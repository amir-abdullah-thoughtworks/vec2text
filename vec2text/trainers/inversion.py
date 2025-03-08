import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_outputs import BaseModelOutput


from typing import Dict, Optional

from vec2text.trainers.base import BaseTrainer


class InversionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ######################################################
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model
        self.embedder = self.model.embedder
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ):
        """
        1) If self.args.use_rl is False => run normal cross-entropy (supervised).
        2) If self.args.use_rl is True  => do a GRPO-style RL update:
           - expand each embedding group_size times
           - generate text with sampling
           - re-embed generated text using embed_and_project
           - compute advantage = reward - group_mean
           - compute policy gradient loss = - sum( advantage * log_prob )
        """

        # Check if we are in RL mode
        if not getattr(self.args, "use_rl", False):
            # =========================================
            # Normal cross-entropy route
            # =========================================
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        # ================================================================
        # RL / GRPO route
        # ================================================================
        device = inputs["frozen_embeddings"].device
        batch_size = inputs["frozen_embeddings"].shape[0]

        # 'group_size' determines how many times we sample per input
        group_size = getattr(self.args, "rl_group_size", 4)

        # Expand each embedding [B, embed_dim] -> [B*N, embed_dim]
        repeated_frozen_embs = (
            inputs["frozen_embeddings"]
            .unsqueeze(1)                              # [B,1,dim]
            .expand(batch_size, group_size, -1)        # [B,N,dim]
            .reshape(batch_size * group_size, -1)      # [B*N,dim]
            .to(device)
        )

        # Generate text for each repeated embedding
        gen_outputs = model.generate(
            inputs={"frozen_embeddings": repeated_frozen_embs},
            generation_kwargs={
                "max_new_tokens": 64,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
                # "temperature": 1.0,   # optional
            },
        )
        gen_texts = model.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)

        # Now we compute a reward by re-embedding the generated text
        # using your model's embed_and_project -> T5 encoder -> final pooled representation
        repeated_target_embs = (
            inputs["target_embeddings"]
            .unsqueeze(1)
            .expand(batch_size, group_size, -1)
            .reshape(batch_size * group_size, -1)
            .to(device)
        )

        with torch.no_grad():
            # Convert generated text back to input_ids
            tokenized = model.tokenizer(
                gen_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # Use your model's embed_and_project(...) to get the encoder inputs
            gen_inputs_embeds, gen_attention_mask = model.embed_and_project(
                embedder_input_ids=tokenized["input_ids"],
                embedder_attention_mask=tokenized["attention_mask"],
                frozen_embeddings=None
            )
            # Then run T5 encoder
            enc_out = model.encoder_decoder.encoder(
                inputs_embeds=gen_inputs_embeds,
                attention_mask=gen_attention_mask,
                output_hidden_states=True
            )
            # Convert to a standard BaseModelOutput
            enc_out_struct = BaseModelOutput(
                last_hidden_state=enc_out.last_hidden_state,
                hidden_states=enc_out.hidden_states,
                attentions=None,
            )
            # Your existing _process_embedder_output does mean-pool or something similar
            gen_embs = model._process_embedder_output(enc_out_struct, gen_attention_mask)

        # Cosine similarity vs. target => reward
        rewards = F.cosine_similarity(gen_embs, repeated_target_embs, dim=-1)
        # Optionally scale or offset the reward if you want
        # e.g. rewards = 10.0 * rewards

        # Next, gather the log probs of each generated sample from the *current* model.
        dec_inputs = model.tokenizer(
            gen_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # We'll pass repeated_frozen_embs as the T5 encoder input again,
        # plus dec_inputs["input_ids"] as the decoder tokens in forward()
        out = model(
            frozen_embeddings=repeated_frozen_embs,
            decoder_input_ids=dec_inputs["input_ids"],
        )
        logits = out.logits  # [B*N, seq_len, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)

        # For each token in each sequence, gather log-prob of the generated token
        chosen_logprobs = log_probs.gather(
            -1, dec_inputs["input_ids"].unsqueeze(-1)
        ).squeeze(-1)

        # Mask out the padding tokens
        pad_mask = (dec_inputs["input_ids"] != model.tokenizer.pad_token_id).float()
        chosen_logprobs = chosen_logprobs * pad_mask
        sum_logprobs = chosen_logprobs.sum(dim=-1)  # shape [B*N]

        # Compute group advantage => advantage_i = reward_i - mean_of_group
        rewards_2d = rewards.view(batch_size, group_size)         # [B,N]
        sum_logprobs_2d = sum_logprobs.view(batch_size, group_size)

        group_mean = rewards_2d.mean(dim=1, keepdim=True)         # [B,1]
        advantages_2d = rewards_2d - group_mean                   # [B,N]
        advantages = advantages_2d.reshape(-1)                    # [B*N]

        # Final GRPO loss = - sum( advantage * sum_logprobs )
        sum_logprobs_flat = sum_logprobs_2d.view(-1)
        pg_loss = -sum_logprobs_flat * advantages
        loss = pg_loss.mean()

        # If we want to track average reward in the logs:
        self.log({"train/avg_reward": rewards.mean().item()})

        if return_outputs:
            # Some tasks or Trainer hooks might want the outputs
            return (loss, {"logits": logits, "rewards": rewards})
        else:
            return loss

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Performs a training step. we override to compute data-specific metrics.
        """
        # TODO: Log training metrics from below... (How to do with huggingface?)
        self._compute_data_metrics(inputs=inputs)
        # self.log({ f"train/{k}": v for k,v in metrics.items() })
        return super().training_step(model, inputs)

    def evaluation_loop(
        self, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics.

        Override to compute ppl from eval loss.
        """
        output = super().evaluation_loop(*args, **kwargs)

        metric_key_prefix = kwargs["metric_key_prefix"]
        try:
            perplexity = math.exp(output.metrics[f"{metric_key_prefix}_loss"])
        except KeyError:
            perplexity = -1
        except OverflowError:
            perplexity = float("inf")
        output.metrics[f"{metric_key_prefix}_perplexity"] = perplexity

        return output

    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        """Edit keys posthumously on model load."""
        # Rename keys for backward compatibility w/ model trained before
        # we added extra dropout to the model
        if {
            "embedding_transform.2.weight",
            "embedding_transform.2.bias",
        } <= state_dict.keys():
            print(
                "Renaming keys",
                {"embedding_transform.2.weight", "embedding_transform.2.bias"},
                "for backward compatibility.",
            )
            state_dict["embedding_transform.3.weight"] = state_dict.pop(
                "embedding_transform.2.weight"
            )
            state_dict["embedding_transform.3.bias"] = state_dict.pop(
                "embedding_transform.2.bias"
            )
        return state_dict
