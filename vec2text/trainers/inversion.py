import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

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
           - re-embed generated text *via call_embedding_model(...)*
           - compute advantage = reward - group_mean
           - compute policy gradient loss = - sum( advantage * log_prob )
        """

        print(f"INPUTS: {inputs}")

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
        # unwrap DDP if needed:
        model_to_use = model.module if hasattr(model, "module") else model
        gen_outputs = model_to_use.generate(
            inputs={"frozen_embeddings": repeated_frozen_embs},
            generation_kwargs={
                "max_new_tokens": 64,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
                # "temperature": 1.0,   # optional
            },
        )
        gen_texts = model_to_use.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)

        # Instead of embed_and_project, we simply embed the generated text with
        # model.call_embedding_model(...) which returns the final sentence embedding.
        repeated_target_embs = (
            inputs["target_embeddings"]
            .unsqueeze(1)
            .expand(batch_size, group_size, -1)
            .reshape(batch_size * group_size, -1)
            .to(device)
        )

        with torch.no_grad():
            # Convert generated text back to input_ids
            tokenized = model_to_use.tokenizer(
                gen_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # Now call call_embedding_model to get the final embedding
            # for each generated text.
            gen_embs = model_to_use.call_embedding_model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            )
            # shape: [B*N, embedder_dim]

        # Cosine similarity => reward
        rewards = F.cosine_similarity(gen_embs, repeated_target_embs, dim=-1)
        # Optionally scale the reward 
        # e.g.: rewards = 10.0 * rewards

        # Next, gather the log probs of each generated sample from the *current* model.
        dec_inputs = model_to_use.tokenizer(
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

        # For each token in each sequence, gather log-prob of that token
        chosen_logprobs = log_probs.gather(
            -1, dec_inputs["input_ids"].unsqueeze(-1)
        ).squeeze(-1)

        # Mask out the padding tokens
        pad_mask = (dec_inputs["input_ids"] != model_to_use.tokenizer.pad_token_id).float()
        chosen_logprobs = chosen_logprobs * pad_mask
        sum_logprobs = chosen_logprobs.sum(dim=-1)  # shape [B*N]

        # Group advantage => advantage_i = reward_i - mean_of_group
        rewards_2d = rewards.view(batch_size, group_size)         # [B,N]
        sum_logprobs_2d = sum_logprobs.view(batch_size, group_size)

        group_mean = rewards_2d.mean(dim=1, keepdim=True)         # [B,1]
        advantages_2d = rewards_2d - group_mean                   # [B,N]
        advantages = advantages_2d.reshape(-1)                    # [B*N]

        # Final GRPO loss = - sum( advantage * sum_logprobs )
        sum_logprobs_flat = sum_logprobs_2d.view(-1)
        pg_loss = -sum_logprobs_flat * advantages
        loss = pg_loss.mean()

        # Log average reward if desired
        self.log({"train/avg_reward": rewards.mean().item()})

        if return_outputs:
            return (loss, {"logits": logits, "rewards": rewards})
        else:
            return loss

    # def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
    #     # If you ever call trainer.generate(...), route it to model.generate(...).
    #     model_to_generate = self.model.module if hasattr(self.model, "module") else self.model
    #     return model_to_generate.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Exactly as before: we log data-specific metrics, then rely on HF Trainer 
        to call compute_loss behind the scenes.
        """
        self._compute_data_metrics(inputs=inputs)
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
