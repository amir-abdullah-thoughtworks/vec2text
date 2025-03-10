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
        2) If self.args.use_rl is True => do a GRPO-style RL update + optional MLE anchor
        """

        # (A) If not RL, do normal MLE
        if not getattr(self.args, "use_rl", False):
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        # ─────────────────────────────────────────────────────────
        # (B) RL Settings
        # ─────────────────────────────────────────────────────────
        device = inputs["frozen_embeddings"].device
        batch_size = inputs["frozen_embeddings"].shape[0]

        group_size = getattr(self.args, "rl_group_size", 4)
        temp = getattr(self.args, "rl_temperature", 1.0)

        # Expand each embedding => [B*N, embed_dim]
        repeated_frozen_embs = (
            inputs["frozen_embeddings"]
            .unsqueeze(1)
            .expand(batch_size, group_size, -1)
            .reshape(batch_size * group_size, -1)
            .to(device)
        )

        # ─────────────────────────────────────────────────────────
        # (C) Generate text with sampling
        # ─────────────────────────────────────────────────────────
        model_to_use = model.module if hasattr(model, "module") else model
        gen_outputs = model_to_use.generate(
            inputs={"frozen_embeddings": repeated_frozen_embs},
            generation_kwargs={
                "max_new_tokens": getattr(self.args, "max_seq_length", 128),
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
                "temperature": temp,
            },
        )
        gen_texts = model_to_use.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)

        # ─────────────────────────────────────────────────────────
        # (D) Reward from embedder
        # ─────────────────────────────────────────────────────────
        repeated_target_embs = (
            inputs["frozen_embeddings"]
            .unsqueeze(1)
            .expand(batch_size, group_size, -1)
            .reshape(batch_size * group_size, -1)
            .to(device)
        )
        with torch.no_grad():
            tokenized = model_to_use.tokenizer(
                gen_texts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            gen_embs = model_to_use.call_embedding_model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            )
        rewards = F.cosine_similarity(gen_embs, repeated_target_embs, dim=-1)

        # ─────────────────────────────────────────────────────────
        # (E) Log-probs from new policy
        # ─────────────────────────────────────────────────────────
        dec_inputs = model_to_use.tokenizer(
            gen_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        out = model(
            embedder_input_ids=None,
            embedder_attention_mask=None,
            frozen_embeddings=repeated_frozen_embs,
            decoder_input_ids=dec_inputs["input_ids"],
        )
        logits = out.logits
        log_probs = F.log_softmax(logits, dim=-1)

        chosen_logprobs = log_probs.gather(
            -1, dec_inputs["input_ids"].unsqueeze(-1)
        ).squeeze(-1)

        pad_mask = (dec_inputs["input_ids"] != model_to_use.tokenizer.pad_token_id).float()
        chosen_logprobs *= pad_mask
        sum_logprobs = chosen_logprobs.sum(dim=-1)  # shape [B*N]

        # Advantage
        rewards_2d = rewards.view(batch_size, group_size)
        sum_logprobs_2d = sum_logprobs.view(batch_size, group_size)

        group_mean = rewards_2d.mean(dim=1, keepdim=True)
        advantages_2d = rewards_2d - group_mean
        advantages = advantages_2d.reshape(-1)

        sum_logprobs_flat = sum_logprobs_2d.view(-1)
        pg_loss = -sum_logprobs_flat * advantages
        rl_loss = pg_loss.mean()  # RL objective

        # ─────────────────────────────────────────────────────────
        # (F) MLE Anchor
        # ─────────────────────────────────────────────────────────
        mle_alpha = getattr(self.args, "mle_anchor_alpha", 0.01)
        if "labels" in inputs:
            # Do a forward pass in teacher-forcing mode with the original text
            # so we compute cross-entropy
            anchor_out = model(
                embedder_input_ids=inputs["embedder_input_ids"],
                embedder_attention_mask=inputs["embedder_attention_mask"],
                labels=inputs["labels"],
            )
            mle_loss = anchor_out.loss  # cross-entropy from HF model
        else:
            # If there's no "labels", we can't do MLE anchor
            mle_loss = torch.zeros(1, device=device)

        total_loss = rl_loss + mle_alpha * mle_loss

        # (G) Logging
        self.log({
            "train/avg_reward": rewards.mean().item(),
            "train/rl_loss": rl_loss.item(),
            "train/mle_loss": mle_loss.item() if mle_loss.numel() > 0 else 0.0,
        })

        # Return final loss
        if return_outputs:
            return (total_loss, {"logits": logits, "rewards": rewards, "mle_loss": mle_loss})
        else:
            return total_loss

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        model_to_generate = self.model.module if hasattr(self.model, "module") else self.model
        return model_to_generate.generate(inputs=inputs, generation_kwargs=generation_kwargs)

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
