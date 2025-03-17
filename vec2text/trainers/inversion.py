import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from trl import GRPOTrainer, GRPOConfig

from vec2text.trainers.base import BaseTrainer


class InversionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model
        self.embedder = self.model.embedder

        # --- Only create a GRPO sub-trainer if RL is enabled
        if getattr(self.args, "use_rl", False):
            # Example GRPOConfig
            self._grpo_config = GRPOConfig(
                output_dir=getattr(self.args, "output_dir", "outputs/"),
                logging_steps=10,
                learning_rate=1e-5,
                beta=getattr(self.args, "kl_coeff_beta", 0.2),
                num_generations=getattr(self.args, "rl_num_generations", 4),
                max_steps=getattr(self.args, "max_train_steps", 10000),
                temperature=getattr(self.args, "rl_temperature", 1.0),
                max_completion_length=getattr(self.args, "max_seq_length", 128),
                use_vllm=False, # So we can use all GPUs for training
            )

            # Reward function: compare generated text to "frozen_embeddings" via cos sim
            def embedder_reward_fn(samples, prompts):
                device = next(self.model.parameters()).device
                with torch.no_grad():
                    target_embs = []
                    for p in prompts:
                        # each "p" is a dict with "frozen_embeddings" from your dataset
                        target_embs.append(p["frozen_embeddings"].to(device))
                    target_embs = torch.stack(target_embs, dim=0)

                    tokenized = self.tokenizer(
                        samples, padding=True, truncation=True, return_tensors="pt"
                    ).to(device)

                    # forward pass through your embedder to get embeddings for the generated text
                    gen_embs = self.call_embedding_model(
                        input_ids=tokenized["input_ids"],
                        attention_mask=tokenized["attention_mask"],
                    )
                    rewards = F.cosine_similarity(gen_embs, target_embs, dim=-1)

                # Return as a Python list
                return rewards.cpu().tolist()

            # Create the sub-trainer from TRL
            self._grpo_trainer = GRPOTrainer(
                model=self.model,
                args=self._grpo_config,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                reward_funcs=embedder_reward_fn,
            )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        """
        If use_rl=False => standard cross-entropy (supervised).
        If use_rl=True  => delegate to GRPOTrainer's RL logic (no MLE anchor).
        """
        if not getattr(self.args, "use_rl", False):
            # Normal cross-entropy path
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        # RL path => rely entirely on GRPO
        loss_dict = self._grpo_trainer.compute_loss(model, inputs, return_outputs=False)
        rl_loss = loss_dict["loss"]

        # Optional: Log something about RL loss
        self.log({"train/rl_loss": rl_loss.item()})

        if return_outputs and getattr(self.args, "use_rl", False) == False:
            return (rl_loss, loss_dict)
        else:
            return rl_loss

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        model_to_generate = self.model.module if hasattr(self.model, "module") else self.model
        return model_to_generate.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch=None
    ) -> torch.Tensor:
        self._compute_data_metrics(inputs=inputs)
        return super().training_step(model, inputs)

    def evaluation_loop(
        self, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:
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