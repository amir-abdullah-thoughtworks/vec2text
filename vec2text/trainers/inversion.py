import math
from typing import Dict

import torch
import torch.nn as nn
import transformers
from transformers import TrainerCallback

from vec2text.trainers.base import BaseTrainer

class _UnfreezeDiffusionCallback(TrainerCallback):
    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps
    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if getattr(model, "train_diffusion", False) and state.global_step == self.warmup_steps:
            for p in model.diffusion_sampler.parameters():
                p.requires_grad_(True)
            if model.is_main_process:
                print(f"[two-stage] diffusion unfrozen at step {state.global_step}")

class InversionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ######################################################
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model
        self.embedder = self.model.embedder

        warmup = self.model.config.diffusion_warmup_steps
        if self.model.train_diffusion:
            self.add_callback(_UnfreezeDiffusionCallback(warmup))
    
    # ------------------------------------------------------------------
    # log per-component losses coming from InversionModel.forward
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Runs the usual forward pass (so uncertainty weighting happens inside
        the model) and logs raw & weighted losses.
        """
        outputs = model(**inputs)
        loss    = outputs.loss

        if hasattr(outputs, "extra_losses"):
            log_dict = {k: v.item() for k, v in outputs.extra_losses.items()}
            # Trainer.log takes care of sync-dist if required.
            self.log(log_dict)

        return (loss, outputs) if return_outputs else loss

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