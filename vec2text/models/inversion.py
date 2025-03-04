import copy
import logging
import weakref  # add weakref for parent_model references

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sentence_transformers import SentenceTransformer
from transformers.modeling_outputs import Seq2SeqLMOutput

from vec2text.models.config import InversionConfig
from vec2text.models.model_utils import (
    FREEZE_STRATEGIES,
    disable_dropout,
    freeze_params,
    load_embedder_and_tokenizer,
    load_encoder_decoder,
    load_tokenizer,
    mean_pool,
)
from vec2text.utils import embed_api

logger = logging.getLogger(__name__)

def is_main_process() -> bool:
    """Helper to check if we're on rank 0 when using torchrun --nproc_per_node."""
    return int(os.environ.get("RANK", "0")) == 0

def debug_print(*args, **kwargs) -> None:
    """
    Print only on the main process, to avoid clutter from multiple GPUs.
    Flush output so we see it right away.
    """
    if is_main_process():
        print(*args, **kwargs, flush=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [GUIDED DIFFUSION ADDED]
# Full submodule implementing advanced diffusion with optional guidance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CosineScheduler:
    """
    Modified Cosine schedule for betas over T timesteps, 
    with an additional clamp so that alpha = 1 - beta never exceeds 0.995.
    This avoids alpha being too close to 1.0.
    """
    def __init__(self, timesteps: int = 50, beta_start: float = 1e-6, beta_end: float = 1e-4):
        import math
        self.timesteps = timesteps
        betas = []
        max_alpha = 0.995  # clamp alpha <= 0.995

        for i in range(timesteps):
            f = i / (timesteps - 1)
            cos_val = 0.5 * (1 - math.cos(math.pi * f))
            raw_beta = beta_start + cos_val * (beta_end - beta_start)
            alpha = 1.0 - raw_beta
            alpha = min(alpha, max_alpha)
            beta = 1.0 - alpha
            betas.append(beta)

        self.betas = torch.tensor(betas, dtype=torch.float32)


class SelfConditionedDenoiser(nn.Module):
    """
    A basic Transformer-based denoiser that self-conditions on previous step's predicted x0.
    We'll use it to predict v (velocity) instead of raw noise, for stability.
    """
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 1024,
        time_embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t: torch.Tensor, z_hat: Optional[torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        if z_hat is None:
            z_hat = torch.zeros_like(z_t)
        x = torch.cat([z_t, z_hat], dim=-1)  # (B,L,2*latent_dim)
        h = self.input_proj(x)

        # time embedding
        t_emb = self.time_mlp(t.unsqueeze(-1).float())  # (B, time_embed_dim)
        t_emb = t_emb.unsqueeze(1).expand(h.size(0), h.size(1), t_emb.size(-1))
        if t_emb.size(-1) < h.size(-1):
            pad_dim = h.size(-1) - t_emb.size(-1)
            t_emb = F.pad(t_emb, (0, pad_dim), "constant", 0)
        h = h + t_emb

        h = self.transformer(h)
        out = self.out_proj(h)
        return out


class GuidedDiffusion(nn.Module):
    """
    Full diffusion submodule with:
      - Cosine schedule (with alpha clamp)
      - Self-conditioning
      - Anchor loss
      - v-pred for stable training
      - Optional guidance calls
      - ...
    """
    def __init__(
        self,
        parent_model: nn.Module,
        latent_dim: int = 256,
        timesteps: int = 50,
        anchor_weight: float = 0.01,
        guidance_mode: str = "cosine",
        guidance_finite_diff_eps: float = 1e-3,
    ):
        super().__init__()
        self._parent_ref = weakref.ref(parent_model)

        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.anchor_weight = anchor_weight
        self.guidance_mode = guidance_mode
        self.guidance_finite_diff_eps = guidance_finite_diff_eps

        sched = CosineScheduler(timesteps, beta_start=1e-6, beta_end=1e-4)
        self.register_buffer("betas", sched.betas)

        alphas = 1.0 - sched.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = torch.clamp(alphas_cumprod, 0.0, 1.0)

        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.denoiser = SelfConditionedDenoiser(
            latent_dim=latent_dim,
            hidden_dim=4 * latent_dim,
            time_embed_dim=256,
            num_layers=2,
            num_heads=4,
        )

        self.final_layernorm = nn.LayerNorm(latent_dim)

    def _parent(self) -> Optional[nn.Module]:
        return self._parent_ref()

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(z0)
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
        return alpha_bar.sqrt() * z0 + (1 - alpha_bar).sqrt() * noise

    def forward(
        self,
        z0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        z_hat: Optional[torch.Tensor] = None,
        training_guidance_scale: float = 0.0,
        training_guidance_freq: int = 99999999,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (diffusion_loss, x0_pred) so that calling code can
        use x0_pred for additional losses, expansions, etc.
        """
        B, L, D = z0.shape
        if t is None:
            t = torch.randint(0, self.timesteps, (B,), device=z0.device)

        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise=noise)

        if z_hat is None:
            z_hat = torch.zeros_like(z_t)

        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
        v_target = alpha_bar.sqrt() * noise - (1 - alpha_bar).sqrt() * z0

        v_pred = self.denoiser(z_t, z_hat, t)
        v_mse = F.mse_loss(v_pred, v_target)

        # Convert v_pred => predicted noise => x0_pred
        pred_noise = (v_pred + (1 - alpha_bar).sqrt() * z0) / (alpha_bar.sqrt() + 1e-7)
        x0_pred = (z_t - (1 - alpha_bar).sqrt() * pred_noise) / (alpha_bar.sqrt() + 1e-7)

        #x0_pred = self._percentile_dynamic_threshold(x0_pred, percentile=0.995)

        x0_pred = self.final_layernorm(x0_pred)

        # Anchor loss after layernorm
        anchor_loss = F.mse_loss(x0_pred, z0)

        total_loss = v_mse + self.anchor_weight * anchor_loss

        # Partial-decode guidance
        if (training_guidance_scale > 0.0) and (training_guidance_freq < 99999999):
            x0_guided = self.apply_embedding_guidance_training(
                x0_pred, z0, training_guidance_scale
            )
            guidance_loss = self._compute_guidance_loss(x0_guided, z0)
            total_loss = total_loss + guidance_loss

        return total_loss, x0_pred

    def apply_embedding_guidance_training(
        self,
        x0_pred: torch.Tensor,
        z0: torch.Tensor,
        guidance_scale: float
    ) -> torch.Tensor:
        pm = self._parent()
        if pm is None:
            return x0_pred

        device = x0_pred.device
        B, L, D = x0_pred.shape
        if L > 1:
            lat_mean = x0_pred.mean(dim=1, keepdim=True)
        else:
            lat_mean = x0_pred

        texts = pm._decode_latent_to_text(lat_mean)

        # embed
        new_embs = []
        for text in texts:
            tokenized = pm.embedder_tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
            emb = pm.call_embedding_model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            )
            new_embs.append(emb)
        new_embs = torch.cat(new_embs, dim=0)

        # if no target embedding, do nothing
        if not hasattr(pm, "_training_target_emb"):
            return x0_pred
        target_embedding = pm._training_target_emb

        # compute baseline score
        if self.guidance_mode == "cosine":
            cos_sim = F.cosine_similarity(new_embs, target_embedding, dim=-1)
            avg_score = cos_sim.mean()
        else:
            mse_val = F.mse_loss(new_embs, target_embedding, reduction="none").mean(dim=-1)
            avg_score = -mse_val.mean()

        # naive finite diff
        partial_dims = min(10, lat_mean.numel())
        dim_list = torch.randperm(lat_mean.numel(), device=device)[:partial_dims]
        best_update = torch.zeros_like(lat_mean)
        eps = self.guidance_finite_diff_eps

        old_flat = lat_mean.flatten()
        for d in dim_list:
            old_val = old_flat[d].item()
            old_flat[d] = old_val + eps
            lat_perturbed = old_flat.view(lat_mean.shape)

            texts_pert = pm._decode_latent_to_text(lat_perturbed)
            new_embs_pert = []
            for txt in texts_pert:
                tkn = pm.embedder_tokenizer([txt], return_tensors="pt").to(device)
                emb_pert = pm.call_embedding_model(
                    input_ids=tkn["input_ids"],
                    attention_mask=tkn["attention_mask"],
                )
                new_embs_pert.append(emb_pert)
            new_embs_pert = torch.cat(new_embs_pert, dim=0)

            if self.guidance_mode == "cosine":
                new_score = F.cosine_similarity(new_embs_pert, target_embedding, dim=-1).mean().item()
            else:
                new_score = -F.mse_loss(new_embs_pert, target_embedding).item()

            grad_approx = (new_score - avg_score.item()) / eps
            old_flat[d] = old_val

            best_update_flat = best_update.flatten()
            best_update_flat[d] = grad_approx
            best_update = best_update_flat.view(best_update.shape)

        lat_new = lat_mean + guidance_scale * best_update
        if L > 1:
            lat_new = lat_new.expand(B, L, D)
        return lat_new

    def _compute_guidance_loss(self, x0_pred: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        pm = self._parent()
        if pm is None or not hasattr(pm, "_training_target_emb"):
            return torch.tensor(0.0, device=x0_pred.device)

        target_emb = pm._training_target_emb
        device = x0_pred.device

        B, L, D = x0_pred.shape
        lat_mean = x0_pred.mean(dim=1, keepdim=True)
        texts = pm._decode_latent_to_text(lat_mean)
        new_embs = []
        for text in texts:
            tkn = pm.embedder_tokenizer([text], return_tensors="pt").to(device)
            emb = pm.call_embedding_model(
                input_ids=tkn["input_ids"],
                attention_mask=tkn["attention_mask"]
            )
            new_embs.append(emb)
        new_embs = torch.cat(new_embs, dim=0)

        cos_sim = F.cosine_similarity(new_embs, target_emb, dim=-1).mean()
        guidance_loss = -cos_sim
        return guidance_loss

    def _percentile_dynamic_threshold(self, x0: torch.Tensor, percentile: float = 0.995) -> torch.Tensor:
        abs_vals = x0.abs().reshape(-1)
        thresh_val = torch.quantile(abs_vals, percentile)
        if thresh_val > 1.0:
            scale = 1.0 / (thresh_val + 1e-7)
            x0 = x0 * scale
        return x0

    def p_sample_loop(
        self,
        x_T: torch.Tensor,
        steps: Optional[int] = None,
        inference_guidance_scale: float = 1.0,
        inference_guidance_freq: int = 1,
        target_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if steps is None:
            steps = self.timesteps

        x_t = x_T.clone()
        z_hat = torch.zeros_like(x_t)
        percentile = 0.995

        for i in reversed(range(steps)):
            t_tensor = torch.full((x_t.size(0),), i, device=x_t.device, dtype=torch.long)
            if i % 5 == 0:
                print(
                    f"[p_sample_loop] step={i}, x_t min/mean/max: "
                    f"{x_t.min().item()}, {x_t.mean().item()}, {x_t.max().item()}",
                    flush=True
                )

            v_pred = self.denoiser(x_t, z_hat, t_tensor)
            alpha_bar = self.alphas_cumprod[t_tensor].view(-1,1,1)
            eps_pred = (v_pred + (1 - alpha_bar).sqrt() * x_t) / (alpha_bar.sqrt() + 1e-7)
            x0_pred = (x_t - (1 - alpha_bar).sqrt() * eps_pred) / (alpha_bar.sqrt() + 1e-7)

            #x0_pred = self._percentile_dynamic_threshold(x0_pred, percentile=percentile)
            x0_pred = self.final_layernorm(x0_pred)
            if i % 5 == 0:
                print(
                    f"[p_sample_loop] step={i}, x0_pred min/mean/max: "
                    f"{x0_pred.min().item()}, {x0_pred.mean().item()}, {x0_pred.max().item()}",
                    flush=True
                )

            # Guidance
            if (
                inference_guidance_scale > 0.0
                and (i % inference_guidance_freq == 0)
                and (target_embedding is not None)
            ):
                x0_pred = self.apply_embedding_guidance(
                    x0_pred, inference_guidance_scale, target_embedding
                )
                #x0_pred = self._percentile_dynamic_threshold(x0_pred, percentile=percentile)

            z_hat = x0_pred.detach()

            if i > 0:
                beta_t = self.betas[t_tensor].view(-1,1,1)
                alpha_bar_prev = self.alphas_cumprod[t_tensor - 1].view(-1,1,1)
                safe_denom = torch.clamp((1 - alpha_bar).sqrt(), min=1e-4)
                mean = (
                    x0_pred * alpha_bar_prev.sqrt()
                    + (x_t - x0_pred * alpha_bar.sqrt()) * (1 - alpha_bar_prev).sqrt()
                ) / (safe_denom + 1e-7)

                var = beta_t * (1 - alpha_bar_prev) / torch.clamp((1 - alpha_bar), min=1e-4)
                noise = torch.randn_like(x_t)

                if i % 5 == 0:
                    print(
                        f"[p_sample_loop] step={i}, mean min/mean/max: "
                        f"{mean.min().item()}, {mean.mean().item()}, {mean.max().item()}",
                        flush=True
                    )
                    print(
                        f"[p_sample_loop] step={i}, var min/mean/max: "
                        f"{var.min().item()}, {var.mean().item()}, {var.max().item()}",
                        flush=True
                    )

                x_t = mean + var.sqrt() * noise

        return x0_pred

    def apply_embedding_guidance(
        self,
        x0_pred: torch.Tensor,
        guidance_scale: float,
        target_embedding: torch.Tensor
    ) -> torch.Tensor:
        pm = self._parent()
        if pm is None:
            return x0_pred

        device = x0_pred.device
        B, L, D = x0_pred.shape

        if L > 1:
            lat_mean = x0_pred.mean(dim=1, keepdim=True)
        else:
            lat_mean = x0_pred

        texts = pm._decode_latent_to_text(lat_mean)
        new_embs = []
        for text in texts:
            tokenized = pm.embedder_tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
            emb = pm.call_embedding_model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"]
            )
            new_embs.append(emb)
        new_embs = torch.cat(new_embs, dim=0)

        if self.guidance_mode == "cosine":
            cos_sim = F.cosine_similarity(new_embs, target_embedding, dim=-1).mean()
            avg_score = cos_sim.item()
        else:
            mse_val = F.mse_loss(new_embs, target_embedding)
            avg_score = -mse_val.item()

        partial_dims = min(10, lat_mean.numel())
        dim_list = torch.randperm(lat_mean.numel(), device=device)[:partial_dims]
        best_update = torch.zeros_like(lat_mean)
        eps = self.guidance_finite_diff_eps

        lat_flat = lat_mean.flatten()
        for d in dim_list:
            old_val = lat_flat[d].item()
            lat_flat[d] = old_val + eps
            lat_perturbed = lat_flat.view(lat_mean.shape)

            texts_pert = pm._decode_latent_to_text(lat_perturbed)
            new_embs_pert = []
            for txt in texts_pert:
                tkn = pm.embedder_tokenizer([txt], return_tensors="pt").to(device)
                emb_pert = pm.call_embedding_model(
                    input_ids=tkn["input_ids"],
                    attention_mask=tkn["attention_mask"],
                )
                new_embs_pert.append(emb_pert)
            new_embs_pert = torch.cat(new_embs_pert, dim=0)

            if self.guidance_mode == "cosine":
                new_score = F.cosine_similarity(new_embs_pert, target_embedding, dim=-1).mean().item()
            else:
                new_score = -F.mse_loss(new_embs_pert, target_embedding).item()

            grad_approx = (new_score - avg_score) / eps
            lat_flat[d] = old_val

            best_update_flat = best_update.flatten()
            best_update_flat[d] = grad_approx
            best_update = best_update_flat.view(best_update.shape)

        lat_new = lat_mean + guidance_scale * best_update
        if L > 1:
            lat_new_exp = lat_new.expand(B, L, D)
        else:
            lat_new_exp = lat_new
        return lat_new_exp


@dataclass
class DiffusionSeq2SeqLMOutput(Seq2SeqLMOutput):
    """
    Extends Seq2SeqLMOutput with extra fields for CE loss and diffusion loss.
    """
    ce_loss: Optional[torch.FloatTensor] = None
    diffusion_loss: Optional[torch.FloatTensor] = None


class InversionModel(transformers.PreTrainedModel):
    """
    A model that inverts embeddings to text, now with optional guided diffusion.
    Always ensures compress/expand are in the gradient path (dummy usage if needed).
    """

    config_class = InversionConfig
    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    encoder_decoder_lora: bool
    tokenizer: transformers.PreTrainedTokenizer
    embedding_transform: nn.Module
    bottleneck_dim: int
    num_repeat_tokens: int
    embedder_dim: int
    embedder_no_grad: bool
    embedder_fake_with_zeros: bool
    embedding_transform_strategy: str
    use_frozen_embeddings_as_input: bool
    embedded_tokens: torch.Tensor
    embedder_model_api: Optional[str]

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        embedder_model_api = config.embedder_model_api
        embedder_fake_with_zeros = config.embedder_fake_with_zeros
        use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input
        encoder_dropout_disabled = config.encoder_dropout_disabled
        decoder_dropout_disabled = config.decoder_dropout_disabled
        embeddings_from_layer_n = config.embeddings_from_layer_n

        encoder_decoder = load_encoder_decoder(
            model_name=config.model_name_or_path,
            lora=config.use_lora,
        )

        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name, torch_dtype=config.embedder_torch_dtype
        )

        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )
        num_repeat_tokens = config.num_repeat_tokens
        embedder_no_grad = config.embedder_no_grad

        self.encoder_decoder = encoder_decoder
        self.num_repeat_tokens = num_repeat_tokens

        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        if embedder_model_api:
            self.embedder_dim = 1536
            bottleneck_dim = self.embedder_dim
        elif isinstance(embedder, SentenceTransformer):
            self.embedder_dim = embedder.get_sentence_embedding_dimension()
            bottleneck_dim = self.embedder_dim
        else:
            if hasattr(embedder.config, "text_config"):
                self.embedder_dim = embedder.config.text_config.hidden_size
            elif hasattr(embedder.config, "hidden_size"):
                self.embedder_dim = embedder.config.hidden_size
            else:
                raise ValueError(
                    f"could not determine hidden size for embedder {embedder} of type {type(embedder)}"
                )
            bottleneck_dim = self.embedder_dim

        self.embedder_no_grad = embedder_no_grad
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        self.bottleneck_dim = bottleneck_dim

        # ~~~ Trainable LayerNorm+MLP in place of 0.1 scale ~~~
        self.latent_preproc = nn.Sequential(
            nn.LayerNorm(self.embedder_dim),
            nn.Linear(self.embedder_dim, self.embedder_dim),
            nn.SiLU(),
            nn.Linear(self.embedder_dim, self.embedder_dim),
        )

        # ~~~ embedding_transform => repeated hidden states for T5 encoder ~~~
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens),
        )

        if encoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.encoder)
        if decoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.decoder)
            disable_dropout(self.encoder_decoder.lm_head)

        self.tokenizer = tokenizer
        self.embedder = embedder
        if self.embedder_no_grad:
            for param in self.embedder.parameters():
                param.requires_grad = False
            self.embedder.eval()

        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        self.embedder_fake_with_zeros = embedder_fake_with_zeros
        self.embedding_transform_strategy = "repeat"
        self.embeddings_from_layer_n = embeddings_from_layer_n
        self.noise_level = vars(config).get("embedder_gaussian_noise_level", 0.0)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # [GUIDED DIFFUSION ADDED]
        self.use_diffusion = getattr(config, "use_diffusion", False)
        self.latent_dim = getattr(config, "latent_dim", 256)
        self.diffusion_timesteps = getattr(config, "diffusion_timesteps", 50)
        self.diffusion_anchor_weight = getattr(config, "diffusion_anchor_weight", 0.01)

        # Guidance parameters
        self.diffusion_training_guidance_scale = getattr(config, "diffusion_training_guidance_scale", 0.0)
        self.diffusion_training_guidance_freq = getattr(config, "diffusion_training_guidance_freq", 9999999999)

        self.diffusion_inference_guidance_scale = getattr(config, "diffusion_inference_guidance_scale", 0.0)
        self.diffusion_inference_guidance_freq = getattr(config, "diffusion_inference_guidance_freq", 9999999999)

        # optional config fields
        self.diffusion_guidance_mode = getattr(config, "diffusion_guidance_mode", "cosine")
        self.diffusion_guidance_finite_diff_eps = getattr(config, "diffusion_guidance_finite_diff_eps", 1e-3)

        # **Crucial**: define these in __init__ so they're recognized & optimized
        self._diffusion_compress = nn.Linear(encoder_hidden_dim, self.latent_dim)
        self._diffusion_expand   = nn.Linear(self.latent_dim, encoder_hidden_dim)

        if self.use_diffusion:
            self.guided_diffusion = GuidedDiffusion(
                parent_model=self,
                latent_dim=self.latent_dim,
                timesteps=self.diffusion_timesteps,
                anchor_weight=self.diffusion_anchor_weight,
                guidance_mode=self.diffusion_guidance_mode,
                guidance_finite_diff_eps=self.diffusion_guidance_finite_diff_eps,
            )

    def _freeze_encoder(self) -> None:
        freeze_params(self.encoder_decoder.encoder)

    def _freeze_decoder(self) -> None:
        freeze_params(self.encoder_decoder.decoder)
        freeze_params(self.encoder_decoder.lm_head)

    def freeze(self, freeze_strategy: str) -> None:
        if freeze_strategy not in FREEZE_STRATEGIES:
            raise ValueError(f"invalid freezing strategy {freeze_strategy}")
        if freeze_strategy == "decoder":
            self._freeze_decoder()
        elif freeze_strategy == "encoder":
            self._freeze_encoder()
        elif freeze_strategy == "encoder_and_decoder":
            self._freeze_encoder()
            self._freeze_decoder()
            freeze_params(self.encoder_decoder.shared)
        elif freeze_strategy == "none":
            pass

    @property
    def embedder_device(self) -> torch.device:
        return next(self.embedder.parameters()).device

    def _process_embedder_output(
        self,
        outputs: transformers.modeling_outputs.BaseModelOutput,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and (outputs.pooler_output is not None):
            return outputs.pooler_output
        else:
            if self.embeddings_from_layer_n is not None:
                assert hasattr(outputs, "hidden_states"), "need output_hidden_states=True"
                hidden_state = outputs.hidden_states[self.embeddings_from_layer_n]
                embeddings = mean_pool(hidden_state, attention_mask)
            else:
                embeddings = mean_pool(outputs.last_hidden_state, attention_mask)
            return embeddings

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.embedder_no_grad:
            self.embedder.eval()

        if self.embedder_fake_with_zeros:
            B = input_ids.size(0)
            return torch.zeros((B, self.embedder_dim), dtype=torch.float32, device=self.embedder_device)
        elif self.embedder_model_api:
            embeddings = embed_api(
                input_ids=input_ids,
                embedder_tokenizer=self.embedder_tokenizer,
                api_name=self.embedder_model_api,
            )
        elif isinstance(self.embedder, SentenceTransformer):
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids
            model_output = self.embedder(model_inputs)
            embeddings = model_output["sentence_embedding"]
        else:
            model_output = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self._process_embedder_output(model_output, attention_mask)

        if self.training and self.noise_level > 0:
            embeddings += self.noise_level * torch.randn_like(embeddings)
        return embeddings

    def embed_and_project(
        self,
        embedder_input_ids: Optional[torch.Tensor],
        embedder_attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (embedder_input_ids is None) and (frozen_embeddings is None):
            raise ValueError("Need either embedder_input_ids or frozen_embeddings")

        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
        else:
            embeddings = self.call_embedding_model(embedder_input_ids, embedder_attention_mask)

        # Instead of 0.1 scale => LN + MLP
        embeddings = self.latent_preproc(embeddings)

        repeated_embeddings = self.embedding_transform(embeddings)
        B = repeated_embeddings.size(0)
        hidden_dim = self.encoder_decoder.config.hidden_size
        repeated_embeddings = repeated_embeddings.view(B, self.num_repeat_tokens, hidden_dim)
        attention_mask = torch.ones((B, self.num_repeat_tokens), device=repeated_embeddings.device)

        return repeated_embeddings, attention_mask

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
        debug_compare_latents: bool = True,
    ) -> torch.Tensor:
        if not self.use_diffusion:
            debug_print("---Not using diffusion---")
            return self._naive_generate(inputs, generation_kwargs)

        inputs_embeds, _ = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )
        debug_print(f"--inputs_embeds.shape: {inputs_embeds.shape}")
        lat_mean = inputs_embeds.mean(dim=1, keepdim=True)

        z0 = self._diffusion_compress(lat_mean)
        debug_print(f"--generate() :: z0 min/mean/max: {z0.min()}, {z0.mean()}, {z0.max()}")

        ### ADDED #3a: decode z0 if debug
        if debug_compare_latents:
            expanded_init = self._diffusion_expand(z0)
            attn_mask_init = torch.ones_like(expanded_init[..., 0], dtype=torch.long)
            out_ids_init = self.encoder_decoder.generate(
                inputs_embeds=expanded_init,
                attention_mask=attn_mask_init,
                max_new_tokens=30,
                do_sample=False,
                num_beams=1,
            )
            text_initial = self.tokenizer.batch_decode(out_ids_init, skip_special_tokens=True)
            print("=== [generate debug] INITIAL LATENTS => DECODE ===")
            print(f"z0: min={z0.min():.3f}, mean={z0.mean():.3f}, max={z0.max():.3f}")
            if text_initial:
                print(f"sample text_initial: {text_initial[0]}")
            print("==============================================")

        # partial noising => xT
        T = self.diffusion_timesteps
        noise = torch.randn_like(z0)
        xT = self.guided_diffusion.q_sample(z0, t=T-1, noise=noise)

        refine_steps = generation_kwargs.pop("refine_steps", T)
        target_emb = inputs.get("target_embedding", None)
        x0 = self.guided_diffusion.p_sample_loop(
            x_T=xT,
            steps=refine_steps,
            inference_guidance_scale=self.diffusion_inference_guidance_scale,
            inference_guidance_freq=self.diffusion_inference_guidance_freq,
            target_embedding=target_emb,
        )

        if debug_compare_latents:
            expanded_final = self._diffusion_expand(x0)
            attn_mask_final = torch.ones_like(expanded_final[..., 0], dtype=torch.long)
            out_ids_final = self.encoder_decoder.generate(
                inputs_embeds=expanded_final,
                attention_mask=attn_mask_final,
                max_new_tokens=30,
                do_sample=False,
                num_beams=1,
            )
            text_final = self.tokenizer.batch_decode(out_ids_final, skip_special_tokens=True)
            print("=== [generate debug] FINAL LATENTS => DECODE ===")
            print(f"x0: min={x0.min():.3f}, mean={x0.mean():.3f}, max={x0.max():.3f}")
            if text_final:
                print(f"sample text_final: {text_final[0]}")
            print("==============================================")

        expanded = self._diffusion_expand(x0).expand(-1, inputs_embeds.size(1), -1)
        debug_print(f"---expanded: {expanded}---")

        attention_mask = torch.ones(
            (expanded.size(0), expanded.size(1)), dtype=torch.long, device=expanded.device
        )
        debug_print(f"---attention_mask: {attention_mask}---")

        out_ids = self.encoder_decoder.generate(
            inputs_embeds=expanded,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        debug_print(f"---out_ids: {out_ids}---")

        max_len = generation_kwargs.get("max_length", 128)
        if out_ids.size(1) < max_len:
            pad_amount = max_len - out_ids.size(1)
            pad_tokens = torch.full(
                (out_ids.size(0), pad_amount),
                self.tokenizer.pad_token_id,
                dtype=out_ids.dtype,
                device=out_ids.device,
            )
            out_ids = torch.cat([out_ids, pad_tokens], dim=1)

        debug_print(f"---out_ids FINAL: {out_ids}---")
        return out_ids

    def _naive_generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        debug_print("--Doing _naive_generate--")
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )
        return self.encoder_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        train_step: int = 0,
        max_steps: int = 100000,
        **kwargs,
    ) -> DiffusionSeq2SeqLMOutput:

        # ~~~ Normal T5 forward ~~~
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=frozen_embeddings,
        )
        seq2seq_outputs = self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )

        ce_loss = seq2seq_outputs.loss
        total_loss = ce_loss
        diffusion_loss = None

        # Weighted schedule (same as you had)
        progress = min(float(train_step) / float(max_steps), 1.0)
        ce_weight_start, ce_weight_end = 1.0, 1.0
        diff_weight_start, diff_weight_end = 1.0, 1.0
        base_ce_weight = ce_weight_start + (ce_weight_end - ce_weight_start) * progress
        base_diff_weight = diff_weight_start + (diff_weight_end - diff_weight_start) * progress

        if self.use_diffusion and self.training:
            # figure out current epoch (same logic you had)
            num_epochs = getattr(self.config, "num_train_epochs", 30.0)
            steps_per_epoch = max_steps / (num_epochs + 1e-7)
            current_epoch = train_step / (steps_per_epoch + 1e-7)
            delay_epoch = getattr(self.config, "diffusion_guidance_delay_epoch", 5)

            if current_epoch < delay_epoch:
                actual_guidance_scale = 0.0
                actual_guidance_freq  = 999999999
            else:
                actual_guidance_scale = self.diffusion_training_guidance_scale
                actual_guidance_freq  = self.diffusion_training_guidance_freq

            # Compress T5 input embeddings -> z0 for diffusion
            lat_mean = inputs_embeds.mean(dim=1, keepdim=True)
            z0 = self._diffusion_compress(lat_mean)

            # Instead of returning just "diff_loss", we now get both:
            diff_loss, x0_pred = self.guided_diffusion(
                z0,
                t=None,
                z_hat=None,
                training_guidance_scale=actual_guidance_scale,
                training_guidance_freq=actual_guidance_freq,
            )

            # 2a) Weighted sum of T5 CE + diffusion loss
            diff_component = base_diff_weight * diff_loss
            ce_component = base_ce_weight * ce_loss

            # 2b) Do an extra T5 pass on the final x0_pred
            #     so that we train T5 from the diffusion latents too.
            expanded_x0 = self._diffusion_expand(x0_pred)
            attn_mask_x0 = torch.ones(
                (expanded_x0.size(0), expanded_x0.size(1)), dtype=torch.long, device=expanded_x0.device
            )
            # teacher-forcing with the same labels
            seq2seq_diff_out = self.encoder_decoder(
                inputs_embeds=expanded_x0,
                attention_mask=attn_mask_x0,
                labels=labels,
            )
            # an additional CE that measures if final latents decode to the ground truth
            ce_diff_loss = seq2seq_diff_out.loss

            # Weighting very simply for now
            # second_ce_weight = 2.0  
            #diff_ce_component = base_diff_weight * second_ce_weight * ce_diff_loss

            # Final total loss
            #total_loss = ce_component + diff_component + diff_ce_component
            total_loss = ce_diff_loss

            diffusion_loss = diff_loss  # store for logging

            print(
                f"train_step: {train_step} || epoch={current_epoch:.2f} || progress={progress:.2f} "
                f"|| ce_loss={ce_loss:.4f}  diff_loss={diff_loss:.4f}  diffCE={ce_diff_loss:.4f} "
                f"|| total_loss={total_loss:.4f}"
            )
        else:
            # If not using diffusion or not in training, just do the normal CE
            total_loss = ce_loss

        return DiffusionSeq2SeqLMOutput(
            loss=total_loss,
            logits=seq2seq_outputs.logits,
            ce_loss=ce_loss,
            diffusion_loss=diffusion_loss,
            past_key_values=seq2seq_outputs.past_key_values,
            decoder_hidden_states=seq2seq_outputs.decoder_hidden_states,
            decoder_attentions=seq2seq_outputs.decoder_attentions,
            cross_attentions=seq2seq_outputs.cross_attentions,
            encoder_last_hidden_state=seq2seq_outputs.encoder_last_hidden_state,
            encoder_hidden_states=seq2seq_outputs.encoder_hidden_states,
            encoder_attentions=seq2seq_outputs.encoder_attentions,
        )


    def _decode_latent_to_text(self, latent: torch.Tensor) -> list:
        """
        Used by apply_embedding_guidance in p_sample_loop to do partial decode 
        from a single-step latent.
        """
        B, L, D = latent.shape

        hidden = self._diffusion_expand(latent)
        attn_mask = torch.ones((B, L), dtype=torch.long, device=latent.device)

        # short, deterministic decode => avoid chaos
        out_ids = self.encoder_decoder.generate(
            inputs_embeds=hidden,
            attention_mask=attn_mask,
            max_new_tokens=15,
            do_sample=False,
            num_beams=1,
        )
        texts = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return texts
