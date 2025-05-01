from __future__ import annotations

import copy
import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from transformers.modeling_outputs import Seq2SeqLMOutput
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

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

@dataclass
class InvOutput(Seq2SeqLMOutput):
    extra_losses: Dict[str, torch.Tensor] = None

# ===========================================================================
# Diffusion utilities -------------------------------------------------------
# ===========================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Classic 1‑D sinusoidal embedding used in diffusion works."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device, half = t.device, self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / half)
        ang = t[:, None].float() * freqs[None]
        emb = torch.cat((ang.sin(), ang.cos()), dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat((emb, emb.new_zeros((t.size(0), 1))), dim=1)
        return emb


class TransformerDenoiser(nn.Module):
    """Small GPT‑style network predicting xₜ₋₁ from xₜ and the target embedding."""

    def __init__(self, vocab_size: int, hidden_size: int = 768, n_layers: int = 6):
        super().__init__()
        cfg = transformers.GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=n_layers,
            n_head=hidden_size // 64,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            pad_token_id=0,
        )
        self.backbone = transformers.GPT2LMHeadModel(cfg)
        self.time_emb = SinusoidalTimeEmbedding(hidden_size)
        self.cond_proj = nn.Linear(1536, hidden_size)

    def forward(
        self,
        x_t: torch.Tensor,               # (B, L)
        t: torch.Tensor,                 # (B,)
        cond: Optional[torch.Tensor],    # (B, 1536) or None
    ) -> torch.Tensor:                   # logits (B, L, V)
        tok_emb = self.backbone.transformer.wte(x_t)
        pos_emb = self.backbone.transformer.wpe(torch.arange(x_t.size(1), device=x_t.device)[None])
        h = tok_emb + pos_emb + self.time_emb(t)[:, None, :]
        if cond is not None:
            h = h + self.cond_proj(cond)[:, None, :]
        return self.backbone(inputs_embeds=h).logits


class DiffusionSampler(nn.Module):
    """DDIM sampler with classifier‑free guidance  and rejection beam."""

    def __init__(
        self,
        inv_model: "InversionModel",
        num_steps: int = 20,
        guidance_scale: float = 2.0,
        num_candidates_default: int = 1,
    ):
        super().__init__()

        self._call_embed = inv_model.call_embedding_model  # function handle
        self.tokenizer = inv_model.tokenizer               # not an nn.Module
        self.embedder_tokenizer = inv_model.embedder_tokenizer

        # hyper‑params
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.num_candidates_default = num_candidates_default
        self.device = next(inv_model.parameters()).device

        # denoiser network
        self.denoiser = TransformerDenoiser(self.tokenizer.vocab_size).to(self.device)

        # simple linear beta schedule  → deterministic DDIM
        self.register_buffer(
            "betas", torch.linspace(0.0001, 0.9, num_steps, dtype=torch.float32)
        )
        self.mask_id = self.tokenizer.mask_token_id or self.tokenizer.pad_token_id

    # ----------------------------- public API -----------------------------

    @torch.no_grad()
    def sample(
        self,
        target_emb: torch.Tensor,            # (B, 1536)
        *,
        max_length: int,
        num_candidates: int | None = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns best_sequences, best_cosines."""
        if seed is not None:
            torch.manual_seed(seed)
        num_candidates = num_candidates or self.num_candidates_default
        B = target_emb.size(0)
        device = self.device

        def run_chain():
            x_t = torch.full((B, max_length), self.mask_id, device=device)
            for step in reversed(range(self.num_steps)):
                t = torch.full((B,), step, device=device, dtype=torch.long)
                logits_cond = self.denoiser(x_t, t, cond=target_emb)
                logits_un = self.denoiser(x_t, t, cond=None)
                logits = logits_un + self.guidance_scale * (logits_cond - logits_un)
                x_t = logits.argmax(-1)
            return x_t

        seqs, cosines = [], []
        for _ in range(num_candidates):
            seq = run_chain()
            seqs.append(seq)
            cosines.append(self._embed(seq).cosine_similarity(target_emb).to(device))
        seqs = torch.stack(seqs, dim=1)   # (B,N,L)
        cosines = torch.stack(cosines, 1) # (B,N)
        best_idx = cosines.argmax(1)
        best_seq = seqs[torch.arange(B, device=device), best_idx]
        best_cos = cosines.max(1).values
        return best_seq, best_cos

    # ------------------------- helpers -----------------------------------

    def _embed(self, seq: torch.Tensor) -> torch.Tensor:
        texts = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
        tok = self.embedder_tokenizer(
            texts, padding=True, return_tensors="pt", truncation=False
        ).to(self.device)
        return self._call_embed(
            input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]
        )

class InversionModel(transformers.PreTrainedModel):
    """Extends the original autoregressive inverter with optional discrete diffusion sampling."""

    config_class = InversionConfig

    # --------------------------- init -------------------------------------

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        encoder_decoder = load_encoder_decoder(config.model_name_or_path, lora=config.use_lora)
        embedder, embedder_tok = load_embedder_and_tokenizer(
            name=config.embedder_model_name, torch_dtype=config.embedder_torch_dtype
        )
        tokenizer = load_tokenizer(config.model_name_or_path, max_length=config.max_seq_length)

        self.encoder_decoder = encoder_decoder
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.embedder_tokenizer = embedder_tok
        # freeze embedder if requested
        self.embedder_no_grad = getattr(config, "embedder_no_grad", True)
        if self.embedder_no_grad:
            for p in self.embedder.parameters():
                p.requires_grad = False
            self.embedder.eval()
        self.embedder_model_api = config.embedder_model_api
        self.embedder_dim = 1536 if self.embedder_model_api else (
            embedder.get_sentence_embedding_dimension()
            if isinstance(embedder, SentenceTransformer)
            else embedder.config.hidden_size
        )
        self.num_repeat_tokens = config.num_repeat_tokens
        self.noise_level = vars(config).get("embedder_gaussian_noise_level", 0.0)
        # simple projection → kept for backwards compat even if diffusion is used
        h_dim = self.encoder_decoder.config.hidden_size
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, self.embedder_dim),
            nn.GELU(),
            nn.Linear(self.embedder_dim, h_dim * self.num_repeat_tokens),
        )

        # learnable log-variances for uncertainty weighting
        self.loss_logvars = nn.ParameterDict({
            "emb" : nn.Parameter(torch.zeros(())),
            "nce" : nn.Parameter(torch.zeros(())),
            "margin" : nn.Parameter(torch.zeros(())),
        })

        # we will lazily create the diffusion sampler so DDP stops complaining
        # about unused params
        self._diffusion_cfg = {
            "enabled": getattr(config, "use_diffusion", False),
            "num_steps": getattr(config, "diffusion_num_steps", 20),
            "guidance_scale": getattr(config, "diffusion_guidance_scale", 2.0),
            "num_candidates": getattr(config, "diffusion_num_candidates", 1),
        }

    # Core embedder routine
    def call_embedding_model(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.embedder_model_api:
            return embed_api(input_ids=input_ids, embedder_tokenizer=self.embedder_tokenizer, api_name=self.embedder_model_api)
        elif isinstance(self.embedder, SentenceTransformer):
            out = self.embedder({"input_ids": input_ids, "attention_mask": attention_mask})
            emb = out["sentence_embedding"]
        else:
            out = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
            emb = mean_pool(out.last_hidden_state, attention_mask)
        if self.training and self.noise_level > 0:
            emb = emb + self.noise_level * torch.randn_like(emb)
        return emb

    def _embed_and_project(self, *, frozen_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rep = self.embedding_transform(frozen_embeddings).view(
            frozen_embeddings.size(0), self.num_repeat_tokens, -1
        )
        attn = torch.ones(rep.shape[:2], device=rep.device)
        return rep, attn

    def generate_greedy(self, *, frozen_embeddings: torch.Tensor, **gen_kwargs) -> torch.Tensor:
        embeds, attn = self._embed_and_project(frozen_embeddings=frozen_embeddings)
        return self.encoder_decoder.generate(inputs_embeds=embeds, attention_mask=attn, **gen_kwargs)

    # ------------------------------------------------------------------
    # PUBLIC GENERATE ---------------------------------------------------
    # ------------------------------------------------------------------

    def generate(self, inputs: Dict[str, torch.Tensor], generation_kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:  # noqa: D401
        """Unified entry‑point.  Pass ``use_diffusion=True`` in
        ``generation_kwargs`` to switch to the diffusion sampler; otherwise the
        original autoregressive path is used.
        """
        generation_kwargs = copy.copy(generation_kwargs)
        use_diffusion = generation_kwargs.pop("use_diffusion", False) and self._diffusion_cfg["enabled"]

        # obtain target embedding
        if "frozen_embeddings" in inputs:
            frozen = inputs["frozen_embeddings"].to(self.device)
        else:
            frozen = self.call_embedding_model(
                input_ids=inputs["embedder_input_ids"].to(self.device),
                attention_mask=inputs["embedder_attention_mask"].to(self.device),
            )

        if use_diffusion:
            # lazily create sampler to keep params out of DDP graph
            if not hasattr(self, "_diffusion_sampler"):
                self._diffusion_sampler = DiffusionSampler(
                    self,
                    num_steps=self._diffusion_cfg["num_steps"],
                    guidance_scale=self._diffusion_cfg["guidance_scale"],
                    num_candidates_default=self._diffusion_cfg["num_candidates"],
                )
                for p in self._diffusion_sampler.parameters():
                    p.requires_grad=False
            max_len = generation_kwargs.get("max_length", self.config.max_seq_length)
            num_cand = generation_kwargs.pop("num_candidates", None)
            seq, _ = self._diffusion_sampler.sample(
                target_emb=frozen,
                max_length=max_len,
                num_candidates=num_cand,
            )
            return seq
        else:
            return self.generate_greedy(frozen_embeddings=frozen, **generation_kwargs)
    
    def forward(
        self,
        *,
        embedder_input_ids: Optional[torch.Tensor] = None,
        embedder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Default training-forward path (always autoregressive)."""
        # Unused: input_ids, attention_mask
        if frozen_embeddings is None:
            assert embedder_input_ids is not None, "Need embedder_input_ids when frozen_embeddings absent"
            frozen_embeddings = self.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            )
        embeds, attn = self._embed_and_project(frozen_embeddings=frozen_embeddings)
        dec_out = self.encoder_decoder(
            inputs_embeds=embeds,
            attention_mask=attn,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )

        loss = dec_out.loss # cross-entropy
        extra_losses = {}
        # ============ auxiliary losses (train mode only) =================
        if self.training and labels is not None:
            B, D = frozen_embeddings.size()
            pad_id = self.tokenizer.pad_token_id

            # sentence-level cosine loss
            with torch.no_grad():
                pred_ids = dec_out.logits.argmax(-1)
                pred_emb = self.call_embedding_model(
                    input_ids=pred_ids,
                    attention_mask=(pred_ids != pad_id)
                )
            target_emb = frozen_embeddings # (B, D)
            cos_sim    = torch.nn.functional.cosine_similarity(pred_emb, target_emb, dim=1)
            l_emb      = (1.0 - cos_sim).mean()

            # In-batch InfoNCE
            tau   = self.config.nce_temperature
            sim   = torch.nn.functional.cosine_similarity(
                        pred_emb.unsqueeze(1), target_emb.unsqueeze(0), dim=-1)  # (B, B)
            l_nce = (-torch.log_softmax(sim / tau, dim=1).diag()).mean()

            # margin loss on bottleneck
            h = self.embedding_transform[0](frozen_embeddings)
            h = self.embedding_transform[1](h)             # GELU
            neg = target_emb[torch.randperm(B, device=h.device)]
            cos_pos = torch.nn.functional.cosine_similarity(h, target_emb, dim=1)
            cos_neg = torch.nn.functional.cosine_similarity(h, neg,        dim=1)
            m = self.config.margin_m
            l_margin = torch.nn.functional.relu(m - cos_pos + cos_neg).mean()

            # uncertainty-weighted sum
            def uw(term, logvar):
                return 0.5 * torch.exp(-logvar) * term + 0.5 * logvar
            
            loss = (
                loss
                + uw(l_emb,    self.loss_logvars["emb"])
                + uw(l_nce,    self.loss_logvars["nce"])
                + uw(l_margin, self.loss_logvars["margin"])
            )
            dec_out.loss = loss

            extra_losses = {
                "ce_loss"      : dec_out.loss.detach(),
                "emb_loss"     : l_emb.detach(),
                "nce_loss"     : l_nce.detach(),
                "margin_loss"  : l_margin.detach(),
                "logvar_emb"   : self.loss_logvars["emb"].detach(),
                "logvar_nce"   : self.loss_logvars["nce"].detach(),
                "logvar_margin": self.loss_logvars["margin"].detach(),
            }
        output = InvOutput(**dec_out.to_dict(), extra_losses=extra)
        print(f"keys in output from forward(): {output.keys()}")
        print(f"accessing output.extra_losses in forward(): {output.extra_losses}")
        return output