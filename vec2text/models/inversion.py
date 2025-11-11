import copy
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import LogitsProcessor, GenerationConfig
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
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


# Define a custom LogitsProcessor for partial-chunk embedding checks
class SimilarityGuidedProcessor(LogitsProcessor):
    """
    A custom processor that, every `checkpoint_interval` tokens, re-embeds the partially generated
    text and adjusts next-token scores based on the similarity to a target embedding.

    Args:
        model: InversionModel instance (used to call embedding_model)
        tokenizer: the HF tokenizer for the decoder
        target_emb: shape (batch_size, emb_dim) or (emb_dim,) of the target embedding(s)
        alpha, beta: weighting factors for combining LM log-probs and embedding similarity
        checkpoint_interval: how many tokens between each embedding check
        batch_size: how many examples in the batch
    """
    def __init__(
        self,
        model: "InversionModel",
        tokenizer: transformers.PreTrainedTokenizer,
        target_emb: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
        checkpoint_interval: int = 10,
        batch_size: int = 1
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.target_emb = target_emb
        self.alpha = alpha
        self.beta = beta
        self.checkpoint_interval = checkpoint_interval
        self.batch_size = batch_size

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Called automatically by HF beam search at each decoding step.
            input_ids: (batch_size*num_beams, cur_seq_len)
            scores:    (batch_size*num_beams, vocab_size) => raw logits for the next token
        Return:
            new scores (same shape) after modifying them based on embedding similarity.
        """
        seq_len = input_ids.shape[1]
        # Only do an embedding check every `checkpoint_interval` tokens
        if (seq_len % self.checkpoint_interval) != 0:
            return scores  # no change

        # 1) Decode partial text for all beams
        partial_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        device = scores.device
        # 2) Single batch call to the embedding model
        emb_inp = self.model.embedder_tokenizer(
            partial_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            partial_embs = self.model.call_embedding_model(
                input_ids=emb_inp["input_ids"],
                attention_mask=emb_inp["attention_mask"],
            )  # shape: [batch_size*num_beams, emb_dim]

        # If target_emb is shape [emb_dim], repeat for all beams
        if len(self.target_emb.shape) == 1:
            target_expand = self.target_emb.unsqueeze(0).expand_as(partial_embs)
        else:
            # shape [B, emb_dim] => we might need to chunk by (num_beams).
            # For simple case: assume no beam reordering, so chunk_size = partial_embs.size(0) // batch_size
            chunk_size = partial_embs.size(0) // self.batch_size
            chunks = []
            for i in range(self.batch_size):
                # repeat row i of target_emb for chunk_size times
                t_i = self.target_emb[i].unsqueeze(0).expand(chunk_size, -1)
                chunks.append(t_i)
            target_expand = torch.cat(chunks, dim=0)

        cos_sim = F.cosine_similarity(partial_embs, target_expand, dim=-1)
        # We'll combine with LM log-probs => log_softmax(scores)
        log_probs = torch.log_softmax(scores, dim=-1)

        # If alpha != 1.0, do alpha * log_probs
        # We'll do: new_log_probs = alpha*log_probs + beta*cos_sim
        # We apply cos_sim as a constant shift across the entire vocab distribution for that beam
        new_log_probs = self.alpha * log_probs + self.beta * cos_sim.view(-1, 1)

        return new_log_probs


# ---------------------------------------------------------------------
# 2) The main InversionModel with new HF-based generation
# ---------------------------------------------------------------------
class InversionModel(transformers.PreTrainedModel):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively, now using HF's built-in beam search if desired.
    """

    config_class = InversionConfig
    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    tokenizer: transformers.PreTrainedTokenizer
    embedding_transform: nn.Module

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        embedder_model_api = config.embedder_model_api
        embedder_fake_with_zeros = config.embedder_fake_with_zeros
        use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input
        encoder_dropout_disabled = config.encoder_dropout_disabled
        decoder_dropout_disabled = config.decoder_dropout_disabled
        embeddings_from_layer_n = config.embeddings_from_layer_n

        # 1) Load your underlying seq2seq model
        encoder_decoder = load_encoder_decoder(
            model_name=config.model_name_or_path,
            lora=config.use_lora,
        )

        # 2) Load the embedder model + tokenizer
        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name,
            torch_dtype=config.embedder_torch_dtype
        )

        # 3) Load your main tokenizer
        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )

        num_repeat_tokens = config.num_repeat_tokens
        embedder_no_grad = config.embedder_no_grad

        self.encoder_decoder = encoder_decoder
        self.num_repeat_tokens = num_repeat_tokens

        # Decide dims
        if embedder_model_api:
            # e.g. OpenAI embeddings are dimension=1536, or as needed
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

        # Linear transform from embedder dimension -> repeated T5 encoder input
        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
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
        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        self.embedder_fake_with_zeros = embedder_fake_with_zeros
        self.embedding_transform_strategy = "repeat"
        self.embeddings_from_layer_n = embeddings_from_layer_n
        self.noise_level = vars(config).get("embedder_gaussian_noise_level", 0.0)

        if self.embedder_no_grad:
            for param in self.embedder.parameters():
                param.requires_grad = False
            self.embedder.eval()

    @property
    def embedder_device(self) -> torch.device:
        return next(self.embedder.parameters()).device

    def _process_embedder_output(
        self,
        outputs: transformers.modeling_outputs.BaseModelOutput,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Helper to extract or pool embeddings from a huggingface model output."""
        if hasattr(outputs, "pooler_output") and (outputs.pooler_output is not None):
            return outputs.pooler_output
        else:
            if self.embeddings_from_layer_n is not None:
                assert hasattr(outputs, "hidden_states"), (
                    "output missing hidden states - did you set output_hidden_states=True?"
                )
                hidden_state = outputs.hidden_states[self.embeddings_from_layer_n]
                embeddings = mean_pool(hidden_state, attention_mask)
            else:
                hidden_state = outputs.last_hidden_state
                embeddings = mean_pool(hidden_state, attention_mask)
            return embeddings

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Call the embedding model, which might be remote (embedder_model_api) or local."""
        if self.embedder_no_grad:
            self.embedder.eval()

        if self.embedder_fake_with_zeros:
            batch_size = input_ids.shape[0]
            return torch.zeros(
                (batch_size, self.embedder_dim),
                dtype=torch.float32,
                device=self.embedder_device,
            )
        elif self.embedder_model_api:
            # e.g. OpenAI or other remote
            embeddings = embed_api(
                input_ids=input_ids,
                embedder_tokenizer=self.embedder_tokenizer,
                api_name=self.embedder_model_api,
            )
        elif isinstance(self.embedder, SentenceTransformer):
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids
            model_output = self.embedder(model_inputs)
            embeddings = model_output["sentence_embedding"]
        else:
            model_output = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self._process_embedder_output(model_output, attention_mask)

        if self.training and self.noise_level > 0:
            embeddings += self.noise_level * torch.randn(
                embeddings.shape, device=embeddings.device
            )
        return embeddings

    def embed_and_project(
        self,
        embedder_input_ids: Optional[torch.Tensor],
        embedder_attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs embedding model (or uses frozen) + transforms into T5 input embeddings."""
        # Step 1: figure out which embeddings to use
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert embeddings.ndim == 2, f"frozen_embeddings must be [B, emb_dim], got {embeddings.shape}"
        else:
            # We must embed from scratch
            if self.embedder_no_grad:
                with torch.no_grad():
                    embeddings = self.call_embedding_model(
                        input_ids=embedder_input_ids,
                        attention_mask=embedder_attention_mask,
                    )
            else:
                embeddings = self.call_embedding_model(
                    input_ids=embedder_input_ids,
                    attention_mask=embedder_attention_mask,
                )

        # Step 2: transform into repeated sequence if "repeat" strategy
        if self.embedding_transform_strategy == "repeat":
            if embeddings.dtype != self.dtype:
                embeddings = embeddings.to(self.dtype)
            repeated_embeddings = self.embedding_transform(embeddings)
            # shape => [B, encoder_hidden_dim * num_repeat_tokens]
            # reshape => [B, num_repeat_tokens, encoder_hidden_dim]
            B = repeated_embeddings.shape[0]
            repeated_embeddings = repeated_embeddings.reshape(
                B, self.num_repeat_tokens, -1
            )
            inputs_embeds = repeated_embeddings  # [B, seq_len, hidden_dim]
        else:
            raise NotImplementedError("Only 'repeat' strategy is implemented currently.")

        # Step 3: attention_mask = ones for that entire sequence
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device
        )
        return inputs_embeds, attention_mask

    # ---------------------------------------------------------------------
    # 2A) Single-pass HF-based generate
    # ---------------------------------------------------------------------
    def _hf_generate_single_pass(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
        guided: bool,
        checkpoint_interval: int,
        alpha: float,
        beta: float
    ) -> torch.Tensor:
        """
        Uses HF's .generate() with optional custom LogitsProcessor for partial-chunk embedding guidance.
        Equivalent to your old single pass (guided or not).
        """
        # 1) Prepare T5 encoder inputs
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )

        # 2) See if we have a target embedding to guide toward
        target_emb = inputs.get("target_embedding", None)
        if target_emb is not None and guided:
            # We have a target embedding & want partial-chunk checks
            if len(target_emb.shape) == 1:
                target_emb = target_emb.unsqueeze(0)  # shape [1, emb_dim]
            batch_size = target_emb.shape[0]
            # Build the processor
            processor = SimilarityGuidedProcessor(
                model=self,
                tokenizer=self.tokenizer,
                target_emb=target_emb,
                alpha=alpha,
                beta=beta,
                checkpoint_interval=checkpoint_interval,
                batch_size=batch_size
            )

            # Add mild repetition penalty
            repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=1.1)

            # Build a GenerationConfig
            gen_config = GenerationConfig.from_pretrained(self.encoder_decoder.config)
            # Update gen_config with user overrides
            for k, v in generation_kwargs.items():
                setattr(gen_config, k, v)
            gen_config.logits_processor = [
                processor,
                repetition_processor
                ]
            outputs = self.encoder_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                generation_config=gen_config
            )
            return outputs
        else:
            # either not guided, or no target embedding => do normal .generate()
            return self.encoder_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs
            )

    # ---------------------------------------------------------------------
    # 2B) Multi-pass approach using HF-based single_pass each time
    # ---------------------------------------------------------------------
    def _hf_generate_multipass(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
        num_passes: int,
        checkpoint_interval: int,
        alpha: float,
        beta: float,
        guided: bool,
    ) -> torch.Tensor:
        """
        Multi-pass iterative refinement:
          - in each pass, we do _hf_generate_single_pass()
          - we embed the best final output
          - feed as "frozen_embeddings" for the next pass
        """
        best_output_ids = None
        new_inputs = dict(inputs)

        for pass_idx in range(num_passes):
            outputs = self._hf_generate_single_pass(
                inputs=new_inputs,
                generation_kwargs=generation_kwargs,
                guided=guided,
                checkpoint_interval=checkpoint_interval,
                alpha=alpha,
                beta=beta,
            )
            best_output_ids = outputs

            # Re-embed the final best text, pass it as frozen embeddings for next iteration
            txts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            device = self.device
            new_embs = []
            for txt in txts:
                emb_inp = self.embedder_tokenizer(txt, return_tensors="pt").to(device)
                with torch.no_grad():
                    e = self.call_embedding_model(
                        input_ids=emb_inp["input_ids"],
                        attention_mask=emb_inp["attention_mask"]
                    )
                new_embs.append(e)
            new_embs = torch.cat(new_embs, dim=0)
            new_inputs["frozen_embeddings"] = new_embs

        return best_output_ids

    # ---------------------------------------------------------------------
    # 3) Public .generate() method
    # ---------------------------------------------------------------------
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
        guided: bool = False,
        checkpoint_interval: int = 10,
        beam_size: int = 5,
        multipass: int = 1,
        alpha: float = 1.0,
        beta: float = 1.0,
        length_penalty: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        A unified .generate() that does:
          - Normal single-pass: (guided=False, multipass=1)
          - Single-pass partial-chunk logic: (guided=True, multipass=1)
          - Multi-pass iterative refinement: (guided=any, multipass>1)

        We now rely on HF beam search with an optional SimilarityGuidedProcessor.
        """
        # Pull out any custom args from generation_kwargs if present
        if "guided" in generation_kwargs:
            guided = generation_kwargs.pop("guided")
        if "checkpoint_interval" in generation_kwargs:
            checkpoint_interval = generation_kwargs.pop("checkpoint_interval")
        if "beam_size" in generation_kwargs:
            beam_size = generation_kwargs.pop("beam_size")
        if "multipass" in generation_kwargs:
            multipass = generation_kwargs.pop("multipass")
        if "alpha" in generation_kwargs:
            alpha = generation_kwargs.pop("alpha")
        if "beta" in generation_kwargs:
            beta = generation_kwargs.pop("beta")
        if "length_penalty" in generation_kwargs:
            length_penalty = generation_kwargs.pop("length_penalty")

        # We'll feed HF's generate() some standard arguments:
        generation_kwargs.setdefault("num_beams", beam_size)
        generation_kwargs.setdefault("length_penalty", length_penalty)

        # Decide single-pass vs. multi-pass
        if multipass <= 1:
            # Single pass
            return self._hf_generate_single_pass(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                guided=guided,
                checkpoint_interval=checkpoint_interval,
                alpha=alpha,
                beta=beta,
            )
        else:
            # Multi-pass
            return self._hf_generate_multipass(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                num_passes=multipass,
                checkpoint_interval=checkpoint_interval,
                alpha=alpha,
                beta=beta,
                guided=guided,
            )

    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Standard forward for training
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=frozen_embeddings,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )

