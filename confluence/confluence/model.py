"""Relational transformer model for database learning.

Implements:
- ValueEncoder: column-name encoding + type-specific value encoding + null/target gating
- DecoderHeads: null, numerical, boolean, timestamp, categorical
- RelationalTransformer: full model combining encoder, transformer layers, and decoder
"""

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from confluence.config import ModelConfig
from confluence.layers import (
    TransformerLayer,
    ZeroCenteredRMSNorm,
)

# Semantic type constants (must match Rust SemanticType enum)
STYPE_IDENTIFIER = 0
STYPE_NUMERICAL = 1
STYPE_TIMESTAMP = 2
STYPE_BOOLEAN = 3
STYPE_CATEGORICAL = 4
STYPE_TEXT = 5
STYPE_IGNORED = 6


class ModelOutput(NamedTuple):
    """Output of the relational transformer forward pass."""

    h: jnp.ndarray  # [B, S, D] final hidden states
    null_logits: jnp.ndarray  # [B, S] null head (raw logits)
    num_preds: jnp.ndarray  # [B, S] numerical predictions (z-score scale)
    bool_logits: jnp.ndarray  # [B, S] boolean head (raw logits)
    ts_preds: jnp.ndarray  # [B, S, 15] timestamp predictions
    cat_preds: jnp.ndarray  # [B, S, D] categorical projections


class ValueEncoder(nn.Module):
    """Encode cell values into initial hidden states h0.

    h0 = RMSNorm(col_enc + val_final)

    Where col_enc = Linear(D_t -> D)(column_embedding_table[column_ids])
    and val_final is the type-specific value, gated by null and target masks.
    """

    config: ModelConfig

    @nn.compact
    def __call__(self, batch, col_emb_table, cat_emb_table, text_batch_emb):
        """
        Args:
            batch: dict of batch tensors from the sampler.
            col_emb_table: [C, D_t] column-name embeddings (GPU-resident, bf16).
            cat_emb_table: [Vc, D_t] categorical embeddings (GPU-resident, bf16).
            text_batch_emb: [U, D_t] per-batch text embeddings (bf16).

        Returns:
            h0: [B, S, D] initial hidden states.
        """
        cfg = self.config
        d = cfg.d_model
        d_t = cfg.d_text

        semantic_types = batch["semantic_types"]  # [B, S] int8
        column_ids = batch["column_ids"]  # [B, S] int32
        is_null = batch["is_null"].astype(jnp.float32)  # [B, S]
        is_target = batch["is_target"].astype(jnp.float32)  # [B, S]
        is_padding = batch["is_padding"].astype(jnp.float32)  # [B, S]

        # Column-name encoding: lookup + project
        col_raw = col_emb_table[column_ids]  # [B, S, D_t]
        col_raw = col_raw.astype(jnp.bfloat16)
        col_enc = nn.Dense(d, use_bias=True, name="column_name_encoder")(col_raw)

        # Type-specific value encoders (all run in parallel on dense layout)
        # Identifier: learned constant
        identifier_emb = self.param(
            "identifier_emb",
            nn.initializers.normal(stddev=0.02),
            (d,),
        )
        id_val = jnp.broadcast_to(identifier_emb, col_enc.shape)

        # Numerical: Linear(1 -> D)
        num_input = batch["numeric_values"][..., None].astype(jnp.bfloat16)  # [B,S,1]
        num_val = nn.Dense(d, use_bias=True, name="numerical_encoder")(num_input)

        # Timestamp: Linear(15 -> D)
        ts_input = batch["timestamp_values"].astype(jnp.bfloat16)  # [B, S, 15]
        ts_val = nn.Dense(d, use_bias=True, name="timestamp_encoder")(ts_input)

        # Boolean: Embedding(2, D)
        bool_input = batch["bool_values"].astype(jnp.int32)  # [B, S]
        bool_emb_table = self.param(
            "boolean_encoder",
            nn.initializers.normal(stddev=0.02),
            (cfg.num_bool_values, d),
        )
        bool_val = bool_emb_table[bool_input]  # [B, S, D]

        # Categorical: Linear(D_t -> D) on frozen embedding lookup
        cat_ids = batch["categorical_embed_ids"]  # [B, S] uint32
        cat_raw = cat_emb_table[cat_ids].astype(jnp.bfloat16)  # [B, S, D_t]
        cat_val = nn.Dense(d, use_bias=True, name="categorical_encoder")(cat_raw)

        # Text: Linear(D_t -> D) on batch-local text embedding lookup
        text_ids = batch["text_embed_ids"]  # [B, S] uint32
        # Clamp text_ids to valid range for text_batch_emb
        max_text_idx = text_batch_emb.shape[0] - 1
        safe_text_ids = jnp.clip(text_ids, 0, jnp.maximum(max_text_idx, 0))
        text_raw = text_batch_emb[safe_text_ids].astype(jnp.bfloat16)  # [B, S, D_t]
        text_val = nn.Dense(d, use_bias=True, name="text_encoder")(text_raw)

        # Dense type dispatch: sum of one_hot(stype, t) * encoder_t(values)
        stypes = semantic_types.astype(jnp.int32)  # [B, S]
        type_one_hot = jax.nn.one_hot(stypes, cfg.num_semantic_types)  # [B,S,7]

        # Stack all encoders: [B, S, 7, D]
        all_vals = jnp.stack(
            [
                id_val,  # 0: Identifier
                num_val,  # 1: Numerical
                ts_val,  # 2: Timestamp
                bool_val,  # 3: Boolean
                cat_val,  # 4: Categorical
                text_val,  # 5: Text
                jnp.zeros_like(id_val),  # 6: Ignored (should never appear)
            ],
            axis=2,
        )  # [B, S, 7, D]

        # Select: [B, S, D]
        raw_val = jnp.einsum("bst,bstd->bsd", type_one_hot, all_vals)

        # Null gating: replace value with null_emb if is_null
        null_emb = self.param("null_emb", nn.initializers.normal(stddev=0.02), (d,))
        is_null_expanded = is_null[..., None]  # [B, S, 1]
        val_or_null = is_null_expanded * null_emb + (1.0 - is_null_expanded) * raw_val

        # Target masking: replace with mask_emb if is_target (priority over null)
        mask_emb = self.param("mask_emb", nn.initializers.normal(stddev=0.02), (d,))
        is_target_expanded = is_target[..., None]  # [B, S, 1]
        val_final = is_target_expanded * mask_emb + (1.0 - is_target_expanded) * val_or_null

        # Combine column encoding + value encoding
        h0 = ZeroCenteredRMSNorm(eps=cfg.rms_norm_eps, name="h0_norm")(col_enc + val_final)

        # Note: we intentionally do NOT zero out padding positions here.
        # Attention masks already exclude padding from affecting non-padding
        # positions, and zeroing creates exact-zero hidden states whose
        # gradients through L2 norm / RMSNorm produce NaN.

        return h0


class DecoderHeads(nn.Module):
    """All five decoder heads, run unconditionally on every position."""

    config: ModelConfig

    @nn.compact
    def __call__(self, h):
        """
        Args:
            h: [B, S, D] final hidden states.

        Returns:
            ModelOutput with all prediction tensors.
        """
        d = self.config.d_model

        null_logits = nn.Dense(1, use_bias=True, name="null_head")(h).squeeze(-1)
        num_preds = nn.Dense(1, use_bias=True, name="numerical_decoder")(h).squeeze(-1)
        bool_logits = nn.Dense(1, use_bias=True, name="boolean_decoder")(h).squeeze(-1)
        ts_preds = nn.Dense(self.config.d_time, use_bias=True, name="timestamp_decoder")(h)
        cat_preds = nn.Dense(d, use_bias=True, name="categorical_decoder")(h)

        return ModelOutput(
            h=h,
            null_logits=null_logits,
            num_preds=num_preds,
            bool_logits=bool_logits,
            ts_preds=ts_preds,
            cat_preds=cat_preds,
        )


def _csr_to_dense_mask(
    row_ptr: jnp.ndarray,
    seq_offsets: jnp.ndarray,
    col_idx: jnp.ndarray,
    s: int,
) -> jnp.ndarray:
    """Decode packed CSR payload into dense [B,S,S] bool."""
    positions = jnp.arange(col_idx.shape[0], dtype=jnp.int32)
    col_idx_i32 = col_idx.astype(jnp.int32)

    def decode_one(rp: jnp.ndarray, seq_start: jnp.ndarray, seq_end: jnp.ndarray) -> jnp.ndarray:
        seq_start_i32 = seq_start.astype(jnp.int32)
        seq_nnz = (seq_end - seq_start).astype(jnp.int32)
        local_pos = positions - seq_start_i32
        valid = (local_pos >= 0) & (local_pos < seq_nnz)
        local_pos = jnp.where(valid, local_pos, 0)
        row_ids = jnp.searchsorted(rp.astype(jnp.int32), local_pos, side="right") - 1
        rows = jnp.where(valid, row_ids, 0)
        cols = jnp.where(valid, col_idx_i32, 0)
        dense = jnp.zeros((s, s), dtype=jnp.bool_)
        return dense.at[rows, cols].set(valid)

    return jax.vmap(decode_one)(row_ptr, seq_offsets[:-1], seq_offsets[1:])


def build_attention_masks(batch):
    """Build dense masks from CSR transport."""
    s = int(batch["seq_row_ids"].shape[1])
    return (
        _csr_to_dense_mask(
            batch["outbound_csr_row_ptr"],
            batch["outbound_csr_seq_offsets"],
            batch["outbound_csr_col_idx"],
            s,
        ),
        _csr_to_dense_mask(
            batch["inbound_csr_row_ptr"],
            batch["inbound_csr_seq_offsets"],
            batch["inbound_csr_col_idx"],
            s,
        ),
        _csr_to_dense_mask(
            batch["column_csr_row_ptr"],
            batch["column_csr_seq_offsets"],
            batch["column_csr_col_idx"],
            s,
        ),
    )


class RelationalTransformer(nn.Module):
    """Full relational transformer model.

    Forward pass:
    1. Value encoding -> h0
    2. N transformer layers (outbound, inbound, column attention + FFN)
    3. Final RMSNorm
    4. Decoder heads
    """

    config: ModelConfig

    @nn.compact
    def __call__(self, batch, col_emb_table, cat_emb_table):
        cfg = self.config

        # Get text embeddings from batch (already transferred as part of batch)
        text_emb_u16 = batch["text_batch_embeddings"]  # [U, D_t] as uint16
        # Convert from uint16 bit representation back to bf16
        # The sampler stores f16 as u16 bits; we reinterpret as bf16-compatible
        # For now, cast uint16 -> float32 -> bfloat16 via jax.lax.bitcast_convert_type
        # Actually, the bits are IEEE float16, not bfloat16.
        # We need to convert f16 bits -> f32 first.
        text_batch_emb = jnp.array(text_emb_u16, dtype=jnp.float16).astype(jnp.bfloat16)

        # Value encoding
        h = ValueEncoder(config=cfg, name="value_encoder")(batch, col_emb_table, cat_emb_table, text_batch_emb)

        # Build attention masks
        outbound_mask, inbound_mask, column_mask = build_attention_masks(batch)

        # Transformer layers
        for i in range(cfg.n_layers):
            h = TransformerLayer(config=cfg, layer_idx=i, name=f"layer_{i}")(
                h,
                outbound_mask,
                inbound_mask,
                column_mask,
            )

        # Final RMSNorm
        h = ZeroCenteredRMSNorm(eps=cfg.rms_norm_eps, name="final_norm")(h)

        # Decoder heads
        output = DecoderHeads(config=cfg, name="decoder")(h)

        return output
