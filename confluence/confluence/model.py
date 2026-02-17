"""Relational transformer model for database learning.

Implements:
- ValueEncoder: column-name encoding + type-specific value encoding + null/target gating
- DecoderHeads: null, numerical, boolean, timestamp, categorical
- RelationalTransformer: full model combining encoder, transformer layers, and decoder
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from confluence.config import ModelConfig
from confluence.layers import (
    ZeroCenteredRMSNorm,
    TransformerLayer,
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
    h: jnp.ndarray            # [B, S, D] final hidden states
    null_logits: jnp.ndarray  # [B, S] null head (raw logits)
    num_preds: jnp.ndarray    # [B, S] numerical predictions (z-score scale)
    bool_logits: jnp.ndarray  # [B, S] boolean head (raw logits)
    ts_preds: jnp.ndarray     # [B, S, 15] timestamp predictions
    cat_preds: jnp.ndarray    # [B, S, D] categorical projections


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

        semantic_types = batch["semantic_types"]        # [B, S] int8
        column_ids = batch["column_ids"]                # [B, S] int32
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
        all_vals = jnp.stack([
            id_val,    # 0: Identifier
            num_val,   # 1: Numerical
            ts_val,    # 2: Timestamp
            bool_val,  # 3: Boolean
            cat_val,   # 4: Categorical
            text_val,  # 5: Text
            jnp.zeros_like(id_val),  # 6: Ignored (should never appear)
        ], axis=2)  # [B, S, 7, D]

        # Select: [B, S, D]
        raw_val = jnp.einsum("bst,bstd->bsd", type_one_hot, all_vals)

        # Null gating: replace value with null_emb if is_null
        null_emb = self.param(
            "null_emb", nn.initializers.normal(stddev=0.02), (d,)
        )
        is_null_expanded = is_null[..., None]  # [B, S, 1]
        val_or_null = (
            is_null_expanded * null_emb + (1.0 - is_null_expanded) * raw_val
        )

        # Target masking: replace with mask_emb if is_target (priority over null)
        mask_emb = self.param(
            "mask_emb", nn.initializers.normal(stddev=0.02), (d,)
        )
        is_target_expanded = is_target[..., None]  # [B, S, 1]
        val_final = (
            is_target_expanded * mask_emb
            + (1.0 - is_target_expanded) * val_or_null
        )

        # Combine column encoding + value encoding
        h0 = ZeroCenteredRMSNorm(eps=cfg.rms_norm_eps, name="h0_norm")(
            col_enc + val_final
        )

        # Zero out padding positions
        h0 = h0 * (1.0 - is_padding[..., None])

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
        ts_preds = nn.Dense(
            self.config.d_time, use_bias=True, name="timestamp_decoder"
        )(h)
        cat_preds = nn.Dense(d, use_bias=True, name="categorical_decoder")(h)

        return ModelOutput(
            h=h,
            null_logits=null_logits,
            num_preds=num_preds,
            bool_logits=bool_logits,
            ts_preds=ts_preds,
            cat_preds=cat_preds,
        )


def build_attention_masks(batch, max_rows):
    """Build the three attention masks from batch tensors.

    Args:
        batch: dict with seq_row_ids [B,S], fk_adj [B,R,R], is_padding [B,S].
        max_rows: R dimension.

    Returns:
        outbound_mask, inbound_mask, column_mask: each [B, S, S] bool.
    """
    seq_row_ids = batch["seq_row_ids"].astype(jnp.int32)  # [B, S]
    fk_adj = batch["fk_adj"].astype(jnp.bool_)             # [B, R, R]
    is_padding = batch["is_padding"].astype(jnp.bool_)     # [B, S]
    column_ids = batch["column_ids"]                        # [B, S]
    b, s = seq_row_ids.shape

    # Row indices for mask expansion
    ri = seq_row_ids[:, :, None]  # [B, S, 1]
    rj = seq_row_ids[:, None, :]  # [B, 1, S]

    # Gather fk_adj[b, ri, rj] for all (i, j) pairs
    # fk_adj_ij[b, i, j] = fk_adj[b, ri, rj]
    ri_flat = ri.reshape(b, s, 1).astype(jnp.int32)
    rj_flat = rj.reshape(b, 1, s).astype(jnp.int32)

    # Index into fk_adj: use advanced indexing
    batch_idx = jnp.arange(b)[:, None, None]
    ri_broadcast = jnp.broadcast_to(ri_flat, (b, s, s))
    rj_broadcast = jnp.broadcast_to(rj_flat, (b, s, s))
    fk_ij = fk_adj[batch_idx, ri_broadcast, rj_broadcast]  # [B, S, S]

    # Same row mask
    same_row = (ri == rj)  # [B, S, S]

    # Outbound mask: same_row OR fk_adj[ri, rj]
    outbound_mask = same_row | fk_ij

    # Inbound mask: fk_adj[rj, ri] (transposed)
    fk_ji = fk_adj[batch_idx, rj_broadcast, ri_broadcast]  # [B, S, S]
    inbound_mask = fk_ji

    # Column mask: same column
    col_i = column_ids[:, :, None]  # [B, S, 1]
    col_j = column_ids[:, None, :]  # [B, 1, S]
    column_mask = (col_i == col_j)

    # Exclude padding from all masks
    not_padding_i = ~is_padding[:, :, None]  # [B, S, 1]
    not_padding_j = ~is_padding[:, None, :]  # [B, 1, S]
    valid_mask = not_padding_i & not_padding_j

    outbound_mask = outbound_mask & valid_mask
    inbound_mask = inbound_mask & valid_mask
    column_mask = column_mask & valid_mask

    return outbound_mask, inbound_mask, column_mask


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
        text_batch_emb = jnp.array(text_emb_u16, dtype=jnp.float16).astype(
            jnp.bfloat16
        )

        # Value encoding
        h = ValueEncoder(config=cfg, name="value_encoder")(
            batch, col_emb_table, cat_emb_table, text_batch_emb
        )

        # Build attention masks
        outbound_mask, inbound_mask, column_mask = build_attention_masks(
            batch, cfg.max_rows
        )

        # Get permutations from batch
        out_perm = batch["out_perm"]  # [B, S]
        in_perm = batch["in_perm"]    # [B, S]
        col_perm = batch["col_perm"]  # [B, S]

        # Transformer layers
        for i in range(cfg.n_layers):
            h = TransformerLayer(config=cfg, layer_idx=i, name=f"layer_{i}")(
                h, outbound_mask, inbound_mask, column_mask,
                out_perm, in_perm, col_perm,
            )

        # Final RMSNorm
        h = ZeroCenteredRMSNorm(eps=cfg.rms_norm_eps, name="final_norm")(h)

        # Decoder heads
        output = DecoderHeads(config=cfg, name="decoder")(h)

        return output
