"""Loss computation for the relational transformer.

Implements:
- Null loss (BCE on null head)
- Type-specific losses (Huber for numerical/timestamp, BCE for boolean, CE for categorical)
- Graph-break-free type selection via one-hot masking
- Z-loss regularization for categorical predictions
"""

import jax
import jax.numpy as jnp

from confluence.model import ModelOutput, STYPE_NUMERICAL, STYPE_BOOLEAN, STYPE_TIMESTAMP, STYPE_CATEGORICAL


def binary_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable BCE from raw logits."""
    # sigmoid_cross_entropy_with_logits
    return jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def huber_loss(error: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Huber loss (smooth L1): quadratic for |e| <= delta, linear beyond.

    Branchless formulation suitable for JAX JIT.
    """
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def compute_loss(
    output: ModelOutput,
    batch: dict,
    cat_emb_table: jnp.ndarray,
    categorical_encoder_fn,
    z_loss_weight: float = 1e-4,
    ts_scalar_weight: float = 2.0,
    huber_delta: float = 1.0,
    max_k: int = 256,
):
    """Compute training loss for a batch.

    Loss = mean_over_targets[ null_loss + (1 - is_null) * type_loss ]

    All decoder heads run on all positions. Type selection is via one-hot masking
    on target_stype (no graph breaks).

    Args:
        output: ModelOutput from the forward pass.
        batch: dict of batch tensors.
        cat_emb_table: [Vc, D_t] categorical embedding table (GPU-resident).
        categorical_encoder_fn: function that applies the categorical encoder
            Linear(D_t -> D) to a tensor. Used for cross-entropy over categories.
        z_loss_weight: coefficient for z-loss regularization.
        ts_scalar_weight: weight for the z-scored scalar component of
            timestamp loss relative to the averaged cyclic components.
        huber_delta: threshold for Huber loss on numerical/timestamp heads.
            Errors below delta are penalized quadratically; above, linearly.
        max_k: maximum category set size (for static shapes).

    Returns:
        scalar loss value (fp32).
    """
    is_target = batch["is_target"].astype(jnp.float32)  # [B, S]
    is_null = batch["is_null"].astype(jnp.float32)       # [B, S]
    target_stype = batch["target_stype"][0]               # scalar uint8

    # Upcast to fp32 for loss computation
    null_logits = output.null_logits.astype(jnp.float32)  # [B, S]
    num_preds = output.num_preds.astype(jnp.float32)      # [B, S]
    bool_logits = output.bool_logits.astype(jnp.float32)  # [B, S]
    ts_preds = output.ts_preds.astype(jnp.float32)        # [B, S, 15]
    cat_preds = output.cat_preds.astype(jnp.float32)      # [B, S, D]

    # Ground-truth values
    gt_numeric = batch["numeric_values"].astype(jnp.float32)       # [B, S]
    gt_bool = batch["bool_values"].astype(jnp.float32)             # [B, S]
    gt_timestamp = batch["timestamp_values"].astype(jnp.float32)   # [B, S, 15]
    gt_cat_ids = batch["categorical_embed_ids"].astype(jnp.int32)  # [B, S]

    # ---- Null loss: BCE at all target positions ----
    null_loss = binary_cross_entropy(null_logits, is_null)  # [B, S]

    # ---- Numerical loss: Huber ----
    num_loss = huber_loss(num_preds - gt_numeric, delta=huber_delta)  # [B, S]

    # ---- Boolean loss: BCE ----
    bool_loss = binary_cross_entropy(bool_logits, gt_bool)  # [B, S]

    # ---- Timestamp loss: Huber with upweighted scalar ----
    # 14 cyclic dims (7 sin/cos pairs) averaged + scalar with higher weight
    ts_cyclic_loss = jnp.mean(
        huber_loss(ts_preds[..., :14] - gt_timestamp[..., :14], delta=huber_delta),
        axis=-1,
    )  # [B, S]
    ts_scalar_loss = huber_loss(
        ts_preds[..., 14] - gt_timestamp[..., 14], delta=huber_delta
    )  # [B, S]
    ts_loss = ts_cyclic_loss + ts_scalar_weight * ts_scalar_loss  # [B, S]

    # ---- Categorical loss: cross-entropy over category set ----
    # Get category set for the target column
    cat_emb_start = batch.get("cat_emb_start", jnp.zeros(1, dtype=jnp.uint32))[0]
    cat_emb_count = batch.get("cat_emb_count", jnp.ones(1, dtype=jnp.uint32) * max_k)[0]

    # Gather category embeddings and project through categorical encoder
    cat_indices = jnp.arange(max_k) + cat_emb_start.astype(jnp.int32)
    # Clamp to valid range
    cat_indices = jnp.clip(cat_indices, 0, cat_emb_table.shape[0] - 1)
    cat_embs = cat_emb_table[cat_indices]  # [max_K, D_t]
    cat_embs_bf16 = cat_embs.astype(jnp.bfloat16)
    col_cat_proj = categorical_encoder_fn(cat_embs_bf16).astype(jnp.float32)  # [max_K, D]

    # Logits: cat_preds @ col_cat_proj.T -> [B, S, max_K]
    cat_logits = jnp.einsum("bsd,kd->bsk", cat_preds, col_cat_proj)

    # Mask out padding categories (indices >= cat_emb_count)
    k_range = jnp.arange(max_k)
    cat_valid_mask = k_range < cat_emb_count.astype(jnp.int32)  # [max_K]
    cat_logits = jnp.where(cat_valid_mask[None, None, :], cat_logits, -1e9)

    # Target category index (relative to cat_emb_start)
    target_cat_idx = (gt_cat_ids - cat_emb_start.astype(jnp.int32))  # [B, S]
    target_cat_idx = jnp.clip(target_cat_idx, 0, max_k - 1)

    # Cross-entropy
    cat_log_softmax = jax.nn.log_softmax(cat_logits, axis=-1)  # [B, S, max_K]
    cat_ce = -jnp.take_along_axis(
        cat_log_softmax, target_cat_idx[..., None], axis=-1
    ).squeeze(-1)  # [B, S]

    # Z-loss: 1e-4 * mean(log(sum(exp(logits)))^2)
    log_z = jax.nn.logsumexp(cat_logits, axis=-1)  # [B, S]
    z_loss = z_loss_weight * log_z ** 2  # [B, S]

    cat_loss = cat_ce + z_loss  # [B, S]

    # ---- Type-specific loss selection (graph-break-free) ----
    # Map target_stype to type loss index:
    # Numerical=1 -> idx 0, Boolean=3 -> idx 1, Timestamp=2 -> idx 2, Categorical=4 -> idx 3
    type_losses = jnp.stack([num_loss, bool_loss, ts_loss, cat_loss], axis=0)  # [4, B, S]

    # Create type selector from target_stype
    # Map semantic types to type_loss indices
    stype_to_loss_idx = jnp.array([
        -1,  # 0: Identifier (never target)
        0,   # 1: Numerical -> type_losses[0]
        2,   # 2: Timestamp -> type_losses[2]
        1,   # 3: Boolean -> type_losses[1]
        3,   # 4: Categorical -> type_losses[3]
        -1,  # 5: Text (never target)
        -1,  # 6: Ignored (never target)
    ])
    loss_idx = stype_to_loss_idx[target_stype.astype(jnp.int32)]
    type_selector = jax.nn.one_hot(loss_idx, 4)  # [4]

    # Select type loss: [4] @ [4, B, S] -> [B, S]
    type_loss = jnp.einsum("t,tbs->bs", type_selector, type_losses)

    # ---- Combined loss ----
    # cell_loss = null_loss + (1 - is_null) * type_loss
    cell_loss = null_loss + (1.0 - is_null) * type_loss  # [B, S]

    # Average over target positions only
    target_count = jnp.maximum(jnp.sum(is_target), 1.0)
    batch_loss = jnp.sum(cell_loss * is_target) / target_count

    return batch_loss
