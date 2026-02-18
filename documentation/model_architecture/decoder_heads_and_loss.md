# Decoder Heads and Loss Computation

## Decoder Heads

Decoder heads map the transformer's final hidden state `h[B, S, D]` into per-type predictions. To avoid JIT graph
breaks in JAX, **every head runs on every position in every forward pass** — the type-specific loss selects the
relevant outputs via masking. The wasted compute is negligible (single `nn.Linear` per head vs. attention cost).

| Head                    | Module               | Output shape    | Activation  | Purpose                           |
|-------------------------|----------------------|-----------------|-------------|-----------------------------------|
| **Null head**           | `Linear(D, 1)`      | `[B, S]`        | sigmoid → p | Binary: is the cell null?         |
| **Numerical decoder**   | `Linear(D, 1)`      | `[B, S]`        | none (raw)  | Predicted z-score value           |
| **Boolean decoder**     | `Linear(D, 1)`      | `[B, S]`        | sigmoid → p | Binary: true or false?            |
| **Timestamp decoder**   | `Linear(D, D_time)` | `[B, S, 15]`    | none (raw)  | Predicted 15-dim timestamp encoding |
| **Categorical decoder** | `Linear(D, D)`      | `[B, S, D]`     | none (raw)  | Projected vector for category retrieval |

There is **no text decoder** — text columns are never prediction targets (see [semantic_types.md](../semantic_types.md)).

**Categorical decoder** projects the hidden state into the **same space** that `categorical_encoder` maps frozen
category embeddings into. At training time, the output is scored against all K category embeddings for the target
column via cross-entropy. This deliberately reuses the same representation space as the input encoder.

### Model Output Structure

```
ModelOutput:
    h:            [B, S, D]       # final hidden states
    null_logits:  [B, S]          # null head (raw logits)
    num_preds:    [B, S]          # numerical predictions (z-score scale)
    bool_logits:  [B, S]          # boolean head (raw logits)
    ts_preds:     [B, S, D_time]  # timestamp predictions (15-dim)
    cat_preds:    [B, S, D]       # categorical projections (ℝ^D)
```

All fields are dense tensors computed unconditionally. `is_target` and `target_stype` determine which fields are
consumed during loss/inference.

### Parameter Summary

| Parameter              | Shape            | Count (D=256, D_time=15) |
|------------------------|------------------|--------------------------|
| `null_head`            | `Linear(D, 1)`  | 257                      |
| `numerical_decoder`    | `Linear(D, 1)`  | 257                      |
| `boolean_decoder`      | `Linear(D, 1)`  | 257                      |
| `timestamp_decoder`    | `Linear(D, 15)` | 3,855                    |
| `categorical_decoder`  | `Linear(D, D)`  | 65,792                   |
| **Total**              |                  | **70,418**               |

---

## Loss Computation

Loss is computed **only at target positions** (`is_target = true`). Currently one target per sequence → B positions.
Each target produces two loss terms:

```
cell_loss = null_loss  +  (1 - is_null) · type_loss
```

When the ground truth is null, only the null head receives gradient. When non-null, both terms contribute.

### Null Loss

Binary cross-entropy, computed for every target regardless of `target_stype`:

```
null_loss = BCE(sigmoid(null_logits[b, t]), is_null[b, t])
```

### Type-Specific Losses

| Target type     | Loss        | Prediction                 | Ground truth                                                            |
|-----------------|-------------|----------------------------|-------------------------------------------------------------------------|
| **Numerical**   | Huber       | `num_preds[b, t]`          | `numeric_values[b, t]`                                                  |
| **Boolean**     | BCE         | `sigmoid(bool_logits[b, t])` | `bool_values[b, t]`                                                  |
| **Timestamp**   | Huber (15-dim, weighted) | `ts_preds[b, t]` | `timestamp_values[b, t]`                                                |
| **Categorical** | Cross-entropy | `cat_preds[b, t]`       | `categorical_encoder(cat_emb_table[cat_emb_start : cat_emb_start + K])` |

**Huber loss** (smooth L1) is used for numerical and timestamp heads instead of MSE. It is quadratic for small errors
(|e| ≤ δ, default δ=1.0) and linear for large errors, providing robustness to outliers in z-scored values while
preserving smooth gradients near zero.

**Timestamp weighting.** The 15-dim timestamp vector contains 14 cyclic (sin/cos) components and 1 z-scored scalar.
The cyclic components are averaged into a single loss term while the scalar is weighted separately
(`ts_scalar_weight`, default 2.0) to prevent it from being drowned out by the 14 cyclic dimensions. With default
settings the scalar receives ~22% of the total timestamp gradient budget.

**Categorical loss** in detail — cross-entropy over the full category set for the target column:

```
col_cat_embeds = cat_emb_table[cat_emb_start : cat_emb_start + K]    # [K, D_t]
col_cat_proj   = categorical_encoder(col_cat_embeds)                   # [K, D]
logits         = cat_preds[b, t] @ col_cat_proj.T                     # [K]
cat_loss       = cross_entropy(logits, target_cat_index)               # scalar
```

Where `target_cat_index` (0..K-1) is derived from `categorical_embed_ids[b, t] - cat_emb_start`.

Because `categorical_encoder` is shared between encoding and loss, it receives gradient from both directions.

**Padding for static shapes.** K varies across categorical columns; JAX requires static shapes for JIT. The category
set is padded to `max_K` (known at preprocessing time), with padding logits masked to `-1e9` before softmax.

### Graph-Break-Free Type Selection

Every batch is homogeneous (one `target_stype`), but for a fully static computation graph we compute **all four
type losses** and select via masking:

```python
type_losses = stack([num_loss, bool_loss, ts_loss, cat_loss])  # [4, B]
type_selector = one_hot(target_stype, 4)                       # [4]
type_loss = type_selector @ type_losses                        # [B]
```

### Batch Loss

```
batch_loss = mean_b[ null_loss_b  +  (1 - is_null[b, t]) · type_loss_b ]
```

---

## Inference Output

| Target type     | Inference prediction                                                                |
|-----------------|-------------------------------------------------------------------------------------|
| **Null**        | `is_null = sigmoid(null_logits[b, t]) > 0.5` (checked first for all types)         |
| **Numerical**   | Denormalize: `pred = num_preds[b, t] * col_std + col_mean`                         |
| **Boolean**     | Threshold: `sigmoid(bool_logits[b, t]) > 0.5`                                      |
| **Timestamp**   | Denormalize epoch component, decode cyclic components to calendar fields             |
| **Categorical** | `argmax_k( cat_preds[b,t] @ categorical_encoder(cat_emb_table[start:start+K]).T )` |

The null head is always checked first: if `p_null > 0.5`, the prediction is NULL regardless of the type-specific output.
