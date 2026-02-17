# Value Encoding

Value encoding produces the initial hidden state **h₀** `[B, S, D]` fed into the first transformer layer. Each cell's
representation is the **sum** of two projected components:

1. A **column-name embedding** — *what table/column* this cell belongs to (schema identity).
2. A **value embedding** — *what data* this cell holds, gated by null status and target masking.

## Column-Name Encoder (always present)

Look up the frozen column-name embedding from the GPU-resident table using `column_ids`, then project into model space:

```
col_raw[b, s]  = column_embedding_table[column_ids[b, s]]     // [D_t]
col_enc[b, s]  = Linear(D_t → D, bias=True)(col_raw[b, s])    // [D]
```

This is the same for all rows sharing a column and is **never** gated or replaced.

## Type-Specific Value Encoders

Each semantic type has its own encoder mapping the raw cell value into model space (D).
See [semantic_types.md](../semantic_types.md) for each type's preprocessing.

| Semantic Type   | Module                       | Input dim   | Learnable module                        |
|-----------------|------------------------------|-------------|-----------------------------------------|
| **Identifier**  | Learned constant             | —           | `identifier_emb ∈ ℝ^D`                 |
| **Numerical**   | `nn.Linear(1, D, bias)`     | 1 (scalar)  | `numerical_encoder: Linear(1 → D)`     |
| **Timestamp**   | `nn.Linear(15, D, bias)`    | 15          | `timestamp_encoder: Linear(15 → D)`    |
| **Boolean**     | `nn.Embedding(2, D)`        | {0, 1}      | `boolean_encoder: Embedding(2, D)`     |
| **Categorical** | `nn.Linear(D_t, D, bias)`   | D_t         | `categorical_encoder: Linear(D_t → D)` |
| **Text**        | `nn.Linear(D_t, D, bias)`   | D_t         | `text_encoder: Linear(D_t → D)`        |

**Identifier** cells carry no value — they signal entity presence only. Since identifiers are never null, every
identifier cell receives the same learned constant vector.

**Categorical** and **Text** use separate `nn.Linear` modules (rather than a shared one) because categorical values
are low-cardinality and repeated across rows, while text values are typically unique. The `categorical_encoder` is
especially important because frozen text embeddings for categories within the same column are often very close in
cosine similarity (e.g., `"status is enabled"` vs `"status is disabled"`). The learned projection amplifies the small
directional differences. This is reinforced at loss time: the same `categorical_encoder` is applied to all category
embeddings for the target column during cross-entropy (see [decoder_heads_and_loss.md](decoder_heads_and_loss.md)).

TODO(mrdmnd): experiment with replacing the single `nn.Linear` categorical encoder with a small MLP
(e.g., `Linear(D_t, D) → GELU → Linear(D, D)`) to see if nonlinear separation improves categorical prediction.

Because the batch uses a dense value layout, all six encoder paths run in parallel. The correct output per position
is selected via `semantic_types`:

```
raw_val[b, s] = Σ_t  one_hot(semantic_types[b, s], t) · encoder_t(values[b, s])    // [D]
```

In practice the sum collapses to a single non-zero term per position.

## Null Gating

The `is_null` tensor **replaces** the type-specific value with a learned null embedding for cells whose database
value is NULL:

```
val_or_null[b, s] = is_null[b, s] · null_emb
                  + (1 − is_null[b, s]) · raw_val[b, s]          // [D]
```

`null_emb ∈ ℝ^D` is a single learned vector shared across all semantic types. It gives the model an explicit signal
for missingness — distinct from a zero value and distinct from padding. A shared null embedding (rather than per-type)
is sufficient because the column encoding already disambiguates types.

## Target Masking

For cells where `is_target` is true, the value embedding is replaced by a learned mask embedding:

```
val_final[b, s] = is_target[b, s] · mask_emb
               + (1 − is_target[b, s]) · val_or_null[b, s]       // [D]
```

`mask_emb ∈ ℝ^D` signals "predict me." The column encoding is preserved.

Target masking takes **priority** over null gating: even if the target cell is null, the forward pass sees `mask_emb`,
not `null_emb`. Ground-truth null status is only consulted at loss time.

| | `is_null` | `is_target` |
|---|---|---|
| **Represents** | Data property (SQL NULL) | Training intervention (hidden for prediction) |
| **Effect on value embedding** | Replace with `null_emb` | Replace with `mask_emb` |
| **Column embedding affected?** | No | No |
| **Can co-occur?** | Yes — the model must predict that the target is null |

## Combination

```
h₀[b, s] = RMSNorm( col_enc[b, s]  +  val_final[b, s] )           // [D]
```

Additive combination (analogous to positional encodings added to token embeddings). Padding positions receive h₀ = 0.

## Parameter Summary

| Parameter              | Shape            | Count (D=256, D_t=256) |
|------------------------|------------------|------------------------|
| `column_name_encoder`  | `Linear(D_t, D)` | 65,792                 |
| `numerical_encoder`    | `Linear(1, D)`   | 512                    |
| `timestamp_encoder`    | `Linear(15, D)`  | 4,096                  |
| `boolean_encoder`      | `Embed(2, D)`    | 512                    |
| `categorical_encoder`  | `Linear(D_t, D)` | 65,792                 |
| `text_encoder`         | `Linear(D_t, D)` | 65,792                 |
| `identifier_emb`       | `[D]`            | 256                    |
| `null_emb`             | `[D]`            | 256                    |
| `mask_emb`             | `[D]`            | 256                    |
| **Total**              |                  | **203,264**            |
