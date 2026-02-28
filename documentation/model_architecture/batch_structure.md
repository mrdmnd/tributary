# Batch Structure

A `TrainingBatch` is the unit of data transferred from CPU to GPU. Every batch comes from the **same task** within
the **same database**, so targets are homogeneous (one semantic type, one decoder head, one loss function per batch).

See [task_framework.md](../task_framework.md) for details on tasks.

## Dimensions

| Symbol | Meaning                                | Typical value |
|--------|----------------------------------------|---------------|
| B      | Batch size (number of subgraphs/seeds) | 32            |
| S      | Sequence length (padded/truncated)     | 1024          |
| D_t    | Frozen text embedding dimension        | 256           |
| D      | Model hidden dimension                 | 512           |

Each batch element is a **cell sequence**: the BFS sampler walks outward from a task's seed row over FK edges, collects
rows (up to a hop budget and optional temporal cutoff), then linearizes all non-ignored cells from collected rows
into a flat sequence of length S.

See [sampling.md](../sampling.md) for the sampling algorithm.

## Index spaces

Three distinct index spaces appear in a `TrainingBatch`.

| Index space | Scope | Range | Batch tensors | Remapping |
|-------------|-------|-------|---------------|-----------|
| **Global** | Entire database | `ColumnIdx` 0..C-1 | `column_ids` | None — column embedding table is GPU-resident |
| **Global** | Entire database | `CategoricalEmbeddingIdx` 0..Vc-1 | `categorical_embed_ids` | None — categorical embedding table is GPU-resident |
| **Sequence-local** | Per sequence (one of B) | 0..R-1 | `seq_row_ids` | `HashMap<RowIdx, u16>` built per-sequence during BFS |
| **Batch-local** | Entire batch (all B sequences) | 0..U-1 | `text_embed_ids`, `text_batch_embeddings` | `HashMap<TextEmbeddingIdx, u32>` built batch-wide after all sequences |

The model never sees global `RowIdx` or global `TextEmbeddingIdx` values. The sampler remaps them to compact
local indices before shipping to GPU. Categorical cells use global indices directly since the table is GPU-resident.

---

## 1. Cell identity tensors

These tell the model *what* each position in the sequence represents.

`semantic_types` — `[B, S]` `int8`. The semantic type of each cell.
See [semantic_types.md](../semantic_types.md). Cells with the `ignored` semantic type are never included.

`column_ids` — `[B, S]` `int32`. The **global** column index (`ColumnIdx`). Used directly as lookup keys into the
GPU-resident column-name embedding table (Section 5a).

`seq_row_ids` — `[B, S]` `uint16`. The **sequence-local** row index (0..R-1), assigned during BFS via a per-sequence
`HashMap<RowIdx, u16>`. Cells from the same row share the same `seq_row_id`. Used for:
- **Same-row detection** in outbound attention's "own row" rule.
- **Sampler-side CSR mask construction** before batch export.

---

## 2. Per-type cell value tensors (dense layout)

Every sequence position has storage for every type, with zeros at non-matching positions.
This dense layout avoids scatter/gather — the forward pass multiplies by a type mask.

`is_null` — `[B, S]` `u8` (0/1). Whether the cell's database value was NULL. Null cells have their value embedding
replaced by a learned `null_embedding` vector (see [value_encoding.md](value_encoding.md)). The column-name
embedding is preserved.

`numeric_values` — `[B, S]` `f32`. Z-score normalized numerical value.

`timestamp_values` — `[B, S, 15]` `f32`. Cyclic timestamp encoding: 7 sin/cos pairs + z-scored epoch microseconds.

`bool_values` — `[B, S]` `u8` (0/1). Boolean value.

`categorical_embed_ids` — `[B, S]` `u32`. **Global** `CategoricalEmbeddingIdx` into the GPU-resident categorical
embedding table (Section 5b). No batch-local remapping needed.

`text_embed_ids` — `[B, S]` `u32`. **Batch-local** index (0..U-1) into `text_batch_embeddings`. The sampler remaps
global `TextEmbeddingIdx` values during batch construction (Section 6).

**Sentinel values:** Positions where a type slot is not active are set to 0. This is safe because the dense layout
multiplies by a type mask, `is_padding` excludes padding from attention/loss, and `is_null` zeroes out the value
embedding for null cells.

---

## 3. Validity and masking tensors

`is_target` — `[B, S]` `u8` (0/1). Which cells are prediction targets for the task. Currently one target per sequence.
Target cells have their value embedding replaced by a learnable `mask_embedding` vector; the column-name embedding
remains visible. No separate ground-truth tensors are needed — ground-truth values are stored in the per-type
value tensors (Section 2) at the target position.

**Null prediction.** The target cell's ground truth may be NULL (indicated by `is_null` at `is_target` positions).
The model includes a binary "is null?" head. The overall loss for a target cell is:

```
loss = null_bce_loss + (1 - is_null) * type_specific_loss
```

When the ground truth is null, only the null head receives gradient. See [decoder_heads_and_loss.md](decoder_heads_and_loss.md).

`is_padding` — `[B, S]` `u8` (0/1). True for positions beyond the actual cell sequence. Padded positions are excluded
from all attention patterns (OR'd into each mask), from loss computation, and from target selection.

---

## 4. Attention mask transport (CSR)

See [attention.md](attention.md) for the full attention mechanism. This section covers the batch tensors involved.

The sampler ships packed CSR payloads for each relation:

- `mask_format` — `[1]` `u8`, currently always `1` (CSR).
- `outbound_csr_row_ptr` — `[B, S+1]` `u32`
- `outbound_csr_seq_offsets` — `[B+1]` `u32`
- `outbound_csr_col_idx` — `[total_out_nnz]` `u16`
- `inbound_csr_row_ptr` — `[B, S+1]` `u32`
- `inbound_csr_seq_offsets` — `[B+1]` `u32`
- `inbound_csr_col_idx` — `[total_in_nnz]` `u16`
- `column_csr_row_ptr` — `[B, S+1]` `u32`
- `column_csr_seq_offsets` — `[B+1]` `u32`
- `column_csr_col_idx` — `[total_col_nnz]` `u16`

Interpretation:
- `*_csr_row_ptr[b]` is sequence-local CSR row pointers for sequence `b`.
- `*_csr_seq_offsets[b:b+2]` gives the packed `col_idx` slice for sequence `b`.
- `*_csr_col_idx` stores sequence-local key indices (0..S-1).

In the model input pipeline, these CSR payloads are decoded to dense `[B, S, S]` boolean masks before SDPA.

---

## 5a. Column-Name Embedding Table (GPU-resident)

Column-name embeddings are loaded from `column_embeddings.bin` into a `[num_columns, D_t]` tensor on GPU at training
start. Small enough to keep resident for the entire run (`num_columns * D_t * 2` bytes). Not part of the per-batch
transfer.

## 5b. Categorical Embedding Table (GPU-resident)

Categorical value embeddings from `categorical_embeddings.bin`, stored as a `[Vc, D_t]` `f16` tensor on GPU. Because
categorical columns are low-cardinality (typically < 100 distinct values per column), the total table is small
(usually under 20 MB).

Each categorical column's categories occupy a **contiguous block** starting at `ColumnStats::Categorical::cat_emb_start`.
Category `i` maps to `CategoricalEmbeddingIdx(cat_emb_start + i)`.

GPU-resident design enables:
1. Cross-entropy loss over the full category set at loss time (see [decoder_heads_and_loss.md](decoder_heads_and_loss.md)).
2. Nearest-neighbor inference via matmul against a contiguous slice.

## 6. Text Embeddings (per-batch)

Text embeddings are high-cardinality and stored in `text_embeddings.bin`. The full table can be very large, so we ship
a per-batch subset.

`text_batch_embeddings` — `[U, D_t]` `f16`. U = number of unique text embeddings referenced by all B sequences.

#### Batch construction: text embedding remapping

**Phase 1 — Per-sequence (parallelizable across B seeds):**
BFS, linearize cells, record values. Text cells temporarily store **global** `TextEmbeddingIdx`. Build per-sequence
row map and CSR attention payloads.

**Phase 2 — Batch-wide text dedup (after all B sequences):**
1. Collect unique global `TextEmbeddingIdx` values across the batch.
2. Assign dense batch-local indices: `HashMap<TextEmbeddingIdx, u32>` mapping global → 0..U-1.
3. Gather U embedding vectors from mmap'd text embedding table into `text_batch_embeddings`.
4. Rewrite `text_embed_ids[B, S]` with batch-local indices.

Phase 1 parallelizes across seeds; Phase 2 is a single sync point.
