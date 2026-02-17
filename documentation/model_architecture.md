# Custom Transformer Model Architecture

## Batch Structure

A `TrainingBatch` is the unit of data transferred from CPU to GPU. Every batch comes from the **same task** within
the **same database**, so targets are homogeneous (one semantic type, one decoder head, one loss function per batch).

See [task_framework.md](task_framework.md) for more details on the task and structure.

### Dimensions

| Symbol | Meaning                                  | Typical value        |
|--------|------------------------------------------|----------------------|
| B      | Batch size (number of subgraphs/seeds)   | 32                   |
| S      | Sequence length (padded/truncated cells) | 1024                 |
| D_t    | Frozen text embedding dimension          | 256                  |
| D      | Model hidden dimension                   | 256                  |

Each batch element is a **cell sequence**: the BFS sampler walks outward from a task's seed row over FK edges, collects
rows (up to a hop budget and optional temporal cutoff), then linearizes all non-ignored cells from collected rows
into a flat sequence of length S.

See [sampling.md](sampling.md) for more details on the sampling algorithm -- it's a non-standard custom order intended
to apply the inductive bias that "parent rows are more important than child rows".

### Index spaces

Three distinct index spaces appear in a `TrainingBatch`. Keeping them straight is critical for the sampler
implementation and for understanding what the model sees.

| Index space | Scope | Range | Batch tensors that use it | Remapping |
|-------------|-------|-------|---------------------------|-----------|
| **Global** | Disk / entire database | `ColumnIdx` 0..C-1 | `column_ids` | None needed — column-name embedding table is small and GPU-resident |
| **Global** | Disk / entire database | `CategoricalEmbeddingIdx` 0..Vc-1 | `categorical_embed_ids` | None needed — categorical embedding table is small and GPU-resident |
| **Sequence-local** | Per sequence (one of B) | 0..R-1 | `seq_row_ids`, adjacency matrices | `HashMap<RowIdx, u16>` built per-sequence during BFS |
| **Batch-local** | Entire batch (all B sequences) | 0..U-1 | `text_embed_ids`, `text_batch_embeddings` | `HashMap<TextEmbeddingIdx, u32>` built batch-wide after all sequences are linearized |

The model never sees global `RowIdx` or global `TextEmbeddingIdx` values for text cells. The sampler remaps
them to compact batch-local indices before shipping the batch to the GPU. Categorical cells use global indices
directly since the categorical embedding table is GPU-resident. See Sections 1 and 6 for details.

---

### 1. Cell identity tensors

These tell the model *what* each position in the sequence represents.

`semantic_types` is a `[B, S]` tensor of `int8` values. For each cell, this indicates the semantic type of the cell.
See [semantic_types.md](semantic_types.md) for more details on the semantic types.
Note that we will never include any cells with the `ignored` semantic type in any batch.

`column_ids` is a `[B, S]` tensor of `int32` values. For each cell, this indicates the **global** column index
(`ColumnIdx`). Column-name embeddings are small and GPU-resident (see Section 5), so global column indices can be
used directly as lookup keys with no remapping.

`seq_row_ids` is a `[B, S]` tensor of `uint16` values. For each cell, this indicates the **sequence-local** row
index (0..R-1), assigned during BFS. Each of the B sequences independently maps the global `RowIdx` values it visits
to a dense local range via a per-sequence `HashMap<RowIdx, u16>`. Cells from the same row share the same
`seq_row_id`. These local indices serve two purposes:
- **Same-row detection**: identifying which cells belong to the same row (for outbound attention's "own row" rule).
- **Adjacency matrix indexing**: the `[B, R, R]` row-level adjacency matrices (Section 4) are indexed by
  `seq_row_ids`, not global `RowIdx`.

The model never needs the absolute global row number — only row equality and adjacency structure matter.


### 2. Per-type cell value tensors (dense layout)

Every sequence position has storage for every type, with zeros at non-matching positions.
This dense layout avoids scatter/gather ops — the forward pass simply multiplies by a type mask.
Categorical cells store **global indices into the GPU-resident categorical embedding table** (see Section 5b).
Text cells store **batch-local indices into `text_batch_embeddings`** (see Section 6)
rather than full 256-dim vectors (512 bytes), keeping the batch compact.

`is_null` is a `[B, S]` tensor of `bool` values. 
For each cell, this indicates whether the cell's value was NULL.
Null cells have their type-specific value embedding replaced by a learned `null_embedding` vector (see Value Encoding).
The column-name embedding is preserved, so the model knows *which* column is null. This lets the model learn from
the *absence* of a value (e.g., a NULL rating means "not yet reviewed") with an explicit trainable signal rather than
relying on a zero vector.

`numeric_values` is a `[B, S]` tensor of `f32` values.
For each cell, this contains the z-score normalized numerical value.

`timestamp_values` is a `[B, S, 15]` tensor of `f32` values.
For each cell, this contains the cyclic timestamp encoding: 7 sin/cos pairs + z-scored epoch microseconds.

`bool_values` is a `[B, S]` tensor of `bool` values.
For each cell, this contains the boolean value.

`categorical_embed_ids` is a `[B, S]` tensor of `u32` values.
For each categorical cell, this contains the **global** `CategoricalEmbeddingIdx` into the GPU-resident
categorical embedding table (see Section 5b). No batch-local remapping is needed — the table is small enough
to keep permanently on GPU.

`text_embed_ids` is a `[B, S]` tensor of `u32` values.
For each text cell, this contains the **batch-local** index (0..U-1) into `text_batch_embeddings` for the text
value embedding. These are *not* the global `TextEmbeddingIdx` values from disk — the sampler remaps them
during batch construction (see Section 6).

**Sentinel values for non-applicable positions.** Positions where a given type slot is not active (e.g., a numerical
cell's `categorical_embed_ids`, or a padding position) are set to 0. This is safe because:
- The dense layout multiplies by a type mask, so non-matching slots are zeroed out regardless.
- `is_padding` excludes padding positions from attention and loss.
- `is_null` zeroes out the value embedding for null cells (their embed IDs are also 0).

---

### 3. Validity and masking tensors

`is_target` is a `[B, S]` tensor of `bool` values.
This is a mask indicating which cells are the target cells for the task.
We're starting with a single target cell per sequence (the one from the task definition).
Target cells are ultimately predicted by the model and the ground truth is used to compute the loss.
No separate prediction target tensors are needed — the ground-truth values (whether raw cell values for cell masking
tasks or derived values for aggregation tasks) are stored in the per-type cell value tensors (Section 2) at the
target position. The sampler writes the appropriate value into the matching type slot when constructing the batch.
At loss time, the model indexes into the per-type tensors using `is_target` to retrieve the ground truth.

Masked cells have their value embedding replaced by a learnable `mask_embedding` vector;
however, the column-name embedding remains visible (the model knows *which* column is masked, just not the value).

**Null prediction.** The target cell's ground truth may be NULL (indicated by `is_null` at `is_target` positions).
The model includes a binary "is null?" prediction head for each target position, trained with binary cross-entropy
against the ground truth from `is_null`. The overall loss for a target cell is:

```
loss = null_bce_loss + (1 - is_null) * type_specific_loss
```

When the ground truth is null, only the null head receives gradient signal — there is no meaningful type-specific
value to regress or classify. When the ground truth is not null, both the null head and the type-specific decoder
head contribute. This lets the model learn that nullness itself carries semantic information (e.g., "no accepted
answer yet", "rating not provided") and produce calibrated null predictions at inference time.

`is_padding` is a `[B, S]` tensor of `bool` values.
If the cell is a padding cell (positions beyond the end of the actual cell sequence), this is `true`. 
Padded positions are excluded 
- from all attention patterns (OR'd into each mask) 
- from loss computation (no gradient signal)
- from masking (they should never be selected for masking in the first place)

---

### 4. Row-level adjacency tensor and attention permutations

Instead of shipping three full `[B, S, S]` attention masks (~100 MB at B=32, S=1024), we ship one compact `[B, R, R]`
row-level boolean adjacency matrix plus three `[B, S]` permutation vectors. Both outbound and inbound attention masks
are derived from the same FK adjacency matrix — one reads it directly, the other reads it transposed — so a single
matrix suffices. The column attention mask needs no adjacency matrix at all (it is derived purely from `column_ids`).

#### `fk_adj` — FK adjacency matrix

`fk_adj` is a `[B, R, R]` tensor of `bool` values, where R is the maximum number of distinct rows across all B
sequences (padded with `false` for sequences with fewer rows). Entry `fk_adj[b, r1, r2]` is `true` when row `r1` has
a foreign-key column whose value points to row `r2` in sequence `b`'s sampled subgraph — i.e., `r2` is a **parent**
of `r1` in FK terminology (the F→P direction).

The sampler builds this matrix during BFS: whenever it follows an outgoing FK edge from a visited row to its parent
row, it records the edge in `fk_adj` using the sequence-local row indices from the same `HashMap<RowIdx, u16>` that
produces `seq_row_ids` (Section 1). Only edges between rows that are *both* present in the sampled subgraph are
recorded.

**Size:** At B=32, R=200: 32 × 200 × 200 = 1.28 MB — roughly **80× smaller** than shipping full `[B, S, S]`
cell-level masks.

#### Deriving cell-level attention masks

The forward pass expands `fk_adj` into `[B, S, S]` cell-level masks at each attention layer. All three attention
types use the same mechanism — gather via `seq_row_ids` — but with different row-level predicates. Let
`ri = seq_row_ids[b, i]` and `rj = seq_row_ids[b, j]` throughout.

**Outbound attention** — a cell sees its own row + FK parent rows:

```
outbound_mask[b, i, j] = (ri == rj) || fk_adj[b, ri, rj]
```

The first term is the self-loop (own row); the second follows FK edges forward (child → parent).

**Inbound attention** — a cell sees only FK child rows (not its own row):

```
inbound_mask[b, i, j] = fk_adj[b, rj, ri]
```

Note the swapped indices `fk_adj[b, rj, ri]`: this reads the matrix transposed, which captures edges pointing
*into* row `ri` — i.e., `rj` is a child that has an FK to `ri`. No explicit transpose is needed; swapping the
gather indices achieves the same result. There is no self-loop term: own-row visibility is provided by outbound
attention, and the residual connection carries representations through inbound layers for rows with no children.

**Column attention** — a cell sees other cells from the same column:

```
column_mask[b, i, j] = (column_ids[b, i] == column_ids[b, j])
```

No adjacency matrix needed — column identity alone determines visibility.

All three masks are AND'd with `~is_padding[b, j]` to exclude padding positions from being attended to,
and with `~is_padding[b, i]` to prevent padding positions from attending to anything.

The outbound mask includes a self-loop (the `ri == rj` term), so every cell can always attend to other cells in its
own row. This matches the original Relational Transformer's "feature attention" design. The inbound mask has **no**
self-loop — it is purely an aggregation channel from child rows, mirroring the RT's "neighbor attention." Rows with
no children (e.g., leaf order rows) produce an all-zero inbound mask; the residual connection ensures their
representations pass through unchanged.

#### Concrete example

Using the bookstore schema from [preprocessing.md](preprocessing.md), suppose BFS from order 1 collects six rows:

| seq_row_id | Row                | Table     |
|------------|--------------------|-----------|
| 0          | order 1 (seed)     | orders    |
| 1          | customer 23        | customers |
| 2          | book 42            | books     |
| 3          | order 7            | orders    |
| 4          | order 12           | orders    |
| 5          | order 5            | orders    |

FK edges in the subgraph (child → parent):
- order 1 → customer 23, book 42 (via `customer_id`, `book_id`)
- order 7 → customer 23 (via `customer_id`)
- order 12 → customer 23 (via `customer_id`)
- order 5 → book 42 (via `book_id`)

The resulting `fk_adj[b]` matrix (`.` = false for readability):

```
     r0  r1  r2  r3  r4  r5
r0: [ .   1   1   .   .   . ]   ← order 1 has FKs to customer 23 (r1) and book 42 (r2)
r1: [ .   .   .   .   .   . ]   ← customer 23 has no FK columns
r2: [ .   .   .   .   .   . ]   ← book 42 has no FK columns
r3: [ .   1   .   .   .   . ]   ← order 7 has FK to customer 23 (r1)
r4: [ .   1   .   .   .   . ]   ← order 12 has FK to customer 23 (r1)
r5: [ .   .   1   .   .   . ]   ← order 5 has FK to book 42 (r2)
```

From this single matrix, the forward pass derives:

**Outbound** (`I + fk_adj`) — each row sees itself and its parents:
- Order 1 sees: {order 1, customer 23, book 42}
- Customer 23 sees: {customer 23} only (no FK columns)
- Order 7 sees: {order 7, customer 23}

**Inbound** (`fk_adj^T`) — each row sees only its children:
- Customer 23 sees: {order 1, order 7, order 12}
- Book 42 sees: {order 1, order 5}
- Order 7 sees: ∅ (no rows have FKs pointing to it — residual carries its state through)

#### Attention permutations for block sparsity

After expanding `fk_adj` to a cell-level mask, the mask is typically sparse but **scattered** — nonzero entries are
distributed irregularly across the S×S matrix. Block-sparse attention kernels (e.g., Triton block-sparse, FlashAttention
with block masks) divide the S×S matrix into fixed-size tiles (commonly 64×64 or 128×128) and skip all-zero tiles
entirely. A scattered mask wastes this optimization: nonzero entries land in many different tiles, causing most tiles to
be marked non-zero even though they're mostly empty.

The fix is to **permute sequence positions** before each attention layer so that cells that can attend to each other are
clustered together. After permutation, nonzeros concentrate into fewer tiles, and the block-sparse kernel skips more
tiles. The sampler precomputes three permutations and ships them as batch tensors:

`col_perm` is a `[B, S]` tensor of `uint16` values.
Permutation for column attention. Computed as `argsort(column_ids[b, :])`, which groups cells by column. This creates
a perfectly block-diagonal mask: each column's cells form a contiguous dense block along the diagonal, with all
off-diagonal blocks being zero.

`out_perm` is a `[B, S]` tensor of `uint16` values.
Permutation for outbound attention. The sampler groups cells by `seq_row_id` (cells in the same row are always
contiguous in the permuted sequence), then orders the row groups to minimize bandwidth of the outbound adjacency
matrix (`I + fk_adj`). In practice, a reverse Cuthill-McKee (RCM) ordering on the row-level graph concentrates
nonzeros near the diagonal. Since R ≤ 200, RCM is negligible cost. In the bookstore example above, an outbound-
optimized ordering might be `[r1, r0, r3, r4, r2, r5]` — customer 23 first, then the three orders that reference
it, then book 42 and its child order — which clusters the FK-parent relationships into a tight block.

`in_perm` is a `[B, S]` tensor of `uint16` values.
Permutation for inbound attention. Same approach as `out_perm`, but optimizing for the inbound adjacency matrix
(`fk_adj^T`, no self-loop). The optimal row ordering often differs from `out_perm` because the inbound mask's
sparsity structure is the transpose of the outbound mask's.

Padding positions are placed at the end of each permuted sequence so that trailing all-padding tiles are trivially
skippable by the kernel.

**Size:** Three `[B, S]` uint16 tensors at B=32, S=1024: 3 × 32 × 1024 × 2 = 192 KB per batch.

#### Forward-pass usage

At each attention layer, the forward pass:

1. **Gather** hidden states into permuted order using the layer's permutation:
   `h_perm[b, i, :] = h[b, perm[b, i], :]`

2. **Expand** the attention mask in the permuted index space. For outbound attention:

```
ri = seq_row_ids[b, out_perm[b, i]]
rj = seq_row_ids[b, out_perm[b, j]]
perm_mask[b, i, j] = (ri == rj) || fk_adj[b, ri, rj]
```

The full `[B, S, S]` mask is never materialized — the block-sparse kernel computes each tile's mask on-the-fly
from `fk_adj` and the permuted row IDs, preserving the 80× compression from the compact representation.

3. **Attend** using block-sparse attention on `h_perm` with the on-the-fly mask.

4. **Scatter** the output back to original position order using the inverse permutation:
   `h_out[b, perm[b, i], :] = attn_out[b, i, :]`

The inverse permutation can be computed on-device in O(S) from `perm` (`inv[perm[i]] = i`), or precomputed by the
sampler and shipped as three additional `[B, S]` uint16 tensors (another 192 KB). Either approach is fine.

---

### 5a. Column-Name Embedding Table (GPU-resident, not part of the batch)

Column-name embeddings are stored in `column_embeddings.bin` and loaded into `ColumnMetadata.embedding` at startup, one per global
column. At the start of the training loop, we collect them into a `[num_columns, D_t]` tensor and upload it to GPU
shared memory once. This avoids per-batch CPU-to-GPU transfer for column identity information. The table is small
— `num_columns * D_t * 2` bytes — so it's fine to keep it resident for the entire run.

### 5b. Categorical Embedding Table (GPU-resident, not part of the batch)

Categorical value embeddings are stored in a separate file (`categorical_embeddings.bin`), distinct from text
embeddings. Because categorical columns are low-cardinality (typically < 100 distinct values per column), the
total number of categorical embeddings across the entire database is small — usually a few thousand entries,
well under 20 MB — making it feasible to keep the entire table GPU-resident permanently, just like column-name
embeddings.

At training start, the categorical embedding table is uploaded to GPU as a `[Vc, D_t]` tensor of `f16` values,
where Vc is the total number of distinct categorical values across all columns. Each categorical column's
categories occupy a **contiguous block** in this table, starting at the offset stored in
`ColumnStats::Categorical::cat_emb_start`. Category `i` (in the column's sorted category list) maps to
`CategoricalEmbeddingIdx(cat_emb_start + i)`.

On disk, each categorical cell stores a **global** `CategoricalEmbeddingIdx` — a 0-based index directly into
`categorical_embeddings.bin`. Because the table is GPU-resident, no batch-local remapping is needed: the
`categorical_embed_ids` tensor in each batch contains global indices that the model uses directly for lookup.

This GPU-resident design has two benefits beyond eliminating per-batch data movement:
1. The model can project **all** categories for a target column through `categorical_encoder` at loss time,
   enabling cross-entropy loss over the full category set (see Loss Computation).
2. At inference, nearest-neighbor search over a target column's categories is a single matmul against a
   pre-projected, contiguous slice of the table.

### 6. Text Embeddings (per-batch, from `text_embeddings.bin`)

Text cell values are high-cardinality (often unique per row) and their embeddings are stored in a separate file
(`text_embeddings.bin`). The full table can be very large (millions of entries), so we cannot keep it GPU-resident.
Instead, the sampler ships a per-batch subset containing only the text embeddings needed by that batch.

On disk, each text cell stores a **global** `TextEmbeddingIdx` — a 0-based index directly into
`text_embeddings.bin`. The sampler must remap these to **batch-local** indices during batch construction so that
the GPU only receives the embeddings it needs.

`text_batch_embeddings` is a `[U, D_t]` tensor of `f16` values, where U is the number of unique text embeddings
referenced by all text cells across all B sequences in the batch. The `text_embed_ids` tensor (Section 2) contains
batch-local indices (0..U-1) into this tensor.

#### Batch construction: text embedding remapping

The sampler builds batches in two phases:

**Phase 1 — Per-sequence (parallelizable across B seeds):**
For each seed, the sampler performs BFS, linearizes cells into a sequence, and records cell values. During this phase,
text cells temporarily store their **global** `TextEmbeddingIdx` values. Categorical cells store their global
`CategoricalEmbeddingIdx` values directly (no remapping needed). Simultaneously, the sampler builds the per-sequence
row map (`HashMap<RowIdx, u16>`) to produce `seq_row_ids` and the `[R, R]` adjacency matrices.

**Phase 2 — Batch-wide text embedding dedup (after all B sequences are built):**
1. Collect all unique global `TextEmbeddingIdx` values from every non-null text cell across the entire batch.
2. Assign each unique value a dense batch-local index: `HashMap<TextEmbeddingIdx, u32>` mapping global to 0..U-1.
3. Gather the U embedding vectors from the mmap'd text embedding table into `text_batch_embeddings[U, D_t]`.
4. Rewrite `text_embed_ids[B, S]`, replacing each global `TextEmbeddingIdx` with its batch-local index.

This two-phase approach allows Phase 1 to run in parallel across seeds while Phase 2 is a single synchronization
point. The resulting `text_batch_embeddings` tensor is typically much smaller than shipping dense embeddings for
every cell — most sequence positions are non-text and would be zeros. Categorical embeddings are not included
here since they are already GPU-resident.


## Attention Mechanisms 

Confluence implements a "relational transformer" operating on sequences of "cells" - a cell is a (row, column) pair in a
relational database, attached to a specific table.

Our sampler implements a custom algorithm to produces sequences of cells (starting from a seed anchor in a task) that
it determines may relevant context towards predicting the masked cell. We mask *only one cell* in each sequence - the
value from the target in the task definition.

These cell sequences have *three* attention mask patterns associated with them (unlike the RT paper, Confluence does
not use a fourth "full attention" layer — all cell visibility is governed entirely by relational structure):

1) Column attention: a cell may attend to other cells in the sequence that come from the same column in the table.

2) Outbound attention (called "feature attention" in the RT paper): a cell may attend to _other cells in its own row_,
as well as _rows reachable by following foreign keys outward_ (F→P direction). For example, if a cell comes from an
"orders" row with `customer_id` pointing to a "customers" table:

orders
|   id   |   value   |   customer_id   |
|   1    |   <MASK>  |   23            |

customers
|   id   |   birthdate   |
|   23   |   01-02-92    |

then outbound attention allows all cells in orders row 1 to attend to cells both *in that row* as well as *in the
customers row for customer_id 23* — because the foreign key on orders points outward to customers.

3) Inbound attention (called "neighbor attention" in the RT paper): a cell may attend to cells from rows whose foreign
keys point *into* the cell's row (P→F direction) — the reverse of outbound attention. Where outbound attention lets a
child row see its parents, inbound attention lets a parent row see its children. This is analogous to message-passing
in GNNs: a parent entity aggregates signals from its child rows.

Continuing the example above, suppose customer 23 has several orders:

orders
|   id   |   value   |   customer_id   |
|   1    |   <MASK>  |   23            |
|   7    |   42.00   |   23            |
|   12   |   18.50   |   23            |

customers
|   id   |   birthdate   |
|   23   |   01-02-92    |

Inbound attention allows the cells in the *customers* row (id 23) to attend to cells in all three order rows (1, 7,
12), because those orders have a foreign key (`customer_id`) pointing into customer 23. Note that order row 1 does
*not* directly see sibling orders 7 and 12 through inbound attention — that information flows indirectly over two
layers: orders 7/12 →(inbound attn)→ customers 23 →(outbound attn)→ orders 1.

## Value Encoding

Value encoding produces the initial hidden state **h₀** `[B, S, D]` fed into the first transformer layer. Inspired
by the Relational Transformer (Ranjan et al., 2025), each cell's representation is the **sum** of two projected
components:

1. A **column-name embedding** — *what table/column* this cell belongs to (schema identity).
2. A **value embedding** — *what data* this cell holds, gated by null status and target masking.

### Column-Name Encoder (always present)

For each cell, look up the frozen column-name embedding from the GPU-resident table (Section 5) using `column_ids`,
then project it into model space:

```
col_raw[b, s]  = column_embedding_table[column_ids[b, s]]          // [D_t]
col_enc[b, s]  = column_name_encoder(col_raw[b, s])                // [D]
               = Linear(D_t → D, bias=True)(col_raw[b, s])
```

This projected vector encodes the column's schema identity (column name, table name, and optional description). It is
the same for all rows sharing a column and is **never** gated or replaced — the model always knows which attribute it
is looking at.

### Type-Specific Value Encoders

Each semantic type has its own encoder that maps the raw cell value **directly into model space** (D). Every encoder
is an independent `nn.Linear` or `nn.Embedding` — there is no shared projection layer.
See [semantic_types.md](semantic_types.md) for details on each type's preprocessing and normalization.

| Semantic Type   | Module                     | Input dim    | Learnable module                              |
|-----------------|----------------------------|--------------|-----------------------------------------------|
| **Identifier**  | Learned constant           | —            | `identifier_emb ∈ ℝ^D`                       |
| **Numerical**   | `nn.Linear(1, D, bias)`    | 1 (scalar)   | `numerical_encoder: Linear(1 → D)`           |
| **Timestamp**   | `nn.Linear(D_time, D, bias)` | D_time (15) | `timestamp_encoder: Linear(15 → D)`          |
| **Boolean**     | `nn.Embedding(2, D)`       | {0, 1}       | `boolean_encoder: Embedding(2, D)`           |
| **Categorical** | `nn.Linear(D_t, D, bias)`  | D_t          | `categorical_encoder: Linear(D_t → D)`       |
| **Text**        | `nn.Linear(D_t, D, bias)`  | D_t          | `text_encoder: Linear(D_t → D)`              |

**Identifier** cells carry no value — they signal entity presence only. Since identifiers are never null
(see [semantic_types.md](semantic_types.md)), every identifier cell receives the same learned constant vector in ℝ^D.

**Numerical** values are already z-score normalized during preprocessing, so the linear layer maps a zero-mean,
unit-variance scalar into a D-dimensional vector. This is the simplest value encoder: one weight vector plus a bias.

**Timestamp** values arrive as a 15-dimensional encoding (7 cyclic sin/cos pairs + z-scored epoch microseconds),
so the encoder is a small matrix multiply from ℝ^15 into ℝ^D.

**Boolean** values index into a 2-row embedding table in ℝ^D (row 0 = false, row 1 = true). Using `nn.Embedding`
rather than a linear layer is natural here because boolean is inherently categorical with exactly two values — there
is no meaningful interpolation between true and false.

**Categorical** values are stored as frozen D_t-dimensional vectors in the GPU-resident categorical embedding table
(Section 5b). The encoder gathers the frozen vector using `categorical_embed_ids` (global indices) and applies a
learned linear projection from D_t into D.

**Text** values are stored as frozen D_t-dimensional vectors in the per-batch `text_batch_embeddings` tensor
(Section 6). The encoder gathers the frozen vector using `text_embed_ids` (batch-local indices) and applies a
learned linear projection from D_t into D.

Using separate `nn.Linear` modules for categorical and text (rather than a shared one) allows the model to learn
distinct projections — categorical values are low-cardinality and repeated across rows, while text values are
typically unique, so the optimal projection may differ.

The learned projection for categorical values is especially important because frozen text embeddings for categories
within the same column are often very close in cosine similarity (e.g., `"status is enabled"`, `"status is disabled"`,
`"status is active"` share most of their tokens). The `categorical_encoder` linear layer can learn to amplify the
small directional differences that the frozen encoder does preserve, effectively pushing apart categories that
the model needs to distinguish. This is further reinforced at loss time: the same `categorical_encoder` projection
is applied to the full set of category embeddings for the target column when computing cross-entropy loss, so the
projection is trained from both the input (context encoding) and output (loss computation) sides. See Loss
Computation for details.

TODO(mrdmnd): experiment with replacing the single `nn.Linear` categorical encoder with a small MLP
(e.g., `Linear(D_t, D) → GELU → Linear(D, D)`) to see if nonlinear separation of close category embeddings
improves prediction accuracy on high-cardinality categorical targets.

Because the batch uses a dense value layout (Section 2), all six encoder paths can run in parallel across all
positions. Non-matching type slots are zero-filled by construction, so the correct output per position is
selected via `semantic_types`:

```
raw_val[b, s] = Σ_t  one_hot(semantic_types[b, s], t) · encoder_t(values[b, s])    // [D]
```

In practice the sum collapses to a single non-zero term per position.

### Null Gating (data-level missingness)

The `is_null` tensor participates directly in value encoding by **replacing** the type-specific value with a
learned **null embedding** for cells whose database value is genuinely NULL:

```
val_or_null[b, s] = is_null[b, s] · null_emb
                  + (1 − is_null[b, s]) · raw_val[b, s]          // [D]
```

`null_emb ∈ ℝ^D` is a single learned vector shared across all semantic types. It gives the model an explicit,
trainable signal for missingness — distinct from a zero value (which is a valid z-score for numericals), and distinct
from padding (which is excluded from attention entirely). The column encoding is **unaffected**: the model still
knows *which* column is null; only the value content is replaced.

A shared null embedding (rather than per-type) is sufficient because the column encoding already disambiguates types.
The model can learn type-specific null behavior through the interaction of column identity and the null signal in
downstream attention layers.

**`is_null` vs. `is_target` — these are not the same thing:**

| | `is_null` | `is_target` |
|---|---|---|
| **What it represents** | Data property — the database cell was SQL `NULL` | Training intervention — we hide the value for prediction |
| **When it is set** | Determined by the source data; fixed for a given cell | Determined by the task definition; one per sequence |
| **Effect on value embedding** | Replace with `null_emb` | Replace with `mask_emb` |
| **Column embedding affected?** | No | No |
| **Can co-occur?** | Yes — a target cell may be null (the model must predict that it is null) | Yes |

### Target Masking (training-level hiding)

For cells where `is_target` is true, the value embedding — whether null-gated or not — is further replaced by
a learned **mask embedding**:

```
val_final[b, s] = is_target[b, s] · mask_emb
               + (1 − is_target[b, s]) · val_or_null[b, s]       // [D]
```

`mask_emb ∈ ℝ^D` signals "predict me." The column encoding is preserved, so the model knows *which column* to
predict but not the value.

Target masking takes **priority** over null gating: even if the target cell happens to be null, the forward pass
sees `mask_emb`, not `null_emb`. The ground-truth null status is only consulted at loss time, where it determines
whether the null-prediction head contributes to the loss (see Loss Computation).

### Combination

The final cell embedding is the sum of the column encoding and the (gated) value encoding, followed by
RMSNorm (see Training Stability & Numerics):

```
h₀[b, s] = RMSNorm( col_enc[b, s]  +  val_final[b, s] )           // [D]
```

Both `col_enc` and `val_final` are already in ℝ^D — the column-name encoder and each type-specific value encoder
project into model space independently. This additive combination (analogous to how positional encodings are added
to token embeddings in standard transformers) allows the model to weight schema identity and data content through the
learned projections: the column encoder can emphasize features useful for attention routing (query/key computation),
while the value encoders can emphasize features that carry content signal.

Padding positions (`is_padding = true`) receive **h₀ = 0**: both encodings are zeroed out before the sum.
Combined with the padding mask on attention (Section 3), these positions are invisible.

### Summary of Value-Encoding Parameters

| Parameter              | Shape              | Count (D=256, D_t=256, D_time=15) |
|------------------------|--------------------|-------------------------------------|
| `column_name_encoder`  | `Linear(D_t, D)`  | 65,792                              |
| `numerical_encoder`    | `Linear(1, D)`    | 512                                 |
| `timestamp_encoder`    | `Linear(15, D)`   | 4,096                               |
| `boolean_encoder`      | `Embed(2, D)`     | 512                                 |
| `categorical_encoder`  | `Linear(D_t, D)`  | 65,792                              |
| `text_encoder`         | `Linear(D_t, D)`  | 65,792                              |
| `identifier_emb`       | `[D]`             | 256                                 |
| `null_emb`             | `[D]`             | 256                                 |
| `mask_emb`             | `[D]`             | 256                                 |
| **Total**              |                    | **203,264**                         |

These ~203K parameters are a negligible fraction of the overall model and are trained end-to-end alongside the
attention layers.

## Decoder Heads

Decoder heads are lightweight modules that map the transformer's final hidden state `h[B, S, D]` into per-type
predictions. To avoid JIT graph breaks in JAX, **every decoder head runs on every position in every forward pass**,
regardless of which type the batch actually targets. The type-specific loss computation then selects the relevant
outputs via masking — the computation graph is fully static with no conditional branches.

The wasted compute is negligible: each head is a single `nn.Linear`, and the attention layers dominate cost by
orders of magnitude.

### Head Inventory

| Head                    | Module              | Output shape     | Activation    | Purpose                             |
|-------------------------|---------------------|------------------|---------------|-------------------------------------|
| **Null head**           | `Linear(D, 1)`     | `[B, S]`         | sigmoid → p   | Binary: is the cell null?           |
| **Numerical decoder**   | `Linear(D, 1)`     | `[B, S]`         | none (raw)    | Predicted z-score value             |
| **Boolean decoder**     | `Linear(D, 1)`     | `[B, S]`         | sigmoid → p   | Binary: true or false?              |
| **Timestamp decoder**   | `Linear(D, D_time)` | `[B, S, 15]`   | none (raw)    | Predicted 15-dim timestamp encoding |
| **Categorical decoder** | `Linear(D, D)`     | `[B, S, D]`     | none (raw)    | Projected vector for category retrieval |

There is **no text decoder** — text columns (`Identifier`, `Text`, `Ignored`) are never prediction targets
(see [semantic_types.md](semantic_types.md)).

**Null head** is used for every target regardless of type. It produces a logit that, after sigmoid, gives the
probability that the target cell is null.

**Numerical decoder** produces a single scalar — the predicted z-score normalized value. At inference, this is
denormalized using the target column's stored mean and std from `ColumnStats::Numerical`.

**Boolean decoder** produces a logit that, after sigmoid, gives the probability that the value is true.

**Timestamp decoder** produces a 15-dimensional vector matching the encoding format of `timestamp_values`
(7 cyclic sin/cos pairs + z-scored epoch microseconds). At inference, the epoch component is denormalized using
global timestamp statistics and the cyclic components can be decoded back to calendar fields.

**Categorical decoder** projects the hidden state into ℝ^D — the **same space** that `categorical_encoder` maps
frozen category embeddings into during value encoding. At training time, the decoder's output is scored against all
K category embeddings for the target column via cross-entropy (see Loss Computation). At inference, the predicted
category is the one with the highest score. This deliberately reuses the same representation space as the input
encoder, so the learned `categorical_encoder` projection is optimized from both the input side (context encoding)
and the output side (loss computation). Because the categorical embedding table is GPU-resident (Section 5b),
all K category embeddings for any column are always available without per-batch shipping.

### Model Output Structure

The forward pass produces a `ModelOutput` with the following fields:

```
ModelOutput:
    h:            [B, S, D]       # final hidden states from transformer
    null_logits:  [B, S]          # null head output (raw logits)
    num_preds:    [B, S]          # numerical predictions (z-score scale)
    bool_logits:  [B, S]          # boolean head output (raw logits)
    ts_preds:     [B, S, D_time]  # timestamp predictions (15-dim)
    cat_preds:    [B, S, D]       # categorical projections (ℝ^D)
```

All fields are dense `[B, S, ...]` tensors computed unconditionally. The `is_target` and `target_stype` determine
which fields are consumed during loss computation and inference.

### Decoder Parameter Summary

| Parameter              | Shape             | Count (D=256, D_time=15) |
|------------------------|-------------------|--------------------------|
| `null_head`            | `Linear(D, 1)`   | 257                      |
| `numerical_decoder`    | `Linear(D, 1)`   | 257                      |
| `boolean_decoder`      | `Linear(D, 1)`   | 257                      |
| `timestamp_decoder`    | `Linear(D, 15)`  | 3,855                    |
| `categorical_decoder`  | `Linear(D, D)`   | 65,792                   |
| **Total**              |                   | **70,418**               |

---

## Loss Computation

### Overview

Loss is computed **only at target positions** (where `is_target = true`). Currently there is exactly one target
per sequence, so this selects B positions from the `[B, S]` tensors. Each target cell produces two loss terms:

1. **Null loss** — always computed, regardless of type. "Is this cell null?"
2. **Type-specific loss** — computed for the type matching `target_stype`, **gated by non-nullness**.

The combined per-cell loss is:

```
cell_loss = null_loss  +  (1 - is_null) · type_loss
```

When the ground truth is null (`is_null = 1`), the type-specific term vanishes — there is no meaningful value to
regress or classify, so only the null head receives gradient. When the ground truth is non-null, both terms
contribute. This lets the model learn that nullness carries semantic information (e.g., "no accepted answer yet")
while still producing calibrated value predictions.

### Null Loss

Binary cross-entropy between the null head's output and the ground-truth null status:

```
p_null    = sigmoid(null_logits[b, t])
null_loss = BCE(p_null, is_null[b, t])
```

This is computed for every target regardless of `target_stype`.

### Type-Specific Losses

| Target type     | Loss function       | Prediction                | Ground truth                                                              |
|-----------------|---------------------|---------------------------|---------------------------------------------------------------------------|
| **Numerical**   | MSE                 | `num_preds[b, t]`         | `numeric_values[b, t]`                                                    |
| **Boolean**     | BCE                 | `sigmoid(bool_logits[b, t])` | `bool_values[b, t]`                                                    |
| **Timestamp**   | MSE (15-dim)        | `ts_preds[b, t]`          | `timestamp_values[b, t]`                                                  |
| **Categorical** | Cross-entropy       | `cat_preds[b, t]`         | `categorical_encoder(cat_emb_table[cat_emb_start : cat_emb_start + K])`   |

**Numerical**: Mean squared error between predicted and ground-truth z-score values. Since both are normalized per
column during preprocessing, the loss is scale-invariant across columns with different ranges. At inference,
predictions are denormalized using the column's stored mean and std.

**Boolean**: Binary cross-entropy — straightforward classification.

**Timestamp**: Mean squared error across all 15 dimensions. The 14 cyclic sin/cos components are bounded in [-1, 1]
and the epoch component is z-scored, so they are roughly comparable in scale. The model learns to predict at multiple
temporal resolutions simultaneously (second, minute, hour, day-of-week, etc.).

**Categorical**: Cross-entropy over the full category set for the target column. Because the categorical embedding
table is GPU-resident (Section 5b), all K category embeddings for the target column are always available. The loss
computes logits by projecting all K frozen embeddings through `categorical_encoder` and taking a dot product with
the decoder's output:

```
# cat_emb_table is the full GPU-resident categorical embedding table [Vc, D_t]
# cat_emb_start and K come from the target column's ColumnStats::Categorical
col_cat_embeds = cat_emb_table[cat_emb_start : cat_emb_start + K]          # [K, D_t]
col_cat_proj   = categorical_encoder(col_cat_embeds)                        # [K, D]
logits         = cat_preds[b, t] @ col_cat_proj.T                          # [K]
cat_loss       = cross_entropy(logits, target_cat_index)                    # scalar
```

Where `target_cat_index` is the index (0..K-1) of the ground-truth category within the column's sorted category
list. This is derived from `categorical_embed_ids[b, t]` by subtracting `cat_emb_start`.

Because `categorical_encoder` is shared between encoding and loss, it receives gradient from both directions:
input encoding pulls toward useful representations, while loss computation pushes apart all K categories in the
column. Cross-entropy is more discriminative than cosine distance because it explicitly pushes probability mass
away from *all* wrong categories, not just implicitly via directional similarity.

**Padding for static shapes.** Because K varies across categorical columns, and JAX requires static tensor shapes
for JIT compilation, the category set is padded to a fixed `max_K` (the maximum cardinality across all categorical
columns in the database, known at preprocessing time). Padding positions are masked out before softmax:

```
logits = jnp.where(valid_mask, logits, -1e9)     # kill padding logits
cat_loss = cross_entropy(logits, target_cat_index)
```

The wasted compute is trivial: one `[D] @ [max_K, D].T` matmul at B target positions.

### Graph-Break-Free Type Selection

Every batch is homogeneous — one task, one `target_stype` — so we could branch on `target_stype` and only compute
the matching type loss. JAX would JIT-trace once per type (4 traces), which is acceptable. However, for a fully
static computation graph we instead compute **all four type losses** and select via masking:

```python
# Extract target positions (B scalar values each)
t = target_positions                       # [B] indices where is_target is true

# All type losses computed unconditionally (static graph)
num_loss  = mse(num_preds[t],   numeric_values[t])           # [B]
bool_loss = bce(bool_logits[t], bool_values[t])               # [B]
ts_loss   = mse(ts_preds[t],   timestamp_values[t])           # [B]
cat_loss  = cat_cross_entropy(cat_preds[t], cat_emb_table, cat_emb_start, K)  # [B]

# Select the relevant loss via dot product with one-hot (no branching)
type_losses = stack([num_loss, bool_loss, ts_loss, cat_loss])  # [4, B]
type_selector = one_hot(target_stype, 4)                       # [4] (same for all B)
type_loss = type_selector @ type_losses                        # [B]
```

The wasted compute is trivial: three extra `Linear(D, 1)` evaluations and one `Linear(D, D)` at B positions.

### Batch Loss

The final scalar loss averages over all B target cells:

```
batch_loss = mean_b[ null_loss_b  +  (1 - is_null[b, t]) · type_loss_b ]
```

### Inference Output

At inference the model returns the same `ModelOutput`. Post-processing selects the relevant head based on
`target_stype`:

| Target type     | Inference prediction                                                                |
|-----------------|-------------------------------------------------------------------------------------|
| **Numerical**   | Denormalize: `pred = num_preds[b, t] * col_std + col_mean`                         |
| **Boolean**     | Threshold: `pred = sigmoid(bool_logits[b, t]) > 0.5`                               |
| **Timestamp**   | Denormalize epoch component, decode cyclic components to calendar fields             |
| **Categorical** | `argmax_k( cat_preds[b,t] @ categorical_encoder(cat_emb_table[start:start+K]).T )` |
| **Null**        | `is_null = sigmoid(null_logits[b, t]) > 0.5` (checked first for all types)         |

The null head is always checked first: if `p_null > 0.5`, the prediction is NULL regardless of the type-specific
output.

---

## Transformer Block Structure

Before discussing stability and numerics, we need to pin down the actual repeated block. Confluence uses a
**pre-norm** architecture (normalize → sublayer → residual add), which is strictly more stable than post-norm
(residual add → normalize) because the residual stream stays on a clean linear path without passing through
normalization. All modern large-scale transformers (LLaMA, Gemma, etc.) use pre-norm.

Each "layer" in Confluence is a **triple** of attention sublayers followed by a feed-forward sublayer.
Each attention sublayer applies **SDPA output gating** (see below) before the residual add:

```
# One Confluence layer (repeated N_layers times)
x_norm = RMSNorm(x)
x = x + GatedAttention(OutboundSDPA, x_norm, W_gate_out)    # outbound (self + FK parents)
x_norm = RMSNorm(x)
x = x + GatedAttention(InboundSDPA, x_norm, W_gate_in)      # inbound (FK children)
x_norm = RMSNorm(x)
x = x + GatedAttention(ColumnSDPA, x_norm, W_gate_col)      # same-column peers
x = x + FFN(RMSNorm(x))                                     # feed-forward
```

Where `GatedAttention` is defined as:

```
def GatedAttention(sdpa_fn, x_norm, W_gate):
    attn_out = sdpa_fn(x_norm)                  # permute → block-sparse MHA → unpermute → W_O
    gate = sigmoid(x_norm @ W_gate)             # [B, S, D] — head-specific elementwise gate
    return attn_out * gate                      # elementwise modulation before residual add
```

A final RMSNorm is applied after the last layer, before the decoder heads:

```
h_final = RMSNorm(x)                        # [B, S, D]
```

Each attention sublayer internally performs: permute → QK-normed multi-head attention → unpermute (see
Attention Mechanisms above). The FFN sublayer uses SwiGLU (see below).

### SDPA Output Gating

Following [Qiu et al., 2025 — "Gated Attention for Large Language Models"](https://arxiv.org/abs/2505.06708),
each attention sublayer applies a **head-specific sigmoid gate** to the SDPA output before the residual
connection. The gate modulates the attention output elementwise:

```
gated_output[b, s] = attn_output[b, s] * sigmoid(x_norm[b, s] @ W_gate)
```

Where `W_gate ∈ ℝ^{D × D}` is a learnable weight matrix (one per attention sublayer, three per layer). The
gate scores are computed from the same normalized input `x_norm` that feeds the attention sublayer's Q/K/V
projections.

**Why this helps — non-linearity.** In standard multi-head attention, the value and output projections
(W_V and W_O) compose into a single low-rank linear map: `o_i^k = Σ_j S_ij · X_j (W_V^k · W_O^k)`. With
d_head < D, this is a rank-d_head bottleneck. GQA (sharing K/V heads across query groups) further reduces
expressiveness. Inserting a nonlinear gate between SDPA output and W_O breaks this bottleneck, increasing the
effective capacity of the value-to-output mapping. This argument is *scale-independent* — it applies equally
at D=256 as at D=4096 — and is especially relevant to Confluence because we have three attention sublayers per
layer, each with its own W_V/W_O bottleneck.

**Why this helps — sparsity.** The sigmoid gate learns to produce mostly near-zero scores, acting as an
input-dependent filter that suppresses irrelevant context from the attention output. This is particularly
useful for inbound attention, where parent rows aggregate over all children via softmax (which forces a
distribution summing to 1 and cannot "say nothing"). A sparse gate after SDPA gives the model an explicit
mechanism to suppress uninformative child signals. The paper demonstrates that this sparsity also eliminates
the "attention sink" phenomenon (initial tokens hoarding attention mass) and improves training stability by
reducing massive activations in the residual stream.

**Initialization.** `W_gate` is initialized with Xavier uniform, consistent with the other projection matrices.
At initialization, the gate outputs are `sigmoid(small random values) ≈ 0.5`, so the gate begins as an
approximate half-scaling of the attention output. This is compatible with the scaled residual initialization
(W_O is already scaled by `1/√(4 · N_layers)`), but the effective initial residual contribution is halved.
Consider adjusting W_O's initialization scale by `1/√(4 · N_layers) · √2` to compensate if needed.

**No bias.** Following the same convention as W_Q, W_K, W_V, and W_O, the gate projection omits the bias
term.

**Optimizer.** `W_gate` is a 2D weight matrix, so it falls under **Muon** in the split optimizer strategy.

**Interaction with block-sparse SDPA — no fused kernel required.** The gate operates entirely *outside* the
block-sparse attention kernel boundary. The computation flow is:

```
┌──────────────────────────────────────────────┐
│  Block-sparse SDPA kernel (unchanged)        │
│  permute → Q,K,V proj → block-sparse attn   │
│  → unpermute → W_O projection                │
│  (custom Triton kernel, handles masks, etc.) │
└──────────────────┬───────────────────────────┘
                   │ attn_out [B, S, D]
                   ▼
         ┌─────────────────────┐
         │  Gate (standard ops) │    x_norm @ W_gate → sigmoid
         │  attn_out * gate     │    elementwise multiply
         └─────────┬───────────┘
                   │ gated_out [B, S, D]
                   ▼
              residual add: x = x + gated_out
```

The gate computation is two standard ops: one matmul (`x_norm @ W_gate`) and one fused sigmoid-multiply.
XLA will automatically fuse the sigmoid and elementwise multiply into a single kernel. The backward pass
is equally clean: the SDPA kernel receives `d(gated_out) * gate` as its upstream gradient (just a
scaled version of the incoming gradient), requiring no modification to the SDPA backward kernel. The gate
gradient `d(gate) = d(gated_out) * attn_out` flows through sigmoid and then the W_gate matmul, all
standard autograd.

**Parameter overhead per attention sublayer:**

| Parameter     | Shape       | Count (D=256) |
|---------------|-------------|----------------|
| `W_gate`      | `[D, D]`   | 65,536         |

With three attention sublayers per layer, this adds **196,608 parameters per layer** — comparable to a single
W_Q or W_V matrix. At the model scale this is negligible.

**Wall-time overhead.** The paper reports <2% latency increase for their 15B MoE model. At our smaller scale,
the matmul + sigmoid + multiply adds one D×D matmul per attention sublayer (three per layer), which is dwarfed
by the block-sparse attention and SwiGLU FFN compute.

---

## Training Stability & Numerics

### Precision: BF16

All forward and backward computation uses **BF16** (bfloat16). BF16 is preferred over FP16 for training because:

| Property | BF16 | FP16 |
|----------|------|------|
| Exponent bits | 8 (same as FP32) | 5 |
| Mantissa bits | 7 | 10 |
| Dynamic range | ±3.4 × 10³⁸ | ±65504 |
| Loss scaling needed? | No | Yes |

FP16's narrow exponent range means gradients frequently overflow or underflow during training, requiring a
loss-scaling mechanism (dynamic or static) to keep values representable. BF16 has the same exponent range as
FP32, so gradients and activations almost never leave the representable range — **no loss scaling is needed**.
The tradeoff is 3 fewer mantissa bits (lower precision), but in practice this has negligible impact on model
quality: the noise from reduced precision acts as a mild regularizer, and stochastic rounding in accumulations
compensates for most of the precision loss.

In JAX, BF16 training is configured via `jax.default_matmul_precision` and explicit dtype policies:

- **Activations**: BF16. Hidden states, attention logits, and FFN intermediates are all stored and computed in BF16.
- **Master weights**: FP32. The optimizer maintains a full-precision copy of all parameters. At each step, weights
  are cast to BF16 for the forward/backward pass, and the FP32 master copy is updated with the gradient.
- **Optimizer state**: FP32. Momentum buffers and second-moment estimates (for AdamW) are FP32 to avoid
  accumulation drift.
- **Loss computation**: FP32. The final loss reduction (mean over batch) and any log/exp operations (BCE, cross-
  entropy softmax) are upcast to FP32 to avoid numerical issues in the tail of the distribution.
- **Gradient accumulation**: FP32. When using gradient accumulation across micro-batches, gradients are accumulated
  in FP32.

This "BF16 compute, FP32 state" pattern is standard across JAX training codebases (e.g., MaxText, PaLM, Gemma).

### Normalization: Zero-Centered RMSNorm

All normalization layers use **zero-centered RMSNorm** rather than standard LayerNorm or vanilla RMSNorm.

Standard LayerNorm centers by the mean and scales by the standard deviation:

```
LayerNorm(x) = γ ⊙ (x − μ) / √(σ² + ε)
```

Vanilla RMSNorm drops the mean-centering step and normalizes by the RMS alone:

```
RMSNorm(x) = γ ⊙ x / √(mean(x²) + ε)
```

Zero-centered RMSNorm reparameterizes the scale as `(1 + γ)` instead of `γ`, with `γ` initialized to **zero**:

```
ZC-RMSNorm(x) = (1 + γ) ⊙ x / √(mean(x²) + ε)
```

This has three benefits over LayerNorm:
1. **Faster**: Like vanilla RMSNorm, no mean computation is needed — one fewer reduction pass per norm site.
2. **Equally effective**: Empirical results across LLaMA, Gemma, and other large-scale models show no quality
   degradation from dropping the centering step.
3. **Better gradient flow at initialization**: With `γ = 0`, the layer starts as a pure normalization (scale = 1
   everywhere). This means the pre-norm residual blocks begin as near-identity functions — the normalized input
   flows straight through the residual connection with minimal distortion. During early training, the network
   behaves like a shallow model and deepens gradually as `γ` moves away from zero. This is strictly better than
   initializing `γ = 1` (vanilla RMSNorm), where the initial scale is already a free parameter that can amplify
   or suppress dimensions before any learning has occurred.

This technique was introduced in Gemma 2 and has since been adopted in other architectures for its improved
training stability, particularly in deep networks.

RMSNorm is applied at these sites:
- Before each of the three attention sublayers (outbound, inbound, column)
- Before the FFN sublayer
- After value encoding (producing h₀)
- After the final layer (before decoder heads)

The learnable scale parameter `γ ∈ ℝ^D` is initialized to **0.0** at every site (giving an effective scale
of 1.0 via the `(1 + γ)` reparameterization).

### QK Norm

To prevent attention logit explosion — where the dot products `Q·K^T` grow large enough to push softmax into
a near-one-hot regime, killing gradient flow — we apply **L2 normalization to queries and keys** after their
respective linear projections:

```
Q = L2Norm(W_Q @ x) · √d_head
K = L2Norm(W_K @ x)
attn_logits = Q @ K^T / √d_head     # equivalent to: cosine_similarity(q, k) scaled by √d_head
```

After L2 normalization, every query and key vector has unit norm, so the dot products are bounded in [-1, 1]
before the `√d_head` scaling. This provides a hard upper bound on attention logit magnitude regardless of hidden
dimension, layer depth, or training stage.

The `√d_head` factor is applied as a **learnable scalar temperature per head**, initialized to `√d_head` to
recover the standard scaled dot-product attention at initialization. During training, each head can learn to
sharpen or broaden its attention pattern by adjusting this temperature.

QK norm is critical for stable BF16 training: without it, attention logits can reach magnitudes that exceed
BF16's effective precision (which has only 7 mantissa bits), causing softmax to saturate and gradients to vanish.
With QK norm, logits stay in a numerically comfortable range throughout training. This technique is used in
ViT-22B, Gemma 2, and Chameleon, among others.

### Feed-Forward Network: SwiGLU

The feed-forward sublayer uses **SwiGLU** (SiLU-gated linear unit) instead of the standard GELU FFN:

Standard FFN:
```
FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2
```

SwiGLU FFN:
```
SwiGLU(x) = W_2 · (SiLU(W_gate · x) ⊙ (W_up · x))
```

Where `SiLU(z) = z · sigmoid(z)` is the Sigmoid Linear Unit, `⊙` is element-wise multiplication, and the three
weight matrices are:
- `W_gate ∈ ℝ^{D × D_ff}` — gate projection
- `W_up   ∈ ℝ^{D × D_ff}` — value projection
- `W_2    ∈ ℝ^{D_ff × D}` — down projection

The gating mechanism allows the network to selectively pass or suppress features, improving expressivity over
a simple nonlinearity. Empirically, SwiGLU consistently outperforms GELU and ReLU FFNs at the same parameter
count across model scales (PaLM, LLaMA, Gemma all use it).

**Hidden dimension sizing.** Standard FFN uses `D_ff = 4D`. SwiGLU has three weight matrices instead of two,
so to match parameter count we use `D_ff = ⌈(8/3)D⌉` rounded up to the nearest multiple of 256 for hardware
alignment. At D=256: `D_ff = ceil(682.67 / 256) × 256 = 768`.

**No bias terms.** Following modern practice (LLaMA, Gemma, PaLM), all linear layers in the attention
projections (W_Q, W_K, W_V, W_O) and FFN (W_gate, W_up, W_2) omit bias terms. Biases add negligible
capacity but complicate optimizer logic (they're 1D parameters requiring different treatment under Muon)
and prevent clean fusion with preceding RMSNorm layers in optimized kernels.

The value encoders and decoder heads (Section 5 and Decoder Heads) retain biases, since they are small and
operate at the boundary between frozen embeddings and model space where a bias offset is more meaningful.

### Optimizer: Muon + AdamW

Training uses a **split optimizer** strategy:

- **Muon** for all 2D weight matrices (attention projections, FFN weights)
- **AdamW** for everything else (embeddings, RMSNorm scales, encoder/decoder biases, 1D parameters)

#### Muon

Muon (Momentum with Orthogonalization via Newton-Schulz) is a recently developed optimizer that applies an
approximate Newton step in the spectral domain for matrix-shaped parameters. The key insight is that for
2D weight matrices, the optimal preconditioner in the Shampoo/SOAP family can be efficiently approximated
using Newton-Schulz iterations on the gradient's momentum, without maintaining or inverting large second-moment
matrices.

For each 2D parameter W with gradient G and momentum buffer M:

```
M ← β₁ · M + (1 − β₁) · G                    # standard momentum
M̃ = NewtonSchulz5(M)                           # ~5 iterations of NS orthogonalization
W ← W − lr · M̃                                 # update
```

The `NewtonSchulz5` procedure orthogonalizes the momentum matrix (bringing all singular values to 1),
effectively applying an optimal per-direction learning rate in weight space. This is significantly cheaper
than full Shampoo (no matrix square roots) while capturing most of the benefit.

Muon hyperparameters:
- `β₁ = 0.95` (momentum coefficient)
- `lr_muon`: typically 0.02 (higher than AdamW because the NS step normalizes gradient scale)
- Newton-Schulz iterations: 5 (sufficient for convergence in practice)

#### AdamW

Standard AdamW is used for all non-2D parameters:

- `β₁ = 0.9`, `β₂ = 0.95`
- `weight_decay = 0.1` (applied to embeddings and encoder/decoder weights, not to biases or norms)
- `ε = 1e-8`

#### Learning Rate Schedule

Both optimizers share a synchronized schedule:

```
lr(t) = lr_peak · warmup(t) · decay(t)

warmup(t) = min(1, t / T_warmup)                              # linear warmup
decay(t)  = 0.1 + 0.9 · (1 + cos(π · (t - T_warmup) / (T_total - T_warmup))) / 2   # cosine decay to 10% of peak
```

- `T_warmup`: ~2000 steps (or ~1% of total training steps, whichever is larger)
- `lr_peak` for Muon: ~0.02
- `lr_peak` for AdamW: ~3e-4

The cosine schedule decays to 10% of peak (not zero) to maintain some learning capacity in the final phase.
The warmup is critical for stability with BF16: large initial gradients before the optimizer state has
warmed up can cause divergence.

### Weight Initialization

Proper initialization prevents signal explosion or collapse in the residual stream:

- **Attention projections** (W_Q, W_K, W_V): Xavier uniform, `U(-√(6/(D+D)), √(6/(D+D)))`.
- **Attention output projection** (W_O): Xavier uniform, **scaled by `1/√(4 · N_layers)`**. This ensures that
  at initialization, the variance contribution of each residual branch is O(1/N_layers), preventing the residual
  stream from growing as O(√N_layers). The factor of 4 accounts for the four residual-adding sublayers
  (three attention + one FFN) per logical layer.
- **SDPA gate projection** (W_gate): Xavier uniform (unscaled). At initialization, the gate outputs
  `sigmoid(small random values) ≈ 0.5`, which halves the attention output before the residual add. This
  effective halving compounds with the W_O residual scaling to produce well-controlled initial residual
  contributions. If training shows instability in early steps, consider compensating by scaling W_O init
  by an additional factor of `√2`.
- **FFN down projection** (W_2 in SwiGLU): Xavier uniform, scaled by `1/√(4 · N_layers)` (same reasoning as W_O).
- **FFN gate and up projections** (W_ffn_gate, W_up): Xavier uniform (unscaled — the down projection handles
  the residual scaling).
- **RMSNorm scales** (γ): initialized to 0.0 (effective scale is `1 + γ = 1.0` at init).
- **Embeddings** (null_emb, mask_emb, identifier_emb, boolean_encoder): normal, std=0.02.
- **Value encoder Linear layers**: Xavier uniform (unscaled — these feed into the first RMSNorm).

This initialization scheme is a variant of the GPT-2 / μP approach adapted for our 4-sublayer residual structure.

### Gradient Clipping

Global gradient norm clipping at **max_norm = 1.0**, applied to all parameters jointly before the optimizer
step. This is a safety net against rare exploding-gradient events (e.g., an unusually adversarial batch);
under normal training with the above stability measures (BF16, RMSNorm, QK norm, proper init), the gradient
norm should rarely exceed the clip threshold after warmup.

Gradient clipping is applied in FP32 (on the FP32 gradients before they enter the optimizer).

### Z-Loss Regularization (Categorical Cross-Entropy)

For categorical targets, the cross-entropy loss includes a small **z-loss** penalty that discourages the
pre-softmax logits from growing large:

```
z_loss = 1e-4 · mean(log(Σ_k exp(logits_k))²)
cat_loss_total = cross_entropy(logits, target) + z_loss
```

This penalizes the log-partition function (log-sum-exp of logits), which grows when logits have large
magnitude. Without z-loss, confident predictions push logits to extreme values, which in BF16 can cause
softmax saturation and training instability. The coefficient (1e-4) is small enough to not meaningfully affect
converged accuracy but large enough to prevent logit drift during training.

Z-loss was introduced in the ST-MoE paper and is used in PaLM and Gemma for the same stability purpose.

### Summary of Stability Techniques

| Technique | Where | Why |
|-----------|-------|-----|
| BF16 mixed precision | All compute | Wide dynamic range, no loss scaling needed |
| FP32 master weights | Optimizer state | Prevent weight update drift |
| FP32 loss reduction | Loss computation | Numerical accuracy in log/exp operations |
| Pre-norm zero-centered RMSNorm | Every sublayer | Stable residual stream, near-identity init |
| QK norm + learned temperature | Attention layers | Bounded logits, prevents softmax saturation |
| SDPA output gating | Attention sublayers | Non-linearity between W_V/W_O, input-dependent sparsity |
| SwiGLU FFN | Feed-forward layers | Better expressivity per parameter |
| No bias in core layers | Attention + FFN + gate | Cleaner optimization, kernel fusion |
| Muon (2D) + AdamW (1D) | Optimizer | Spectral preconditioning for matrix weights |
| Scaled residual init | W_O, W_2 | O(1) residual variance at init |
| Linear warmup + cosine decay | LR schedule | Stable early training, controlled annealing |
| Gradient clipping (max_norm=1.0) | All parameters | Safety net against rare gradient spikes |
| Z-loss | Categorical CE | Prevents logit drift in BF16 |

