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

---

### 1. Cell identity tensors

These tell the model *what* each position in the sequence represents.

`semantic_types` is a `[B, S]` tensor of `int8` values. For each cell, this indicates the semantic type of the cell.
See [semantic_types.md](semantic_types.md) for more details on the semantic types.
Note that we will never include any cells with the `ignored` semantic type in any batch.

`column_ids` is a `[B, S]` tensor of `int32` values. For each cell, this indicates the global column index.

`row_ids` is a `[B, S]` tensor of `int32` values. For each cell, this indicates the global row index.


### 2. Per-type cell value tensors (dense layout)

Every sequence position has storage for every type, with zeros at non-matching positions.
This dense layout avoids scatter/gather ops — the forward pass simply multiplies by a type mask.
Categorical and text cells store **local indices into a `batch_embeddings` tensor**
rather than full 256-dim vectors (512 bytes), keeping the batch compact.
The model gathers from `batch_embeddings` during the forward pass.

`null_mask` is a `[B, S]` tensor of `bool` values. 
For each cell, this indicates whether the cell's value was NULL.
Null cells contribute only their column-name embedding — the value embedding is zeroed out.
The model can still learn from the *absence* of a value (e.g., a NULL rating means "not yet reviewed").

`numeric_values` is a `[B, S]` tensor of `f32` values.
For each cell, this contains the z-score normalized numerical value.

`timestamp_values` is a `[B, S, 15]` tensor of `f32` values.
For each cell, this contains the cyclic timestamp encoding: 7 sin/cos pairs + z-scored epoch microseconds.

`bool_values` is a `[B, S]` tensor of `bool` values.
For each cell, this contains the boolean value.

`categorical_embed_ids` is a `[B, S]` tensor of `u32` values.
For each categorical cell, this contains the local index into `batch_embeddings` for the categorical value embedding.

`text_embed_ids` is a `[B, S]` tensor of `u32` values.
For each text cell, this contains the local index into `batch_embeddings` for the text value embedding.

---

### 3. Validity and masking tensors

`target_mask` is a `[B, S]` tensor of `bool` values.
This is a mask indicating which cells are the target cells for the task.
We're starting with a single target cell per sequence (the one from the task definition).
Target cells are ultimately predicted by the model and the ground truth is used to compute the loss.
No separate prediction target tensors are needed — the ground-truth values (whether raw cell values for cell masking
tasks or derived values for aggregation tasks) are stored in the per-type cell value tensors (Section 2) at the
target position. The sampler writes the appropriate value into the matching type slot when constructing the batch.
At loss time, the model indexes into the per-type tensors using `target_mask` to retrieve the ground truth.

Masked cells have their value embedding replaced by a learnable `mask_embedding` vector;
however, the column-name embedding remains visible (the model knows *which* column is masked, just not the value).

**Null prediction.** The target cell's ground truth may be NULL (indicated by `null_mask` at `target_mask` positions).
The model includes a binary "is null?" prediction head for each target position, trained with binary cross-entropy
against the ground truth from `null_mask`. The overall loss for a target cell is:

```
loss = null_bce_loss + (1 - is_null) * type_specific_loss
```

When the ground truth is null, only the null head receives gradient signal — there is no meaningful type-specific
value to regress or classify. When the ground truth is not null, both the null head and the type-specific decoder
head contribute. This lets the model learn that nullness itself carries semantic information (e.g., "no accepted
answer yet", "rating not provided") and produce calibrated null predictions at inference time.

`padding_mask` is a `[B, S]` tensor of `bool` values.
If the cell is a padding cell (positions beyond the end of the actual cell sequence), this is `true`. 
Padded positions are excluded 
- from all attention patterns (OR'd into each mask) 
- from loss computation (no gradient signal)
- from masking (they should never be selected for masking in the first place)

---

### 4. Row-level adjacency tensors (compact attention mask inputs)

TODO(mrdmnd): fill this in

Instead of shipping three full `[B, S, S]` attention masks (~100 MB at B=32, S=1024), we ship two compact `[B, R, R]` row-level boolean adjacency matrices (~2.4 MB at R=200).
Note: this is using the LOCAL row indexing, not the global row indexing for the sequence.

The column attention mask needs no adjacency matrix — it is derived purely from `column_ids`.
The outbound and inbound attention masks are derived from the adjacency matrices here, though - they're computed at the
start of the forward pass.

TODO(mrdmnd): compute permutation matrices in the sampler so that these masks end up "block sparse" in the kernels, then
apply the permutation + inverse permutation in the forward pass.

---

### 5. Column-Name Embedding Table (resident on GPU, not part of the batch)

Column-name embeddings are stored in `column_embeddings.bin` and loaded into `ColumnMetadata.embedding` at startup, one per global
column. At the start of the training loop, we collect them into a `[num_columns, D_t]` tensor and upload it to GPU
shared memory once. This avoids per-batch CPU-to-GPU transfer for column identity information. The table is small
— `num_columns * D_t * 2` bytes — so it's fine to keep it resident for the entire run.

### 6. Vocabulary Embeddings (per-batch, from `vocab_embeddings.bin`)

For categorical and text cells, the universe of possible value embeddings is much larger and lives in a separate
file (`vocab_embeddings.bin`). We cannot afford to keep all of them in VRAM all the time, which means we ship a
subset to the GPU each batch.

Thankfully, we can do some clever deduplication:
All of the categorical and text values in the batch can be put into a single large tensor — let's say that our batch has
U unique categorical and text values. At batch-creation time, we gather the relevant embeddings from the vocab table.
We ship THIS deduplicated tensor to the GPU each batch, which should usually be smaller than including dense embeddings
for each cell in the batch (most of which would be zeros).

Note: `EmbeddingIdx` values stored in `ColumnSlice::Embedded` cells index directly into `vocab_embeddings.bin`
(0-based), with no offset for column-name embeddings.


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

