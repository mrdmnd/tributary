# Attention Mechanisms

Confluence uses three structured attention patterns per layer. Unlike the RT paper, there is no fourth "full attention"
layer — all cell visibility is governed entirely by relational structure.

## Attention patterns

**Column attention:** A cell attends to other cells from the same column.

```
column_mask[b, i, j] = (column_ids[b, i] == column_ids[b, j])
```

**Outbound attention** (RT's "feature attention"): A cell attends to cells in its own row and rows reachable by
following FK edges outward (F→P direction). For example, cells in an `orders` row with `customer_id = 23` can attend
to cells in `orders` row 1 *and* cells in `customers` row 23.

```
outbound_mask[b, i, j] = (ri == rj) || fk_adj[b, ri, rj]
```

The first term is the self-loop (own row); the second follows FK edges forward (child → parent).

**Inbound attention** (RT's "neighbor attention"): A cell attends to cells from rows whose FK columns point *into*
the cell's row (P→F direction) — the reverse of outbound. A parent aggregates signals from its children.

```
inbound_mask[b, i, j] = fk_adj[b, rj, ri]
```

Swapped indices read `fk_adj` transposed without materializing a transpose. No self-loop — own-row visibility is
provided by outbound attention, and the residual connection carries representations through for rows with no children.

All three masks are AND'd with `~is_padding` to exclude padding positions.

## Deriving cell-level masks from `fk_adj`

`fk_adj[B, R, R]` is expanded to `[B, S, S]` cell-level masks using `seq_row_ids` as gather indices
(`ri = seq_row_ids[b, i]`, `rj = seq_row_ids[b, j]`). Both outbound and inbound masks derive from the same matrix
(one direct, one transposed), so a single `fk_adj` suffices. Column attention needs no adjacency matrix.

### Concrete example

Using the bookstore schema from [preprocessing.md](../preprocessing.md). BFS from order 1 collects:

| seq_row_id | Row            | Table     |
|------------|----------------|-----------|
| 0          | order 1 (seed) | orders    |
| 1          | customer 23    | customers |
| 2          | book 42        | books     |
| 3          | order 7        | orders    |
| 4          | order 12       | orders    |
| 5          | order 5        | orders    |

FK edges (child → parent): order 1 → {customer 23, book 42}, order 7 → customer 23,
order 12 → customer 23, order 5 → book 42.

```
fk_adj[b]:
     r0  r1  r2  r3  r4  r5
r0: [ .   1   1   .   .   . ]   ← order 1 → customer 23, book 42
r1: [ .   .   .   .   .   . ]   ← customer 23 (no FKs)
r2: [ .   .   .   .   .   . ]   ← book 42 (no FKs)
r3: [ .   1   .   .   .   . ]   ← order 7 → customer 23
r4: [ .   1   .   .   .   . ]   ← order 12 → customer 23
r5: [ .   .   1   .   .   . ]   ← order 5 → book 42
```

**Outbound** (`I + fk_adj`): order 1 sees {order 1, customer 23, book 42}; customer 23 sees {customer 23} only.

**Inbound** (`fk_adj^T`): customer 23 sees {order 1, order 7, order 12}; book 42 sees {order 1, order 5};
order 7 sees ∅ (residual carries its state through).

Note: order 1 does *not* directly see sibling orders 7/12 through inbound attention — that information flows
indirectly: orders 7/12 →(inbound)→ customer 23 →(outbound)→ order 1.

## Block-sparse permutations and forward-pass usage

At each attention layer, the forward pass:

1. **Gather** hidden states into permuted order: `h_perm[b, i, :] = h[b, perm[b, i], :]`

2. **Expand** the attention mask in permuted index space using `seq_row_ids` and `fk_adj`. The full `[B, S, S]` mask
   is never materialized — the block-sparse kernel computes each tile's mask on-the-fly, preserving the ~80×
   compression.

3. **Attend** using block-sparse attention on `h_perm`.

4. **Scatter** back: `h_out[b, perm[b, i], :] = attn_out[b, i, :]`

The inverse permutation is either computed on-device in O(S) or precomputed by the sampler.

See [batch_structure.md](batch_structure.md) Section 4 for permutation tensor definitions (`col_perm`, `out_perm`,
`in_perm`) and sizing.
