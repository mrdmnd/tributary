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

**Inbound attention** (RT's "neighbor attention"): A cell attends to cells from rows whose FK columns point *into*
the cell's row (P→F direction) — the reverse of outbound. A parent aggregates signals from its children.
No self-loop — own-row visibility is provided by outbound attention, and the residual connection carries
representations through for rows with no children.

All three masks are AND'd with `~is_padding` to exclude padding positions.

## Mask transport

The sampler ships CSR-packed masks in the batch payload:

- `outbound_csr_row_ptr`, `outbound_csr_seq_offsets`, `outbound_csr_col_idx`
- `inbound_csr_row_ptr`, `inbound_csr_seq_offsets`, `inbound_csr_col_idx`
- `column_csr_row_ptr`, `column_csr_seq_offsets`, `column_csr_col_idx`
- `mask_format` is `[1]` with value `1` (CSR)

In model input assembly, these CSR tensors are decoded to dense `[B, S, S]` boolean masks.

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

**Outbound**: order 1 sees {order 1, customer 23, book 42}; customer 23 sees {customer 23} only.

**Inbound**: customer 23 sees {order 1, order 7, order 12}; book 42 sees {order 1, order 5};
order 7 sees ∅ (residual carries its state through).

Note: order 1 does *not* directly see sibling orders 7/12 through inbound attention — that information flows
indirectly: orders 7/12 →(inbound)→ customer 23 →(outbound)→ order 1.

## Forward-pass usage

At each attention layer, the forward pass:

1. **Decode CSR masks** to dense `[B, S, S]` bool masks.
2. **Apply padding mask** (already encoded by sampler and preserved in decode).
3. **Attend** with dense masked attention logits.

See [batch_structure.md](batch_structure.md) Section 4 for tensor definitions.
