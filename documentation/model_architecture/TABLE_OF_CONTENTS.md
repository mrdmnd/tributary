# Model Architecture

Confluence is a relational transformer operating on sequences of database cells.
These documents specify the model architecture, training configuration, and batch format.

## Documents

| Document | Contents |
|----------|----------|
| [batch_structure.md](batch_structure.md) | `TrainingBatch` layout: dimensions, index spaces, all tensor definitions, GPU-resident tables, text embedding remapping |
| [attention.md](attention.md) | Three attention patterns (outbound, inbound, column), FK adjacency matrix, mask derivation, block-sparse permutations |
| [value_encoding.md](value_encoding.md) | Column-name encoder, per-type value encoders, null gating, target masking, combination into h₀ |
| [decoder_heads_and_loss.md](decoder_heads_and_loss.md) | Decoder head inventory, loss computation (null + type-specific), inference output |
| [transformer_blocks.md](transformer_blocks.md) | Pre-norm block structure, SDPA output gating, SwiGLU FFN |
| [training_config.md](training_config.md) | BF16 precision policy, normalization, QK norm, optimizer (Muon + AdamW), initialization, LR schedule, gradient clipping, z-loss |

## Related documents

- [task_framework.md](../task_framework.md) — Task definitions and target selection
- [sampling.md](../sampling.md) — BFS sampling algorithm
- [semantic_types.md](../semantic_types.md) — Semantic type system
- [preprocessing.md](../preprocessing.md) — Database preprocessing and binary format
