"""Model and training configuration."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the RelationalTransformer model."""

    # Core dimensions
    d_model: int = 256          # Model hidden dimension (D)
    d_text: int = 256           # Frozen text embedding dimension (D_t)
    d_ff: int = 768             # FFN hidden dimension (ceil(8/3 * D), rounded to 256)
    n_layers: int = 6           # Number of transformer layers
    n_heads: int = 8            # Number of attention heads
    d_head: int = 32            # Per-head dimension (D / n_heads)
    d_time: int = 15            # Timestamp encoding dimension

    # Sequence dimensions
    max_seq_len: int = 1024     # Maximum sequence length (S)
    max_rows: int = 200         # Maximum rows per sequence (R)

    # Semantic types (must match Rust SemanticType enum)
    num_semantic_types: int = 7  # Identifier=0..Ignored=6

    # Attention
    dropout_rate: float = 0.0   # No dropout in initial version

    # Normalization
    rms_norm_eps: float = 1e-6

    # Number of boolean values for embedding
    num_bool_values: int = 2    # false=0, true=1


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    # Batch dimensions
    batch_size: int = 32
    sequence_length: int = 1024

    # Optimization
    num_steps: int = 100_000
    eval_interval: int = 1000
    num_val_steps: int = 50

    # Learning rates
    muon_lr_peak: float = 0.02
    adamw_lr_peak: float = 3e-4
    warmup_steps: int = 2000

    # AdamW
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_weight_decay: float = 0.1
    adamw_eps: float = 1e-8

    # Muon
    muon_beta1: float = 0.95
    muon_ns_iters: int = 5  # Newton-Schulz iterations

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Z-loss for categorical
    z_loss_weight: float = 1e-4

    # LR schedule: cosine decay to this fraction of peak
    lr_min_ratio: float = 0.1

    # Sampler
    bfs_child_width: int = 16
    num_prefetch: int = 3
    split_ratios: tuple = (0.8, 0.1, 0.1)
    split_seed: int = 123
    seed: int = 42
