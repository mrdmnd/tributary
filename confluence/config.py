"""Model and training configuration."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the RelationalTransformer model."""

    # Core dimensions
    d_model: int = 1024  # Model hidden dimension (D)
    d_text: int = 256  # Frozen text embedding dimension (D_t)
    d_ff: int = 4096  # FFN hidden dimension
    n_layers: int = 6  # Number of transformer layers
    n_heads: int = 8  # Number of attention heads (d_head = d_model // n_heads)

    # Sequence dimensions
    max_seq_len: int = 1024  # Maximum sequence length (S)

    # Attention
    dropout_rate: float = 0.0  # No dropout in initial version

    # Normalization
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")

    @property
    def d_head(self) -> int:
        """Per-head dimension (d_model // n_heads)."""
        return self.d_model // self.n_heads


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
    num_prefetch: int = 2
    split_ratios: tuple = (0.8, 0.1, 0.1)
    split_seed: int = 123
    seed: int = 42
