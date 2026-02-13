use crate::model::ModelConfig;
use burn::config::Config;

use burn::optim::{AdamW, AdamWConfig, MuonConfig};

#[derive(Config, Debug)]
pub struct OptimizerConfig {
    /// The learning rate to use
    #[config(default = 1e-3)]
    pub learning_rate: f64,

    /// The weight decay to use
    #[config(default = 0.1)]
    pub weight_decay: f64,

    /// Gradient clipping
    #[config(default = 1.0)]
    pub max_grad_norm: f64,

    /// The AdamW optimizer configuration
    /// No default, you must spell this one out.
    pub adamw_config: AdamWConfig,

    /// The Muon optimizer configuration
    /// Only used for 2D parameters - use the AdamW optimizer for 1D params.
    pub muon_config: MuonConfig,
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// The model configuration
    model_config: ModelConfig,

    /// The optimizer configuration
    optimizer_config: OptimizerConfig,

    /// The number of batches to train for, total.
    #[config(default = 1000)]
    num_batches: usize,

    /// The batch size to use (number of sequences to process in parallel)
    #[config(default = 32)]
    batch_size: usize,

    /// Number of cells (tokens) per sequence
    #[config(default = 1024)]
    seq_len: usize,

    /// Evaluation interval - how often should we use the validation set to evaluate?
    #[config(default = 100)]
    eval_interval: usize,

    /// Metric logging interval - how often should we log metrics to our sink?
    #[config(default = 10)]
    log_interval: usize,

    /// Snapshot interval - how often should we save a snapshot of the model?
    /// We can use this to resume training from a checkpoint, or to load a model for inference.
    #[config(default = 100)]
    snapshot_interval: usize,
}
