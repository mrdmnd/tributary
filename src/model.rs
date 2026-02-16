use burn::{
    module::Param,
    nn::{
        Linear, LinearConfig, RmsNorm, RmsNormConfig,
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    },
    prelude::*,
    tensor::activation::silu,
};

#[derive(Config, Debug)]
pub struct ModelConfig {
    /// The hidden dimension of the model (the embedding dimension)
    #[config(default = 256)]
    dim_model: usize,

    /// The dimension of the feedforward network, usually 4x the model dimension
    #[config(default = 1024)]
    dim_ffn: usize,

    /// The dimension of our frozen text embeddings - must match the embedding dimension of the text embedder
    #[config(default = 768)]
    dim_text: usize,

    /// The number of transformer blocks to use
    #[config(default = 12)]
    num_blocks: usize,

    /// The number of attention heads to use per block
    #[config(default = 8)]
    num_heads: usize,

    /// Dropout rate for attention layers
    #[config(default = 0.0)]
    dropout: f64,
}

#[derive(Module, Debug)]
pub struct SwigluFFN<B: Backend> {
    up: Linear<B>,   // [dim_model, dim_ffn] (no bias)
    gate: Linear<B>, // [dim_model, dim_ffn] (no bias)
    down: Linear<B>, // [dim_ffn, dim_model] (no bias)
}

// Author's note: this could maybe be replaced with the built in burn SwiGlu and then the down layer.
// Not sure which is faster / better kernel perf / better fusion.
impl<B: Backend> SwigluFFN<B> {
    pub fn new(device: &B::Device, dim_model: usize, dim_ffn: usize) -> Self {
        let up = LinearConfig::new(dim_model, dim_ffn)
            .with_bias(false)
            .init(device);
        let gate = LinearConfig::new(dim_model, dim_ffn)
            .with_bias(false)
            .init(device);
        let down = LinearConfig::new(dim_ffn, dim_model)
            .with_bias(false)
            .init(device);
        Self { up, gate, down }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let gate = silu(self.gate.forward(x.clone()));
        let up = self.up.forward(x);
        self.down.forward(gate * up)
    }
}

#[derive(Module, Debug)]
pub struct SparseAttentionBlock<B: Backend> {
    mha: MultiHeadAttention<B>,
}

impl<B: Backend> SparseAttentionBlock<B> {
    pub fn new(device: &B::Device, dim_model: usize, num_heads: usize, dropout: f64) -> Self {
        Self {
            mha: MultiHeadAttentionConfig::new(dim_model, num_heads)
                .with_dropout(dropout)
                .init(device),
        }
    }

    // Input x: [B, S, D]
    // Input mask: [B, S, S] (true = masked, i.e. positions to ignore)
    // Output: [B, S, D]
    pub fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 3, Bool>) -> Tensor<B, 3> {
        let mha_input = MhaInput::self_attn(x).mask_attn(mask);
        self.mha.forward(mha_input).context
    }
}

#[derive(Module, Debug)]
pub struct RelationalBlock<B: Backend> {
    column_norm: RmsNorm<B>,
    feature_norm: RmsNorm<B>,
    neighbor_norm: RmsNorm<B>,
    ffn_norm: RmsNorm<B>,

    column_attention: SparseAttentionBlock<B>,
    feature_attention: SparseAttentionBlock<B>,
    neighbor_attention: SparseAttentionBlock<B>,
    ffn: SwigluFFN<B>,
}

impl<B: Backend> RelationalBlock<B> {
    pub fn new(
        device: &B::Device,
        dim_model: usize,
        dim_ffn: usize,
        num_heads: usize,
        dropout: f64,
    ) -> Self {
        Self {
            column_norm: RmsNormConfig::new(dim_model).init(device),
            feature_norm: RmsNormConfig::new(dim_model).init(device),
            neighbor_norm: RmsNormConfig::new(dim_model).init(device),
            ffn_norm: RmsNormConfig::new(dim_model).init(device),

            column_attention: SparseAttentionBlock::new(device, dim_model, num_heads, dropout),
            feature_attention: SparseAttentionBlock::new(device, dim_model, num_heads, dropout),
            neighbor_attention: SparseAttentionBlock::new(device, dim_model, num_heads, dropout),
            ffn: SwigluFFN::new(device, dim_model, dim_ffn),
        }
    }

    /// Pre-norm relational attention block with three sparse attention sub-layers + SwiGLU FFN.
    /// Each sub-layer uses a different mask to attend over columns, features, and neighbors.
    ///
    /// All masks have shape [B, S, S] where true = masked (position ignored in attention).
    ///
    /// Input x: [B, S, D]
    /// Output:  [B, S, D]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        column_mask: Tensor<B, 3, Bool>,
        feature_mask: Tensor<B, 3, Bool>,
        neighbor_mask: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 3> {
        // Column attention with pre-norm + residual
        let x = x.clone()
            + self
                .column_attention
                .forward(self.column_norm.forward(x.clone()), column_mask);

        // Feature attention with pre-norm + residual
        let x = x.clone()
            + self
                .feature_attention
                .forward(self.feature_norm.forward(x.clone()), feature_mask);

        // Neighbor attention with pre-norm + residual
        let x = x.clone()
            + self
                .neighbor_attention
                .forward(self.neighbor_norm.forward(x.clone()), neighbor_mask);

        // SwiGLU FFN with pre-norm + residual
        x.clone() + self.ffn.forward(self.ffn_norm.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    // Initial data encoders
    column_encoder: Linear<B>, // Specifically, column *name* encoder for "foo of bar" inputs
    numerical_encoder: Linear<B>,
    timestamp_encoder: Linear<B>,
    boolean_encoder: Linear<B>,
    categorical_encoder: Linear<B>,
    text_encoder: Linear<B>,

    // Categorical projection layer: learn to separate category embeddings that may be too close in the frozen text
    // encoder space. For example, "status is A" and "status is B" have 0.93 cosine similarity with BGE.
    // This projection is applied to both input and target, so the loss becomes more discriminative as the projection
    // learns to push apart categories.
    category_projection: Linear<B>,

    // Norms
    column_norm: RmsNorm<B>,
    numerical_norm: RmsNorm<B>,
    timestamp_norm: RmsNorm<B>,
    boolean_norm: RmsNorm<B>,
    categorical_norm: RmsNorm<B>,
    text_norm: RmsNorm<B>,

    // Mask embeddings - one per maskable semantic type.
    // Shape is (num_maskable_types, dim_model) where num_maskable_types = 5
    // Index mapping (0-based into this table):
    //   0 = Numerical, 1 = Timestamp, 2 = Boolean, 3 = Categorical, 4 = Text
    // Note: Identifier and Ignored types are never masked, so they have no entry here.
    // Initialized to random normal.
    mask_embeddings: Param<Tensor<B, 2>>,

    // Main relational blocks
    blocks: Vec<RelationalBlock<B>>,

    // Output norm
    out_norm: RmsNorm<B>,

    // Decoder heads
    numerical_decoder: Linear<B>,
    timestamp_decoder: Linear<B>,
    boolean_decoder: Linear<B>,
    categorical_decoder: Linear<B>,
    text_decoder: Linear<B>,
}

impl ModelConfig {
    /// Returns an initialized model with the given configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        // TODO
        unimplemented!()
    }
}
