//! Text embedding module.
//!
//! Provides GPU-accelerated text embeddings using ONNX Runtime with CUDA EP.
//!
//! # Performance optimizations
//!
//! - **FP16 pooling in ONNX** — mean-pooling + L2 normalisation run in FP16,
//!   with a single FP16→FP32 cast only on the reduced `[B, 768]` output.
//!   This avoids materialising the full `[B, S, 768]` tensor in FP32.
//! - **INT32 inputs** — the model accepts INT32 directly (matching the ORT
//!   kernel's internal type), halving host→device input transfer size vs INT64.
//! - **IoBinding with CUDA-pinned I/O** — both input **and** output tensors
//!   are allocated in page-locked host memory and bound via ORT's `IoBinding`
//!   API, enabling truly asynchronous DMA in both directions.
//! - **Pipelined tokenization** — a background thread tokenizes chunk N+1 while
//!   the GPU processes chunk N.
//! - **Optional CUDA graph capture** — eliminates per-kernel launch overhead
//!   for fixed-shape workloads.  Enable with
//!   [`EmbedderConfig::with_cuda_graph`].

use std::path::PathBuf;

use half::f16;
use hf_hub::{Repo, RepoType, api::sync::Api};
use ort::io_binding::IoBinding;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use parking_lot::Mutex;
use thiserror::Error;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};
use tracing::info;

// ============================================================================
// Configuration
// ============================================================================

/// Default embedding model — BGE is a reasonably high-quality off-the-shelf
/// embedding model.
pub const DEFAULT_EMBEDDING_REPO: &str = "BAAI/bge-base-en-v1.5";

/// Default embedding dimension for bge-base-en-v1.5
pub const EMBEDDING_DIM: usize = 768;

/// Default batch size for embedding operations.
/// 2048 keeps the GPU well-saturated on modern GPUs (Blackwell+) for short
/// sequences without excessive padding waste.
pub const DEFAULT_BATCH_SIZE: usize = 2048;

/// Default work queue capacity
pub const DEFAULT_QUEUE_CAPACITY: usize = 10_000;

/// Max sequence length (longest number of tokens we support embedding)
pub const MAX_SEQ_LEN: usize = 512;

/// Default ONNX model path (relative to the project root).
/// Points to the **pooled** model that includes fused mean-pooling + L2 norm.
/// Generate with `uv run ort/scripts/export_onnx.py`.
pub const DEFAULT_ONNX_MODEL_PATH: &str = "ort/models/bge-base-en-v1.5-onnx/model_fp16_pooled.onnx";

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum EmbedderError {
    #[error("Failed to initialize embedding model: {0}")]
    InitError(String),

    #[error("Embedding failed: {0}")]
    EmbedError(String),

    #[error("Tokenization failed: {0}")]
    TokenizeError(String),

    #[error("Work queue error: {0}")]
    QueueError(String),

    #[error("Model not loaded")]
    NotLoaded,

    #[error("HF Hub error: {0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),

    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),
}

pub type Result<T> = std::result::Result<T, EmbedderError>;

// ============================================================================
// Embedder Configuration
// ============================================================================

/// Configuration for the embedder.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Model repository on HuggingFace (used to download the tokenizer).
    pub model_repo: String,
    /// Model revision (branch, tag, or commit)
    pub model_revision: String,
    /// Path to the optimized `.onnx` model file.
    pub onnx_model_path: PathBuf,
    /// CUDA device ordinal
    pub cuda_device: usize,
    /// Batch size for embedding operations
    pub batch_size: usize,
    /// Work queue capacity for background processing
    pub queue_capacity: usize,
    /// Max sequence length
    pub max_seq_len: usize,
    /// Enable CUDA graph capture for reduced kernel-launch overhead.
    ///
    /// When enabled, ORT captures the GPU kernel launch sequence on the first
    /// run and replays it on subsequent runs, eliminating per-kernel launch
    /// overhead.
    ///
    /// **Constraints** (enforced by ORT, not by this crate):
    /// - Input / output **shapes must not change** between runs.
    /// - Input / output **addresses must not change** (requires `IoBinding`).
    /// - The session is **not `Send` or `Sync`** at the ORT level; we still
    ///   wrap in a `Mutex` for Rust safety but you must not move the embedder
    ///   across OS threads when this flag is set.
    /// - Models with control-flow ops (`If`, `Loop`, `Scan`) are unsupported.
    pub enable_cuda_graph: bool,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            model_repo: DEFAULT_EMBEDDING_REPO.to_string(),
            model_revision: "main".to_string(),
            onnx_model_path: PathBuf::from(DEFAULT_ONNX_MODEL_PATH),
            cuda_device: 0,
            batch_size: DEFAULT_BATCH_SIZE,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            max_seq_len: MAX_SEQ_LEN,
            enable_cuda_graph: false,
        }
    }
}

impl EmbedderConfig {
    /// Set a custom model repository
    pub fn with_model(mut self, model_repo: &str) -> Self {
        self.model_repo = model_repo.to_string();
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set CUDA device
    pub fn with_cuda_device(mut self, device: usize) -> Self {
        self.cuda_device = device;
        self
    }

    /// Set the ONNX model path
    pub fn with_onnx_model(mut self, path: impl Into<PathBuf>) -> Self {
        self.onnx_model_path = path.into();
        self
    }

    /// Enable or disable CUDA graph capture.
    ///
    /// See [`EmbedderConfig::enable_cuda_graph`] for constraints.
    pub fn with_cuda_graph(mut self, enable: bool) -> Self {
        self.enable_cuda_graph = enable;
        self
    }
}

// ============================================================================
// Pre-tokenized batch
// ============================================================================

/// A pre-tokenized batch ready for inference.
///
/// Arrays use `i32` — the optimized ONNX model declares INT32 inputs directly
/// (the ORT BERT optimizer's `EmbedLayerNormalization` kernel uses INT32
/// internally, so there is no reason to send INT64 from the host).
pub struct TokenizedBatch {
    pub input_ids: Vec<i32>,
    pub attention_mask: Vec<i32>,
    pub batch_size: usize,
    pub seq_len: usize,
}

// ============================================================================
// Tokenization
// ============================================================================

/// Tokenize a batch of texts into a [`TokenizedBatch`].
///
/// Delegates to the HuggingFace `encode_batch` method which parallelises
/// across strings via Rayon **and** handles padding internally (via
/// [`PaddingStrategy::BatchLongest`] configured at load time).
pub fn tokenize_batch(tokenizer: &Tokenizer, texts: &[&str]) -> Result<TokenizedBatch> {
    let batch_size = texts.len();

    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| EmbedderError::TokenizeError(e.to_string()))?;

    let max_len = encodings.first().map_or(0, |e| e.get_ids().len());

    let total = batch_size * max_len;
    let mut flat_ids = Vec::with_capacity(total);
    let mut flat_mask = Vec::with_capacity(total);
    for enc in &encodings {
        flat_ids.extend(enc.get_ids().iter().map(|&x| x as i32));
        flat_mask.extend(enc.get_attention_mask().iter().map(|&x| x as i32));
    }

    Ok(TokenizedBatch {
        input_ids: flat_ids,
        attention_mask: flat_mask,
        batch_size,
        seq_len: max_len,
    })
}

// ============================================================================
// Inference context (session + pinned allocator)
// ============================================================================

/// Cached pinned input tensors for a specific batch shape.
///
/// Reusing these across calls with the same `(batch_size, seq_len)` avoids
/// repeated `cudaMallocHost` / `cudaFreeHost` round-trips for the three
/// input tensors (input_ids, attention_mask, token_type_ids).
///
/// `token_type_ids` is always zero for single-segment models like BGE, but
/// we keep it as a model input (rather than a graph-computed constant)
/// because ORT's `ConstantOfShape` would allocate a fresh `[B, S]` tensor
/// on every call for dynamic shapes.  A cached pinned tensor + `memset` is
/// much cheaper.
///
/// The `IoBinding` is **not** cached — it is created fresh each call.
/// Caching it caused CUDA cleanup issues (error 700 on `cudaEventDestroy`)
/// because ORT's internal event tracking does not expect bindings to
/// outlive the run that populated them.
struct CachedInputs {
    batch_size: usize,
    seq_len: usize,
    /// Pre-allocated pinned input tensors (data overwritten each call).
    input_ids: Tensor<i32>,
    attn_mask: Tensor<i32>,
    token_type: Tensor<i32>,
}

/// Bundles the ORT session with CUDA-pinned memory allocators.
///
/// Held behind a single [`Mutex`] so that the allocators (which are tied
/// to the session's CUDA EP) are only accessed under the same lock.
///
/// **Drop order matters**: `cached` is declared first so that pinned
/// tensors are freed while the CUDA context (owned by `session`) and the
/// allocators are still alive.  Rust drops struct fields in declaration
/// order.
struct InferenceCtx {
    /// Cached pinned input tensors — MUST be dropped before the allocators
    /// and session (declaration-order drop).
    cached: Option<CachedInputs>,
    /// CUDA-pinned allocator for **input** tensors (`MemoryType::CPUInput`).
    pinned_input_alloc: Allocator,
    /// CUDA-pinned allocator for **output** tensors (`MemoryType::CPUOutput`).
    pinned_output_alloc: Allocator,
    /// The ORT session — dropped last (owns the CUDA context).
    session: Session,
    /// Name of the first model output (e.g. `"embeddings"`).
    output_name: String,
}

// SAFETY: `InferenceCtx` is only ever accessed through a `Mutex`, which
// guarantees exclusive access.  The inner `Allocator` pointer is tied to the
// ORT session's CUDA EP and does not carry thread-affine state beyond what
// the Mutex already serialises.
unsafe impl Send for InferenceCtx {}

// ============================================================================
// ORT Inference
// ============================================================================

/// Run inference on a pre-tokenized batch using **IoBinding** with cached
/// pinned input tensors.
///
/// On the first call (or when `(batch_size, seq_len)` changes), three
/// pinned input tensors are allocated and cached in [`InferenceCtx`].
/// Subsequent calls with the **same shape** reuse the existing tensors —
/// only the data is overwritten via [`Tensor::data_ptr_mut`].  This
/// eliminates per-call `cudaMallocHost` / `cudaFreeHost` round-trips.
///
/// A fresh [`IoBinding`] is created each call (binding creation is cheap;
/// caching it caused CUDA cleanup errors).  All tensors — inputs **and**
/// the output — live in CUDA-pinned (page-locked) host memory, enabling
/// truly asynchronous DMA via `cudaMemcpyAsync` in both directions.
///
/// The model is expected to be **pooled** (output `[B, H]`) with fused
/// mean-pooling and L2 normalisation baked into the ONNX graph.
fn ort_infer(ctx: &mut InferenceCtx, batch: TokenizedBatch) -> Result<Vec<Vec<f32>>> {
    let TokenizedBatch {
        input_ids,
        attention_mask,
        batch_size,
        seq_len,
    } = batch;
    let n = batch_size * seq_len;

    // ── Ensure cached input tensors match the current batch shape ──────
    let need_new = match &ctx.cached {
        Some(c) => c.batch_size != batch_size || c.seq_len != seq_len,
        None => true,
    };

    if need_new {
        // Drop stale cache (frees old pinned input tensors).
        ctx.cached = None;

        // Allocate fresh pinned input tensors.
        let input_ids_tensor =
            Tensor::<i32>::new(&ctx.pinned_input_alloc, [batch_size, seq_len])
                .map_err(|e| EmbedderError::EmbedError(format!("pinned input_ids: {e}")))?;
        let attn_mask_tensor =
            Tensor::<i32>::new(&ctx.pinned_input_alloc, [batch_size, seq_len])
                .map_err(|e| EmbedderError::EmbedError(format!("pinned attn_mask: {e}")))?;
        let token_type_tensor = Tensor::<i32>::new(&ctx.pinned_input_alloc, [batch_size, seq_len])
            .map_err(|e| EmbedderError::EmbedError(format!("pinned token_type: {e}")))?;

        ctx.cached = Some(CachedInputs {
            batch_size,
            seq_len,
            input_ids: input_ids_tensor,
            attn_mask: attn_mask_tensor,
            token_type: token_type_tensor,
        });
    }

    // ── Overwrite cached tensor data ───────────────────────────────────
    // The pinned buffers keep their addresses; only the contents change.
    let cached = ctx.cached.as_mut().unwrap();
    // SAFETY: CUDA_PINNED memory is CPU-accessible (page-locked host RAM).
    unsafe {
        let ptr = cached.input_ids.data_ptr_mut() as *mut i32;
        std::ptr::copy_nonoverlapping(input_ids.as_ptr(), ptr, n);

        let ptr = cached.attn_mask.data_ptr_mut() as *mut i32;
        std::ptr::copy_nonoverlapping(attention_mask.as_ptr(), ptr, n);

        // Token-type IDs — always zero for single-segment models.
        let ptr = cached.token_type.data_ptr_mut() as *mut i32;
        std::ptr::write_bytes(ptr, 0, n);
    }

    // ── Set up IoBinding (fresh each call) ─────────────────────────────
    let mut binding: IoBinding = ctx
        .session
        .create_binding()
        .map_err(|e| EmbedderError::EmbedError(format!("IoBinding create: {e}")))?;

    binding
        .bind_input("input_ids", &cached.input_ids)
        .map_err(|e| EmbedderError::EmbedError(format!("bind input_ids: {e}")))?;
    binding
        .bind_input("attention_mask", &cached.attn_mask)
        .map_err(|e| EmbedderError::EmbedError(format!("bind attention_mask: {e}")))?;
    binding
        .bind_input("token_type_ids", &cached.token_type)
        .map_err(|e| EmbedderError::EmbedError(format!("bind token_type_ids: {e}")))?;

    // Allocate pinned output ([B, H]) and bind.
    let output_tensor = Tensor::<f32>::new(&ctx.pinned_output_alloc, [batch_size, EMBEDDING_DIM])
        .map_err(|e| EmbedderError::EmbedError(format!("pinned output: {e}")))?;
    binding
        .bind_output(&ctx.output_name, output_tensor)
        .map_err(|e| EmbedderError::EmbedError(format!("bind output: {e}")))?;

    // ── Run inference via IoBinding ─────────────────────────────────────
    let outputs = ctx
        .session
        .run_binding(&binding)
        .map_err(|e| EmbedderError::EmbedError(format!("ORT run_binding: {e}")))?;

    // ── Extract results ────────────────────────────────────────────────
    // The output tensor lives in CUDA_PINNED memory.  We downcast from
    // `DynValue` → `TensorValueType<f32>` and use `data_ptr()` to get
    // the raw host pointer (bypassing the `ort` crate's conservative
    // CPU-accessibility check).
    let output_ref = outputs[&*ctx.output_name]
        .downcast_ref::<ort::value::TensorValueType<f32>>()
        .map_err(|e| EmbedderError::EmbedError(format!("downcast output: {e}")))?;

    // Output is already [B, H] (pooled + L2-normalised by the ONNX graph).
    let total = batch_size * EMBEDDING_DIM;
    // SAFETY: CUDA_PINNED memory is CPU-accessible.
    let data: &[f32] = unsafe {
        let ptr = output_ref.data_ptr() as *const f32;
        std::slice::from_raw_parts(ptr, total)
    };
    let embeddings: Vec<Vec<f32>> = data
        .chunks_exact(EMBEDDING_DIM)
        .map(|chunk| chunk.to_vec())
        .collect();
    Ok(embeddings)
}

// ============================================================================
// OrtEmbedder
// ============================================================================

/// Text embedder backed by ONNX Runtime with CUDA acceleration.
///
/// The [`OrtEmbedder`] owns the tokenizer and an ORT session with a
/// CUDA-pinned memory allocator.  Large batches processed via
/// [`embed_batch_chunked`](Self::embed_batch_chunked) benefit from
/// **pipelined tokenization**: a background thread tokenizes chunk N+1 while
/// the GPU processes chunk N.
pub struct OrtEmbedder {
    /// ORT session + pinned allocator (wrapped in Mutex because
    /// `Session::run` requires `&mut self`).
    ctx: Mutex<InferenceCtx>,
    /// Shared tokenizer.
    tokenizer: Tokenizer,
    /// Configuration (public for external access to batch_size, etc.)
    pub config: EmbedderConfig,
}

impl OrtEmbedder {
    /// Create a new embedder with the given configuration.
    ///
    /// This downloads the tokenizer from HuggingFace Hub (if not cached) and
    /// loads the ONNX model into an ORT session with CUDA EP.  A CUDA-pinned
    /// memory allocator is created for fast host→device transfers.
    pub fn new(config: EmbedderConfig) -> Result<Self> {
        info!("Initializing embedder with model: {}", config.model_repo);

        // ── Load tokenizer ───────────────────────────────────────────────
        let api = Api::new()?;
        let repo = api.repo(Repo::new(config.model_repo.clone(), RepoType::Model));
        let tokenizer_path = repo.get("tokenizer.json")?;

        info!("Loading tokenizer...");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedderError::InitError(format!("Failed to load tokenizer: {e}")))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        if let Err(e) = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: config.max_seq_len,
            ..Default::default()
        })) {
            return Err(EmbedderError::InitError(format!(
                "Failed to configure truncation: {e}"
            )));
        }

        // ── Load ORT session ─────────────────────────────────────────────
        info!("Loading ORT model from: {:?}", config.onnx_model_path);

        if !config.onnx_model_path.exists() {
            return Err(EmbedderError::InitError(format!(
                "ONNX model file not found: {:?}. Run `uv run ort/scripts/export_onnx.py` first.",
                config.onnx_model_path
            )));
        }

        // Build CUDA EP — optionally with CUDA-graph capture.
        let mut cuda_ep = ort::execution_providers::CUDAExecutionProvider::default()
            .with_device_id(config.cuda_device as i32);
        if config.enable_cuda_graph {
            info!("CUDA graph capture enabled");
            cuda_ep = cuda_ep.with_cuda_graph(true);
        }

        let session = Session::builder()
            .map_err(|e| EmbedderError::InitError(format!("ORT session builder: {e}")))?
            .with_execution_providers([cuda_ep.build()])
            .map_err(|e| EmbedderError::InitError(format!("ORT execution providers: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EmbedderError::InitError(format!("ORT optimization level: {e}")))?
            .commit_from_file(&config.onnx_model_path)
            .map_err(|e| {
                EmbedderError::InitError(format!(
                    "ORT load model from {:?}: {e}",
                    config.onnx_model_path,
                ))
            })?;

        // Log model input/output info.
        info!("ORT model inputs:");
        for input in session.inputs() {
            info!("  {}", input.name());
        }
        info!("ORT model outputs:");
        for output in session.outputs() {
            info!("  {}", output.name());
        }

        // ── Create CUDA-pinned allocators (input + output) ──────────────
        let pinned_input_alloc = Allocator::new(
            &session,
            MemoryInfo::new(
                AllocationDevice::CUDA_PINNED,
                config.cuda_device as i32,
                AllocatorType::Device,
                MemoryType::CPUInput,
            )
            .map_err(|e| EmbedderError::InitError(format!("pinned input MemoryInfo: {e}")))?,
        )
        .map_err(|e| EmbedderError::InitError(format!("pinned input Allocator: {e}")))?;

        let pinned_output_alloc = Allocator::new(
            &session,
            MemoryInfo::new(
                AllocationDevice::CUDA_PINNED,
                config.cuda_device as i32,
                AllocatorType::Device,
                MemoryType::CPUOutput,
            )
            .map_err(|e| EmbedderError::InitError(format!("pinned output MemoryInfo: {e}")))?,
        )
        .map_err(|e| EmbedderError::InitError(format!("pinned output Allocator: {e}")))?;
        info!("CUDA-pinned allocators ready (input + output)");

        // ── Verify the model is pooled ──────────────────────────────────
        // A pooled model outputs [B, H] (2 dims).  We require this — unpooled
        // models (output [B, S, H]) are not supported.
        let output_info = session.outputs();
        let first_output = output_info
            .first()
            .ok_or_else(|| EmbedderError::InitError("Model has no outputs".into()))?;
        let output_name = first_output.name().to_string();

        match first_output.dtype() {
            ort::value::ValueType::Tensor { ty: _, shape, .. } => {
                info!(
                    "Output '{}' shape: {:?} ({} dims)",
                    output_name,
                    shape,
                    shape.len()
                );
                if shape.len() != 2 {
                    return Err(EmbedderError::InitError(format!(
                        "Expected a pooled model with 2-dim output [B, H], but got {} dims. \
                         Re-export with `uv run ort/scripts/export_onnx.py` (without --skip-fuse-pooling).",
                        shape.len()
                    )));
                }
            }
            other => {
                return Err(EmbedderError::InitError(format!(
                    "Unexpected output type: {:?}",
                    other
                )));
            }
        };

        info!("Embedder ready (cuda_graph={})", config.enable_cuda_graph);

        Ok(Self {
            ctx: Mutex::new(InferenceCtx {
                cached: None,
                pinned_input_alloc,
                pinned_output_alloc,
                session,
                output_name,
            }),
            tokenizer,
            config,
        })
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        EMBEDDING_DIM
    }

    // ====================================================================
    // Single Embedding API
    // ====================================================================

    /// Embed a single text string, returning f32 embeddings.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::EmbedError("No embedding returned".to_string()))
    }

    /// Embed a single text string, returning f16 embeddings (for storage).
    pub fn embed_one_f16(&self, text: &str) -> Result<Vec<f16>> {
        self.embed_one(text)
            .map(|v| v.into_iter().map(f16::from_f32).collect())
    }

    // ====================================================================
    // Batch Embedding API
    // ====================================================================

    /// Embed a batch of text strings, returning f32 embeddings.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let batch = tokenize_batch(&self.tokenizer, texts)?;
        let mut ctx = self.ctx.lock();
        ort_infer(&mut ctx, batch)
    }

    /// Embed a batch of text strings, returning f16 embeddings (for storage).
    pub fn embed_batch_f16(&self, texts: &[&str]) -> Result<Vec<Vec<f16>>> {
        self.embed_batch(texts).map(|batch| {
            batch
                .into_iter()
                .map(|v| v.into_iter().map(f16::from_f32).collect())
                .collect()
        })
    }

    /// Embed owned strings in batch.
    pub fn embed_batch_owned(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.embed_batch(&refs)
    }

    /// Embed owned strings in batch, returning f16.
    pub fn embed_batch_owned_f16(&self, texts: &[String]) -> Result<Vec<Vec<f16>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.embed_batch_f16(&refs)
    }

    // ====================================================================
    // Chunked Batch Processing (with pipelined tokenization)
    // ====================================================================

    /// Process a large batch in chunks with pipelined tokenization.
    ///
    /// Strings are **sorted by byte-length** before chunking so that each GPU
    /// batch contains strings of similar token counts.  This minimises padding
    /// within each chunk and can dramatically reduce wasted compute when the
    /// input corpus has heterogeneous lengths.  Results are returned in the
    /// original input order.
    ///
    /// Tokenization of chunk N+1 runs on a **background thread** while the GPU
    /// processes chunk N, hiding most of the CPU tokenization latency.
    pub fn embed_batch_chunked(&self, texts: &[&str], chunk_size: usize) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut indexed: Vec<(usize, &str)> = texts.iter().copied().enumerate().collect();
        indexed.sort_unstable_by_key(|&(_, s)| s.len());

        let sorted_texts: Vec<&str> = indexed.iter().map(|&(_, s)| s).collect();
        let num_chunks = (sorted_texts.len() + chunk_size - 1) / chunk_size;

        // ── Pipelined tokenization ─────────────────────────────────────
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<TokenizedBatch>>(2);
        let tok = self.tokenizer.clone();

        let chunks_owned: Vec<Vec<String>> = sorted_texts
            .chunks(chunk_size)
            .map(|c| c.iter().map(|s| s.to_string()).collect())
            .collect();

        let tok_handle = std::thread::spawn(move || {
            for chunk_strings in chunks_owned {
                let refs: Vec<&str> = chunk_strings.iter().map(|s| s.as_str()).collect();
                let result = tokenize_batch(&tok, &refs);
                if tx.send(result).is_err() {
                    break;
                }
            }
        });

        // ── Inference on main thread ───────────────────────────────────
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        {
            let mut ctx = self.ctx.lock();
            for _ in 0..num_chunks {
                let batch = rx.recv().map_err(|_| {
                    EmbedderError::EmbedError("Tokenization channel closed".into())
                })??;
                let embeddings = ort_infer(&mut ctx, batch)?;
                all_embeddings.extend(embeddings);
            }
        } // Mutex released before joining the tokenization thread.

        tok_handle
            .join()
            .map_err(|_| EmbedderError::EmbedError("Tokenization thread panicked".into()))?;

        // ── Scatter back to original order ─────────────────────────────
        let mut result: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        result.resize_with(texts.len(), Vec::new);
        for (sorted_idx, embedding) in all_embeddings.into_iter().enumerate() {
            let original_idx = indexed[sorted_idx].0;
            result[original_idx] = embedding;
        }

        Ok(result)
    }

    /// Process a large batch in chunks, returning f16.
    pub fn embed_batch_chunked_f16(
        &self,
        texts: &[&str],
        chunk_size: usize,
    ) -> Result<Vec<Vec<f16>>> {
        self.embed_batch_chunked(texts, chunk_size).map(|batch| {
            batch
                .into_iter()
                .map(|v| v.into_iter().map(f16::from_f32).collect())
                .collect()
        })
    }
}

/// Backwards-compatible type alias.
pub type Embedder = OrtEmbedder;

// ============================================================================
// Shared utilities
// ============================================================================

/// Create a placeholder embedding (zeros) for lazy initialization.
pub fn placeholder_embedding(dim: usize) -> Vec<f16> {
    vec![f16::ZERO; dim]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = EmbedderConfig::default();
        assert_eq!(config.model_repo, DEFAULT_EMBEDDING_REPO);
        assert_eq!(config.cuda_device, 0);
        assert_eq!(
            config.onnx_model_path,
            PathBuf::from(DEFAULT_ONNX_MODEL_PATH)
        );
        assert!(!config.enable_cuda_graph);
    }

    #[test]
    fn test_placeholder_embedding() {
        let placeholder = placeholder_embedding(1024);
        assert_eq!(placeholder.len(), 1024);
        assert!(placeholder.iter().all(|&v| v == f16::ZERO));
    }

    // ========================================================================
    // Integration Tests (require model download + CUDA + ONNX model)
    // Run with: cargo test --release -- --ignored
    // ========================================================================

    mod ort_tests {
        use super::*;
        use std::time::Instant;

        /// Generate sample strings for testing
        fn generate_test_strings(count: usize) -> Vec<String> {
            let sample_texts = [
                "user_id of customers",
                "order_date of orders",
                "product_name of products",
                "total_amount of transactions",
                "email_address of users",
                "created_at of sessions",
                "category_id of items",
                "description of products",
                "first_name of employees",
                "last_name of employees",
                "phone_number of contacts",
                "street_address of addresses",
                "city of locations",
                "country_code of regions",
                "price of line_items",
                "quantity of inventory",
                "status of orders",
                "rating of reviews",
                "comment_text of comments",
                "timestamp of events",
            ];

            (0..count)
                .map(|i| {
                    let base = &sample_texts[i % sample_texts.len()];
                    if i < sample_texts.len() {
                        base.to_string()
                    } else {
                        format!("{} variant_{}", base, i)
                    }
                })
                .collect()
        }

        #[test]
        #[ignore] // Run with: cargo test --release -- --ignored test_embed_single
        fn test_embed_single() {
            println!("Initializing embedder...");
            let embedder =
                Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

            println!("Embedding single text...");
            let text = "user_id of customers";
            let embedding = embedder.embed_one(text).expect("Failed to embed");

            println!("Embedding dimension: {}", embedding.len());
            assert!(!embedding.is_empty(), "Embedding should not be empty");

            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!("L2 norm: {}", norm);
            assert!(
                (norm - 1.0).abs() < 0.01,
                "Embedding should be normalized, got norm: {}",
                norm
            );
        }

        #[test]
        #[ignore] // Run with: cargo test --release -- --ignored test_embed_batch_small
        fn test_embed_batch_small() {
            println!("Initializing embedder...");
            let embedder =
                Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

            let texts = vec![
                "user_id of customers",
                "order_date of orders",
                "product_name of products",
            ];

            println!("Embedding {} texts...", texts.len());
            let start = Instant::now();
            let embeddings = embedder.embed_batch(&texts).expect("Failed to embed batch");
            let elapsed = start.elapsed();

            println!(
                "Embedded {} texts in {:?} ({:.2} texts/sec)",
                texts.len(),
                elapsed,
                texts.len() as f64 / elapsed.as_secs_f64()
            );

            assert_eq!(embeddings.len(), texts.len());
            for (i, embedding) in embeddings.iter().enumerate() {
                assert!(!embedding.is_empty(), "Embedding {} should not be empty", i);
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    (norm - 1.0).abs() < 0.01,
                    "Embedding {} should be normalized, got norm: {}",
                    i,
                    norm
                );
            }
        }

        #[test]
        #[ignore]
        fn test_embed_1000_strings() {
            println!("Initializing embedder...");
            let embedder =
                Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

            let strings = generate_test_strings(1000);
            let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

            println!(
                "Embedding 1000 strings in batches of {}...",
                DEFAULT_BATCH_SIZE
            );
            let start = Instant::now();
            let embeddings = embedder
                .embed_batch_chunked(&refs, DEFAULT_BATCH_SIZE)
                .expect("Failed to embed 1000 strings");
            let elapsed = start.elapsed();

            println!(
                "Embedded 1000 strings in {:?} ({:.2} strings/sec)",
                elapsed,
                1000.0 / elapsed.as_secs_f64()
            );

            assert_eq!(embeddings.len(), 1000);
            for (i, embedding) in embeddings.iter().enumerate() {
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    (norm - 1.0).abs() < 0.01,
                    "Embedding {} should be normalized, got norm: {}",
                    i,
                    norm
                );
            }
        }

        #[test]
        #[ignore]
        fn test_similarity_sanity() {
            println!("Initializing embedder...");
            let embedder =
                Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

            let texts = vec![
                "user_id of customers",
                "customer_id of users",
                "price of products",
                "cost of items",
                "completely unrelated text",
            ];

            let embeddings = embedder.embed_batch(&texts).expect("Failed to embed");

            let cosine_sim =
                |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

            let sim_0_1 = cosine_sim(&embeddings[0], &embeddings[1]);
            let sim_0_4 = cosine_sim(&embeddings[0], &embeddings[4]);

            assert!(
                sim_0_1 > sim_0_4,
                "Similar texts should have higher similarity: {} vs {}",
                sim_0_1,
                sim_0_4
            );
        }
    }
}
