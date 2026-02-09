//! Text embedding module.
//!
//! Provides CUDA-accelerated text embeddings using ONNX Runtime with the CUDA
//! execution provider and graph optimizations (fused attention, FP16, CUDA
//! graphs).
//!
//! The [`Embedder`] owns both the tokenizer and the ORT session.  Large batches
//! processed via [`embed_batch_chunked`](Embedder::embed_batch_chunked) benefit
//! from **pipelined tokenization**: a background thread tokenizes chunk N+1
//! while the GPU processes chunk N.

use std::path::PathBuf;

use half::f16;
use hf_hub::{Repo, RepoType, api::sync::Api};
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
/// 512 saturates the GPU well for short sequences with ORT.
pub const DEFAULT_BATCH_SIZE: usize = 512;

/// Default work queue capacity
pub const DEFAULT_QUEUE_CAPACITY: usize = 10_000;

/// Max sequence length
pub const MAX_SEQ_LEN: usize = 512;

/// Default ONNX model path (relative to the project root).
pub const DEFAULT_ONNX_MODEL_PATH: &str = "ort/models/bge-base-en-v1.5-onnx/model_fp16.onnx";

/// Minimum batch size at which we use `encode_batch` (Rayon-parallel) instead
/// of a sequential loop.  Below this threshold the thread-pool overhead is not
/// worth it.
const PARALLEL_TOKENIZE_THRESHOLD: usize = 32;

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
    /// Whether to normalize embeddings (L2 norm)
    pub normalize: bool,
    /// Work queue capacity for background processing
    pub queue_capacity: usize,
    /// Max sequence length
    pub max_seq_len: usize,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            model_repo: DEFAULT_EMBEDDING_REPO.to_string(),
            model_revision: "main".to_string(),
            onnx_model_path: PathBuf::from(DEFAULT_ONNX_MODEL_PATH),
            cuda_device: 0,
            batch_size: DEFAULT_BATCH_SIZE,
            normalize: true,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            max_seq_len: MAX_SEQ_LEN,
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
}

// ============================================================================
// Pre-tokenized batch
// ============================================================================

/// A pre-tokenized batch ready for inference.
///
/// All arrays use `i64` (the ONNX convention for integer tensors).
pub struct TokenizedBatch {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub batch_size: usize,
    pub seq_len: usize,
}

// ============================================================================
// Shared tokenization
// ============================================================================

/// Tokenize a batch of texts into a [`TokenizedBatch`].
///
/// For batches >= [`PARALLEL_TOKENIZE_THRESHOLD`] we delegate to the
/// HuggingFace `encode_batch` method which parallelises across strings via
/// Rayon **and** handles padding internally (via [`PaddingStrategy::BatchLongest`]
/// configured at load time).  Smaller batches use a simple sequential loop to
/// avoid thread-pool overhead, with a fast manual padding pass.
fn tokenize_batch(
    tokenizer: &Tokenizer,
    texts: &[&str],
    max_seq_len: usize,
) -> Result<TokenizedBatch> {
    let batch_size = texts.len();

    if batch_size >= PARALLEL_TOKENIZE_THRESHOLD {
        // ── Fast parallel path ──────────────────────────────────────────
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedderError::TokenizeError(e.to_string()))?;

        let max_len = encodings.first().map_or(0, |e| e.get_ids().len());

        let total = batch_size * max_len;
        let mut flat_ids = Vec::with_capacity(total);
        let mut flat_mask = Vec::with_capacity(total);
        for enc in &encodings {
            flat_ids.extend(enc.get_ids().iter().map(|&x| x as i64));
            flat_mask.extend(enc.get_attention_mask().iter().map(|&x| x as i64));
        }

        Ok(TokenizedBatch {
            input_ids: flat_ids,
            attention_mask: flat_mask,
            batch_size,
            seq_len: max_len,
        })
    } else {
        // ── Sequential path (small batches) ─────────────────────────────
        let mut all_input_ids = Vec::with_capacity(batch_size);
        let mut max_len: usize = 0;

        for text in texts {
            let encoding = tokenizer
                .encode(*text, true)
                .map_err(|e| EmbedderError::TokenizeError(e.to_string()))?;

            let ids: Vec<u32> = encoding.get_ids().to_vec();
            let len = ids.len().min(max_seq_len);
            max_len = max_len.max(len);
            all_input_ids.push(ids);
        }

        let total = batch_size * max_len;
        let mut padded_ids = vec![0i64; total];
        let mut padded_mask = vec![0i64; total];

        for (i, ids) in all_input_ids.iter().enumerate() {
            let len = ids.len().min(max_len);
            let row = i * max_len;
            for j in 0..len {
                padded_ids[row + j] = ids[j] as i64;
                padded_mask[row + j] = 1;
            }
        }

        Ok(TokenizedBatch {
            input_ids: padded_ids,
            attention_mask: padded_mask,
            batch_size,
            seq_len: max_len,
        })
    }
}

// ============================================================================
// ORT Inference
// ============================================================================

/// Run inference on a pre-tokenized batch using the ORT session.
///
/// Returns `[batch_size, EMBEDDING_DIM]` embeddings as `f32`.
fn ort_infer(
    session: &Mutex<Session>,
    input_ids: &[i64],
    attention_mask: &[i64],
    batch_size: usize,
    seq_len: usize,
    normalize: bool,
) -> Result<Vec<Vec<f32>>> {
    // Create ort Tensors from flat i64 slices.
    let input_ids_tensor =
        Tensor::<i64>::from_array(([batch_size, seq_len], input_ids.to_vec()))
            .map_err(|e| EmbedderError::EmbedError(format!("input_ids tensor: {}", e)))?;

    let attention_mask_tensor =
        Tensor::<i64>::from_array(([batch_size, seq_len], attention_mask.to_vec()))
            .map_err(|e| EmbedderError::EmbedError(format!("attention_mask tensor: {}", e)))?;

    let token_type_ids_tensor =
        Tensor::<i64>::from_array(([batch_size, seq_len], vec![0i64; batch_size * seq_len]))
            .map_err(|e| EmbedderError::EmbedError(format!("token_type_ids tensor: {}", e)))?;

    // Run inference (Session::run requires &mut self).
    let mut session = session.lock();
    let outputs = session
        .run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])
        .map_err(|e| EmbedderError::EmbedError(format!("ORT inference: {}", e)))?;

    // The first output is last_hidden_state [B, S, H].
    let (shape, hidden_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| EmbedderError::EmbedError(format!("ORT extract output: {}", e)))?;

    let hidden_dim = if shape.len() == 3 {
        shape[2] as usize
    } else {
        EMBEDDING_DIM
    };

    Ok(mean_pool_and_normalize(
        hidden_data,
        attention_mask,
        batch_size,
        seq_len,
        hidden_dim,
        normalize,
    ))
}

/// Mean pool hidden states and optionally L2-normalize.
///
/// `hidden_states` is a flat `[batch_size * seq_len * hidden_dim]` array in
/// row-major order.  `attention_mask` is `[batch_size * seq_len]`.
fn mean_pool_and_normalize(
    hidden_states: &[f32],
    attention_mask: &[i64],
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    normalize: bool,
) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let mut embedding = vec![0.0f32; hidden_dim];
        let mut count = 0.0f32;

        for s in 0..seq_len {
            let mask_val = attention_mask[b * seq_len + s] as f32;
            if mask_val > 0.0 {
                let offset = (b * seq_len + s) * hidden_dim;
                for d in 0..hidden_dim {
                    embedding[d] += hidden_states[offset + d] * mask_val;
                }
                count += mask_val;
            }
        }

        // Divide by token count.
        let count = count.max(1e-7);
        for val in &mut embedding {
            *val /= count;
        }

        // L2 normalize.
        if normalize {
            let norm: f32 = embedding
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(1e-7);
            for val in &mut embedding {
                *val /= norm;
            }
        }

        result.push(embedding);
    }

    result
}

// ============================================================================
// Embedder (public API)
// ============================================================================

/// Text embedder backed by ONNX Runtime with CUDA acceleration.
///
/// The [`Embedder`] owns the tokenizer and an ORT session.  Large batches
/// processed via [`embed_batch_chunked`](Self::embed_batch_chunked) benefit
/// from **pipelined tokenization**: a background thread tokenizes chunk N+1
/// while the GPU processes chunk N.
pub struct Embedder {
    /// ORT session (wrapped in Mutex because `Session::run` requires `&mut`).
    session: Mutex<Session>,
    /// Shared tokenizer.
    tokenizer: Tokenizer,
    /// Whether to L2-normalize output embeddings.
    normalize: bool,
    /// Configuration (public for external access to batch_size, etc.)
    pub config: EmbedderConfig,
}

impl Embedder {
    /// Create a new embedder with the given configuration.
    ///
    /// This downloads the tokenizer from HuggingFace Hub (if not cached) and
    /// loads the ONNX model into an ORT session with CUDA EP.
    pub fn new(config: EmbedderConfig) -> Result<Self> {
        info!("Initializing embedder with model: {}", config.model_repo);

        // ── Load tokenizer ───────────────────────────────────────────────
        let api = Api::new()?;
        let repo = api.repo(Repo::new(config.model_repo.clone(), RepoType::Model));
        let tokenizer_path = repo.get("tokenizer.json")?;

        info!("Loading tokenizer...");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedderError::InitError(format!("Failed to load tokenizer: {}", e)))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        if let Err(e) = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: config.max_seq_len,
            ..Default::default()
        })) {
            return Err(EmbedderError::InitError(format!(
                "Failed to configure truncation: {}",
                e
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

        let session = Session::builder()
            .map_err(|e| EmbedderError::InitError(format!("ORT session builder: {}", e)))?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(config.cuda_device as i32)
                    .build(),
            ])
            .map_err(|e| EmbedderError::InitError(format!("ORT execution providers: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EmbedderError::InitError(format!("ORT optimization level: {}", e)))?
            .commit_from_file(&config.onnx_model_path)
            .map_err(|e| {
                EmbedderError::InitError(format!(
                    "ORT load model from {:?}: {}",
                    config.onnx_model_path, e
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

        let normalize = config.normalize;
        info!("Embedder ready (normalize={})", normalize);

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            normalize,
            config,
        })
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        EMBEDDING_DIM
    }

    // ========================================================================
    // Single Embedding API
    // ========================================================================

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

    // ========================================================================
    // Batch Embedding API
    // ========================================================================

    /// Embed a batch of text strings, returning f32 embeddings.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let batch = tokenize_batch(&self.tokenizer, texts, self.config.max_seq_len)?;
        ort_infer(
            &self.session,
            &batch.input_ids,
            &batch.attention_mask,
            batch.batch_size,
            batch.seq_len,
            self.normalize,
        )
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

    // ========================================================================
    // Chunked Batch Processing (with pipelined tokenization)
    // ========================================================================

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

        // Build (original_index, text) pairs and sort by byte-length (a cheap
        // but effective proxy for token count).
        let mut indexed: Vec<(usize, &str)> = texts.iter().copied().enumerate().collect();
        indexed.sort_unstable_by_key(|&(_, s)| s.len());

        let sorted_texts: Vec<&str> = indexed.iter().map(|&(_, s)| s).collect();
        let num_chunks = (sorted_texts.len() + chunk_size - 1) / chunk_size;

        // ── Pipelined tokenization ─────────────────────────────────────
        // Clone the tokenizer for the background thread (~2 MB, cheap).
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<TokenizedBatch>>(2);
        let tok = self.tokenizer.clone();
        let max_seq_len = self.config.max_seq_len;

        // Collect chunks as owned Strings so the thread can outlive `texts`.
        let chunks_owned: Vec<Vec<String>> = sorted_texts
            .chunks(chunk_size)
            .map(|c| c.iter().map(|s| s.to_string()).collect())
            .collect();

        let tok_handle = std::thread::spawn(move || {
            for chunk_strings in chunks_owned {
                let refs: Vec<&str> = chunk_strings.iter().map(|s| s.as_str()).collect();
                let result = tokenize_batch(&tok, &refs, max_seq_len);
                if tx.send(result).is_err() {
                    break; // receiver dropped
                }
            }
        });

        // ── Inference on main thread ───────────────────────────────────
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for _ in 0..num_chunks {
            let batch = rx
                .recv()
                .map_err(|_| EmbedderError::EmbedError("Tokenization channel closed".into()))??;
            let embeddings = ort_infer(
                &self.session,
                &batch.input_ids,
                &batch.attention_mask,
                batch.batch_size,
                batch.seq_len,
                self.normalize,
            )?;
            all_embeddings.extend(embeddings);
        }

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
    use std::time::Instant;

    #[test]
    fn test_config_default() {
        let config = EmbedderConfig::default();
        assert_eq!(config.model_repo, DEFAULT_EMBEDDING_REPO);
        assert!(config.normalize);
        assert_eq!(config.cuda_device, 0);
        assert_eq!(
            config.onnx_model_path,
            PathBuf::from(DEFAULT_ONNX_MODEL_PATH)
        );
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
        let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

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
        let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

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
        let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

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
        let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

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
