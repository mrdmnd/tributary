//! Text embedding module.
//!
//! Provides CUDA-accelerated text embeddings with:
//! - Batch embedding API for efficient processing
//! - Work queue for background embedding of discovered text values
//! - Integration with the Database types for column/table embeddings

use std::sync::Arc;

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use half::f16;
use hf_hub::{Repo, RepoType, api::sync::Api};
use parking_lot::RwLock;
use thiserror::Error;
use tokenizers::Tokenizer;
use tracing::info;

// ============================================================================
// Configuration
// ============================================================================

/// Default embedding model - BGE is a reasonably high-quality off-the-shelf embedding model
pub const DEFAULT_EMBEDDING_REPO: &str = "BAAI/bge-base-en-v1.5";

/// Default embedding dimension for bge-base-en-v1.5
pub const EMBEDDING_DIM: usize = 768;

/// Default batch size for embedding operations
pub const DEFAULT_BATCH_SIZE: usize = 256;

/// Default work queue capacity
pub const DEFAULT_QUEUE_CAPACITY: usize = 10_000;

/// Max sequence length
pub const MAX_SEQ_LEN: usize = 512;

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

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("HF Hub error: {0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),
}

pub type Result<T> = std::result::Result<T, EmbedderError>;

// ============================================================================
// Embedder Configuration
// ============================================================================

/// Configuration for the embedder
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Model repository on HuggingFace
    pub model_repo: String,
    /// Model revision (branch, tag, or commit)
    pub model_revision: String,
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
}

// ============================================================================
// Embedding Model
// ============================================================================

/// The loaded embedding model and tokenizer
struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    config: EmbedderConfig,
}

impl EmbeddingModel {
    /// Load the model from HuggingFace Hub
    fn load(config: &EmbedderConfig) -> Result<Self> {
        info!("Loading model from: {}", config.model_repo);

        // Set up device
        let device = Device::new_cuda(config.cuda_device)?;
        info!("Using CUDA device: {:?}", device);

        // Download model files from HuggingFace
        let api = Api::new()?;
        let repo = api.repo(Repo::new(config.model_repo.clone(), RepoType::Model));

        info!("Downloading model files...");
        let tokenizer_path = repo.get("tokenizer.json")?;
        let config_path = repo.get("config.json")?;

        // Try model.safetensors first, fall back to pytorch_model.bin via safetensors
        let weights_path = repo.get("model.safetensors").or_else(|_| {
            info!("model.safetensors not found, trying pytorch_model.bin...");
            repo.get("pytorch_model.bin")
        })?;

        // Load tokenizer
        info!("Loading tokenizer...");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedderError::InitError(format!("Failed to load tokenizer: {}", e)))?;

        // Load model config
        info!("Loading model config...");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| EmbedderError::InitError(format!("Failed to read config: {}", e)))?;
        let model_config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| EmbedderError::InitError(format!("Failed to parse config: {}", e)))?;

        // Load model weights
        info!("Loading model weights from {:?}...", weights_path);
        let vb = if weights_path
            .extension()
            .map(|e| e == "safetensors")
            .unwrap_or(false)
        {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? }
        } else {
            VarBuilder::from_pth(&weights_path, DTYPE, &device)?
        };

        // Build model
        info!("Building model...");
        let model = BertModel::load(vb, &model_config)?;

        info!("Model loaded successfully!");
        Ok(Self {
            model,
            tokenizer,
            device,
            config: config.clone(),
        })
    }

    /// Minimum batch size at which we use `encode_batch` (Rayon-parallel) instead
    /// of a sequential loop.  Below this threshold the thread-pool overhead is not
    /// worth it.
    const PARALLEL_TOKENIZE_THRESHOLD: usize = 32;

    /// Tokenize a batch of texts synchronously.
    ///
    /// For batches >= [`Self::PARALLEL_TOKENIZE_THRESHOLD`] we delegate to the
    /// HuggingFace `encode_batch` method which parallelises across strings via
    /// Rayon.  Smaller batches use a simple sequential loop to avoid thread-pool
    /// overhead.
    fn tokenize(&self, texts: &[&str]) -> Result<(Tensor, Tensor, Tensor)> {
        let mut all_input_ids = Vec::with_capacity(texts.len());
        let mut all_attention_mask = Vec::with_capacity(texts.len());
        let mut max_len = 0;

        if texts.len() >= Self::PARALLEL_TOKENIZE_THRESHOLD {
            // Parallel path: encode_batch uses Rayon internally.
            let encodings = self
                .tokenizer
                .encode_batch(texts.to_vec(), true)
                .map_err(|e| EmbedderError::TokenizeError(e.to_string()))?;

            for encoding in &encodings {
                let ids: Vec<u32> = encoding.get_ids().to_vec();
                let len = ids.len().min(self.config.max_seq_len);
                max_len = max_len.max(len);
                all_attention_mask.push(vec![1u32; len]);
                all_input_ids.push(ids);
            }
        } else {
            // Sequential path: avoid Rayon thread-pool overhead for small batches.
            for text in texts {
                let encoding = self
                    .tokenizer
                    .encode(*text, true)
                    .map_err(|e| EmbedderError::TokenizeError(e.to_string()))?;

                let ids: Vec<u32> = encoding.get_ids().to_vec();
                let len = ids.len().min(self.config.max_seq_len);
                max_len = max_len.max(len);
                all_attention_mask.push(vec![1u32; len]);
                all_input_ids.push(ids);
            }
        }

        // Pad to max length
        let batch_size = texts.len();
        let mut padded_ids = vec![0u32; batch_size * max_len];
        let mut padded_mask = vec![0u32; batch_size * max_len];
        let padded_token_types = vec![0u32; batch_size * max_len]; // All zeros for single-sentence

        for (i, (ids, mask)) in all_input_ids
            .iter()
            .zip(all_attention_mask.iter())
            .enumerate()
        {
            let len = ids.len().min(max_len);
            for j in 0..len {
                padded_ids[i * max_len + j] = ids[j];
                padded_mask[i * max_len + j] = mask[j];
            }
        }

        // Create tensors
        let input_ids = Tensor::from_vec(padded_ids, (batch_size, max_len), &self.device)?;
        let token_type_ids =
            Tensor::from_vec(padded_token_types, (batch_size, max_len), &self.device)?;
        let attention_mask = Tensor::from_vec(padded_mask, (batch_size, max_len), &self.device)?;

        Ok((input_ids, token_type_ids, attention_mask))
    }

    /// Compute embeddings for a batch of texts
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize
        let (input_ids, token_type_ids, attention_mask) = self.tokenize(texts)?;

        // Forward pass through BERT model
        let hidden_states =
            self.model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pooling over sequence length (considering attention mask)
        let embeddings = self.mean_pool(&hidden_states, &attention_mask)?;

        // Normalize if configured
        let embeddings = if self.config.normalize {
            self.l2_normalize(&embeddings)?
        } else {
            embeddings
        };

        // Convert to Vec<Vec<f32>>
        let embeddings = embeddings.to_dtype(DType::F32)?.to_vec2()?;

        Ok(embeddings)
    }

    /// Mean pooling over sequence length
    fn mean_pool(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        // hidden_states: [batch, seq_len, hidden_dim]
        // attention_mask: [batch, seq_len]

        // Expand mask to hidden dim
        let mask = attention_mask
            .to_dtype(hidden_states.dtype())?
            .unsqueeze(2)?; // [batch, seq_len, 1]

        // Apply mask and sum
        let masked = hidden_states.broadcast_mul(&mask)?;
        let sum = masked.sum(1)?; // [batch, hidden_dim]

        // Divide by sum of mask (number of valid tokens)
        let count = mask.sum(1)?; // [batch, 1]
        let count = count.broadcast_add(&Tensor::new(&[1e-9f32], &self.device)?)?; // Avoid div by zero

        sum.broadcast_div(&count)
    }

    /// L2 normalize embeddings
    fn l2_normalize(&self, embeddings: &Tensor) -> CandleResult<Tensor> {
        let norm = embeddings.sqr()?.sum_keepdim(1)?.sqrt()?;
        let norm = norm.broadcast_add(&Tensor::new(&[1e-9f32], &self.device)?)?;
        embeddings.broadcast_div(&norm)
    }
}

// ============================================================================
// Embedder
// ============================================================================

/// Text embedder using BERT with CUDA support
pub struct Embedder {
    /// The loaded model
    inner: Arc<RwLock<Option<EmbeddingModel>>>,
    /// Configuration (public for external access to batch_size, etc.)
    pub config: EmbedderConfig,
}

impl Embedder {
    /// Create a new embedder with the given configuration
    pub fn new(config: EmbedderConfig) -> Result<Self> {
        info!("Initializing embedder with model: {}", config.model_repo);

        let model = EmbeddingModel::load(&config)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(Some(model))),
            config,
        })
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        EMBEDDING_DIM
    }

    // ========================================================================
    // Single Embedding API
    // ========================================================================

    /// Embed a single text string, returning f32 embeddings
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::EmbedError("No embedding returned".to_string()))
    }

    /// Embed a single text string, returning f16 embeddings (for storage)
    pub fn embed_one_f16(&self, text: &str) -> Result<Vec<f16>> {
        self.embed_one(text)
            .map(|v| v.into_iter().map(f16::from_f32).collect())
    }

    // ========================================================================
    // Batch Embedding API
    // ========================================================================

    /// Embed a batch of text strings, returning f32 embeddings
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut guard = self.inner.write();
        let model = guard.as_mut().ok_or(EmbedderError::NotLoaded)?;
        model.embed_batch(texts)
    }

    /// Embed a batch of text strings, returning f16 embeddings (for storage)
    pub fn embed_batch_f16(&self, texts: &[&str]) -> Result<Vec<Vec<f16>>> {
        self.embed_batch(texts).map(|batch| {
            batch
                .into_iter()
                .map(|v| v.into_iter().map(f16::from_f32).collect())
                .collect()
        })
    }

    /// Embed owned strings in batch
    pub fn embed_batch_owned(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.embed_batch(&refs)
    }

    /// Embed owned strings in batch, returning f16
    pub fn embed_batch_owned_f16(&self, texts: &[String]) -> Result<Vec<Vec<f16>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.embed_batch_f16(&refs)
    }

    // ========================================================================
    // Chunked Batch Processing
    // ========================================================================

    /// Process a large batch in chunks
    pub fn embed_batch_chunked(&self, texts: &[&str], chunk_size: usize) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(chunk_size) {
            let chunk_embeddings = self.embed_batch(chunk)?;
            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
    }

    /// Process a large batch in chunks, returning f16
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

    /// Unload the model to free memory
    pub fn unload(&self) {
        let mut guard = self.inner.write();
        *guard = None;
        info!("Model unloaded");
    }
}

/// Create a placeholder embedding (zeros) for lazy initialization
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
    }

    #[test]
    fn test_placeholder_embedding() {
        let placeholder = placeholder_embedding(1024);
        assert_eq!(placeholder.len(), 1024);
        assert!(placeholder.iter().all(|&v| v == f16::ZERO));
    }

    // ========================================================================
    // Integration Tests (require model download + CUDA/CPU)
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

        // Check that embedding is normalized (L2 norm ≈ 1.0)
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
    #[ignore] // Run with: cargo test --release -- --ignored test_embed_1000_strings
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

        assert_eq!(embeddings.len(), 1000, "Should have 1000 embeddings");

        // Verify all embeddings are valid
        let embedding_dim = embeddings[0].len();
        println!("Embedding dimension: {}", embedding_dim);

        for (i, embedding) in embeddings.iter().enumerate() {
            assert_eq!(
                embedding.len(),
                embedding_dim,
                "Embedding {} has wrong dimension",
                i
            );

            // Check normalization
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "Embedding {} should be normalized, got norm: {}",
                i,
                norm
            );
        }

        // Check that different texts produce different embeddings
        let first = &embeddings[0];
        let second = &embeddings[1];
        let dot_product: f32 = first.iter().zip(second.iter()).map(|(a, b)| a * b).sum();
        println!(
            "Cosine similarity between first two embeddings: {}",
            dot_product
        );
        assert!(
            dot_product < 0.9999,
            "Different texts should produce different embeddings"
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored test_embed_1000_strings_f16
    fn test_embed_1000_strings_f16() {
        println!("Initializing embedder...");
        let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

        let strings = generate_test_strings(1000);
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

        println!("Embedding 1000 strings to f16...");
        let start = Instant::now();
        let embeddings = embedder
            .embed_batch_chunked_f16(&refs, DEFAULT_BATCH_SIZE)
            .expect("Failed to embed 1000 strings to f16");
        let elapsed = start.elapsed();

        println!(
            "Embedded 1000 strings to f16 in {:?} ({:.2} strings/sec)",
            elapsed,
            1000.0 / elapsed.as_secs_f64()
        );

        assert_eq!(embeddings.len(), 1000, "Should have 1000 embeddings");

        // Verify f16 embeddings
        for (i, embedding) in embeddings.iter().enumerate() {
            assert!(!embedding.is_empty(), "Embedding {} should not be empty", i);

            // Convert to f32 and check normalization
            let f32_embedding: Vec<f32> = embedding.iter().map(|x| x.to_f32()).collect();
            let norm: f32 = f32_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.02, // Slightly looser tolerance for f16
                "Embedding {} should be normalized, got norm: {}",
                i,
                norm
            );
        }

        // Calculate memory usage
        let bytes_per_embedding = embeddings[0].len() * std::mem::size_of::<f16>();
        let total_bytes = bytes_per_embedding * embeddings.len();
        println!(
            "Memory usage: {} bytes per embedding, {} KB total for 1000 embeddings",
            bytes_per_embedding,
            total_bytes / 1024
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored test_embed_varying_lengths
    fn test_embed_varying_lengths() {
        println!("Initializing embedder...");
        let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

        // Generate strings of varying lengths
        let strings: Vec<String> = (0..100)
            .map(|i| {
                let base = "word ".repeat(1 + (i % 50));
                format!("Text {}: {}", i, base.trim())
            })
            .collect();
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

        println!("Embedding 100 strings of varying lengths...");
        let start = Instant::now();
        let embeddings = embedder
            .embed_batch_chunked(&refs, 16)
            .expect("Failed to embed varying length strings");
        let elapsed = start.elapsed();

        println!("Embedded 100 varying-length strings in {:?}", elapsed);

        assert_eq!(embeddings.len(), 100);

        // All embeddings should have the same dimension regardless of input length
        let dim = embeddings[0].len();
        for (i, embedding) in embeddings.iter().enumerate() {
            assert_eq!(
                embedding.len(),
                dim,
                "All embeddings should have same dimension, embedding {} has {}",
                i,
                embedding.len()
            );
        }
    }

    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored test_embed_special_characters
    fn test_embed_special_characters() {
        println!("Initializing embedder...");
        let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

        let texts = vec![
            "normal text",
            "text with numbers 12345",
            "text with symbols !@#$%^&*()",
            "text with émojis 🎉🚀",
            "text with unicode: 日本語 中文 العربية",
            "text\nwith\nnewlines",
            "text\twith\ttabs",
            "",  // Empty string
            " ", // Just whitespace
            "a", // Single character
        ];

        println!("Embedding {} special texts...", texts.len());
        let embeddings = embedder
            .embed_batch(&texts)
            .expect("Failed to embed special texts");

        assert_eq!(embeddings.len(), texts.len());
        println!("All {} special texts embedded successfully", texts.len());

        for (i, (text, embedding)) in texts.iter().zip(embeddings.iter()).enumerate() {
            println!(
                "  [{}] '{}' -> {} dims",
                i,
                text.chars().take(30).collect::<String>(),
                embedding.len()
            );
            assert!(
                !embedding.is_empty(),
                "Embedding for '{}' should not be empty",
                text
            );
        }
    }

    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored test_similarity_sanity
    fn test_similarity_sanity() {
        println!("Initializing embedder...");
        let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

        let texts = vec![
            "user_id of customers",      // Similar to index 1
            "customer_id of users",      // Similar to index 0
            "price of products",         // Different topic
            "cost of items",             // Similar to index 2
            "completely unrelated text", // Different
        ];

        let embeddings = embedder.embed_batch(&texts).expect("Failed to embed");

        // Helper to compute cosine similarity
        let cosine_sim =
            |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

        println!("\nCosine similarity matrix:");
        for i in 0..texts.len() {
            for j in 0..texts.len() {
                let sim = cosine_sim(&embeddings[i], &embeddings[j]);
                print!("{:.3} ", sim);
            }
            println!(" <- '{}'", texts[i]);
        }

        // Sanity checks
        let sim_0_1 = cosine_sim(&embeddings[0], &embeddings[1]);
        let sim_2_3 = cosine_sim(&embeddings[2], &embeddings[3]);
        let sim_0_4 = cosine_sim(&embeddings[0], &embeddings[4]);

        println!("\nSimilarity checks:");
        println!("  '{}' vs '{}' = {:.3}", texts[0], texts[1], sim_0_1);
        println!("  '{}' vs '{}' = {:.3}", texts[2], texts[3], sim_2_3);
        println!("  '{}' vs '{}' = {:.3}", texts[0], texts[4], sim_0_4);

        // Similar texts should have higher similarity than unrelated ones
        assert!(
            sim_0_1 > sim_0_4,
            "Similar texts should have higher similarity: {} vs {}",
            sim_0_1,
            sim_0_4
        );
        assert!(
            sim_2_3 > sim_0_4,
            "Similar texts should have higher similarity: {} vs {}",
            sim_2_3,
            sim_0_4
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored test_benchmark_batch_sizes --nocapture
    fn test_benchmark_batch_sizes() {
        println!("Initializing embedder...");
        let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

        // Generate 10,000 test strings
        let num_strings = 10_000;
        let strings = generate_test_strings(num_strings);
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

        println!("\n=== Batch Size Benchmark ({} strings) ===\n", num_strings);
        println!(
            "{:>12} {:>12} {:>12} {:>12}",
            "Batch Size", "Time (s)", "Strings/sec", "Speedup"
        );
        println!("{}", "-".repeat(52));

        let batch_sizes = [8, 16, 32, 64, 128, 256, 512];
        let mut baseline_throughput = 0.0;

        for &batch_size in &batch_sizes {
            // Warm-up run
            let _ = embedder.embed_batch_chunked(&refs[..batch_size.min(100)], batch_size);

            // Timed run
            let start = Instant::now();
            let embeddings = embedder
                .embed_batch_chunked(&refs, batch_size)
                .expect("Failed to embed");
            let elapsed = start.elapsed();

            let throughput = num_strings as f64 / elapsed.as_secs_f64();
            let speedup = if baseline_throughput > 0.0 {
                throughput / baseline_throughput
            } else {
                baseline_throughput = throughput;
                1.0
            };

            println!(
                "{:>12} {:>12.3} {:>12.1} {:>12.2}x",
                batch_size,
                elapsed.as_secs_f64(),
                throughput,
                speedup
            );

            assert_eq!(embeddings.len(), num_strings);
        }

        println!("\n=== Memory & Dimension Info ===");
        let sample_embedding = embedder.embed_one("test").unwrap();
        println!("Embedding dimension: {}", sample_embedding.len());
        println!(
            "Memory per embedding (f32): {} bytes",
            sample_embedding.len() * 4
        );
        println!(
            "Memory per embedding (f16): {} bytes",
            sample_embedding.len() * 2
        );
        println!(
            "Memory for {} embeddings (f16): {:.2} MB",
            num_strings,
            (num_strings * sample_embedding.len() * 2) as f64 / 1024.0 / 1024.0
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored test_sustained_throughput --nocapture
    fn test_sustained_throughput() {
        println!("Initializing embedder...");
        let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");

        let batch_size = 128; // Use larger batch for sustained test
        let num_batches = 100;
        let strings_per_batch = batch_size;
        let total_strings = num_batches * strings_per_batch;

        let strings = generate_test_strings(strings_per_batch);
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

        println!("\n=== Sustained Throughput Test ===");
        println!("Batch size: {}", batch_size);
        println!("Number of batches: {}", num_batches);
        println!("Total strings: {}", total_strings);
        println!();

        // Warm-up
        let _ = embedder.embed_batch(&refs);

        let start = Instant::now();
        let mut total_embedded = 0;

        for batch_num in 0..num_batches {
            let embeddings = embedder.embed_batch(&refs).expect("Failed to embed");
            total_embedded += embeddings.len();

            if (batch_num + 1) % 10 == 0 {
                let elapsed = start.elapsed();
                let throughput = total_embedded as f64 / elapsed.as_secs_f64();
                println!(
                    "Batch {:>3}/{}: {:>6} strings, {:.1} strings/sec (running avg)",
                    batch_num + 1,
                    num_batches,
                    total_embedded,
                    throughput
                );
            }
        }

        let total_elapsed = start.elapsed();
        let final_throughput = total_embedded as f64 / total_elapsed.as_secs_f64();

        println!("\n=== Final Results ===");
        println!("Total time: {:.3}s", total_elapsed.as_secs_f64());
        println!("Total strings: {}", total_embedded);
        println!("Throughput: {:.1} strings/sec", final_throughput);
        println!(
            "Latency per batch: {:.2}ms",
            total_elapsed.as_millis() as f64 / num_batches as f64
        );
    }

}
