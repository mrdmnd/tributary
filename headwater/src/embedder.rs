use std::time::{Duration, Instant};

use half::f16;
use rand::RngExt;
use reqwest::{
    Client,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::time::sleep;

/// The full dimensionality of the embedding model's output.
const MODEL_DIM: usize = 1024;

/// The truncated MRL dimensionality we actually store (as f16).
pub const EMBEDDING_DIM: usize = 256;

// ============================================================================
// Batch Timing
// ============================================================================

/// Per-phase timing breakdown for a single batch embedding request.
#[derive(Debug, Clone, Default)]
pub struct BatchTimings {
    /// Time to serialize the request body as JSON (microseconds).
    pub serialize_us: u64,
    /// Time for the HTTP round-trip: send request through receiving full response body (microseconds).
    pub http_roundtrip_us: u64,
    /// Time to deserialize the JSON response (microseconds).
    pub deserialize_us: u64,
    /// Time to truncate, normalize, and convert to f16 (microseconds).
    pub convert_us: u64,
    /// Number of strings in this batch.
    pub num_strings: usize,
    /// Size of the serialized request body in bytes.
    pub request_bytes: usize,
    /// Size of the response body in bytes.
    pub response_bytes: usize,
}

impl BatchTimings {
    /// Total wall-clock time for this batch (microseconds).
    pub fn total_us(&self) -> u64 {
        self.serialize_us + self.http_roundtrip_us + self.deserialize_us + self.convert_us
    }
}

// ============================================================================
// Error Type
// ============================================================================

#[derive(Debug, Error)]
pub enum EmbedderError {
    #[error("base URL not found in environment: {0}")]
    BaseUrlNotFound(String),

    #[error("API key not found in environment: {0}")]
    ApiKeyNotFound(String),

    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("failed to parse API response: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("invalid header value: {0}")]
    InvalidHeaderValue(#[from] reqwest::header::InvalidHeaderValue),

    #[error("API returned an error: {0}")]
    ApiError(String),
}

pub trait EmbedderTrait {
    /// Embeds a batch of text strings, returning a flat buffer of `f16` values.
    /// The returned buffer has length `texts.len() * EMBEDDING_DIM`, laid out contiguously:
    /// the embedding for `texts[i]` occupies indices `i * EMBEDDING_DIM..(i + 1) * EMBEDDING_DIM`.
    fn embed_texts(
        &self,
        texts: &[&str],
    ) -> impl std::future::Future<Output = Result<Vec<f16>, EmbedderError>> + Send;
}

// ============================================================================
// Configuration
// ============================================================================

#[derive(Clone, Debug)]
pub struct EmbedderConfig {
    /// The base URL for the OpenAI-compatible embeddings endpoint
    pub base_url: String,
    /// The API key for the OpenAI-compatible embeddings endpoint
    pub api_key: String,
    /// The model to use for the embeddings. Ignored for BASETEN; model is encoded in the URL directly.
    pub model: String,
    /// The timeout for the HTTP request to the API.
    /// Requests that exceed this timeout will be retried according to the retry policy.
    pub request_timeout: Duration,
}

impl EmbedderConfig {
    pub fn from_baseten_env() -> Result<Self, EmbedderError> {
        let base_url = std::env::var("BASETEN_EMBEDDER_URL")
            .map_err(|e| EmbedderError::BaseUrlNotFound(e.to_string()))?;
        let api_key = std::env::var("BASETEN_API_KEY")
            .map_err(|e| EmbedderError::ApiKeyNotFound(e.to_string()))?;
        Ok(Self {
            base_url,
            api_key,
            model: "ignored".to_string(),
            request_timeout: Duration::from_secs(30),
        })
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================
#[derive(Serialize)]
struct ApiRequest<'a> {
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct ApiResponseItem {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct ApiResponse {
    data: Vec<ApiResponseItem>,
}

// ============================================================================
// Implementation
// ============================================================================

pub struct Embedder {
    client: Client,
    config: EmbedderConfig,
}

impl Embedder {
    pub fn new(config: EmbedderConfig) -> Result<Self, EmbedderError> {
        let mut default_headers = HeaderMap::new();
        default_headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Api-Key {}", &config.api_key))?,
        );
        default_headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = Client::builder()
            .timeout(config.request_timeout)
            .default_headers(default_headers)
            .pool_max_idle_per_host(256)
            .tcp_nodelay(true)
            .tcp_keepalive(Duration::from_secs(60))
            .build()?;

        Ok(Self { client, config })
    }

    pub async fn send_batch(&self, texts: &[&str]) -> Result<Vec<f16>, EmbedderError> {
        tracing::debug!("Embedder API call: embedding {} texts", texts.len());

        let payload = ApiRequest { input: texts };

        let response = self
            .client
            .post(&self.config.base_url)
            .json(&payload)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let bytes = response.bytes().await?;
            let body_str = String::from_utf8_lossy(&bytes);
            return Err(EmbedderError::ApiError(format!(
                "API error: status={}, body={}",
                status, body_str
            )));
        }

        let bytes = response.bytes().await?;
        let parsed: ApiResponse = serde_json::from_slice(&bytes)?;

        if parsed.data.len() != texts.len() {
            return Err(EmbedderError::ApiError(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                parsed.data.len()
            )));
        }

        // Truncate to EMBEDDING_DIM (MRL), L2-normalize, and convert f32 → f16
        let mut flat = Vec::with_capacity(texts.len() * EMBEDDING_DIM);
        for (i, item) in parsed.data.into_iter().enumerate() {
            if item.embedding.len() != MODEL_DIM {
                return Err(EmbedderError::ApiError(format!(
                    "embedding[{}] has dimension {}, expected {}",
                    i,
                    item.embedding.len(),
                    MODEL_DIM
                )));
            }

            let truncated = &item.embedding[..EMBEDDING_DIM];
            let norm = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
            flat.extend(truncated.iter().map(|&x| f16::from_f32(x * inv_norm)));
        }

        Ok(flat)
    }

    /// Like [`send_batch`], but returns per-phase timing information alongside
    /// the embeddings. Used for profiling and throughput analysis.
    pub async fn send_batch_timed(
        &self,
        texts: &[&str],
    ) -> Result<(Vec<f16>, BatchTimings), EmbedderError> {
        let mut timings = BatchTimings {
            num_strings: texts.len(),
            ..Default::default()
        };

        // Phase 1: Serialize request body
        let t = Instant::now();
        let payload = ApiRequest { input: texts };
        let body = serde_json::to_vec(&payload)?;
        timings.serialize_us = t.elapsed().as_micros() as u64;
        timings.request_bytes = body.len();

        // Phase 2: HTTP round-trip (send request + receive full response body)
        let t = Instant::now();
        let response = self
            .client
            .post(&self.config.base_url)
            .body(body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let bytes = response.bytes().await?;
            let body_str = String::from_utf8_lossy(&bytes);
            return Err(EmbedderError::ApiError(format!(
                "API error: status={}, body={}",
                status, body_str
            )));
        }

        let bytes = response.bytes().await?;
        timings.http_roundtrip_us = t.elapsed().as_micros() as u64;
        timings.response_bytes = bytes.len();

        // Phase 3: Deserialize JSON response
        let t = Instant::now();
        let parsed: ApiResponse = serde_json::from_slice(&bytes)?;
        timings.deserialize_us = t.elapsed().as_micros() as u64;

        if parsed.data.len() != texts.len() {
            return Err(EmbedderError::ApiError(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                parsed.data.len()
            )));
        }

        // Phase 4: Truncate to EMBEDDING_DIM (MRL), L2-normalize, convert f32 → f16
        let t = Instant::now();
        let mut flat = Vec::with_capacity(texts.len() * EMBEDDING_DIM);
        for (i, item) in parsed.data.into_iter().enumerate() {
            if item.embedding.len() != MODEL_DIM {
                return Err(EmbedderError::ApiError(format!(
                    "embedding[{}] has dimension {}, expected {}",
                    i,
                    item.embedding.len(),
                    MODEL_DIM
                )));
            }

            let truncated = &item.embedding[..EMBEDDING_DIM];
            let norm = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
            flat.extend(truncated.iter().map(|&x| f16::from_f32(x * inv_norm)));
        }
        timings.convert_us = t.elapsed().as_micros() as u64;

        Ok((flat, timings))
    }
}

impl EmbedderTrait for Embedder {
    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<f16>, EmbedderError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let max_retries = 5;
        let mut current_attempt = 0;
        let mut current_backoff = Duration::from_millis(200);
        let max_backoff = Duration::from_secs(10);

        loop {
            let result = self.send_batch(texts).await;
            match result {
                Ok(embeddings) => return Ok(embeddings),
                Err(e) => {
                    current_attempt += 1;
                    if current_attempt >= max_retries {
                        tracing::error!(
                            "Failed to embed texts after {} attempts: {}",
                            current_attempt,
                            e
                        );
                        return Err(e);
                    }

                    let jitter: f64 = rand::rng().random_range(0.5..1.5);
                    let jittered = current_backoff.mul_f64(jitter);

                    tracing::debug!(
                        "Embedding request failed (attempt {}/{}), retrying in {}ms: {}",
                        current_attempt,
                        max_retries,
                        jittered.as_millis(),
                        e
                    );
                    sleep(jittered).await;
                    current_backoff = std::cmp::min(current_backoff * 2, max_backoff);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Waits for the remote embedder to become responsive (handles cold starts).
    /// Sends a trivial 1-string probe with generous patience before giving up.
    async fn wait_for_server(embedder: &Embedder, timeout: Duration) {
        let deadline = tokio::time::Instant::now() + timeout;
        let probe: &[&str] = &["warmup"];
        let mut interval = Duration::from_secs(5);
        let max_interval = Duration::from_secs(30);

        loop {
            match embedder.send_batch(probe).await {
                Ok(_) => {
                    tracing::info!("Embedder server is warm and responding");
                    return;
                }
                Err(e) => {
                    if tokio::time::Instant::now() >= deadline {
                        panic!("Embedder server did not become responsive within {timeout:?}: {e}");
                    }
                    tracing::warn!(
                        "Server not ready yet ({}), retrying in {}s...",
                        e,
                        interval.as_secs()
                    );
                    sleep(interval).await;
                    interval = std::cmp::min(interval * 2, max_interval);
                }
            }
        }
    }

    #[tokio::test]
    #[ignore] // requires BASETEN_EMBEDDER_URL and BASETEN_API_KEY env vars
    async fn test_embed_texts_live() {
        tracing_subscriber::fmt()
            .with_env_filter("info")
            .with_test_writer()
            .try_init()
            .ok();

        let config = EmbedderConfig::from_baseten_env()
            .expect("BASETEN_EMBEDDER_URL and BASETEN_API_KEY must be set");
        let embedder = Embedder::new(config).expect("failed to build embedder");

        // Wait up to 5 minutes for a cold-starting server
        wait_for_server(&embedder, Duration::from_secs(300)).await;

        let texts: &[&str] = &[
            "The quick brown fox jumps over the lazy dog.",
            "Rust is a systems programming language.",
            "Matryoshka representation learning enables flexible truncation.",
        ];

        let result = embedder
            .embed_texts(texts)
            .await
            .expect("embed_texts failed");

        // Validate flat buffer shape
        assert_eq!(
            result.len(),
            texts.len() * EMBEDDING_DIM,
            "expected {} f16 values, got {}",
            texts.len() * EMBEDDING_DIM,
            result.len()
        );

        // Validate that each truncated embedding is approximately unit-normalized
        for (i, chunk) in result.chunks_exact(EMBEDDING_DIM).enumerate() {
            let norm_sq: f32 = chunk
                .iter()
                .map(|x| {
                    let v = x.to_f32();
                    v * v
                })
                .sum();
            let norm = norm_sq.sqrt();
            assert!(
                (norm - 1.0).abs() < 0.05,
                "embedding[{i}] L2 norm = {norm:.4}, expected ~1.0"
            );
        }

        // Sanity check: different strings should produce different embeddings
        let emb0 = &result[..EMBEDDING_DIM];
        let emb1 = &result[EMBEDDING_DIM..2 * EMBEDDING_DIM];
        let cosine_sim: f32 = emb0
            .iter()
            .zip(emb1.iter())
            .map(|(a, b)| a.to_f32() * b.to_f32())
            .sum();
        assert!(
            cosine_sim < 0.99,
            "embeddings for different strings are suspiciously similar (cosine={cosine_sim:.4})"
        );

        tracing::info!(
            "Embedded {} texts → {} f16 values, cosine(0,1) = {:.4}",
            texts.len(),
            result.len(),
            cosine_sim
        );
    }
}
