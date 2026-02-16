//! Embedder throughput benchmarks.
//!
//! Benchmarks the ORT backend on two workloads:
//! - **chunk_size_sweep**: Short, uniform strings — reveals the GPU saturation
//!   curve and optimal `batch_size`.
//! - **mixed_length**: Realistic log-normal token-length distribution — measures
//!   real-world throughput including padding overhead.
//!
//! # Running
//!
//! ```sh
//! cargo bench --bench embedder_throughput
//! ```
//!
//! # GPU clock-locking for reproducible results
//!
//! NVIDIA GPUs dynamically adjust clock frequencies based on power and thermal
//! limits.  For stable, reproducible benchmark numbers, lock the GPU clocks
//! before running:
//!
//! ```sh
//! # Query available clock frequencies:
//! nvidia-smi -q -d SUPPORTED_CLOCKS
//!
//! # Lock the graphics clock to a fixed frequency (pick one your GPU supports):
//! sudo nvidia-smi -lgc <freq>,<freq>
//!
//! # Optionally lock the power limit:
//! sudo nvidia-smi -pl <watts>
//!
//! # Run the benchmarks:
//! cargo bench --bench embedder_throughput
//!
//! # Reset clocks when done:
//! sudo nvidia-smi -rgc
//! sudo nvidia-smi -rpl
//! ```

use criterion::{BenchmarkId, Criterion, Throughput};
use rand::Rng;
use rand_distr::LogNormal;
use tributary::embedder::{DEFAULT_EMBEDDING_REPO, EmbedderConfig, OrtEmbedder};

// ============================================================================
// Test data generators
// ============================================================================

/// Short, column-description-style strings (3-6 tokens each).
fn generate_short_strings(count: usize) -> Vec<String> {
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

/// Generate strings with a log-normal word-count distribution.
///
/// This produces a realistic mix of short and long strings within a single
/// batch — much closer to production traffic than uniform-length corpora.
/// Token lengths are clamped to `[3, max_words]` to stay within BERT's
/// sequence-length limit.
fn generate_mixed_length_strings(count: usize, max_words: usize) -> Vec<String> {
    let sample_phrases = [
        "the customer placed an order for multiple products",
        "this field contains the unique identifier for each record",
        "timestamp indicating when the transaction was completed",
        "foreign key reference to the parent table entry",
        "calculated value based on quantity and unit price",
        "status flag showing whether the item is active",
        "description text providing additional context and details",
        "numeric value representing the total amount due",
        "date field for tracking creation and modification times",
        "category classification used for filtering and grouping",
    ];

    // Log-normal with mu=3.0, sigma=0.7 gives a median of ~20 words and a
    // long right tail, which is a reasonable approximation of real-world
    // column description / field documentation lengths.
    let ln = LogNormal::new(3.0, 0.7).unwrap();
    let mut rng = rand::rng();

    (0..count)
        .map(|i| {
            let target_words: usize = (rng.sample::<f64, _>(&ln) as usize).clamp(3, max_words);
            let mut result = String::new();
            let mut word_count = 0;
            while word_count < target_words {
                let phrase = &sample_phrases[i % sample_phrases.len()];
                if !result.is_empty() {
                    result.push_str(". ");
                }
                result.push_str(phrase);
                word_count += phrase.split_whitespace().count();
            }
            format!("Field {}: {}", i, result)
        })
        .collect()
}

/// Load a tokenizer (for counting tokens in the mixed-length benchmark).
fn load_tokenizer() -> tokenizers::Tokenizer {
    let api = hf_hub::api::sync::Api::new().unwrap();
    let repo = api.repo(hf_hub::Repo::new(
        DEFAULT_EMBEDDING_REPO.to_string(),
        hf_hub::RepoType::Model,
    ));
    let path = repo.get("tokenizer.json").unwrap();
    tokenizers::Tokenizer::from_file(&path).unwrap()
}

// ============================================================================
// ORT Benchmarks
// ============================================================================

/// Resolve the ONNX model path from the environment or use the default.
fn ort_model_path() -> String {
    std::env::var("ONNX_MODEL_PATH")
        .unwrap_or_else(|_| "ort/models/bge-base-en-v1.5-onnx/model_fp16_pooled.onnx".to_string())
}

/// Create an ORT embedder, or panic with a helpful message if the model is missing.
///
/// Set `ORT_PROFILE_DIR` to enable ORT's built-in profiler (writes a JSON
/// trace to the given directory).  Call [`OrtEmbedder::end_profiling`] to
/// flush after the benchmark completes.
fn create_ort_embedder() -> OrtEmbedder {
    let path = ort_model_path();
    if !std::path::Path::new(&path).exists() {
        panic!(
            "ONNX model not found at '{}'. Run `uv run scripts/export_onnx.py` first, \
             or set ONNX_MODEL_PATH.",
            path
        );
    }
    let mut config = EmbedderConfig::default().with_onnx_model(&path);
    if let Ok(profile_prefix) = std::env::var("ORT_PROFILE_PREFIX") {
        eprintln!("[profile] ORT profiling enabled → {profile_prefix}");
        config = config.with_profiling(&profile_prefix);
    }
    OrtEmbedder::new(config).expect("Failed to create ORT embedder")
}

fn bench_chunk_size_sweep(c: &mut Criterion) {
    let num_strings = 8_192;
    let strings = generate_short_strings(num_strings);
    let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

    let embedder = create_ort_embedder();

    let mut group = c.benchmark_group("chunk_size_sweep");
    group.sample_size(50);
    group.noise_threshold(0.05);
    group.throughput(Throughput::Elements(num_strings as u64));

    for chunk_size in [512, 1024, 2048, 4096, 8192] {
        group.bench_with_input(
            BenchmarkId::new("embed_batch_chunked", chunk_size),
            &chunk_size,
            |b, &chunk_size| {
                b.iter(|| embedder.embed_batch_chunked(&refs, chunk_size).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_mixed_length(c: &mut Criterion) {
    let tokenizer = load_tokenizer();

    let num_strings = 8_192;
    let strings = generate_mixed_length_strings(num_strings, 60);
    let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

    let total_tokens: usize = strings
        .iter()
        .map(|s| tokenizer.encode(s.as_str(), true).unwrap().get_ids().len())
        .sum();

    let embedder = create_ort_embedder();

    let mut group = c.benchmark_group("mixed_length");
    group.sample_size(10);
    group.noise_threshold(0.05);
    group.throughput(Throughput::Elements(total_tokens as u64));

    // Keep chunk sizes ≤ 2048 for mixed-length strings — larger chunks
    // cause catastrophic padding waste because the longest string in the
    // chunk determines seq_len for the entire batch.
    for chunk_size in [256, 512, 1024, 2048] {
        group.bench_with_input(
            BenchmarkId::new("embed_batch_chunked", chunk_size),
            &chunk_size,
            |b, &chunk_size| {
                b.iter(|| embedder.embed_batch_chunked(&refs, chunk_size).unwrap());
            },
        );
    }
    group.finish();
}

// ============================================================================
// Criterion main
// ============================================================================

fn main() {
    let mut criterion = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(15))
        .configure_from_args();

    bench_chunk_size_sweep(&mut criterion);
    bench_mixed_length(&mut criterion);

    criterion.final_summary();
}
