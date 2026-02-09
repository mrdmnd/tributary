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
//! # Make sure ORT_DYLIB_PATH is set to the CUDA 13 ORT library:
//! export ORT_DYLIB_PATH=/path/to/ort/ort-libs/onnxruntime-linux-x64-gpu-1.24.1/lib/libonnxruntime.so.1.24.1
//! export LD_LIBRARY_PATH=/path/to/ort/ort-libs/onnxruntime-linux-x64-gpu-1.24.1/lib:$LD_LIBRARY_PATH
//!
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

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::Rng;
use rand_distr::LogNormal;
use tributary::embedder::{DEFAULT_EMBEDDING_REPO, Embedder, EmbedderConfig};

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

// ============================================================================
// Benchmark helpers
// ============================================================================

/// Resolve the ONNX model path from the environment or use the default.
fn ort_model_path() -> String {
    std::env::var("ONNX_MODEL_PATH")
        .unwrap_or_else(|_| "ort/models/bge-base-en-v1.5-onnx/model_fp16.onnx".to_string())
}

/// Create an embedder, or panic with a helpful message if the model is missing.
fn create_embedder() -> Embedder {
    let path = ort_model_path();
    if !std::path::Path::new(&path).exists() {
        panic!(
            "ONNX model not found at '{}'. Run `uv run ort/scripts/export_onnx.py` first, \
             or set ONNX_MODEL_PATH.",
            path
        );
    }
    Embedder::new(EmbedderConfig::default().with_onnx_model(&path))
        .expect("Failed to create embedder")
}

// ============================================================================
// Benchmarks
// ============================================================================

/// Sweep `chunk_size` (the GPU batch size) with a fixed corpus of short
/// strings.  This reveals the GPU saturation curve and is the most actionable
/// benchmark for tuning `DEFAULT_BATCH_SIZE`.
fn bench_chunk_size_sweep(c: &mut Criterion) {
    let num_strings = 5_000;
    let strings = generate_short_strings(num_strings);
    let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

    let embedder = create_embedder();

    let mut group = c.benchmark_group("chunk_size_sweep");
    group.noise_threshold(0.05);
    group.throughput(Throughput::Elements(num_strings as u64));

    for chunk_size in [128, 256, 512, 1024, 2048, 4096] {
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

/// Benchmark with a realistic mixed-length corpus (log-normal token-length
/// distribution).  Reports throughput in tokens/sec so we can see how padding
/// overhead and variable sequence lengths affect real-world performance.
fn bench_mixed_length(c: &mut Criterion) {
    // Load the tokenizer separately so we can count tokens for the throughput
    // metric without including tokenizer load time in the measurement.
    let tokenizer = {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let repo = api.repo(hf_hub::Repo::new(
            DEFAULT_EMBEDDING_REPO.to_string(),
            hf_hub::RepoType::Model,
        ));
        let path = repo.get("tokenizer.json").unwrap();
        tokenizers::Tokenizer::from_file(&path).unwrap()
    };

    let num_strings = 5_000;
    // Cap word count so that after tokenization we stay within BERT's 512
    // token limit.  ~60 words ≈ ~80-100 tokens with BPE.
    let strings = generate_mixed_length_strings(num_strings, 60);
    let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

    let total_tokens: usize = strings
        .iter()
        .map(|s| tokenizer.encode(s.as_str(), true).unwrap().get_ids().len())
        .sum();

    let embedder = create_embedder();

    let mut group = c.benchmark_group("mixed_length");
    group.noise_threshold(0.05);
    group.throughput(Throughput::Elements(total_tokens as u64));

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

criterion_group! {
    name = benches;
    config = Criterion::default()
        // Each iteration runs a full GPU batch, so we don't need many samples.
        .sample_size(10)
        // Give the GPU time to stabilize.
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(20));
    targets = bench_chunk_size_sweep, bench_mixed_length
}
criterion_main!(benches);
