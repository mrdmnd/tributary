//! Profiling harness for the embedder pipeline.
//!
//! Designed to be run both standalone (for coarse timing) and under NVIDIA
//! Nsight Systems for GPU kernel-level analysis:
//!
//! ```sh
//! # Build in release mode:
//! cargo build --release --bin profile_embedder
//!
//! # Standalone with timing breakdown:
//! cargo run --release --bin profile_embedder
//!
//! # Under Nsight Systems (full CPU+GPU timeline):
//! nsys profile --stats=true --cuda-memory-usage=true \
//!     -o embedder_profile \
//!     ./target/release/profile_embedder
//!
//! # Per-kernel metrics with Nsight Compute (pick a specific kernel):
//! ncu --set full -o embedder_kernels \
//!     ./target/release/profile_embedder -- --iterations 1
//! ```

use std::time::Instant;

use tributary::embedder::{Embedder, EmbedderConfig};

/// Short, column-description-style strings (the target workload).
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

fn main() {
    // Parse simple CLI flags
    let args: Vec<String> = std::env::args().collect();
    let iterations: usize = args
        .iter()
        .position(|a| a == "--iterations")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let num_strings: usize = args
        .iter()
        .position(|a| a == "--strings")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(5_000);
    let chunk_size: usize = args
        .iter()
        .position(|a| a == "--chunk-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);

    println!("=== Embedder Profiling Harness ===");
    println!(
        "  strings: {}  chunk_size: {}  iterations: {}",
        num_strings, chunk_size, iterations
    );
    println!();

    // ── Model loading ───────────────────────────────────────────────────
    let t_load = Instant::now();
    let embedder = Embedder::new(EmbedderConfig::default()).expect("Failed to create embedder");
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    println!("Model load:        {:.1} ms", load_ms);
    println!("  dtype:           {:?}", embedder.config.dtype);
    println!("  max_seq_len:     {}", embedder.config.max_seq_len);
    println!();

    // ── Data generation ─────────────────────────────────────────────────
    let strings = generate_short_strings(num_strings);
    let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

    // ── Warm-up (1 iteration, not timed against main loop) ──────────────
    println!("Warm-up...");
    let t_warmup = Instant::now();
    let _ = embedder.embed_batch_chunked(&refs, chunk_size).unwrap();
    let warmup_ms = t_warmup.elapsed().as_secs_f64() * 1000.0;
    println!("Warm-up:           {:.1} ms", warmup_ms);
    println!();

    // ── Measured iterations ─────────────────────────────────────────────
    println!("--- Timed iterations ---");
    let mut times_ms = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let t_iter = Instant::now();
        let result = embedder.embed_batch_chunked(&refs, chunk_size).unwrap();
        let iter_ms = t_iter.elapsed().as_secs_f64() * 1000.0;
        times_ms.push(iter_ms);

        let throughput = num_strings as f64 / (iter_ms / 1000.0);
        println!(
            "  iter {}: {:.2} ms  ({:.0} strings/sec)  [result len={}]",
            i,
            iter_ms,
            throughput,
            result.len()
        );
    }

    // ── Summary statistics ──────────────────────────────────────────────
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times_ms[times_ms.len() / 2];
    let min = times_ms[0];
    let max = times_ms[times_ms.len() - 1];
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;

    println!();
    println!("=== Summary ({} strings, chunk_size={}) ===", num_strings, chunk_size);
    println!("  min:    {:.2} ms  ({:.0} strings/sec)", min, num_strings as f64 / (min / 1000.0));
    println!("  median: {:.2} ms  ({:.0} strings/sec)", median, num_strings as f64 / (median / 1000.0));
    println!("  mean:   {:.2} ms  ({:.0} strings/sec)", mean, num_strings as f64 / (mean / 1000.0));
    println!("  max:    {:.2} ms  ({:.0} strings/sec)", max, num_strings as f64 / (max / 1000.0));
    println!();

    // ── Per-chunk breakdown (single pass) ───────────────────────────────
    println!("=== Per-chunk breakdown (single pass) ===");
    let mut chunk_idx = 0;
    let mut total_embed_ms = 0.0;
    for chunk in refs.chunks(chunk_size) {
        let t_chunk = Instant::now();
        let _ = embedder.embed_batch(chunk).unwrap();
        let chunk_ms = t_chunk.elapsed().as_secs_f64() * 1000.0;
        total_embed_ms += chunk_ms;
        println!(
            "  chunk {}: {} strings -> {:.2} ms  ({:.2} ms/string)",
            chunk_idx,
            chunk.len(),
            chunk_ms,
            chunk_ms / chunk.len() as f64
        );
        chunk_idx += 1;
    }
    println!("  total embed: {:.2} ms", total_embed_ms);
}
