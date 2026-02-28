//! Benchmark sampler throughput and latency under multi-process load.
//!
//! Coordinator mode (default) spawns N worker processes, each configured with
//! a unique rank in the same world_size. Workers warm up, then measure
//! next_train_batch() latency for a fixed number of batches and emit JSON
//! results to temp files. Coordinator aggregates wall-clock throughput and
//! merged latency quantiles.
//!
//! Usage:
//!   cargo run --release --bin benchmark_sampler -- \
//!     --db-dir data/processed/rel-stack --workers 4 --batches-per-worker 200

use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use headwater::sampler::{Sampler, SamplerConfig};

#[derive(Parser, Debug, Clone)]
#[command(about = "Benchmark headwater sampler throughput and latency")]
struct Args {
    /// Path to the preprocessed database directory.
    #[arg(long)]
    db_dir: PathBuf,

    /// Number of worker processes to run in parallel.
    #[arg(long, default_value_t = 1)]
    workers: usize,

    /// Number of measured batches per worker (after warmup).
    #[arg(long, default_value_t = 200)]
    batches_per_worker: usize,

    /// Number of warmup batches per worker before timing starts.
    #[arg(long, default_value_t = 20)]
    warmup_batches: usize,

    /// Batch size B.
    #[arg(long, default_value_t = 32)]
    batch_size: u32,

    /// Sequence length S.
    #[arg(long, default_value_t = 1024)]
    sequence_length: u32,

    /// Max sampled children per P->F edge during BFS.
    #[arg(long, default_value_t = 16)]
    bfs_child_width: u32,

    /// Sampler prefetch depth per split queue.
    #[arg(long, default_value_t = 3)]
    num_prefetch: usize,

    /// Split seed for deterministic split assignment.
    #[arg(long, default_value_t = 123)]
    split_seed: u64,

    /// Base RNG seed; each worker offsets by rank.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Comma-separated split ratios train,val,test.
    #[arg(long, default_value = "0.8,0.1,0.1")]
    split_ratios: String,

    /// Optional fixed Rayon thread count per worker process.
    #[arg(long)]
    worker_rayon_threads: Option<usize>,

    /// Optional path to write aggregated JSON results.
    #[arg(long)]
    output_json: Option<PathBuf>,

    /// Optional path to write a CPU flamegraph SVG for the worker.
    /// Best used with --workers 1.
    #[arg(long)]
    cpu_flamegraph: Option<PathBuf>,

    /// Enable internal headwater sampler profiling logs.
    #[arg(long, default_value_t = false)]
    headwater_profile: bool,

    /// Internal worker mode.
    #[arg(long, hide = true, default_value_t = false)]
    worker: bool,

    /// Internal worker rank.
    #[arg(long, hide = true, default_value_t = 0)]
    rank: u32,

    /// Internal world size.
    #[arg(long, hide = true, default_value_t = 1)]
    world_size: u32,

    /// Internal worker result path (JSON).
    #[arg(long, hide = true)]
    result_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerResult {
    worker_rank: u32,
    world_size: u32,
    warmup_batches: usize,
    measured_batches: usize,
    batch_size: u32,
    sequence_length: u32,
    elapsed_s: f64,
    throughput_batches_per_s: f64,
    throughput_cells_per_s: f64,
    latency_ms_avg: f64,
    latency_ms_p50: f64,
    latency_ms_p90: f64,
    latency_ms_p95: f64,
    latency_ms_p99: f64,
    latency_ms_min: f64,
    latency_ms_max: f64,
    latencies_ms: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AggregateResult {
    workers: usize,
    warmup_batches_per_worker: usize,
    measured_batches_per_worker: usize,
    batch_size: u32,
    sequence_length: u32,
    total_measured_batches: usize,
    wall_elapsed_s: f64,
    aggregate_throughput_batches_per_s: f64,
    aggregate_throughput_cells_per_s: f64,
    merged_latency_ms_avg: f64,
    merged_latency_ms_p50: f64,
    merged_latency_ms_p90: f64,
    merged_latency_ms_p95: f64,
    merged_latency_ms_p99: f64,
    merged_latency_ms_min: f64,
    merged_latency_ms_max: f64,
    worker_results: Vec<WorkerResult>,
}

fn parse_split_ratios(raw: &str) -> Result<(f32, f32, f32), String> {
    let parts: Vec<&str> = raw.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return Err(format!(
            "split_ratios must have 3 comma-separated values, got: {raw}"
        ));
    }
    let a = parts[0]
        .parse::<f32>()
        .map_err(|e| format!("invalid train ratio '{}': {e}", parts[0]))?;
    let b = parts[1]
        .parse::<f32>()
        .map_err(|e| format!("invalid val ratio '{}': {e}", parts[1]))?;
    let c = parts[2]
        .parse::<f32>()
        .map_err(|e| format!("invalid test ratio '{}': {e}", parts[2]))?;
    let sum = a + b + c;
    if (sum - 1.0).abs() > 1e-6 {
        return Err(format!("split_ratios must sum to 1.0, got {sum:.6}"));
    }
    Ok((a, b, c))
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let q = q.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn summarize_latencies(latencies_ms: &[f64]) -> (f64, f64, f64, f64, f64, f64, f64) {
    if latencies_ms.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let mut sorted = latencies_ms.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let sum: f64 = sorted.iter().sum();
    let avg = sum / sorted.len() as f64;
    let p50 = percentile(&sorted, 0.50);
    let p90 = percentile(&sorted, 0.90);
    let p95 = percentile(&sorted, 0.95);
    let p99 = percentile(&sorted, 0.99);
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    (avg, p50, p90, p95, p99, min, max)
}

fn run_worker(args: &Args) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let split_ratios = parse_split_ratios(&args.split_ratios)?;
    let result_path = args
        .result_path
        .clone()
        .ok_or_else(|| "worker mode requires --result-path".to_string())?;
    let config = SamplerConfig {
        db_path: args.db_dir.to_string_lossy().to_string(),
        rank: args.rank,
        world_size: args.world_size,
        split_ratios,
        split_seed: args.split_seed,
        seed: args.seed.wrapping_add(args.rank as u64),
        num_prefetch: args.num_prefetch,
        batch_size: args.batch_size,
        sequence_length: args.sequence_length,
        bfs_child_width: args.bfs_child_width,
    };

    let mut sampler = Sampler::new(config)?;
    for _ in 0..args.warmup_batches {
        let _ = sampler.next_train_batch()?;
    }

    let profiler = if args.cpu_flamegraph.is_some() {
        Some(
            pprof::ProfilerGuardBuilder::default()
                .frequency(999)
                .blocklist(&["libc", "libgcc", "pthread", "vdso"])
                .build()?,
        )
    } else {
        None
    };

    let mut latencies_ms = Vec::with_capacity(args.batches_per_worker);
    let mut total_cells: usize = 0;
    let measured_start = Instant::now();
    for _ in 0..args.batches_per_worker {
        let t0 = Instant::now();
        let batch = sampler.next_train_batch()?;
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        latencies_ms.push(dt_ms);
        total_cells += batch.batch_size * batch.sequence_length;
    }
    let elapsed_s = measured_start.elapsed().as_secs_f64();
    sampler.shutdown();

    if let (Some(path), Some(guard)) = (&args.cpu_flamegraph, profiler) {
        let report = guard.report().build()?;
        let mut file = File::create(path)?;
        report.flamegraph(&mut file)?;
    }

    let (avg, p50, p90, p95, p99, min, max) = summarize_latencies(&latencies_ms);
    let throughput_batches_per_s = if elapsed_s > 0.0 {
        args.batches_per_worker as f64 / elapsed_s
    } else {
        0.0
    };
    let throughput_cells_per_s = if elapsed_s > 0.0 {
        total_cells as f64 / elapsed_s
    } else {
        0.0
    };

    let result = WorkerResult {
        worker_rank: args.rank,
        world_size: args.world_size,
        warmup_batches: args.warmup_batches,
        measured_batches: args.batches_per_worker,
        batch_size: args.batch_size,
        sequence_length: args.sequence_length,
        elapsed_s,
        throughput_batches_per_s,
        throughput_cells_per_s,
        latency_ms_avg: avg,
        latency_ms_p50: p50,
        latency_ms_p90: p90,
        latency_ms_p95: p95,
        latency_ms_p99: p99,
        latency_ms_min: min,
        latency_ms_max: max,
        latencies_ms,
    };

    let payload = serde_json::to_string_pretty(&result)?;
    fs::write(result_path, payload)?;
    Ok(())
}

fn spawn_worker(
    exe: &Path,
    args: &Args,
    rank: usize,
    world_size: usize,
    result_path: &Path,
) -> Result<Child, Box<dyn std::error::Error + Send + Sync>> {
    let mut cmd = Command::new(exe);
    cmd.arg("--worker")
        .arg("--db-dir")
        .arg(&args.db_dir)
        .arg("--workers")
        .arg(args.workers.to_string())
        .arg("--batches-per-worker")
        .arg(args.batches_per_worker.to_string())
        .arg("--warmup-batches")
        .arg(args.warmup_batches.to_string())
        .arg("--batch-size")
        .arg(args.batch_size.to_string())
        .arg("--sequence-length")
        .arg(args.sequence_length.to_string())
        .arg("--bfs-child-width")
        .arg(args.bfs_child_width.to_string())
        .arg("--num-prefetch")
        .arg(args.num_prefetch.to_string())
        .arg("--split-seed")
        .arg(args.split_seed.to_string())
        .arg("--seed")
        .arg(args.seed.to_string())
        .arg("--split-ratios")
        .arg(&args.split_ratios)
        .arg("--rank")
        .arg(rank.to_string())
        .arg("--world-size")
        .arg(world_size.to_string())
        .arg("--result-path")
        .arg(result_path);

    if let Some(path) = &args.cpu_flamegraph {
        cmd.arg("--cpu-flamegraph").arg(path);
    }

    if let Some(threads) = args.worker_rayon_threads {
        cmd.env("RAYON_NUM_THREADS", threads.to_string());
    }
    if args.headwater_profile {
        cmd.env("HEADWATER_PROFILE", "1");
    }

    Ok(cmd.spawn()?)
}

fn run_coordinator(args: &Args) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if args.workers == 0 {
        return Err("workers must be >= 1".into());
    }
    if args.cpu_flamegraph.is_some() && args.workers > 1 {
        return Err("cpu_flamegraph currently supports only --workers 1".into());
    }

    let exe = std::env::current_exe()?;
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let temp_root = std::env::temp_dir().join(format!(
        "headwater-benchmark-{}-{}",
        std::process::id(),
        now_ns
    ));
    fs::create_dir_all(&temp_root)?;

    let total_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let threads_per_worker = args
        .worker_rayon_threads
        .unwrap_or_else(|| (total_cores / args.workers).max(1));

    info!(
        "Starting sampler benchmark: workers={}, warmup_batches/worker={}, measured_batches/worker={}, batch_size={}, sequence_length={}",
        args.workers,
        args.warmup_batches,
        args.batches_per_worker,
        args.batch_size,
        args.sequence_length
    );
    info!(
        "CPU cores={}, worker RAYON_NUM_THREADS={} (override with --worker-rayon-threads)",
        total_cores, threads_per_worker
    );

    let mut children = Vec::with_capacity(args.workers);
    let mut result_paths = Vec::with_capacity(args.workers);
    let wall_start = Instant::now();

    for rank in 0..args.workers {
        let result_path = temp_root.join(format!("worker-{rank}.json"));
        let mut worker_args = args.clone();
        worker_args.worker_rayon_threads = Some(threads_per_worker);
        let child = spawn_worker(&exe, &worker_args, rank, args.workers, &result_path)?;
        children.push((rank, child));
        result_paths.push(result_path);
    }

    let mut failed = false;
    for (rank, mut child) in children {
        let status = child.wait()?;
        if !status.success() {
            warn!("Worker rank {} exited with status {}", rank, status);
            failed = true;
        }
    }
    if failed {
        return Err("one or more workers failed".into());
    }

    let wall_elapsed_s = wall_start.elapsed().as_secs_f64();
    let mut worker_results = Vec::with_capacity(args.workers);
    for path in &result_paths {
        let data = fs::read_to_string(path)?;
        let wr: WorkerResult = serde_json::from_str(&data)?;
        worker_results.push(wr);
    }
    worker_results.sort_by_key(|w| w.worker_rank);

    let total_measured_batches: usize = worker_results.iter().map(|w| w.measured_batches).sum();
    let total_cells: usize = worker_results
        .iter()
        .map(|w| w.measured_batches * w.batch_size as usize * w.sequence_length as usize)
        .sum();

    let mut merged_latencies = Vec::new();
    for wr in &worker_results {
        merged_latencies.extend_from_slice(&wr.latencies_ms);
    }
    let (avg, p50, p90, p95, p99, min, max) = summarize_latencies(&merged_latencies);

    let aggregate = AggregateResult {
        workers: args.workers,
        warmup_batches_per_worker: args.warmup_batches,
        measured_batches_per_worker: args.batches_per_worker,
        batch_size: args.batch_size,
        sequence_length: args.sequence_length,
        total_measured_batches,
        wall_elapsed_s,
        aggregate_throughput_batches_per_s: if wall_elapsed_s > 0.0 {
            total_measured_batches as f64 / wall_elapsed_s
        } else {
            0.0
        },
        aggregate_throughput_cells_per_s: if wall_elapsed_s > 0.0 {
            total_cells as f64 / wall_elapsed_s
        } else {
            0.0
        },
        merged_latency_ms_avg: avg,
        merged_latency_ms_p50: p50,
        merged_latency_ms_p90: p90,
        merged_latency_ms_p95: p95,
        merged_latency_ms_p99: p99,
        merged_latency_ms_min: min,
        merged_latency_ms_max: max,
        worker_results,
    };

    info!(
        "Aggregate throughput: {:.2} batches/s, {:.2} cells/s, wall_elapsed={:.3}s",
        aggregate.aggregate_throughput_batches_per_s,
        aggregate.aggregate_throughput_cells_per_s,
        aggregate.wall_elapsed_s
    );
    info!(
        "Merged latency ms: avg={:.3}, p50={:.3}, p95={:.3}, p99={:.3}, max={:.3}",
        aggregate.merged_latency_ms_avg,
        aggregate.merged_latency_ms_p50,
        aggregate.merged_latency_ms_p95,
        aggregate.merged_latency_ms_p99,
        aggregate.merged_latency_ms_max
    );

    let json = serde_json::to_string_pretty(&aggregate)?;
    if let Some(path) = &args.output_json {
        fs::write(path, &json)?;
        info!("Wrote benchmark JSON to {}", path.display());
    } else {
        info!("Benchmark JSON output path not set; pass --output-json to persist full results");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    if args.worker {
        run_worker(&args)
    } else {
        run_coordinator(&args)
    }
}
