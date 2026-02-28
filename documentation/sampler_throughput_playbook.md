# Sampler Throughput Playbook

This playbook focuses on maximizing end-to-end sample throughput for training.

## 1) Run an offline sampler sweep

Use the sweep helper to benchmark `headwater` across worker/thread/prefetch/shape configs.

```bash
./scripts/run_sampler_throughput_sweep.sh /path/to/processed_db
```

Environment knobs:

- `WORKERS_LIST` (default: `1 2 4 8`)
- `PREFETCH_LIST` (default: `2 3 6`)
- `BATCH_LIST` (default: `16 32`)
- `SEQLEN_LIST` (default: `512 1024`)
- `BFS_LIST` (default: `16 32`)
- `WARMUP_BATCHES` (default: `30`)
- `MEASURED_BATCHES` (default: `300`)
- `OUTPUT_DIR` to choose output location.

The script writes benchmark JSON files and prints a top-k summary sorted by cells/sec.

## 2) Launch one trainer process per GPU

For single-node multi-GPU runs:

```bash
./scripts/launch_confluence_8gpu.sh /path/to/processed_db --num-steps 10000 --batch-size 32 --seq-length 1024
```

Important environment knobs:

- `NUM_PROCS` (default: `8`)
- `RAYON_NUM_THREADS` (default: `nproc / NUM_PROCS`)
- `JAX_COORDINATOR_ADDRESS` (default: `127.0.0.1:12355`)
- `HEADWATER_PROFILE=1` to emit sampler profile logs at shutdown

Per-rank logs are written to `/tmp/confluence-<timestamp>/rank-*.log` by default.

## 3) Read the sampler profile logs

With `HEADWATER_PROFILE=1`, the sampler prints:

- batch phase timing (`phase1`, `text_dedup`, `text_gather`, `collate`)
- queue behavior (`producer_blocked_*`, `consumer_blocked_*`)
- produced/consumed train/val counts

Interpretation:

- High `consumer_blocked_avg_ms`: training is starving for samples.
- High `producer_blocked_avg_ms`: sampler is overproducing and waiting on full queue.
- High `phase1_ms`: BFS/sequence construction is the dominant bottleneck.
