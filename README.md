# tributary

A foundation model transformer for relational databases.

Assumes a modern (Blackwell+) NVIDIA GPU with CUDA 13 drivers.

## Prerequisites

- **Blackwell GPU** with CUDA 13+ drivers
- **[uv](https://docs.astral.sh/uv/)** — runs the ONNX export script
- **lld** — used as the linker (set in `.cargo/config.toml`)

## Setup

### 1. Download ONNX Runtime

The project dynamically loads ONNX Runtime with the CUDA execution provider.
Download the pre-built binary and extract it:

```bash
mkdir -p ort/ort-libs && cd ort/ort-libs
curl -L -o ort-cuda13.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.24.1/onnxruntime-linux-x64-gpu_cuda13-1.24.1.tgz
tar xzf ort-cuda13.tgz
cd ../..
```

The library path is configured automatically via `.cargo/config.toml`.

### 2. Export the ONNX model

```bash
uv run ort/scripts/export_onnx.py
```

This runs a four-step pipeline (export, BERT graph fusion, FP16 conversion,
post-processing) and produces `ort/models/bge-base-en-v1.5-onnx/model_fp16_pooled.onnx`
(~105 MB). See the script's docstring for details.

The final model includes:

- **BERT-specific graph fusions** — `Attention`, `BiasGelu`,
  `SkipLayerNormalization`, `EmbedLayerNormalization`.
- **FP16 weights and activations** — enables tensor-core acceleration.
- **INT32 inputs** — halves host-to-device transfer size vs the default INT64.
- **Fused FP16 mean-pooling + L2 normalisation** — output is `[B, 768]` FP32.
  Uses only CUDA-friendly primitives to avoid a CPU fallback on
  `LpNormalization`.

### 3. Build

```bash
cargo build --release
```

## Project structure

```
src/
  lib.rs            — crate root (mimalloc global allocator)
  embedder.rs       — ORT-backed GPU embedding (tokenization, IoBinding, pipelining)
  types.rs          — shared types (placeholder)
  utils.rs          — shared utilities (placeholder)
  bin/
    preprocess.rs   — data preprocessing binary (WIP)
    train.rs        — model training binary (WIP)
benches/
  embedder_throughput.rs — Criterion benchmarks (chunk-size sweep + mixed-length)
ort/
  scripts/
    export_onnx.py  — ONNX export & optimisation pipeline
  ort-libs/         — ONNX Runtime shared library (gitignored)
  models/           — exported ONNX models (gitignored)
```

## Benchmarking

```bash
cargo bench --bench embedder_throughput
```

For reproducible results, lock GPU clocks before benchmarking:

```bash
# Lock clocks (pick a frequency your GPU supports):
sudo nvidia-smi -lgc <freq>,<freq>

cargo bench --bench embedder_throughput

# Reset when done:
sudo nvidia-smi -rgc
```
