# tributary
A foundation model transformer for Relational Databases

OPINIONATED: Assumes a modern (Blackwell+) NVIDIA setup with CUDA 13 or better.


## Prerequisites

- **BLACKWELL GPU** with CUDA 13+ drivers
- **[uv](https://docs.astral.sh/uv/)** (for running the ONNX export script)


## Setup

The database preprocessing binary uses a high-performance local embedding library based on ONNX and BGE-base-en-v1.5.

You need to set this up:

### 1. Download the ONNX Runtime CUDA 13 binary

```bash
mkdir -p ort/ort-libs && cd ort/ort-libs
curl -L -o ort-cuda13.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.24.1/onnxruntime-linux-x64-gpu_cuda13-1.24.1.tgz
tar xzf ort-cuda13.tgz
cd ../..
```

The library path is configured automatically via `.cargo/config.toml` — no
environment variables needed.

### 2. Export the ONNX model

The embedding model (BGE-base-en-v1.5) must be exported to an optimized FP16
ONNX format before use. This applies BERT-specific graph fusions (fused
attention, layer norm, etc.) and FP16 conversion for tensor core acceleration.

```bash
uv run ort/scripts/export_onnx.py
```

This creates `ort/models/bge-base-en-v1.5-onnx/model_fp16.onnx` (~208 MB).

### 3. Build and run

```bash
cargo build --release
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
