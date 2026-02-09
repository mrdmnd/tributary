# tributary
A foundation model transformer for Relational Databases

OPINIONATED: Assumes a modern (Blackwell+) NVIDIA setup with CUDA 13 or better.


## Prerequisites

- **BLACKWELL GPU** with CUDA 13+ drivers
- **[uv](https://docs.astral.sh/uv/)** (only needed for the ONNX export script)


## Embedding

The embedder uses ONNX Runtime with CUDA EP for GPU-accelerated text embeddings.

### Setup

1. **Download the ONNX Runtime CUDA 13 binary:**

```bash
mkdir -p ort/ort-libs && cd ort/ort-libs
curl -L -o ort-cuda13.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.24.1/onnxruntime-linux-x64-gpu_cuda13-1.24.1.tgz
tar xzf ort-cuda13.tgz
cd ../..
```

The library path is configured automatically via `.cargo/config.toml`.

2. **Export the ONNX model:**

```bash
uv run ort/scripts/export_onnx.py
```

This creates `ort/models/bge-base-en-v1.5-onnx/model_fp16.onnx` (~208 MB).

3. **Build:**

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
