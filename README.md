# tributary

A foundation model for learning internal representations of relational databases.

## Overview

Tributary is a system that is design to train models against relational databases such
that the underlying model, through the process of masked-cell regression/classification,
learns internal representations of the database schema and the data itself.

This is a work in progress! I'm still figuring out the best way to do this.

## Representations

We first define the objects we're interested in studying.

First, the relational database.

A relational database is a collection of tables, some of which may be "joinable" to each
other through a primary key/foreign key relationship. We can represent this as a schema
graph, with directed edges between tables.

Each table is also a collection of rows, each of which has some columns. A particular
"cell" is identified by a (table, row, column) tuple. Cells can be "null" or "not null".

These columns in a table can have different "data types" - these are the primitive types
that the column is stored as - for example, in MySQL, this could be "INT", "FLOAT",
"VARCHAR", "BOOLEAN", etc.

However, columns ALSO have meaningful "semantic types" - these encode the meaning of the
column in the context of what it's attempting to represent. For example, a column that
holds a primary key might be an "Identifier" column, whereas a column that holds a date
might be a "Timestamp" column, and a column that holds a string might be a "Text" column
(when the contents are semantically meaningful) or a "Categorical" column (when the
contents do not have meaningful semantics, but serve as an enumeration of some kind.)

Our model supports the following semantic types:

- `Identifier`
- `Numerical`
- `Timestamp`
- `Boolean`
- `Categorical`
- `Text`

Every column that is not one of the above semantic types is considered "Ignored"
and is skipped entirely during preprocessing and training.

## Preprocessing

Assumption: the database comes to us in the form of a collection of parquet files, and a
special "metadata.json" file that has been human-annotated with some information about
the schema. We need information on the semantic types (see above) for each column in the
database, as well as some information about the signal columns that are worth masking
and predicting.

There is a bit of work done here at preprocessing time - we construct a graph
representation of the rows in the database (bidirectional CSR graph), and we also
encode each semantic type cell into a special value.

Identifiers are not encoded at all - merely "present" / "absent". The inductive bias
here is that there might be some signal in the presence / absence of an identifier, but
it's not clear what that signal is - the model will learn.

Numerical values are encoded as z-scored f32 values, with a validity bitmap (null / present).
The scores are normalized _per-column_ for numerical values.

Timestamp values are cyclically encoded, with a validity bitmap (null / present).

- second of minute
- minute of hour
- hour of day
- day of week
- day of month
- month of year
- day of year

The timestamp itself (i64 microseconds since epoch) is also part of the feature, but it's
z-score normalized to an f32 based on all timestamp values across all tables in the database.

Boolean values are encoded as 0 / 1 values, with a validity bitmap (null / present).

Categorical values are encoded as an _index_ into an embedding table, for the string
"column name is X". For example, if the column name is "color", and the value is "red",
we use a frozen text embedder to embed "color is red", put that embedding into a lookup
table, and use the index of that embedding in the binary representation.

Text values are similarly encoded - we use a frozen text embedder for non-identifier
(semantically meaningful) text values.

## Training

We train the model using a masked language model (MLM) objective.

For numeric and timestamp values, we use Huber regression loss.
For boolean values, we use binary cross-entropy loss.
For categorical values, we use InfoNCE loss

## Prerequisites

TODO(mrdmnd): Add prerequisites.

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
uv run scripts/export_onnx.py
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
  common.rs         — shared types, graph structures, binary format I/O
  embedder.rs       — ORT-backed GPU embedding (tokenization, IoBinding, pipelining)
  model.rs          — transformer model definition (WIP)
  training.rs       — training configuration (WIP)
  bin/
    preprocess.rs   — data preprocessing binary
    inspect.rs      — preprocessed database inspector / debugger
    train.rs        — model training binary (WIP)
benches/
  embedder_throughput.rs — Criterion benchmarks (chunk-size sweep + mixed-length)
ort/
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
