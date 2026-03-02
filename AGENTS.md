# Instructions for LLM Programming Agents

This repository contains code designed to build a "relational transformer" machine learning model.
There are two pieces. One is written in Rust, and the other in Python.
You can see lots of documentation in the `documentation` directory - read pieces of it as needed.

When writing markdown files, stick to one short and concise sentence per line.

## Components

### Metadata Generation Python Script

`generate_metadata.py` is a Python script that uses an LLM agent to process a collection of parquet files.
The intention is it builds a draft metadata file that can be fixed by a human.
This metadata file is used by the `headwater` preprocessor to generate a binary format for our sampler to load.

### Rust (`headwater`)

Headwater is a Rust library with two responsibilities:

1. Preprocessing databases into graphs + materializing tasks

The preprocessor converts parquet files + metadata JSON file into a custom processed binary format for our sampler.
You can see the file at `headwater/src/bin/preprocess.rs`.

1. Sampling from processed databases into batches

The sampler runs in a single Rust process and produces batches of sequences of training (and validation) cells.
It is callable from Python code via PyO3, via Maturin.
See `headwater/src/sampler.rs` for the BFS-based sampling, batch packing, prefetch pipeline, and DDP sharding.
See `headwater/src/python_bindings.rs` for the PyO3 bindings.

### Python (`confluence`)

Confluence is the main training JAX python project.
It defines a custom relational transformer model and uses the sampler from`headwater` to feed batches in.

The model, training loop, loss, and optimizer are all in `confluence/`.
Confluence is intended to be run in a DistributedDataParallel way - multiple GPU nodes per host.

## Development

### General Style

- In particular, functions should generally be no longer than about 60 lines (one screen of text).
- Use a hard cutover approach and never implement backward compatibility.

### Environmental Assumptions

You can assume the machine you are running on is an x86-64 machine with many CPU cores that are SIMD-capable, and at least one GPU.
In local testing, you're probably being run on a 1x 5090 Blackwell.
In a production training scenarios you're probably being run on an 8x B200 node.

You can ALWAYS ASSUME that you are running in an environment with CUDA support; no need for CPU fallbacks anywhere.

### Rust Guidelines

This project uses cargo, like a normal rust project.
Binaries should be built and compiled in --release mode.

Use `cargo fmt` for consistent code formatting; `cargo clippy` to ensure lints and idiomatic suggestions are followed.
Write tests and use `cargo test` where appropriate.

Avoid anti-patterns like `unwrap`.

Where possible, use the newtypes defined in `headwater/src/common.rs` to represent indices and other values.
The existence of "as" and ".0" are usually code smells; try to find these and avoid them.

Use tokio-rs `tracing` for logging instead or print statements.

### Python Guidelines

The repository root is a **uv workspace** and the `confluence` Python package.
The workspace root `pyproject.toml` contains both the package definition and shared tool configuration (ruff).
`headwater` is a workspace member whose Python bindings are built via maturin.
A single `uv.lock` at the root manages dependency resolution.

- **`confluence/`** — the importable Python package (model, training loop, loss, optimizer, config).
- **`headwater/`** — the Rust crate's Python bindings, built by maturin as a workspace member.

Run training with `uv run train` (entry point defined in `[project.scripts]`).

Use `ty` for typechecking.
Use `deptry` to check for dependency updates, and ensure you're including what you use.
Use `pytest` for testing.
Use `ruff` for linting/formatting (config lives in the root `pyproject.toml`).

You should always ensure that functions are no longer than about 60 lines (one screen of text).
You should always include strong type hints for all functions and variables.
You should _especially_ be sure to use the `jaxtyping` library to put tensor shape types in place.

For example:

```python
from jaxtyping import Float

def my_function(x: Float[Tensor, "N M"]) -> Float[Tensor, "N M"]:
    return x + 1
```

When done making changes, always run the git precommit hooks and fix any issues.
You can do this by running `pre-commit` from the project root.
