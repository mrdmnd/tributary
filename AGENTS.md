# Instructions for LLM Programming Agents

This repository contains code designed to build a "relational transformer" machine learning model.

There are two pieces. One is written in Rust, and the other in Python.

You can see lots of documentation in the `documentation` directory - read pieces of it as needed.

## Components

### Rust (`headwater`)

Headwater is a Rust library with two responsibilities:

1) preprocessing databases into graphs + materializing tasks

The preprocessor converts collections of parquet files plus a metadata JSON file describing database semantics into a
custom processed binary format for our samplers to load. **This is implemented.**

2) sampling from processed databases into batches

The sampler runs in a single Rust process. It produces sequences of training cells from our pre-processed graphs. The
sampler both *constructs the sequences* and *packs the batches* for downstream use. It is callable from Python code via
PyO3. **This is implemented.** See `src/sampler.rs` for the BFS-based sampling, batch packing, prefetch pipeline, and
DDP sharding. See `src/python.rs` for the PyO3 bindings.


### Python (`confluence`)

Confluence is a JAX python project. It defines a custom relational transformer model and uses the sampler from
`headwater` to feed batches in. **This is implemented.** The model, training loop, loss, and optimizer are all in
`confluence/confluence/`. A standalone smoke test (`confluence/smoke_test.py`) validates the pipeline end-to-end without
the Rust sampler.

Confluence is intended to be run in a DistributedDataParallel way - multiple GPU nodes per host.

## Development

### General Style

- In particular, functions should generally be no longer than about 60 lines (one screen of text).
- Use a hard cutover approach and never implement backward compatibility.

### Environmental Assumptions

The machine you are running on is an x86-64 machine with many CPU cores that are SIMD-capable, and at least one GPU
acceleration device. In local testing, you're probably being run on a 1x 5090 Blackwell; in real production training
scenarios you're probably being run on an 8x B200 node.

You can ALWAYS ASSUME that you are running in an environment with CUDA support; no need for CPU fallbacks anywhere.

### Rust (`headwater`)

This project uses cargo, like a normal rust project.
Binaries should be built and compiled in --release mode. 

Use `cargo fmt` for consistent code formatting; `cargo clippy` to ensure lints and idiomatic suggestions are followed;
write tests and use `cargo test` where appropriate.

Avoid anti-patterns like `unwrap`.

Use tokio-rs `tracing` for logging instead or print statements.

### Python (`confluence`)

This project uses `uv` for package management with dependencies listed in `pyproject.toml`. The `headwater` Rust crate
is a local editable dependency (built via maturin). Use `ruff` for linting/formatting.

### Python (`scripts`)

Helper scripts live in the `scripts/` directory with their own `pyproject.toml`.
If you want to do ANY one-off python tasks, you can use `uv run --project scripts <script>`.