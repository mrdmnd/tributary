# Instructions for LLM Programming Agents

This repository contains code designed to build a "relational transformer" machine learning model.

There are two main pieces. One is written in rust, and the other is written in python.

You can see lots of documentation in the `documentation` directory - read pieces of it as needed.

## Components

### Rust (`headwater`)

Headwater is a rust library with two responsibilities: 

1) preprocessing databases into graphs + materializing tasks

The preprocessor converts collections of parquet files plus a metadata JSON file describing database semantics into a
custom processed binary format for our samplers to load.

2) sampling from processed databases into batches

The sampler runs in a single rust process. It is responsible for producing sequences of training cells from our 
pre-processed graphs. The sampler both *constructs the sequences* as well as *packs the batches* for downstream use.
The sampler is callable from python code via PyO3.


### Python (`confluence`)

Confluence (the model) is a JAX python project.
We define a custom relational transformer model, and use the sampler from `headwater` to feed batches in.

Confluence is intended to be run in a DistributedDataParallel way - mutiple GPU nodes per host.

## Development

### General Style

- In particular, functions should generally be no longer than about 60 lines (one screen of text).

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

This project uses `uv` for package management. All dependencies should be listed in `pyproject.toml`.
If you want to do ANY one-off python tasks, you can use `uv run <script>`.