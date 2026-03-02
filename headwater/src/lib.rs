use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod batch;
pub mod common;
pub mod embedder;
pub mod sampler;
pub mod seed_manager;

#[cfg(feature = "python")]
mod python_bindings;
