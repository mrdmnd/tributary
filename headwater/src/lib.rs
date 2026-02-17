use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod common;
pub mod embedder;
pub mod sampler;

#[cfg(feature = "python")]
mod python;
