use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod common;
pub mod embedder;
pub mod model;
pub mod training;
