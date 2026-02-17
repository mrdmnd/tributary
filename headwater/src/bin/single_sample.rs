//! Generate a single batch from a preprocessed database and print summary stats.
//!
//! Usage:
//!   cargo run --release --bin single_sample -- --db-dir data/processed/rel-stack

use std::path::PathBuf;

use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use headwater::sampler::{Sampler, SamplerConfig};

#[derive(Parser, Debug)]
#[command(about = "Generate a single sample batch from a preprocessed database")]
struct Args {
    /// Path to the preprocessed database directory.
    #[arg(long)]
    db_dir: PathBuf,

    /// Batch size (number of sequences per batch).
    #[arg(long, default_value = "4")]
    batch_size: u32,

    /// Sequence length (cells per sequence).
    #[arg(long, default_value = "256")]
    sequence_length: u32,

    /// Max children per P->F edge during BFS.
    #[arg(long, default_value = "16")]
    bfs_child_width: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    info!("Loading database from: {}", args.db_dir.display());

    let config = SamplerConfig {
        db_path: args.db_dir.to_string_lossy().to_string(),
        rank: 0,
        world_size: 1,
        split_ratios: (0.8, 0.1, 0.1),
        split_seed: 123,
        seed: 42,
        num_prefetch: 1,
        batch_size: args.batch_size,
        sequence_length: args.sequence_length,
        bfs_child_width: args.bfs_child_width,
        max_rows_per_seq: 200,
    };

    let mut sampler = Sampler::new(config)?;
    let db = sampler.database();

    info!("Database loaded:");
    info!("  Tables: {}", db.metadata.table_metadata.len());
    info!("  Columns: {}", db.metadata.column_metadata.len());
    info!("  Tasks: {}", db.metadata.task_metadata.len());
    for (i, tm) in db.metadata.task_metadata.iter().enumerate() {
        info!(
            "  Task {}: {} (anchor={}, target_stype={:?}, seeds={})",
            i,
            tm.name,
            db.metadata.table_metadata[tm.anchor_table.0 as usize].name,
            tm.target_stype,
            tm.num_seeds
        );
    }

    info!("\nPulling a train batch...");
    let batch = sampler.next_train_batch()?;

    info!("Batch summary:");
    info!("  batch_size (B): {}", batch.batch_size);
    info!("  sequence_length (S): {}", batch.sequence_length);
    info!("  max_rows (R): {}", batch.max_rows);
    info!("  task_idx: {}", batch.task_idx);
    info!(
        "  task_name: {}",
        db.metadata.task_metadata[batch.task_idx as usize].name
    );
    info!("  target_stype: {:?}", batch.target_stype);
    info!("  num_unique_texts (U): {}", batch.num_unique_texts);

    // Per-sequence stats
    let b = batch.batch_size;
    let s = batch.sequence_length;
    for bi in 0..b {
        let offset = bi * s;
        let non_padding: usize = (0..s)
            .filter(|&j| batch.is_padding[offset + j] == 0)
            .count();
        let num_targets: usize = (0..s).filter(|&j| batch.is_target[offset + j] == 1).count();
        let num_nulls: usize = (0..s)
            .filter(|&j| batch.is_null[offset + j] == 1 && batch.is_padding[offset + j] == 0)
            .count();

        // Count distinct rows
        let distinct_rows: std::collections::HashSet<u16> = (0..s)
            .filter(|&j| batch.is_padding[offset + j] == 0)
            .map(|j| batch.seq_row_ids[offset + j])
            .collect();

        // Count FK edges in adjacency
        let r = batch.max_rows;
        let adj_offset = bi * r * r;
        let num_edges: usize = batch.fk_adj[adj_offset..adj_offset + r * r]
            .iter()
            .filter(|&&v| v == 1)
            .count();

        info!(
            "  seq[{}]: {} cells, {} rows, {} edges, {} targets, {} nulls",
            bi,
            non_padding,
            distinct_rows.len(),
            num_edges,
            num_targets,
            num_nulls
        );
    }

    // Tensor shape summary
    info!("\nTensor shapes:");
    info!(
        "  semantic_types:        [{}, {}] = {} elements",
        b,
        s,
        batch.semantic_types.len()
    );
    info!(
        "  column_ids:            [{}, {}] = {} elements",
        b,
        s,
        batch.column_ids.len()
    );
    info!(
        "  numeric_values:        [{}, {}] = {} elements",
        b,
        s,
        batch.numeric_values.len()
    );
    info!(
        "  timestamp_values:      [{}, {}, {}] = {} elements",
        b,
        s,
        headwater::common::TIMESTAMP_DIM,
        batch.timestamp_values.len()
    );
    info!(
        "  fk_adj:                [{}, {}, {}] = {} elements",
        b,
        batch.max_rows,
        batch.max_rows,
        batch.fk_adj.len()
    );
    info!(
        "  text_batch_embeddings: [{}, {}] = {} elements",
        batch.num_unique_texts,
        headwater::embedder::EMBEDDING_DIM,
        batch.text_batch_embeddings.len()
    );

    info!("\nPulling a val batch...");
    let val_batch = sampler.next_val_batch()?;
    info!(
        "Val batch: task={}, {} cells non-padding in seq[0]",
        db.metadata.task_metadata[val_batch.task_idx as usize].name,
        (0..val_batch.sequence_length)
            .filter(|&j| val_batch.is_padding[j] == 0)
            .count()
    );

    info!("\nShutting down sampler...");
    sampler.shutdown();
    info!("Done!");

    Ok(())
}
