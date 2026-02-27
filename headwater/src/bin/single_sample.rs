//! Generate a single batch from a preprocessed database and print summary stats.
//!
//! Usage:
//!   cargo run --release --bin single_sample -- --db-dir data/processed/rel-stack

use std::path::PathBuf;

use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use headwater::sampler::{RawBatch, Sampler, SamplerConfig};

#[derive(Parser, Debug)]
#[command(about = "Generate a single sample batch of one sequence from a preprocessed database")]
struct Args {
    /// Path to the preprocessed database directory.
    #[arg(long)]
    db_dir: PathBuf,

    /// Sequence length (cells per sequence).
    #[arg(long, default_value = "64")]
    sequence_length: u32,

    /// Max children per P->F edge during BFS.
    #[arg(long, default_value = "16")]
    bfs_child_width: u32,

    /// Tile width/height for coarse sparsity visualization.
    #[arg(long, default_value = "32")]
    tile_size: u32,
}

fn build_attention_masks_for_sequence(
    batch: &RawBatch,
    seq_idx: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let s = batch.sequence_length;
    let r = batch.max_rows;
    let seq_offset = seq_idx * s;
    let adj_offset = seq_idx * r * r;
    let fk_adj = &batch.fk_adj[adj_offset..adj_offset + r * r];

    let mut outbound = vec![0u8; s * s];
    let mut inbound = vec![0u8; s * s];
    let mut column = vec![0u8; s * s];

    for i in 0..s {
        for j in 0..s {
            let idx = i * s + j;
            let i_pad = batch.is_padding[seq_offset + i] == 1;
            let j_pad = batch.is_padding[seq_offset + j] == 1;
            let valid = !i_pad && !j_pad;

            let row_i = batch.seq_row_ids[seq_offset + i] as usize;
            let row_j = batch.seq_row_ids[seq_offset + j] as usize;
            let same_row = row_i == row_j;
            let fk_ij = row_i < r && row_j < r && fk_adj[row_i * r + row_j] == 1;
            let fk_ji = row_i < r && row_j < r && fk_adj[row_j * r + row_i] == 1;
            let same_col = batch.column_ids[seq_offset + i] == batch.column_ids[seq_offset + j];

            let out = valid && (same_row || fk_ij);
            let inn = valid && fk_ji;
            let col = valid && same_col;

            outbound[idx] = if out { 1 } else { 0 };
            inbound[idx] = if inn { 1 } else { 0 };
            column[idx] = if col { 1 } else { 0 };
        }
    }

    (outbound, inbound, column)
}

fn print_tile_separator(side: usize, tile: usize) {
    let mut line = String::new();
    for j in 0..side {
        if j % tile == 0 {
            line.push('+');
        }
        line.push('-');
    }
    line.push('+');
    println!("{line}");
}

fn print_mask_ascii(name: &str, mask: &[u8], stride: usize, side: usize, tile: usize) {
    println!("\n{name} ({side}x{side}, tile={tile})");
    print_tile_separator(side, tile);
    for i in 0..side {
        if i > 0 && i % tile == 0 {
            print_tile_separator(side, tile);
        }
        let mut row = String::with_capacity(side);
        for j in 0..side {
            if j % tile == 0 {
                row.push('|');
            }
            let ch = if mask[i * stride + j] == 1 { '#' } else { '.' };
            row.push(ch);
        }
        row.push('|');
        println!("{row}");
    }
    print_tile_separator(side, tile);
}

fn print_permuted_mask_ascii(
    name: &str,
    mask: &[u8],
    stride: usize,
    perm: &[u16],
    side: usize,
    tile: usize,
) {
    println!("\n{name} ({side}x{side}, tile={tile})");
    print_tile_separator(side, tile);
    for i in 0..side {
        if i > 0 && i % tile == 0 {
            print_tile_separator(side, tile);
        }
        let mut row = String::with_capacity(side);
        let pi = perm[i] as usize;
        for j in 0..side {
            if j % tile == 0 {
                row.push('|');
            }
            let pj = perm[j] as usize;
            let ch = if mask[pi * stride + pj] == 1 { '#' } else { '.' };
            row.push(ch);
        }
        row.push('|');
        println!("{row}");
    }
    print_tile_separator(side, tile);
}

fn tile_density_char(enabled: usize, total: usize) -> char {
    if enabled == 0 {
        return '.';
    }
    let density = enabled as f32 / total as f32;
    if density < 0.25 {
        '+'
    } else if density < 0.75 {
        '*'
    } else {
        '#'
    }
}

fn print_tiled_mask_ascii(name: &str, mask: &[u8], stride: usize, side: usize, tile: usize) {
    let tiles = side.div_ceil(tile);
    println!("\n{name} tiled ({tiles}x{tiles} tiles, tile={tile})");
    for ti in 0..tiles {
        let mut row = String::with_capacity(tiles);
        let i0 = ti * tile;
        let i1 = (i0 + tile).min(side);
        for tj in 0..tiles {
            let j0 = tj * tile;
            let j1 = (j0 + tile).min(side);
            let mut enabled = 0usize;
            for i in i0..i1 {
                for j in j0..j1 {
                    if mask[i * stride + j] == 1 {
                        enabled += 1;
                    }
                }
            }
            let total = (i1 - i0) * (j1 - j0);
            row.push(tile_density_char(enabled, total));
        }
        println!("{row}");
    }
}

fn print_tiled_permuted_mask_ascii(
    name: &str,
    mask: &[u8],
    stride: usize,
    perm: &[u16],
    side: usize,
    tile: usize,
) {
    let tiles = side.div_ceil(tile);
    println!("\n{name} tiled ({tiles}x{tiles} tiles, tile={tile})");
    for ti in 0..tiles {
        let mut row = String::with_capacity(tiles);
        let i0 = ti * tile;
        let i1 = (i0 + tile).min(side);
        for tj in 0..tiles {
            let j0 = tj * tile;
            let j1 = (j0 + tile).min(side);
            let mut enabled = 0usize;
            for i in i0..i1 {
                let pi = perm[i] as usize;
                for j in j0..j1 {
                    let pj = perm[j] as usize;
                    if mask[pi * stride + pj] == 1 {
                        enabled += 1;
                    }
                }
            }
            let total = (i1 - i0) * (j1 - j0);
            row.push(tile_density_char(enabled, total));
        }
        println!("{row}");
    }
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
        batch_size: 1,
        sequence_length: args.sequence_length,
        bfs_child_width: args.bfs_child_width,
        max_rows_per_seq: 200,
        perm_tile_size: args.tile_size,
        perm_stats_every: 1,
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

    let s = batch.sequence_length;
    let mask_side = s.min(256);
    let tile_size = (args.tile_size as usize).max(1).min(mask_side);
    let (outbound_mask, inbound_mask, column_mask) = build_attention_masks_for_sequence(&batch, 0);
    print_mask_ascii(
        "Outbound attention mask",
        &outbound_mask,
        s,
        mask_side,
        tile_size,
    );
    print_mask_ascii(
        "Inbound attention mask",
        &inbound_mask,
        s,
        mask_side,
        tile_size,
    );
    print_mask_ascii(
        "Column attention mask",
        &column_mask,
        s,
        mask_side,
        tile_size,
    );
    print_tiled_mask_ascii(
        "Outbound attention mask",
        &outbound_mask,
        s,
        mask_side,
        tile_size,
    );
    print_tiled_mask_ascii(
        "Inbound attention mask",
        &inbound_mask,
        s,
        mask_side,
        tile_size,
    );
    print_tiled_mask_ascii(
        "Column attention mask",
        &column_mask,
        s,
        mask_side,
        tile_size,
    );

    let seq_offset = 0;
    let out_perm = &batch.out_perm[seq_offset..seq_offset + s];
    let in_perm = &batch.in_perm[seq_offset..seq_offset + s];
    let col_perm = &batch.col_perm[seq_offset..seq_offset + s];
    print_permuted_mask_ascii(
        "Outbound attention mask (after out_perm)",
        &outbound_mask,
        s,
        out_perm,
        mask_side,
        tile_size,
    );
    print_permuted_mask_ascii(
        "Inbound attention mask (after in_perm)",
        &inbound_mask,
        s,
        in_perm,
        mask_side,
        tile_size,
    );
    print_permuted_mask_ascii(
        "Column attention mask (after col_perm)",
        &column_mask,
        s,
        col_perm,
        mask_side,
        tile_size,
    );
    print_tiled_permuted_mask_ascii(
        "Outbound attention mask (after out_perm)",
        &outbound_mask,
        s,
        out_perm,
        mask_side,
        tile_size,
    );
    print_tiled_permuted_mask_ascii(
        "Inbound attention mask (after in_perm)",
        &inbound_mask,
        s,
        in_perm,
        mask_side,
        tile_size,
    );
    print_tiled_permuted_mask_ascii(
        "Column attention mask (after col_perm)",
        &column_mask,
        s,
        col_perm,
        mask_side,
        tile_size,
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
