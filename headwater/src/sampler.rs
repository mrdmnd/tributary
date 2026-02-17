//! BFS-based relational database sampler for training batch construction.
//!
//! The sampler walks FK edges outward from task seed rows, collects cells into
//! sequences, and packs them into dense batches for the relational transformer.
//!
//! See `documentation/sampling.md` and `documentation/system_architecture.md`
//! for the full design.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam::channel::{self, Receiver, Sender};
use half::f16;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use tracing::{debug, info, warn};

use crate::common::{Database, RowIdx, SemanticType, TIMESTAMP_DIM, TaskIdx};
use crate::embedder::EMBEDDING_DIM;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the sampler.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Path to the preprocessed database directory.
    pub db_path: String,
    /// This process's rank in the DDP group.
    pub rank: u32,
    /// Total number of DDP processes.
    pub world_size: u32,
    /// Train/val/test split ratios (must sum to 1.0).
    pub split_ratios: (f32, f32, f32),
    /// Hash seed for deterministic split assignment.
    pub split_seed: u64,
    /// RNG seed for sampling randomness.
    pub seed: u64,
    /// Number of prefetch batches per channel.
    pub num_prefetch: usize,
    /// Batch size B (number of subgraphs/seeds per batch).
    pub batch_size: u32,
    /// Sequence length S (cells per sequence, padded/truncated).
    pub sequence_length: u32,
    /// Max children sampled per P->F edge during BFS.
    pub bfs_child_width: u32,
    /// Max distinct rows per sequence (R dimension for fk_adj).
    pub max_rows_per_seq: u32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            db_path: String::new(),
            rank: 0,
            world_size: 1,
            split_ratios: (0.8, 0.1, 0.1),
            split_seed: 123,
            seed: 42,
            num_prefetch: 3,
            batch_size: 32,
            sequence_length: 1024,
            bfs_child_width: 16,
            max_rows_per_seq: 200,
        }
    }
}

// ============================================================================
// Split Assignment
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Val,
    Test,
}

/// Deterministic hash-based split assignment.
fn assign_split(
    task_idx: u32,
    anchor_row: u32,
    split_seed: u64,
    train_ratio: f32,
    val_ratio: f32,
) -> Split {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};
    task_idx.hash(&mut hasher);
    anchor_row.hash(&mut hasher);
    split_seed.hash(&mut hasher);
    let bucket = (hasher.finish() % 1000) as f32;
    let train_thresh = train_ratio * 1000.0;
    let val_thresh = (train_ratio + val_ratio) * 1000.0;
    if bucket < train_thresh {
        Split::Train
    } else if bucket < val_thresh {
        Split::Val
    } else {
        Split::Test
    }
}

// ============================================================================
// Seed Manager
// ============================================================================

/// Manages per-task seed lists for each split, sharded by rank.
struct SeedManager {
    /// For each task, the seed indices assigned to this rank for train split.
    train_seeds: Vec<Vec<usize>>,
    /// For each task, the seed indices assigned to this rank for val split.
    val_seeds: Vec<Vec<usize>>,
    /// For each task, the seed indices assigned to this rank for test split.
    #[allow(dead_code)]
    test_seeds: Vec<Vec<usize>>,
}

impl SeedManager {
    fn new(db: &Database, config: &SamplerConfig) -> Self {
        let num_tasks = db.metadata.task_metadata.len();
        let mut train_seeds = vec![Vec::new(); num_tasks];
        let mut val_seeds = vec![Vec::new(); num_tasks];
        let mut test_seeds = vec![Vec::new(); num_tasks];

        for ti in 0..num_tasks {
            let task_view = db.task(TaskIdx(ti as u32));
            let num_seeds = task_view.num_seeds();
            let mut train_for_task = Vec::new();
            let mut val_for_task = Vec::new();
            let mut test_for_task = Vec::new();

            for seed_idx in 0..num_seeds {
                let anchor_row = task_view.anchor_row(seed_idx);
                let split = assign_split(
                    ti as u32,
                    anchor_row.0,
                    config.split_seed,
                    config.split_ratios.0,
                    config.split_ratios.1,
                );
                // Round-robin sharding by rank
                match split {
                    Split::Train => train_for_task.push(seed_idx),
                    Split::Val => val_for_task.push(seed_idx),
                    Split::Test => test_for_task.push(seed_idx),
                }
            }

            // Shard by rank (round-robin)
            train_seeds[ti] = train_for_task
                .into_iter()
                .enumerate()
                .filter(|(i, _)| (*i as u32) % config.world_size == config.rank)
                .map(|(_, s)| s)
                .collect();
            val_seeds[ti] = val_for_task
                .into_iter()
                .enumerate()
                .filter(|(i, _)| (*i as u32) % config.world_size == config.rank)
                .map(|(_, s)| s)
                .collect();
            test_seeds[ti] = test_for_task
                .into_iter()
                .enumerate()
                .filter(|(i, _)| (*i as u32) % config.world_size == config.rank)
                .map(|(_, s)| s)
                .collect();
        }

        info!(
            "SeedManager: {} tasks, train seeds/rank: [{}], val seeds/rank: [{}]",
            num_tasks,
            train_seeds
                .iter()
                .map(|s| s.len())
                .collect::<Vec<_>>()
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            val_seeds
                .iter()
                .map(|s| s.len())
                .collect::<Vec<_>>()
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", "),
        );

        Self {
            train_seeds,
            val_seeds,
            test_seeds,
        }
    }

    fn seeds_for_split(&self, split: Split) -> &[Vec<usize>] {
        match split {
            Split::Train => &self.train_seeds,
            Split::Val => &self.val_seeds,
            Split::Test => &self.test_seeds,
        }
    }

    /// Total number of seeds across all tasks for a given split.
    fn total_seeds(&self, split: Split) -> usize {
        self.seeds_for_split(split).iter().map(|s| s.len()).sum()
    }
}

// ============================================================================
// Raw Batch (output of sampling)
// ============================================================================

/// A complete batch of B sequences, ready for transfer to Python/GPU.
///
/// All tensors are flat `Vec<T>` with shapes documented per field.
/// The Python binding converts these to NumPy arrays via zero-copy.
pub struct RawBatch {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub max_rows: usize,

    // Cell identity: [B, S]
    pub semantic_types: Vec<i8>,
    pub column_ids: Vec<i32>,
    pub seq_row_ids: Vec<u16>,

    // Per-type values: [B, S] or [B, S, 15]
    pub numeric_values: Vec<f32>,
    pub timestamp_values: Vec<f32>, // [B, S, 15]
    pub bool_values: Vec<u8>,
    pub categorical_embed_ids: Vec<u32>,
    pub text_embed_ids: Vec<u32>,

    // Masks: [B, S]
    pub is_null: Vec<u8>,
    pub is_target: Vec<u8>,
    pub is_padding: Vec<u8>,

    // Adjacency: [B, R, R]
    pub fk_adj: Vec<u8>,

    // Permutations: [B, S]
    pub col_perm: Vec<u16>,
    pub out_perm: Vec<u16>,
    pub in_perm: Vec<u16>,

    // Text embeddings: [U, EMBEDDING_DIM]
    pub text_batch_embeddings: Vec<f16>,
    pub num_unique_texts: usize,

    // Batch-level metadata
    pub target_stype: u8,
    pub task_idx: u32,
}

// ============================================================================
// BFS Frontier
// ============================================================================

/// Priority tag for BFS frontier entries.
/// F->P (parent) rows have higher priority than P->F (child) rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrontierPriority {
    /// Reached via F->P edge (parent). Always explored first.
    Parent,
    /// Reached via P->F edge (child). Explored after all parents.
    Child,
}

struct FrontierEntry {
    row: RowIdx,
    priority: FrontierPriority,
    #[allow(dead_code)]
    hops: u32,
}

// ============================================================================
// Per-Sequence Builder
// ============================================================================

/// Intermediate result of BFS + linearization for a single sequence.
#[allow(dead_code)]
struct SequenceData {
    num_cells: usize,
    num_rows: usize,

    // [S] arrays
    semantic_types: Vec<i8>,
    column_ids: Vec<i32>,
    seq_row_ids: Vec<u16>,
    numeric_values: Vec<f32>,
    timestamp_values: Vec<f32>, // [S * 15]
    bool_values: Vec<u8>,
    categorical_embed_ids: Vec<u32>,
    text_embed_ids: Vec<u32>, // global TextEmbeddingIdx temporarily
    is_null: Vec<u8>,
    is_target: Vec<u8>,
    is_padding: Vec<u8>,

    // [R, R] adjacency (R = num_rows)
    fk_adj: Vec<u8>,

    // [S] permutations
    col_perm: Vec<u16>,
    out_perm: Vec<u16>,
    in_perm: Vec<u16>,
}

/// Build a single sequence from a seed via BFS.
fn build_sequence(
    db: &Database,
    task_idx: TaskIdx,
    seed_idx: usize,
    config: &SamplerConfig,
    rng: &mut SmallRng,
) -> SequenceData {
    let s = config.sequence_length as usize;
    let max_r = config.max_rows_per_seq as usize;
    let _task_meta = db.task_metadata(task_idx);
    let task_view = db.task(task_idx);

    let seed_row = task_view.anchor_row(seed_idx);
    let obs_time = task_view.observation_time(seed_idx);

    // Allocate per-sequence buffers (initialized to zeros / padding)
    let mut semantic_types = vec![0i8; s];
    let mut column_ids = vec![0i32; s];
    let mut seq_row_ids = vec![0u16; s];
    let mut numeric_values = vec![0.0f32; s];
    let mut timestamp_values = vec![0.0f32; s * TIMESTAMP_DIM];
    let mut bool_values = vec![0u8; s];
    let mut categorical_embed_ids = vec![0u32; s];
    let mut text_embed_ids = vec![0u32; s];
    let mut is_null = vec![0u8; s];
    let mut is_target = vec![0u8; s];
    let mut is_padding = vec![1u8; s]; // start all padding, clear as we fill

    // Row tracking
    let mut row_map: HashMap<RowIdx, u16> = HashMap::new();
    let mut row_order: Vec<RowIdx> = Vec::new(); // insertion order for adjacency
    let mut cell_count: usize = 0;

    // BFS frontier
    let mut frontier: VecDeque<FrontierEntry> = VecDeque::new();
    let mut visited: std::collections::HashSet<RowIdx> = std::collections::HashSet::new();

    // Seed the frontier
    frontier.push_back(FrontierEntry {
        row: seed_row,
        priority: FrontierPriority::Parent,
        hops: 0,
    });

    while let Some(entry) = pop_next(&mut frontier) {
        if cell_count >= s || row_map.len() >= max_r {
            break;
        }
        if visited.contains(&entry.row) {
            continue;
        }
        visited.insert(entry.row);

        let row = entry.row;
        let table_idx = db.row_table(row);
        let table_meta = &db.metadata.table_metadata[table_idx.0 as usize];
        let table_view = db.table(table_idx);
        let local_row = (row.0 - table_meta.row_range.0.0) as usize;

        // Assign sequence-local row id
        let seq_row_id = row_map.len() as u16;
        row_map.insert(row, seq_row_id);
        row_order.push(row);

        // Visit all non-ignored columns of this row
        let col_start = table_meta.col_range.0.0;
        let col_end = table_meta.col_range.1.0;
        for global_col in col_start..col_end {
            if cell_count >= s {
                break;
            }
            let local_col = (global_col - col_start) as usize;
            let col_meta = &db.metadata.column_metadata[global_col as usize];
            if col_meta.stype == SemanticType::Ignored {
                continue;
            }

            let pos = cell_count;
            cell_count += 1;
            is_padding[pos] = 0;

            semantic_types[pos] = col_meta.stype as i8;
            column_ids[pos] = global_col as i32;
            seq_row_ids[pos] = seq_row_id;

            let cell_is_null = match col_meta.stype {
                SemanticType::Identifier => false,
                _ => table_view.is_null(local_col, local_row),
            };
            is_null[pos] = cell_is_null as u8;

            // Write value based on semantic type
            if !cell_is_null {
                match col_meta.stype {
                    SemanticType::Identifier => {
                        // No value to write - presence is the signal
                    }
                    SemanticType::Numerical => {
                        numeric_values[pos] = table_view.numerical(local_col, local_row);
                    }
                    SemanticType::Timestamp => {
                        let ts = table_view.timestamp(local_col, local_row);
                        let start = pos * TIMESTAMP_DIM;
                        timestamp_values[start..start + TIMESTAMP_DIM].copy_from_slice(ts);
                    }
                    SemanticType::Boolean => {
                        bool_values[pos] = table_view.boolean(local_col, local_row) as u8;
                    }
                    SemanticType::Categorical => {
                        categorical_embed_ids[pos] =
                            table_view.categorical_embedding_idx(local_col, local_row).0;
                    }
                    SemanticType::Text => {
                        text_embed_ids[pos] = table_view.text_embedding_idx(local_col, local_row).0;
                    }
                    SemanticType::Ignored => unreachable!(),
                }
            }
        }

        // Add FK neighbors to frontier
        // Outgoing edges (F->P): this row's FK columns point to parent rows
        for &neighbor_raw in db.graph.outgoing_neighbors(row) {
            let neighbor = RowIdx(neighbor_raw);
            if !visited.contains(&neighbor) {
                frontier.push_back(FrontierEntry {
                    row: neighbor,
                    priority: FrontierPriority::Parent,
                    hops: entry.hops + 1,
                });
            }
        }

        // Incoming edges (P->F): other rows' FK columns point to this row (children)
        let children: Vec<u32> = db.graph.incoming_neighbors(row).to_vec();
        let mut eligible: Vec<RowIdx> = Vec::new();
        for &child_raw in &children {
            let child_row = RowIdx(child_raw);
            if visited.contains(&child_row) {
                continue;
            }
            // Temporal filtering: if the child's table has a temporal column,
            // exclude rows with timestamp after obs_time
            if obs_time != i64::MAX {
                let child_table_idx = db.row_table(child_row);
                let child_table_meta = &db.metadata.table_metadata[child_table_idx.0 as usize];
                if let Some(tc) = child_table_meta.temporal_col {
                    let child_local_row = (child_row.0 - child_table_meta.row_range.0.0) as usize;
                    let child_table_view = db.table(child_table_idx);
                    let tc_local = (tc.0 - child_table_meta.col_range.0.0) as usize;
                    if !child_table_view.is_null(tc_local, child_local_row) {
                        let z = child_table_view.timestamp_zscore_epoch(tc_local, child_local_row);
                        let raw_us = (z as f64 * db.metadata.global_ts_std_us
                            + db.metadata.global_ts_mean_us)
                            as i64;
                        if raw_us > obs_time {
                            continue; // Future row, skip
                        }
                    }
                }
            }
            eligible.push(child_row);
        }

        // Subsample children to bfs_child_width
        if eligible.len() > config.bfs_child_width as usize {
            eligible.shuffle(rng);
            eligible.truncate(config.bfs_child_width as usize);
        }
        for child_row in eligible {
            frontier.push_back(FrontierEntry {
                row: child_row,
                priority: FrontierPriority::Child,
                hops: entry.hops + 1,
            });
        }
    }

    // Mark target cell: find the target column on the seed row
    // For the seed row, we need to mark the appropriate cell as target
    // and overwrite its value with the ground-truth from the task
    mark_target(
        &mut is_target,
        &mut is_null,
        &mut numeric_values,
        &mut timestamp_values,
        &mut bool_values,
        &mut categorical_embed_ids,
        &semantic_types,
        db,
        task_idx,
        seed_idx,
        seed_row,
        &row_map,
        s,
    );

    // Build fk_adj[R, R] where R = max_rows_per_seq (padded)
    let r = max_r;
    let mut fk_adj = vec![0u8; r * r];
    for (&row, &seq_id) in &row_map {
        // Outgoing neighbors that are also in this sequence
        for &neighbor_raw in db.graph.outgoing_neighbors(row) {
            if let Some(&neighbor_seq_id) = row_map.get(&RowIdx(neighbor_raw)) {
                fk_adj[seq_id as usize * r + neighbor_seq_id as usize] = 1;
            }
        }
    }

    // Compute permutations
    let col_perm = compute_col_perm(&column_ids, &is_padding, s);
    let out_perm = compute_rcm_perm(&fk_adj, &seq_row_ids, &is_padding, s, r, true);
    let in_perm = compute_rcm_perm(&fk_adj, &seq_row_ids, &is_padding, s, r, false);

    SequenceData {
        num_cells: cell_count,
        num_rows: row_map.len(),
        semantic_types,
        column_ids,
        seq_row_ids,
        numeric_values,
        timestamp_values,
        bool_values,
        categorical_embed_ids,
        text_embed_ids,
        is_null,
        is_target,
        is_padding,
        fk_adj,
        col_perm,
        out_perm,
        in_perm,
    }
}

/// Pop the next entry from the frontier with parent priority.
fn pop_next(frontier: &mut VecDeque<FrontierEntry>) -> Option<FrontierEntry> {
    // Prefer Parent-priority entries; if none, take from front (closest-first BFS)
    if let Some(pos) = frontier
        .iter()
        .position(|e| e.priority == FrontierPriority::Parent)
    {
        frontier.remove(pos)
    } else {
        frontier.pop_front()
    }
}

/// Mark the target cell in the sequence and write ground-truth values.
///
/// The seed row is always visited first during BFS, so its non-ignored cells
/// occupy the first positions of the sequence. We scan those positions to find
/// a cell whose semantic type matches `target_stype`.
#[allow(clippy::too_many_arguments)]
fn mark_target(
    is_target: &mut [u8],
    is_null: &mut [u8],
    numeric_values: &mut [f32],
    timestamp_values: &mut [f32],
    bool_values: &mut [u8],
    categorical_embed_ids: &mut [u32],
    _semantic_types: &[i8],
    db: &Database,
    task_idx: TaskIdx,
    seed_idx: usize,
    _seed_row: RowIdx,
    _row_map: &HashMap<RowIdx, u16>,
    s: usize,
) {
    let task_meta = db.task_metadata(task_idx);
    let task_view = db.task(task_idx);
    let target_stype = task_meta.target_stype;

    // The seed row's cells were placed first in the sequence. Walk the anchor
    // table's columns in order to find the position of the first cell matching
    // target_stype.
    let anchor_table_meta = &db.metadata.table_metadata[task_meta.anchor_table.0 as usize];
    let col_start = anchor_table_meta.col_range.0.0;
    let col_end = anchor_table_meta.col_range.1.0;

    let mut target_pos: Option<usize> = None;
    let mut pos_in_seq = 0usize;
    for global_col in col_start..col_end {
        let col_meta = &db.metadata.column_metadata[global_col as usize];
        if col_meta.stype == SemanticType::Ignored {
            continue;
        }
        if pos_in_seq >= s {
            break;
        }
        if col_meta.stype == target_stype && target_pos.is_none() {
            target_pos = Some(pos_in_seq);
        }
        pos_in_seq += 1;
    }

    // Fallback: use position 0 (first cell of seed row) for derived tasks where
    // target_stype may not exist as a column in the anchor table.
    let pos = target_pos.unwrap_or(0);
    if pos >= s {
        return;
    }

    is_target[pos] = 1;

    // Write ground-truth target value from the materialized task
    let gt_is_null = task_view.target_is_null(seed_idx);
    is_null[pos] = gt_is_null as u8;

    if !gt_is_null {
        match target_stype {
            SemanticType::Numerical => {
                numeric_values[pos] = task_view.target_numerical(seed_idx);
            }
            SemanticType::Timestamp => {
                let ts = task_view.target_timestamp(seed_idx);
                let start = pos * TIMESTAMP_DIM;
                timestamp_values[start..start + TIMESTAMP_DIM].copy_from_slice(ts);
            }
            SemanticType::Boolean => {
                bool_values[pos] = task_view.target_boolean(seed_idx) as u8;
            }
            SemanticType::Categorical => {
                categorical_embed_ids[pos] = task_view.target_categorical_idx(seed_idx).0;
            }
            _ => {}
        }
    }
}

// ============================================================================
// Permutation Computation
// ============================================================================

/// Compute column-based permutation: argsort by column_ids.
/// Padding positions go to the end.
fn compute_col_perm(column_ids: &[i32], is_padding: &[u8], s: usize) -> Vec<u16> {
    let mut indices: Vec<u16> = (0..s as u16).collect();
    indices.sort_by(|&a, &b| {
        let a_pad = is_padding[a as usize];
        let b_pad = is_padding[b as usize];
        a_pad
            .cmp(&b_pad)
            .then_with(|| column_ids[a as usize].cmp(&column_ids[b as usize]))
            .then_with(|| a.cmp(&b))
    });
    indices
}

/// Compute reverse Cuthill-McKee permutation for block-sparse attention.
/// `outbound=true` uses (I + fk_adj), `outbound=false` uses fk_adj^T.
fn compute_rcm_perm(
    fk_adj: &[u8],
    seq_row_ids: &[u16],
    is_padding: &[u8],
    s: usize,
    r: usize,
    outbound: bool,
) -> Vec<u16> {
    // Build row-level adjacency for RCM
    let mut adj = vec![vec![]; r];
    for i in 0..r {
        for j in 0..r {
            let connected = if outbound {
                (i == j) || fk_adj[i * r + j] == 1 // I + fk_adj
            } else {
                fk_adj[j * r + i] == 1 // fk_adj^T
            };
            if connected && i != j {
                adj[i].push(j);
            }
        }
    }

    // RCM: start from the node with minimum degree
    let mut row_order: Vec<usize> = Vec::with_capacity(r);
    let mut row_visited = vec![false; r];

    // Find all rows that are actually used
    let mut used_rows: Vec<bool> = vec![false; r];
    for pos in 0..s {
        if is_padding[pos] == 0 {
            let row_id = seq_row_ids[pos] as usize;
            if row_id < r {
                used_rows[row_id] = true;
            }
        }
    }

    // Simple BFS-based RCM for used rows
    while row_order.len() < r {
        // Find unvisited used row with minimum degree
        let start = (0..r)
            .filter(|&i| !row_visited[i] && used_rows[i])
            .min_by_key(|&i| adj[i].len())
            .or_else(|| (0..r).find(|&i| !row_visited[i]));

        let start = match start {
            Some(s) => s,
            None => break,
        };

        // BFS from start
        let mut queue = VecDeque::new();
        queue.push_back(start);
        row_visited[start] = true;

        while let Some(node) = queue.pop_front() {
            row_order.push(node);
            // Sort neighbors by degree (ascending) for Cuthill-McKee
            let mut neighbors: Vec<usize> = adj[node]
                .iter()
                .filter(|&&n| !row_visited[n])
                .copied()
                .collect();
            neighbors.sort_by_key(|&n| adj[n].len());
            for n in neighbors {
                if !row_visited[n] {
                    row_visited[n] = true;
                    queue.push_back(n);
                }
            }
        }
    }

    // Reverse for RCM
    row_order.reverse();

    // Fill remaining unused rows
    for (i, visited) in row_visited.iter().enumerate().take(r) {
        if !visited {
            row_order.push(i);
        }
    }

    // Build row_id -> rcm_rank mapping
    let mut row_rank = vec![0usize; r];
    for (rank, &row_id) in row_order.iter().enumerate() {
        if row_id < r {
            row_rank[row_id] = rank;
        }
    }

    // Sort cell positions by (is_padding, row_rank, column_id)
    let mut indices: Vec<u16> = (0..s as u16).collect();
    indices.sort_by(|&a, &b| {
        let a_pad = is_padding[a as usize];
        let b_pad = is_padding[b as usize];
        a_pad.cmp(&b_pad).then_with(|| {
            let a_row = seq_row_ids[a as usize] as usize;
            let b_row = seq_row_ids[b as usize] as usize;
            let a_rank = if a_row < r {
                row_rank[a_row]
            } else {
                usize::MAX
            };
            let b_rank = if b_row < r {
                row_rank[b_row]
            } else {
                usize::MAX
            };
            a_rank.cmp(&b_rank).then_with(|| a.cmp(&b))
        })
    });
    indices
}

// ============================================================================
// Batch Construction
// ============================================================================

/// Build a complete batch from B seeds.
fn build_batch(
    db: &Database,
    seed_manager: &SeedManager,
    config: &SamplerConfig,
    split: Split,
    rng: &mut SmallRng,
) -> Option<RawBatch> {
    let b = config.batch_size as usize;
    let s = config.sequence_length as usize;
    let r = config.max_rows_per_seq as usize;

    let seeds_by_task = seed_manager.seeds_for_split(split);
    let num_tasks = seeds_by_task.len();

    // Find tasks that have seeds
    let tasks_with_seeds: Vec<usize> = (0..num_tasks)
        .filter(|&ti| !seeds_by_task[ti].is_empty())
        .collect();

    if tasks_with_seeds.is_empty() {
        warn!("No tasks with seeds for {:?} split", split);
        return None;
    }

    // Pick a task (uniform random among those with seeds)
    let chosen_task_idx = tasks_with_seeds[rng.random_range(0..tasks_with_seeds.len())];
    let task_seeds = &seeds_by_task[chosen_task_idx];
    let task_meta = &db.metadata.task_metadata[chosen_task_idx];

    // Draw B seed indices (with replacement if needed)
    let seed_indices: Vec<usize> = (0..b)
        .map(|_| task_seeds[rng.random_range(0..task_seeds.len())])
        .collect();

    // Phase 1: Parallel BFS over B seeds
    // Each thread gets its own RNG derived from the batch RNG
    let thread_seeds: Vec<u64> = (0..b).map(|_| rng.random()).collect();
    let task_idx = TaskIdx(chosen_task_idx as u32);

    let sequences: Vec<SequenceData> = thread_seeds
        .par_iter()
        .zip(seed_indices.par_iter())
        .map(|(&thread_seed, &seed_idx)| {
            let mut thread_rng = SmallRng::seed_from_u64(thread_seed);
            build_sequence(db, task_idx, seed_idx, config, &mut thread_rng)
        })
        .collect();

    // Phase 2: Text embedding dedup and collation
    let mut text_global_to_local: HashMap<u32, u32> = HashMap::new();
    let mut unique_text_indices: Vec<u32> = Vec::new();

    for seq in &sequences {
        for pos in 0..s {
            if seq.is_padding[pos] == 1 {
                continue;
            }
            if seq.semantic_types[pos] == SemanticType::Text as i8 && seq.is_null[pos] == 0 {
                let global_idx = seq.text_embed_ids[pos];
                if let std::collections::hash_map::Entry::Vacant(e) =
                    text_global_to_local.entry(global_idx)
                {
                    let local = unique_text_indices.len() as u32;
                    e.insert(local);
                    unique_text_indices.push(global_idx);
                }
            }
        }
    }

    // Gather text embeddings
    let u = unique_text_indices.len();
    let mut text_batch_embeddings = vec![f16::ZERO; u * EMBEDDING_DIM];
    for (local_idx, &global_idx) in unique_text_indices.iter().enumerate() {
        let emb = db.text_embeddings.get(global_idx);
        let start = local_idx * EMBEDDING_DIM;
        text_batch_embeddings[start..start + EMBEDDING_DIM].copy_from_slice(emb);
    }

    // Assemble final batch tensors
    let total_cells = b * s;
    let total_ts = b * s * TIMESTAMP_DIM;
    let total_adj = b * r * r;

    let mut batch_semantic_types = vec![0i8; total_cells];
    let mut batch_column_ids = vec![0i32; total_cells];
    let mut batch_seq_row_ids = vec![0u16; total_cells];
    let mut batch_numeric_values = vec![0.0f32; total_cells];
    let mut batch_timestamp_values = vec![0.0f32; total_ts];
    let mut batch_bool_values = vec![0u8; total_cells];
    let mut batch_categorical_embed_ids = vec![0u32; total_cells];
    let mut batch_text_embed_ids = vec![0u32; total_cells];
    let mut batch_is_null = vec![0u8; total_cells];
    let mut batch_is_target = vec![0u8; total_cells];
    let mut batch_is_padding = vec![1u8; total_cells];
    let mut batch_fk_adj = vec![0u8; total_adj];
    let mut batch_col_perm = vec![0u16; total_cells];
    let mut batch_out_perm = vec![0u16; total_cells];
    let mut batch_in_perm = vec![0u16; total_cells];

    for (bi, seq) in sequences.iter().enumerate() {
        let cell_offset = bi * s;
        let ts_offset = bi * s * TIMESTAMP_DIM;
        let adj_offset = bi * r * r;

        batch_semantic_types[cell_offset..cell_offset + s].copy_from_slice(&seq.semantic_types);
        batch_column_ids[cell_offset..cell_offset + s].copy_from_slice(&seq.column_ids);
        batch_seq_row_ids[cell_offset..cell_offset + s].copy_from_slice(&seq.seq_row_ids);
        batch_numeric_values[cell_offset..cell_offset + s].copy_from_slice(&seq.numeric_values);
        batch_timestamp_values[ts_offset..ts_offset + s * TIMESTAMP_DIM]
            .copy_from_slice(&seq.timestamp_values);
        batch_bool_values[cell_offset..cell_offset + s].copy_from_slice(&seq.bool_values);
        batch_categorical_embed_ids[cell_offset..cell_offset + s]
            .copy_from_slice(&seq.categorical_embed_ids);
        batch_is_null[cell_offset..cell_offset + s].copy_from_slice(&seq.is_null);
        batch_is_target[cell_offset..cell_offset + s].copy_from_slice(&seq.is_target);
        batch_is_padding[cell_offset..cell_offset + s].copy_from_slice(&seq.is_padding);
        batch_fk_adj[adj_offset..adj_offset + r * r].copy_from_slice(&seq.fk_adj);
        batch_col_perm[cell_offset..cell_offset + s].copy_from_slice(&seq.col_perm);
        batch_out_perm[cell_offset..cell_offset + s].copy_from_slice(&seq.out_perm);
        batch_in_perm[cell_offset..cell_offset + s].copy_from_slice(&seq.in_perm);

        // Remap text_embed_ids to batch-local indices
        for pos in 0..s {
            if seq.is_padding[pos] == 0
                && seq.semantic_types[pos] == SemanticType::Text as i8
                && seq.is_null[pos] == 0
            {
                let global = seq.text_embed_ids[pos];
                batch_text_embed_ids[cell_offset + pos] =
                    *text_global_to_local.get(&global).unwrap_or(&0);
            }
        }
    }

    Some(RawBatch {
        batch_size: b,
        sequence_length: s,
        max_rows: r,
        semantic_types: batch_semantic_types,
        column_ids: batch_column_ids,
        seq_row_ids: batch_seq_row_ids,
        numeric_values: batch_numeric_values,
        timestamp_values: batch_timestamp_values,
        bool_values: batch_bool_values,
        categorical_embed_ids: batch_categorical_embed_ids,
        text_embed_ids: batch_text_embed_ids,
        is_null: batch_is_null,
        is_target: batch_is_target,
        is_padding: batch_is_padding,
        fk_adj: batch_fk_adj,
        col_perm: batch_col_perm,
        out_perm: batch_out_perm,
        in_perm: batch_in_perm,
        text_batch_embeddings,
        num_unique_texts: u,
        target_stype: task_meta.target_stype as u8,
        task_idx: chosen_task_idx as u32,
    })
}

// ============================================================================
// Sampler (with prefetch pipeline)
// ============================================================================

/// The main sampler struct, managing prefetch channels and background threads.
#[allow(dead_code)]
pub struct Sampler {
    db: Arc<Database>,
    config: SamplerConfig,
    seed_manager: Arc<SeedManager>,
    train_rx: Receiver<RawBatch>,
    val_rx: Receiver<RawBatch>,
    shutdown: Arc<AtomicBool>,
    train_handle: Option<std::thread::JoinHandle<()>>,
    val_handle: Option<std::thread::JoinHandle<()>>,
}

impl Sampler {
    /// Create a new sampler. Opens the preprocessed database, assigns splits,
    /// shards seeds, and spawns prefetch producer threads.
    pub fn new(config: SamplerConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        info!(
            "Sampler::new: rank={}, world_size={}, db={}",
            config.rank, config.world_size, config.db_path
        );

        let db = Arc::new(Database::load(Path::new(&config.db_path))?);
        let seed_manager = Arc::new(SeedManager::new(&db, &config));

        let train_total = seed_manager.total_seeds(Split::Train);
        let val_total = seed_manager.total_seeds(Split::Val);
        info!(
            "Sampler: train_seeds={}, val_seeds={} (for this rank)",
            train_total, val_total
        );

        let shutdown = Arc::new(AtomicBool::new(false));

        // Create bounded channels
        let (train_tx, train_rx) = channel::bounded(config.num_prefetch);
        let (val_tx, val_rx) = channel::bounded(config.num_prefetch.max(1));

        // Spawn train producer thread
        let train_handle = {
            let db = Arc::clone(&db);
            let sm = Arc::clone(&seed_manager);
            let cfg = config.clone();
            let stop = Arc::clone(&shutdown);
            std::thread::Builder::new()
                .name("sampler-train".into())
                .spawn(move || {
                    producer_loop(db, sm, cfg, Split::Train, train_tx, stop);
                })?
        };

        // Spawn val producer thread
        let val_handle = {
            let db = Arc::clone(&db);
            let sm = Arc::clone(&seed_manager);
            let cfg = config.clone();
            let stop = Arc::clone(&shutdown);
            std::thread::Builder::new()
                .name("sampler-val".into())
                .spawn(move || {
                    producer_loop(db, sm, cfg, Split::Val, val_tx, stop);
                })?
        };

        Ok(Self {
            db,
            config,
            seed_manager,
            train_rx,
            val_rx,
            shutdown,
            train_handle: Some(train_handle),
            val_handle: Some(val_handle),
        })
    }

    /// Pull the next training batch. Blocks until one is available.
    pub fn next_train_batch(&self) -> Result<RawBatch, SamplerError> {
        self.train_rx.recv().map_err(|_| SamplerError::Shutdown)
    }

    /// Pull the next validation batch. Blocks until one is available.
    pub fn next_val_batch(&self) -> Result<RawBatch, SamplerError> {
        self.val_rx.recv().map_err(|_| SamplerError::Shutdown)
    }

    /// Get column embeddings as a flat f16 slice: [C, EMBEDDING_DIM].
    pub fn column_embeddings(&self) -> Vec<f16> {
        let num_cols = self.db.metadata.column_metadata.len();
        let mut out = vec![f16::ZERO; num_cols * EMBEDDING_DIM];
        for (i, cm) in self.db.metadata.column_metadata.iter().enumerate() {
            let start = i * EMBEDDING_DIM;
            if cm.embedding.len() == EMBEDDING_DIM {
                out[start..start + EMBEDDING_DIM].copy_from_slice(&cm.embedding);
            }
        }
        out
    }

    /// Get categorical embeddings as a flat f16 slice: [Vc, EMBEDDING_DIM].
    pub fn categorical_embeddings(&self) -> Vec<f16> {
        let n = self.db.categorical_embeddings.num_embeddings();
        let mut out = vec![f16::ZERO; n * EMBEDDING_DIM];
        for i in 0..n {
            let emb = self.db.categorical_embeddings.get(i as u32);
            let start = i * EMBEDDING_DIM;
            out[start..start + EMBEDDING_DIM].copy_from_slice(emb);
        }
        out
    }

    /// Get a reference to the loaded database.
    pub fn database(&self) -> &Database {
        &self.db
    }

    /// Shut down the sampler: signal producer threads to stop, drain channels,
    /// and join threads.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);

        // Drain channels so producers can unblock
        while self.train_rx.try_recv().is_ok() {}
        while self.val_rx.try_recv().is_ok() {}

        if let Some(h) = self.train_handle.take() {
            let _ = h.join();
        }
        if let Some(h) = self.val_handle.take() {
            let _ = h.join();
        }

        info!("Sampler shut down.");
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Error type for sampler operations.
#[derive(Debug, thiserror::Error)]
pub enum SamplerError {
    #[error("sampler has been shut down")]
    Shutdown,
}

/// Background producer loop: continuously builds batches and pushes to channel.
fn producer_loop(
    db: Arc<Database>,
    seed_manager: Arc<SeedManager>,
    config: SamplerConfig,
    split: Split,
    tx: Sender<RawBatch>,
    shutdown: Arc<AtomicBool>,
) {
    // Derive per-thread RNG from config seed + rank + split
    let split_offset = match split {
        Split::Train => 0u64,
        Split::Val => 1,
        Split::Test => 2,
    };
    let mut rng = SmallRng::seed_from_u64(
        config
            .seed
            .wrapping_add(config.rank as u64 * 1000)
            .wrapping_add(split_offset),
    );

    debug!("Producer {:?} started", split);

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        match build_batch(&db, &seed_manager, &config, split, &mut rng) {
            Some(batch) => {
                if tx.send(batch).is_err() {
                    break; // receiver dropped
                }
            }
            None => {
                // No seeds available, sleep briefly and retry
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }

    debug!("Producer {:?} exiting", split);
}
