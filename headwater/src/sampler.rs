//! BFS-based relational database sampler for training batch construction.
//!
//! The sampler walks FK edges outward from task seed rows, collects cells into
//! sequences, and packs them into batches for the relational transformer.
//! Attention masks are built per sequence and shipped as packed CSR (row_ptr,
//! seq_offsets, col_idx); the model decodes them to dense bool masks.
//!
//! See `documentation/sampling.md` and `documentation/system_architecture.md`
//! for the full design.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use crossbeam::channel::{self, Receiver, Sender, TryRecvError, TrySendError};
use half::f16;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use tracing::{debug, info, warn};

use crate::common::{ColumnSlice, Database, RowIdx, SemanticType, TIMESTAMP_DIM, TaskIdx};
use crate::embedder::EMBEDDING_DIM;

static PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static PROFILE_BATCH_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_SEQUENCE_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_BATCH_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_BATCH_PHASE1_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_BATCH_TEXT_DEDUP_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_BATCH_TEXT_GATHER_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_BATCH_COLLATE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_SEQUENCE_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_ENCODE_MASK_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_FRONTIER_POP_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_VISITED_CHECK_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_ROW_MATERIALIZE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_OUT_NEIGHBOR_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_IN_NEIGHBOR_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_TEMPORAL_FILTER_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_CHILD_SUBSAMPLE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_ROWS_VISITED: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_ROWS_SKIPPED_VISITED: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_OUT_NEIGHBOR_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_IN_NEIGHBOR_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_TEMPORAL_CHECK_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_TEMPORAL_REJECT_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_PARENT_PUSH_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_PHASE1_CHILD_PUSH_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_PRODUCED_TRAIN_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_PRODUCED_VAL_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_CONSUMED_TRAIN_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_CONSUMED_VAL_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_PRODUCER_BLOCKED_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_PRODUCER_BLOCKED_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_CONSUMER_BLOCKED_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_CONSUMER_BLOCKED_NS: AtomicU64 = AtomicU64::new(0);

/// Word size in bits for position bitsets (used in CSR mask encoding).
const BITSET_WORD_BITS: usize = u64::BITS as usize;

/// True when `HEADWATER_PROFILE=1`; enables per-phase timing and profile logs.
fn profile_enabled() -> bool {
    *PROFILE_ENABLED.get_or_init(|| {
        std::env::var("HEADWATER_PROFILE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn add_elapsed_ns(counter: &AtomicU64, start: Option<Instant>) {
    if let Some(t0) = start {
        counter.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// True if the bit for the given row is set in the packed validity bitmap.
#[inline]
fn validity_bit_is_set(validity: &[u8], row_byte: usize, row_bit_mask: u8) -> bool {
    (validity[row_byte] & row_bit_mask) != 0
}

/// Set one bit in a position bitset (used for CSR mask construction).
#[inline]
fn bitset_set(bits: &mut [u64], idx: usize) {
    let word = idx / BITSET_WORD_BITS;
    let bit = idx % BITSET_WORD_BITS;
    bits[word] |= 1u64 << bit;
}

/// OR `src` into `dst` in place; both must have the same length.
#[inline]
fn bitset_or_inplace(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d |= *s;
    }
}

/// Append sequence positions (0..S) corresponding to set bits, in ascending order.
#[inline]
fn bitset_extend_sorted_indices(bits: &[u64], out: &mut Vec<u16>) {
    for (word_idx, &word) in bits.iter().enumerate() {
        let mut w = word;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            let idx = word_idx * BITSET_WORD_BITS + tz;
            out.push(idx as u16);
            w &= w - 1;
        }
    }
}

/// Copy one row's timestamp encoding (length `TIMESTAMP_DIM`) into the batch slot at position `pos`.
#[inline]
fn copy_timestamp_values(dst: &mut [f32], pos: usize, ts: &[f32]) {
    let start = pos * TIMESTAMP_DIM;
    dst[start..start + TIMESTAMP_DIM].copy_from_slice(ts);
}

fn log_profile_report() {
    let batches = PROFILE_BATCH_COUNT.load(Ordering::Relaxed);
    let seqs = PROFILE_SEQUENCE_COUNT.load(Ordering::Relaxed);
    if batches == 0 && seqs == 0 {
        return;
    }

    let batch_total_ns = PROFILE_BATCH_TOTAL_NS.load(Ordering::Relaxed);
    let phase1_ns = PROFILE_BATCH_PHASE1_NS.load(Ordering::Relaxed);
    let text_dedup_ns = PROFILE_BATCH_TEXT_DEDUP_NS.load(Ordering::Relaxed);
    let text_gather_ns = PROFILE_BATCH_TEXT_GATHER_NS.load(Ordering::Relaxed);
    let collate_ns = PROFILE_BATCH_COLLATE_NS.load(Ordering::Relaxed);
    let seq_total_ns = PROFILE_SEQUENCE_TOTAL_NS.load(Ordering::Relaxed);
    let encode_mask_ns = PROFILE_ENCODE_MASK_NS.load(Ordering::Relaxed);
    let phase1_frontier_pop_ns = PROFILE_PHASE1_FRONTIER_POP_NS.load(Ordering::Relaxed);
    let phase1_visited_check_ns = PROFILE_PHASE1_VISITED_CHECK_NS.load(Ordering::Relaxed);
    let phase1_row_materialize_ns = PROFILE_PHASE1_ROW_MATERIALIZE_NS.load(Ordering::Relaxed);
    let phase1_out_neighbor_ns = PROFILE_PHASE1_OUT_NEIGHBOR_NS.load(Ordering::Relaxed);
    let phase1_in_neighbor_ns = PROFILE_PHASE1_IN_NEIGHBOR_NS.load(Ordering::Relaxed);
    let phase1_temporal_filter_ns = PROFILE_PHASE1_TEMPORAL_FILTER_NS.load(Ordering::Relaxed);
    let phase1_child_subsample_ns = PROFILE_PHASE1_CHILD_SUBSAMPLE_NS.load(Ordering::Relaxed);
    let phase1_rows_visited = PROFILE_PHASE1_ROWS_VISITED.load(Ordering::Relaxed);
    let phase1_rows_skipped_visited = PROFILE_PHASE1_ROWS_SKIPPED_VISITED.load(Ordering::Relaxed);
    let phase1_out_neighbor_count = PROFILE_PHASE1_OUT_NEIGHBOR_COUNT.load(Ordering::Relaxed);
    let phase1_in_neighbor_count = PROFILE_PHASE1_IN_NEIGHBOR_COUNT.load(Ordering::Relaxed);
    let phase1_temporal_check_count = PROFILE_PHASE1_TEMPORAL_CHECK_COUNT.load(Ordering::Relaxed);
    let phase1_temporal_reject_count = PROFILE_PHASE1_TEMPORAL_REJECT_COUNT.load(Ordering::Relaxed);
    let phase1_parent_push_count = PROFILE_PHASE1_PARENT_PUSH_COUNT.load(Ordering::Relaxed);
    let phase1_child_push_count = PROFILE_PHASE1_CHILD_PUSH_COUNT.load(Ordering::Relaxed);
    let produced_train = PROFILE_PRODUCED_TRAIN_COUNT.load(Ordering::Relaxed);
    let produced_val = PROFILE_PRODUCED_VAL_COUNT.load(Ordering::Relaxed);
    let consumed_train = PROFILE_CONSUMED_TRAIN_COUNT.load(Ordering::Relaxed);
    let consumed_val = PROFILE_CONSUMED_VAL_COUNT.load(Ordering::Relaxed);
    let producer_blocked_count = PROFILE_PRODUCER_BLOCKED_COUNT.load(Ordering::Relaxed);
    let producer_blocked_ns = PROFILE_PRODUCER_BLOCKED_NS.load(Ordering::Relaxed);
    let consumer_blocked_count = PROFILE_CONSUMER_BLOCKED_COUNT.load(Ordering::Relaxed);
    let consumer_blocked_ns = PROFILE_CONSUMER_BLOCKED_NS.load(Ordering::Relaxed);

    let batch_denom = (batches as f64).max(1.0);
    let seq_denom = (seqs as f64).max(1.0);
    info!(
        "Sampler profile: batches={}, sequences={}, batch_avg_ms={:.3}, phase1_ms={:.3}, text_dedup_ms={:.3}, text_gather_ms={:.3}, collate_ms={:.3}, sequence_avg_ms={:.3}, encode_mask_ms={:.3}",
        batches,
        seqs,
        batch_total_ns as f64 / batch_denom / 1e6,
        phase1_ns as f64 / batch_denom / 1e6,
        text_dedup_ns as f64 / batch_denom / 1e6,
        text_gather_ns as f64 / batch_denom / 1e6,
        collate_ns as f64 / batch_denom / 1e6,
        seq_total_ns as f64 / seq_denom / 1e6,
        encode_mask_ns as f64 / seq_denom / 1e6,
    );
    info!(
        "Sampler phase1 detail: frontier_pop_ms={:.3}, visited_check_ms={:.3}, row_materialize_ms={:.3}, out_neighbor_ms={:.3}, in_neighbor_ms={:.3}, temporal_filter_ms={:.3}, child_subsample_ms={:.3}, rows_visited_avg={:.2}, rows_skipped_avg={:.2}, out_neighbors_avg={:.2}, in_neighbors_avg={:.2}, temporal_checks_avg={:.2}, temporal_reject_rate={:.3}, parent_push_avg={:.2}, child_push_avg={:.2}",
        phase1_frontier_pop_ns as f64 / seq_denom / 1e6,
        phase1_visited_check_ns as f64 / seq_denom / 1e6,
        phase1_row_materialize_ns as f64 / seq_denom / 1e6,
        phase1_out_neighbor_ns as f64 / seq_denom / 1e6,
        phase1_in_neighbor_ns as f64 / seq_denom / 1e6,
        phase1_temporal_filter_ns as f64 / seq_denom / 1e6,
        phase1_child_subsample_ns as f64 / seq_denom / 1e6,
        phase1_rows_visited as f64 / seq_denom,
        phase1_rows_skipped_visited as f64 / seq_denom,
        phase1_out_neighbor_count as f64 / seq_denom,
        phase1_in_neighbor_count as f64 / seq_denom,
        phase1_temporal_check_count as f64 / seq_denom,
        if phase1_temporal_check_count > 0 {
            phase1_temporal_reject_count as f64 / phase1_temporal_check_count as f64
        } else {
            0.0
        },
        phase1_parent_push_count as f64 / seq_denom,
        phase1_child_push_count as f64 / seq_denom,
    );
    info!(
        "Sampler queue profile: produced_train={}, consumed_train={}, produced_val={}, consumed_val={}, producer_blocked_events={}, producer_blocked_avg_ms={:.3}, consumer_blocked_events={}, consumer_blocked_avg_ms={:.3}",
        produced_train,
        consumed_train,
        produced_val,
        consumed_val,
        producer_blocked_count,
        if producer_blocked_count > 0 {
            producer_blocked_ns as f64 / producer_blocked_count as f64 / 1e6
        } else {
            0.0
        },
        consumer_blocked_count,
        if consumer_blocked_count > 0 {
            consumer_blocked_ns as f64 / consumer_blocked_count as f64 / 1e6
        } else {
            0.0
        },
    );
}

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

/// Deterministic hash-based train/val/test assignment per (task, anchor_row, split_seed).
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
    train_tasks_with_seeds: Vec<usize>,
    val_tasks_with_seeds: Vec<usize>,
    test_tasks_with_seeds: Vec<usize>,
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

        let train_tasks_with_seeds = train_seeds
            .iter()
            .enumerate()
            .filter(|(_, seeds)| !seeds.is_empty())
            .map(|(ti, _)| ti)
            .collect();
        let val_tasks_with_seeds = val_seeds
            .iter()
            .enumerate()
            .filter(|(_, seeds)| !seeds.is_empty())
            .map(|(ti, _)| ti)
            .collect();
        let test_tasks_with_seeds = test_seeds
            .iter()
            .enumerate()
            .filter(|(_, seeds)| !seeds.is_empty())
            .map(|(ti, _)| ti)
            .collect();

        Self {
            train_seeds,
            val_seeds,
            test_seeds,
            train_tasks_with_seeds,
            val_tasks_with_seeds,
            test_tasks_with_seeds,
        }
    }

    fn seeds_for_split(&self, split: Split) -> &[Vec<usize>] {
        match split {
            Split::Train => &self.train_seeds,
            Split::Val => &self.val_seeds,
            Split::Test => &self.test_seeds,
        }
    }

    fn tasks_with_seeds(&self, split: Split) -> &[usize] {
        match split {
            Split::Train => &self.train_tasks_with_seeds,
            Split::Val => &self.val_tasks_with_seeds,
            Split::Test => &self.test_tasks_with_seeds,
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
/// All tensors are flat `Vec<T>`; shapes are `[B, S]`, `[B, S, TIMESTAMP_DIM]`, or
/// packed CSR. The Python binding converts these to NumPy arrays via zero-copy.
pub struct RawBatch {
    pub batch_size: usize,
    pub sequence_length: usize,
    /// Max distinct rows in any one sequence in this batch (R dimension).
    pub max_rows: usize,

    // --- Cell identity [B, S] ---
    pub semantic_types: Vec<i8>,
    pub column_ids: Vec<i32>,
    pub seq_row_ids: Vec<u16>,

    // --- Per-type values [B, S] or [B, S, TIMESTAMP_DIM] ---
    pub numeric_values: Vec<f32>,
    pub timestamp_values: Vec<f32>,
    pub bool_values: Vec<u8>,
    pub categorical_embed_ids: Vec<u32>,
    pub text_embed_ids: Vec<u32>,

    // --- Validity and masking [B, S] ---
    pub is_null: Vec<u8>,
    pub is_target: Vec<u8>,
    pub is_padding: Vec<u8>,

    // --- Attention masks (CSR transport) ---
    /// Always 1 (CSR). Row ptrs [B, S+1]; seq_offsets [B+1]; col_idx packed.
    pub mask_format: u8,
    pub outbound_csr_row_ptr: Vec<u32>,
    pub outbound_csr_seq_offsets: Vec<u32>,
    pub outbound_csr_col_idx: Vec<u16>,
    pub inbound_csr_row_ptr: Vec<u32>,
    pub inbound_csr_seq_offsets: Vec<u32>,
    pub inbound_csr_col_idx: Vec<u16>,
    pub column_csr_row_ptr: Vec<u32>,
    pub column_csr_seq_offsets: Vec<u32>,
    pub column_csr_col_idx: Vec<u16>,

    // --- Text embeddings (per-batch subset) ---
    /// [U, EMBEDDING_DIM] f16; U = num unique text ids in this batch.
    pub text_batch_embeddings: Vec<f16>,
    pub num_unique_texts: usize,

    // --- Batch-level metadata ---
    pub target_stype: u8,
    pub task_idx: u32,
}

/// Per-batch buffers filled during Phase 1; converted into `RawBatch` by `into_raw_batch`.
struct BatchBuffers {
    semantic_types: Vec<i8>,
    column_ids: Vec<i32>,
    seq_row_ids: Vec<u16>,
    numeric_values: Vec<f32>,
    timestamp_values: Vec<f32>,
    bool_values: Vec<u8>,
    categorical_embed_ids: Vec<u32>,
    text_embed_ids: Vec<u32>,
    is_null: Vec<u8>,
    is_target: Vec<u8>,
    is_padding: Vec<u8>,
}

/// Per-sequence CSR attention masks (outbound, inbound, column) before packing into batch-wide arrays.
#[derive(Default)]
struct SequenceCsrMasks {
    outbound_row_ptr: Vec<u32>,
    outbound_col_idx: Vec<u16>,
    inbound_row_ptr: Vec<u32>,
    inbound_col_idx: Vec<u16>,
    column_row_ptr: Vec<u32>,
    column_col_idx: Vec<u16>,
}

impl SequenceCsrMasks {
    fn clear(&mut self) {
        self.outbound_row_ptr.clear();
        self.outbound_col_idx.clear();
        self.inbound_row_ptr.clear();
        self.inbound_col_idx.clear();
        self.column_row_ptr.clear();
        self.column_col_idx.clear();
    }
}

/// Mutable slices for one sequence's slot within `BatchBuffers` (one of B sequences).
struct SequenceSlotMut<'a> {
    semantic_types: &'a mut [i8],
    column_ids: &'a mut [i32],
    seq_row_ids: &'a mut [u16],
    numeric_values: &'a mut [f32],
    timestamp_values: &'a mut [f32],
    bool_values: &'a mut [u8],
    categorical_embed_ids: &'a mut [u32],
    text_embed_ids: &'a mut [u32],
    is_null: &'a mut [u8],
    is_target: &'a mut [u8],
    is_padding: &'a mut [u8],
}

impl BatchBuffers {
    fn new(b: usize, s: usize) -> Self {
        let total_cells = b * s;
        let total_ts = b * s * TIMESTAMP_DIM;
        Self {
            semantic_types: vec![0i8; total_cells],
            column_ids: vec![0i32; total_cells],
            seq_row_ids: vec![0u16; total_cells],
            numeric_values: vec![0.0f32; total_cells],
            timestamp_values: vec![0.0f32; total_ts],
            bool_values: vec![0u8; total_cells],
            categorical_embed_ids: vec![0u32; total_cells],
            text_embed_ids: vec![0u32; total_cells],
            is_null: vec![0u8; total_cells],
            is_target: vec![0u8; total_cells],
            is_padding: vec![1u8; total_cells],
        }
    }

    fn into_raw_batch(
        self,
        seq_csr_masks: &mut [SequenceCsrMasks],
        b: usize,
        s: usize,
        max_rows: usize,
        text_batch_embeddings: Vec<f16>,
        num_unique_texts: usize,
        target_stype: u8,
        task_idx: u32,
    ) -> RawBatch {
        let mut outbound_csr_row_ptr = vec![0u32; b * (s + 1)];
        let mut inbound_csr_row_ptr = vec![0u32; b * (s + 1)];
        let mut column_csr_row_ptr = vec![0u32; b * (s + 1)];
        let mut outbound_csr_seq_offsets = vec![0u32; b + 1];
        let mut inbound_csr_seq_offsets = vec![0u32; b + 1];
        let mut column_csr_seq_offsets = vec![0u32; b + 1];

        for bi in 0..b {
            outbound_csr_seq_offsets[bi + 1] =
                outbound_csr_seq_offsets[bi] + seq_csr_masks[bi].outbound_col_idx.len() as u32;
            inbound_csr_seq_offsets[bi + 1] =
                inbound_csr_seq_offsets[bi] + seq_csr_masks[bi].inbound_col_idx.len() as u32;
            column_csr_seq_offsets[bi + 1] =
                column_csr_seq_offsets[bi] + seq_csr_masks[bi].column_col_idx.len() as u32;
        }

        let mut outbound_csr_col_idx = vec![0u16; outbound_csr_seq_offsets[b] as usize];
        let mut inbound_csr_col_idx = vec![0u16; inbound_csr_seq_offsets[b] as usize];
        let mut column_csr_col_idx = vec![0u16; column_csr_seq_offsets[b] as usize];

        for (bi, seq) in seq_csr_masks.iter().enumerate() {
            let row_off = bi * (s + 1);
            let out_col_off = outbound_csr_seq_offsets[bi] as usize;
            let in_col_off = inbound_csr_seq_offsets[bi] as usize;
            let col_col_off = column_csr_seq_offsets[bi] as usize;
            let out_nnz = seq.outbound_col_idx.len();
            let in_nnz = seq.inbound_col_idx.len();
            let col_nnz = seq.column_col_idx.len();
            debug_assert_eq!(seq.outbound_row_ptr.len(), s + 1);
            debug_assert_eq!(seq.inbound_row_ptr.len(), s + 1);
            debug_assert_eq!(seq.column_row_ptr.len(), s + 1);
            outbound_csr_row_ptr[row_off..row_off + (s + 1)].copy_from_slice(&seq.outbound_row_ptr);
            inbound_csr_row_ptr[row_off..row_off + (s + 1)].copy_from_slice(&seq.inbound_row_ptr);
            column_csr_row_ptr[row_off..row_off + (s + 1)].copy_from_slice(&seq.column_row_ptr);
            outbound_csr_col_idx[out_col_off..out_col_off + out_nnz]
                .copy_from_slice(&seq.outbound_col_idx);
            inbound_csr_col_idx[in_col_off..in_col_off + in_nnz]
                .copy_from_slice(&seq.inbound_col_idx);
            column_csr_col_idx[col_col_off..col_col_off + col_nnz]
                .copy_from_slice(&seq.column_col_idx);
        }
        for seq in seq_csr_masks.iter_mut() {
            seq.clear();
        }

        RawBatch {
            batch_size: b,
            sequence_length: s,
            max_rows,
            semantic_types: self.semantic_types,
            column_ids: self.column_ids,
            seq_row_ids: self.seq_row_ids,
            numeric_values: self.numeric_values,
            timestamp_values: self.timestamp_values,
            bool_values: self.bool_values,
            categorical_embed_ids: self.categorical_embed_ids,
            text_embed_ids: self.text_embed_ids,
            is_null: self.is_null,
            is_target: self.is_target,
            is_padding: self.is_padding,
            mask_format: 1,
            outbound_csr_row_ptr,
            outbound_csr_seq_offsets,
            outbound_csr_col_idx,
            inbound_csr_row_ptr,
            inbound_csr_seq_offsets,
            inbound_csr_col_idx,
            column_csr_row_ptr,
            column_csr_seq_offsets,
            column_csr_col_idx,
            text_batch_embeddings,
            num_unique_texts,
            target_stype,
            task_idx,
        }
    }
}

/// Reused across batch builds: per-sequence CSR mask buffers, seed indices, and text dedup state.
struct BatchBuildWorkspace {
    seq_csr_masks: Vec<SequenceCsrMasks>,
    seq_num_rows: Vec<usize>,
    seed_indices: Vec<usize>,
    thread_seeds: Vec<u64>,
    text_global_to_local: FxHashMap<u32, u32>,
    unique_text_indices: Vec<u32>,
}

impl BatchBuildWorkspace {
    fn new(batch_size: usize) -> Self {
        let mut seq_csr_masks = Vec::with_capacity(batch_size);
        seq_csr_masks.resize_with(batch_size, SequenceCsrMasks::default);
        Self {
            seq_csr_masks,
            seq_num_rows: vec![0usize; batch_size],
            seed_indices: Vec::with_capacity(batch_size),
            thread_seeds: Vec::with_capacity(batch_size),
            text_global_to_local: FxHashMap::default(),
            unique_text_indices: Vec::new(),
        }
    }

    fn reset_for_batch(&mut self, batch_size: usize) {
        if self.seq_csr_masks.len() != batch_size {
            self.seq_csr_masks
                .resize_with(batch_size, SequenceCsrMasks::default);
        }
        for seq in &mut self.seq_csr_masks {
            seq.clear();
        }
        self.seq_num_rows.resize(batch_size, 0);
        self.seq_num_rows.fill(0);
        self.seed_indices.clear();
        self.thread_seeds.clear();
        self.text_global_to_local.clear();
        self.unique_text_indices.clear();
    }
}

/// Per-thread scratch for building one sequence: BFS state, row adjacency, and bitsets for CSR mask encoding.
#[derive(Default)]
struct SequenceScratch {
    row_map: FxHashMap<RowIdx, u16>,
    parent_frontier: VecDeque<RowIdx>,
    child_frontier: VecDeque<RowIdx>,
    eligible_children: Vec<RowIdx>,
    active_positions: Vec<usize>,
    /// For each row index ri, bitset of sequence positions (0..S) in that row; used for CSR mask OR.
    row_position_bits: Vec<Vec<u64>>,
    out_neighbors: Vec<Vec<usize>>,
    in_neighbors: Vec<Vec<usize>>,
    col_pairs: Vec<(i32, usize)>,
    tmp_bits: Vec<u64>,
    col_group_start: Vec<usize>,
    col_group_end: Vec<usize>,
}

// ============================================================================
// Per-Sequence Builder
// ============================================================================

/// Per-sequence summary stats written during slot construction.
struct SequenceStats {
    num_rows: usize,
}

/// Build one sequence: BFS from seed, materialize cells into slot, build adjacency, encode CSR masks.
#[allow(clippy::too_many_arguments)]
fn build_sequence_into(
    db: &Database,
    task_idx: TaskIdx,
    seed_idx: usize,
    target_pos_in_seed: usize,
    config: &SamplerConfig,
    rng: &mut SmallRng,
    scratch: &mut SequenceScratch,
    slot: &mut SequenceSlotMut<'_>,
    seq_csr_masks: &mut SequenceCsrMasks,
) -> SequenceStats {
    let prof = profile_enabled();
    let t_sequence_total = if prof { Some(Instant::now()) } else { None };
    // Profiling accumulators (only used when HEADWATER_PROFILE=1)
    let mut p_frontier_pop_ns = 0u64;
    let mut p_visited_check_ns = 0u64;
    let mut p_row_materialize_ns = 0u64;
    let mut p_out_neighbor_ns = 0u64;
    let mut p_in_neighbor_ns = 0u64;
    let mut p_temporal_filter_ns = 0u64;
    let mut p_child_subsample_ns = 0u64;
    let mut p_rows_visited = 0u64;
    let mut p_rows_skipped_visited = 0u64;
    let mut p_out_neighbor_count = 0u64;
    let mut p_in_neighbor_count = 0u64;
    let mut p_temporal_check_count = 0u64;
    let mut p_temporal_reject_count = 0u64;
    let mut p_parent_push_count = 0u64;
    let mut p_child_push_count = 0u64;
    let s = config.sequence_length as usize;
    let _task_meta = db.task_metadata(task_idx);
    let task_view = db.task(task_idx);

    let seed_row = task_view.anchor_row(seed_idx);
    let obs_time = task_view.observation_time(seed_idx);
    let use_temporal_filter = obs_time != i64::MAX;
    let obs_time_z_cutoff = if use_temporal_filter && db.metadata.global_ts_std_us > 0.0 {
        Some(((obs_time as f64) - db.metadata.global_ts_mean_us) / db.metadata.global_ts_std_us)
    } else {
        None
    };

    debug_assert_eq!(slot.semantic_types.len(), s);
    debug_assert_eq!(slot.column_ids.len(), s);
    debug_assert_eq!(slot.seq_row_ids.len(), s);
    debug_assert_eq!(slot.numeric_values.len(), s);
    debug_assert_eq!(slot.timestamp_values.len(), s * TIMESTAMP_DIM);
    debug_assert_eq!(slot.bool_values.len(), s);
    debug_assert_eq!(slot.categorical_embed_ids.len(), s);
    debug_assert_eq!(slot.text_embed_ids.len(), s);
    debug_assert_eq!(slot.is_null.len(), s);
    debug_assert_eq!(slot.is_target.len(), s);
    debug_assert_eq!(slot.is_padding.len(), s);

    let semantic_types = &mut slot.semantic_types;
    let column_ids = &mut slot.column_ids;
    let seq_row_ids = &mut slot.seq_row_ids;
    let numeric_values = &mut slot.numeric_values;
    let timestamp_values = &mut slot.timestamp_values;
    let bool_values = &mut slot.bool_values;
    let categorical_embed_ids = &mut slot.categorical_embed_ids;
    let text_embed_ids = &mut slot.text_embed_ids;
    let is_null = &mut slot.is_null;
    let is_target = &mut slot.is_target;
    let is_padding = &mut slot.is_padding;

    // Reuse per-thread scratch buffers to reduce allocator churn.
    scratch.row_map.clear();
    scratch.parent_frontier.clear();
    scratch.child_frontier.clear();
    scratch.eligible_children.clear();

    let row_map = &mut scratch.row_map;
    let mut cell_count: usize = 0;
    let parent_frontier = &mut scratch.parent_frontier;
    let child_frontier = &mut scratch.child_frontier;

    // Seed the frontier
    parent_frontier.push_back(seed_row);

    loop {
        let t_frontier_pop = if prof { Some(Instant::now()) } else { None };
        let row = match parent_frontier
            .pop_front()
            .or_else(|| child_frontier.pop_front())
        {
            Some(row) => row,
            None => break,
        };
        if let Some(t0) = t_frontier_pop {
            p_frontier_pop_ns += t0.elapsed().as_nanos() as u64;
        }
        if cell_count >= s {
            break;
        }
        let t_visited_check = if prof { Some(Instant::now()) } else { None };
        let next_seq_row_id = row_map.len() as u16;
        let is_new_row = if let std::collections::hash_map::Entry::Vacant(slot) = row_map.entry(row)
        {
            slot.insert(next_seq_row_id);
            true
        } else {
            false
        };
        if let Some(t0) = t_visited_check {
            p_visited_check_ns += t0.elapsed().as_nanos() as u64;
        }
        if !is_new_row {
            p_rows_skipped_visited += 1;
            continue;
        }
        p_rows_visited += 1;
        let t_row_materialize = if prof { Some(Instant::now()) } else { None };
        let table_idx = db.row_table(row);
        let table_meta = &db.metadata.table_metadata[table_idx.0 as usize];
        let table_view = db.table(table_idx);
        let local_row = (row.0 - table_meta.row_range.0.0) as usize;

        // Assigned above via row_map.entry()
        let seq_row_id = row_map[&row];

        // Precompute bitmap indices for this row (used for validity/boolean bits in column slices).
        let row_byte = local_row / 8;
        let row_bit_mask = 1u8 << (local_row % 8);
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

            // Match once on the column backing slice to avoid repeated dispatch
            // through TableView helpers in this hot row-materialization loop.
            match table_view.column(local_col) {
                ColumnSlice::Identifier { .. } => {
                    is_null[pos] = 0;
                }
                ColumnSlice::Numerical { validity, values } => {
                    let is_valid = validity_bit_is_set(validity, row_byte, row_bit_mask);
                    is_null[pos] = (!is_valid) as u8;
                    if is_valid {
                        numeric_values[pos] = values[local_row];
                    }
                }
                ColumnSlice::Timestamp { validity, values } => {
                    let is_valid = validity_bit_is_set(validity, row_byte, row_bit_mask);
                    is_null[pos] = (!is_valid) as u8;
                    if is_valid {
                        let ts_start = local_row * TIMESTAMP_DIM;
                        copy_timestamp_values(
                            timestamp_values,
                            pos,
                            &values[ts_start..ts_start + TIMESTAMP_DIM],
                        );
                    }
                }
                ColumnSlice::Boolean { validity, bits } => {
                    let is_valid = validity_bit_is_set(validity, row_byte, row_bit_mask);
                    is_null[pos] = (!is_valid) as u8;
                    if is_valid {
                        bool_values[pos] = validity_bit_is_set(bits, row_byte, row_bit_mask) as u8;
                    }
                }
                ColumnSlice::Embedded { validity, indices } => {
                    let is_valid = validity_bit_is_set(validity, row_byte, row_bit_mask);
                    is_null[pos] = (!is_valid) as u8;
                    if is_valid {
                        match col_meta.stype {
                            SemanticType::Categorical => {
                                categorical_embed_ids[pos] = indices[local_row];
                            }
                            SemanticType::Text => {
                                text_embed_ids[pos] = indices[local_row];
                            }
                            _ => {
                                unreachable!("embedded storage is only valid for categorical/text")
                            }
                        }
                    }
                }
                ColumnSlice::Ignored => unreachable!("ignored columns are filtered above"),
            }
        }
        if let Some(t0) = t_row_materialize {
            p_row_materialize_ns += t0.elapsed().as_nanos() as u64;
        }

        if cell_count >= s {
            break;
        }

        // Add FK neighbors to frontier
        // Outgoing edges (F->P): this row's FK columns point to parent rows
        let t_out_neighbors = if prof { Some(Instant::now()) } else { None };
        for &neighbor_raw in db.graph.outgoing_neighbors(row) {
            p_out_neighbor_count += 1;
            let neighbor = RowIdx(neighbor_raw);
            if !row_map.contains_key(&neighbor) {
                parent_frontier.push_back(neighbor);
                p_parent_push_count += 1;
            }
        }
        if let Some(t0) = t_out_neighbors {
            p_out_neighbor_ns += t0.elapsed().as_nanos() as u64;
        }

        // Incoming edges (P->F): rows with FKs into this row (children).
        // Graph stores a temporal-aware partition:
        //   - static incoming: rows without usable temporal value
        //   - temporal incoming: rows sorted by timestamp z-score
        let t_in_neighbors = if prof { Some(Instant::now()) } else { None };
        scratch.eligible_children.clear();
        let static_children = db.graph.incoming_static_neighbors(row);
        let (temporal_children, temporal_z) = db.graph.incoming_temporal_neighbors(row);
        p_in_neighbor_count += (static_children.len() + temporal_children.len()) as u64;

        for &child_raw in static_children {
            let child_row = RowIdx(child_raw);
            if !row_map.contains_key(&child_row) {
                scratch.eligible_children.push(child_row);
            }
        }

        let temporal_prefix_len = if use_temporal_filter {
            let t_temporal_filter = if prof { Some(Instant::now()) } else { None };
            p_temporal_check_count += temporal_children.len() as u64;
            let prefix_len = if let Some(z_cutoff) = obs_time_z_cutoff {
                temporal_z.partition_point(|&z| (z as f64) <= z_cutoff)
            } else {
                temporal_children.len()
            };
            p_temporal_reject_count += (temporal_children.len() - prefix_len) as u64;
            if let Some(t0) = t_temporal_filter {
                p_temporal_filter_ns += t0.elapsed().as_nanos() as u64;
            }
            prefix_len
        } else {
            temporal_children.len()
        };

        for &child_raw in &temporal_children[..temporal_prefix_len] {
            let child_row = RowIdx(child_raw);
            if !row_map.contains_key(&child_row) {
                scratch.eligible_children.push(child_row);
            }
        }
        if let Some(t0) = t_in_neighbors {
            p_in_neighbor_ns += t0.elapsed().as_nanos() as u64;
        }

        // Subsample children to bfs_child_width
        let t_child_subsample = if prof { Some(Instant::now()) } else { None };
        if scratch.eligible_children.len() > config.bfs_child_width as usize {
            scratch.eligible_children.shuffle(rng);
            scratch
                .eligible_children
                .truncate(config.bfs_child_width as usize);
        }
        for &child_row in &scratch.eligible_children {
            if !row_map.contains_key(&child_row) {
                child_frontier.push_back(child_row);
                p_child_push_count += 1;
            }
        }
        if let Some(t0) = t_child_subsample {
            p_child_subsample_ns += t0.elapsed().as_nanos() as u64;
        }
    }

    // Mark target cell: find the target column on the seed row
    // For the seed row, we need to mark the appropriate cell as target
    // and overwrite its value with the ground-truth from the task
    mark_target(
        is_target,
        is_null,
        numeric_values,
        timestamp_values,
        bool_values,
        categorical_embed_ids,
        db,
        task_idx,
        seed_idx,
        target_pos_in_seed,
        s,
    );

    // Build row-level directed adjacency for rows materialized in this sequence.
    let r = row_map.len();
    if scratch.out_neighbors.len() < r {
        scratch.out_neighbors.resize_with(r, Vec::new);
    }
    if scratch.in_neighbors.len() < r {
        scratch.in_neighbors.resize_with(r, Vec::new);
    }
    for ri in 0..r {
        scratch.out_neighbors[ri].clear();
        scratch.in_neighbors[ri].clear();
    }
    for (&row, &seq_id) in row_map.iter() {
        for &neighbor_raw in db.graph.outgoing_neighbors(row) {
            if let Some(&neighbor_seq_id) = row_map.get(&RowIdx(neighbor_raw)) {
                let ri = seq_id as usize;
                let rj = neighbor_seq_id as usize;
                scratch.out_neighbors[ri].push(rj);
                scratch.in_neighbors[rj].push(ri);
            }
        }
    }

    let row_count = row_map.len();
    *seq_csr_masks =
        encode_attention_masks_csr(seq_row_ids, column_ids, is_padding, s, r, scratch);

    if t_sequence_total.is_some() {
        PROFILE_SEQUENCE_COUNT.fetch_add(1, Ordering::Relaxed);
        PROFILE_PHASE1_FRONTIER_POP_NS.fetch_add(p_frontier_pop_ns, Ordering::Relaxed);
        PROFILE_PHASE1_VISITED_CHECK_NS.fetch_add(p_visited_check_ns, Ordering::Relaxed);
        PROFILE_PHASE1_ROW_MATERIALIZE_NS.fetch_add(p_row_materialize_ns, Ordering::Relaxed);
        PROFILE_PHASE1_OUT_NEIGHBOR_NS.fetch_add(p_out_neighbor_ns, Ordering::Relaxed);
        PROFILE_PHASE1_IN_NEIGHBOR_NS.fetch_add(p_in_neighbor_ns, Ordering::Relaxed);
        PROFILE_PHASE1_TEMPORAL_FILTER_NS.fetch_add(p_temporal_filter_ns, Ordering::Relaxed);
        PROFILE_PHASE1_CHILD_SUBSAMPLE_NS.fetch_add(p_child_subsample_ns, Ordering::Relaxed);
        PROFILE_PHASE1_ROWS_VISITED.fetch_add(p_rows_visited, Ordering::Relaxed);
        PROFILE_PHASE1_ROWS_SKIPPED_VISITED.fetch_add(p_rows_skipped_visited, Ordering::Relaxed);
        PROFILE_PHASE1_OUT_NEIGHBOR_COUNT.fetch_add(p_out_neighbor_count, Ordering::Relaxed);
        PROFILE_PHASE1_IN_NEIGHBOR_COUNT.fetch_add(p_in_neighbor_count, Ordering::Relaxed);
        PROFILE_PHASE1_TEMPORAL_CHECK_COUNT.fetch_add(p_temporal_check_count, Ordering::Relaxed);
        PROFILE_PHASE1_TEMPORAL_REJECT_COUNT.fetch_add(p_temporal_reject_count, Ordering::Relaxed);
        PROFILE_PHASE1_PARENT_PUSH_COUNT.fetch_add(p_parent_push_count, Ordering::Relaxed);
        PROFILE_PHASE1_CHILD_PUSH_COUNT.fetch_add(p_child_push_count, Ordering::Relaxed);
    }
    add_elapsed_ns(&PROFILE_SEQUENCE_TOTAL_NS, t_sequence_total);

    let _ = cell_count;
    SequenceStats {
        num_rows: row_count,
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
    db: &Database,
    task_idx: TaskIdx,
    seed_idx: usize,
    target_pos_in_seed: usize,
    s: usize,
) {
    let task_meta = db.task_metadata(task_idx);
    let task_view = db.task(task_idx);
    let target_stype = task_meta.target_stype;

    // Fallback position for derived tasks where target_stype doesn't appear
    // in the anchor table is precomputed to 0.
    let pos = target_pos_in_seed;
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
                copy_timestamp_values(timestamp_values, pos, ts);
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

fn target_pos_in_seed_row(db: &Database, task_idx: usize) -> usize {
    let task_meta = &db.metadata.task_metadata[task_idx];
    let target_stype = task_meta.target_stype;
    let anchor_table_meta = &db.metadata.table_metadata[task_meta.anchor_table.0 as usize];
    let col_start = anchor_table_meta.col_range.0.0;
    let col_end = anchor_table_meta.col_range.1.0;

    let mut pos_in_seq = 0usize;
    for global_col in col_start..col_end {
        let col_meta = &db.metadata.column_metadata[global_col as usize];
        if col_meta.stype == SemanticType::Ignored {
            continue;
        }
        if col_meta.stype == target_stype {
            return pos_in_seq;
        }
        pos_in_seq += 1;
    }
    0
}

/// Build per-sequence CSR attention masks (outbound, inbound, column) using position bitsets and OR.
/// Uses `row_position_bits` for row→positions, then outbound = self row ∪ out_neighbors' positions,
/// inbound = union of in_neighbors' positions; column = same-column position pairs.
fn encode_attention_masks_csr(
    seq_row_ids: &[u16],
    column_ids: &[i32],
    is_padding: &[u8],
    s: usize,
    r: usize,
    scratch: &mut SequenceScratch,
) -> SequenceCsrMasks {
    let t_encode = if profile_enabled() {
        Some(Instant::now())
    } else {
        None
    };
    let active_positions = &mut scratch.active_positions;
    active_positions.clear();
    active_positions.reserve(s);

    let col_pairs = &mut scratch.col_pairs;
    col_pairs.clear();
    col_pairs.reserve(s);
    let mut out_row_ptr = Vec::with_capacity(s + 1);
    let mut out_col_idx = Vec::new();
    out_row_ptr.push(0);
    let mut in_row_ptr = Vec::with_capacity(s + 1);
    let mut in_col_idx = Vec::new();
    in_row_ptr.push(0);

    let words_per_row = s.div_ceil(BITSET_WORD_BITS);
    if scratch.row_position_bits.len() < r {
        scratch.row_position_bits.resize_with(r, Vec::new);
    }
    for ri in 0..r {
        let bits = &mut scratch.row_position_bits[ri];
        if bits.len() != words_per_row {
            bits.resize(words_per_row, 0);
        }
        bits.fill(0);
    }
    for i in 0..s {
        if is_padding[i] == 1 {
            continue;
        }
        active_positions.push(i);
        let ri = seq_row_ids[i] as usize;
        if ri < r {
            bitset_set(&mut scratch.row_position_bits[ri], i);
        }
        col_pairs.push((column_ids[i], i));
    }

    let tmp_bits = &mut scratch.tmp_bits;
    if tmp_bits.len() != words_per_row {
        tmp_bits.resize(words_per_row, 0);
    }
    for i in 0..s {
        if is_padding[i] == 1 {
            out_row_ptr.push(out_col_idx.len() as u32);
            continue;
        }
        let ri = seq_row_ids[i] as usize;
        if ri >= r {
            out_row_ptr.push(out_col_idx.len() as u32);
            continue;
        }
        tmp_bits.copy_from_slice(&scratch.row_position_bits[ri]);
        for &rj in &scratch.out_neighbors[ri] {
            bitset_or_inplace(tmp_bits, &scratch.row_position_bits[rj]);
        }
        bitset_extend_sorted_indices(tmp_bits, &mut out_col_idx);
        out_row_ptr.push(out_col_idx.len() as u32);
    }

    for i in 0..s {
        if is_padding[i] == 1 {
            in_row_ptr.push(in_col_idx.len() as u32);
            continue;
        }
        let ri = seq_row_ids[i] as usize;
        if ri >= r {
            in_row_ptr.push(in_col_idx.len() as u32);
            continue;
        }
        tmp_bits.fill(0);
        for &rj in &scratch.in_neighbors[ri] {
            bitset_or_inplace(tmp_bits, &scratch.row_position_bits[rj]);
        }
        bitset_extend_sorted_indices(tmp_bits, &mut in_col_idx);
        in_row_ptr.push(in_col_idx.len() as u32);
    }
    if scratch.col_group_start.len() < s {
        scratch.col_group_start.resize(s, usize::MAX);
    }
    if scratch.col_group_end.len() < s {
        scratch.col_group_end.resize(s, usize::MAX);
    }
    for i in 0..s {
        scratch.col_group_start[i] = usize::MAX;
        scratch.col_group_end[i] = usize::MAX;
    }
    col_pairs.sort_unstable_by_key(|&(col, _)| col);
    let mut start = 0usize;
    while start < col_pairs.len() {
        let col = col_pairs[start].0;
        let mut end = start + 1;
        while end < col_pairs.len() && col_pairs[end].0 == col {
            end += 1;
        }
        for &(_, pos) in &col_pairs[start..end] {
            scratch.col_group_start[pos] = start;
            scratch.col_group_end[pos] = end;
        }
        start = end;
    }

    let mut col_row_ptr = Vec::with_capacity(s + 1);
    let mut col_col_idx = Vec::new();
    col_row_ptr.push(0);
    for i in 0..s {
        if is_padding[i] == 1 {
            col_row_ptr.push(col_col_idx.len() as u32);
            continue;
        }
        let g_start = scratch.col_group_start[i];
        let g_end = scratch.col_group_end[i];
        if g_start == usize::MAX {
            col_row_ptr.push(col_col_idx.len() as u32);
            continue;
        }
        for k in g_start..g_end {
            col_col_idx.push(col_pairs[k].1 as u16);
        }
        col_row_ptr.push(col_col_idx.len() as u32);
    }

    add_elapsed_ns(&PROFILE_ENCODE_MASK_NS, t_encode);
    SequenceCsrMasks {
        outbound_row_ptr: out_row_ptr,
        outbound_col_idx: out_col_idx,
        inbound_row_ptr: in_row_ptr,
        inbound_col_idx: in_col_idx,
        column_row_ptr: col_row_ptr,
        column_col_idx: col_col_idx,
    }
}

// ============================================================================
// Batch Construction
// ============================================================================

/// Build one batch: choose task, draw B seeds, Phase 1 parallel BFS + CSR masks, Phase 2 text dedup/gather/collate.
fn build_batch(
    db: &Database,
    seed_manager: &SeedManager,
    config: &SamplerConfig,
    split: Split,
    rng: &mut SmallRng,
    workspace: &mut BatchBuildWorkspace,
) -> Option<RawBatch> {
    let t_batch_total = if profile_enabled() {
        Some(Instant::now())
    } else {
        None
    };
    let b = config.batch_size as usize;
    let s = config.sequence_length as usize;

    let seeds_by_task = seed_manager.seeds_for_split(split);
    let tasks_with_seeds = seed_manager.tasks_with_seeds(split);

    if tasks_with_seeds.is_empty() {
        warn!("No tasks with seeds for {:?} split", split);
        return None;
    }

    // Pick a task (uniform random among those with seeds)
    let chosen_task_idx = tasks_with_seeds[rng.random_range(0..tasks_with_seeds.len())];
    let task_seeds = &seeds_by_task[chosen_task_idx];
    let task_meta = &db.metadata.task_metadata[chosen_task_idx];
    let target_pos_in_seed = target_pos_in_seed_row(db, chosen_task_idx);

    // Draw B seed indices (with replacement if needed)
    workspace.reset_for_batch(b);
    workspace
        .seed_indices
        .extend((0..b).map(|_| task_seeds[rng.random_range(0..task_seeds.len())]));

    // Pre-allocate final batch tensors and build each sequence directly in-place.
    let mut buffers = BatchBuffers::new(b, s);
    let seq_csr_masks = &mut workspace.seq_csr_masks;
    let seq_num_rows = &mut workspace.seq_num_rows;

    // Phase 1: Parallel BFS over B seeds with direct writes into final buffers.
    workspace
        .thread_seeds
        .extend((0..b).map(|_| rng.random::<u64>()));
    let seed_indices = &workspace.seed_indices;
    let thread_seeds = &workspace.thread_seeds;
    let task_idx = TaskIdx(chosen_task_idx as u32);
    let t_phase1 = if profile_enabled() {
        Some(Instant::now())
    } else {
        None
    };
    seq_num_rows
        .par_iter_mut()
        .zip(seed_indices.par_iter())
        .zip(thread_seeds.par_iter())
        .zip(buffers.semantic_types.par_chunks_mut(s))
        .zip(buffers.column_ids.par_chunks_mut(s))
        .zip(buffers.seq_row_ids.par_chunks_mut(s))
        .zip(buffers.numeric_values.par_chunks_mut(s))
        .zip(buffers.timestamp_values.par_chunks_mut(s * TIMESTAMP_DIM))
        .zip(buffers.bool_values.par_chunks_mut(s))
        .zip(buffers.categorical_embed_ids.par_chunks_mut(s))
        .zip(buffers.text_embed_ids.par_chunks_mut(s))
        .zip(buffers.is_null.par_chunks_mut(s))
        .zip(buffers.is_target.par_chunks_mut(s))
        .zip(buffers.is_padding.par_chunks_mut(s))
        .zip(seq_csr_masks.par_iter_mut())
        .for_each_init(SequenceScratch::default, |scratch, item| {
            let (item, seq_csr_masks_out) = item;
            let (item, is_padding) = item;
            let (item, is_target) = item;
            let (item, is_null) = item;
            let (item, text_embed_ids) = item;
            let (item, categorical_embed_ids) = item;
            let (item, bool_values) = item;
            let (item, timestamp_values) = item;
            let (item, numeric_values) = item;
            let (item, seq_row_ids) = item;
            let (item, column_ids) = item;
            let (item, semantic_types) = item;
            let ((num_rows_out, seed_idx), thread_seed) = item;

            let mut thread_rng = SmallRng::seed_from_u64(*thread_seed);
            let mut slot = SequenceSlotMut {
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
            };
            let stats = build_sequence_into(
                db,
                task_idx,
                *seed_idx,
                target_pos_in_seed,
                config,
                &mut thread_rng,
                scratch,
                &mut slot,
                seq_csr_masks_out,
            );
            *num_rows_out = stats.num_rows;
        });
    add_elapsed_ns(&PROFILE_BATCH_PHASE1_NS, t_phase1);

    // Phase 2: Text embedding dedup, gather, and remap to batch-local indices.
    let text_global_to_local = &mut workspace.text_global_to_local;
    let unique_text_indices = &mut workspace.unique_text_indices;

    let t_text_dedup = if profile_enabled() {
        Some(Instant::now())
    } else {
        None
    };
    for bi in 0..b {
        let cell_offset = bi * s;
        for pos in 0..s {
            let idx = cell_offset + pos;
            if buffers.is_padding[idx] == 1 {
                continue;
            }
            if buffers.semantic_types[idx] == SemanticType::Text as i8 && buffers.is_null[idx] == 0
            {
                let global_idx = buffers.text_embed_ids[idx];
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
    add_elapsed_ns(&PROFILE_BATCH_TEXT_DEDUP_NS, t_text_dedup);

    // Gather unique text embedding vectors from DB into text_batch_embeddings.
    let t_text_gather = if profile_enabled() {
        Some(Instant::now())
    } else {
        None
    };
    let u = unique_text_indices.len();
    let mut text_batch_embeddings = vec![f16::ZERO; u * EMBEDDING_DIM];
    for (local_idx, &global_idx) in unique_text_indices.iter().enumerate() {
        let emb = db.text_embeddings.get(global_idx);
        let start = local_idx * EMBEDDING_DIM;
        text_batch_embeddings[start..start + EMBEDDING_DIM].copy_from_slice(emb);
    }
    add_elapsed_ns(&PROFILE_BATCH_TEXT_GATHER_NS, t_text_gather);

    // Rewrite text_embed_ids from global to batch-local indices.
    let t_collate = if profile_enabled() {
        Some(Instant::now())
    } else {
        None
    };
    for bi in 0..b {
        let cell_offset = bi * s;
        for pos in 0..s {
            let idx = cell_offset + pos;
            if buffers.is_padding[idx] == 0
                && buffers.semantic_types[idx] == SemanticType::Text as i8
                && buffers.is_null[idx] == 0
            {
                let global = buffers.text_embed_ids[idx];
                buffers.text_embed_ids[idx] = *text_global_to_local.get(&global).unwrap_or(&0);
            }
        }
    }
    add_elapsed_ns(&PROFILE_BATCH_COLLATE_NS, t_collate);
    if t_batch_total.is_some() {
        PROFILE_BATCH_COUNT.fetch_add(1, Ordering::Relaxed);
    }
    add_elapsed_ns(&PROFILE_BATCH_TOTAL_NS, t_batch_total);

    Some(buffers.into_raw_batch(
        seq_csr_masks,
        b,
        s,
        seq_num_rows.iter().copied().max().unwrap_or(0),
        text_batch_embeddings,
        u,
        task_meta.target_stype as u8,
        chosen_task_idx as u32,
    ))
}

// ============================================================================
// Sampler (prefetch pipeline)
// ============================================================================

/// Main sampler: holds DB, seed sharding, and prefetch channels; used from Python and benchmark binaries.
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
    profile_reported: bool,
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
            profile_reported: false,
        })
    }

    /// Pull the next training batch. Blocks until one is available.
    pub fn next_train_batch(&self) -> Result<RawBatch, SamplerError> {
        let batch = match self.train_rx.try_recv() {
            Ok(batch) => batch,
            Err(TryRecvError::Empty) => {
                let t_wait = if profile_enabled() {
                    Some(Instant::now())
                } else {
                    None
                };
                let recv = self.train_rx.recv().map_err(|_| SamplerError::Shutdown)?;
                if t_wait.is_some() {
                    PROFILE_CONSUMER_BLOCKED_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                add_elapsed_ns(&PROFILE_CONSUMER_BLOCKED_NS, t_wait);
                recv
            }
            Err(TryRecvError::Disconnected) => return Err(SamplerError::Shutdown),
        };
        if profile_enabled() {
            PROFILE_CONSUMED_TRAIN_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        Ok(batch)
    }

    /// Pull the next validation batch. Blocks until one is available.
    pub fn next_val_batch(&self) -> Result<RawBatch, SamplerError> {
        let batch = match self.val_rx.try_recv() {
            Ok(batch) => batch,
            Err(TryRecvError::Empty) => {
                let t_wait = if profile_enabled() {
                    Some(Instant::now())
                } else {
                    None
                };
                let recv = self.val_rx.recv().map_err(|_| SamplerError::Shutdown)?;
                if t_wait.is_some() {
                    PROFILE_CONSUMER_BLOCKED_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                add_elapsed_ns(&PROFILE_CONSUMER_BLOCKED_NS, t_wait);
                recv
            }
            Err(TryRecvError::Disconnected) => return Err(SamplerError::Shutdown),
        };
        if profile_enabled() {
            PROFILE_CONSUMED_VAL_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        Ok(batch)
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

        if profile_enabled() && !self.profile_reported {
            log_profile_report();
            self.profile_reported = true;
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

/// Background producer: loop build_batch → send; runs in a dedicated thread per split.
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
    let mut workspace = BatchBuildWorkspace::new(config.batch_size as usize);

    debug!("Producer {:?} started", split);

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        match build_batch(&db, &seed_manager, &config, split, &mut rng, &mut workspace) {
            Some(batch) => {
                let send_ok = match tx.try_send(batch) {
                    Ok(()) => true,
                    Err(TrySendError::Full(batch)) => {
                        let t_wait = if profile_enabled() {
                            Some(Instant::now())
                        } else {
                            None
                        };
                        let send_result = tx.send(batch);
                        if t_wait.is_some() {
                            PROFILE_PRODUCER_BLOCKED_COUNT.fetch_add(1, Ordering::Relaxed);
                        }
                        add_elapsed_ns(&PROFILE_PRODUCER_BLOCKED_NS, t_wait);
                        send_result.is_ok()
                    }
                    Err(TrySendError::Disconnected(_)) => false,
                };
                if !send_ok {
                    break;
                }
                if profile_enabled() {
                    match split {
                        Split::Train => {
                            PROFILE_PRODUCED_TRAIN_COUNT.fetch_add(1, Ordering::Relaxed);
                        }
                        Split::Val => {
                            PROFILE_PRODUCED_VAL_COUNT.fetch_add(1, Ordering::Relaxed);
                        }
                        Split::Test => {}
                    }
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
