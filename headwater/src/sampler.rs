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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam::channel::{self, Receiver, Sender, TryRecvError, TrySendError};
use half::f16;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use tracing::{debug, info, warn};

use crate::batch::{
    BatchBuffers, BatchBuildWorkspace, RawBatch, SequenceCsrMasks, SequenceSlotMut,
};
use crate::common::{
    ColumnIdx, ColumnSlice, Database, RowIdx, SemanticType, TIMESTAMP_DIM, TaskIdx,
};
use crate::embedder::EMBEDDING_DIM;
use crate::seed_manager::{SeedManager, Split};

/// Word size in bits for position bitsets (used in CSR mask encoding).
const BITSET_WORD_BITS: usize = u64::BITS as usize;

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
// Per-Sequence Builder
// ============================================================================

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
    col_pairs: Vec<(u32, usize)>,
    tmp_bits: Vec<u64>,
    col_group_start: Vec<usize>,
    col_group_end: Vec<usize>,
    /// Sequence-local row IDs used internally for CSR mask construction.
    seq_row_ids: Vec<u16>,
}

/// Build one sequence: BFS from task table row, materialize cells, build adjacency, encode CSR masks.
///
/// The BFS starts from the task table row (which contains obs_time + target cells),
/// follows the FK edge to the anchor row, then expands outward through the real
/// database graph. Columns listed in `columns_to_drop` are skipped to prevent
/// autocomplete leakage.
#[allow(clippy::too_many_arguments)]
fn build_sequence_into(
    db: &Database,
    task_idx: TaskIdx,
    seed_idx: usize,
    config: &SamplerConfig,
    rng: &mut SmallRng,
    scratch: &mut SequenceScratch,
    slot: &mut SequenceSlotMut<'_>,
    seq_csr_masks: &mut SequenceCsrMasks,
) {
    let s = config.sequence_length as usize;
    let task_meta = db.task_metadata(task_idx);
    let task_view = db.task(task_idx);

    let task_table_meta = &db.metadata.table_metadata[task_meta.task_table.0 as usize];
    let task_row = RowIdx(task_table_meta.row_range.0.0 + seed_idx as u32);

    let obs_time = task_view.observation_time(seed_idx);
    let use_temporal_filter = obs_time != i64::MAX;
    let obs_time_z_cutoff = if use_temporal_filter && db.metadata.global_ts_std_us > 0.0 {
        Some(((obs_time as f64) - db.metadata.global_ts_mean_us) / db.metadata.global_ts_std_us)
    } else {
        None
    };

    debug_assert_eq!(slot.semantic_types.len(), s);
    debug_assert_eq!(slot.column_embed_ids.len(), s);
    debug_assert_eq!(slot.numeric_values.len(), s);
    debug_assert_eq!(slot.timestamp_values.len(), s * TIMESTAMP_DIM);
    debug_assert_eq!(slot.bool_values.len(), s);
    debug_assert_eq!(slot.categorical_embed_ids.len(), s);
    debug_assert_eq!(slot.text_embed_ids.len(), s);
    debug_assert_eq!(slot.is_null.len(), s);
    debug_assert_eq!(slot.is_padding.len(), s);

    let semantic_types = &mut slot.semantic_types;
    let column_embed_ids = &mut slot.column_embed_ids;
    let numeric_values = &mut slot.numeric_values;
    let timestamp_values = &mut slot.timestamp_values;
    let bool_values = &mut slot.bool_values;
    let categorical_embed_ids = &mut slot.categorical_embed_ids;
    let text_embed_ids = &mut slot.text_embed_ids;
    let is_null = &mut slot.is_null;
    let is_padding = &mut slot.is_padding;
    let columns_to_drop = &task_meta.columns_to_drop;

    scratch.row_map.clear();
    scratch.parent_frontier.clear();
    scratch.child_frontier.clear();
    scratch.eligible_children.clear();
    scratch.seq_row_ids.clear();
    scratch.seq_row_ids.resize(s, 0u16);

    let row_map = &mut scratch.row_map;
    let seq_row_ids = &mut scratch.seq_row_ids;
    let mut cell_count: usize = 0;
    let parent_frontier = &mut scratch.parent_frontier;
    let child_frontier = &mut scratch.child_frontier;

    parent_frontier.push_back(task_row);

    loop {
        let row = match parent_frontier
            .pop_front()
            .or_else(|| child_frontier.pop_front())
        {
            Some(row) => row,
            None => break,
        };
        if cell_count >= s {
            break;
        }
        let next_seq_row_id = row_map.len() as u16;
        let is_new_row = if let std::collections::hash_map::Entry::Vacant(slot) = row_map.entry(row)
        {
            slot.insert(next_seq_row_id);
            true
        } else {
            false
        };
        if !is_new_row {
            continue;
        }
        let table_idx = db.row_table(row);
        let table_meta = &db.metadata.table_metadata[table_idx.0 as usize];
        let table_view = db.table(table_idx);
        let local_row = (row.0 - table_meta.row_range.0.0) as usize;

        let seq_row_id = row_map[&row];

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
            if columns_to_drop.contains(&ColumnIdx(global_col)) {
                continue;
            }

            let pos = cell_count;
            cell_count += 1;
            is_padding[pos] = 0;

            semantic_types[pos] = col_meta.stype as u8;
            column_embed_ids[pos] = global_col;
            seq_row_ids[pos] = seq_row_id;

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
        if cell_count >= s {
            break;
        }

        for &neighbor_raw in db.graph.outgoing_neighbors(row) {
            let neighbor = RowIdx(neighbor_raw);
            if !row_map.contains_key(&neighbor) {
                parent_frontier.push_back(neighbor);
            }
        }

        scratch.eligible_children.clear();
        let static_children = db.graph.incoming_static_neighbors(row);
        let (temporal_children, temporal_z) = db.graph.incoming_temporal_neighbors(row);

        for &child_raw in static_children {
            let child_row = RowIdx(child_raw);
            if !row_map.contains_key(&child_row) {
                scratch.eligible_children.push(child_row);
            }
        }

        let temporal_prefix_len = if use_temporal_filter {
            if let Some(z_cutoff) = obs_time_z_cutoff {
                temporal_z.partition_point(|&z| (z as f64) <= z_cutoff)
            } else {
                temporal_children.len()
            }
        } else {
            temporal_children.len()
        };

        for &child_raw in &temporal_children[..temporal_prefix_len] {
            let child_row = RowIdx(child_raw);
            if !row_map.contains_key(&child_row) {
                scratch.eligible_children.push(child_row);
            }
        }

        if scratch.eligible_children.len() > config.bfs_child_width as usize {
            scratch.eligible_children.shuffle(rng);
            scratch
                .eligible_children
                .truncate(config.bfs_child_width as usize);
        }
        for &child_row in &scratch.eligible_children {
            if !row_map.contains_key(&child_row) {
                child_frontier.push_back(child_row);
            }
        }
    }

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

    *seq_csr_masks = encode_attention_masks_csr(column_embed_ids, is_padding, s, r, scratch);
}

// ============================================================================
// Attention Mask Encoding
// ============================================================================

/// Build per-sequence CSR attention masks (outbound, inbound, column) using position bitsets and OR.
/// Uses `row_position_bits` for row->positions, then outbound = self row | out_neighbors' positions,
/// inbound = union of in_neighbors' positions; column = same-column position pairs.
fn encode_attention_masks_csr(
    column_embed_ids: &[u32],
    is_padding: &[u8],
    s: usize,
    r: usize,
    scratch: &mut SequenceScratch,
) -> SequenceCsrMasks {
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
        let ri = scratch.seq_row_ids[i] as usize;
        if ri < r {
            bitset_set(&mut scratch.row_position_bits[ri], i);
        }
        col_pairs.push((column_embed_ids[i], i));
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
        let ri = scratch.seq_row_ids[i] as usize;
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
        let ri = scratch.seq_row_ids[i] as usize;
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
    let b = config.batch_size as usize;
    let s = config.sequence_length as usize;

    let seeds_by_task = seed_manager.seeds_for_split(split);
    let tasks_with_seeds = seed_manager.tasks_with_seeds(split);

    if tasks_with_seeds.is_empty() {
        warn!("No tasks with seeds for {:?} split", split);
        return None;
    }

    let chosen_task_idx = tasks_with_seeds[rng.random_range(0..tasks_with_seeds.len())];
    let task_seeds = &seeds_by_task[chosen_task_idx];
    let task_meta = &db.metadata.task_metadata[chosen_task_idx];

    workspace.reset_for_batch(b);
    workspace
        .seed_indices
        .extend((0..b).map(|_| task_seeds[rng.random_range(0..task_seeds.len())]));

    let mut buffers = BatchBuffers::new(b, s);
    let seq_csr_masks = &mut workspace.seq_csr_masks;

    workspace
        .thread_seeds
        .extend((0..b).map(|_| rng.random::<u64>()));
    let seed_indices = &workspace.seed_indices;
    let thread_seeds = &workspace.thread_seeds;
    let task_idx = TaskIdx(chosen_task_idx as u32);
    seed_indices
        .par_iter()
        .zip(thread_seeds.par_iter())
        .zip(buffers.semantic_types.par_chunks_mut(s))
        .zip(buffers.column_embed_ids.par_chunks_mut(s))
        .zip(buffers.numeric_values.par_chunks_mut(s))
        .zip(buffers.timestamp_values.par_chunks_mut(s * TIMESTAMP_DIM))
        .zip(buffers.bool_values.par_chunks_mut(s))
        .zip(buffers.categorical_embed_ids.par_chunks_mut(s))
        .zip(buffers.text_embed_ids.par_chunks_mut(s))
        .zip(buffers.is_null.par_chunks_mut(s))
        .zip(buffers.is_padding.par_chunks_mut(s))
        .zip(seq_csr_masks.par_iter_mut())
        .for_each_init(SequenceScratch::default, |scratch, item| {
            let (item, seq_csr_masks_out) = item;
            let (item, is_padding) = item;
            let (item, is_null) = item;
            let (item, text_embed_ids) = item;
            let (item, categorical_embed_ids) = item;
            let (item, bool_values) = item;
            let (item, timestamp_values) = item;
            let (item, numeric_values) = item;
            let (item, column_embed_ids) = item;
            let (item, semantic_types) = item;
            let (seed_idx, thread_seed) = item;

            let mut thread_rng = SmallRng::seed_from_u64(*thread_seed);
            let mut slot = SequenceSlotMut {
                semantic_types,
                column_embed_ids,
                numeric_values,
                timestamp_values,
                bool_values,
                categorical_embed_ids,
                text_embed_ids,
                is_null,
                is_padding,
            };
            build_sequence_into(
                db,
                task_idx,
                *seed_idx,
                config,
                &mut thread_rng,
                scratch,
                &mut slot,
                seq_csr_masks_out,
            );
        });

    let (text_batch_embeddings, num_unique_texts) =
        gather_text_embeddings(&mut buffers, workspace, db, b, s);

    Some(buffers.into_raw_batch(
        &mut workspace.seq_csr_masks,
        b,
        s,
        text_batch_embeddings,
        num_unique_texts,
        task_meta.target_stype as u8,
        chosen_task_idx as u32,
    ))
}

/// Phase 2: Text embedding dedup, gather, and remap to batch-local indices.
/// Returns (text_batch_embeddings, num_unique_texts).
fn gather_text_embeddings(
    buffers: &mut BatchBuffers,
    workspace: &mut BatchBuildWorkspace,
    db: &Database,
    b: usize,
    s: usize,
) -> (Vec<f16>, usize) {
    let text_global_to_local = &mut workspace.text_global_to_local;
    let unique_text_indices = &mut workspace.unique_text_indices;

    for bi in 0..b {
        let cell_offset = bi * s;
        for pos in 0..s {
            let idx = cell_offset + pos;
            if buffers.is_padding[idx] == 1 {
                continue;
            }
            if buffers.semantic_types[idx] == SemanticType::Text as u8 && buffers.is_null[idx] == 0
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

    let u = unique_text_indices.len();
    let mut text_batch_embeddings = vec![f16::ZERO; u * EMBEDDING_DIM];
    for (local_idx, &global_idx) in unique_text_indices.iter().enumerate() {
        let emb = db.text_embeddings.get(global_idx);
        let start = local_idx * EMBEDDING_DIM;
        text_batch_embeddings[start..start + EMBEDDING_DIM].copy_from_slice(emb);
    }

    for bi in 0..b {
        let cell_offset = bi * s;
        for pos in 0..s {
            let idx = cell_offset + pos;
            if buffers.is_padding[idx] == 0
                && buffers.semantic_types[idx] == SemanticType::Text as u8
                && buffers.is_null[idx] == 0
            {
                let global = buffers.text_embed_ids[idx];
                buffers.text_embed_ids[idx] = *text_global_to_local.get(&global).unwrap_or(&0);
            }
        }
    }

    (text_batch_embeddings, u)
}

// ============================================================================
// Sampler (prefetch pipeline)
// ============================================================================

/// Mutable state for on-demand validation batch construction.
struct ValBatchState {
    rng: SmallRng,
    workspace: BatchBuildWorkspace,
}

/// Main sampler: holds DB, seed sharding, train prefetch channel, and on-demand val state.
#[allow(dead_code)]
pub struct Sampler {
    db: Arc<Database>,
    config: SamplerConfig,
    seed_manager: Arc<SeedManager>,
    train_rx: Receiver<RawBatch>,
    shutdown: Arc<AtomicBool>,
    train_handle: Option<std::thread::JoinHandle<()>>,
    val_state: Mutex<ValBatchState>,
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

        let (train_tx, train_rx) = channel::bounded(config.num_prefetch);
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

        let val_rng = SmallRng::seed_from_u64(
            config
                .seed
                .wrapping_add(config.rank as u64 * 1000)
                .wrapping_add(1),
        );
        let val_state = Mutex::new(ValBatchState {
            rng: val_rng,
            workspace: BatchBuildWorkspace::new(config.batch_size as usize),
        });

        Ok(Self {
            db,
            config,
            seed_manager,
            train_rx,
            shutdown,
            train_handle: Some(train_handle),
            val_state,
        })
    }

    /// Pull the next training batch. Blocks until one is available.
    pub fn next_train_batch(&self) -> Result<RawBatch, SamplerError> {
        let batch = match self.train_rx.try_recv() {
            Ok(batch) => batch,
            Err(TryRecvError::Empty) => self.train_rx.recv().map_err(|_| SamplerError::Shutdown)?,
            Err(TryRecvError::Disconnected) => return Err(SamplerError::Shutdown),
        };
        Ok(batch)
    }

    /// Build and return the next validation batch on demand.
    pub fn next_val_batch(&self) -> Result<RawBatch, SamplerError> {
        let mut state = self.val_state.lock().map_err(|_| SamplerError::Shutdown)?;
        let ValBatchState { rng, workspace } = &mut *state;
        let batch = build_batch(
            &self.db,
            &self.seed_manager,
            &self.config,
            Split::Val,
            rng,
            workspace,
        )
        .ok_or(SamplerError::NoSeeds)?;
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

    /// Get a reference to the sampler configuration.
    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Get a reference to the loaded database.
    pub fn database(&self) -> &Database {
        &self.db
    }

    /// Shut down the sampler: signal producer threads to stop, drain channels,
    /// and join threads.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);

        while self.train_rx.try_recv().is_ok() {}

        if let Some(h) = self.train_handle.take() {
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
    #[error("no seeds available for the requested split")]
    NoSeeds,
}

/// Background producer: loop build_batch -> send; runs in a dedicated thread per split.
fn producer_loop(
    db: Arc<Database>,
    seed_manager: Arc<SeedManager>,
    config: SamplerConfig,
    split: Split,
    tx: Sender<RawBatch>,
    shutdown: Arc<AtomicBool>,
) {
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
                    Err(TrySendError::Full(batch)) => tx.send(batch).is_ok(),
                    Err(TrySendError::Disconnected(_)) => false,
                };
                if !send_ok {
                    break;
                }
            }
            None => {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }

    debug!("Producer {:?} exiting", split);
}
