//! Batch data types and reusable buffers for the relational transformer sampler.
//!
//! Contains the output `RawBatch`, intermediate `BatchBuffers`, CSR mask structures,
//! and the `BatchBuildWorkspace` for buffer reuse across batch builds.

use half::f16;
use rustc_hash::FxHashMap;

use crate::common::TIMESTAMP_DIM;

/// A complete batch of B sequences, ready for transfer to Python/GPU.
///
/// All tensors are flat `Vec<T>`; shapes are `[B, S]`, `[B, S, TIMESTAMP_DIM]`, or
/// packed CSR. The Python binding converts these to NumPy arrays via zero-copy.
pub struct RawBatch {
    /// The semantic type of each cell. See [semantic_types.md](../semantic_types.md).
    pub semantic_types: Vec<u8>,
    /// Z-score normalized numerical values.
    pub numeric_values: Vec<f32>,
    /// Cyclic timestamp encoding: 7 sin/cos pairs + z-scored epoch microseconds.
    pub timestamp_values: Vec<f32>,
    /// Boolean values (not bit-packed, one byte per cell).
    pub bool_values: Vec<u8>,
    /// Used for lookup into the global categorical embedding table.
    pub categorical_embed_ids: Vec<u32>,
    /// Used for lookup into the global column-description embedding table.
    pub column_embed_ids: Vec<u32>,
    /// Used for lookup into the BATCH LOCAL text embedding table.
    pub text_embed_ids: Vec<u32>,

    // --- Validity and masking [B, S] ---
    pub is_null: Vec<u8>,
    pub is_padding: Vec<u8>,

    // --- Attention masks (CSR transport layer) ---
    /// Row ptrs [B, S+1]; seq_offsets [B+1]; col_idx packed.
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
    pub num_unique_texts: usize,
    pub text_batch_embeddings: Vec<f16>,

    // --- Batch-level metadata ---
    pub target_stype: u8,
    pub task_idx: u32,
}

/// Per-batch buffers filled during Phase 1; converted into `RawBatch` by `into_raw_batch`.
pub(crate) struct BatchBuffers {
    pub(crate) semantic_types: Vec<u8>,
    pub(crate) column_embed_ids: Vec<u32>,
    pub(crate) numeric_values: Vec<f32>,
    pub(crate) timestamp_values: Vec<f32>,
    pub(crate) bool_values: Vec<u8>,
    pub(crate) categorical_embed_ids: Vec<u32>,
    pub(crate) text_embed_ids: Vec<u32>,
    pub(crate) is_null: Vec<u8>,
    pub(crate) is_padding: Vec<u8>,
}

/// Per-sequence CSR attention masks (outbound, inbound, column) before packing into batch-wide arrays.
#[derive(Default)]
pub(crate) struct SequenceCsrMasks {
    pub(crate) outbound_row_ptr: Vec<u32>,
    pub(crate) outbound_col_idx: Vec<u16>,
    pub(crate) inbound_row_ptr: Vec<u32>,
    pub(crate) inbound_col_idx: Vec<u16>,
    pub(crate) column_row_ptr: Vec<u32>,
    pub(crate) column_col_idx: Vec<u16>,
}

impl SequenceCsrMasks {
    pub(crate) fn clear(&mut self) {
        self.outbound_row_ptr.clear();
        self.outbound_col_idx.clear();
        self.inbound_row_ptr.clear();
        self.inbound_col_idx.clear();
        self.column_row_ptr.clear();
        self.column_col_idx.clear();
    }
}

/// Mutable slices for one sequence's slot within `BatchBuffers` (one of B sequences).
pub(crate) struct SequenceSlotMut<'a> {
    pub(crate) semantic_types: &'a mut [u8],
    pub(crate) column_embed_ids: &'a mut [u32],
    pub(crate) numeric_values: &'a mut [f32],
    pub(crate) timestamp_values: &'a mut [f32],
    pub(crate) bool_values: &'a mut [u8],
    pub(crate) categorical_embed_ids: &'a mut [u32],
    pub(crate) text_embed_ids: &'a mut [u32],
    pub(crate) is_null: &'a mut [u8],
    pub(crate) is_padding: &'a mut [u8],
}

impl BatchBuffers {
    pub(crate) fn new(b: usize, s: usize) -> Self {
        let total_cells = b * s;
        let total_ts = b * s * TIMESTAMP_DIM;
        Self {
            semantic_types: vec![0u8; total_cells],
            column_embed_ids: vec![0u32; total_cells],
            numeric_values: vec![0.0f32; total_cells],
            timestamp_values: vec![0.0f32; total_ts],
            bool_values: vec![0u8; total_cells],
            categorical_embed_ids: vec![0u32; total_cells],
            text_embed_ids: vec![0u32; total_cells],
            is_null: vec![0u8; total_cells],
            is_padding: vec![1u8; total_cells],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn into_raw_batch(
        self,
        seq_csr_masks: &mut [SequenceCsrMasks],
        b: usize,
        s: usize,
        text_batch_embeddings: Vec<f16>,
        num_unique_texts: usize,
        target_stype: u8,
        task_idx: u32,
    ) -> RawBatch {
        let (outbound_csr_row_ptr, outbound_csr_seq_offsets, outbound_csr_col_idx) =
            pack_csr_masks(seq_csr_masks, b, s, CsrKind::Outbound);
        let (inbound_csr_row_ptr, inbound_csr_seq_offsets, inbound_csr_col_idx) =
            pack_csr_masks(seq_csr_masks, b, s, CsrKind::Inbound);
        let (column_csr_row_ptr, column_csr_seq_offsets, column_csr_col_idx) =
            pack_csr_masks(seq_csr_masks, b, s, CsrKind::Column);

        for seq in seq_csr_masks.iter_mut() {
            seq.clear();
        }

        RawBatch {
            semantic_types: self.semantic_types,
            column_embed_ids: self.column_embed_ids,
            numeric_values: self.numeric_values,
            timestamp_values: self.timestamp_values,
            bool_values: self.bool_values,
            categorical_embed_ids: self.categorical_embed_ids,
            text_embed_ids: self.text_embed_ids,
            is_null: self.is_null,
            is_padding: self.is_padding,
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

enum CsrKind {
    Outbound,
    Inbound,
    Column,
}

/// Pack per-sequence CSR masks of the given kind into batch-wide arrays.
fn pack_csr_masks(
    seq_csr_masks: &[SequenceCsrMasks],
    b: usize,
    s: usize,
    kind: CsrKind,
) -> (Vec<u32>, Vec<u32>, Vec<u16>) {
    let mut row_ptr = vec![0u32; b * (s + 1)];
    let mut seq_offsets = vec![0u32; b + 1];

    for (bi, seq) in seq_csr_masks.iter().enumerate() {
        let (rp, ci) = match kind {
            CsrKind::Outbound => (&seq.outbound_row_ptr, &seq.outbound_col_idx),
            CsrKind::Inbound => (&seq.inbound_row_ptr, &seq.inbound_col_idx),
            CsrKind::Column => (&seq.column_row_ptr, &seq.column_col_idx),
        };
        debug_assert_eq!(rp.len(), s + 1);
        seq_offsets[bi + 1] = seq_offsets[bi] + ci.len() as u32;
    }

    let mut col_idx = vec![0u16; seq_offsets[b] as usize];

    for (bi, seq) in seq_csr_masks.iter().enumerate() {
        let (rp, ci) = match kind {
            CsrKind::Outbound => (&seq.outbound_row_ptr, &seq.outbound_col_idx),
            CsrKind::Inbound => (&seq.inbound_row_ptr, &seq.inbound_col_idx),
            CsrKind::Column => (&seq.column_row_ptr, &seq.column_col_idx),
        };
        let row_off = bi * (s + 1);
        let col_off = seq_offsets[bi] as usize;
        row_ptr[row_off..row_off + (s + 1)].copy_from_slice(rp);
        col_idx[col_off..col_off + ci.len()].copy_from_slice(ci);
    }

    (row_ptr, seq_offsets, col_idx)
}

/// Reused across batch builds: per-sequence CSR mask buffers, seed indices, and text dedup state.
pub(crate) struct BatchBuildWorkspace {
    pub(crate) seq_csr_masks: Vec<SequenceCsrMasks>,
    pub(crate) seed_indices: Vec<usize>,
    pub(crate) thread_seeds: Vec<u64>,
    pub(crate) text_global_to_local: FxHashMap<u32, u32>,
    pub(crate) unique_text_indices: Vec<u32>,
}

impl BatchBuildWorkspace {
    pub(crate) fn new(batch_size: usize) -> Self {
        let mut seq_csr_masks = Vec::with_capacity(batch_size);
        seq_csr_masks.resize_with(batch_size, SequenceCsrMasks::default);
        Self {
            seq_csr_masks,
            seed_indices: Vec::with_capacity(batch_size),
            thread_seeds: Vec::with_capacity(batch_size),
            text_global_to_local: FxHashMap::default(),
            unique_text_indices: Vec::new(),
        }
    }

    pub(crate) fn reset_for_batch(&mut self, batch_size: usize) {
        if self.seq_csr_masks.len() != batch_size {
            self.seq_csr_masks
                .resize_with(batch_size, SequenceCsrMasks::default);
        }
        for seq in &mut self.seq_csr_masks {
            seq.clear();
        }
        self.seed_indices.clear();
        self.thread_seeds.clear();
        self.text_global_to_local.clear();
        self.unique_text_indices.clear();
    }
}
