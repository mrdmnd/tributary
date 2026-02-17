//! Common types and constants used throughout the project.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use half::f16;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::embedder::EMBEDDING_DIM;

/// Database columns have "data types" — the primitive types a column is stored as.
/// They also have "semantic types" — the *meaning* of the data.
/// The preprocessor assigns semantic types based on a human-annotated metadata JSON file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SemanticType {
    Identifier = 0,
    Numerical = 1,
    Timestamp = 2,
    Boolean = 3,
    Categorical = 4,
    Text = 5,
    Ignored = 6,
}

// ============================================================================
// Index NewTypes
// ============================================================================
/// Global table index in the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TableIdx(pub u32);

/// Global column index (unique across all tables) in the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ColumnIdx(pub u32);

/// Global row index (unique across all tables) in the database.
/// Uses `u32` to match the CSR graph representation, supporting up to ~4B total rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RowIdx(pub u32);

/// Global task index in the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskIdx(pub u32);

/// Index into the interned vocabulary embedding table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EmbeddingIdx(pub u32);

// ============================================================================
// Metadata / Schema Objects
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMetadata {
    pub name: String,
    pub stype: SemanticType,
    pub fkey_target_col: Option<ColumnIdx>,

    /// Pre-computed text embedding for this column's name (+ table name + description).
    ///
    /// The embedded text is `"<col_name> of <table_name>: <description>"` (or
    /// `"<col_name> of <table_name>"` if no description). This vector has length
    /// [`EMBEDDING_DIM`] and contains L2-normalized f16 values.
    ///
    /// **Not serialized to JSON.** Column embeddings are stored in a separate
    /// binary file (`column_embeddings.bin`) and populated at load time by
    /// [`Database::load`].
    ///
    /// At training time, the full set of column embeddings is uploaded to GPU
    /// shared memory once and kept resident for the entire run.
    #[serde(skip)]
    pub embedding: Vec<f16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableMetadata {
    pub name: String,
    pub col_range: (ColumnIdx, ColumnIdx), // The range of global column indices for the table [start, end)
    pub row_range: (RowIdx, RowIdx), // The range of global row indices for the table [start, end)
    pub pkey_col: Option<ColumnIdx>, // The primary key column index for the table, if it has one.
}

/// Metadata for a single prediction task.
///
/// Each task is defined by a SQL query that materializes ground-truth labels as
/// `(anchor_row, observation_time, target_value)` tuples. The preprocessor
/// executes the query and stores the results in a `<task_name>.bin` file.
///
/// During training, the sampler picks a task, draws seeds from its anchor rows,
/// builds BFS subgraphs, and constructs batches. The target column comes from
/// the materialized task table, not from any column in the original database tables
/// (though for simple cell-masking tasks, they may coincide).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    /// Human-readable task name (e.g. "predict_vote_type").
    pub name: String,

    /// Which table anchors the subgraph sampling for this task.
    pub anchor_table: TableIdx,

    /// Semantic type of the prediction target.
    /// Only `Numerical`, `Categorical`, `Boolean`, and `Timestamp` are valid here.
    pub target_stype: SemanticType,

    /// Number of seed rows (anchor_row, target_value pairs) in the materialized task table.
    pub num_seeds: u32,

    /// Statistics for the target column, used for normalization and loss computation.
    /// Variant matches `target_stype` (e.g. `ColumnStats::Numerical` for numerical targets).
    pub target_stats: ColumnStats,
}

/// Per-column statistics, determined during preprocessing.
/// Each variant carries only the stats meaningful for that semantic type.
/// This lives in the DatabaseMetadata object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnStats {
    /// Identifiers: we only care about null rate.
    Identifier { num_nulls: u64 },

    /// Numerical columns: z-score normalization parameters.
    Numerical {
        num_nulls: u64,
        min: f64,
        max: f64,
        mean: f64,
        std: f64,
    },

    /// Timestamps: normalization parameters for cyclic encoding.
    ///
    /// `min` and `max` are stored as **epoch microseconds** (`i64`), matching
    /// Arrow's `TimestampMicrosecondArray` internal representation. Convert to
    /// chrono via `chrono::DateTime::from_timestamp_micros(value)`.
    ///
    /// `mean` and `std` are `f64` epoch microseconds to avoid overflow during
    /// accumulation and to represent fractional microseconds.
    Timestamp {
        num_nulls: u64,
        /// Earliest timestamp in the column (epoch microseconds).
        min_us: i64,
        /// Latest timestamp in the column (epoch microseconds).
        max_us: i64,
        /// Arithmetic mean of all non-null timestamps (epoch microseconds, f64).
        mean_us: f64,
        /// Standard deviation of all non-null timestamps (microseconds, f64).
        std_us: f64,
    },

    /// Booleans: just counts.
    Boolean {
        num_nulls: u64,
        num_true: u64,
        num_false: u64,
    },

    /// Categoricals: the vocabulary of distinct values.
    /// Cardinality is just `categories.len()`.
    Categorical {
        num_nulls: u64,
        categories: Vec<String>,
    },

    /// Text: null count
    Text { num_nulls: u64 },

    /// Ignored columns carry no stats.
    Ignored,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseMetadata {
    pub table_metadata: Vec<TableMetadata>, // List of tables in the database, indexed by TableIdx
    pub column_metadata: Vec<ColumnMetadata>, // List of columns in the database, indexed by ColumnIdx
    pub column_stats: Vec<ColumnStats>,       // List of column stats, indexed by ColumnIdx
    pub task_metadata: Vec<TaskMetadata>,     // List of prediction tasks, indexed by TaskIdx

    /// Global mean of all non-null timestamp values across every table (epoch microseconds).
    /// Used for the z-score component of timestamp encoding so that all timestamps are
    /// on the same scale regardless of which column they came from.
    pub global_ts_mean_us: f64,
    /// Global standard deviation of all non-null timestamp values across every table (microseconds).
    pub global_ts_std_us: f64,
}

// ============================================================================
// Topology (Graph) Components
// ============================================================================

/// CSR (Compressed Sparse Row) representation of a directed graph.
///
/// This owned type is used by the **preprocessor** for graph construction
/// ([`from_sorted_edges`](Self::from_sorted_edges), [`transpose`](Self::transpose),
/// [`build_pair`](Self::build_pair)). For zero-copy mmap'd reading at training
/// time, see [`GraphView`].
#[derive(Debug, Clone)]
pub struct CsrGraph {
    /// `row_ptr[i]` is the start offset in `col_idx` for node `i`.
    /// `row_ptr[i+1] - row_ptr[i]` is the degree of node `i`.
    /// Length = `num_nodes + 1`.
    pub row_ptr: Vec<u32>,
    /// Packed neighbor lists. Neighbors of node `i` are
    /// `col_idx[row_ptr[i]..row_ptr[i+1]]`.
    pub col_idx: Vec<u32>,
}

impl CsrGraph {
    /// Number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.row_ptr.len().saturating_sub(1)
    }

    /// Number of edges in the graph.
    pub fn num_edges(&self) -> usize {
        self.col_idx.len()
    }

    /// Return the neighbors of `node` as a slice.
    pub fn neighbors(&self, node: u32) -> &[u32] {
        let start = self.row_ptr[node as usize] as usize;
        let end = self.row_ptr[node as usize + 1] as usize;
        &self.col_idx[start..end]
    }

    /// Degree (number of neighbors) of `node`.
    pub fn degree(&self, node: u32) -> u32 {
        self.row_ptr[node as usize + 1] - self.row_ptr[node as usize]
    }

    /// Build a CSR from edges that are **already sorted by source node**.
    ///
    /// # Panics (debug builds)
    /// - If `edges` is not sorted by source node.
    /// - If the number of edges exceeds `u32::MAX`.
    pub fn from_sorted_edges(num_nodes: usize, edges: &[(u32, u32)]) -> Self {
        debug_assert!(
            edges.windows(2).all(|w| w[0].0 <= w[1].0),
            "edges must be sorted by source node"
        );
        debug_assert!(
            edges.len() <= u32::MAX as usize,
            "edge count {} exceeds u32::MAX",
            edges.len()
        );

        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        let mut col_idx = Vec::with_capacity(edges.len());
        let mut current_node = 0u32;
        row_ptr.push(0);

        for &(src, dst) in edges {
            // Fill in row_ptr entries for nodes with no outgoing edges.
            while current_node < src {
                row_ptr.push(col_idx.len() as u32);
                current_node += 1;
            }
            col_idx.push(dst);
        }

        // Pad trailing nodes that have no edges.
        while row_ptr.len() <= num_nodes {
            row_ptr.push(col_idx.len() as u32);
        }

        Self { row_ptr, col_idx }
    }

    /// Compute the transpose (reverse all edge directions) in O(V + E) time
    /// using a counting-sort, without allocating an intermediate edge list.
    pub fn transpose(&self) -> Self {
        let num_nodes = self.num_nodes();
        let num_edges = self.num_edges();

        // Pass 1: count in-degree of each node → shifted by one so the
        // prefix sum directly produces row_ptr.
        let mut row_ptr = vec![0u32; num_nodes + 1];
        for &dst in &self.col_idx {
            row_ptr[dst as usize + 1] += 1;
        }
        // Prefix sum.
        for i in 1..=num_nodes {
            row_ptr[i] += row_ptr[i - 1];
        }

        // Pass 2: scatter source nodes into col_idx.
        let mut col_idx = vec![0u32; num_edges];
        let mut write_cursor = row_ptr[..num_nodes].to_vec();
        for src in 0..num_nodes {
            let start = self.row_ptr[src] as usize;
            let end = self.row_ptr[src + 1] as usize;
            for &dst in &self.col_idx[start..end] {
                let pos = write_cursor[dst as usize] as usize;
                col_idx[pos] = src as u32;
                write_cursor[dst as usize] += 1;
            }
        }

        Self { row_ptr, col_idx }
    }

    /// Build outgoing and incoming CSR graphs from an unsorted edge list.
    ///
    /// Each edge `(src, dst)` represents a directed edge from `src` to `dst`.
    /// The edges are sorted in-place. The incoming graph is computed via an
    /// O(V + E) CSR transpose — no second sort is needed.
    ///
    /// Returns `(outgoing, incoming)`.
    pub fn build_pair(num_nodes: usize, mut edges: Vec<(u32, u32)>) -> (Self, Self) {
        edges.sort_unstable_by_key(|&(src, _)| src);
        let outgoing = Self::from_sorted_edges(num_nodes, &edges);
        let incoming = outgoing.transpose();
        (outgoing, incoming)
    }
}

// ============================================================================
// Graph View (zero-copy mmap'd CSR)
// ============================================================================

/// Header size for `graph.bin`: four `u32` values.
const GRAPH_HEADER_U32S: usize = 4;

/// A zero-copy, memory-mapped view of a bidirectional CSR graph.
///
/// Backed by a flat binary file (`graph.bin`) containing a 16-byte header
/// followed by four packed `u32` arrays (outgoing row_ptr, outgoing col_idx,
/// incoming row_ptr, incoming col_idx). Multiple processes mmapping the same
/// file share the same physical pages via the OS page cache.
///
/// For **construction** (preprocessing), see [`CsrGraph`] and [`write_graph_bin`].
///
/// ## File layout
///
/// ```text
/// Header (16 bytes):
///   [0] num_nodes  (u32) — outgoing
///   [1] num_edges  (u32) — outgoing
///   [2] num_nodes  (u32) — incoming (must match [0])
///   [3] num_edges  (u32) — incoming (must match [1])
/// Data (packed u32 arrays):
///   outgoing.row_ptr  [num_nodes + 1 elements]
///   outgoing.col_idx  [num_edges elements]
///   incoming.row_ptr  [num_nodes + 1 elements]
///   incoming.col_idx  [num_edges elements]
/// ```
pub struct GraphView {
    /// Keeps the memory map alive for the lifetime of the view.
    _mmap: Arc<Mmap>,
    /// Outgoing CSR row pointers (`num_nodes + 1` elements).
    out_row_ptr: &'static [u32],
    /// Outgoing CSR column indices (packed neighbor lists).
    out_col_idx: &'static [u32],
    /// Incoming CSR row pointers (`num_nodes + 1` elements).
    in_row_ptr: &'static [u32],
    /// Incoming CSR column indices (packed neighbor lists).
    in_col_idx: &'static [u32],
    /// Number of nodes in the graph.
    num_nodes: usize,
    /// Number of directed edges in the graph.
    num_edges: usize,
}

impl GraphView {
    /// Create a [`GraphView`] from a memory-mapped `graph.bin` file.
    ///
    /// Parses the 16-byte header, validates sizes, and creates four `&[u32]`
    /// slices pointing directly into the mmap'd pages.
    ///
    /// # Panics
    /// Panics if the file is too small for the header or if the total size
    /// does not match the header's declared dimensions.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Self {
        let byte_len = mmap.len();
        let header_bytes = GRAPH_HEADER_U32S * std::mem::size_of::<u32>();
        assert!(
            byte_len >= header_bytes,
            "graph.bin too small for header ({byte_len} < {header_bytes} bytes)",
        );

        // SAFETY: Mmap is page-aligned (>= 4-byte aligned). Header is 4 u32s.
        let header: &[u32] =
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const u32, GRAPH_HEADER_U32S) };
        let num_nodes = header[0] as usize;
        let num_edges = header[1] as usize;
        assert_eq!(
            header[2] as usize, num_nodes,
            "incoming num_nodes ({}) != outgoing num_nodes ({num_nodes})",
            header[2],
        );
        assert_eq!(
            header[3] as usize, num_edges,
            "incoming num_edges ({}) != outgoing num_edges ({num_edges})",
            header[3],
        );

        let row_ptr_len = num_nodes + 1;
        let expected_u32s = GRAPH_HEADER_U32S + 2 * row_ptr_len + 2 * num_edges;
        let expected_bytes = expected_u32s * std::mem::size_of::<u32>();
        assert_eq!(
            byte_len, expected_bytes,
            "graph.bin size mismatch: expected {expected_bytes} bytes \
             ({num_nodes} nodes, {num_edges} edges), got {byte_len}",
        );

        // SAFETY: The mmap is read-only and immutable. The Arc keeps the
        // backing memory alive for as long as this struct exists. We extend
        // the slice lifetimes to 'static because the Arc prevents deallocation.
        let base = unsafe { (mmap.as_ptr() as *const u32).add(GRAPH_HEADER_U32S) };
        let (out_row_ptr, out_col_idx, in_row_ptr, in_col_idx) = unsafe {
            let out_rp = std::slice::from_raw_parts(base, row_ptr_len);
            let out_ci = std::slice::from_raw_parts(base.add(row_ptr_len), num_edges);
            let in_rp = std::slice::from_raw_parts(base.add(row_ptr_len + num_edges), row_ptr_len);
            let in_ci =
                std::slice::from_raw_parts(base.add(2 * row_ptr_len + num_edges), num_edges);
            (out_rp, out_ci, in_rp, in_ci)
        };

        Self {
            _mmap: mmap,
            out_row_ptr,
            out_col_idx,
            in_row_ptr,
            in_col_idx,
            num_nodes,
            num_edges,
        }
    }

    /// Total number of nodes (rows) in the graph.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Total number of directed edges in the graph.
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Rows that `row` references via foreign keys (forward FK direction).
    ///
    /// Returns a slice of raw `u32` row indices for zero-overhead access.
    pub fn outgoing_neighbors(&self, row: RowIdx) -> &[u32] {
        let start = self.out_row_ptr[row.0 as usize] as usize;
        let end = self.out_row_ptr[row.0 as usize + 1] as usize;
        &self.out_col_idx[start..end]
    }

    /// Rows that reference `row` via foreign keys (reverse FK direction).
    ///
    /// Returns a slice of raw `u32` row indices for zero-overhead access.
    pub fn incoming_neighbors(&self, row: RowIdx) -> &[u32] {
        let start = self.in_row_ptr[row.0 as usize] as usize;
        let end = self.in_row_ptr[row.0 as usize + 1] as usize;
        &self.in_col_idx[start..end]
    }

    /// Number of outgoing edges (forward FK references) from `row`.
    pub fn out_degree(&self, row: RowIdx) -> u32 {
        self.out_row_ptr[row.0 as usize + 1] - self.out_row_ptr[row.0 as usize]
    }

    /// Number of incoming edges (reverse FK references) to `row`.
    pub fn in_degree(&self, row: RowIdx) -> u32 {
        self.in_row_ptr[row.0 as usize + 1] - self.in_row_ptr[row.0 as usize]
    }
}

/// Write a bidirectional CSR graph to a flat binary file (`graph.bin`).
///
/// See [`GraphView`] for the file layout. The outgoing and incoming graphs
/// must have the same number of nodes and edges (the incoming graph is
/// typically produced by [`CsrGraph::transpose`]).
///
/// # Panics
/// Panics if the outgoing and incoming graphs have different dimensions.
pub fn write_graph_bin(
    outgoing: &CsrGraph,
    incoming: &CsrGraph,
    path: &Path,
) -> std::io::Result<()> {
    assert_eq!(
        outgoing.num_nodes(),
        incoming.num_nodes(),
        "outgoing num_nodes ({}) != incoming num_nodes ({})",
        outgoing.num_nodes(),
        incoming.num_nodes(),
    );
    assert_eq!(
        outgoing.num_edges(),
        incoming.num_edges(),
        "outgoing num_edges ({}) != incoming num_edges ({})",
        outgoing.num_edges(),
        incoming.num_edges(),
    );

    let num_nodes = outgoing.num_nodes() as u32;
    let num_edges = outgoing.num_edges() as u32;

    let mut w = BufWriter::new(File::create(path)?);

    // Header: [num_nodes, num_edges, num_nodes, num_edges]
    w.write_all(&num_nodes.to_ne_bytes())?;
    w.write_all(&num_edges.to_ne_bytes())?;
    w.write_all(&num_nodes.to_ne_bytes())?;
    w.write_all(&num_edges.to_ne_bytes())?;

    /// Write a `&[u32]` slice as raw bytes.
    fn write_u32_slice(w: &mut BufWriter<File>, s: &[u32]) -> std::io::Result<()> {
        // SAFETY: u32 has no padding and a well-defined memory layout.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                s.as_ptr() as *const u8,
                s.len() * std::mem::size_of::<u32>(),
            )
        };
        w.write_all(bytes)
    }

    write_u32_slice(&mut w, &outgoing.row_ptr)?;
    write_u32_slice(&mut w, &outgoing.col_idx)?;
    write_u32_slice(&mut w, &incoming.row_ptr)?;
    write_u32_slice(&mut w, &incoming.col_idx)?;

    w.flush()?;
    Ok(())
}

// ============================================================================
// Timestamp Encoding Constants
// ============================================================================

/// Number of cyclic component pairs in the timestamp encoding.
/// Levels: second_of_minute, minute_of_hour, hour_of_day,
///         day_of_week, day_of_month, month_of_year, day_of_year.
pub const TIMESTAMP_CYCLIC_PAIRS: usize = 7;

/// Total float dimension per timestamp cell: 7 sin/cos pairs + 1 z-scored value of epoch microseconds.
pub const TIMESTAMP_DIM: usize = TIMESTAMP_CYCLIC_PAIRS * 2 + 1; // = 15

/// Indices into the 15-element timestamp feature vector.
/// Each cyclic level occupies two consecutive slots: [sin, cos].
pub const TS_SECOND_OF_MINUTE: usize = 0; // slots 0, 1
pub const TS_MINUTE_OF_HOUR: usize = 2; // slots 2, 3
pub const TS_HOUR_OF_DAY: usize = 4; // slots 4, 5
pub const TS_DAY_OF_WEEK: usize = 6; // slots 6, 7
pub const TS_DAY_OF_MONTH: usize = 8; // slots 8, 9
pub const TS_MONTH_OF_YEAR: usize = 10; // slots 10, 11
pub const TS_DAY_OF_YEAR: usize = 12; // slots 12, 13
pub const TS_ZSCORE_EPOCH: usize = 14; // slot 14

// ============================================================================
// Bitmap & Alignment Utilities
// ============================================================================

/// Number of bytes needed to store `num_bits` packed bits (little-endian bit order).
pub const fn packed_bit_bytes(num_bits: usize) -> usize {
    (num_bits + 7) / 8
}

/// Check if bit `index` is set in a packed little-endian bitmap.
///
/// Bit `i` is stored as bit `(i % 8)` of byte `(i / 8)`.
/// Returns `true` if the bit is 1, `false` if 0.
#[inline]
pub fn bit_is_set(bitmap: &[u8], index: usize) -> bool {
    bitmap[index / 8] & (1 << (index % 8)) != 0
}

/// Round `offset` up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two.
#[inline]
const fn align_up(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

/// Alignment guarantee for all sections within table binary files.
const TABLE_ALIGNMENT: usize = 8;

// ============================================================================
// Vocab Embedding Table
// ============================================================================

/// Interned vocabulary embedding table backed by a memory-mapped flat binary file.
///
/// Contains embeddings for categorical and text cell **values** only.
/// Column-name embeddings are stored separately in [`ColumnMetadata::embedding`].
///
/// The file (`vocab_embeddings.bin`) contains a packed `[num_embeddings, EMBEDDING_DIM]`
/// array of `f16` values (native byte order). The mmap is read-only and shared
/// across processes.
///
/// Indices stored in [`ColumnSlice::Embedded`] point directly into this table
/// (0-based, no offset).
pub struct VocabEmbeddingTable {
    /// Keeps the memory map alive for the lifetime of the table.
    _mmap: Arc<Mmap>,
    /// View into the mmap as a flat f16 slice.
    ///
    /// # Safety
    /// Lifetime is logically tied to `_mmap` which is kept alive by Arc.
    /// The `'static` lifetime is safe because the Arc prevents deallocation.
    data: &'static [f16],
    /// Number of distinct embeddings in the table.
    num_embeddings: usize,
}

impl VocabEmbeddingTable {
    /// Create a [`VocabEmbeddingTable`] from a memory-mapped flat binary file.
    ///
    /// The file must contain `num_embeddings * EMBEDDING_DIM` packed `f16` values
    /// (i.e., its byte length must be a multiple of `EMBEDDING_DIM * 2`).
    ///
    /// # Panics
    /// Panics if the file size is not a multiple of `EMBEDDING_DIM * sizeof(f16)`.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Self {
        let byte_len = mmap.len();
        let stride = EMBEDDING_DIM * std::mem::size_of::<f16>();
        assert!(
            byte_len % stride == 0,
            "vocab embedding file size ({byte_len} bytes) is not a multiple of \
             EMBEDDING_DIM ({EMBEDDING_DIM}) * sizeof(f16) = {stride} bytes",
        );
        let num_embeddings = byte_len / stride;
        let num_f16s = num_embeddings * EMBEDDING_DIM;

        // SAFETY: The mmap is read-only and immutable. The Arc keeps the
        // backing memory alive for as long as this struct exists. We extend
        // the slice lifetime to 'static because the Arc prevents deallocation.
        // f16 is repr(transparent) over u16 (2-byte aligned), and mmap
        // pointers are page-aligned, so alignment is satisfied.
        let data: &'static [f16] = unsafe {
            let ptr = mmap.as_ptr() as *const f16;
            std::slice::from_raw_parts(ptr, num_f16s)
        };

        Self {
            _mmap: mmap,
            data,
            num_embeddings,
        }
    }

    /// Number of distinct vocabulary embeddings in the table.
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Look up a vocabulary embedding vector by index.
    ///
    /// Returns a slice of [`EMBEDDING_DIM`] `f16` values.
    ///
    /// # Panics
    /// Panics if `idx` is out of bounds.
    pub fn get(&self, idx: EmbeddingIdx) -> &[f16] {
        let i = idx.0 as usize;
        assert!(
            i < self.num_embeddings,
            "EmbeddingIdx {i} out of bounds (num_vocab_embeddings = {})",
            self.num_embeddings,
        );
        let start = i * EMBEDDING_DIM;
        &self.data[start..start + EMBEDDING_DIM]
    }
}

// ============================================================================
// Table Storage (zero-copy mmap'd column store)
// ============================================================================

/// Typed, zero-copy view into a single column's data within a [`TableView`].
///
/// Each variant holds `&'static` slices that point directly into the
/// memory-mapped `<table_name>.bin` file. The `Arc<Mmap>` in the parent
/// [`TableView`] keeps the backing memory alive.
///
/// ## Encoding summary
///
/// | SemanticType  | Storage                            | Content                          |
/// |---------------|------------------------------------|----------------------------------|
/// | Identifier    | packed bits (non-nullable)         | true = present, false = DB NULL  |
/// | Numerical     | validity bitmap + `[f32]`          | z-scored value, null = DB NULL   |
/// | Timestamp     | validity bitmap + `[f32; 15/row]`  | cyclic + z-epoch, null = DB NULL |
/// | Boolean       | validity bitmap + packed bits      | true/false, null = DB NULL       |
/// | Categorical   | validity bitmap + `[u32]`          | EmbeddingIdx, null = DB NULL     |
/// | Text          | validity bitmap + `[u32]`          | EmbeddingIdx, null = DB NULL     |
/// | Ignored   | nothing stored                     | column is skipped entirely       |
pub enum ColumnSlice {
    /// Identifier: packed bits, non-nullable.
    /// `true` = value present in source DB, `false` = source DB NULL.
    Identifier { bits: &'static [u8] },

    /// Numerical: z-scored f32 values with validity bitmap.
    Numerical {
        validity: &'static [u8],
        values: &'static [f32],
    },

    /// Timestamp: [`TIMESTAMP_DIM`] × f32 cyclic encoding per row, with validity bitmap.
    Timestamp {
        validity: &'static [u8],
        /// Flat array of length `num_rows * TIMESTAMP_DIM`.
        /// Row `i`'s encoding is `values[i * TIMESTAMP_DIM .. (i+1) * TIMESTAMP_DIM]`.
        values: &'static [f32],
    },

    /// Boolean: packed bits with validity bitmap.
    Boolean {
        validity: &'static [u8],
        bits: &'static [u8],
    },

    /// Categorical or Text: `u32` indices into [`VocabEmbeddingTable`], with validity bitmap.
    Embedded {
        validity: &'static [u8],
        indices: &'static [u32],
    },

    /// Ignored columns store no data.
    Ignored,
}

/// A zero-copy, memory-mapped view of a single preprocessed table.
///
/// Backed by a flat binary file (`<table_name>.bin`) containing a small header
/// followed by packed column data. Multiple processes mmapping the same file
/// share the same physical pages via the OS page cache.
///
/// ## File layout
///
/// ```text
/// Header (8 bytes, 8-byte aligned):
///   num_rows  : u32
///   num_cols  : u32
///
/// Column data (packed sequentially, each section 8-byte aligned):
///   For each column (in metadata col_range order):
///     [validity bitmap — ceil(num_rows/8) bytes, padded to 8B]  (nullable types only)
///     [values — type-dependent size, padded to 8B]
///     (Ignored columns contribute 0 bytes)
/// ```
///
/// Column types are **not** stored in the file — they are read from
/// [`DatabaseMetadata::column_metadata`] at load time.
pub struct TableView {
    /// Keeps the memory map alive for the lifetime of the view.
    _mmap: Arc<Mmap>,
    /// Number of rows in this table.
    num_rows: usize,
    /// One [`ColumnSlice`] per column, in metadata col_range order.
    columns: Vec<ColumnSlice>,
}

impl TableView {
    /// Create a [`TableView`] from a memory-mapped `<table_name>.bin` file.
    ///
    /// `column_types` must list the [`SemanticType`] for every column in the
    /// table (matching the metadata's `col_range`). The function walks the
    /// file deterministically and creates zero-copy slices into the mmap.
    ///
    /// # Panics
    /// Panics if the file is too small, if the header column count doesn't
    /// match `column_types.len()`, or if the computed file size doesn't
    /// match the actual file size.
    pub fn from_mmap(mmap: Arc<Mmap>, column_types: &[SemanticType]) -> Self {
        let byte_len = mmap.len();
        let header_bytes = 2 * std::mem::size_of::<u32>();
        assert!(
            byte_len >= header_bytes,
            "table file too small for header ({byte_len} < {header_bytes} bytes)",
        );

        // SAFETY: Mmap is page-aligned (>= 4-byte aligned). Header is 2 u32s.
        let header: &[u32] = unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const u32, 2) };
        let num_rows = header[0] as usize;
        let num_cols = header[1] as usize;
        assert_eq!(
            num_cols,
            column_types.len(),
            "table header says {num_cols} columns, but metadata has {} column types",
            column_types.len(),
        );

        let bitmap_len = packed_bit_bytes(num_rows);
        let mut offset = align_up(header_bytes, TABLE_ALIGNMENT);
        let mut columns = Vec::with_capacity(num_cols);
        let base = mmap.as_ptr();

        for &stype in column_types {
            // SAFETY for all slices below:
            //   - The mmap is read-only and immutable.
            //   - The Arc keeps the backing memory alive for as long as this
            //     struct exists. We extend slice lifetimes to 'static because
            //     the Arc prevents deallocation.
            //   - Offsets are 8-byte aligned, satisfying alignment for u32/f32.
            match stype {
                SemanticType::Ignored => {
                    columns.push(ColumnSlice::Ignored);
                }
                SemanticType::Identifier => {
                    let bits = unsafe { std::slice::from_raw_parts(base.add(offset), bitmap_len) };
                    offset = align_up(offset + bitmap_len, TABLE_ALIGNMENT);
                    columns.push(ColumnSlice::Identifier { bits });
                }
                SemanticType::Numerical => {
                    let validity =
                        unsafe { std::slice::from_raw_parts(base.add(offset), bitmap_len) };
                    offset = align_up(offset + bitmap_len, TABLE_ALIGNMENT);
                    let values = unsafe {
                        std::slice::from_raw_parts(base.add(offset) as *const f32, num_rows)
                    };
                    offset = align_up(
                        offset + num_rows * std::mem::size_of::<f32>(),
                        TABLE_ALIGNMENT,
                    );
                    columns.push(ColumnSlice::Numerical { validity, values });
                }
                SemanticType::Timestamp => {
                    let validity =
                        unsafe { std::slice::from_raw_parts(base.add(offset), bitmap_len) };
                    offset = align_up(offset + bitmap_len, TABLE_ALIGNMENT);
                    let total_floats = num_rows * TIMESTAMP_DIM;
                    let values = unsafe {
                        std::slice::from_raw_parts(base.add(offset) as *const f32, total_floats)
                    };
                    offset = align_up(
                        offset + total_floats * std::mem::size_of::<f32>(),
                        TABLE_ALIGNMENT,
                    );
                    columns.push(ColumnSlice::Timestamp { validity, values });
                }
                SemanticType::Boolean => {
                    let validity =
                        unsafe { std::slice::from_raw_parts(base.add(offset), bitmap_len) };
                    offset = align_up(offset + bitmap_len, TABLE_ALIGNMENT);
                    let bits = unsafe { std::slice::from_raw_parts(base.add(offset), bitmap_len) };
                    offset = align_up(offset + bitmap_len, TABLE_ALIGNMENT);
                    columns.push(ColumnSlice::Boolean { validity, bits });
                }
                SemanticType::Categorical | SemanticType::Text => {
                    let validity =
                        unsafe { std::slice::from_raw_parts(base.add(offset), bitmap_len) };
                    offset = align_up(offset + bitmap_len, TABLE_ALIGNMENT);
                    let indices = unsafe {
                        std::slice::from_raw_parts(base.add(offset) as *const u32, num_rows)
                    };
                    offset = align_up(
                        offset + num_rows * std::mem::size_of::<u32>(),
                        TABLE_ALIGNMENT,
                    );
                    columns.push(ColumnSlice::Embedded { validity, indices });
                }
            }
        }

        assert_eq!(
            offset, byte_len,
            "table file size mismatch: column layout expects {offset} bytes, file is {byte_len}",
        );

        Self {
            _mmap: mmap,
            num_rows,
            columns,
        }
    }

    /// Number of rows in this table.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of columns in this table (including Ignored placeholders).
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get the [`ColumnSlice`] for a local column index.
    pub fn column(&self, col: usize) -> &ColumnSlice {
        &self.columns[col]
    }

    /// Returns `true` if the cell at (`col`, `row`) is null.
    ///
    /// Identifier columns are never null (they use presence/absence encoding).
    /// Ignored columns always return `true` (treated as null).
    #[inline]
    pub fn is_null(&self, col: usize, row: usize) -> bool {
        match &self.columns[col] {
            ColumnSlice::Identifier { .. } => false,
            ColumnSlice::Numerical { validity, .. }
            | ColumnSlice::Timestamp { validity, .. }
            | ColumnSlice::Boolean { validity, .. }
            | ColumnSlice::Embedded { validity, .. } => !bit_is_set(validity, row),
            ColumnSlice::Ignored => true,
        }
    }

    /// Returns `true` if the identifier at (`col`, `row`) indicates presence
    /// in the source database.
    ///
    /// # Panics
    /// Panics if column `col` is not an Identifier.
    #[inline]
    pub fn identifier_present(&self, col: usize, row: usize) -> bool {
        match &self.columns[col] {
            ColumnSlice::Identifier { bits } => bit_is_set(bits, row),
            _ => panic!("column {col} is not an Identifier"),
        }
    }

    /// Returns the z-scored `f32` value of a Numerical column at the given row.
    ///
    /// Caller should check [`is_null`](Self::is_null) first.
    ///
    /// # Panics
    /// Panics if column `col` is not Numerical.
    #[inline]
    pub fn numerical(&self, col: usize, row: usize) -> f32 {
        match &self.columns[col] {
            ColumnSlice::Numerical { values, .. } => values[row],
            _ => panic!("column {col} is not Numerical"),
        }
    }

    /// Returns the [`TIMESTAMP_DIM`]-element timestamp encoding at the given row.
    ///
    /// Caller should check [`is_null`](Self::is_null) first.
    ///
    /// # Panics
    /// Panics if column `col` is not Timestamp.
    #[inline]
    pub fn timestamp(&self, col: usize, row: usize) -> &[f32] {
        match &self.columns[col] {
            ColumnSlice::Timestamp { values, .. } => {
                let start = row * TIMESTAMP_DIM;
                &values[start..start + TIMESTAMP_DIM]
            }
            _ => panic!("column {col} is not Timestamp"),
        }
    }

    /// Returns the boolean value at (`col`, `row`).
    ///
    /// Caller should check [`is_null`](Self::is_null) first.
    ///
    /// # Panics
    /// Panics if column `col` is not Boolean.
    #[inline]
    pub fn boolean(&self, col: usize, row: usize) -> bool {
        match &self.columns[col] {
            ColumnSlice::Boolean { bits, .. } => bit_is_set(bits, row),
            _ => panic!("column {col} is not Boolean"),
        }
    }

    /// Returns the [`EmbeddingIdx`] for a Categorical or Text column at the given row.
    ///
    /// Caller should check [`is_null`](Self::is_null) first.
    ///
    /// # Panics
    /// Panics if column `col` is not Categorical/Text (Embedded).
    #[inline]
    pub fn embedding_idx(&self, col: usize, row: usize) -> EmbeddingIdx {
        match &self.columns[col] {
            ColumnSlice::Embedded { indices, .. } => EmbeddingIdx(indices[row]),
            _ => panic!("column {col} is not Categorical/Text"),
        }
    }
}

// ============================================================================
// Table Binary Writer (used by preprocessor)
// ============================================================================

/// Column data to be written by [`write_table_bin`].
///
/// Each variant carries borrowed slices of the raw buffers. Validity bitmaps
/// use little-endian bit order: bit `i` is `byte[i/8] & (1 << (i%8))`.
/// A set bit (1) means the value is **valid** (non-null).
pub enum ColumnWriter<'a> {
    /// Identifier: packed presence bits, non-nullable.
    Identifier { bits: &'a [u8] },
    /// Numerical: validity bitmap + z-scored f32 values.
    Numerical {
        validity: &'a [u8],
        values: &'a [f32],
    },
    /// Timestamp: validity bitmap + `TIMESTAMP_DIM` × f32 per row.
    Timestamp {
        validity: &'a [u8],
        values: &'a [f32],
    },
    /// Boolean: validity bitmap + packed value bits.
    Boolean { validity: &'a [u8], bits: &'a [u8] },
    /// Categorical or Text: validity bitmap + u32 indices into [`VocabEmbeddingTable`].
    Embedded {
        validity: &'a [u8],
        indices: &'a [u32],
    },
    /// Ignored: nothing written.
    Ignored,
}

/// Write a preprocessed table to a flat binary file (`<table_name>.bin`).
///
/// See [`TableView`] for the file layout. The column order must match the
/// metadata's `col_range` for the table.
///
/// # Panics (debug builds)
/// Panics if buffer lengths don't match `num_rows`.
pub fn write_table_bin(
    path: &Path,
    num_rows: u32,
    columns: &[ColumnWriter],
) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);

    // -- Header (8 bytes) ---------------------------------------------------
    let num_cols = columns.len() as u32;
    w.write_all(&num_rows.to_ne_bytes())?;
    w.write_all(&num_cols.to_ne_bytes())?;
    let mut offset = align_up(8, TABLE_ALIGNMENT); // already 8

    let bitmap_len = packed_bit_bytes(num_rows as usize);

    // Helper: write `data` bytes followed by zero-padding to TABLE_ALIGNMENT.
    fn write_padded(
        w: &mut BufWriter<File>,
        data: &[u8],
        offset: &mut usize,
    ) -> std::io::Result<()> {
        w.write_all(data)?;
        *offset += data.len();
        let aligned = align_up(*offset, TABLE_ALIGNMENT);
        let pad = aligned - *offset;
        if pad > 0 {
            w.write_all(&[0u8; 8][..pad])?;
        }
        *offset = aligned;
        Ok(())
    }

    /// Reinterpret a `&[f32]` as raw bytes.
    fn f32_as_bytes(s: &[f32]) -> &[u8] {
        // SAFETY: f32 has no padding, well-defined layout.
        unsafe {
            std::slice::from_raw_parts(
                s.as_ptr() as *const u8,
                s.len() * std::mem::size_of::<f32>(),
            )
        }
    }

    /// Reinterpret a `&[u32]` as raw bytes.
    fn u32_as_bytes(s: &[u32]) -> &[u8] {
        // SAFETY: u32 has no padding, well-defined layout.
        unsafe {
            std::slice::from_raw_parts(
                s.as_ptr() as *const u8,
                s.len() * std::mem::size_of::<u32>(),
            )
        }
    }

    for col in columns {
        match col {
            ColumnWriter::Ignored => {}
            ColumnWriter::Identifier { bits } => {
                debug_assert_eq!(bits.len(), bitmap_len);
                write_padded(&mut w, bits, &mut offset)?;
            }
            ColumnWriter::Numerical { validity, values } => {
                debug_assert_eq!(validity.len(), bitmap_len);
                debug_assert_eq!(values.len(), num_rows as usize);
                write_padded(&mut w, validity, &mut offset)?;
                write_padded(&mut w, f32_as_bytes(values), &mut offset)?;
            }
            ColumnWriter::Timestamp { validity, values } => {
                debug_assert_eq!(validity.len(), bitmap_len);
                debug_assert_eq!(values.len(), num_rows as usize * TIMESTAMP_DIM);
                write_padded(&mut w, validity, &mut offset)?;
                write_padded(&mut w, f32_as_bytes(values), &mut offset)?;
            }
            ColumnWriter::Boolean { validity, bits } => {
                debug_assert_eq!(validity.len(), bitmap_len);
                debug_assert_eq!(bits.len(), bitmap_len);
                write_padded(&mut w, validity, &mut offset)?;
                write_padded(&mut w, bits, &mut offset)?;
            }
            ColumnWriter::Embedded { validity, indices } => {
                debug_assert_eq!(validity.len(), bitmap_len);
                debug_assert_eq!(indices.len(), num_rows as usize);
                write_padded(&mut w, validity, &mut offset)?;
                write_padded(&mut w, u32_as_bytes(indices), &mut offset)?;
            }
        }
    }

    w.flush()?;
    Ok(())
}

// ============================================================================
// Task Storage (zero-copy mmap'd materialized task tables)
// ============================================================================

/// A zero-copy, memory-mapped view of a single materialized prediction task.
///
/// Each task is produced by executing a SQL query during preprocessing, yielding
/// `(anchor_row, observation_time, target_value)` tuples. The binary file
/// packs this data densely for zero-copy access at training time.
///
/// Observation times are **always** present. The preprocessor resolves the
/// observation time for every seed row at preprocessing time:
/// - If the task defines an `observation_time_column`, that value is used.
/// - Otherwise, if the anchor table has a `temporal_column`, the anchor row's
///   temporal value is used.
/// - Otherwise, `i64::MAX` is stored (no temporal filtering).
///
/// ## File layout
///
/// ```text
/// Header (4 bytes, 8-byte aligned):
///   num_seeds : u32     — number of (anchor_row, observation_time, target) triples
///   _reserved : u32     — reserved (must be 0)
///
/// Body (packed sequentially, each section 8-byte aligned):
///   anchor_rows       : [u32; num_seeds]                  — global RowIdx per seed
///   observation_times : [i64; num_seeds]                  — epoch μs (always present)
///   target validity   : packed bits [ceil(num_seeds/8)]   — 1 = valid, 0 = null
///   target values     : type-dependent:
///     Numerical   → [f32; num_seeds]                      — z-scored values
///     Categorical → [u32; num_seeds]                      — EmbeddingIdx
///     Boolean     → packed bits [ceil(num_seeds/8)]       — true/false
///     Timestamp   → [f32; num_seeds × TIMESTAMP_DIM]      — cyclic + z-epoch
/// ```
///
/// The target's semantic type is **not** stored in the file — it is read from
/// [`TaskMetadata::target_stype`] at load time.
pub struct TaskView {
    /// Keeps the memory map alive for the lifetime of the view.
    _mmap: Arc<Mmap>,
    /// Number of seed rows in this task.
    num_seeds: usize,
    /// Global RowIdx for each seed (length = num_seeds).
    anchor_rows: &'static [u32],
    /// Observation times in epoch microseconds (length = num_seeds, always present).
    /// `i64::MAX` means no temporal filtering for that seed.
    observation_times: &'static [i64],
    /// The target column, encoded identically to a [`ColumnSlice`].
    target: ColumnSlice,
}

impl TaskView {
    /// Create a [`TaskView`] from a memory-mapped `<task_name>.bin` file.
    ///
    /// `target_stype` must be the task's target semantic type from
    /// [`TaskMetadata`]. Only `Numerical`, `Categorical`, `Boolean`, and
    /// `Timestamp` are valid target types.
    ///
    /// # Panics
    /// Panics if the file is too small, if `target_stype` is not a valid
    /// target type, or if the computed file size doesn't match the actual size.
    pub fn from_mmap(mmap: Arc<Mmap>, target_stype: SemanticType) -> Self {
        let byte_len = mmap.len();
        let header_bytes = 2 * std::mem::size_of::<u32>();
        assert!(
            byte_len >= header_bytes,
            "task file too small for header ({byte_len} < {header_bytes} bytes)",
        );

        // SAFETY: Mmap is page-aligned (>= 4-byte aligned). Header is 2 u32s.
        let header: &[u32] = unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const u32, 2) };
        let num_seeds = header[0] as usize;
        let _reserved = header[1];

        let bitmap_len = packed_bit_bytes(num_seeds);
        let mut offset = align_up(header_bytes, TABLE_ALIGNMENT);
        let base = mmap.as_ptr();

        // SAFETY for all slices below:
        //   - The mmap is read-only and immutable.
        //   - The Arc keeps the backing memory alive for as long as this struct
        //     exists. We extend slice lifetimes to 'static because the Arc
        //     prevents deallocation.
        //   - Offsets are 8-byte aligned, satisfying alignment for u32/f32/i64.

        // -- anchor_rows: [u32; num_seeds] --
        let anchor_rows: &'static [u32] =
            unsafe { std::slice::from_raw_parts(base.add(offset) as *const u32, num_seeds) };
        offset = align_up(
            offset + num_seeds * std::mem::size_of::<u32>(),
            TABLE_ALIGNMENT,
        );

        // -- observation_times: [i64; num_seeds] (always present) --
        let observation_times: &'static [i64] =
            unsafe { std::slice::from_raw_parts(base.add(offset) as *const i64, num_seeds) };
        offset = align_up(
            offset + num_seeds * std::mem::size_of::<i64>(),
            TABLE_ALIGNMENT,
        );

        // -- target column: validity bitmap + typed values --
        let target = match target_stype {
            SemanticType::Numerical => {
                let validity = unsafe { std::slice::from_raw_parts(base.add(offset), bitmap_len) };
                offset = align_up(offset + bitmap_len, TABLE_ALIGNMENT);
                let values = unsafe {
                    std::slice::from_raw_parts(base.add(offset) as *const f32, num_seeds)
                };
                offset = align_up(
                    offset + num_seeds * std::mem::size_of::<f32>(),
                    TABLE_ALIGNMENT,
                );
                ColumnSlice::Numerical { validity, values }
            }
            SemanticType::Timestamp => {
                let validity = unsafe { std::slice::from_raw_parts(base.add(offset), bitmap_len) };
                offset = align_up(offset + bitmap_len, TABLE_ALIGNMENT);
                let total_floats = num_seeds * TIMESTAMP_DIM;
                let values = unsafe {
                    std::slice::from_raw_parts(base.add(offset) as *const f32, total_floats)
                };
                offset = align_up(
                    offset + total_floats * std::mem::size_of::<f32>(),
                    TABLE_ALIGNMENT,
                );
                ColumnSlice::Timestamp { validity, values }
            }
            SemanticType::Boolean => {
                let validity = unsafe { std::slice::from_raw_parts(base.add(offset), bitmap_len) };
                offset = align_up(offset + bitmap_len, TABLE_ALIGNMENT);
                let bits = unsafe { std::slice::from_raw_parts(base.add(offset), bitmap_len) };
                offset = align_up(offset + bitmap_len, TABLE_ALIGNMENT);
                ColumnSlice::Boolean { validity, bits }
            }
            SemanticType::Categorical => {
                let validity = unsafe { std::slice::from_raw_parts(base.add(offset), bitmap_len) };
                offset = align_up(offset + bitmap_len, TABLE_ALIGNMENT);
                let indices = unsafe {
                    std::slice::from_raw_parts(base.add(offset) as *const u32, num_seeds)
                };
                offset = align_up(
                    offset + num_seeds * std::mem::size_of::<u32>(),
                    TABLE_ALIGNMENT,
                );
                ColumnSlice::Embedded { validity, indices }
            }
            _ => panic!(
                "invalid target_stype {:?} for task (must be Numerical, Categorical, Boolean, or Timestamp)",
                target_stype,
            ),
        };

        assert_eq!(
            offset, byte_len,
            "task file size mismatch: layout expects {offset} bytes, file is {byte_len}",
        );

        Self {
            _mmap: mmap,
            num_seeds,
            anchor_rows,
            observation_times,
            target,
        }
    }

    /// Number of seed rows in this task.
    pub fn num_seeds(&self) -> usize {
        self.num_seeds
    }

    /// Get the global [`RowIdx`] for a seed.
    ///
    /// # Panics
    /// Panics if `seed` is out of bounds.
    #[inline]
    pub fn anchor_row(&self, seed: usize) -> RowIdx {
        RowIdx(self.anchor_rows[seed])
    }

    /// Get the observation time (epoch microseconds) for a seed.
    ///
    /// A value of `i64::MAX` means no temporal filtering applies for this seed.
    ///
    /// # Panics
    /// Panics if `seed` is out of bounds.
    #[inline]
    pub fn observation_time(&self, seed: usize) -> i64 {
        self.observation_times[seed]
    }

    /// Get the target column as a [`ColumnSlice`].
    ///
    /// The variant matches the task's `target_stype`:
    /// - `Numerical` → [`ColumnSlice::Numerical`]
    /// - `Categorical` → [`ColumnSlice::Embedded`]
    /// - `Boolean` → [`ColumnSlice::Boolean`]
    /// - `Timestamp` → [`ColumnSlice::Timestamp`]
    pub fn target(&self) -> &ColumnSlice {
        &self.target
    }

    /// Returns `true` if the target value at `seed` is null.
    #[inline]
    pub fn target_is_null(&self, seed: usize) -> bool {
        match &self.target {
            ColumnSlice::Numerical { validity, .. }
            | ColumnSlice::Timestamp { validity, .. }
            | ColumnSlice::Boolean { validity, .. }
            | ColumnSlice::Embedded { validity, .. } => !bit_is_set(validity, seed),
            _ => unreachable!("task targets are always nullable typed columns"),
        }
    }
}

// ============================================================================
// Task Binary Writer (used by preprocessor)
// ============================================================================

/// Write a materialized task to a flat binary file (`<task_name>.bin`).
///
/// See [`TaskView`] for the file layout.
///
/// `observation_times` must always be provided (one `i64` per seed). Use
/// `i64::MAX` for seeds with no temporal constraint.
///
/// `target` must be a [`ColumnWriter`] variant matching the task's `target_stype`:
/// `Numerical`, `Timestamp`, `Boolean`, or `Embedded` (for categoricals).
///
/// # Panics (debug builds)
/// Panics if buffer lengths don't match `num_seeds`.
pub fn write_task_bin(
    path: &Path,
    num_seeds: u32,
    anchor_rows: &[u32],
    observation_times: &[i64],
    target: &ColumnWriter,
) -> std::io::Result<()> {
    debug_assert_eq!(anchor_rows.len(), num_seeds as usize);
    debug_assert_eq!(observation_times.len(), num_seeds as usize);

    let mut w = BufWriter::new(File::create(path)?);

    // -- Header (8 bytes) ---------------------------------------------------
    let reserved: u32 = 0;
    w.write_all(&num_seeds.to_ne_bytes())?;
    w.write_all(&reserved.to_ne_bytes())?;
    let mut offset = align_up(8, TABLE_ALIGNMENT); // already 8

    let bitmap_len = packed_bit_bytes(num_seeds as usize);

    fn write_padded(
        w: &mut BufWriter<File>,
        data: &[u8],
        offset: &mut usize,
    ) -> std::io::Result<()> {
        w.write_all(data)?;
        *offset += data.len();
        let aligned = align_up(*offset, TABLE_ALIGNMENT);
        let pad = aligned - *offset;
        if pad > 0 {
            w.write_all(&[0u8; 8][..pad])?;
        }
        *offset = aligned;
        Ok(())
    }

    fn u32_as_bytes(s: &[u32]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                s.as_ptr() as *const u8,
                s.len() * std::mem::size_of::<u32>(),
            )
        }
    }

    fn i64_as_bytes(s: &[i64]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                s.as_ptr() as *const u8,
                s.len() * std::mem::size_of::<i64>(),
            )
        }
    }

    fn f32_as_bytes(s: &[f32]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                s.as_ptr() as *const u8,
                s.len() * std::mem::size_of::<f32>(),
            )
        }
    }

    // -- anchor_rows --
    write_padded(&mut w, u32_as_bytes(anchor_rows), &mut offset)?;

    // -- observation_times (always present) --
    write_padded(&mut w, i64_as_bytes(observation_times), &mut offset)?;

    // -- target column --
    match target {
        ColumnWriter::Numerical { validity, values } => {
            debug_assert_eq!(validity.len(), bitmap_len);
            debug_assert_eq!(values.len(), num_seeds as usize);
            write_padded(&mut w, validity, &mut offset)?;
            write_padded(&mut w, f32_as_bytes(values), &mut offset)?;
        }
        ColumnWriter::Timestamp { validity, values } => {
            debug_assert_eq!(validity.len(), bitmap_len);
            debug_assert_eq!(values.len(), num_seeds as usize * TIMESTAMP_DIM);
            write_padded(&mut w, validity, &mut offset)?;
            write_padded(&mut w, f32_as_bytes(values), &mut offset)?;
        }
        ColumnWriter::Boolean { validity, bits } => {
            debug_assert_eq!(validity.len(), bitmap_len);
            debug_assert_eq!(bits.len(), bitmap_len);
            write_padded(&mut w, validity, &mut offset)?;
            write_padded(&mut w, bits, &mut offset)?;
        }
        ColumnWriter::Embedded { validity, indices } => {
            debug_assert_eq!(validity.len(), bitmap_len);
            debug_assert_eq!(indices.len(), num_seeds as usize);
            write_padded(&mut w, validity, &mut offset)?;
            write_padded(&mut w, u32_as_bytes(indices), &mut offset)?;
        }
        _ => panic!(
            "invalid ColumnWriter variant for task target (must be Numerical, Timestamp, Boolean, or Embedded)"
        ),
    }

    w.flush()?;
    Ok(())
}

// ============================================================================
// Database Wrapper
// ============================================================================

/// A preprocessed relational database, ready for training.
///
/// Assembles five independently-stored components, **all** backed by
/// memory-mapped files for zero-copy sharing across DDP processes:
///
/// - **metadata** — schema, column stats, task definitions (JSON, deserialized into owned types; small KB)
/// - **column_embeddings** — pre-computed column-name embeddings (flat binary, read at load into [`ColumnMetadata::embedding`])
/// - **graph** — bidirectional CSR topology (flat binary, mmap'd zero-copy via [`GraphView`])
/// - **tables** — preprocessed column data (flat binary, mmap'd zero-copy via [`TableView`])
/// - **tasks** — materialized prediction tasks (flat binary, mmap'd zero-copy via [`TaskView`])
/// - **vocab_embeddings** — interned text embeddings for categorical/text values (flat binary, mmap'd via [`VocabEmbeddingTable`])
///
/// Column-name embeddings are loaded from `column_embeddings.bin` and populated
/// into [`ColumnMetadata::embedding`] at load time, so they can be uploaded to
/// GPU shared memory once at training start.
///
/// The preprocessed database directory layout:
///
/// ```text
/// db_out/
///   metadata.json              — DatabaseMetadata (JSON, schema + stats + tasks)
///   column_embeddings.bin      — flat [C, EMBEDDING_DIM] f16 array (one per column)
///   vocab_embeddings.bin       — flat [V, EMBEDDING_DIM] f16 array (categorical/text value embeddings)
///   graph.bin                  — bidirectional CSR (flat binary, see GraphView)
///   tables/
///     badges.bin               — flat binary column store (see TableView)
///     comments.bin
///     ...
///   tasks/
///     predict_vote_type.bin    — materialized task (see TaskView)
///     predict_post_type.bin
///     ...
/// ```
pub struct Database {
    /// Schema, column stats, column-name embeddings, task definitions.
    /// Deserialized from `metadata.json`; column embeddings populated from `column_embeddings.bin`.
    pub metadata: DatabaseMetadata,

    /// Bidirectional CSR graph topology, zero-copy from mmap'd `graph.bin`.
    pub graph: GraphView,

    /// Preprocessed column data: one [`TableView`] per table.
    /// Each view is backed by a mmap'd `<table_name>.bin` file.
    pub tables: Vec<TableView>,

    /// Materialized prediction tasks: one [`TaskView`] per task.
    /// Each view is backed by a mmap'd `<task_name>.bin` file.
    pub tasks: Vec<TaskView>,

    /// Interned vocabulary embeddings for categorical and text cell values.
    /// Indexed by [`EmbeddingIdx`] stored in Embedded columns of `tables`.
    /// Column-name embeddings are **not** in this table — they live in
    /// [`ColumnMetadata::embedding`].
    pub vocab_embeddings: VocabEmbeddingTable,

    /// Sorted table row-range starts for O(log T) row-to-table lookup.
    /// One entry per table: `row_starts[i] = table_metadata[i].row_range.0`.
    row_starts: Vec<u32>,

    /// Sorted table col-range starts for O(log T) col-to-table lookup.
    /// One entry per table: `col_starts[i] = table_metadata[i].col_range.0`.
    col_starts: Vec<u32>,
}

impl Database {
    /// Load a preprocessed database from the given directory.
    ///
    /// Expects the directory layout described in the [`Database`] docs.
    pub fn load(dir: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // -- Metadata (JSON, deserialized into owned types) ------------------
        let metadata_path = dir.join("metadata.json");
        let metadata_bytes = std::fs::read(&metadata_path)?;
        let mut metadata: DatabaseMetadata = serde_json::from_slice(&metadata_bytes)
            .map_err(|e| format!("failed to deserialize metadata: {e}"))?;

        // -- Column embeddings (flat binary) --------------------------------
        // Populate ColumnMetadata.embedding from the separate binary file.
        let col_emb_path = dir.join("column_embeddings.bin");
        let col_emb_bytes = std::fs::read(&col_emb_path)?;
        let stride = EMBEDDING_DIM * std::mem::size_of::<f16>();
        let num_col_embeddings = col_emb_bytes.len() / stride;
        assert_eq!(
            num_col_embeddings,
            metadata.column_metadata.len(),
            "column_embeddings.bin has {} embeddings but metadata has {} columns",
            num_col_embeddings,
            metadata.column_metadata.len(),
        );
        for (i, cmeta) in metadata.column_metadata.iter_mut().enumerate() {
            let start = i * stride;
            let chunk = &col_emb_bytes[start..start + stride];
            // SAFETY: f16 is repr(transparent) over u16 (2 bytes), and the
            // slice is properly aligned because stride is a multiple of 2.
            let f16_slice: &[f16] =
                unsafe { std::slice::from_raw_parts(chunk.as_ptr() as *const f16, EMBEDDING_DIM) };
            cmeta.embedding = f16_slice.to_vec();
        }

        // -- Graph (flat binary, mmap'd zero-copy) --------------------------
        let graph_path = dir.join("graph.bin");
        let graph_file = File::open(&graph_path)?;
        let graph_mmap = Arc::new(unsafe { Mmap::map(&graph_file)? });
        let graph = GraphView::from_mmap(graph_mmap);

        // -- Precompute row_starts / col_starts for O(log T) lookups ---------
        let row_starts: Vec<u32> = metadata
            .table_metadata
            .iter()
            .map(|t| t.row_range.0.0)
            .collect();
        let col_starts: Vec<u32> = metadata
            .table_metadata
            .iter()
            .map(|t| t.col_range.0.0)
            .collect();

        // -- Tables (flat binary, mmap'd zero-copy) ---------------------------
        let tables_dir = dir.join("tables");
        let num_tables = metadata.table_metadata.len();
        let mut tables = Vec::with_capacity(num_tables);
        for i in 0..num_tables {
            let table_name = &metadata.table_metadata[i].name;
            let bin_path = tables_dir.join(format!("{table_name}.bin"));
            let file = File::open(&bin_path)?;
            let mmap = Arc::new(unsafe { Mmap::map(&file)? });

            // Derive column types from metadata for this table.
            let tmeta = &metadata.table_metadata[i];
            let col_start = tmeta.col_range.0.0 as usize;
            let col_end = tmeta.col_range.1.0 as usize;
            let column_types: Vec<SemanticType> = (col_start..col_end)
                .map(|ci| metadata.column_metadata[ci].stype)
                .collect();

            tables.push(TableView::from_mmap(mmap, &column_types));
        }

        // -- Tasks (flat binary, mmap'd zero-copy) ----------------------------
        let tasks_dir = dir.join("tasks");
        let num_tasks = metadata.task_metadata.len();
        let mut tasks = Vec::with_capacity(num_tasks);
        for i in 0..num_tasks {
            let task_name = &metadata.task_metadata[i].name;
            let bin_path = tasks_dir.join(format!("{task_name}.bin"));
            let file = File::open(&bin_path)?;
            let mmap = Arc::new(unsafe { Mmap::map(&file)? });
            let target_stype = metadata.task_metadata[i].target_stype;
            tasks.push(TaskView::from_mmap(mmap, target_stype));
        }

        // -- Vocab embeddings (flat binary, mmap'd) ---------------------------
        let vocab_path = dir.join("vocab_embeddings.bin");
        let vocab_file = File::open(&vocab_path)?;
        let vocab_mmap = Arc::new(unsafe { Mmap::map(&vocab_file)? });
        let vocab_embeddings = VocabEmbeddingTable::from_mmap(vocab_mmap);

        Ok(Self {
            metadata,
            graph,
            tables,
            tasks,
            vocab_embeddings,
            row_starts,
            col_starts,
        })
    }

    /// Number of tables in the database.
    pub fn num_tables(&self) -> usize {
        self.metadata.table_metadata.len()
    }

    /// Total number of rows across all tables.
    pub fn num_rows(&self) -> usize {
        self.graph.num_nodes()
    }

    /// Resolve a [`RowIdx`] to the [`TableIdx`] of the table that contains it.
    ///
    /// Uses binary search over precomputed table row-range boundaries.
    /// O(log T) where T is the number of tables.
    ///
    /// # Panics
    /// Panics if `row` does not belong to any table.
    pub fn row_table(&self, row: RowIdx) -> TableIdx {
        // row_starts is sorted (tables have contiguous, non-overlapping row
        // ranges). partition_point finds the first index where start > row,
        // so (index - 1) is our table.
        let pos = self.row_starts.partition_point(|&start| start <= row.0);
        assert!(pos > 0, "RowIdx {} is before the first table", row.0);
        TableIdx((pos - 1) as u32)
    }

    /// Get the [`TableView`] for a specific table.
    pub fn table(&self, idx: TableIdx) -> &TableView {
        &self.tables[idx.0 as usize]
    }

    /// Look up a vocabulary embedding (categorical / text value) by index.
    ///
    /// Returns a slice of [`EMBEDDING_DIM`] `f16` values.
    pub fn vocab_embedding(&self, idx: EmbeddingIdx) -> &[f16] {
        self.vocab_embeddings.get(idx)
    }

    /// Look up the column-name embedding for a given column.
    ///
    /// Column embeddings are stored directly in [`ColumnMetadata`].
    pub fn column_embedding(&self, col: ColumnIdx) -> &[f16] {
        &self.metadata.column_metadata[col.0 as usize].embedding
    }

    /// Convert a global [`RowIdx`] to the local row offset within its table.
    ///
    /// This is the row index you pass to [`TableView`] accessor methods.
    pub fn local_row(&self, row: RowIdx) -> usize {
        let ti = self.row_table(row);
        let row_start = self.metadata.table_metadata[ti.0 as usize].row_range.0.0;
        (row.0 - row_start) as usize
    }

    /// Resolve a global [`ColumnIdx`] to `(table_index, local_column_offset)`.
    ///
    /// Uses binary search over precomputed table col-range boundaries.
    /// O(log T) where T is the number of tables.
    ///
    /// # Panics
    /// Panics if `col` does not belong to any table.
    pub fn resolve_column(&self, col: ColumnIdx) -> (usize, usize) {
        let pos = self.col_starts.partition_point(|&start| start <= col.0);
        assert!(pos > 0, "ColumnIdx {} is before the first table", col.0);
        let ti = pos - 1;
        let local = (col.0 - self.col_starts[ti]) as usize;
        (ti, local)
    }

    /// Number of prediction tasks.
    pub fn num_tasks(&self) -> usize {
        self.metadata.task_metadata.len()
    }

    /// Get the [`TaskView`] for a specific task.
    pub fn task(&self, idx: TaskIdx) -> &TaskView {
        &self.tasks[idx.0 as usize]
    }

    /// Get the [`TaskMetadata`] for a specific task.
    pub fn task_metadata(&self, idx: TaskIdx) -> &TaskMetadata {
        &self.metadata.task_metadata[idx.0 as usize]
    }
}
