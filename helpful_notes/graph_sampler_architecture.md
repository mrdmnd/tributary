# Graph Subsampler Architecture for Tributary

> Design document for a high-performance graph subsampler that produces batches directly
> from BFS over relational-database-derived graphs, designed to integrate with Burn's DDP
> training framework.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Why Produce Batches Directly](#why-produce-batches-directly)
- [Architecture Overview](#architecture-overview)
- [Component Design](#component-design)
  - [SeedPool — Seed Node Management](#seedpool--seed-node-management)
  - [SubgraphSampler — BFS Extraction](#subgraphsampler--bfs-extraction)
  - [BatchAssembler — Tensor Construction](#batchassembler--tensor-construction)
  - [GraphBatchLoader — Prefetching Pipeline](#graphbatchloader--prefetching-pipeline)
- [What a Batch Looks Like](#what-a-batch-looks-like)
- [DDP Integration](#ddp-integration)
- [Performance Considerations](#performance-considerations)
- [Appendix: Burn DataLoader Trait Compatibility](#appendix-burn-dataloader-trait-compatibility)

---

## Problem Statement

We need to train a relational foundation model where:

- The "dataset" is a relational database represented as a bidirectional CSR graph
  (nodes = rows, edges = FK relationships)
- Each training sample is a **subgraph** extracted via BFS from a seed node
- The subgraph is flattened into a sequence of cells (tokens) of length `seq_len`
- The model requires three attention masks per sequence: **column**, **feature** (same-row),
  and **neighbor** (FK-connected rows)
- Samples are generated **dynamically** — the BFS walk, neighbor sampling, and cell ordering
  happen at sample time, not from a pre-materialized dataset

The sampler must:
1. Produce batches at GPU-saturating throughput
2. Integrate with Burn's DDP for multi-device gradient synchronization
3. Share the read-only `Database` (mmap'd) across all worker threads without copying

---

## Why Produce Batches Directly

In standard ML data loading (images, text), each item is independent and cheap to produce.
The pattern is: produce N items independently → collate into a batch. Burn's `DataLoader`
follows this: `Dataset::get(index) -> Item`, then `Batcher::batch(Vec<Item>) -> Batch`.

Graph subsampling breaks this pattern for several reasons:

1. **BFS is not a pure function of a single index.** The subgraph shape depends on the
   graph topology, and two seeds may produce overlapping or wildly different subgraphs.
   There's no meaningful "item index" beyond the seed node.

2. **Collation is expensive and tightly coupled to sampling.** Building the three attention
   masks requires knowing which cells belong to which rows and columns — information that's
   produced during BFS, not after. Separating sampling from collation means either:
   - Materializing intermediate subgraph representations (wasteful allocations), or
   - Re-deriving structure during collation (redundant work)

3. **Batch-level parallelism is natural.** Each seed's BFS can run on a separate thread,
   and the batch can be assembled once all BFS walks complete. This maps cleanly to a
   thread-pool model.

4. **Variable-size subgraphs need batch-level padding.** The number of cells per subgraph
   varies. Padding/truncation to `seq_len` is a batch-level decision (you could pack
   multiple small subgraphs into one sequence, or truncate large ones).

**Bottom line**: the sampler should own the full pipeline from seed selection through tensor
construction, producing ready-to-train `GraphBatch` values directly.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GraphBatchLoader                        │
│                                                             │
│  ┌──────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │ SeedPool │───►│ SubgraphSampler  │───►│BatchAssembler│  │
│  │          │    │ (thread pool)    │    │              │  │
│  │ shuffled │    │                  │    │ builds masks │  │
│  │ seed     │    │ BFS per seed,    │    │ encodes cells│  │
│  │ nodes    │    │ neighbor sample, │    │ pads/packs   │  │
│  │          │    │ cell extraction  │    │              │  │
│  └──────────┘    └──────────────────┘    └──────────────┘  │
│                                                │            │
│                          ┌─────────────────────┘            │
│                          ▼                                  │
│                   ┌─────────────┐                           │
│                   │ Prefetch    │  1–2 batches buffered     │
│                   │ Channel     │  ahead of training loop   │
│                   └──────┬──────┘                           │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           ▼
                    Training Loop
                    (forward, backward, all-reduce, optimizer step)
```

### Shared, Read-Only State

```rust
/// Everything the sampler needs, immutable and shared across all threads.
/// All backed by mmap — the OS page cache handles sharing automatically.
struct SamplerState {
    db: Arc<Database>,       // graph, tables, embeddings, metadata
    config: SamplerConfig,   // seq_len, max_hops, neighbor_budget, etc.
}
```

Since `Database` is entirely mmap-backed with `&'static` slices held alive by `Arc<Mmap>`,
it is `Send + Sync` and can be shared across threads at zero cost. Multiple DDP workers
(threads) on the same node share the same physical pages via the OS page cache.

---

## Component Design

### SeedPool — Seed Node Management

The SeedPool manages which nodes are valid BFS roots and handles shuffling / epoch cycling.

```rust
/// Configuration for seed node selection.
struct SeedPoolConfig {
    /// Which tables contain valid seed nodes.
    /// Typically the "target" table(s) for prediction.
    seed_tables: Vec<TableIdx>,

    /// Optional: filter to rows where the prediction target is non-null.
    /// (Only rows with a valid label should be seeds during training.)
    require_label: bool,

    /// RNG seed for reproducibility.
    rng_seed: u64,
}

/// Manages the pool of seed nodes for one rank.
struct SeedPool {
    /// All valid seed node indices (global RowIdx values).
    seeds: Vec<RowIdx>,

    /// Current position in the shuffled order.
    cursor: usize,

    /// RNG for shuffling.
    rng: StdRng,
}

impl SeedPool {
    /// Create a new pool from the database, filtering to valid seeds.
    fn new(db: &Database, config: &SeedPoolConfig) -> Self { ... }

    /// Partition this pool for DDP: keep only seeds for rank `rank` of `world_size`.
    /// Uses interleaved assignment (like PyTorch DistributedSampler) for better
    /// statistical diversity than contiguous slicing.
    fn partition_for_rank(&mut self, rank: usize, world_size: usize) { ... }

    /// Draw the next `batch_size` seeds. Reshuffles when exhausted (new epoch).
    fn next_batch(&mut self, batch_size: usize) -> Vec<RowIdx> { ... }

    /// Number of seeds remaining before reshuffle.
    fn remaining(&self) -> usize { ... }

    /// Total seeds in this rank's partition.
    fn len(&self) -> usize { ... }
}
```

**Key design choice: interleaved partitioning for DDP.**

Unlike Burn's default `split_dataloader` which does contiguous slicing
(`rank 0 gets [0..N/k), rank 1 gets [N/k..2N/k), ...`), we use interleaved assignment:

```
rank 0: seeds[0], seeds[2], seeds[4], ...
rank 1: seeds[1], seeds[3], seeds[5], ...
```

This ensures each rank sees a statistically representative sample of the database,
which matters for relational databases where table structure can create spatial clustering
in contiguous RowIdx ranges. All rows in `table_0000` are contiguous, then all rows in
`table_0001`, etc. — contiguous slicing would give some ranks all seeds from one table.

### SubgraphSampler — BFS Extraction

The core sampling logic: BFS from a seed node, collecting a subgraph of rows.

```rust
struct SamplerConfig {
    /// Maximum BFS hops from the seed node.
    max_hops: usize,

    /// Maximum number of neighbors to sample per node per direction (fan-out).
    /// Controls subgraph size. If a node has more neighbors, we sample uniformly.
    neighbor_budget: usize,

    /// Target number of cells (tokens) per sequence.
    seq_len: usize,

    /// Maximum number of rows per subgraph (hard cap).
    max_rows: usize,
}

/// A raw subgraph extracted by BFS. Not yet tensorized.
struct RawSubgraph {
    /// The seed node (always the first element).
    seed: RowIdx,

    /// Ordered list of rows in the subgraph (BFS order).
    /// Includes the seed as the first element.
    rows: Vec<RowIdx>,

    /// For each row in `rows`, its table index.
    row_tables: Vec<TableIdx>,

    /// For each row in `rows`, its BFS depth from seed.
    row_depths: Vec<u8>,

    /// Adjacency within the subgraph: (local_src, local_dst) pairs.
    /// Indices are into `rows`, not global RowIdx.
    /// Includes both outgoing and incoming edges.
    local_edges: Vec<(u16, u16)>,
}

impl SubgraphSampler {
    /// Extract a subgraph via BFS from `seed`.
    ///
    /// The BFS explores both outgoing (forward FK) and incoming (reverse FK)
    /// neighbors, sampling up to `neighbor_budget` per direction per node.
    ///
    /// Returns a RawSubgraph in BFS order.
    fn sample(
        &self,
        seed: RowIdx,
        db: &Database,
        config: &SamplerConfig,
        rng: &mut impl Rng,
    ) -> RawSubgraph {
        // 1. Initialize BFS queue with seed
        // 2. For each node in queue (up to max_hops):
        //    a. Get outgoing_neighbors and incoming_neighbors from db.graph
        //    b. If degree > neighbor_budget, sample uniformly without replacement
        //    c. Add unvisited neighbors to queue
        // 3. Stop when queue is empty or max_rows reached
        // 4. Record local edge list for adjacency within subgraph
        ...
    }
}
```

**BFS details:**

```
Seed (table: orders, row: 42)
  ├─ hop 1 (outgoing FK): customer_37, product_102, store_5
  ├─ hop 1 (incoming FK): return_88, review_201
  ├─ hop 2 (outgoing FK from customer_37): address_15, segment_3
  ├─ hop 2 (incoming FK to customer_37): order_99, order_157
  └─ ...
```

The BFS traverses both FK directions because:
- **Outgoing** (forward FK): "this order references customer_37" → learn about the customer
- **Incoming** (reverse FK): "customer_37 has these other orders" → learn aggregate patterns

**Neighbor sampling** is critical for controlling subgraph size. High-degree nodes (e.g., a
popular product referenced by 10,000 orders) would explode the subgraph without budgeting.

### BatchAssembler — Tensor Construction

Converts a batch of `RawSubgraph`s into the tensors the model expects.

```rust
/// A fully tensorized batch, ready for the model's forward pass.
/// All tensors are on CPU — the training loop moves them to device.
struct GraphBatch<B: Backend> {
    // ====== Input Tensors ======

    /// Cell feature vectors: [batch_size, seq_len, dim_model]
    /// Each cell is one (row, column) pair from the subgraph.
    /// Encoded from raw column data using the appropriate encoder.
    features: Tensor<B, 3>,

    /// Column embedding input: [batch_size, seq_len, dim_text]
    /// The frozen column-name embedding for each cell's column.
    column_embeddings: Tensor<B, 3>,

    /// Semantic type of each cell: [batch_size, seq_len] (u8 values)
    /// Used to select the correct mask embedding for masked cells.
    cell_types: Tensor<B, 2, Int>,

    // ====== Attention Masks ======
    // All masks: [batch_size, seq_len, seq_len], true = MASKED (ignored)

    /// Column mask: cells can attend to other cells in the SAME COLUMN
    /// (same ColumnIdx across different rows in the subgraph).
    column_mask: Tensor<B, 3, Bool>,

    /// Feature mask: cells can attend to other cells in the SAME ROW.
    feature_mask: Tensor<B, 3, Bool>,

    /// Neighbor mask: cells can attend to cells in FK-CONNECTED ROWS
    /// (rows connected by an edge in the subgraph's local_edges).
    neighbor_mask: Tensor<B, 3, Bool>,

    /// Padding mask: [batch_size, seq_len], true = padding position.
    padding_mask: Tensor<B, 2, Bool>,

    // ====== MLM Target Tensors ======

    /// Which cells are masked for prediction: [batch_size, seq_len], true = masked.
    masked_positions: Tensor<B, 2, Bool>,

    /// Ground truth values for masked cells (for loss computation).
    /// Structure TBD — depends on per-type loss heads.
    targets: MaskedTargets,
}

/// Targets for each semantic type's decoder head.
/// Only cells that are both (a) masked and (b) non-null have targets.
struct MaskedTargets {
    /// Indices into the flat seq_len dimension where each type's targets live.
    numerical_indices: Tensor<B, 2, Int>,   // [batch_size, max_num_targets]
    numerical_values: Tensor<B, 2>,          // [batch_size, max_num_targets]

    timestamp_indices: Tensor<B, 2, Int>,
    timestamp_values: Tensor<B, 3>,          // [batch_size, max_ts_targets, TIMESTAMP_DIM]

    boolean_indices: Tensor<B, 2, Int>,
    boolean_values: Tensor<B, 2, Bool>,

    categorical_indices: Tensor<B, 2, Int>,
    categorical_embeddings: Tensor<B, 3>,    // [batch_size, max_cat_targets, dim_text]

    text_indices: Tensor<B, 2, Int>,
    text_embeddings: Tensor<B, 3>,           // [batch_size, max_text_targets, dim_text]
}
```

**Cell ordering within a sequence:**

The subgraph's rows need to be flattened into a linear sequence of cells. The ordering
matters for attention pattern locality. Proposed ordering:

```
For each row r in BFS order:
    For each column c in table schema order (skipping Unsupported):
        emit cell (r, c)
```

This means:
- **Feature attention** (same-row cells) always attends to a contiguous block
- **Column attention** (same-column cells) attends to cells at regular stride intervals
  within the same table, and scattered positions across different tables
- **Neighbor attention** is defined by the subgraph's local_edges

**Padding and truncation:**

- If the flattened cells exceed `seq_len`, truncate (drop the deepest BFS rows first —
  they're least relevant to the seed)
- If fewer than `seq_len`, pad with zeros and set `padding_mask = true`

**Mask construction:**

For a batch of B sequences each of length S:

```python
# Feature mask: cells i, j can attend iff they're in the same row
feature_mask[b, i, j] = (cell_row[b, i] != cell_row[b, j])

# Column mask: cells i, j can attend iff they're in the same column
column_mask[b, i, j] = (cell_col[b, i] != cell_col[b, j])

# Neighbor mask: cells i, j can attend iff their rows are FK-connected
#   Build from local_edges: row i's row and row j's row share an edge
neighbor_mask[b, i, j] = !edge_exists(cell_row[b, i], cell_row[b, j])

# All masks must also mask padding positions
feature_mask[b, i, j]  |= padding[b, i] | padding[b, j]
column_mask[b, i, j]   |= padding[b, i] | padding[b, j]
neighbor_mask[b, i, j]  |= padding[b, i] | padding[b, j]
```

**MLM masking:**

Like BERT, randomly mask ~15% of non-padding cells. For each masked cell:
- Replace its feature vector with the learned `mask_embedding[semantic_type]`
- Record its ground truth value in `MaskedTargets` for loss computation

### GraphBatchLoader — Prefetching Pipeline

Wraps everything into a prefetching iterator that stays ahead of the training loop.

```rust
struct GraphBatchLoader<B: Backend> {
    /// Shared, immutable database reference.
    db: Arc<Database>,

    /// Seed pool for this rank.
    seed_pool: SeedPool,

    /// Sampler + assembler configuration.
    config: SamplerConfig,

    /// Prefetch channel: background threads produce batches,
    /// training loop consumes them.
    rx: Receiver<GraphBatch<B>>,

    /// Handle to the background prefetch thread(s).
    _workers: Vec<JoinHandle<()>>,
}

impl<B: Backend> GraphBatchLoader<B> {
    fn new(
        db: Arc<Database>,
        seed_pool: SeedPool,
        config: SamplerConfig,
        prefetch_depth: usize,  // how many batches to buffer (1-3)
        num_workers: usize,     // parallel BFS threads per batch
    ) -> Self {
        // Spawn a prefetch thread that:
        // 1. Draws batch_size seeds from the pool
        // 2. Fans out BFS to a rayon thread pool (num_workers threads)
        // 3. Assembles the batch
        // 4. Sends it through the channel
        //
        // The channel has capacity `prefetch_depth`, so the producer
        // blocks when the consumer (training loop) is behind.
        ...
    }

    /// Get the next batch. Blocks until available.
    fn next_batch(&self) -> Option<GraphBatch<B>> {
        self.rx.recv().ok()
    }

    /// Number of batches per epoch (seeds / batch_size, rounded up).
    fn batches_per_epoch(&self) -> usize {
        (self.seed_pool.len() + self.config.batch_size - 1) / self.config.batch_size
    }
}
```

**Prefetch pipeline timing:**

```
Time ──────────────────────────────────────────────────────►

Prefetch thread:  [BFS batch 1] [BFS batch 2] [BFS batch 3] [BFS batch 4] ...
                        │              │              │
                        ▼              ▼              ▼
Channel buffer:   [batch 1] → [batch 2] → [batch 3]
                      │
Training loop:   [forward 1] [backward 1] [allreduce 1] [optim 1] [forward 2] ...
```

With `prefetch_depth=2`, the BFS sampling is always 1-2 batches ahead of training. The
training loop never waits for data unless sampling is slower than training (unlikely on
CPU-bound BFS with GPU-bound training).

---

## DDP Integration

### Option A: Custom Training Loop (Recommended)

Burn's built-in `Learner` / `DdpTrainingStrategy` expects a `DataLoader` that implements
`num_items()`, `slice()`, and `iter()`. Our graph sampler doesn't map cleanly to this because:

- There's no fixed "item count" — seeds are reshuffled each epoch
- `slice(start, end)` assumes contiguous partitioning, but we want interleaved
- We produce batches directly, not individual items

**The cleanest approach is a custom training loop** that uses `burn-collective` directly
for gradient sync while owning the data pipeline. Burn explicitly supports this pattern
(see their `custom-training-loop` example).

```rust
fn train_ddp<B: AutodiffBackend>(
    db: Arc<Database>,
    model: Model<B>,
    config: TrainingConfig,
    devices: Vec<B::Device>,
    collective_config: CollectiveConfig,
) {
    let world_size = devices.len();

    // One thread per device
    let handles: Vec<_> = devices.into_iter().enumerate().map(|(rank, device)| {
        let db = db.clone();
        let model = model.clone();

        std::thread::spawn(move || {
            // 1. Register with collective
            let peer_id = PeerId::from(rank);
            burn_collective::register::<B::InnerBackend>(peer_id, device, collective_config);

            // 2. Build rank-local seed pool
            let mut seed_pool = SeedPool::new(&db, &seed_config);
            seed_pool.partition_for_rank(rank, world_size);

            // 3. Build prefetching batch loader
            let loader = GraphBatchLoader::new(
                db.clone(),
                seed_pool,
                sampler_config,
                /* prefetch_depth */ 2,
                /* num_workers */ 4,
            );

            // 4. Training loop
            let mut model = model.fork(&device);
            let mut optim = optimizer.fork(&device);

            for step in 0..config.num_batches {
                let batch = loader.next_batch().expect("data exhausted");
                let batch = batch.to_device(&device);  // CPU → GPU transfer

                // Forward + backward
                let output = model.forward_training(batch);
                let loss = output.loss;
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);

                // All-reduce gradients across ranks
                let grads = grads
                    .all_reduce::<B::InnerBackend>(peer_id, ReduceOperation::Mean)
                    .expect("gradient sync failed");

                // Optimizer step
                model = optim.step(config.learning_rate, model, grads);

                // LR scheduling, logging, checkpointing (rank 0 only)
                if rank == 0 && step % config.log_interval == 0 {
                    log_metrics(step, loss);
                }
            }

            burn_collective::finish_collective::<B::InnerBackend>(peer_id);
            model
        })
    }).collect();

    // Join all threads, take model from rank 0
    let model = handles.into_iter().next().unwrap().join().unwrap();
}
```

### Option B: Adapt to Burn's DataLoader (If You Want the Learner TUI)

If you want Burn's training dashboard and checkpoint management, you can adapt to their
`DataLoader` trait with some shims:

```rust
/// A "dataset" where each item is a seed node index.
/// The actual BFS happens in the Batcher, not here.
struct SeedDataset {
    seeds: Vec<RowIdx>,
}

impl Dataset<RowIdx> for SeedDataset {
    fn get(&self, index: usize) -> Option<RowIdx> {
        self.seeds.get(index).copied()
    }
    fn len(&self) -> usize {
        self.seeds.len()
    }
}

/// A Batcher that receives Vec<RowIdx> seeds and produces a GraphBatch.
/// This is where the BFS happens.
struct GraphBatcher<B: Backend> {
    db: Arc<Database>,
    config: SamplerConfig,
}

impl<B: Backend> Batcher<B, RowIdx, GraphBatch<B>> for GraphBatcher<B> {
    fn batch(&self, seeds: Vec<RowIdx>) -> GraphBatch<B> {
        // BFS from each seed, assemble batch
        let subgraphs: Vec<RawSubgraph> = seeds
            .par_iter()
            .map(|&seed| SubgraphSampler::sample(seed, &self.db, &self.config, &mut thread_rng()))
            .collect();
        BatchAssembler::assemble(&self.db, &self.config, subgraphs)
    }
}
```

This works but has downsides:
- BFS parallelism is constrained by Burn's `MultiThreadDataLoader` threading model
  (worker threads produce items, not batches — the batcher runs single-threaded)
- No control over prefetch depth
- You lose the ability to do batch-level packing optimizations

**Recommendation: Option A (custom loop) for production, Option B for quick prototyping.**

### DDP Seed Partitioning

Regardless of which option you choose, the key DDP concern is ensuring each rank gets
disjoint seeds:

```
All seeds (shuffled): [s0, s1, s2, s3, s4, s5, s6, s7, ...]

Rank 0: [s0, s2, s4, s6, ...]  (even indices)
Rank 1: [s1, s3, s5, s7, ...]  (odd indices)
```

Each rank reshuffles independently per epoch, but the partition assignment is fixed.
This is functionally equivalent to PyTorch's `DistributedSampler`.

**Important**: The `Database` itself does NOT need to be partitioned. All ranks mmap the
same files. Only the seed list is partitioned. The BFS from rank 0's seeds may traverse
the same rows that rank 1's BFS visits — this is fine, because the gradients are averaged
via all-reduce anyway. The seeds just need to be disjoint to avoid redundant computation.

---

## Performance Considerations

### Memory Layout: Zero-Copy Sharing

```
                    Physical Memory (OS Page Cache)
                    ┌──────────────────────────────┐
                    │  graph.bin (mmap)             │
                    │  table_0000.bin (mmap)        │
                    │  table_0001.bin (mmap)        │
                    │  embeddings.bin (mmap)        │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         DDP Thread 0    DDP Thread 1    DDP Thread 2
         (GraphView)     (GraphView)     (GraphView)
         (TableViews)    (TableViews)    (TableViews)
```

All threads share the same physical pages. No data duplication. The OS page cache ensures
frequently-accessed graph regions stay hot in memory.

### BFS Hot Path Optimization

The BFS inner loop is the critical path. Key optimizations:

1. **Avoid allocations in the hot loop.** Pre-allocate the BFS queue, visited set, and
   edge buffers. Reuse across samples within a batch.

2. **Use a bitset for `visited`** instead of `HashSet`. Since RowIdx values are dense u32s,
   a `BitVec` of size `num_nodes` is both faster and more memory-efficient. For very large
   graphs (>100M nodes), use a thread-local bitset that's cleared between samples.

3. **Neighbor sampling without allocation.** When `degree > budget`, use a partial
   Fisher-Yates shuffle on the neighbor slice (which is `&[u32]` into the mmap). But since
   the mmap is read-only, copy the neighbor list to a scratch buffer first. For small degree,
   take all neighbors (no copy needed).

4. **Avoid the `row_table()` binary search in the inner loop.** Pre-compute a dense
   `row_to_table: Vec<TableIdx>` lookup (or `Vec<u16>` if tables < 65k). This costs
   `4 * num_nodes` bytes but turns O(log T) lookups into O(1).

### Threading Model

```
┌─────────────────────────────────────────────────┐
│              Prefetch Thread (1 per rank)        │
│                                                  │
│  loop {                                          │
│    seeds = seed_pool.next_batch(batch_size);     │
│                                                  │
│    // Fan out BFS to thread pool                 │
│    subgraphs = rayon::scope(|s| {                │
│      seeds.par_iter()                            │
│        .map(|seed| bfs_sample(seed))             │
│        .collect()                                │
│    });                                           │
│                                                  │
│    batch = assemble(subgraphs);                  │
│    channel.send(batch);  // blocks if full       │
│  }                                               │
└─────────────────────────────────────────────────┘
```

- **1 prefetch thread** per DDP rank (per device)
- **rayon thread pool** (shared across ranks on the same node) for parallel BFS
- **Bounded channel** (capacity 1-3) between prefetch and training
- Total CPU threads: `num_ranks * 1 (prefetch) + rayon pool size (shared)`

### Estimated Throughput

Back-of-envelope for a database with 10M rows, 50 tables, avg 10 columns:

- BFS per seed (3 hops, budget=10): ~100-300 rows visited, ~1-3 ms on mmap'd CSR
- Assembly per batch (32 sequences × 1024 cells): ~5-10 ms
- Prefetch pipeline: at 32 batches/sec, each consuming ~10ms → single thread is enough
- GPU forward+backward: ~50-200 ms per batch → data pipeline is 10-20x faster than training

**The sampler will not be the bottleneck** unless the graph is on network storage or the
BFS budget is extremely large.

---

## Appendix: Burn DataLoader Trait Compatibility

For reference, Burn's `DataLoader` trait (from `burn-core`):

```rust
pub trait DataLoader<B: Backend, O>: Send {
    fn iter(&self) -> Box<dyn DataLoaderIterator<O>>;
    fn num_items(&self) -> usize;
    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, O>>;
    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, O>>;
}
```

If you want to implement this:

- `num_items()` → `seed_pool.len()` (number of seeds, not batches)
- `slice(start, end)` → create a new loader with a subset of seeds
- `iter()` → returns an iterator that produces `GraphBatch` values
- `to_device()` → store the target device, apply in batch assembly

The main awkwardness is that Burn's `DataLoader` expects `num_items()` to return the count
of individual items (seeds), but we produce batches. Burn's `BatchDataLoader` wrapper handles
this by calling the inner `DataLoader` for individual items and grouping them. This means
the `Batcher` receives `Vec<RowIdx>` and must do the BFS there — which works but loses
prefetch control.

The cleaner path is the custom training loop (Option A), using `burn-collective` directly.
