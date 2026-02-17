# System Architecture

How the Rust sampler (headwater) and Python training loop (confluence) connect in a
DistributedDataParallel setup.

---

## 1. Process Topology

Each GPU gets **one Python process** and **one embedded Rust sampler** (loaded as a native
extension via PyO3). On an 8×B200 node, that means 8 independent OS processes, each pinned to
one GPU.

```
┌─────────── Node ──────────────────────────────────────────────────┐
│                                                                    │
│  ┌─── Process 0 (GPU 0) ──────────────────────────────────────┐   │
│  │  Python (confluence)          Rust (headwater, via PyO3)    │   │
│  │  ┌────────────────────┐       ┌──────────────────────┐     │   │
│  │  │ JAX train loop     │◄─────►│ Sampler              │     │   │
│  │  │ forward / backward │       │  rayon thread pool    │     │   │
│  │  │ allreduce grads    │       │  mmap'd graph + data  │     │   │
│  │  └────────────────────┘       └──────────┬───────────┘     │   │
│  └──────────────────────────────────────────┼─────────────────┘   │
│                                              │ shared mmap pages   │
│  ┌─── Process 1 (GPU 1) ────────────────────┼─────────────────┐   │
│  │  ...same structure...                     │                 │   │
│  │                              mmap'd graph + data ◄──────────┘   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ... processes 2–7 ...                                             │
└────────────────────────────────────────────────────────────────────┘
```

**Key properties:**

- Rust samplers across processes **share physical memory** for the read-only preprocessed
  database via OS-level memory-mapped files (`memmap2`). Eight processes mapping the same
  `graph.bin` / `table.bin` files share a single set of physical pages in the kernel page cache.
- Each process's Rust sampler runs its own **rayon thread pool** for parallel BFS across seeds.
- Gradient synchronization between GPUs uses **JAX collective ops** (`jax.lax.pmean`), routed
  over NCCL.
- There is **no Rust-to-Rust communication** between processes. Samplers are fully independent.

---

## 2. Rust Sampler Architecture (headwater)

### 2a. Memory-Mapped Data

At sampler initialization, Rust opens the preprocessed database directory and `mmap`s the
binary files read-only:

| File | Contents | Access pattern |
|------|----------|----------------|
| `graph.bin` | CSR adjacency (F→P and P→F) | Random reads during BFS |
| `tables/*.bin` | Per-table column data (values, validity bitmaps, embedding indices) | Sequential reads when visiting a row |
| `text_embeddings.bin` | `[Vt, D_t]` f16 vectors | Gather by `TextEmbeddingIdx` during batch collation |
| `tasks/*.bin` | Materialized `(anchor_row, obs_time, target_value)` tuples per task | Sequential scan of seed list |
| `metadata.json` | `DatabaseMetadata` (table/column/task metadata, stats) | Read once at init, kept in-memory |
| `column_embeddings.bin` | `[C, D_t]` f16 column-name embeddings | Read once, uploaded to GPU by Python |
| `categorical_embeddings.bin` | `[Vc, D_t]` f16 category embeddings | Read once, uploaded to GPU by Python |

Memory-mapped I/O means the OS kernel manages paging. Hot pages (graph structure, frequently
accessed table columns) stay resident; cold pages (rarely sampled rows, unused text embeddings)
can be evicted under memory pressure. Because the mappings are read-only, all processes sharing
a node share the same physical pages — an 8-process node uses roughly 1× the data size in RAM,
not 8×.

### 2b. Train / Val / Test Splitting

All tasks — temporal or not — use **hash-based splitting**. The observation time on each seed
is used solely for BFS temporal filtering during sampling, not for split assignment.

#### Why not split by time?

Each seed already carries its own `obs_time`, which the BFS respects regardless of split
membership. A val seed with `obs_time = June 2022` gets its BFS filtered to data before June
2022, even though the database may contain data through 2025. The temporal filtering is a
property of the **seed**, not the **split**.

Splitting by time would add a second concern on top of this: "ensure no train seed comes from
after time T." But that conflates two independent properties:
1. **What context does the model see?** → governed by per-seed `obs_time` and BFS filtering.
2. **Which seeds do we train on vs. evaluate on?** → governed by split assignment.

Hash-based splitting decouples them. One mechanism, no special cases, no per-task mode
detection, and no need to pick temporal cutoffs.

If you want to measure temporal generalization (does the model degrade on future data?), you
can always bucket validation metrics by observation time post-hoc.

#### Mechanism

Every seed is assigned to a split via a deterministic hash:

```
bucket = stable_hash(task_idx, anchor_row_id, split_seed) % 1000
split  = Train  if bucket < split_ratios.train * 1000
         Val    if bucket < (split_ratios.train + split_ratios.val) * 1000
         Test   otherwise
```

Properties:
- **Deterministic across ranks**: all processes compute the same hash, so they agree on which
  seeds are train/val/test. No communication needed.
- **Stable across runs**: same `split_seed` + ratios → same assignment, regardless of sampling
  RNG state or `rank`. Changing the sampling `seed` reshuffles iteration order but not split
  membership.
- **Per-task independence**: including `task_idx` in the hash means a given anchor row can be
  train for one task and val for another. This maximizes data utilization — every task sees
  most of its seeds during training.

The default ratios are 80/10/10 (train/val/test), configurable via `split_ratios`.

`split_seed` is a separate parameter from the sampling `seed` so that you can change sampling
randomness (BFS child subsampling, iteration order) without invalidating your split assignments,
and vice versa.

#### What validation actually measures

The model's BFS context for a given seed is governed by that seed's `obs_time`:

- **Temporal seed** (anchor table has timestamps): BFS excludes rows created after `obs_time`.
  The model can't see future data for this seed — regardless of whether other future seeds
  are in the train set.
- **Non-temporal seed** (`obs_time = i64::MAX`): BFS sees the entire graph. The only thing
  hidden is the target cell value (replaced by `mask_emb`).

In both cases, target leakage is prevented by masking. Feature leakage is prevented by per-seed
BFS filtering. The split is **transductive** — a val seed's BFS may traverse rows that are
train seeds for other tasks — but that's fine because BFS only reads features, never targets.
This is standard in GNN and knowledge graph literature.

| Property | Temporal seeds | Non-temporal seeds |
|----------|---------------|-------------------|
| BFS filtering | Rows after `obs_time` excluded | Full graph visible |
| Target leakage | Masked via `is_target` | Masked via `is_target` |
| Feature leakage | Prevented by `obs_time` cutoff | N/A (no "future" exists) |
| What val measures | Predict held-out values from temporally-filtered context | Predict held-out values from full graph |

#### Post-hoc temporal analysis

If deployment realism matters (e.g., "will this model work on next month's data?"), you can
analyze validation metrics by observation time bucket after training:

```python
val_metrics_by_time = defaultdict(list)
for batch in val_batches:
    loss = eval_step(params, batch, ...)
    obs_time = batch['obs_time']  # or retrieve from task metadata
    bucket = time_bucket(obs_time)
    val_metrics_by_time[bucket].append(loss)
```

This gives you the temporal generalization signal without baking it into the split mechanism.

### 2c. Seed Sharding (across ranks)

After splitting, each split's seed list is sharded across DDP ranks. For a given task with N
seeds in a split, rank `r` owns the subset:

```
my_seeds = { seed_i : i % world_size == rank }
```

This deterministic round-robin assignment ensures:
- **No duplication**: every seed is seen by exactly one process per epoch.
- **Load balance**: seeds are evenly distributed (±1).
- **Reproducibility**: given the same `rank`, `world_size`, and `split_seed`, the assignment is
  identical across runs.

Sharding happens *after* splitting: the sampler first assigns all seeds to train/val/test, then
partitions each split's seed list by rank. This guarantees that sharding preserves the split
ratios on each rank (up to rounding).

### 2d. Sampling Pipeline (per batch)

A batch is constructed in two phases, as described in `batch_structure.md`:

**Phase 1 — Parallel BFS (rayon `par_iter` over B seeds):**

Each of the B seeds is processed independently on a rayon worker thread:

1. Pick a task (weighted random or round-robin across tasks).
2. Draw a seed row from this rank's shard of the selected task's split.
3. BFS from the seed row over FK edges, respecting the temporal cutoff and sequence length budget
   L (see `sampling.md`).
4. Linearize visited cells into a flat sequence. Build the per-sequence `HashMap<RowIdx, u16>`
   for row remapping, the `fk_adj` adjacency slice, and the attention permutations.
5. Write cell values into pre-allocated per-sequence buffers. Text cells temporarily store
   global `TextEmbeddingIdx` values.

**Phase 2 — Batch collation (single-threaded sync point):**

1. Collect all B sequences' global `TextEmbeddingIdx` values. Build a batch-wide dedup map
   `HashMap<TextEmbeddingIdx, u32>` → batch-local indices 0..U-1.
2. Gather U text embedding vectors from the mmap'd `text_embeddings.bin` into a contiguous
   `Vec<f16>` of shape `[U, D_t]`.
3. Rewrite each sequence's `text_embed_ids` with batch-local indices.
4. Assemble all batch tensors into flat `Vec<T>` buffers with the correct shapes.

Phase 1 parallelizes cleanly across rayon threads. Phase 2 is fast (a hash-map build + a
gather) and is the only serial bottleneck.

### 2e. Prefetch Pipeline

The sampler runs a **background producer loop** that continuously generates batches ahead of
the Python consumer. This is implemented as a dedicated Rust thread (outside the rayon pool)
that pushes completed batches into a bounded crossbeam channel:

```
 Rust producer thread              crossbeam::bounded(N)         Python consumer
 ┌──────────────────┐              ┌───────────────┐             ┌──────────────┐
 │ loop {           │    push      │               │    pull     │              │
 │   batch = sample │ ──────────►  │  batch queue  │ ──────────► │ next_batch() │
 │   ...            │              │  (capacity 3) │             │              │
 │ }                │              │               │             │              │
 └──────────────────┘              └───────────────┘             └──────────────┘
```

The channel capacity (default: 3) bounds memory usage. When the channel is full, the producer
blocks until the consumer pulls a batch. When the channel is empty, the consumer blocks until
a batch is ready.

Two independent channels are maintained — one for train, one for val — so validation batches
can be produced without tearing down the training prefetch pipeline.

---

## 3. PyO3 API Boundary

### 3a. Python-Visible API

The Rust crate is built as a Python extension module via **maturin** (PEP 517 build backend for
PyO3 crates). Python imports it as a normal package:

```python
import headwater

sampler = headwater.Sampler(
    db_path="data/processed/my_database",
    rank=jax.process_index(),
    world_size=jax.process_count(),
    split_ratios=(0.8, 0.1, 0.1),   # train/val/test hash split ratios
    split_seed=123,                  # deterministic hash seed for split assignment
    seed=42,                         # RNG seed for sampling randomness
    num_prefetch=3,                  # prefetch channel capacity
    default_batch_size=32,
    default_sequence_length=1024,
    bfs_child_width=16,              # max children sampled per P→F edge
)
```

On construction, the sampler:
1. Opens and mmap's all preprocessed files.
2. Deserializes `DatabaseMetadata` from `metadata.json`.
3. Hash-assigns each task's seeds into train/val/test splits.
4. Shards each split's seeds by `rank` / `world_size`.
5. Spawns the rayon thread pool and the two prefetch producer threads.

```python
# Returns a dict of numpy arrays (zero-copy from Rust allocations).
# Blocks until a batch is available in the prefetch channel.
train_batch: dict[str, np.ndarray] = sampler.next_train_batch()
val_batch:   dict[str, np.ndarray] = sampler.next_val_batch()

# Metadata accessors (for one-time GPU table uploads).
col_embeddings: np.ndarray  = sampler.column_embeddings()       # [C, D_t] f16
cat_embeddings: np.ndarray  = sampler.categorical_embeddings()   # [Vc, D_t] f16
metadata:       dict        = sampler.database_metadata()        # JSON-like dict

# Lifecycle.
sampler.shutdown()  # drain channels, join threads, drop mmaps
```

### 3b. Batch Dict Layout

`next_train_batch()` and `next_val_batch()` return a Python `dict[str, numpy.ndarray]`:

| Key | Shape | Rust type | NumPy dtype | Notes |
|-----|-------|-----------|-------------|-------|
| `semantic_types` | `[B, S]` | `i8` | `int8` | |
| `column_ids` | `[B, S]` | `i32` | `int32` | Global `ColumnIdx` |
| `seq_row_ids` | `[B, S]` | `u16` | `uint16` | Sequence-local row index |
| `numeric_values` | `[B, S]` | `f32` | `float32` | Z-score; cast to bf16 on GPU |
| `timestamp_values` | `[B, S, 15]` | `f32` | `float32` | Cyclic encoding; cast to bf16 on GPU |
| `bool_values` | `[B, S]` | `u8` | `uint8` | 0/1; cast to bool on GPU |
| `categorical_embed_ids` | `[B, S]` | `u32` | `uint32` | Global `CategoricalEmbeddingIdx` |
| `text_embed_ids` | `[B, S]` | `u32` | `uint32` | Batch-local text index |
| `is_null` | `[B, S]` | `u8` | `uint8` | 0/1 |
| `is_target` | `[B, S]` | `u8` | `uint8` | 0/1 |
| `is_padding` | `[B, S]` | `u8` | `uint8` | 0/1 |
| `fk_adj` | `[B, R, R]` | `u8` | `uint8` | Boolean adjacency |
| `col_perm` | `[B, S]` | `u16` | `uint16` | Attention permutation |
| `out_perm` | `[B, S]` | `u16` | `uint16` | Attention permutation |
| `in_perm` | `[B, S]` | `u16` | `uint16` | Attention permutation |
| `text_batch_embeddings` | `[U, D_t]` | `f16` | `float16` | Per-batch text subset |
| `target_stype` | `[1]` | `u8` | `uint8` | Batch is homogeneous |
| `task_idx` | `[1]` | `u32` | `uint32` | For stats / logging |
| `cat_emb_start` | `[1]` | `u32` | `uint32` | Offset into categorical table |
| `cat_emb_count` | `[1]` | `u32` | `uint32` | K for the target column |

**Dtype rationale**: Rust produces `f32` for numeric/timestamp values and `u8` for booleans /
masks. The bf16 cast happens on-device during the JAX forward pass — this avoids bf16 rounding
in the sampler and lets XLA fuse the cast with the first matmul.

### 3c. Zero-Copy Memory Transfer

Each batch tensor is backed by a Rust `Vec<T>` allocated during batch collation. PyO3 hands
ownership of the `Vec` to NumPy via `PyArray::from_vec()`:

```rust
// Rust side (simplified)
fn next_train_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    // Release the GIL while we block on the channel.
    let batch: RawBatch = py.allow_threads(|| {
        self.train_channel.recv()
            .map_err(|_| SamplerShutdown)
    })?;

    let dict = PyDict::new(py);

    // Zero-copy: Vec<f32> → numpy array. Numpy now owns this memory.
    // When Python GCs the array, PyO3's destructor drops the Vec.
    let numeric = PyArray2::from_vec(py, batch.numeric_values)
        .reshape([batch.B, batch.S])?;
    dict.set_item("numeric_values", numeric)?;

    // ... repeat for all tensors ...
    Ok(dict)
}
```

The key properties of this transfer:

1. **Rust allocates.** During Phase 1+2, rayon workers fill `Vec<T>` buffers. These live on the
   Rust heap (mimalloc).
2. **Ownership transfers to Python.** `PyArray::from_vec(py, vec)` consumes the `Vec` and
   creates a NumPy array whose data pointer is the Vec's buffer. No copy occurs.
3. **Python frees.** When the NumPy array is garbage-collected (typically at the end of one
   training iteration, when the local `batch` variable goes out of scope), Python invokes the
   release callback, which drops the Rust `Vec`, returning memory to mimalloc.

This gives us **exactly one allocation and zero copies** on the CPU side, per tensor per batch.

### 3d. GIL Management

The GIL must be released during all blocking or compute-heavy Rust operations, so that
Python's main thread (and JAX's background threads) can make progress:

| Operation | GIL status | Why |
|-----------|-----------|-----|
| `Sampler::new()` | Released during mmap + metadata load | File I/O can block |
| `next_train_batch()` / `next_val_batch()` | Released during `channel.recv()` | Main thread must be free to run JAX ops |
| Prefetch producer thread | Never holds GIL | Rust-native thread, no Python interaction |
| `column_embeddings()` / `categorical_embeddings()` | Released during mmap read + copy | One-time; may page-fault |
| `shutdown()` | Released during thread join | Avoid deadlock |

PyO3 pattern: wrap blocking work in `py.allow_threads(|| { ... })`. The prefetch producer
threads are spawned via `std::thread::spawn` (not Python threads), so they never touch the GIL.

---

## 4. Python Training Architecture (confluence)

### 4a. JAX Distributed Setup

Each process initializes the JAX distributed runtime at startup:

```python
jax.distributed.initialize()  # reads SLURM / coordinator env vars
rank = jax.process_index()
world_size = jax.process_count()
device = jax.local_devices()[0]  # one GPU per process
```

Model parameters are **replicated** across all processes. Gradients are **averaged** inside the
JIT-compiled training step via `jax.lax.pmean`. This is standard single-program-multiple-data
(SPMD) DDP.

### 4b. GPU-Resident Tables (uploaded once)

Three embedding tables are loaded from the sampler at startup and placed on-device permanently:

```python
col_emb_table  = jax.device_put(sampler.column_embeddings(), device)       # [C, D_t] f16
cat_emb_table  = jax.device_put(sampler.categorical_embeddings(), device)  # [Vc, D_t] f16
```

These are small (typically < 20 MB combined) and used by every forward pass, so keeping them
GPU-resident avoids per-batch transfer overhead. They are **not** model parameters — they are
frozen lookup tables from the preprocessor.

The text embedding table is too large to keep GPU-resident (potentially gigabytes for
high-cardinality text columns). Instead, the per-batch `text_batch_embeddings` tensor is
transferred with each batch.

### 4c. Training Loop

```python
sampler = headwater.Sampler(
    db_path=db_path,
    rank=rank,
    world_size=world_size,
    split_ratios=(0.8, 0.1, 0.1),
    split_seed=123,
    seed=42,
    num_prefetch=3,
)

col_emb_table = jax.device_put(sampler.column_embeddings(), device)
cat_emb_table = jax.device_put(sampler.categorical_embeddings(), device)

model = RelationalTransformer(config)
params = model.init(rng, dummy_batch, col_emb_table, cat_emb_table)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, batch, col_emb_table, cat_emb_table):
    def loss_fn(params):
        output = model.apply(params, batch, col_emb_table, cat_emb_table)
        loss = compute_loss(output, batch)
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads = jax.lax.pmean(grads, axis_name='devices')      # allreduce
    loss  = jax.lax.pmean(loss,  axis_name='devices')       # sync loss for logging
    params, opt_state = optimizer.update(grads, opt_state, params)
    return params, opt_state, loss

for step in range(num_steps):
    # Pull next pre-built batch from Rust (blocks if not ready yet).
    # GIL is released inside next_train_batch() so JAX async dispatch continues.
    np_batch = sampler.next_train_batch()

    # Host-to-device transfer. XLA handles pinned staging internally (see §5b).
    device_batch = jax.device_put(np_batch, device)

    # Dispatch train step. JAX executes asynchronously — the Python thread
    # returns immediately and can pull the next batch while GPU is busy.
    params, opt_state, loss = train_step(
        params, opt_state, device_batch, col_emb_table, cat_emb_table
    )

    # Periodic validation.
    if step % eval_interval == 0:
        val_metrics = run_validation(sampler, model, params, ...)
```

### 4d. Validation

Validation uses the same sampler, pulling from the val prefetch channel. The validation batch
pipeline runs independently from training — no need to pause or restart prefetching:

```python
def run_validation(sampler, model, params, col_emb_table, cat_emb_table,
                   num_val_steps, device):
    total_loss = 0.0
    for _ in range(num_val_steps):
        np_batch = sampler.next_val_batch()
        device_batch = jax.device_put(np_batch, device)
        loss = eval_step(params, device_batch, col_emb_table, cat_emb_table)
        total_loss += float(loss)
    return total_loss / num_val_steps
```

The val prefetch channel can use a smaller capacity (e.g., 1–2) since validation is periodic
and doesn't need deep pipelining. The Rust producer thread for validation batches runs at lower
priority, yielding rayon workers to the training producer when both are active.

### 4e. Numerical Stability in Mixed Precision

The model operates in bfloat16 for most computation, with targeted float32 upcasts where
low-precision gradients would produce NaN or overflow. Three design decisions enforce this:

**1. No explicit padding zeroing in the value encoder.**  
The original design zeroed padding positions via `h0 = h0 * (1 - is_padding)`. This created
exact-zero hidden states whose gradients through L2 normalization and RMSNorm produced NaN
(the gradient of `x / ||x||` at `x = 0` is `I / eps`, which overflows in bf16). Since attention
masks already exclude padding from affecting non-padding positions, the zeroing was redundant
and is omitted.

**2. Float32 attention with safe L2 normalization.**  
QK-normalized attention (Q and K are L2-normalized before the dot product) is particularly
sensitive to low-precision gradients. The QKV projections use no bias, so padding positions
produce zero Q/K vectors after projection. The attention layer:
- Upcasts Q and K to float32 before normalization and softmax.
- Uses `q / max(||q||, 1e-6)` instead of `q / (||q|| + eps)` for L2 normalization. The
  `jnp.maximum` form is both more numerically stable and avoids the JAX `jnp.where` gradient
  trap (JAX traces both branches of `jnp.where`, so guarding with a condition doesn't prevent
  NaN gradients in the masked branch).
- Uses `-1e9` instead of `jnp.finfo(dtype).min` as the mask fill value, avoiding extreme
  softmax inputs that could cause numerical issues in the backward pass.
- Casts back to the input dtype (bf16) after the softmax-weighted sum.

**3. Diagonal self-attention in all masks.**  
Every attention mask (outbound, inbound, column) includes the identity diagonal, ensuring
every position can attend to at least itself. Without this, padding positions whose mask rows
are all-False produce softmax over all-`-inf` inputs → `0/0 = NaN` in the backward pass.
The diagonal is harmless for non-padding positions (self-attention is semantically valid) and
for padding positions (their hidden states don't contribute to the loss).

### 4f. Optimizer

The optimizer uses `optax.contrib.muon`, which automatically partitions parameters: 2D weight
matrices use Muon (momentum + Newton-Schulz orthogonalization), and everything else (biases,
norms, embeddings, 1D/3D params) uses AdamW. A single `learning_rate` schedule governs Muon;
AdamW's hyperparameters (`adam_b1`, `adam_b2`, `adam_weight_decay`) are set separately.

The LR schedule is linear warmup + cosine decay to 10% of peak. Gradient clipping
(`clip_by_global_norm`) is applied before the optimizer step.

### 4g. Loss Dispatch

Every batch is homogeneous (one `target_stype`). The forward pass computes all decoder heads
unconditionally, and the loss function selects the right one via masking (no graph breaks):

```python
type_losses = jnp.stack([num_loss, bool_loss, ts_loss, cat_loss])  # [4, B]
type_selector = jax.nn.one_hot(batch['target_stype'], 4)           # [4]
type_loss = type_selector @ type_losses                            # [B]
batch_loss = jnp.mean(null_loss + (1 - is_null_at_target) * type_loss)
```

See `decoder_heads_and_loss.md` for full details.

### 4h. Smoke Test

A self-contained smoke test (`confluence/smoke_test.py`) verifies the full JAX pipeline
end-to-end without the Rust sampler. It generates synthetic dummy batches with random values,
initializes a small model (2 layers, d=128), and runs 30 training steps. This exercises:
- Model initialization and JIT compilation
- Forward pass through value encoder, attention masks, transformer layers, and decoder heads
- Loss computation (null loss + type-specific loss with one-hot dispatch)
- Backward pass (gradient computation)
- Optimizer step (Muon for 2D matrices, AdamW for the rest)

Run with: `uv run --project confluence python confluence/smoke_test.py`

---

## 5. End-to-End Data Pipeline

### 5a. Pipeline Stages and Overlap

```
Timeline (steady state):
────────────────────────────────────────────────────────────────────────

Rust rayon pool:    [── sample batch N+2 ──][── sample batch N+3 ──]
                              │
                    push to crossbeam channel
                              │
                              ▼
Python main thread: [pull N+1][device_put N+1][─── train_step(N) on GPU ───]
                                     │
                          XLA async H2D transfer
                                     │
                                     ▼
GPU:                                 [── transfer N+1 ──][── train_step(N) ──]
```

Three pipeline stages run concurrently:
1. **Rust sampling** (CPU, rayon threads): BFS + collation for batch N+2.
2. **Host→device transfer** (XLA staging): batch N+1 moving to GPU memory.
3. **GPU compute** (XLA): forward + backward + allreduce for batch N.

Because JAX dispatches asynchronously, the Python main thread returns from `train_step()` as
soon as the computation is *enqueued*, not when it finishes. This lets Python immediately pull
the next batch from Rust and call `device_put`, overlapping CPU and GPU work.

### 5b. Host→Device Transfer: XLA Pinned Staging

When Python calls `jax.device_put(numpy_array)`, XLA's runtime handles the transfer:

1. **Allocate or reuse** a page-locked (pinned) staging buffer from XLA's internal pool.
2. **memcpy** from the NumPy array's data pointer into the pinned staging buffer.
3. **Async DMA** from the pinned staging buffer to GPU device memory (via `cudaMemcpyAsync`).
4. Return a `jax.Array` future that resolves when the DMA completes.

The staging copy (step 2) is an extra memcpy, but it's fast (memory bandwidth-limited, not
latency-limited) and XLA pools the pinned buffers to avoid repeated `cudaMallocHost` calls.

**Why we don't allocate pinned memory in Rust directly**: While we *could* use `cudarc` to call
`cudaMallocHost` and write batch tensors directly into pinned buffers, XLA has no public API to
adopt a pre-pinned host buffer as a device-put source. We'd need to wrap the pinned memory as a
`jax.Array` or use DLPack with device-type hints, and the XLA backend may still copy it into
its own staging area. The complexity isn't worth the marginal gain — the sampling and GPU
compute stages dominate the pipeline, not the staging memcpy.

### 5c. Memory Lifecycle of a Batch Tensor

```
 ┌────────────────────┐
 │ 1. Rust Vec<f32>   │  mimalloc alloc during Phase 1/2
 │    (rayon workers)  │
 └────────┬───────────┘
          │  PyArray::from_vec() — zero-copy, ownership transfer
          ▼
 ┌────────────────────┐
 │ 2. NumPy ndarray   │  Python holds a reference; data pointer = Rust Vec's buffer
 │    (Python heap)    │
 └────────┬───────────┘
          │  jax.device_put() — XLA copies to pinned staging, then async DMA
          ▼
 ┌────────────────────┐
 │ 3. jax.Array       │  XLA device buffer on GPU
 │    (GPU memory)     │
 └────────┬───────────┘
          │  Python drops reference to the NumPy array (end of iteration)
          ▼
 ┌────────────────────┐
 │ 4. Vec dropped     │  PyO3 release callback fires, mimalloc frees
 │    (CPU memory)     │
 └────────────────────┘
```

At steady state, roughly 3 batches worth of CPU memory are alive:
- Batch N: GPU-resident `jax.Array` (NumPy/Rust buffers already freed).
- Batch N+1: NumPy arrays being transferred via `device_put`.
- Batch N+2: sitting in the crossbeam channel, ready for Python to pull.
- Batch N+3: being constructed by rayon workers.

(N+2 and N+3 account for the channel capacity.)

---

## 6. Library Roles

### rayon

Parallel BFS sampling across the B seeds within a single batch. Each seed's subgraph walk is
independent, making this embarrassingly parallel. Rayon's work-stealing pool ensures even
load distribution when some seeds have deeper/wider BFS trees than others. Also used for
parallel Phase 2 text-embedding gather if U is large.

### PyO3

The FFI boundary between Rust and Python. Exposes the `Sampler` class as a native Python
object with `#[pymethods]`. Handles:
- GIL release (`py.allow_threads`) during blocking operations.
- Ownership transfer of `Vec<T>` → NumPy arrays via `PyArray::from_vec`.
- Type conversions between Rust structs and Python dicts.
- The `#[pymodule]` entry point that maturin compiles into a `.so` / `.pyd`.

### maturin

PEP 517 build backend for PyO3 extension modules. Running `maturin develop` (for dev) or
`maturin build --release` (for wheels) compiles the Rust crate with `--release` optimizations
and produces a Python-importable native extension. Integrates with `pyproject.toml` so that
`uv pip install .` or `pip install .` in the `headwater` directory just works.

### numpy (via pyo3-numpy)

The `pyo3-numpy` crate provides `PyArray::from_vec()` for zero-copy Vec → ndarray conversion,
and `PyReadonlyArray` for accepting numpy arrays from Python (e.g., if we want to pass
configuration arrays in). NumPy arrays are the lingua franca between Rust and JAX —
`jax.device_put` accepts them natively.

### memmap2

Memory-mapped file access for all preprocessed binary data (graph, tables, embeddings, tasks).
Provides `Mmap` (read-only) and `MmapMut` (for the preprocessor). Key benefits:
- **Cross-process sharing**: 8 processes mapping the same file share physical pages.
- **Lazy loading**: pages are faulted in on first access; the entire dataset doesn't need to
  fit in RAM if the working set is smaller.
- **Zero-copy reads**: the sampler reads cell values and embeddings directly from mmap'd
  memory without intermediate buffers.
- `GraphView` and the column data readers use `Mmap` slices as their backing store.

### crossbeam

Lock-free bounded MPMC channels (`crossbeam::channel::bounded`) for the prefetch pipeline.
The producer (Rust background thread) pushes completed `RawBatch` structs; the consumer
(Python via `next_train_batch()`) pulls them. Bounded capacity provides natural backpressure —
if the GPU is slower than sampling, the producer blocks; if sampling is slower, the consumer
blocks. Two channels (train + val) run independently.

### cudarc

**Not used in the initial implementation.** Identified as relevant if we later want to:
- Allocate batch buffers in CUDA pinned memory directly (bypassing XLA staging).
- Perform custom CUDA kernels for on-device batch preprocessing.
- Manage GPU memory pools for the sampler's output buffers.

For now, XLA's internal pinned-staging pool handles host→device transfer efficiently. cudarc
becomes relevant if profiling reveals the staging memcpy as a bottleneck (unlikely given that
GPU compute dominates) or if we want to build a direct Rust→GPU DMA path that bypasses Python
entirely.

---

## 7. Dtype Casting Strategy

Keeping numeric precision high in the sampler and casting on-device avoids rounding errors
during normalization and lets XLA fuse the cast with the first operation that consumes the data:

| Tensor | Sampler (CPU) | After `device_put` | Forward pass (GPU) |
|--------|--------------|--------------------|--------------------|
| `numeric_values` | f32 | f32 | Cast to bf16 at first use |
| `timestamp_values` | f32 | f32 | Cast to bf16 at first use |
| `text_batch_embeddings` | f16 | f16 | Cast to bf16 at first use |
| `bool_values` | u8 | u8 | Cast to bf16 via embedding lookup |
| Index tensors (`column_ids`, `*_embed_ids`, `*_perm`) | u16/u32/i32 | Same | Used as integer indices |
| Mask tensors (`is_null`, `is_target`, `is_padding`) | u8 | u8 | Cast to bool or bf16 as needed |

The model's value encoder handles all casts at the boundary of the value encoding layer, after
which everything is bf16 throughout the transformer — except for attention QK normalization and
softmax, which are computed in float32 for numerical stability (see §4e).

---

## 8. Configuration and Hyperparameters

### Sampler Configuration (Rust-side)

| Parameter | Type | Description |
|-----------|------|-------------|
| `db_path` | `String` | Path to preprocessed database directory |
| `rank` | `u32` | This process's rank in the DDP group |
| `world_size` | `u32` | Total number of processes |
| `split_ratios` | `(f32, f32, f32)` | Train/val/test hash split ratios (default: 0.8/0.1/0.1) |
| `split_seed` | `u64` | Hash seed for split assignment (same across all ranks) |
| `seed` | `u64` | RNG seed for sampling randomness (combined with rank for per-process determinism) |
| `num_prefetch` | `usize` | Crossbeam channel capacity (default: 3) |
| `default_batch_size` | `u32` | B (default: 32) |
| `default_sequence_length` | `u32` | S (default: 1024) |
| `bfs_child_width` | `u32` | Max children per P→F edge during BFS (default: 16) |
| `task_weights` | `Option<Vec<f32>>` | Sampling weights per task (default: uniform) |

### Training Configuration (Python-side)

See `confluence/confluence/config.py` for optimizer, precision, LR schedule, and regularization
settings. These are independent of the sampler — the sampler produces batches, and the training
loop consumes them.

---

## 9. Failure Modes and Safety

| Scenario | Behavior |
|----------|----------|
| Sampler channel drained after `shutdown()` | `next_train_batch()` raises `SamplerShutdown` |
| BFS finds fewer than S cells | Sequence is padded; `is_padding` marks unused positions. Padding positions retain non-zero hidden states (no explicit zeroing) to avoid NaN gradients; attention masks exclude them from computation. |
| Task has zero seeds for this rank's shard | Sampler skips the task; warns via `tracing` |
| mmap'd file is corrupted or truncated | Rust returns `io::Error` at mmap creation; Python sees an exception |
| One process crashes mid-training | NCCL allreduce will hang on other ranks; use a watchdog/timeout to detect and abort |
| OOM on batch allocation | Reduce `num_prefetch`, `default_batch_size`, or `default_sequence_length` |

---

## 10. Future Work

- **Direct Rust→GPU DMA**: If profiling shows the XLA staging copy matters, investigate cudarc-based
  pinned memory allocation with DLPack interop to JAX.
- **Multi-database training**: A single sampler process managing multiple mmap'd databases, switching
  between them per-batch. Requires extending `Sampler::new()` to accept a list of database paths.
- **Dynamic batching**: Variable sequence length S per batch to reduce padding waste. Requires
  changes to both the sampler (bin-packing seeds by BFS tree size) and the model (dynamic shapes
  or bucketed JIT compilation).
- **Gradient accumulation**: For effective batch sizes larger than GPU memory allows, accumulate
  gradients over multiple micro-batches before allreduce. Purely a Python-side change.
