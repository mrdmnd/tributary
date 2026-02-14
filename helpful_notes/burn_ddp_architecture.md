# Burn Framework: Distributed Data Parallel (DDP) Training Architecture

> Analysis of the Burn ML framework's DDP implementation, based on reading the source code of
> `burn-collective`, `burn-communication`, `burn-train`, `burn-core`, and `burn-optim` crates.
> Covers the architecture as of the `0.21.0-pre.1` development branch (post-0.20.0).

---

## Table of Contents

- [Overview](#overview)
- [Process & Thread Model](#process--thread-model)
- [Data Distribution & Samplers](#data-distribution--samplers)
- [Batching](#batching)
- [Gradient Synchronization](#gradient-synchronization)
- [Collective Communication Layer](#collective-communication-layer)
- [Multi-Node Training](#multi-node-training)
- [Communication Protocol](#communication-protocol)
- [Complete Training Flow](#complete-training-flow)
- [Comparison with PyTorch DDP](#comparison-with-pytorch-ddp)

---

## Overview

Burn's DDP training is implemented across several crates:

| Crate | Role |
|-------|------|
| `burn-train` | Learner, training loop, DDP strategy, DDP worker, epoch runner |
| `burn-optim` | `GradientsParams` with `all_reduce()` for per-parameter gradient sync |
| `burn-collective` | Collective communication primitives (all-reduce, reduce, broadcast) |
| `burn-communication` | Protocol-agnostic network transport (WebSocket impl, tensor data service) |
| `burn-core` | Data loading, `split_dataloader()`, `BatchDataLoader`, `PartialDataset` |

The DDP feature is gated behind a feature flag in `burn-train`:

```toml
ddp = ["burn-collective", "burn-optim/collective"]
```

---

## Process & Thread Model

**Key answer: One OS thread per device, NOT one process per rank.**

Unlike PyTorch's DDP (which uses one process per rank, launched via `torchrun`), Burn spawns
**one OS thread per device** within a single process on each node.

### How It Works

1. `TrainingStrategy::DistributedDataParallel` is selected with a list of devices and a
   `CollectiveConfig`.
2. `DdpTrainingStrategy::fit()` spawns N `DdpWorker` threads — one per device.
3. Each worker gets a `PeerId(u32)` (analogous to a "rank") — values 0, 1, 2, ...
4. Device 0's thread is the **main worker** — it handles validation, checkpointing, and event
   processing. All other threads are secondary workers.

```
Single Node (e.g. 4 GPUs)
┌──────────────────────────────────────────────────────────┐
│  Process                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Thread 0 │ │ Thread 1 │ │ Thread 2 │ │ Thread 3 │    │
│  │ PeerId=0 │ │ PeerId=1 │ │ PeerId=2 │ │ PeerId=3 │    │
│  │ GPU 0    │ │ GPU 1    │ │ GPU 2    │ │ GPU 3    │    │
│  │ (main)   │ │          │ │          │ │          │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
│       │             │             │             │         │
│       └─────────────┴─────────────┴─────────────┘         │
│                LocalCollectiveServer                      │
└──────────────────────────────────────────────────────────┘
```

For multi-node training, each node runs its own process with its own set of threads. Nodes
coordinate via a Global Orchestrator (a separate WebSocket server).

### Relevant Source: `strategies/ddp/strategy.rs`

```rust
pub struct DdpTrainingStrategy<LC: LearningComponentsTypes> {
    devices: Vec<Device<TrainingBackend<LC>>>,
    config: CollectiveConfig,
}
```

In `fit()`:
- Calls `split_dataloader()` to partition the training data
- Spawns one `DdpWorker` thread per device
- Device 0 = main worker (handles validation + checkpointing)
- Waits for all threads to join, returns main worker's model

---

## Data Distribution & Samplers

**Key answer: No `DistributedSampler`. Uses contiguous slicing via `split_dataloader()`.**

### `split_dataloader()` — `burn-core/src/data/dataloader/split.rs`

```rust
pub fn split_dataloader<B: Backend, O>(
    dataloader: Arc<dyn DataLoader<B, O>>,
    devices: &[B::Device],
) -> Vec<Arc<dyn DataLoader<B, O>>> {
    let num_splits = devices.len();
    let num_items = dataloader.num_items();
    let step = num_items / num_splits;

    for (i, device) in devices.iter().enumerate() {
        let end = if i == (num_splits - 1) { num_items } else { start + step };
        let dataloader = dataloader.slice(start, end).to_device(device);
        dataloaders.push(dataloader);
        start = end;
    }
}
```

Each call to `dataloader.slice(start, end)` creates a new `BatchDataLoader` backed by a
`PartialDataset` — an offset+length view into the original dataset.

### How Each Rank Gets Its Data

Given a dataset of 10,000 items and 4 devices:

```
Device 0 (PeerId=0): items [0..2500)
Device 1 (PeerId=1): items [2500..5000)
Device 2 (PeerId=2): items [5000..7500)
Device 3 (PeerId=3): items [7500..10000)
```

The last device absorbs any remainder from integer division.

### Shuffling

Shuffling happens **per-epoch, within each slice**, not globally:

- Each `BatchDataLoader` wraps its dataset in a `ShuffledDataset` when `iter()` is called
- The RNG is forked per split, so different ranks see different orderings
- However, each rank always sees the **same subset** of data — just in different orders per epoch

This is simpler than PyTorch's `DistributedSampler` (which interleaves indices across ranks, so
each rank sees different data each epoch drawn from the full dataset).

### Other Dataset Utilities (Not DDP-Specific)

| Type | Purpose |
|------|---------|
| `PartialDataset` | Offset+length view into a parent dataset |
| `ShuffledDataset` | Fisher-Yates shuffled view of a dataset |
| `SamplerDataset` | Random sampling with/without replacement |
| `SelectionDataset` | Selects specific indices from a dataset |

---

## Batching

Batching is a two-stage process, happening independently on each rank:

### Stage 1: `BatchStrategy` — Accumulates Items

```rust
pub trait BatchStrategy<I>: Send {
    fn add(&mut self, item: I);
    fn batch(&mut self, force: bool) -> Option<Vec<I>>;
    fn new_like(&self) -> Box<dyn BatchStrategy<I>>;
}
```

Currently only `FixBatchStrategy` exists — it accumulates items one-by-one until the batch size
is reached, then flushes. When `force=true` (end of dataset), it flushes remaining items
regardless of count.

### Stage 2: `Batcher` — Converts to Tensors

```rust
pub trait Batcher<B: Backend, I, O>: Send + Sync {
    fn batch(&self, items: Vec<I>) -> O;
}
```

This is user-implemented (analogous to PyTorch's `collate_fn`). It takes a `Vec<I>` and produces
tensors on the target device.

### Multi-Threaded Data Loading

`MultiThreadDataLoader` uses a thread pool to pre-fetch batches. It:

1. Shuffles the full (sliced) dataset
2. Splits shuffled indices across worker threads
3. Each worker thread produces batches from its portion
4. Batches are collected in round-robin order

---

## Gradient Synchronization

### Training Step Flow

In `DdpTrainEpoch::run()`:

```rust
while let Some(item) = iterator.next() {
    // Forward + backward pass (local)
    let output = learner.train_step(item);

    // Gradient accumulation (optional)
    accumulator.accumulate(&learner.model(), output.grads);

    if accumulation_complete {
        let grads = accumulator.grads();

        // ALL-REDUCE gradients across all peers
        let grads = grads_syncer.sync(grads);

        // Optimizer step with averaged gradients
        learner.optimizer_step(grads);
    }
}
```

### `GradsSyncer` — Background Gradient Synchronization

The `GradsSyncer` runs a dedicated background thread for the all-reduce:

```rust
fn run_worker(double_buffering, peer_id, send, recv) {
    while let Ok(new_grads) = recv.recv() {
        let new_grads = new_grads
            .all_reduce::<B::InnerBackend>(peer_id, ReduceOperation::Mean)
            .expect("DDP worker could not sync gradients!");

        if double_buffering {
            // Pipeline: return previous iteration's grads,
            // buffer current for next iteration
            let old_grads = grads_buffer.take();
            grads_buffer = Some(new_grads);
            send.send(old_grads).unwrap();
        } else {
            send.send(Some(new_grads)).unwrap();
        }
    }
}
```

- **`double_buffering = false`** (current default): Synchronous — blocks until all-reduce completes
- **`double_buffering = true`**: Overlaps gradient sync with next forward pass — optimizer uses
  *previous* iteration's all-reduced gradients while current ones are being synchronized

### `GradientsParams::all_reduce()` — Per-Parameter Sync

```rust
pub fn all_reduce<B: Backend>(
    mut self,
    peer_id: PeerId,
    op: ReduceOperation,
) -> Result<Self, CollectiveError> {
    let mut ids = self.container.ids().into_iter().copied().collect::<Vec<ParamId>>();
    ids.sort(); // CRUCIAL: all peers must all-reduce in the same order!

    for id in ids {
        let grad = self.container.remove::<B>(&id).unwrap();
        let grad = burn_collective::all_reduce::<B>(peer_id, grad, op)?;
        self.container.register::<B>(id, grad);
    }
    Ok(self)
}
```

**Critical detail**: Parameter IDs are **sorted** to ensure all peers perform the all-reduce
operations in the **same deterministic order** — required for correctness since the collective
server pairs up tensors by submission order.

### LR Scheduler Alignment

Each worker advances the LR scheduler by `peer_count` steps per batch (since all peers process
batches in parallel):

```rust
for _ in 0..peer_count {
    iteration += 1;
    learner.lr_step();
}
```

---

## Collective Communication Layer

### Crate: `burn-collective`

#### Architecture

```
burn-collective/
├── api.rs              # Public API: register, all_reduce, broadcast, reduce, finish
├── config.rs           # CollectiveConfig, strategies, PeerId
├── local/
│   ├── server.rs       # Per-backend singleton server (coordinates all local peers)
│   ├── client.rs       # Thread-local client (sends messages to server)
│   ├── all_reduce/     # Centralized, Tree, Ring implementations
│   ├── reduce/         # Centralized, Tree implementations
│   └── broadcast/      # Centralized, Tree implementations
└── global/
    ├── orchestrator/   # WebSocket server for multi-node coordination
    └── node/           # Multi-node client, barrier sync, P2P tensor transfer
```

#### Public API

```rust
pub fn register<B: Backend>(id: PeerId, device: B::Device, config: CollectiveConfig) -> Result<(), CollectiveError>;
pub fn all_reduce<B: Backend>(id: PeerId, tensor: B::FloatTensorPrimitive, op: ReduceOperation) -> Result<B::FloatTensorPrimitive, CollectiveError>;
pub fn broadcast<B: Backend>(id: PeerId, tensor: Option<B::FloatTensorPrimitive>) -> Result<B::FloatTensorPrimitive, CollectiveError>;
pub fn reduce<B: Backend>(id: PeerId, tensor: B::FloatTensorPrimitive, op: ReduceOperation, root: PeerId) -> Result<Option<B::FloatTensorPrimitive>, CollectiveError>;
pub fn finish_collective<B: Backend>(id: PeerId) -> Result<(), CollectiveError>;
```

#### `CollectiveConfig`

```rust
pub struct CollectiveConfig {
    pub num_devices: usize,
    pub local_all_reduce_strategy: AllReduceStrategy,
    pub local_reduce_strategy: ReduceStrategy,
    pub local_broadcast_strategy: BroadcastStrategy,

    // Multi-node parameters (all optional, but all-or-nothing)
    pub num_nodes: Option<u32>,
    pub global_address: Option<Address>,
    pub node_address: Option<Address>,
    pub data_service_port: Option<u16>,
    pub global_all_reduce_strategy: Option<AllReduceStrategy>,
    pub global_reduce_strategy: Option<ReduceStrategy>,
    pub global_broadcast_strategy: Option<BroadcastStrategy>,
}
```

#### All-Reduce Strategies

```rust
pub enum AllReduceStrategy {
    Centralized,   // One device reduces all, then broadcasts
    Tree(u32),     // B-tree structured reduce + broadcast (arity parameter)
    Ring,          // Ring all-reduce (scatter-reduce + all-gather)
}
```

Defaults: `Tree(2)` locally, `Ring` globally.

#### `LocalCollectiveServer` — Intra-Node Coordination

A per-backend singleton that runs as a background tokio task. Each device thread gets a
`LocalCollectiveClient` that communicates with the server via channels:

```
Thread 0 ──► LocalCollectiveClient ──┐
Thread 1 ──► LocalCollectiveClient ──┤
Thread 2 ──► LocalCollectiveClient ──┼──► LocalCollectiveServer
Thread 3 ──► LocalCollectiveClient ──┘      (background tokio task)
```

The server:
1. **Registration**: Waits until `num_devices` peers register
2. **Collective op**: Collects inputs from all peers, executes the chosen algorithm
3. **Result**: Sends results back through callback channels

```rust
enum Message<B: Backend> {
    Register { device_id, device, config, callback },
    AllReduce { device_id, tensor, op, callback },
    Reduce { device_id, tensor, op, root, callback },
    Broadcast { device_id, tensor, callback },
    Reset,
    Finish { id, callback },
}
```

#### All-Reduce Algorithms (Local)

**Centralized:**
```
GPU0 ──tensor──►│
GPU1 ──tensor──►│ Central   ──sum──► Central ──broadcast──► GPU0, GPU1, GPU2, GPU3
GPU2 ──tensor──►│ (GPU0)              (GPU0)
GPU3 ──tensor──►│
```

**Tree(2):**
```
       ┌──GPU0──┐
   ┌───┤        ├───┐   reduce up     broadcast down
   GPU1         GPU2
   │                │
   GPU3
```
Recursively groups tensors, sums at parent nodes, then redistributes down the tree.

**Ring:**
Classic two-phase ring all-reduce:
1. **Scatter-Reduce**: Tensor sliced into N parts, slices circulate around the ring accumulating
   partial sums. After N-1 steps, each peer has one fully-reduced slice.
2. **All-Gather**: Reduced slices circulate again. After N-1 more steps, every peer has the
   complete result.

Falls back to Tree if the tensor is too small to slice effectively.

---

## Multi-Node Training

### Two-Level Hierarchy

For multi-node training, each all-reduce happens in three phases:

```
Node A (4 GPUs)                    Node B (4 GPUs)
┌─────────────────┐                ┌─────────────────┐
│ GPU0 GPU1       │                │ GPU0 GPU1       │
│ GPU2 GPU3       │                │ GPU2 GPU3       │
│                 │                │                 │
│ ① Local Reduce  │                │ ① Local Reduce  │
│ (4 tensors → 1) │                │ (4 tensors → 1) │
└────────┬────────┘                └────────┬────────┘
         │                                  │
         │     ② Global All-Reduce          │
         │     (WebSocket + P2P)            │
         └──────────────┬───────────────────┘
                        │
         ┌──────────────┴───────────────────┐
         │                                  │
┌────────┴────────┐                ┌────────┴────────┐
│ ③ Local         │                │ ③ Local         │
│    Broadcast    │                │    Broadcast    │
│ (1 → 4 GPUs)   │                │ (1 → 4 GPUs)   │
└─────────────────┘                └─────────────────┘
```

1. **Local reduce** — aggregate all local device gradients to one tensor on the node
2. **Global all-reduce** — exchange across nodes via WebSocket + P2P tensor data service
3. **Local broadcast** — distribute the result back to all local devices

### Global Orchestrator

A standalone WebSocket server (`start_global_orchestrator(port)`) that:
- Accepts connections from nodes on `/request` and `/response` routes
- Manages sessions, registration, and teardown
- Assigns each node a `NodeId` and shares the topology map (addresses of all nodes)

**The orchestrator is only for coordination — actual tensor data flows P2P between nodes.**

### Global All-Reduce Algorithms

The same three algorithms (Centralized, Tree, Ring) are available at the global level, but
transfers happen over the network via `TensorDataService`:

- **Ring**: Each node exposes a slice, downloads from its predecessor, accumulates/replaces
- **Tree**: Leaf nodes upload to parents, parents reduce and propagate up, root broadcasts down
- **Centralized**: One node downloads from all others, sums, and exposes the result

### Barrier Synchronization (`SyncService`)

After each global collective operation, a barrier sync ensures all nodes have completed:

```rust
pub struct SyncService<P: Protocol> {
    node_state: Arc<RwLock<Option<NodeState>>>,
    syncing_peers: Mutex<Vec<NodeId>>,
    sync_notif: Notify,
}
```

Each node sends a `SyncRequest` to all other nodes and waits until all have reported in.

---

## Communication Protocol

### Crate: `burn-communication`

Provides protocol-agnostic transport abstractions:

```rust
pub trait Protocol: Clone + Send + Sync + 'static {
    type Client: ProtocolClient;
    type Server: ProtocolServer;
}

pub trait CommunicationChannel: Send + 'static {
    fn send(&mut self, message: Message) -> impl Future<Output = Result<(), Error>>;
    fn recv(&mut self) -> impl Future<Output = Result<Option<Message>, Error>>;
    fn close(&mut self) -> impl Future<Output = Result<(), Error>>;
}
```

### Current Implementation: WebSocket Only

| Component | Technology |
|-----------|-----------|
| Client | `tokio-tungstenite` |
| Server | `axum` with WebSocket support |
| Serialization | MessagePack (`rmp-serde`) |
| Frame size | Up to 512 MB per frame, no message size limit |

No MPI, NCCL, or Gloo backends exist yet — the `Protocol` trait is designed for extensibility.

### `TensorDataService` — P2P Tensor Transfer

Enables direct server-to-server tensor transfer without routing through the orchestrator:

```rust
// Sender side:
data_service.expose(tensor, max_downloads, transfer_id).await;

// Receiver side:
let data = data_service.download_tensor(remote_address, transfer_id).await;
```

- Tensors are serialized to `TensorData` via MessagePack and stored in memory
- `max_downloads` tracks how many nodes will download (auto-removes after all downloads complete)
- `TensorTransferId(u64)` coordinates which tensor is being requested

---

## Complete Training Flow

Here is the end-to-end DDP training flow:

```
1. User configures SupervisedTraining with TrainingStrategy::DistributedDataParallel
   ├── devices: [GPU0, GPU1, GPU2, GPU3]
   └── config: CollectiveConfig { num_devices: 4, local_strategy: Tree(2), ... }

2. DdpTrainingStrategy::fit()
   ├── split_dataloader(train_dl, devices) → [dl_0, dl_1, dl_2, dl_3]
   ├── move valid_dl to main device (GPU0)
   └── spawn N DdpWorker threads

3. Each DdpWorker::fit()
   ├── burn_collective::register(peer_id, device, config)
   ├── learner.fork(device)  // move model to assigned device
   │
   ├── for epoch in 1..num_epochs:
   │   ├── DdpTrainEpoch::run()
   │   │   ├── iterator = my_dataloader.iter()  // shuffled each epoch
   │   │   └── for batch in iterator:
   │   │       ├── lr_scheduler.step() × peer_count  // align LR across ranks
   │   │       ├── grads = learner.train_step(batch)  // forward + backward
   │   │       ├── [optional: accumulate gradients for K steps]
   │   │       ├── grads = GradsSyncer.sync(grads)
   │   │       │   └── GradientsParams::all_reduce(peer_id, Mean)
   │   │       │       └── for param_id in sorted(param_ids):
   │   │       │           └── burn_collective::all_reduce(peer_id, grad_tensor, Mean)
   │   │       │               └── LocalCollectiveServer waits for all peers,
   │   │       │                   executes Tree/Ring/Centralized algorithm
   │   │       └── learner.optimizer_step(averaged_grads)
   │   │
   │   ├── [main worker only] validation epoch
   │   └── [main worker only] checkpoint
   │
   └── burn_collective::finish_collective(peer_id)

4. All threads join → return main worker's trained model
```

---

## Comparison with PyTorch DDP

| Aspect | Burn DDP | PyTorch DDP |
|--------|----------|-------------|
| **Process model** | One thread per device (single process per node) | One process per rank |
| **Rank concept** | `PeerId(u32)` | Integer rank from `torch.distributed` |
| **Launch mechanism** | Direct thread spawning | `torchrun` / `torch.distributed.launch` |
| **Data partitioning** | Contiguous slicing (`split_dataloader`) | `DistributedSampler` (interleaved indices) |
| **Shuffling** | Per-epoch within each fixed slice | Per-epoch across full dataset, then partitioned |
| **Gradient sync trigger** | Explicit `all_reduce()` call after backward | Automatic hooks on backward pass |
| **Gradient bucketing** | None — one all-reduce per parameter | Bucketed — groups small parameters for efficiency |
| **Communication** | WebSocket + MessagePack | NCCL / Gloo / MPI |
| **All-reduce algorithms** | Centralized, Tree, Ring (user-configurable) | NCCL handles internally (ring, tree, etc.) |
| **Multi-node coordination** | Global Orchestrator (WebSocket server) | `torchrun` + `c10d` process group |
| **Gradient accumulation** | Built into `DdpTrainEpoch` | Manual (user wraps with `no_sync()` context) |
| **Double buffering** | Optional (overlap sync with next forward) | Default behavior via async all-reduce |
| **Backend support** | Generic over `B: Backend` (NdArray, WGPU, CUDA, etc.) | CUDA-focused (CPU via Gloo) |

### Key Differences in Data Sampling

**PyTorch `DistributedSampler`:**
- Generates indices `[0, 1, 2, ..., N-1]`, shuffles globally, then each rank takes every
  `world_size`-th index (interleaved)
- Each epoch, every rank can see different data points from the full dataset
- Pads the dataset to be evenly divisible by `world_size`

**Burn `split_dataloader()`:**
- Divides the dataset into N contiguous slices: `[0..N/k), [N/k..2N/k), ...`
- Each rank always sees the **same subset** of data — only the ordering within that subset
  changes per epoch
- Simpler implementation, but less data diversity per rank across epochs

### What's Not Yet Implemented

As of this code snapshot:
- Global `reduce()` and `broadcast()` are `unimplemented!()` at the multi-node level
- No NCCL, MPI, or Gloo communication backends (only WebSocket)
- No gradient bucketing optimization
- Double buffering is disabled by default
- Quantized gradient all-reduce is `unimplemented!()`
