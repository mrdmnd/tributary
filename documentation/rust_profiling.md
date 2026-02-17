# Best Practices for Profiling and Benchmarking CPU-Heavy Rust Code

## On Ubuntu Linux

---

## 1. Environment Setup and Prerequisites

### System Configuration

Before any profiling or benchmarking session, ensure the environment is stable and reproducible.

```bash
# Install essential tools
sudo apt update
sudo apt install -y linux-tools-common linux-tools-$(uname -r) valgrind build-essential pkg-config libssl-dev

# Install Rust toolchain (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable

# Install profiling/benchmarking cargo subcommands
cargo install flamegraph
cargo install cargo-criterion
cargo install critcmp
cargo install cargo-asm
cargo install hyperfine
cargo install samply
```

### Kernel and OS Tuning for Reproducible Results

CPU-heavy benchmarks are extremely sensitive to system noise. Apply these before benchmarking.

```bash
# Allow perf events without root (persists until reboot)
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

# Disable ASLR for consistent memory layouts
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space

# Set CPU governor to 'performance' to disable frequency scaling
sudo cpupower frequency-set -g performance

# Optionally, isolate CPUs for benchmark processes (e.g., cores 4-7)
# Add to kernel boot params: isolcpus=4-7 nohz_full=4-7
# Then pin your process:
taskset -c 4-7 ./target/release/my_benchmark
```

Additionally, close all unnecessary applications, disable background services like automatic updates, and avoid running on a laptop on battery power.

### Compilation Profiles

The `Cargo.toml` should define distinct profiles for benchmarking and profiling.

```toml
# Maximum optimization for benchmarks
[profile.bench]
opt-level = 3
lto = "fat"
codegen-units = 1

# Profiling: optimized but with debug symbols
[profile.profiling]
inherits = "release"
debug = true        # DWARF debug info for symbol resolution
strip = false       # Do NOT strip symbols

# Release profile (your normal release build)
[profile.release]
opt-level = 3
lto = "thin"
```

Always profile against the `profiling` profile (or `release` with `debug = true`). 
Never profile unoptimized debug builds — the results will be meaningless for production performance analysis.

---

## 2. Benchmarking: Measuring Performance

Benchmarking answers "how fast is this code?" Profiling answers "why is it this fast (or slow)?" 
Always benchmark first, then profile the areas that need improvement.

### 2.1 Criterion.rs (Micro-Benchmarks)

Criterion is the gold standard for Rust micro-benchmarks. It provides statistical rigor, regression detection, and HTML reports.

```bash
cargo add --dev criterion
```

Add to `Cargo.toml`:

```toml
[[bench]]
name = "my_bench"
harness = false
```

Create `benches/my_bench.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, BatchSize};

fn bench_sorting(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting");

    for size in [100, 1_000, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::new("my_sort", size),
            &size,
            |b, &size| {
                // Use iter_batched to separate setup from measurement
                b.iter_batched(
                    || generate_random_vec(size),          // setup (not timed)
                    |mut data| black_box(my_sort(&mut data)), // routine (timed)
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_sorting);
criterion_main!(benches);
```

Run:

```bash
cargo bench                           # Run all benchmarks
cargo bench -- sorting                # Filter by name
cargo bench -- --save-baseline before # Save a named baseline
# ... make changes ...
cargo bench -- --baseline before      # Compare against baseline
```

Key practices for Criterion:

- Always use `black_box()` on inputs and outputs to prevent the optimizer from eliminating your code.
- Use `iter_batched` or `iter_batched_ref` when setup is expensive (e.g., generating random data), so setup cost doesn't pollute measurements.
- Use `BenchmarkId::new` with parameterized inputs to generate comparison charts across input sizes.
- Examine the HTML report in `target/criterion/report/index.html` for distributions and regression analysis.
- Use `critcmp` for comparing JSON exports across git commits or branches.

### 2.2 Hyperfine (Binary-Level Benchmarks)

For benchmarking entire compiled binaries or comparing different implementations:

```bash
cargo build --release --bin solver_v1
cargo build --release --bin solver_v2

hyperfine \
    --warmup 5 \
    --min-runs 20 \
    --export-json results.json \
    './target/release/solver_v1 input.dat' \
    './target/release/solver_v2 input.dat'
```

Hyperfine automatically detects shell startup overhead, provides statistical summaries, and can export to JSON/Markdown for comparisons.

### 2.3 Custom Wall-Clock Timing

For coarse measurements in application code:

```rust
use std::time::Instant;

let start = Instant::now();
let result = expensive_computation();
let elapsed = start.elapsed();
println!("Computation took: {:?}", elapsed);
std::hint::black_box(&result); // prevent dead-code elimination
```

This is fine for quick sanity checks but lacks the statistical rigor of Criterion. Never rely on a single measurement.

---

## 3. Profiling: Finding Bottlenecks

### 3.1 Sampling Profilers

Sampling profilers periodically interrupt the program and record the call stack.
They have low overhead and are ideal for finding hot spots.

#### `perf` (Linux Kernel Profiler)

The most powerful and lowest-overhead profiler on Linux.

```bash
# Record profile data
perf record -g --call-graph dwarf \
    ./target/profiling/my_binary --my-args

# Or for a cargo bench run:
perf record -g --call-graph dwarf \
    cargo bench --profile profiling -- my_benchmark_name --profile-time 10

# View interactive TUI report
perf report

# Generate flamegraph from perf data
perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg
```

Key `perf` flags:

- `-g` enables call-graph recording.
- `--call-graph dwarf` uses DWARF debug info for accurate stack unwinding. This is essential for Rust; the default (frame-pointer) often produces broken stacks because Rust omits frame pointers by default.
- `-F 999` sets sampling frequency (default 4000 Hz; lower for less overhead).
- `perf stat ./binary` gives a quick overview of hardware counters (cycles, instructions, cache misses, branch mispredicts) without recording a full profile.

#### `samply` (Modern Profiler with Firefox Profiler UI)

Samply wraps `perf` and opens results in the Firefox Profiler web UI, which is excellent for interactive exploration.

```bash
samply record ./target/profiling/my_binary --my-args
# Opens browser automatically with interactive flame chart
```

This is often the fastest path from "I want to profile" to "I can see where time is spent."

#### Flamegraphs via `cargo flamegraph`

```bash
cargo flamegraph --profile profiling -- --my-args
# Produces flamegraph.svg in the current directory
```

Flamegraphs are the single most useful visualization for CPU-bound code. 
Read them bottom-up: the x-axis is *not* time but rather the set of sampled stacks sorted alphabetically; width represents the proportion of total samples.

### 3.2 Instrumentation Profilers

These measure exact call counts and durations but have higher overhead (often 10–100× slowdown).

#### Cachegrind / Callgrind (Valgrind)

```bash
# Instruction-level profiling (simulates CPU cache)
valgrind --tool=cachegrind ./target/profiling/my_binary
cg_annotate cachegrind.out.<pid>

# Call-graph profiling with exact call counts
valgrind --tool=callgrind ./target/profiling/my_binary
callgrind_annotate callgrind.out.<pid>

# Visualize with KCachegrind
kcachegrind callgrind.out.<pid>
```

Callgrind is invaluable for understanding exact call counts and instruction-level hotspots, especially when sampling profilers give ambiguous results.
The major downside is extreme slowdown (20–50×), so use it on smaller inputs.

### 3.3 Hardware Performance Counters

For deep CPU micro-architecture analysis:

```bash
# Quick overview of key metrics
perf stat -d ./target/release/my_binary

# Specific counters
perf stat -e cycles,instructions,cache-references,cache-misses,\
branches,branch-misses,L1-dcache-load-misses,LLC-load-misses \
    ./target/release/my_binary
```

Key metrics to watch:

- **IPC (Instructions Per Cycle)**: Modern CPUs can do 3–6 IPC. If you're below 1, you likely have memory stalls or branch mispredictions.
- **Cache miss rate**: High L1/LLC miss rates indicate poor data locality. Consider restructuring data layouts (e.g., SoA vs AoS).
- **Branch misprediction rate**: Above 5% suggests unpredictable branching patterns. Consider branchless algorithms or sorting data to improve predictability.

### 3.4 Memory / Allocation Profiling

For CPU-heavy code, allocation pressure is often a hidden bottleneck.

#### DHAT (Valgrind)

```bash
valgrind --tool=dhat ./target/profiling/my_binary
# Open dhat output in the DHAT viewer: https://nnethercote.github.io/dh_view/dh_view.html
```

DHAT shows every allocation site, how many bytes were allocated, how long allocations lived, and how many were "short-lived" (potential optimization targets).

#### Custom Global Allocator Tracking

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;
```

This lets you assert "this hot loop performs zero allocations" — a powerful invariant for CPU-bound code.

---

## 4. Analyzing Assembly Output

When micro-optimizing, inspect what the compiler actually generates.

```bash
# View assembly for a specific function
cargo asm --profile release my_crate::my_module::my_function

# Or use Compiler Explorer (godbolt.org) for quick iteration

# With perf, annotate source with assembly
perf annotate -M intel
```

Things to look for:

- **Auto-vectorization**: Look for SIMD instructions (e.g., `vpaddd`, `vmovdqu`, `vpshufb`). If you expected vectorization and see scalar code, consider restructuring data or adding `#[target_feature(enable = "avx2")]`.
- **Unnecessary bounds checks**: Rust inserts bounds checks on indexing. Look for call sites to `core::panicking::panic_bounds_check`. Use iterators, `get_unchecked` (with safety justification), or assert lengths before loops to help the compiler elide them.
- **Unexpected function calls**: In tight loops, calls to allocator functions (`__rust_alloc`), formatting machinery, or drop glue indicate avoidable overhead.

---

## 5. Common CPU-Bound Optimization Patterns

These are the most impactful optimizations, ordered roughly by effort-to-impact ratio.

### Algorithmic Improvements

Always start here. No amount of micro-optimization compensates for a suboptimal algorithm.
Profile first, then check: is this O(n²) when it could be O(n log n)?

### Reduce Allocations

Pre-allocate buffers, reuse `Vec`s across iterations, use `SmallVec` for small-but-variable-size collections, use `String`/`Vec::with_capacity()`.
In hot loops, allocations dominate — each one is a syscall or at minimum a lock-free CAS on the allocator.

### Data Layout and Cache Efficiency

Restructure data for sequential access.
Prefer Structure-of-Arrays (SoA) over Array-of-Structures (AoS) when iterating over one field at a time.
Use `#[repr(C)]` when layout matters. Consider `[u8; N]` or `ArrayVec` over heap-allocated containers for small fixed-size data.

### Parallelism with Rayon

For embarrassingly parallel workloads:

```rust
use rayon::prelude::*;

let results: Vec<_> = inputs
    .par_iter()
    .map(|input| expensive_computation(input))
    .collect();
```

Rayon's work-stealing scheduler handles load balancing.
Ensure each unit of work is substantial enough (microseconds, not nanoseconds) to amortize the scheduling overhead.

### SIMD and Auto-Vectorization

Help the compiler auto-vectorize by using iterators, avoiding early exits in loops, and ensuring data is aligned. For manual SIMD:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(data: &[f32]) -> f32 {
    // ... manual SIMD implementation
}
```

Verify vectorization by examining assembly output with `cargo asm`.

---

## 6. Continuous Benchmarking and Regression Detection

### In CI/CD

Use Criterion's baseline comparison in CI to catch performance regressions:

```bash
# On main branch, save baseline
cargo bench -- --save-baseline main

# On PR branch, compare
cargo bench -- --baseline main
```

### Tracking Over Time

- Store Criterion JSON output in a database or tracking service.
- Set regression thresholds (e.g., fail if >5% slower).
- Run benchmarks on dedicated hardware or at minimum on isolated CI runners to reduce noise.
- Always compare the same input sizes and configurations.

---

## 7. Checklist: Before You Profile or Benchmark

1. **Build with the right profile**: `--profile profiling` for profiling (optimized + debug symbols), `--release` or `--profile bench` for benchmarking.
2. **Set CPU governor to `performance`**.
3. **Close unnecessary applications** and background processes.
4. **Use `taskset` to pin to specific cores** if possible.
5. **Disable Turbo Boost** for reproducibility: `echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo` (Intel) or equivalent for AMD.
6. **Warm up**: Run a few iterations before measuring (Criterion and Hyperfine do this automatically).
7. **Use statistical methods**: Never report a single measurement. Report median, standard deviation, and confidence intervals.
8. **Profile with realistic inputs**: Synthetic micro-benchmarks can miss real-world performance characteristics (e.g., cache effects at scale, branch prediction patterns).
9. **Iterate**: Benchmark → Profile → Optimize → Benchmark again. Never optimize without measuring the impact.

---

## 8. Quick Reference: Tool Selection

| Goal | Tool |
|---|---|
| Micro-benchmark a function | Criterion.rs |
| Benchmark a whole binary | Hyperfine |
| Find hot functions (low overhead) | `perf record` + `perf report`, or `samply` |
| Visualize hot paths | Flamegraph (`cargo flamegraph` or `samply`) |
| Exact call counts and instruction costs | Callgrind (Valgrind) |
| Cache miss analysis | `perf stat -d` or Cachegrind |
| Allocation profiling | DHAT (Valgrind) or custom global allocator |
| Assembly inspection | `cargo asm` or Compiler Explorer |
| Hardware counter deep-dive | `perf stat -e <counters>` |
| Compare two binaries | Hyperfine with multiple commands |
| CI regression detection | Criterion baselines + `critcmp` |