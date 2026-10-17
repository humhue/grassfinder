# How grassfinder works

## The input

A Minecraft grass plant's displacement is determined by a coordinate hash. From a screenshot, reconstruct the relative block coordinates of several plants and their visual offsets, then search for the world position whose calculated offsets fit best. See [polymetric's original program](https://github.com/polymetric/grassfinder) for the foundation of this technique.

Each nonempty line in `offsets.txt` contains six integers:

```text
observed_x observed_y observed_z offset_x offset_y offset_z
```

Offsets are in `0..15`. The first position is an *observed origin*. Other positions are treated as relative to it; search results give the corresponding world coordinate and translation. Lines beginning with `#` (or trailing comments) are allowed.

For screenshots from **Minecraft 1.8 onward**, use `--post-1.8`: the grass hash does not depend on world Y, so only X and Z can be recovered by this technique. In older versions, the three-dimensional hash includes Y. The older `--post-1.12` spelling remains accepted as an alias.

## Hash and error metric

The coordinate-mixing function is equivalent, for grass offsets, to:

```rust
let seed = (x as u32).wrapping_mul(3129871)
    ^ (z as u32).wrapping_mul(116129781)
    ^ (y as u32);
let hash = seed.wrapping_mul(seed).wrapping_mul(42317861)
    .wrapping_add(seed.wrapping_mul(11));
let offsets = [(hash >> 16) & 15, (hash >> 20) & 15, (hash >> 24) & 15];
```

These are the same bits used by the original 64-bit arithmetic: only bits 16–27 matter, so wrapping 32-bit operations produce identical relevant results. The implementation evaluates the sum of absolute differences across all three offset components for each observed plant.

A candidate matches when:

```text
sum of all component differences < maximum average delta × number of plants
```

An average delta of 5 is useful for imperfectly reconstructed screenshots but is not guaranteed to work for every input. A lower threshold rejects more candidates; a higher threshold allows more false positives. Additional correctly reconstructed plants usually improve discrimination.

## CPU optimizations

The CPU backend uses [Rayon](https://github.com/rayon-rs/rayon) to split the search across available logical CPU threads. By default it selects the `grouped` strategy automatically; `RAYON_NUM_THREADS` can override the worker count for benchmarking.

**Column-based search.** Instead of converting a flattened index back into X, Y and Z for every candidate, the CPU iterates over X/Z columns and evaluates their Y levels in sequence. For each column and grass plant, it calculates the X/Z-dependent portion of the hash seed once and reuses it at every Y level. This avoids repeated 64-bit quotient/remainder operations and two coordinate multiplications per plant per Y level. Work is still divided into batches and columns across Rayon workers.

**Grouped pruning (CPU default).** The first four plants are evaluated together without a data-dependent early exit, allowing the CPU more freedom to schedule their independent arithmetic. If their accumulated error already reaches the cutoff, the candidate is rejected. Otherwise, the remaining plants are evaluated individually, with early rejection after each one. This keeps the same match criterion as a full evaluation. A plain `column` mode checks the cutoff after *every* plant; for this workload, grouped pruning measured substantially faster.

**Plant ordering.** Before searching, plants are sorted once by expected absolute error against a uniformly distributed four-bit offset. Observed offsets near 0 or 15 typically differ more from random candidate offsets than values near 7 or 8, so testing them first often improves pruning. Sorting does not change which candidates match.

**Available CPU strategies:**

```sh
cargo run --release -- --backend cpu --cpu-loop grouped --no-matches
cargo run --release -- --backend cpu --cpu-loop column  --no-matches
cargo run --release -- --backend cpu --cpu-loop full    --no-matches
cargo run --release -- --backend cpu --cpu-loop flat    --no-matches
```

All these modes preserve the same error calculation and accepted candidates. They differ only in traversal and how much computation is skipped. Worst-case work remains `O(positions × plants)`.

## GPU execution

The GPU kernel maps batches of positions into 32-bit coordinates and computes their hashes independently. On macOS, wgpu uses Metal; supported Windows/Linux adapters use DirectX or Vulkan. The default `prune` mode breaks out early for rejected positions.

The default `compact` output uses an atomic counter to write *only matches* into a bounded per-batch buffer. This avoids transferring four bytes for every searched position (32 GB over an eight-billion-position search). If too many matches fill a batch's compact buffer, the program reruns that batch with full-output readback so no results are dropped. The reported throughput includes this work.

Alternative modes are retained for investigation:

```sh
cargo run --release -- --gpu-loop full      --gpu-output compact --no-matches
cargo run --release -- --gpu-loop grouped   --group-size 4 --no-matches
cargo run --release -- --gpu-loop two-stage --stage1-plants 4 --no-matches
cargo run --release -- --backend cpu --no-matches
./benchmark.sh --range 1000 --y-min 62 --y-max 66 --delta 5
```

The `two-stage` mode filters candidates with the first few plants, compacts intermediate survivors and processes survivors in a second dispatch. It can lose to direct pruning because of compaction and dispatch overhead.

## Performance and reproducibility

Measured on an **Apple M2 Pro** with **11 grass plants**, an **average delta threshold of 5**, and **26 billion candidate positions** (20,000 × 20,000 × 65; X/Z `[-10000, 10000)`, Y `[62, 127)`):

| Implementation | Time | Throughput |
|---|---:|---:|
| Original-style single-thread CPU | ~492 s | ~52.9 million positions/s |
| Optimized CPU: grouped pruning, automatic Rayon (12 workers) | **24.859 s** | **1.046 billion positions/s** |
| Metal GPU: pruning and compact output | **9.427 s** | **2.758 billion positions/s** |

Compared with the original-style single-thread baseline, the optimized CPU is approximately **19.8× faster** and the GPU **52.1× faster**. The GPU is **2.64× faster** than the optimized CPU for this workload.

In additional single-worker tests, the CPU `flat` mode processed about **56 million positions/s**, `column` about **81 million/s**, `full` about **102 million/s**, and `grouped` about **139 million/s**.

The original-style reference retains polymetric's full, unpruned per-position comparison and 64-bit grass hash, but replaces its spiral search with rectangular bounds for comparison. All figures are hardware- and input-dependent.

For correctness checks:

```sh
cargo test --release
cargo run --release -- --backend gpu --range 8 --y-min 62 --y-max 65 --file test_known_offsets.txt --verify
```

To reproduce the completed CPU/GPU comparisons on the sample offsets, with the long Y range explicitly enabled:

```sh
cargo run --release -- --backend cpu --cpu-loop grouped --range 10000 --y-min 62 --y-max 127 --delta 5 --no-matches
cargo run --release -- --backend gpu --gpu-loop prune --gpu-output compact --range 10000 --y-min 62 --y-max 127 --delta 5 --no-matches
```

Rayon uses the machine's available logical processors by default; for a controlled single-worker comparison, prefix a CPU command with `RAYON_NUM_THREADS=1`. GPU code and shader logic are independent of the CPU-specific column and grouped optimizations.

## History, license and credits

This builds on polymetric's original grass-coordinate search and manual reconstruction approach, humhue's parallelization and pruning work, and subsequent implementation/benchmarking work developed collaboratively with ChatGPT. The project retains the MIT license and original copyright notice.
