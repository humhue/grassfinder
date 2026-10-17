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

The original sequential search evaluated every observed plant at every candidate position. This implementation divides candidate world positions across Rayon workers.

Within each candidate, accumulated absolute errors can only increase. Therefore, when the running error reaches the maximum permitted *total*, the candidate can be rejected immediately without evaluating the remaining plants. Worst-case work is still `O(positions × plants)`, but average work is often much smaller.

Plants are sorted *once*, before search, by their expected absolute deviation against a uniformly distributed random four-bit offset. Offsets near 0 or 15 usually produce more error against a random candidate than offsets near 7 or 8, so checking them first often helps early rejection. Ordering does not change accepted matches.

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

## Benchmark reproducibly

Keep identical bounds, offsets, threshold, hardware and binary build. For a short correctness test:

```sh
cargo test --release
cargo run --release -- --backend gpu --range 8 --y-min 62 --y-max 65 --file test_known_offsets.txt --verify
```

On one Apple M2 Pro with eleven plants and a delta threshold of five, the default eight-billion-position run took about 16.5 seconds on Rayon CPU and 2.9–3.0 seconds on Metal GPU (approximately 2.7 billion candidate positions/s). These measurements are user-supplied, not independent benchmarks on every supported device.

A historical note in the prior documentation describes an approximately nine-minute original-program search with a *different* 26-billion-position, five-plant workload. Consequently, interpreting the nine-minute and 2.9-second figures as a controlled speedup requires an identical original-program rerun. On nominal wall-clock figures alone, 540/2.9 is roughly 186×.

## History, license and credits

This builds on polymetric's original grass-coordinate search and manual reconstruction approach, humhue's parallelization and pruning work, and subsequent implementation/benchmarking work developed collaboratively with ChatGPT. The project retains the MIT license and original copyright notice.
