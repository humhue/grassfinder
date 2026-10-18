# How grassfinder works

## Input and matching

Grassfinder searches for the world coordinates corresponding to plant offsets reconstructed from Minecraft screenshots. It is based on [polymetric's original program](https://github.com/polymetric/grassfinder).

Each line of `offsets.txt` specifies six integers:

```text
observed_x observed_y observed_z offset_x offset_y offset_z
```

Offsets range from 0 to 15. The first plant defines the observed coordinate origin; subsequent plant coordinates and search results are interpreted relative to it. Blank lines and `#` comments are supported.

For screenshots from Minecraft 1.8 onward use `--post-1.8` (old alias `--post-1.12`). In that mode the relevant grass hash is independent of world Y, so it cannot recover Y.

For each candidate world position and plant, Grassfinder predicts the three 4-bit offsets and sums absolute differences. A match satisfies:

```text
sum of absolute differences < average_delta * plant_count
```

The inequality is strict. The default average delta is 5; increasing it admits more inaccurate observations and false matches.

## Grass hash: 32-bit equivalence

The required hash bits are 16–27. Wrapping 32-bit arithmetic preserves these bits even though the original used 64-bit values:

```rust
let seed = (x as u32).wrapping_mul(3129871)
    ^ (z as u32).wrapping_mul(116129781)
    ^ (y as u32);
let hash = seed.wrapping_mul(seed).wrapping_mul(42317861)
    .wrapping_add(seed.wrapping_mul(11));
let offsets = [(hash >> 16) & 15, (hash >> 20) & 15, (hash >> 24) & 15];
```

The test suite compares this result with the original wrapping 64-bit calculation.

## CPU default: tiled + LUT

Rayon automatically uses available logical CPU threads unless `RAYON_NUM_THREADS` is set.

The CPU searches columns rather than decoding flattened indices for every candidate, and reuses horizontal X/Z portions of each plant's hash across Y levels. Plants are ordered by expected error against random offsets to improve early rejection.

The default `--cpu-loop tiled` constructs overlapping spatial tiles of grass hashes, including a border for the plants' relative positions. Each 12-bit grass offset is stored in a `u16`, so the cache stays relatively small. Tiles are independent and cached per worker; neighboring tile borders may be recomputed. Default `--tile-size 128` used approximately 2.33 MiB per worker with the supplied offsets and 65-level Y range. Storage depends on the plant spread and search height.

The default `--cpu-error lut` uses a 4,096-entry `u8` table for each plant. Each lookup replaces three absolute differences with one indexed error byte. For 11 plants, the full set occupies 44 KiB. The tiled evaluator combines this table with the spatial hash cache and early rejection; it never changes the matching criterion.

Other CPU modes are retained for comparisons:

```sh
cargo run --release -- --backend cpu --cpu-loop grouped --cpu-error direct --no-matches
cargo run --release -- --backend cpu --cpu-loop grouped --cpu-error lut --no-matches
cargo run --release -- --backend cpu --cpu-loop tiled --cpu-error lut --tile-size 64 --no-matches
cargo run --release -- --backend cpu --cpu-loop full --no-matches
cargo run --release -- --backend cpu --cpu-loop flat --no-matches
```

`grouped` evaluates four plants before checking the cutoff, then prunes remaining plants individually. `column` prunes after every plant, `full` evaluates all plants, and `flat` is the earlier per-position CPU baseline. Worst-case asymptotic work remains O(positions × plants).

## GPU default: LUT + compact output

The default `--gpu-loop lut` runs one independent GPU work item per candidate, using the packed per-plant error tables and early rejection. It is the fastest measured GPU method on the test M2 Pro. Other available modes are `prune`, `full`, `grouped`, and `two-stage`.

With `--gpu-output compact` (the default), only matching candidates are written into a bounded result buffer and read back by the CPU. If that buffer overflows, the affected batch is rerun in full-output mode to avoid losing matches. This is essential for high throughput when most candidate positions fail the test.

GPU spatial tiling and simultaneous hybrid CPU/GPU scheduling were explored but excluded from this release because neither outperformed GPU LUT alone in the tested environment. Discrete NVIDIA GPUs may have different performance characteristics.

## Benchmark and validation

Reported measurements on **Apple M2 Pro**, using 11 plants, threshold 5, X/Z `[-10000,10000)`, Y `[62,127)` (26 billion positions):

| Implementation | Runtime | Positions/second |
|---|---:|---:|
| Rayon CPU, grouped + direct | 25.312 s | 1.027 B |
| Rayon CPU, tiled + LUT, 128 tile | **14.742 s** | **1.764 B** |
| Metal GPU, prune + compact | 9.583 s | 2.713 B |
| Metal GPU, LUT + compact | **8.781 s** | **2.961 B** |
| Polymetric's single-thread CPU | ~492 s | ~52.9 M |
| Single-thread CPU | ~125 s | ~208 M |

Correctness:

```sh
cargo test --release
cargo run --release -- --backend gpu --gpu-loop lut --gpu-output compact \
    --range 8 --y-min 62 --y-max 65 --file test_known_offsets.txt --delta 2 --verify
```

For the long full search, use:

```sh
cargo run --release -- --backend cpu --y-max 127 --no-matches
cargo run --release -- --backend gpu --y-max 127 --no-matches
```

## History and credits

This project builds on polymetric's grass-coordinate search and humhue's parallelization and pruning work, with later optimization and benchmarking developed collaboratively with ChatGPT. It retains the original MIT license and attribution.
