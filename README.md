# grassfinder

Find Minecraft grass coordinates from reconstructed plant offsets. GPU-accelerated on Apple Silicon, NVIDIA and other supported GPUs through [wgpu](https://github.com/gfx-rs/wgpu), with parallel CPU support if no GPU is found.

Based on [polymetric's grassfinder](https://github.com/polymetric/grassfinder).

## Install and run

**Windows / macOS:** Download the matching archive from [Releases](https://github.com/humhue/grassfinder/releases), extract it, open a terminal in the extracted `grassfinder` directory and run:

```sh
# macOS
./grassfinder
```

```powershell
# Windows (PowerShell)
.\grassfinder.exe
```

**From source (any supported platform):**

```sh
cargo run --release
```

Edit `offsets.txt` with your plant positions and offsets (six integers per line: `x y z offset_x offset_y offset_z`). The first plant defines the observed origin. The included offsets are example data: replace them with your own.

Defaults: GPU if available, sorted plants, early pruning, compact GPU output, average delta below **5**, X/Z **[-10,000, 10,000)** and Y **[62, 82)**. Set only what you need:

```sh
cargo run --release -- --range 20000 --delta 8
cargo run --release -- --backend cpu
cargo run --release -- --post-1.8  # Minecraft 1.8+ uses a Y-independent hash
cargo run --release -- --help
```

For a downloaded binary, omit `cargo run --release --` and place the options after `./grassfinder` or `grassfinder.exe`. An unsigned macOS release may require approval in **System Settings → Privacy & Security** on first launch.

## Performance

Measured on an **Apple M2 Pro**, searching **26 billion candidate positions**
(20,000 × 20,000 × 65), using **11 grass plants** with average delta **5**:

| Implementation | Runtime | Throughput |
|---|---:|---:|
| Original-style single-thread CPU (11 plants per position) | ~492 s (projected) | 52.9 million positions/s |
| Optimized CPU, grouped pruning (12 Rayon workers) | **24.859 s** | **1.046 billion positions/s** |
| GPU, Metal, pruning + compact output | **9.427 s** | **2.758 billion positions/s** |

The optimized CPU is approximately **19.8× faster** than the original-style
single-thread baseline; the GPU is approximately **52.1× faster**. The GPU is
approximately **2.64× faster** than the optimized CPU on this workload.

Even with **one CPU worker**, grouped pruning reaches approximately **139 million
positions/s**, about a **2.6×** speed up over the original's 52.9 million positions/s.

Results can vary depending on your hardware and input.

## Why it is fast

- **Rayon CPU:** automatic worker count, column-based hash reuse across Y levels, and grouped pruning (first four plants checked together).
- **Early rejection:** stop checking a candidate after its accumulated error reaches `average_delta × plant_count`.
- **Plant ordering:** test offsets with the largest expected mismatch against random positions first.
- **32-bit hash:** uses wrapping 32-bit arithmetic while preserving the relevant offset bits.
- **wgpu GPU compute:** runs on Metal (macOS), Vulkan and DirectX backends.
- **Compact readback:** return matching candidates only rather than one score per world position; batches keep memory bounded.

See [doc.md](doc.md) for the algorithm, input format, implementation details and alternative benchmarking modes. The original code and derivative project are distributed under the [MIT license](LICENSE.txt).
