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

Reported on an Apple M2 Pro, searching **8 billion candidate positions** against **11 grass plants**, at average delta **5**:

| Implementation | Time | Positions/sec |
|---|---:|---:|
| Rayon CPU | 16.5 s | 484 million |
| Metal GPU, pruning + compact output | **2.9–3.0 s** | **2.7 billion** |

The optimized GPU run is approximately **5.7× faster** than the optimized CPU on this workload. Compared with a *reported* nine-minute run of the original implementation, it is approximately **180–186× faster**; this is provided as a historical comparison (it wasn't run on the same machine). Your hardware, sample count and thresholds will affect results.

## Why it is fast

- **Rayon:** parallel CPU search across available cores.
- **Early rejection:** stop checking a candidate after its accumulated error reaches `average_delta × plant_count`.
- **Plant ordering:** test offsets with the largest expected mismatch against random positions first.
- **32-bit hash:** uses wrapping 32-bit arithmetic while preserving the relevant offset bits.
- **wgpu GPU compute:** runs on Metal (macOS), Vulkan and DirectX backends.
- **Compact readback:** return matching candidates only rather than one score per world position; batches keep memory bounded.

See [doc.md](doc.md) for the algorithm, input format, implementation details and alternative benchmarking modes. The original code and derivative project are distributed under the [MIT license](LICENSE.txt).
