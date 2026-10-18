# grassfinder

Find Minecraft grass coordinates from reconstructed plant offsets. Uses [wgpu](https://github.com/gfx-rs/wgpu) on supported GPUs (Metal, Vulkan, DirectX) and [Rayon](https://github.com/rayon-rs/rayon) on CPUs. Based on [polymetric's original grassfinder](https://github.com/polymetric/grassfinder).

## Install and run

**Windows / macOS:** Download the appropriate archive from [Releases](https://github.com/humhue/grassfinder/releases), extract it, and run `./grassfinder.exe` (Windows) or `./grassfinder` (macOS). macOS may require approving an unsigned executable in Privacy & Security.

**From source:**

```sh
cargo run --release
```

Edit `offsets.txt` first: each non-comment line contains `x y z offset_x offset_y offset_z`. The first plant defines the observed origin. The included offsets are example data: replace them with your own.

Defaults: GPU when available (otherwise CPU), GPU LUT + compact output, CPU 128×128 tiled cache + error LUT, automatic Rayon worker count, average delta below 5, X/Z `[-10000, 10000)` and Y `[62, 82)`. Set only what you need:

```sh
cargo run --release -- --backend cpu
cargo run --release -- --y-max 127 --no-matches
cargo run --release -- --range 20000 --delta 8
cargo run --release -- --post-1.8    # Minecraft 1.8+ (Y-independent hash)
cargo run --release -- --help
```

For a downloaded binary, omit `cargo run --release --` and place the options after `./grassfinder` or `grassfinder.exe`.

## Performance

Measured on an **Apple M2 Pro** with **11 plants**, average delta threshold **5**, searching **26 billion positions** (20,000 × 20,000 × 65):

| Implementation | Runtime | Positions/second |
|---|---:|---:|
| Optimized CPU, tiled + LUT, 128 tile | **14.742 s** | **1.764 B** |
| Optimized GPU, LUT + compact | **8.781 s** | **2.961 B** |
| Polymetric's single-thread CPU | ~492 s | ~52.9 M |
| Optimized single-thread CPU | ~125 s | ~208 M |

Approximately **33× faster on CPU** and **56× faster on GPU** versus Polymetric's original implementation. Results depend on hardware, plant layout, and search options. (Hybrid CPU/GPU execution was also tested but did not improve runtime on my machine.)


Even with **one CPU worker**, spatial tiling and the error lookup table (all CPU defaults), the program reaches approximately **208 million positions/s**, about a **4×** speed up over the original's 52.9 million positions/s.

## Why it is fast

- **CPU:** multithreaded column search; early rejection; grass ordering; 32-bit hash arithmetic; overlapping spatial hash tiles; compact 12-bit offsets; 4,096-entry per-plant error lookup tables.
- **GPU:** parallel compute through wgpu; early rejection; packed error lookup table; compact results readback with overflow fallback.
- **No giant world cache:** work is evaluated in bounded batches; CPU tiles are cached per worker. GPU spatial tiling was tested and discarded because it was slower.

See [doc.md](doc.md) for implementation details, correctness checks, and benchmark alternatives. Licensed under [MIT](LICENSE.txt).
