#!/usr/bin/env bash
set -euo pipefail
cargo build --release
printf '\n=== CPU: tiled + LUT ===\n'
cargo run --release -- --backend cpu --y-max 127 --no-matches "$@"
printf '\n=== GPU: LUT + compact ===\n'
cargo run --release -- --backend gpu --y-max 127 --no-matches "$@"
