#!/usr/bin/env bash
set -euo pipefail
# Usage: ./benchmark.sh [additional search args]
# Defaults to a repeatable, nontrivial 16-million-position search:
# X/Z ±1000, Y 62..65. Pass --range 10000 --y-min 62 --y-max 82
# --delta 5 to benchmark your original 8-billion-position workload.
cargo build --release
base=(--backend gpu --range 1000 --y-min 62 --y-max 66 --delta 5 --no-matches)
if (($# > 0)); then
  base=(--backend gpu --no-matches "$@")
fi
for mode in full prune grouped two-stage; do
  printf '\n===== %s / full output =====\n' "$mode"
  target/release/grassfinder "${base[@]}" --gpu-loop "$mode"
done
for mode in full prune grouped two-stage; do
  printf '\n===== %s / compact output =====\n' "$mode"
  target/release/grassfinder "${base[@]}" --gpu-loop "$mode" --gpu-output compact
done
