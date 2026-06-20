#!/usr/bin/env bash
# Fetch the privacy-filter weights for the MLX runtime from the published HF repo.
#
# Default pulls the bf16 checkpoint (~2.6 GB) — the app quantizes at runtime
# (PF_QBITS/PF_QEMBED). Pass `q4-8emb` to fetch the pre-quantized 870 MB variant
# instead (4-bit MoE + 8-bit embeddings, certified 99.4% labels).
#
# Run it yourself (heavy download — not auto-run):
#   bash reference/download.sh                 # -> models/privacy-filter (bf16)
#   bash reference/download.sh q4-8emb         # -> models/q4-8emb (870 MB)
#   bash reference/download.sh bf16 /path/dir  # custom dir
#
# Override the source repo with PF_HF_REPO. `hf` is the huggingface_hub CLI
# (`uvx --from huggingface_hub hf ...` if not installed); the old
# `huggingface-cli` was removed in huggingface_hub 1.x.
set -euo pipefail

REPO="${PF_HF_REPO:-beshkenadze/privacy-filter-mlx}"
VARIANT="${1:-bf16}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "$VARIANT" in
  bf16)
    DIR="${2:-$ROOT/models/privacy-filter}"
    echo "Downloading $REPO (bf16, ~2.6 GB) -> $DIR"
    hf download "$REPO" --local-dir "$DIR" --exclude "q4-8emb/*"
    ;;
  q4-8emb)
    DIR="${2:-$ROOT/models/q4-8emb}"
    echo "Downloading $REPO q4-8emb (~870 MB) -> $DIR"
    hf download "$REPO" --include "q4-8emb/*" --local-dir "$DIR"
    # flatten q4-8emb/* up to DIR root so config.json/model.safetensors/tokenizer.json
    # sit where the loader expects them
    if [ -d "$DIR/q4-8emb" ]; then mv "$DIR"/q4-8emb/* "$DIR"/ && rmdir "$DIR/q4-8emb"; fi
    ;;
  *)
    echo "usage: download.sh [bf16|q4-8emb] [dir]" >&2; exit 2 ;;
esac

echo "done. config.json + model.safetensors + tokenizer.json in: $DIR"
