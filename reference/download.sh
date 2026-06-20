#!/usr/bin/env bash
# Fetch the openai/privacy-filter HF checkpoint (~3 GB) for the ANE experiment.
# We convert FROM the HF safetensors (not the GGUF) because coremltools traces
# torch graphs and config.json gives clean hparams; the HF->our-weights mapping
# is the one in scripts/convert.py.
#
# Run it yourself (heavy download — not auto-run):
#   bash apple/download.sh                       # -> apple/models/privacy-filter
#   bash apple/download.sh /path/to/dir          # custom dir
set -euo pipefail

DIR="${1:-$(cd "$(dirname "$0")" && pwd)/models/privacy-filter}"
mkdir -p "$DIR"

echo "Downloading openai/privacy-filter -> $DIR  (~3 GB, one time)"
huggingface-cli download openai/privacy-filter \
    --local-dir "$DIR" \
    --exclude "*.gguf" "original/*"

echo "done. config.json + model.safetensors + tokenizer.json in: $DIR"
