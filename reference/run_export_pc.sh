#!/usr/bin/env bash
# Heavy step on the CUDA box (pc.lan): build the Core ML .mlpackage AND dump the
# reference .npz, inside tmux so it survives an SSH disconnect.
#
#   bash apple/run_export_pc.sh [model_dir] [window] [quant]
#   tmux attach -t pf-export                 # watch progress
#
# When DONE, copy the two artifacts to the Mac and verify there:
#   scp apple/PrivacyFilter.mlpackage apple/ref.npz  mac:.../privacy-filter.cpp/apple/
#   # on the Mac:
#   uv run --with coremltools python apple/verify_ane.py \
#       apple/PrivacyFilter.mlpackage apple/ref.npz
set -euo pipefail

MODEL="${1:-apple/models/privacy-filter}"
WINDOW="${2:-128}"
QUANT="${3:-6bit}"
SESSION=pf-export
DEPS="--with torch --with safetensors --with tokenizers --with coremltools --with numpy"

tmux new-session -d -s "$SESSION" "
set -e
echo '[1/2] export -> apple/PrivacyFilter.mlpackage';
uv run $DEPS python apple/export_coreml.py '$MODEL' --window '$WINDOW' --quantize '$QUANT' --out apple/PrivacyFilter.mlpackage;
echo '[2/2] dump reference -> apple/ref.npz';
uv run $DEPS python apple/dump_reference.py '$MODEL' apple/ref.npz --window '$WINDOW' --device cuda;
echo 'DONE: apple/PrivacyFilter.mlpackage + apple/ref.npz';
exec bash
"
echo "started tmux session '$SESSION' on $(hostname)."
echo "watch:   tmux attach -t $SESSION"
echo "outputs: apple/PrivacyFilter.mlpackage  +  apple/ref.npz"
