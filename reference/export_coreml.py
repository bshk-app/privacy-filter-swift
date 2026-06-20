#!/usr/bin/env python3
"""Trace PFANE and convert it to a Core ML .mlpackage (step 2).

Heavy — run yourself on an Apple-Silicon Mac (macOS 15+ recommended; needs
coremltools 8+, builds the full 1.5B model in fp32 to trace, ~6 GB RAM):

  uv run --with coremltools --with torch --with safetensors python \
      apple/export_coreml.py apple/models/privacy-filter \
      --window 128 --quantize 6bit --out apple/PrivacyFilter.mlpackage

fp16 is 2.82 GB, which exceeds Core ML's load limit (~2 GB/file macOS, 1 GB
iOS — per Anemll docs/convert.md), so weights are LUT-palettized by default:
6bit ~1.06 GB fits macOS, 4bit ~0.7 GB fits iOS. LUT quant is ANE-native.
`--quantize none` keeps raw fp16 (will likely fail to load on-device).

The model takes precomputed token embeddings ([1,n_embd,1,W]) + an additive
attention mask ([1,1,W,W]) and emits per-token logits ([1,n_cls,1,W]); the
201k-row embedding gather stays on the host (see pf_ane.embed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pf_model import load_hp, load_weights
from pf_ane import PFANE, build_mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--window", type=int, default=128, help="fixed sequence length S")
    ap.add_argument("--out", type=Path, default=Path("apple/PrivacyFilter.mlpackage"))
    ap.add_argument("--compute-units", default="all", choices=["all", "cpu_and_ne", "cpu_only"])
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--quantize", default="6bit", choices=["none", "6bit", "4bit"],
                    help="LUT palettization (default 6bit): fp16 is 2.82GB and "
                         "exceeds Core ML's per-file load limit")
    args = ap.parse_args()

    import coremltools as ct

    hp = load_hp(args.model_dir)
    w = load_weights(args.model_dir, hp)
    W, D = args.window, hp.n_embd
    model = PFANE(w, hp, W)
    model.train(False)

    emb = torch.zeros(1, D, 1, W)
    mask = build_mask(W, W, hp.swa_radius)            # shape-only example; mask is a live input
    print(f"tracing PFANE  (window={W}, n_embd={D}, experts={hp.n_expert}) ...")
    with torch.no_grad():
        traced = torch.jit.trace(model, (emb, mask))

    cu = {"all": ct.ComputeUnit.ALL,
          "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
          "cpu_only": ct.ComputeUnit.CPU_ONLY}[args.compute_units]
    prec = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32

    print(f"converting -> mlprogram ({args.precision}, {args.compute_units}) ...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="emb", shape=(1, D, 1, W), dtype=np.float32),
                ct.TensorType(name="mask", shape=(1, 1, W, W), dtype=np.float32)],
        outputs=[ct.TensorType(name="logits")],
        compute_units=cu,
        compute_precision=prec,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    if args.quantize != "none":
        nbits = 6 if args.quantize == "6bit" else 4
        import coremltools.optimize.coreml as cto
        print(f"palettizing weights -> {nbits}-bit LUT (kmeans) ...")
        # Global palettization. If top-4 routing degrades, exclude the router /
        # embeddings via op_name_configs={name: None} — keep them higher precision.
        cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(nbits=nbits, mode="kmeans"))
        mlmodel = cto.palettize_weights(mlmodel, cfg)

    mlmodel.short_description = "openai-privacy-filter PII token classifier (ANE experiment)"
    mlmodel.user_defined_metadata["labels"] = json.dumps(hp.labels)
    mlmodel.user_defined_metadata["window"] = str(W)
    mlmodel.user_defined_metadata["swa_radius"] = str(hp.swa_radius)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))
    size_gb = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) / 1e9
    print(f"saved {args.out}  ({size_gb:.2f} GB, window={W}, {args.precision}/{args.quantize}, {hp.n_cls} labels)")
    if size_gb > 2.0:
        print("  WARNING: >2 GB exceeds macOS Core ML per-file limit — use --quantize 4bit")
    print("next: apple/verify_ane.py (accuracy) + Anemll ane_profiler.py (ANE residency)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
