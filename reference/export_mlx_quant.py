#!/usr/bin/env python3
"""Export a *pre-quantized* MLX artifact from the bf16 checkpoint.

Quantizes EXACTLY as `pf_mlx.load_weights_mx` does at runtime (fp16 cast ->
`mx.quantize`), then saves the resulting internal `w`-dict layout to a single
safetensors. Because it persists the very tensors the runtime path builds, the
artifact is bit-identical to the runtime-quantized model — no quality drift from
re-quantizing, only the save/load round-trip (which is lossless).

The saved layout is the model's INTERNAL keys (e.g. `l0.gate_w.weight/.scales/
.biases`, `tok_embd.weight/.scales/.biases`, `l0.attn_norm`, ...) — NOT the
upstream HF key names. config.json carries a `quantization` block so a loader
knows to read the triples instead of quantizing.

Usage:
  uv run --with mlx --with numpy --with tokenizers python reference/export_mlx_quant.py \
      models/privacy-filter out/q4-8emb --bits 4 --group 64 --embed-bits 8 --verify

Run it yourself (loads the 2.6 GB bf16 model + quantizes — ~1-2 min, one time).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

materialize = mx.eval  # MLX force-evaluate (lazy graph -> compute)


def flatten_w(w: dict) -> dict[str, mx.array]:
    """Internal `w` dict -> flat save dict. Quantized triples (lists) become
    `<key>.weight/.scales/.biases`; plain arrays are saved under their key."""
    flat: dict[str, mx.array] = {}
    for key, val in w.items():
        if isinstance(val, (tuple, list)):  # [w_q, scales, biases]
            wq, sc, qb = val
            flat[f"{key}.weight"] = wq
            flat[f"{key}.scales"] = sc
            flat[f"{key}.biases"] = qb
        else:
            flat[key] = val
    return flat


def unflatten_w(flat: dict[str, mx.array]) -> dict:
    """Inverse of flatten_w: rebuild the internal `w` dict, reassembling
    `<base>.weight/.scales/.biases` into a [w_q, scales, biases] list."""
    w: dict = {}
    triples: dict[str, dict] = {}
    for key, val in flat.items():
        if key.endswith((".weight", ".scales", ".biases")):
            base, sub = key.rsplit(".", 1)
            triples.setdefault(base, {})[sub] = val
        else:
            w[key] = val
    for base, parts in triples.items():
        w[base] = [parts["weight"], parts["scales"], parts["biases"]]
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path, help="bf16 checkpoint dir")
    ap.add_argument("out_dir", type=Path, help="output dir for the quantized artifact")
    ap.add_argument("--bits", type=int, default=4, help="MoE expert bits")
    ap.add_argument("--group", type=int, default=64, help="quant group size")
    ap.add_argument("--embed-bits", type=int, default=8, help="token-embedding bits")
    ap.add_argument("--verify", action="store_true", help="reload + parity-check vs runtime")
    args = ap.parse_args()

    # Drive pf_mlx's runtime quantization knobs so the saved tensors == runtime tensors.
    os.environ["PF_QBITS"] = str(args.bits)
    os.environ["PF_QGROUP"] = str(args.group)
    os.environ["PF_QEMBED"] = str(args.embed_bits)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_mlx  # noqa: E402

    hp = pf_mlx.load_hp(args.model_dir)
    w = pf_mlx.load_weights_mx(args.model_dir, hp, mx.float16)  # fp16 cast + quantize

    args.out_dir.mkdir(parents=True, exist_ok=True)
    flat = flatten_w(w)
    materialize(*flat.values())
    out_st = args.out_dir / "model.safetensors"
    mx.save_safetensors(str(out_st), flat)

    # config.json: original hparams + quantization block; tokenizer alongside.
    cfg = json.loads((args.model_dir / "config.json").read_text())
    cfg["quantization"] = {"group_size": args.group, "bits": args.bits, "embed_bits": args.embed_bits}
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    shutil.copy(args.model_dir / "tokenizer.json", args.out_dir / "tokenizer.json")

    size_mb = out_st.stat().st_size / 1e6
    print(f"wrote {out_st}  ({size_mb:.1f} MB)  bits={args.bits} group={args.group} embed_bits={args.embed_bits}")

    if args.verify:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
        ref_path = Path(__file__).resolve().parent / "ref.npz"
        ref = np.load(ref_path) if ref_path.exists() else None
        texts = [str(t) for t in ref["texts"][:5]] if ref is not None else [
            "Contact John Smith at john@acme.io, key sk-ABCD1234, call +1-202-555-0173."]

        runtime = pf_mlx.PFMLX(w, hp, mx.float16)
        reloaded = pf_mlx.PFMLX(unflatten_w(mx.load(str(out_st))), hp, mx.float16)

        hit = tot = 0
        coss = []
        for text in texts:
            ids = mx.array(tok.encode(text).ids, dtype=mx.int32)
            a = np.array(runtime(ids).astype(mx.float32))
            b = np.array(reloaded(ids).astype(mx.float32))
            hit += int((a.argmax(-1) == b.argmax(-1)).sum()); tot += a.shape[0]
            coss.append(float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        print(f"PARITY saved-vs-runtime: argmax-agree={hit / tot * 100:.2f}%  mean cosine={np.mean(coss):.6f}  ({len(texts)} texts)")
        if hit != tot or np.mean(coss) < 0.9999:
            print("PARITY FAILED", file=sys.stderr)
            return 1
        print("PARITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
