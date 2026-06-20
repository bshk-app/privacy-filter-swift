#!/usr/bin/env python3
"""Profile the MLX forward at a given sequence length: where does the time go?

Splits per-forward time into three buckets to size the *ceiling* of a banded-
attention optimisation before building it:
  - attention core  = scores @ + sink-softmax + @v   (the O(n^2) part banding shrinks ~32x)
  - MoE             = router + sparse experts (gather_mm)   (linear in n)
  - projections     = q/k/v/o matmuls                       (linear in n)

Reuses the model's own _moe/_rope/_norm; only the ~8-line attention core is
mirrored inline (clearly marked) because it isn't a standalone method.

  uv run --with mlx --with numpy --with tokenizers \
      python apple/profile_mlx.py apple/models/privacy-filter --tokens 8192
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from pf_mlx import PFMLX, load_hp, load_weights_mx

materialize = mx.eval  # MLX force-evaluate; not Python eval
DTYPE = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}


def bench(fn, iters: int = 6, warmup: int = 2) -> float:
    for _ in range(warmup):
        materialize(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        materialize(fn())
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.min(ts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    dtype = DTYPE[args.dtype]
    hp = load_hp(args.model_dir)
    m = PFMLX(load_weights_mx(args.model_dir, hp, dtype), hp, dtype)
    w, n, o = m.w, args.tokens, "l0."
    H, Hkv, dh = hp.n_head, hp.n_head_kv, hp.head_dim
    group, scale, L = H // Hkv, 1.0 / (dh ** 0.5), hp.n_layer

    ids = mx.array((np.arange(n) % w["tok_embd"].shape[0]).astype(np.int32))
    x = w["tok_embd"][ids]
    pos = mx.arange(n).astype(mx.float32)
    theta = pos[:, None] * m.inv[None, :]
    cos = (mx.cos(theta) * m.attn_factor).astype(dtype)
    sin = (mx.sin(theta) * m.attn_factor).astype(dtype)
    idx = mx.arange(n)
    mask = mx.where(mx.abs(idx[None, :] - idx[:, None]) <= hp.swa_radius, 0.0, -1e9).astype(dtype)
    h = m._norm(x, w[o + "attn_norm"])

    # projections (linear): q/k/v on h, o on an attention-shaped tensor
    ao_in = mx.zeros((n, H * dh), dtype=dtype)

    def proj():
        return [h @ w[o + "wq"].T + w[o + "bq"], h @ w[o + "wk"].T + w[o + "bk"],
                h @ w[o + "wv"].T + w[o + "bv"], ao_in @ w[o + "wo"].T + w[o + "bo"]]

    # attention core (O(n^2)) — mirrors PFMLX._forward; q/k/v prepared once outside the timed loop
    q = mx.transpose(m._rope((h @ w[o + "wq"].T + w[o + "bq"]).reshape(n, H, dh), cos, sin), (1, 0, 2))
    k = mx.repeat((h @ w[o + "wk"].T + w[o + "bk"]).reshape(n, Hkv, dh), group, axis=1)
    k = mx.transpose(m._rope(mx.transpose(k, (1, 0, 2)).reshape(n, H, dh), cos, sin), (1, 0, 2))
    v = mx.transpose(mx.repeat((h @ w[o + "wv"].T + w[o + "bv"]).reshape(n, Hkv, dh), group, axis=1), (1, 0, 2))
    sink = w[o + "sinks"].reshape(H, 1, 1)
    materialize(q, k, v)

    def core():
        scores = (q @ mx.swapaxes(k, -1, -2)) * scale + mask
        mm = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink)
        e = mx.exp(scores - mm)
        attn = e / (mx.sum(e, axis=-1, keepdims=True) + mx.exp(sink - mm))
        return mx.transpose(attn @ v, (1, 0, 2)).reshape(n, H * dh)

    t_core, t_proj, t_moe = bench(core) * L, bench(proj) * L, bench(lambda: m._moe(h, o)) * L
    tot = t_core + t_proj + t_moe
    print(f"\nprofile @ {n} tokens, {args.dtype}, x{L} layers (best-of-N):")
    for nm, t in [("attention core O(n^2)", t_core), ("MoE (gather_mm)", t_moe), ("projections q/k/v/o", t_proj)]:
        print(f"  {nm:24} {t:8.1f} ms   {t / tot * 100:5.1f}%")
    print(f"  {'sum (~forward)':24} {tot:8.1f} ms")
    sped = tot - t_core
    print(f"\nceiling: if banding makes the core ~free, forward {tot:.0f} -> ~{sped:.0f} ms  "
          f"({n / (tot / 1000):.0f} -> ~{n / (sped / 1000):.0f} tok/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
