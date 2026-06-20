#!/usr/bin/env python3
"""Equivalence test: windowed `_attn` must equal dense+SWA attention at multi-block n.

ref.npz texts are short (single block), so they never exercise the windowed
attention's block boundaries. This compares the model's windowed `_attn` against
an inline dense oracle (the pre-windowing code) across lengths that straddle
block edges, in fp32 -> must match to fp accumulation tolerance.

  uv run --with mlx --with numpy --with tokenizers \
      python apple/test_windowed.py apple/models/privacy-filter
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ["PF_QBITS"] = "0"   # pure attention equivalence test — no MoE quant
os.environ["PF_QEMBED"] = "0"  # and no embedding quant (test indexes tok_embd directly)

import mlx.core as mx
import numpy as np

from pf_mlx import PFMLX, load_hp, load_weights_mx


def dense_attn(m: PFMLX, h: mx.array, o: str, cos: mx.array, sin: mx.array) -> mx.array:
    """Oracle: dense attention under the SWA mask + sinks (the pre-windowing code)."""
    hp, w = m.hp, m.w
    n = h.shape[0]
    H, Hkv, dh = hp.n_head, hp.n_head_kv, hp.head_dim
    group, scale = H // Hkv, 1.0 / math.sqrt(dh)
    q = m._rope((h @ w[o + "wq"].T + w[o + "bq"]).reshape(n, H, dh), cos, sin)
    k = m._rope((h @ w[o + "wk"].T + w[o + "bk"]).reshape(n, Hkv, dh), cos, sin)
    v = (h @ w[o + "wv"].T + w[o + "bv"]).reshape(n, Hkv, dh)
    k = mx.repeat(k, group, axis=1); v = mx.repeat(v, group, axis=1)
    q = mx.transpose(q, (1, 0, 2)); k = mx.transpose(k, (1, 0, 2)); v = mx.transpose(v, (1, 0, 2))
    idx = mx.arange(n)
    mask = mx.where(mx.abs(idx[None, :] - idx[:, None]) <= hp.swa_radius, 0.0, -1e9).astype(h.dtype)
    scores = (q @ mx.swapaxes(k, -1, -2)) * scale + mask
    sink = w[o + "sinks"].reshape(H, 1, 1)
    mm = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink)
    e = mx.exp(scores - mm)
    attn = e / (mx.sum(e, axis=-1, keepdims=True) + mx.exp(sink - mm))
    ao = mx.transpose(attn @ v, (1, 0, 2)).reshape(n, H * dh)
    return ao @ w[o + "wo"].T + w[o + "bo"]


def main() -> int:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "apple/models/privacy-filter")
    hp = load_hp(model_dir)
    m = PFMLX(load_weights_mx(model_dir, hp, mx.float32), hp, mx.float32)  # fp32 for exact compare
    R = hp.swa_radius
    fail = 0
    print(f"windowed vs dense attention (fp32, R={R}):")
    for n in (19, R, R + 1, 2 * R, 3 * R, 600, 1000):
        ids = mx.array((np.arange(n) * 131 % m.w["tok_embd"].shape[0]).astype(np.int32))
        h = m.w["tok_embd"][ids]
        pos = mx.arange(n).astype(mx.float32)
        theta = pos[:, None] * m.inv[None, :]
        cos = (mx.cos(theta) * m.attn_factor).astype(mx.float32)
        sin = (mx.sin(theta) * m.attn_factor).astype(mx.float32)
        a = np.array(m._attn(h, "l0.", cos, sin))
        b = np.array(dense_attn(m, h, "l0.", cos, sin))
        d = float(np.max(np.abs(a - b)))
        rel = d / (float(np.max(np.abs(b))) + 1e-9)
        ok = rel < 1e-4
        fail += not ok
        print(f"  n={n:5d}  blocks={(n + R - 1) // R:3d}  max|Δ|={d:.2e}  rel={rel:.2e}  {'OK' if ok else 'FAIL'}")
    print("PASS" if not fail else f"{fail} FAILED")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
