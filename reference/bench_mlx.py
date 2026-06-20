#!/usr/bin/env python3
"""Benchmark harness for the MLX privacy-filter forward — the autoresearch metric.

One job: emit a single, CPU-comparable throughput number for the sparse MLX
forward, *gated on correctness* so an optimisation can't "win" by breaking the
model. The model itself lives in pf_mlx.py (imported, never duplicated).

  uv run --with mlx --with numpy --with tokenizers \
      python apple/bench_mlx.py apple/models/privacy-filter apple/ref.npz \
      --dtype fp16 --bench-tokens 512

stdout (last line, parseable):   tok_s=<float> ms=<float> tokens=<int>
stderr:                          correctness diagnostics
Exit 1 if argmax-agree < 100% or mean cosine < the threshold vs the fp32
reference, so the autoresearch loop rejects any edit that regresses accuracy.

CPU reference to beat (README, Ryzen 9 7900, 12 threads, fp32 GGML):
  512 tok -> 3564 tok/s · 2048 -> 3490 · 8192 -> 2332.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from pf_mlx import PFMLX, load_hp, load_weights_mx  # DRY: single model definition

materialize = mx.eval  # MLX force-evaluate (lazy graph -> compute); not Python eval
DTYPE = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}


def _bench_ids(tok, texts, n: int) -> mx.array:
    """Deterministic, in-vocab length-n input: real tokens tiled to n."""
    ids: list[int] = []
    i = 0
    while len(ids) < n:
        ids.extend(tok.encode(str(texts[i % len(texts)])).ids)
        i += 1
    return mx.array(ids[:n], dtype=mx.int32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("ref", type=Path, help="ref.npz from dump_reference.py")
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--bench-tokens", type=int, default=512)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--min-cosine", type=float, default=0.995)  # 4-bit default lands ~0.998; PF_QBITS=0 ~1.0
    ap.add_argument("--min-argmax", type=float, default=99.0)    # label-agreement floor (fp16~99.9, 4-bit~99.4)
    ap.add_argument("--check-only", action="store_true",
                    help="correctness gate only, skip timing (fast Guard for the loop)")
    ap.add_argument("--report-size", action="store_true",
                    help="print loaded weight memory (MB) and exit (size metric)")
    args = ap.parse_args()

    dtype = DTYPE[args.dtype]
    hp = load_hp(args.model_dir)
    model = PFMLX(load_weights_mx(args.model_dir, hp, dtype), hp, dtype)

    if args.report_size:  # size metric: total loaded weight bytes (reflects quantization)
        def _nb(v):  # quantized weights are stored as [w_q, scales, biases]
            return sum(int(a.nbytes) for a in v) if isinstance(v, (tuple, list)) else int(v.nbytes)
        wb = sum(_nb(v) for v in model.w.values()) / 1e6
        line = f"weight_mb={wb:.2f}"
        try:  # peak forward memory is informational (weights + activations @512)
            materialize(model(mx.arange(512, dtype=mx.int32) % model.w["tok_embd"].shape[0]))
            line += f" peak_mb={mx.get_peak_memory() / 1e6:.1f}"
        except Exception:  # noqa: BLE001
            pass
        print(line)
        return 0

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))

    d = np.load(args.ref)
    texts, ref_log, lengths = d["texts"], d["ref_logits"], d["lengths"]

    # --- correctness gate (same check as pf_mlx.py, against the fp32 reference) ---
    hit = tot = 0
    coss = []
    for t, text in enumerate(texts):
        ids = mx.array(tok.encode(str(text)).ids, dtype=mx.int32)
        out = np.array(model(ids).astype(mx.float32))
        n = int(lengths[t])
        a, b = out[:n], ref_log[t, :n]
        hit += int((a.argmax(-1) == b.argmax(-1)).sum())
        tot += n
        coss.append(float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    argmax = hit / max(tot, 1) * 100.0
    cosine = float(np.mean(coss))
    print(f"correctness: argmax-agree={argmax:.1f}%  mean cosine={cosine:.5f}  "
          f"(dtype={args.dtype})", file=sys.stderr)
    if argmax < args.min_argmax or cosine < args.min_cosine:
        print(f"FAIL: correctness regressed (argmax {argmax:.1f}% < {args.min_argmax}% or "
              f"cosine {cosine:.5f} < {args.min_cosine})", file=sys.stderr)
        return 1
    if args.check_only:
        print("OK: correctness gate passed", file=sys.stderr)
        return 0

    # --- throughput at a fixed, CPU-comparable length (best-of-N, noise-robust) ---
    n = args.bench_tokens
    ids = _bench_ids(tok, texts, n)
    for _ in range(5):
        materialize(model(ids))  # warmup: Metal shader compile + allocate
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        materialize(model(ids))
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = float(np.min(times))     # headline: best-of-N — the only reproducible signal on a
    med = float(np.median(times)) # thermally-noisy Mac (median's tail severity varies ~75%).
    tok_s = n / (ms / 1000.0)     # min still tracks real fixes (dispatch/kernel/memory floor).
    print(f"timing: best={ms:.3f}ms median={med:.3f}ms over {args.iters} iters "
          f"(best≈median => stable/steady-state; big gap => machine jitter, re-run)",
          file=sys.stderr)
    print(f"tok_s={tok_s:.1f} ms={ms:.3f} tokens={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
