#!/usr/bin/env python3
"""Export a LONG parity fixture (>256 tokens) for the Swift windowed-attention check.

The shipped apple/pf/parity-fixture.json is n=20 — shorter than the SWA radius
(R=128), so it never exercises the windowed attention's block boundaries. This
synthesizes one PII-rich text long enough to span >=3 blocks (so neighbour-block
concat + band/validity masks all get hit), runs the UNQUANTIZED fp32 MLX model
(the numerical oracle, == apple/pf_mlx.py), and writes the SAME schema as
parity-fixture.json: {text, ids, labels, argmax, logits}.

Because windowed attention is bit-identical to dense+SWA-mask, Swift parity on
this fixture must stay cosine ~1.0 with full argmax agreement.

  PF_QBITS=0 PF_QEMBED=0 uv run --with mlx --with numpy --with tokenizers \
      python apple/scripts/make_parity_fixture.py \
      apple/models/privacy-filter apple/pf/Tests/fixtures/parity-long.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PF_QBITS", "0")   # the oracle MUST be the unquantized fp32 model
os.environ.setdefault("PF_QEMBED", "0")  # (quant is the model default — disable it here)

# pf_mlx.py / make_eval.py live in apple/ (one dir up); import them regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np

from pf_mlx import PFMLX, load_hp, load_weights_mx

# Reuse make_eval.py's PII-diverse sentence generator (kept in sync intentionally).
from make_eval import sentences


def build_text(r: random.Random, tok, min_tokens: int) -> str:
    """Concatenate varied PII sentences until the tokenized length exceeds min_tokens."""
    parts: list[str] = []
    while True:
        parts.extend(sentences(r))          # 8 varied PII sentences per call
        text = " ".join(parts)
        if len(tok.encode(text).ids) > min_tokens:
            return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--min-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if int(os.environ.get("PF_QBITS", "0")) or int(os.environ.get("PF_QEMBED", "0")):
        raise SystemExit("set PF_QBITS=0 PF_QEMBED=0 — the parity oracle must be unquantized fp32")

    hp = load_hp(args.model_dir)
    model = PFMLX(load_weights_mx(args.model_dir, hp, mx.float32), hp, mx.float32)  # fp32 oracle

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))

    text = build_text(random.Random(args.seed), tok, args.min_tokens)
    ids = tok.encode(text).ids
    logits = np.array(model(mx.array(ids, dtype=mx.int32)).astype(mx.float32))   # [n, n_cls]
    argmax = logits.argmax(-1).astype(int).tolist()

    out = {
        "text": text,
        "ids": [int(i) for i in ids],
        "labels": hp.labels,
        "argmax": argmax,
        "logits": logits.astype(float).tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}: n={len(ids)} tokens, blocks={(len(ids) + hp.swa_radius - 1) // hp.swa_radius}, "
          f"C={hp.n_cls}, text={len(text)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
