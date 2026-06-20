#!/usr/bin/env python3
"""Run the fp32 reference on a test set and dump CoreML inputs + reference
logits to a portable .npz  (step 2, PC / CUDA side).

Run on the CUDA box (in tmux for the weight load — see apple/run_export_pc.sh):
  uv run --with torch --with safetensors --with tokenizers python \
      apple/dump_reference.py apple/models/privacy-filter apple/ref.npz \
      --window 128 --device cuda

Loads the fp32 model (~12 GB VRAM on cuda; peak ~11 GB host RAM), runs PFModel
(reference) and PFANE (the ANE-layout rewrite) over a few PII texts, and saves
emb, mask, ref_logits, ane_logits, lengths, offsets, texts, labels. Copy this
.npz + the .mlpackage to the Mac; verify_ane.py then needs only coremltools +
numpy — no weights, torch, tokenizer, or embedding table on the Mac.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pf_model import PFModel, load_hp, load_weights
from pf_ane import PFANE, build_mask

DEFAULT_TEXTS = [
    "Contact John Doe at jdoe@example.com",
    "My SSN is 123-45-6789 and I live at 42 Baker Street, London.",
    "Call Alice on +1 555-0100 or visit https://example.org/u/alice",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("texts", nargs="*")
    args = ap.parse_args()

    texts = args.texts or DEFAULT_TEXTS
    dev = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"device={dev}, {len(texts)} texts, window={args.window}")

    from tokenizers import Tokenizer

    hp = load_hp(args.model_dir)
    w = load_weights(args.model_dir, hp)                       # host (cpu)
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))

    w_dev = {k: v.to(dev) for k, v in w.items()}
    W, D, C, T = args.window, hp.n_embd, hp.n_cls, len(texts)
    ref = PFModel(w_dev, hp)
    ane = PFANE(w, hp, W).to(dev)
    ane.train(False)
    emb = np.zeros((T, 1, D, 1, W), np.float32)
    mask = np.zeros((T, 1, 1, W, W), np.float32)
    ref_log = np.zeros((T, W, C), np.float32)
    ane_log = np.zeros((T, W, C), np.float32)
    lengths = np.zeros((T,), np.int32)
    offsets = np.zeros((T, W, 2), np.int32)

    for t, text in enumerate(texts):
        enc = tok.encode(text)
        ids = torch.tensor(enc.ids[:W])
        n = int(ids.shape[0])
        lengths[t] = n
        for i, (a, b) in enumerate(enc.offsets[:n]):
            offsets[t, i] = (a, b)
        e = torch.zeros(1, D, 1, W)
        e[0, :, 0, :n] = w["tok_embd"][ids].t()               # host embedding lookup
        m = build_mask(n, W, hp.swa_radius)
        emb[t], mask[t] = e.numpy(), m.numpy()
        with torch.no_grad():
            ref_log[t, :n] = ref.forward(ids.to(dev)).cpu().numpy()
            ane_log[t] = ane(e.to(dev), m.to(dev)).reshape(C, W).t().cpu().numpy()

    # rewrite correctness on the REAL weights (reported here, on the PC)
    am = tot = 0
    for t in range(T):
        n = int(lengths[t])
        am += int((ref_log[t, :n].argmax(-1) == ane_log[t, :n].argmax(-1)).sum())
        tot += n
    print(f"PFANE vs PFModel (real weights): argmax-agree={am / tot * 100:.1f}%")

    np.savez_compressed(args.out, emb=emb, mask=mask, ref_logits=ref_log,
                        ane_logits=ane_log, lengths=lengths, offsets=offsets,
                        texts=np.array(texts), labels=np.array(hp.labels))
    print(f"wrote {args.out}  ({T} texts, window {W}, {C} labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
