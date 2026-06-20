#!/usr/bin/env python3
"""Generate a larger, PII-diverse eval set + fp32 reference logits for quant certification.

The shipped ref.npz is 18 short texts — too small to certify aggressive (3/2-bit)
quantization (argmax-agree went non-monotonic on it). This synthesizes ~N varied
PII texts (short..long, many entity types), runs the UNQUANTIZED fp32 MLX model as
the reference (== SSOT; fp32-MLX matches pf_model to cosine 1.0), and saves
apple/eval.npz {texts, ref_logits, lengths} — same schema as ref.npz, so
bench_mlx.py consumes it directly.

  PF_QBITS= uv run --with mlx --with numpy --with tokenizers \
      python apple/make_eval.py apple/models/privacy-filter apple/eval.npz --n 200
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ["PF_QBITS"] = "0"   # the reference MUST be the unquantized fp32 model
os.environ["PF_QEMBED"] = "0"  # (quant is now the model default — disable it here)

import mlx.core as mx
import numpy as np

from pf_mlx import PFMLX, load_hp, load_weights_mx

FIRST = ["James", "Maria", "Wei", "Aisha", "Liam", "Sofia", "Raj", "Emma", "Chen", "Omar",
         "Olivia", "Noah", "Yuki", "Fatima", "Lucas", "Ava", "Diego", "Nadia", "Ivan", "Grace"]
LAST = ["Smith", "Garcia", "Nguyen", "Khan", "Johnson", "Rossi", "Patel", "Muller", "Kim", "Hassan",
        "Brown", "Lopez", "Tanaka", "Ali", "Schmidt", "Costa", "Petrov", "Sato", "Dubois", "Okafor"]
DOMAIN = ["example.com", "mail.org", "corp.io", "university.edu", "clinic.net", "acme.co", "proton.me"]
CITY = ["Berlin", "Toronto", "Osaka", "Lagos", "Austin", "Lyon", "Pune", "Cairo", "Oslo", "Bogota"]
STREET = ["Maple Ave", "2nd Street", "Elm Road", "King Blvd", "Park Lane", "River Way", "Oak Drive"]
ORG = ["Acme Corp", "Globex", "Initech", "Umbrella LLC", "Wayne Enterprises", "Stark Industries"]


def sentences(r: random.Random) -> list[str]:
    fn, ln = r.choice(FIRST), r.choice(LAST)
    email = f"{fn.lower()}.{ln.lower()}@{r.choice(DOMAIN)}"
    phone = f"+1-{r.randint(200, 999)}-{r.randint(200, 999)}-{r.randint(1000, 9999)}"
    return [
        f"Please contact {fn} {ln} at {email} or call {phone}.",
        f"{fn} {ln} lives at {r.randint(1, 9999)} {r.choice(STREET)}, {r.choice(CITY)}.",
        f"Patient {fn} {ln} (DOB {r.randint(1, 28):02d}/{r.randint(1, 12):02d}/19{r.randint(50, 99)}) was admitted.",
        f"Card 4{r.randint(100, 999)} {r.randint(1000, 9999)} {r.randint(1000, 9999)} {r.randint(1000, 9999)} on file.",
        f"SSN {r.randint(100, 899)}-{r.randint(10, 99)}-{r.randint(1000, 9999)} recorded for {fn}.",
        f"The shipment from {r.choice(ORG)} went to {fn} {ln} in {r.choice(CITY)}.",
        f"Login from {email} at IP {r.randint(1, 255)}.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 255)}.",
        f"{r.choice(ORG)} invoiced {fn} {ln}; reply to {email}.",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hp = load_hp(args.model_dir)
    model = PFMLX(load_weights_mx(args.model_dir, hp, mx.float32), hp, mx.float32)  # unquantized fp32 ref
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))

    r = random.Random(args.seed)
    texts = []
    for _ in range(args.n):
        k = r.choice([1, 1, 2, 3, 5, 8, 12, 20])  # vary length: short .. long documents
        sents = []
        for _ in range(k):
            sents.extend(r.sample(sentences(r), r.randint(1, 3)))
        texts.append(" ".join(sents))

    enc = [tok.encode(t).ids for t in texts]
    lengths = np.array([len(e) for e in enc], np.int32)
    W, C = int(lengths.max()), hp.n_cls
    logits = np.zeros((len(texts), W, C), np.float32)
    for i, e in enumerate(enc):
        logits[i, :len(e)] = np.array(model(mx.array(e, dtype=mx.int32)).astype(mx.float32))

    np.savez(args.out, texts=np.array(texts), ref_logits=logits, lengths=lengths)
    print(f"wrote {args.out}: {len(texts)} texts, ~{int(lengths.sum())} tokens, "
          f"len[min/med/max]={lengths.min()}/{int(np.median(lengths))}/{lengths.max()}, C={C}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
