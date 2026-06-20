#!/usr/bin/env python3
"""Synthetic gold PII corpus for eval_prf.py — texts WITH their (category, value) spans.

Reuses the entity pools from make_eval.py (DRY); each template records exactly which
(category, value) it injected, so we get ground-truth spans for precision/recall/F1.
Single-line texts (pf is line-oriented). Synthetic & templated → optimistic and
in-distribution; pair with eval_ai4privacy.py for an out-of-distribution check.

  uv run python reference/make_gold.py reference/gold_synth.jsonl --n 300
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from make_eval import FIRST, LAST, DOMAIN, CITY, STREET, ORG  # DRY: shared entity pools


def _hex(r: random.Random, k: int) -> str:
    return "".join(r.choice("0123456789abcdef") for _ in range(k))


def record(r: random.Random) -> tuple[str, list[dict]]:
    fn, ln = r.choice(FIRST), r.choice(LAST)
    person = f"{fn} {ln}"
    email = f"{fn.lower()}.{ln.lower()}@{r.choice(DOMAIN)}"
    phone = f"+1-{r.randint(200, 999)}-{r.randint(200, 999)}-{r.randint(1000, 9999)}"
    card = f"4{r.randint(100, 999)} {r.randint(1000, 9999)} {r.randint(1000, 9999)} {r.randint(1000, 9999)}"
    ssn = f"{r.randint(100, 899)}-{r.randint(10, 99)}-{r.randint(1000, 9999)}"
    date = f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/19{r.randint(50, 99)}"
    addr = f"{r.randint(1, 9999)} {r.choice(STREET)}, {r.choice(CITY)}"
    sk = f"sk-{r.choice(['proj-', ''])}{_hex(r, r.randint(20, 32))}"
    url = f"https://{r.choice(DOMAIN)}/u/{_hex(r, 8)}"
    dburl = f"postgres://{fn.lower()}:{_hex(r, 10)}@db.{r.choice(DOMAIN)}:5432/prod"

    def P(c: str, v: str) -> dict: return {"category": c, "value": v}

    templates = [
        (f"Please contact {person} at {email} or call {phone}.",
         [P("private_person", person), P("private_email", email), P("private_phone", phone)]),
        (f"{person} lives at {addr}.",
         [P("private_person", person), P("private_address", addr)]),
        (f"Patient {person} (DOB {date}) was admitted.",
         [P("private_person", person), P("private_date", date)]),
        (f"Card {card} on file; SSN {ssn} recorded.",
         [P("account_number", card), P("account_number", ssn)]),
        (f"Login from {email} via {url}.",
         [P("private_email", email), P("private_url", url)]),
        (f"Service API key {sk} must be rotated.",
         [P("secret", sk)]),
        (f"Connect to {dburl} before deploy.",
         [P("private_url", dburl)]),
        (f"{r.choice(ORG)} invoiced {person}; reply to {email}.",
         [P("private_person", person), P("private_email", email)]),
    ]
    return r.choice(templates)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    r = random.Random(args.seed)
    n_spans = 0
    with args.out.open("w") as f:
        for _ in range(args.n):
            text, spans = record(r)
            n_spans += len(spans)
            f.write(json.dumps({"text": text, "spans": spans}) + "\n")
    print(f"wrote {args.out}: {args.n} texts, {n_spans} gold spans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
