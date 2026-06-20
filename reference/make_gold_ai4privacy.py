#!/usr/bin/env python3
"""AI4Privacy → gold JSONL adapter for eval_prf.py (out-of-distribution PII benchmark).

Streams ai4privacy/pii-masking-200k (English), takes the first N rows, and writes the
{text, spans:[{category,value}]} schema eval_prf.py consumes. Two fairness adjustments,
both reported so the result stays interpretable:

  • CONSERVATIVE label mapping. AI4Privacy has ~54 fine-grained labels; our model has 8
    categories. Labels without a clear equivalent (JOBAREA, COMPANYNAME, AGE, …) are
    DROPPED from gold — exactly like leak_rate.sh's AKIA exclusion — so the model is not
    penalised for entities it was never trained to detect. The adapter prints mapped vs
    dropped label counts.
  • SPAN MERGE. AI4Privacy labels an address as separate sub-parts (BUILDINGNUMBER + STREET
    + …); our model emits ONE address span. Adjacent same-category sub-spans (gap ≤ 3 chars,
    bridging ", "/" ") are merged so one detection can match one gold span.

  uv run --with datasets python reference/make_gold_ai4privacy.py reference/gold_ai4privacy.jsonl --n 500
  uv run python reference/eval_prf.py reference/gold_ai4privacy.jsonl --model ../models/q4-8emb
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# AI4Privacy label (UPPER) -> our category. Only the clear intersection; everything else
# is dropped. IP/URL both map to private_url (the model tags IPs as private_url — see
# leak_rate.sh). CITY/STATE/ZIP are intentionally NOT mapped (the model preserves loosely
# person-linked context by design, so scoring bare cities would unfairly depress recall).
MAP = {
    "FIRSTNAME": "private_person", "LASTNAME": "private_person", "MIDDLENAME": "private_person",
    "EMAIL": "private_email",
    "PHONENUMBER": "private_phone",
    "STREET": "private_address", "BUILDINGNUMBER": "private_address",
    "SECONDARYADDRESS": "private_address", "STREETADDRESS": "private_address",
    "DATE": "private_date", "DOB": "private_date",
    "ACCOUNTNUMBER": "account_number", "CREDITCARDNUMBER": "account_number",
    "IBAN": "account_number", "SSN": "account_number",
    "URL": "private_url", "IP": "private_url", "IPV4": "private_url", "IPV6": "private_url",
    "PASSWORD": "secret",
}
MERGE_GAP = 3  # merge adjacent same-category sub-spans separated by ≤ this many chars


def _mask(row) -> list[dict]:
    pm = row.get("privacy_mask")
    return json.loads(pm) if isinstance(pm, str) else (pm or [])


def spans_from_row(row) -> list[dict]:
    src = row["source_text"]
    mapped = []
    for e in _mask(row):
        cat = MAP.get(str(e.get("label", "")).upper())
        if cat is not None:
            mapped.append((int(e["start"]), int(e["end"]), cat))
    mapped.sort()
    merged: list[list] = []
    for s, e, c in mapped:
        if merged and merged[-1][2] == c and s - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e, c])
    return [{"category": c, "value": src[s:e]} for s, e, c in merged]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train", streaming=True)

    seen = n_spans = 0
    kept, dropped = Counter(), Counter()
    with args.out.open("w") as f:
        for row in ds:
            if str(row.get("language", "")).lower() != args.lang:
                continue
            for e in _mask(row):
                lab = str(e.get("label", "")).upper()
                (kept if lab in MAP else dropped)[lab] += 1
            spans = spans_from_row(row)
            text = " ".join(str(row["source_text"]).splitlines())
            f.write(json.dumps({"text": text, "spans": spans}) + "\n")
            n_spans += len(spans)
            seen += 1
            if seen >= args.n:
                break

    print(f"wrote {args.out}: {seen} texts, {n_spans} gold spans (lang={args.lang})")
    print("mapped labels:", dict(kept.most_common()))
    print("dropped (top 15):", dict(dropped.most_common(15)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
