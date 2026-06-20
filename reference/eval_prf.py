#!/usr/bin/env python3
"""Precision / Recall / F1 for the `pf` redactor against gold (category, value) spans.

Decoder-agnostic by design: it runs the actual `pf` BINARY end-to-end, so it scores
whatever decode pf currently uses (per-token argmax today, constrained Viterbi later) —
re-run after a decoder change for a clean before/after. One harness serves any dataset
that emits the gold JSONL schema (synthetic generator, AI4Privacy adapter, …):

    {"text": "...", "spans": [{"category": "private_person", "value": "John Smith"}, ...]}

How detections are recovered (no per-line model reload):
  • feed all texts through `pf --map <tmp>` in ONE pass (one model load);
  • pf replaces each hit with a `<CATEGORY_n>` token and writes {token: value} to the map;
  • per output line, the `<…>` tokens that appear ARE that line's detections — look each up
    in the map to get its (category, value). Attribution is therefore per-line.

Matching is per-line, value-level with substring overlap (so a boundary/partial span still
counts), category-exact. This measures DETECTION accuracy (precision = how much of what pf
redacted was real PII; recall = how much real PII pf caught) — complementary to leak_rate.sh,
which only measures whether a known value survived verbatim.

    uv run python reference/eval_prf.py gold.jsonl --model ../models/q4-8emb
    PF_BIN=pf/.build/xcode/Build/Products/Debug/pf uv run python reference/eval_prf.py gold.jsonl --model ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

PLACEHOLDER = "⟦pf:line-redacted⟧"
TOKEN_RE = re.compile(r"<([A-Z][A-Z0-9_]*?)_(\d+)>")  # <PRIVATE_PERSON_1> -> ("PRIVATE_PERSON","1")


def default_pf_bin() -> str:
    here = Path(__file__).resolve().parent.parent
    for cfg in ("Release", "Debug"):
        p = here / "pf" / ".build" / "xcode" / "Build" / "Products" / cfg / "pf"
        if p.exists():
            return str(p)
    return "pf"  # fall back to PATH


def overlap(a: str, b: str) -> bool:
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def detections_for_line(out_line: str, token_map: dict[str, str]) -> list[tuple[str, str]]:
    """The (category, value) pairs pf redacted in this output line, with repetition."""
    dets: list[tuple[str, str]] = []
    for m in TOKEN_RE.finditer(out_line):
        token = m.group(0)
        if token in token_map:
            dets.append((m.group(1).lower(), token_map[token]))
    return dets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", type=Path, help="gold JSONL: {text, spans:[{category,value}]}")
    ap.add_argument("--model", required=True, help="model dir passed to pf --model")
    ap.add_argument("--pf-bin", default=os.environ.get("PF_BIN", default_pf_bin()))
    ap.add_argument("--decoder", choices=["viterbi", "argmax"], default="viterbi",
                    help="forwarded to pf --decoder (A/B the label decoder)")
    args = ap.parse_args()

    gold_lines = [json.loads(l) for l in args.gold.read_text().splitlines() if l.strip()]
    # pf is line-oriented (one out line per in line); flatten any newlines so alignment holds.
    texts = [" ".join(str(g["text"]).splitlines()) for g in gold_lines]

    with tempfile.TemporaryDirectory() as td:
        map_path = Path(td) / "map.json"
        proc = subprocess.run(
            [args.pf_bin, "--model", args.model, "--map", str(map_path), "--decoder", args.decoder],
            input="\n".join(texts) + "\n", capture_output=True, text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(f"pf failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}\n")
            return 1
        out_lines = proc.stdout.split("\n")
        if out_lines and out_lines[-1] == "":
            out_lines = out_lines[:-1]
        token_map = json.loads(map_path.read_text()) if map_path.exists() else {}

    if len(out_lines) != len(gold_lines):
        sys.stderr.write(f"FAIL: {len(gold_lines)} gold lines vs {len(out_lines)} pf output lines\n")
        return 1

    # Relaxed, overlap-based span P/R (standard for NER; robust to span fragmentation —
    # AI4Privacy labels address sub-parts separately while our model emits one span).
    # Non-consuming: a gold span is recalled if ANY same-category detection overlaps it;
    # a detection is precise if ANY same-category gold overlaps it. So many↔one matches
    # cleanly, and recall-TP and precision-TP are counted independently.
    r_tot, r_hit = defaultdict(int), defaultdict(int)   # per gold span
    p_tot, p_hit = defaultdict(int), defaultdict(int)   # per detection
    withheld = 0
    fn_s, fp_s = defaultdict(list), defaultdict(list)
    for g, out in zip(gold_lines, out_lines):
        if out == PLACEHOLDER:
            withheld += 1
        gold = [(s["category"], str(s["value"])) for s in g.get("spans", [])]
        det = detections_for_line(out, token_map)
        for gc, gv in gold:
            r_tot[gc] += 1
            if any(dc == gc and overlap(dv, gv) for dc, dv in det):
                r_hit[gc] += 1
            elif len(fn_s[gc]) < 3:
                fn_s[gc].append(gv)
        for dc, dv in det:
            p_tot[dc] += 1
            if any(gc == dc and overlap(dv, gv) for gc, gv in gold):
                p_hit[dc] += 1
            elif len(fp_s[dc]) < 3:
                fp_s[dc].append(dv)

    cats = sorted(set(r_tot) | set(p_tot))
    print(f"gold: {len(gold_lines)} texts, {sum(r_tot.values())} spans"
          + (f"   ({withheld} lines withheld by pf)" if withheld else ""))
    print(f"{'category':16s} {'P':>6s} {'R':>6s} {'F1':>6s}   {'gold':>5s} {'det':>5s}")
    f1s = []
    PH = PT = RH = RT = 0
    for c in cats:
        p = p_hit[c] / p_tot[c] if p_tot[c] else 0.0
        r = r_hit[c] / r_tot[c] if r_tot[c] else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1)
        PH += p_hit[c]; PT += p_tot[c]; RH += r_hit[c]; RT += r_tot[c]
        note = ""
        if r_tot[c] - r_hit[c] and fn_s[c]:
            note += f"   miss {fn_s[c]}"
        if p_tot[c] - p_hit[c] and fp_s[c]:
            note += f"   spurious {fp_s[c]}"
        print(f"{c:16s} {p*100:6.1f} {r*100:6.1f} {f1*100:6.1f}   {r_tot[c]:5d} {p_tot[c]:5d}{note}")
    mp = PH / PT if PT else 0.0
    mr = RH / RT if RT else 0.0
    mf = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    print("-" * 52)
    print(f"{'micro':16s} {mp*100:6.1f} {mr*100:6.1f} {mf*100:6.1f}   {RT:5d} {PT:5d}")
    print(f"macro-F1 = {macro*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
