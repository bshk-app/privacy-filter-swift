#!/usr/bin/env python3
"""Generate a PII/secret-rich corpus + per-line ground-truth of the EXACT sensitive
substrings that SHOULD be redacted — the input to the corpus leak-rate test (Task D4).

Why a new generator instead of reusing apple/eval.npz?
  eval.npz stores {texts, ref_logits, lengths} only — it has the texts but NOT the set
  of injected sensitive VALUES per line, so it cannot serve as leak ground truth. This
  script reuses the SAME sentence templates / name+email+phone+card+SSN vocabulary as
  apple/make_eval.py's `sentences()` helper, but additionally records, per generated
  line, the precise substring of each value it injected. It also injects a `secret`
  (sk-…/hex token) per record because make_eval's templates don't cover the `secret`
  class, which is the single most security-critical category for a redactor.

  It does NOT import the model (pure `random` text generation) → fast, no MLX needed.

Outputs (paths are CLI args):
  - leak-corpus.txt   : one record per line (newlines inside a record are stripped — pf
                        is line-oriented and emits exactly one output line per input line).
  - leak-values.json  : list (aligned 1:1 with corpus lines) of objects
                        {"line": <str>, "values": [{"category": <str>, "value": <str>}, ...]}
                        listing every sensitive substring that the redactor SHOULD remove.

Category set tracked as ground truth (must match the model's BIOES label families in
apple/models/privacy-filter/config.json):
    private_person, private_email, private_phone, private_address,
    private_date, private_url (IP addresses), account_number (cards + SSNs), secret

Deliberately EXCLUDED from ground truth (documented in leak_rate.sh, not asserted):
  - AWS access-key IDs (AKIA…): the model has NO label family for them and does not tag
    them (empirically confirmed). Including them would inflate the leak rate for a value
    the model is not designed to catch. We therefore do not inject them as ground truth.
    (The redactor is NOT weakened — there is simply no AKIA value claimed as redactable.)

Usage:
  uv run --with tokenizers python apple/scripts/make_leak_corpus.py \
      apple/pf/Tests/leak-corpus.txt apple/pf/Tests/leak-values.json --n 100 --seed 7
  (tokenizers is optional and unused here; kept for parity with the other generators.)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# --- vocabulary (mirrors apple/make_eval.py) -------------------------------------------
FIRST = ["James", "Maria", "Wei", "Aisha", "Liam", "Sofia", "Raj", "Emma", "Chen", "Omar",
         "Olivia", "Noah", "Yuki", "Fatima", "Lucas", "Ava", "Diego", "Nadia", "Ivan", "Grace"]
LAST = ["Smith", "Garcia", "Nguyen", "Khan", "Johnson", "Rossi", "Patel", "Muller", "Kim", "Hassan",
        "Brown", "Lopez", "Tanaka", "Ali", "Schmidt", "Costa", "Petrov", "Sato", "Dubois", "Okafor"]
DOMAIN = ["example.com", "mail.org", "corp.io", "university.edu", "clinic.net", "acme.co", "proton.me"]
CITY = ["Berlin", "Toronto", "Osaka", "Lagos", "Austin", "Lyon", "Pune", "Cairo", "Oslo", "Bogota"]
STREET = ["Maple Ave", "2nd Street", "Elm Road", "King Blvd", "Park Lane", "River Way", "Oak Drive"]
ORG = ["Acme Corp", "Globex", "Initech", "Umbrella LLC", "Wayne Enterprises", "Stark Industries"]

_HEX = "0123456789abcdef"
_B62 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _secret(r: random.Random) -> str:
    """A high-entropy secret token of the kind the `secret` label is trained on."""
    kind = r.choice(["skproj", "sklive", "ghp", "hex"])
    if kind == "skproj":
        return "sk-proj-" + "".join(r.choice(_B62) for _ in range(r.randint(20, 32)))
    if kind == "sklive":
        return "sk-live-" + "".join(r.choice(_B62) for _ in range(r.randint(20, 32)))
    if kind == "ghp":
        return "ghp_" + "".join(r.choice(_B62) for _ in range(36))
    return "".join(r.choice(_HEX) for _ in range(40))  # bare 40-char hex token


def record(r: random.Random) -> tuple[str, list[dict[str, str]]]:
    """Build one corpus line and its ground-truth list of (category, exact-substring)."""
    fn, ln = r.choice(FIRST), r.choice(LAST)
    name = f"{fn} {ln}"
    email = f"{fn.lower()}.{ln.lower()}@{r.choice(DOMAIN)}"
    phone = f"+1-{r.randint(200, 999)}-{r.randint(200, 999)}-{r.randint(1000, 9999)}"
    card = f"4{r.randint(100, 999)} {r.randint(1000, 9999)} {r.randint(1000, 9999)} {r.randint(1000, 9999)}"
    ssn = f"{r.randint(100, 899)}-{r.randint(10, 99)}-{r.randint(1000, 9999)}"
    dob = f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/19{r.randint(50, 99)}"
    ip = f"{r.randint(1, 255)}.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 255)}"
    secret = _secret(r)

    # Each template carries a known set of injected values. We keep one value per category
    # per line so ground truth is unambiguous (the redactor maps a value -> a single token).
    templates = [
        (f"Please contact {name} at {email} or call {phone}.",
         [("private_person", name), ("private_email", email), ("private_phone", phone)]),
        (f"Patient {name} (DOB {dob}) was admitted by Dr. {r.choice(LAST)}.",
         [("private_person", name), ("private_date", dob)]),
        (f"Card {card} and SSN {ssn} on file for {name}.",
         [("account_number", card), ("account_number", ssn), ("private_person", name)]),
        (f"Deploy used api key {secret}; notify {email}.",
         [("secret", secret), ("private_email", email)]),
        (f"Login from {email} at IP {ip} by {name}.",
         [("private_email", email), ("private_url", ip), ("private_person", name)]),
        (f"{r.choice(ORG)} invoiced {name}; reply to {email} or {phone}.",
         [("private_person", name), ("private_email", email), ("private_phone", phone)]),
        (f"Service token {secret} rotated; SSN {ssn} kept for {fn}.",
         [("secret", secret), ("account_number", ssn)]),
        (f"Charged card {card} for {name} on {dob}.",
         [("account_number", card), ("private_person", name), ("private_date", dob)]),
    ]
    line, values = r.choice(templates)
    # pf is line-oriented: guarantee a single physical line per record.
    line = re.sub(r"\s*\n\s*", " ", line).strip()
    # Drop any value that (after newline-stripping) is no longer a verbatim substring —
    # ground truth must be exactly what appears in the emitted line.
    values = [{"category": c, "value": v} for c, v in values if v in line]
    return line, values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_out", type=Path)
    ap.add_argument("values_out", type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    r = random.Random(args.seed)
    lines: list[str] = []
    records: list[dict] = []
    for _ in range(args.n):
        line, values = record(r)
        assert "\n" not in line, "corpus line must be single-line"
        lines.append(line)
        records.append({"line": line, "values": values})

    args.corpus_out.parent.mkdir(parents=True, exist_ok=True)
    args.corpus_out.write_text("\n".join(lines) + "\n")
    args.values_out.parent.mkdir(parents=True, exist_ok=True)
    args.values_out.write_text(json.dumps(records, indent=0))

    total_vals = sum(len(rec["values"]) for rec in records)
    by_cat: dict[str, int] = {}
    for rec in records:
        for v in rec["values"]:
            by_cat[v["category"]] = by_cat.get(v["category"], 0) + 1
    print(f"wrote {args.corpus_out}: {len(lines)} lines")
    print(f"wrote {args.values_out}: {total_vals} ground-truth values")
    print("by category: " + ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
