#!/usr/bin/env python3
"""Export a tokenizer parity fixture (ids + char offsets) for the Swift port.
  uv run --with tokenizers python apple/scripts/make_tok_fixture.py \
      apple/models/privacy-filter apple/pf/Tests/fixtures/tok-fixture.json
"""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

md = Path(sys.argv[1])
out = Path(sys.argv[2])
tok = Tokenizer.from_file(str(md / "tokenizer.json"))
# Multi-byte case: built from EXPLICIT codepoints so the combining sequences survive
# editor normalization. It deliberately mixes:
#   - "café" / "résumé": base letter + COMBINING ACUTE (U+0301)
#   - "❤️": HEAVY BLACK HEART + VARIATION SELECTOR-16 (emoji presentation)
#   - "日本語": CJK (each 1 grapheme = 1 codepoint, multi-byte in UTF-8)
# Result: 28 codepoints but only 24 extended grapheme clusters. This is the whole point:
# Python `tokenizers` reports offsets as Python `str` indices == CODEPOINTS, so the Swift
# port MUST index by `text.unicodeScalars` (codepoints), NOT `Array(text)` (Character ==
# extended grapheme cluster). A grapheme-based walk produces offsets shifted by the number
# of combining marks and FAILS this case; a codepoint walk reproduces Python exactly.
# (Avoids characters whose UTF-8 bytes get split ACROSS bpe tokens — e.g. many emoji —
#  because such tokens decode to U+FFFD and cannot be anchored by a decode-walk; those
#  belong to the fail-closed path, not to a passing fixture case.)
MULTIBYTE = "café ❤️ résumé 日本語 x@y.io"
cases = ["Contact John Smith at john@acme.com.",
         "key sk-proj-abc123 and AKIAIOSFODNN7EXAMPLE",
         "", "   ", "no pii here at all",
         MULTIBYTE]
data = [{"text": t, "ids": (e := tok.encode(t)).ids, "offsets": [list(o) for o in e.offsets]}
        for t in cases]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=0, ensure_ascii=False))
print(f"wrote {out}: {len(data)} cases")
