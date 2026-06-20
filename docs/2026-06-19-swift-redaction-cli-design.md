# `pf` — Swift CLI for redacting secrets & PII in streams

**Date:** 2026-06-19 · **Status:** design (validated via brainstorming)

A native Swift command-line filter that hides dev secrets (tokens, API keys,
connection strings) and PII (names, emails, phones, addresses, account numbers)
in text streams — log lines, or data piped to an AI agent — using the on-device
MLX `privacy-filter` model built in `apple/`.

```sh
app 2>&1 | pf                 # scrub a live log
cat data.json | pf | agent    # scrub data before an agent reads it
```

## Decisions (resolved during brainstorming)

| fork | choice | why |
|---|---|---|
| run mode | **stdin→stdout streaming filter** | one Unix primitive serves both logs and agent-input |
| detector | **ML-only** (the MLX model) | probe confirmed it catches AWS/`sk-`/JWT/`postgres`/PII via `secret`+PII labels |
| redaction | **typed + stable token** (`<EMAIL_1>`) | same value → same token; agent can still reason; readable in logs |
| reverse | **door open, not built in v1** | optional `--map out.json` persists the mapping; `restore` is a later add |
| Swift↔MLX | **native port on mlx-swift** | a real native binary, no Python at runtime |

### Probe evidence (fp16, `apple/pf_mlx.py`)
The model flags real secrets, not just classic PII:

| input | caught | label |
|---|---|---|
| AWS secret key | ✅ | `secret` |
| `sk-proj-…` bearer | ✅ | `secret` |
| `ghp_…` GitHub token | ✅ | `private_url` (mislabeled, still hidden) |
| `postgres://…:pass@…` | ✅ (span split) | `secret` |
| JWT | ✅ | `secret` |
| name / email / phone | ✅ | clean |
| SSN / card | ✅ | `account_number` |

Caveats → handled in design: spans can be over-greedy (safe) or split (must merge
across small gaps); labels can be wrong (irrelevant when we only hide).

## Architecture

A single binary `pf`. **Key leverage:** the privacy-filter model is gpt-oss-style,
and **`mlx-swift-lm` already ships a tested GPT-OSS implementation** (MoE switch
layers, attention sinks, sliding window, YaRN RoPE, quantized experts). So we do
*not* re-port the forward from scratch — we build a thin classification wrapper on
those layers.

```
stdin ─▶ line reader ─▶ tokenizer ─▶ MLX forward ─▶ BIOES→spans ─▶ redactor ─▶ stdout
                        (offsets)     (gpt-oss)                      │ value→token map (in-mem)
                                                                     ▼ optional --map out.json (0600)
```

### Components
1. **CLI** — swift-argument-parser: `pf [--only|--except CATS] [--style typed|label] [--map FILE] [--model PATH] [--fail-open]`.
2. **Tokenizer** — swift-transformers `Tokenizers`, loads the model's `tokenizer.json` (o200k BPE), encodes each line **with char offsets**.
3. **Model** — mlx-swift, reusing mlx-swift-lm's gpt-oss blocks with **two changes**:
   - **bidirectional** sliding-window mask (not causal);
   - a **33-label token-classification head** (not the LM head).
   `apple/pf_mlx.py` is the exact spec (YaRN `truncate=false` freqs, radius-128 SWA,
   4-bit MoE + 8-bit-embed default).
4. **Span extractor** — argmax → strip BIOES → merge contiguous/near-adjacent
   same-type tokens into char spans (greedy; gap-merge ≤2 chars).
5. **Redactor** — replace spans with `<TYPE_n>` via the in-memory map; rebuild the
   line from original text + offset-located replacements (non-span text byte-exact).

Weights ship separately (`--model PATH`, default `~/.pf/model`); the binary stays small.

## Data flow / streaming
- Load model **once** at startup; long-lived process.
- Line-buffered read; redact; write+flush per line; order preserved (single stream).
- Throughput ~6 300 tok/s (4-bit) → hundreds–thousands of log lines/sec. **Optimization:**
  micro-batch consecutive lines into one padded forward; v1 ships per-line.
- Giant lines capped at N tokens/inference-unit (default 4 096); windowed attention
  handles long context, the cap bounds memory.
- Invalid-UTF-8 bytes pass through unchanged.

## Error handling — **fail-closed**
A redactor must never emit raw text on failure (that is the leak).
- Per-line model/tokenizer/OOM error → emit `⟦pf:error⟧`, log to stderr, **never** the original.
- Missing weights/model/tokenizer → exit non-zero **before** streaming starts.
- `--fail-open` opt-in only for non-sensitive contexts.
- Empty/whitespace lines pass through.

## Testing
- **Numerical oracle:** Swift logits vs `apple/pf_mlx.py` on shared inputs
  (`eval.npz` / `ref.npz`) — argmax/cosine parity gate (the Swift port must match
  the validated Python forward).
- **Golden redaction cases:** the probe set (AWS/`sk-`/JWT/`postgres`/name/email/SSN)
  → expected redactions; locks recall against regressions.
- **Leak-rate metric:** on the 200-text labeled eval, fraction of known
  secrets/PII left visible (target: ≈ the model's own recall ceiling).
- **Stream tests:** line boundaries, huge lines, non-UTF-8 passthrough, injected
  model error proving fail-closed.

## Non-goals (YAGNI, v1)
- Reverse/`restore` path (only the map dump is built).
- Wrap/`exec` mode for agents (`pf exec -- cmd`).
- Regex/entropy backstop (ML-only; revisit only if leak-rate shows gaps).
- Multilingual model, GUI, config files.

## Open items to settle at implementation
- Reuse mlx-swift-lm as a **dependency** vs **vendoring** its few gpt-oss layer files
  (depends on how composably the layers are exposed vs locked in a causal generate loop).
- Whether the gpt-oss MoE switch layer there exposes the sorted/quantized
  `gatherQMM` path we rely on (else add it).
- Exact category default set (all 9 vs secrets+account_number only).

## Next steps
1. Spike: load privacy-filter weights into mlx-swift-lm's gpt-oss blocks, run a
   **bidirectional** forward + classification head, check argmax parity vs `pf_mlx.py`
   on one sentence. (De-risks the whole port.)
2. If parity holds → build span extractor + redactor + streaming + CLI.
3. Golden + leak-rate tests; ship `pf`.
