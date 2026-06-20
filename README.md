# privacy-filter-swift

On-device **secret & PII redactor for Apple Silicon** — a native Swift CLI that runs
the `openai/privacy-filter` token-classifier (a gpt-oss-style MoE) on the GPU via
[MLX](https://github.com/ml-explore/mlx-swift), streaming `stdin → stdout` and replacing
detected tokens/keys/PII with stable typed placeholders. Built for scrubbing logs and
sanitizing data before it reaches an AI agent.

```sh
app 2>&1 | pf                 # scrub a live log
cat data.json | pf | agent    # scrub data before an agent reads it
```
→ `Contact <PRIVATE_PERSON_1> at <PRIVATE_EMAIL_1> key <SECRET_1>` — no raw value leaks.

## Layout

| path | what |
|---|---|
| `pf/` | **the product** — the native Swift CLI (SwiftPM package): `PFCore` (pure redaction logic), `PFModel` (the MLX forward), `PFTokenizer`, the `pf` CLI + `pf-parity` check. |
| `reference/` | the Python MLX/ANE pipeline that produced & validates it — `pf_mlx.py` is the numerical **oracle of record**; plus fixture/eval generators and the ANE experiment. |
| `docs/` | the design + implementation plan. |
| `models/` | (gitignored) the model — put `privacy-filter/` here (`reference/download.sh`). |

## Build & run

**Build via Xcode, not `swift build`/`swift run`** — only Xcode's build system compiles
MLX's Metal kernels (`default.metallib`), which MLX needs at init. The helper does it:

```sh
echo "my secret is sk-proj-abc123" | pf/run.sh pf ../models/privacy-filter   # the redactor
pf/run.sh pf-parity ../models/privacy-filter parity-fixture.json             # parity vs pf_mlx.py
```
First build compiles MLX + Metal (a few minutes), then incremental.

## Status

- **Parity:** the Swift forward is bit-identical to `reference/pf_mlx.py` — cosine **1.0** (20- and 405-token fixtures).
- **Footprint:** ships **4-bit MoE + 8-bit embeddings ≈ 830 MB** (`PF_QBITS=0` for fp16).
- **Quality:** corpus **leak-rate 1.5 %** (secret / email / person / phone / IP categories at 0 %).
- **Tests:** `swift test --package-path pf` (PFCore, fast) · `pf/run.sh pf-parity …` · `pf/Tests/e2e.sh` · `pf/Tests/leak_rate.sh`.

## Next

- A SwiftUI app wrapping `pf` (iOS + macOS).
- A regex/entropy backstop for formats the model has no label for (e.g. AWS access-key IDs `AKIA…`).
