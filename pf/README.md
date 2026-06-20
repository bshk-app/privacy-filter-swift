# `pf` — Swift CLI redactor (native mlx-swift)

Native Swift port of the MLX `privacy-filter` (see `../pf_mlx.py`) as a streaming
secret/PII redactor. Design: [`../../docs/plans/2026-06-19-swift-redaction-cli-design.md`](../../docs/plans/2026-06-19-swift-redaction-cli-design.md).

Direct port of `../pf_mlx.py`'s forward to mlx-swift core (which has `gatherMM`,
`argSort`, `MLXFast.rmsNorm`, `quantize`) — not a fork of mlx-swift-lm.

## Milestones

| # | scope | status |
|---|---|---|
| 0 | foundation: deps build, load safetensors on Metal | ✅ done |
| 1 | tokenizer (swift-transformers, with offsets) + embedding lookup | ✅ done |
| 2 | forward (RMSNorm, attn+sinks+bidir, YaRN RoPE, dense, unsorted MoE) → parity vs `pf_mlx.py` | ✅ **cosine 1.0, 20/20** (`pf-parity`) |
| 3 | BIOES→spans + stable-token redactor | ✅ done |
| 4 | streaming stdin→stdout + CLI flags + fail-closed | ✅ done |
| 5 | windowed attn + sorted/quant MoE (`gatherQMM`) for long/fast; golden + leak-rate tests | ✅ done |

M2 proved parity with dense attention + unsorted `gatherMM` (simplest correct
equivalents; the fixture is < SWA radius so windowing is identity). M5 added the
windowed attention + sorted/quant MoE for long-context/speed/footprint.
`pf_mlx.py` is the parity oracle.

## Build & run

**Must build via Xcode, not `swift build`/`swift run`** — only Xcode's build system
compiles MLX's Metal kernels into `default.metallib`, which MLX needs at init
(mlx-swift README §"SwiftPM (command line) cannot build the Metal shaders"). Use the
helper (mlx-run style: `xcodebuild` + run from DerivedData); first build is several
minutes (Metal compile), then incremental. Two products: `pf` (the streaming
redactor CLI) and `pf-parity` (the parity check vs `pf_mlx.py`).

```sh
# redactor CLI (stdin → stdout)
echo "my secret is sk-abc123" | apple/pf/run.sh pf ../models/privacy-filter
# parity check — prints `PARITY OK` (cosine 1.0)
apple/pf/run.sh pf-parity ../models/privacy-filter parity-fixture.json
# CONFIG=Release apple/pf/run.sh pf ...   for an optimized build
```

The model is **4-bit-quantized by default (~830 MB on disk)** — 4-bit MoE + 8-bit
embed, fp16 elsewhere.

## Tests
- `swift test` — the pure-Swift `PFCore` redaction core (no MLX/Metal, fast).
- `apple/pf/run.sh pf-parity ../models/privacy-filter parity-fixture.json` —
  forward parity vs `pf_mlx.py` (`PARITY OK`, cosine 1.0).
- `Tests/e2e.sh` — end-to-end redactor over stdin→stdout.
- `Tests/leak_rate.sh` — leak-rate check against the golden fixtures.

## Model facts (from milestone 0, for the port)
- weights on disk are **bfloat16**; `vocab = 200064`, `d_model = 640`,
  `intermediate I = 640` (`experts.gate_up_proj [128, 640, 1280]` = `[E, D, 2I]`),
  14 q-heads (`q_proj [896, 640]`), 128 experts, `sinks [14]`, `score [33, 640]`.
- weight keys match `../pf_mlx.py` `load_weights_mx` exactly.
