# privacy-filter on Apple Silicon — ANE vs MLX

An experiment in running the `openai-privacy-filter` model (the gpt-oss-style
MoE token classifier this repo serves) on Apple Silicon two ways:

1. **Core ML / ANE** — convert to a `.mlpackage` and try to land it on the Neural Engine.
2. **MLX** — a sparse forward on the Metal GPU.

**Verdict: use MLX.** A 128-expert MoE needs *sparse* top-k gather, which the ANE
can't do — so the ANE path is accurate but ~44× slower. MLX's `gather_mm` does
only the selected experts and is both exact and fast.

## Result (measured, M-series Mac)

| path | argmax | cosine vs fp32 | latency | op placement |
|---|---|---|---|---|
| ANE — Core ML, **dense** MoE, 6-bit LUT (0.96 GB) | 100% | 0.997 | 336 ms¹ | 14% ANE / 85% GPU / 1% CPU |
| MLX fp32 — **sparse** `gather_mm`, Metal | 100% | 1.00000 | 11.3 ms² | GPU |
| **MLX fp16** — sparse, Metal | 100% | **1.00000** | **7.6 ms** ⭐ | GPU |
| MLX bf16 — sparse, Metal | 100% | 0.99945 | 8.1 ms² | GPU |

¹ padded 128-token window · ² 19 real tokens. The gap dwarfs the token-count difference.

**Why the ANE loses:** the dense MoE turns `gate`/`up` into 1×1 convs with
`128×640 = 81 920` output channels — past the ANE's per-layer channel limit, so
Core ML evicts the FFN to the GPU — and "dense" means computing all 128 experts
when only the top-4 fire (32× wasted work). Sparsity is exactly what the GGUF
engine exploits, and what MLX reproduces.

## Performance & footprint

The MLX forward is the on-device path. It ships **quantized by default** — 4-bit MoE +
8-bit embeddings → **870 MB** (−69 %), with **99.4 %** of token labels identical to fp32
and still **~2.7× the C++/GGML CPU** reference. Set `PF_QBITS=0` for the fp16 path:
2.8 GB but up to **4.0×** CPU. Size/speed/quality is a runtime knob (`PF_QBITS`,
`PF_QGROUP`, `PF_QEMBED`), not a hardcoded choice.

### Speed — throughput vs length (fp16 path, `PF_QBITS=0`)

| tokens | MLX fp16 (tok/s) | CPU/GGML (tok/s) | MLX vs CPU |
|-------:|-----------------:|-----------------:|-----------:|
|    512 |            8 300 |            3 564 |      2.3× |
|  2 048 |            9 400 |            3 490 |      2.7× |
|  8 192 |            9 450 |            2 332 |      4.0× |
| 16 384 |            8 600 |              —   |        —  |
| 32 768 |            7 000 |              —   |        —  |

(best-of-N on an idle M-series Mac; CPU column from the root README, Ryzen 9 7900, fp32.
MLX peaks ~9.5k tok/s at 2–8k and stays high as CPU's throughput declines.) Quantized,
throughput is a flat ~6 300 tok/s (~2.7× CPU) — `gather_qmm`'s dequant overhead is fixed,
so it's bit-width-independent. Two wins drive the fp16 speed:

1. **Sorted-MoE** (`_moe`). The MoE GEMM is the dominant cost (~57 % of an 8k forward,
   ~96 % at 512). Sorting the `n·k` (token, slot) pairs by expert turns the scattered
   `gather_mm` into one contiguous GEMM tile per expert — **+14 % at 512, +2.7× at 8k**.
   (The sorted path needs an explicit `lhs_indices`, else it silently mispairs rows.)
2. **Windowed (banded) attention** (`_attn`). Sliding-window (radius 128), but a dense
   forward still builds the full `[H, n, n]` scores — `1.9 GB` at 8k, `30 GB` at 32k,
   `481 GB` at 131k → OOM around 16–32k (the wall HF Transformers hits). Attending in
   blocks of R (each query block sees its 3 neighbours, `≤ 2R+1` keys) makes memory
   `O(n·window)` → near-flat throughput, no wall. Exact: `test_windowed.py` checks it
   equals dense+SWA to ~1e-7.

### Footprint — quantization frontier (certified on `eval.npz`, 40k tokens)

| `PF_QBITS`/`PF_QGROUP`/`PF_QEMBED` | weight | Δ | labels vs fp32 | cosine |
|---|---:|---:|---:|---:|
| `0` — fp16 | 2 799 MB | — | 99.9 % | 0.99982 |
| `8`/`64`/`0` | 1 619 MB | −42 % | 99.8 % | 0.99957 |
| **`4`/`64`/`8` — default** | **870 MB** | **−69 %** | **99.4 %** | 0.99796 |
| `3`/`128`/`8` | 670 MB | −76 % | 98.9 % | 0.99591 |
| `2`/`32`/`8` | 642 MB | −77 % | 97.9 % | 0.99092 |

MoE experts are 90 % of the weights, so quantizing them is the whole game. Below 870 MB
it's purely size↔label-drift (speed is flat); 3-bit (670 MB) is certified but drifts ~1 %
of labels — risky for a redactor, so the default stops at the 4-bit knee.

### Measuring on a Mac (lessons that cost real time)

- **Thermal noise is huge** (tok/s swings ±15–75 % with load). Compare variants with an
  **interleaved A/B** (back-to-back, repeated). Sorted-MoE first looked 2× *slower* and
  8-bit quant looked −55 % under naive sequential runs (cool baseline vs throttled
  variant); interleaved, sorted is +2.7× and quant costs only ~25 %. `mx.compile`
  likewise faked a "stability win." Use best-of-N; measure on an idle machine.
- **A tiny eval lies.** The 18-text `ref.npz` couldn't certify 3/2-bit (argmax went
  non-monotonic — 2-bit "passed" while 3-bit "failed"). `make_eval.py` builds a 200-text
  / 40k-token PII set (`eval.npz`) that makes the quality curve smooth and trustworthy.

## The model

gpt-oss-style sparse MoE re-purposed as a bidirectional PII classifier: 8 layers,
`d_model=640`, 128 experts top-4, head 14/2×64 GQA, attention sinks, bidirectional
sliding-window (radius 128), interleaved (GPT-J) RoPE + YaRN, o200k tokenizer,
33-label BIOES head. `pf_model.py` reimplements `../src/model.cpp` exactly in
PyTorch and is the **reference of record**; weights map via `../scripts/convert.py`.

## Files

| file | runs on | role |
|---|---|---|
| `pf_model.py` | any | fp32 PyTorch reference (SSOT mirror of `src/model.cpp`) |
| `pf_ane.py` | any | ANE-shaped variant: (B,C,1,S), conv-1×1, **dense** grouped-conv MoE, static shapes |
| `pf_mlx.py` | **Mac** | MLX path: sorted top-4 MoE + windowed attention; **4-bit-quant by default** (`PF_QBITS`) |
| `bench_mlx.py` | **Mac** | correctness-gated harness: `tok_s` (best-of-N), `--report-size`, `--check-only` |
| `profile_mlx.py` | **Mac** | per-component time breakdown (MoE / attn-core / proj) |
| `test_windowed.py` | **Mac** | proves windowed attention ≡ dense+SWA (rel ~1e-7) |
| `make_eval.py` | **Mac** | synth 200-text / 40k-token PII eval + fp32 reference → `eval.npz` |
| `export_coreml.py` | Linux/CUDA or Mac | trace `pf_ane` → `coremltools` → 6-bit-LUT `.mlpackage` |
| `dump_reference.py` | CUDA box | run fp32 ref + ANE-layout on a test set → `ref.npz` (portable) |
| `verify_ane.py` | **Mac** | run the `.mlpackage` on ANE/GPU, diff vs `ref.npz`, residency report |
| `download.sh`, `run_export_pc.sh` | CUDA box | fetch model; tmux driver for export+dump |

The split: a CUDA box does all the heavy lifting (download, convert, reference),
the Mac only runs the actual Apple engines. `ref.npz` carries the reference logits
so the Mac needs no weights/torch for the Core ML check.

## Run

**1. Build artifacts on the Linux/CUDA box (in tmux):**
```sh
bash apple/run_export_pc.sh            # download → PrivacyFilter.mlpackage (6-bit) + ref.npz
```
Copy `PrivacyFilter.mlpackage` and `ref.npz` to the Mac **(boot volume — see gotchas)**.

**2. ANE check on the Mac:**
```sh
uv run --with coremltools --with numpy \
  python apple/verify_ane.py ~/PrivacyFilter.mlpackage ~/ref.npz
```

**3. MLX (the fast path) on the Mac** — copy `config.json` + `tokenizer.json` +
`model.safetensors` into `apple/models/privacy-filter/`, then:
```sh
uv run --with mlx --with numpy --with tokenizers \
  python apple/pf_mlx.py apple/models/privacy-filter apple/ref.npz --dtype fp16
```

**4. Footprint / quantization knob.** The model is 4-bit-quantized by default (870 MB);
measure size and label-quality, or pick another point on the frontier:
```sh
uv run --with mlx --with numpy --with tokenizers python apple/bench_mlx.py \
  apple/models/privacy-filter apple/eval.npz --report-size            # weight_mb + peak_mb
uv run --with mlx --with numpy --with tokenizers python apple/bench_mlx.py \
  apple/models/privacy-filter apple/eval.npz --check-only             # labels/cosine vs fp32
# fp16 (max speed): PF_QBITS=0   ·   smaller (670 MB): PF_QBITS=3 PF_QGROUP=128
# regenerate the eval:  python apple/make_eval.py apple/models/privacy-filter apple/eval.npz
```

## Gotchas (each cost real time)

1. **`noowners` volumes break the CoreML compiler.** If the repo is on a volume
   mounted `noowners` (e.g. an external APFS drive), `xcrun coremlcompiler` fails
   with `NSCocoaErrorDomain 513`. Copy the `.mlpackage` to the **boot volume** first.
2. **coremltools 9.0 `MLModel(.mlpackage)` load is broken on macOS 26** (looks for
   `coremldata.bin` inside the uncompiled package). `verify_ane.py` works around it:
   `xcrun coremlcompiler compile` → load the `.mlmodelc` via `CompiledMLModel`.
3. **coremltools needs static shapes.** Derive sequence length from a fixed python
   int (see `pf_ane.py`'s `window`), never `x.shape[-1]` / `arange(symbolic)` /
   `zeros(symbolic)` — those trigger `aten::Int … 0-dimensional arrays`. 6-bit
   kmeans palettization additionally needs `scikit-learn`.
4. **`huggingface-cli download` is gone in `huggingface_hub` 1.x** → use
   `snapshot_download` (Python) or `hf download`.
5. **fp16 over fp32 is free here** (cosine 1.0) — ship fp16.

## Conclusion

The ANE port is real and correct — it just proved that a 128-expert MoE belongs on
the GPU. For an on-device privacy filter on Apple Silicon, MLX is exact, real-time, and
compact: **4-bit-quantized by default at 870 MB** with 99.4 % of labels intact and
~2.7× the CPU engine, or **fp16 at up to 4.0×** (`PF_QBITS=0`), scaling to 32k+‑token
documents a dense forward would OOM on. The natural next step is wrapping `pf_mlx.py`
in `mlx-swift` for a SwiftUI on-device app (iOS + macOS).
