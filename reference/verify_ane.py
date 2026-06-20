#!/usr/bin/env python3
"""Validate the Core ML model against a PC-dumped reference (.npz) and report
ANE residency.  Mac side — needs ONLY coremltools + numpy (no torch, weights,
tokenizer, or embedding table).

  uv run --with coremltools python apple/verify_ane.py \
      apple/PrivacyFilter.mlpackage apple/ref.npz

Three diffs over real tokens only:
  ANE-layout vs ref (PC)  -> ~1.0; the rewrite is correct on real weights
  CoreML vs ref           -> the real fp16/LUT + ANE/GPU degradation
  CoreML vs ANE-layout    -> isolates Core ML/hardware from the rewrite
plus a CPU-only-vs-ALL latency proxy and a best-effort per-op device map.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    if len(sys.argv) < 3:
        sys.exit("usage: verify_ane.py <mlpackage> <ref.npz>")
    mlpkg, refnpz = Path(sys.argv[1]), Path(sys.argv[2])

    import coremltools as ct

    d = np.load(refnpz)  # native arrays only (texts/labels are <U strings, no pickle)
    emb, mask = d["emb"], d["mask"]                       # [T,1,D,1,W], [T,1,1,W,W]
    ref_log, ane_log = d["ref_logits"], d["ane_logits"]  # [T,W,C]
    lengths, offsets = d["lengths"], d["offsets"]
    texts, labels = d["texts"], d["labels"]
    T, _, _, _, W = emb.shape
    C = len(labels)

    # coremltools 9.0's MLModel(.mlpackage) compile path is broken on this macOS;
    # compile with the OS compiler (works) and load the .mlmodelc directly.
    if str(mlpkg).endswith(".mlpackage"):
        outdir = tempfile.mkdtemp(prefix="pf_ane_")
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(mlpkg), outdir], check=True)
        compiled = str(Path(outdir) / (mlpkg.stem + ".mlmodelc"))
    else:
        compiled = str(mlpkg)

    def load(units: "ct.ComputeUnit | None" = None) -> "ct.models.CompiledMLModel":
        kw = {"compute_units": units} if units is not None else {}
        return ct.models.CompiledMLModel(compiled, **kw)

    model = load()
    cm_log = np.zeros((T, W, C), np.float32)
    for t in range(T):
        out = model.predict({"emb": emb[t], "mask": mask[t]})["logits"]   # [1,C,1,W]
        cm_log[t] = np.array(out).reshape(C, W).T

    def agg(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        hit = tot = 0
        coss = []
        for t in range(T):
            n = int(lengths[t])
            aa, bb = a[t, :n], b[t, :n]
            hit += int((aa.argmax(-1) == bb.argmax(-1)).sum())
            tot += n
            coss.append(_cosine(aa, bb))
        return hit / max(tot, 1) * 100, float(np.mean(coss))

    print(f"\n{T} texts · window {W} · {C} labels")
    for name, a, b in [("ANE-layout vs ref (PC)", ane_log, ref_log),
                       ("CoreML vs ref", cm_log, ref_log),
                       ("CoreML vs ANE-layout", cm_log, ane_log)]:
        am, cs = agg(a, b)
        print(f"  {name:24} argmax-agree={am:5.1f}%   cosine={cs:.5f}")

    print("\npredicted PII (CoreML), first text:")
    n0, txt = int(lengths[0]), str(texts[0])
    pred = cm_log[0, :n0].argmax(-1)
    hits = [(txt[offsets[0, i, 0]:offsets[0, i, 1]], str(labels[pred[i]]))
            for i in range(n0) if str(labels[pred[i]]) != "O"]
    for piece, lab in hits:
        print(f"  {piece!r:24} -> {lab}")
    if not hits:
        print("  (none)")

    # latency proxy: ANE/GPU is in use if ALL beats CPU_ONLY.
    feed = {"emb": emb[0], "mask": mask[0]}

    def bench(units: "ct.ComputeUnit") -> float:
        m = load(units)
        m.predict(feed)
        t = time.perf_counter()
        for _ in range(20):
            m.predict(feed)
        return (time.perf_counter() - t) / 20 * 1000.0

    print("\nlatency (ms/forward):")
    cpu, allu = bench(ct.ComputeUnit.CPU_ONLY), bench(ct.ComputeUnit.ALL)
    print(f"  CPU_ONLY={cpu:.1f}   ALL(ANE/GPU)={allu:.1f}   speedup={cpu / max(allu, 1e-6):.2f}x")

    # per-op device placement (CoreMLTools 9.0+). Anemll's ane_profiler.py is the
    # polished, model-agnostic version of this.
    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(compiled)
        fn = plan.model_structure.program.functions["main"]
        nm = {"MLNeuralEngineComputeDevice": "ANE", "MLGPUComputeDevice": "GPU",
              "MLCPUComputeDevice": "CPU"}
        c: Counter[str] = Counter()
        for op in fn.block.operations:
            du = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if du is not None:
                c[nm.get(type(du.preferred_compute_device).__name__, "?")] += 1
        tot = sum(c.values()) or 1
        print("\ncompute-plan placement:", {k: f"{v} ({v / tot * 100:.0f}%)" for k, v in c.items()})
    except Exception as e:  # noqa: BLE001 — informational, never fatal
        print(f"\n(MLComputePlan unavailable: {e})")
    print(f" deeper residency: python ane_profiler.py -m {mlpkg} --analyze   (Anemll)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
