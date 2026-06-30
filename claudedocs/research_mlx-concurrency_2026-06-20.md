# Research: Concurrency on MLX (for `pf serve`)

**Date:** 2026-06-20  ·  **Depth:** standard + GitHub state verification  ·  **Confidence:** High (verified against live issue state + release notes)

> **Correction (this revision):** an earlier draft concluded "MLX is not thread-safe, must serialize." **That was stale.** Thread-safety support **shipped in MLX 0.31.2 (2026-04-22)**; the tracking issues were closed *completed* on 2026-04-24. The nuanced, current picture is below.

## Question
For a resident `pf serve` daemon holding one MLX model, can we serve requests concurrently, or must we serialize? What's the current MLX concurrency model (mlx-swift 0.31.4)?

## Executive summary
**As of MLX 0.31.2 / mlx-swift 0.31.4, concurrent use from multiple threads for *independent computations* is officially supported** — it no longer crashes in general. So serialization is **no longer a correctness requirement**. However: (1) the **continuous-batching / BatchedEngine multi-stream-worker path is still broken** (mlx-lm #1256, open, confirmed on 0.31.2) — avoid it; (2) thread-safety ≠ free speedup — concurrent forwards still contend for one GPU. **Recommendation for `pf serve` v1: still serialize**, but now as a *simplicity + single-GPU throughput-equivalence* choice, not a safety mandate. In-process concurrency is a safe upgrade path if profiling shows GPU idle time.

## Verified current state (GitHub, 2026-06-20)

| Item | State | Note |
|---|---|---|
| mlx #2133 (thread-safety tracking) | **closed/completed** 2026-04-24 | *"MLX has thread safety support in 0.31.2 now"* — zcbenz |
| mlx #3078 (concurrent independent models) | **closed/completed** 2026-04-24 | same |
| mlx #2067 (eval thread bug) | **closed/completed** 2026-04-24 | same |
| mlx PR #2104 (old "Metal thread safety") | **closed, NOT merged** | superseded by the #33xx series below |
| mlx-lm #1256 (multi-stream worker crash) | **OPEN**, updated 2026-05-07 | still crashes on 0.31.2; see below |
| mlx (latest) | **v0.31.2** (2026-04-22) | thread-safety release |
| mlx-swift (our dep) | **0.31.4** (2026-06-01) | tracks core ≥0.31.2; added an `evalLock` (#410) |
| mlx-lm (latest) | v0.31.3 (2026-04-22) | |

## What actually shipped in 0.31.2
Release notes: *"MLX can be used by multiple threads for independent computations"* via a series of merged PRs:
- `ThreadLocalStream in C++` (#3405), `Make each thread have its own default stream` (#3281)
- `Make CommandEncoder thread local` (#3348), `Merge DeviceStream into CommandEncoder` (#3264)
- `Make Scheduler::enqueue thread safe` (#3423), `Fix synchronize for ThreadLocalStream` (#3429)
- `Fix crashes in multi-threaded process teardown` (#3167), `clear_streams`/`Avoid joining threads on exit` (#3395/#3388)

So the old failure mode (the Metal command-buffer assertions, "no Stream(gpu,N) in current thread") is **largely resolved for independent per-thread computation** — each thread gets its own default stream + thread-local command encoder.

## The remaining hazard (still open)
mlx-lm **#1256 is OPEN**. Latest comment (2026-05-07, on **mlx 0.31.2 / mlx-lm 0.31.3**): `--continuous-batching` still crashes with `There is no Stream(gpu, N) in current thread` from the `BatchedEngine` worker path — and it hits **non-sliding-window models too** (Qwen2.5-Coder-32B, GLM-4.7-Flash), so it's the *batching engine's cross-thread stream handoff*, not our sliding window per se. **Takeaway: avoid mlx-lm's BatchedEngine/continuous-batching path; plain per-thread independent eval is fine.**

## Performance caveat (unchanged by the fix)
Thread-safety means it won't *crash*, not that it's *faster*. On a single GPU two concurrent forwards contend for the same compute; matmul-bound work won't parallelize across threads on one device. Concurrency mainly buys overlap of CPU-side work (tokenize/decode) with GPU forwards, and prevents one request blocking others.

## Implication for `pf serve` (corrected)
- **v1: serialize the forward** on one executor — now justified by *simplicity + single-GPU throughput-equivalence*, **not** thread-safety. For `pf`'s short single-pass forwards (~ms–tens of ms), head-of-line blocking is negligible.
- **Concurrency is now a safe upgrade**, not forbidden: a small worker pool doing *independent* per-thread eval (thread-local default stream) is supported as of 0.31.2 — use it only if profiling shows GPU idle between requests (CPU-bound tokenize/decode).
- **Avoid** mlx-lm's `BatchedEngine`/continuous-batching path (#1256 open). If batching is ever wanted, concat queued lines into one padded forward on the single executor (manual batch), not the BatchedEngine.
- **Process isolation is no longer the *only* scale-out path** — in-process threads are viable now. Still pin to mlx-swift ≥ 0.31.4 and smoke-test concurrency on our sliding-window model before relying on it.

## Sources
- [MLX v0.31.2 release notes — "used by multiple threads for independent computations"](https://github.com/ml-explore/mlx/releases/tag/v0.31.2)
- [MLX #2133 — thread-safety tracking (closed: "support in 0.31.2 now")](https://github.com/ml-explore/mlx/issues/2133)
- [MLX #3078 — concurrent independent models (closed completed)](https://github.com/ml-explore/mlx/issues/3078)
- [MLX PR #2104 — old Metal thread-safety attempt (closed, unmerged)](https://github.com/ml-explore/mlx/pull/2104)
- [mlx-lm #1256 — multi-stream worker crash (OPEN, 2026-05-07, on 0.31.2)](https://github.com/ml-explore/mlx-lm/issues/1256)
- [mlx-swift 0.31.4 release](https://github.com/ml-explore/mlx-swift/releases/tag/0.31.4)
- [mlx-lm HTTP Server (DeepWiki)](https://deepwiki.com/ml-explore/mlx-lm/3.3-http-server)
- [arXiv 2511.05502 — Production-Grade Local LLM Inference on Apple Silicon](https://arxiv.org/pdf/2511.05502)
