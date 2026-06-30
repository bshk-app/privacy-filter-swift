# `pf serve` Phase 1 — Implementation Plan

> **For Claude:** Execute task-by-task via **superpowers:subagent-driven-development** (one fresh
> subagent per task + code review between tasks). Each subagent follows **test-driven-development**.
> Full design context: [`2026-06-21-pf-serve-and-brew-design.md`](2026-06-21-pf-serve-and-brew-design.md).

**Goal:** Resident `pf serve` (unix socket, warm MLX model) + native `pf pull` + a build-from-source
Homebrew formula — all native Swift. Phase 2 (notarized cask + CI) is a separate plan.

**Architecture:** Daemon is transport around the unchanged redaction core. A shared `redactLine`
helper makes `pf serve` byte-identical to one-shot `pf`, so existing `pf-parity`/`leak_rate.sh`/
`eval_prf.py` validate serve for free.

**Tech stack:** Swift 6, SwiftPM, mlx-swift ≥ 0.31.4, swift-argument-parser, Foundation (URLSession,
Network/POSIX sockets). Build MLX targets via `pf/run.sh` (xcodebuild); PFCore tests via `swift test`.

**Branch:** `feat/prequant-and-eval` (has the dependent viterbi/`redactLine` code). Frequent signed commits.

---

### Task 1: Frame codec (`PFCore/Frame.swift`) — pure, TDD

**Files:** Create `pf/Sources/PFCore/Frame.swift`, `pf/Tests/PFCoreTests/FrameTests.swift`.

Length-prefixed framing (design §2): `REQ [len:u32 BE][utf8]`, `RESP [status:u8][len:u32][utf8]`.
- `encodeRequest(_ text: String) -> Data`, `encodeResponse(status: UInt8, _ text: String) -> Data`.
- A streaming `FrameReader` that accumulates bytes and yields complete frames (handles partial/
  coalesced reads — TCP/unix streams don't preserve message boundaries).
- `maxFrameSize` (default 16 MiB); a length header over the cap → throw `FrameError.oversize`.

**TDD:** round-trip encode→decode; partial feed (split a frame across 3 chunks) yields one frame;
two frames in one buffer yield two; oversize header throws; empty payload ok; status byte preserved.
Run `cd pf && swift test --filter PFCoreTests` → all green. **Commit.**

**Why pure/PFCore:** no MLX → fast `swift test`, and the codec is the protocol's SSOT.

---

### Task 2: Shared `redactLine` pipeline (refactor) — parity-preserving

**Files:** Create `pf/Sources/pf/RedactPipeline.swift`; modify `pf/Sources/pf/PF.swift`.

Extract the current `redactLine` body (tokenize → `model.logits` → `viterbiLabels` → `bioesToSpans`
→ `Redactor.redact`, incl. the empty-line + label/offset-count guards) into a reusable
`RedactPipeline` (holds `Model`, `PFTokenizer`, labels/nCls; method `redact(_ line:, into: inout Redactor)`).
One-shot `PF.run()` uses it. **No behaviour change.**

**Verify (not new unit tests — reuse the oracle):** `pf/run.sh pf-parity ../models/privacy-filter
parity-fixture.json ../models/q4-8emb` still passes; `PF_MODEL=../models/q4-8emb bash pf/Tests/e2e.sh`
still `E2E OK`. **Commit.** (Gate-accept: parity-proven refactor.)

---

### Task 3: `pf serve` MVP (`Sources/pf/Serve.swift`) — fair line-interleave

**Files:** Create `pf/Sources/pf/Serve.swift`; add the `serve` subcommand in `PF.swift`.

- Bind unix socket at `$PF_SOCK` ▸ `~/.pf/pf.sock` (`0600`, dir `0700`). (Lock/`--force` is Task 4 —
  here assume a free path; basic "exists → error" stub is fine until Task 4.)
- Accept loop: each connection = its own async `Task` with a fresh `Redactor`; `FrameReader` over the
  socket; per frame, split payload into lines.
- **Single GPU executor** = an `actor` serializing `RedactPipeline.redact`; connections submit lines
  **round-robin** (fair interleave — a big payload never starves a small one). Reply per frame:
  `[status 0][len][redacted]`; on pipeline throw → `[status 1][0][]` (fail-closed, no raw).
- Reuse `RedactPipeline` (Task 2) → **serve output ≡ one-shot** for the same text.
- Micro-batching (design §3 C) is explicitly **deferred**.

**Verify (dev harness):** add `pf/Tests/serve_eq_spawn.sh` — start serve, send framed text via a tiny
client (socat or a Swift `pf-serve-probe`), assert byte-identical to `pf` one-shot; a fairness check
(big conn A + tiny conn B → B returns promptly). **Commit.**

---

### Task 4: Lifecycle — `Lock.swift` + `--force` (design §4)

**Files:** Create `pf/Sources/pf/Lock.swift`, `pf/Tests/PFCoreTests/` lock-decision test (pure part);
wire into `Serve.swift`.

- `flock ~/.pf/pf.lock`; PING probe of existing socket (send a 0-length/handshake frame, expect a pong).
- Decision: live → `already running` (or `--force` → SIGTERM pid, take over); stale (socket silent) →
  **auto-reclaim** (unlink + bind, warn); bind-fails-after-unlink → require `--force`.
- `SIGINT/SIGTERM` → drain + unlink sock/pid. Write `pf.pid`.
- **Pure-testable piece:** factor the stale-vs-live *decision* (inputs: pingOK, pidAlive, force) into a
  pure function → unit test all branches. Socket I/O stays in `Serve`.

**Verify:** unit (decision matrix); dev harness `pf/Tests/lifecycle.sh` (double-start → `already
running`; `kill -9` → auto-reclaim; `--force` displaces). **Commit.**

---

### Task 5: `pf pull` (`Sources/pf/Pull.swift`) — canonical hub cache (design §5)

**Files:** Create `pf/Sources/pf/Pull.swift`; add `pull` subcommand; model-resolution helper used by
`pf`/`serve` (read `refs/main` → `snapshots/<commit>/q4-8emb/`).

- Base: `$HF_HUB_CACHE` ▸ `$HF_HOME/hub` ▸ `~/.cache/huggingface/hub`.
- URLSession GET `…/resolve/main/<path>` for root files + `q4-8emb/*`; from headers take
  `X-Repo-Commit` + (`X-Linked-Etag` ▸ `ETag`, unquoted); write `blobs/<etag>` (atomic), relative
  `snapshots/<commit>/<path>` symlink, `refs/main`. `Authorization: Bearer` from token file if present.
- `--variant bf16|q4-8emb` (default `q4-8emb`); progress on stderr.
- Resolution: missing model → fail-closed `model missing: run \`pf pull\``; `--model <dir>` override.

**Verify:** dev harness `reference/test_pull.sh` — run `pf pull`, assert canonical layout (`blobs/<etag>`,
snapshot symlinks resolve, `refs/main`), and `hf download … --local-dir-use-symlinks` / a python
`huggingface_hub` check sees it as cached (unification). **Commit.**

---

### Task 6: Homebrew source formula (design §6, Phase 1 track)

**Files:** `Formula/pf.rb` in the `bshk-app/tap` repo (or a `packaging/` stub here + PR to the tap).

- `depends_on xcode: [:build]`, `arch: :arm64`, `macos: ">= :sonoma"`; `url` = tagged source tarball.
- `install`: `xcodebuild -scheme pf -configuration Release …` → install `pf` + `default.metallib` +
  dylibs into `libexec`; `bin.install_symlink`. `service do run [opt_bin/"pf","serve"]; keep_alive
  crashed: true; log_path …`. `caveats`: run `pf pull`.

**Verify:** `brew install --build-from-source ./Formula/pf.rb` on this Mac; `pf --help`, `pf pull`,
`brew services start pf`, smoke a redaction. **Commit / open tap PR.**

---

### Final review
After all tasks: dispatch a code-reviewer over the full Phase-1 diff (fail-closed contract intact,
serve≡spawn, no raw leak on any error path, signatures/perms `0600`). Then **finishing-a-development-branch**.

### Out of scope (later)
Phase 2 cask + GH Actions notarization; micro-batching (§3 C); `ViterbiBias` tuning; agentvault Go client.
