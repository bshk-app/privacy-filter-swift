# `pf serve` (resident daemon) + Homebrew distribution — Design

**Date:** 2026-06-21  ·  **Status:** validated (brainstorm)  ·  **Repo:** privacy-filter-swift

**Goal:** Add a resident `pf serve` mode (keeping one-shot `pf` as an option) so the MLX
model loads once and serves many clients (notably agentvault) over a local socket, and ship
`pf` as a brew-installed binary — entirely native Swift (no SH/PY in the shipped path).

**Non-goals:** porting the dev oracle/eval (`reference/*.py|*.sh`) to Swift — they stay as
dev tooling, excluded from brew. The shipped `pf` binary is 100% Swift.

---

## 1. Architecture & components

`pf` stays a single binary with ArgumentParser subcommands; the model loads once in `serve`.

| Subcommand | Role |
|---|---|
| `pf` | one-shot stdin→stdout (current behaviour) — the "spawn" option |
| `pf serve` | resident daemon: unix socket, warm model |
| `pf pull` | native HF download into the canonical hub cache |

**New Swift files**
- `Sources/PFCore/Frame.swift` — length-prefixed frame codec (pure, unit-tested).
- `Sources/pf/Serve.swift` — daemon: socket, accept loop, scheduler.
- `Sources/pf/Lock.swift` — pidfile + socket probe + `--force`.
- `Sources/pf/Pull.swift` — native canonical-hub downloader (URLSession).

**Key refactor (DRY/SSOT):** extract the current `redactLine` (tokenize → `model.logits` →
viterbi → `bioesToSpans` → `Redactor.redact`) into one shared helper used by both one-shot and
serve. **serve is byte-identical to spawn**, so existing `pf-parity` / `leak_rate.sh` /
`eval_prf.py` cover serve for free. The daemon is only transport around an unchanged core.

---

## 2. Wire protocol

- **Transport:** unix socket `~/.pf/pf.sock`, mode `0600` (owner-only; no network exposure).
- **Frame:** `REQ [len:u32 BE][utf8]` → `RESP [status:u8][len:u32][redacted utf8]`.
  - `status`: `0`=ok, `1`=whole-request failure (client treats as full withhold), `2`=proto/frame-error.
  - Max frame cap (e.g. 16 MB); oversize → `status 2` (OOM guard).
  - **Per-line failures are NOT request failures:** a line that can't be processed emits the
    placeholder INSIDE a `status 0` frame — mirroring one-shot `pf`, which withholds the failing
    line and keeps producing output. `status 1` is reserved for whole-request failures (none in
    the MVP — the MVP never emits it); `status 2` is reserved for protocol/frame errors.
- **Semantics = stdin batch:** the frame payload is processed exactly as `pf` would process
  that text on stdin (split into lines, `redactLine` each, rejoin with `\n`).
- **🔑 Per-connection `Redactor`:** stable `<SECRET_1>` tokens live within one client's stream
  (like "one Redactor per stream" in one-shot). **Never shared across connections** — else
  tokens / secret map leak between sessions. Destroyed on connection EOF.
- **No `--map` over the wire:** the daemon never returns raw values, only redacted text.

---

## 3. Concurrency & fairness (agent-friendly)

**Constraint (researched):** a single GPU can't be parallelised — serial-vs-concurrent yields
the same throughput. So the goal is *fairness* (no agent blocks another) + CPU/GPU overlap, via
a scheduler — not GPU threads.

> **MLX thread-safety note (verified 2026-06-20):** thread-safety for *independent* multi-thread
> computation shipped in **MLX 0.31.2** (issues #2133/#3078 closed *completed*); mlx-swift 0.31.4
> includes it. So serialization is a *simplicity/throughput* choice, not a safety mandate. BUT
> the `BatchedEngine`/continuous-batching path is still broken (mlx-lm #1256 open) — **do not use
> it**; batch manually if ever needed. Pin mlx-swift ≥ 0.31.4.

- **Accept concurrently:** each connection = its own `Task`; reads frames, splits to lines,
  enqueues `(connId, lineId, text, reply)` into a central fair queue.
- **One GPU executor (actor):** pulls round-robin across connections → a big payload from A
  never blocks a one-liner from B (interleave at line granularity).
- **CPU/GPU overlap:** tokenize (per-conn) and Viterbi+redact (CPU pool) overlap the GPU forward.
- **MVP = B (fair line-interleave). Target = C (cross-connection micro-batch):** coalesce up to
  K queued lines from *different* connections into one padded `[B, maxLen]` forward (windowed
  attention already handles variable length; pad rows masked). Self-tuning: 1 agent → batch=1.

```
conn A ─┐ tokenize(CPU)            ┌─► Viterbi+redact(CPU pool) ─► reply A
conn B ─┼─► [fair queue] ─► GPU executor (1 padded forward) ─┼─► reply B
conn C ─┘    round-robin                                      └─► reply C
```

---

## 4. Lifecycle (`~/.pf/` = runtime state only; weights live in the HF cache)

State in `~/.pf/` (dir `0700`): `pf.sock` (`0600`), `pf.pid`, `pf.lock`, opt. `pf.log`.
**No weights here** (those are in the HF cache, §5).

**Start (source of truth = socket PING, not file presence):**
1. `flock ~/.pf/pf.lock` — kill the TOCTOU race between two simultaneous starts.
2. PING existing `pf.sock`:
   - **live daemon answers** → without `--force`: exit `already running (pid N)`; with `--force`:
     `SIGTERM` the pid, then take over.
   - **socket present but silent = stale** (crashed instance) → **auto-reclaim** (unlink + bind,
     warn). No `--force` needed — nobody's there. *This self-heals the hung-lock case.*
   - **no socket** → bind.
3. Write pidfile, bind, `chmod 0600`. If bind still fails (zombie holds it) → require `--force`.

`--force` = the guaranteed hammer (displace a live/wedged daemon, clear junk). Stale self-heals.

**Shutdown:** `SIGINT/SIGTERM` → drain in-flight → unlink sock+pid → clean exit.

**launchd / brew services:** `service do run [opt_bin/"pf","serve"]; keep_alive crashed: true;
log_path …`. Restart-on-crash, but **not** on a clean error exit (so "model missing" doesn't
hot-loop). After a crash, restart finds the stale socket → auto-reclaim → comes up.

---

## 5. Model cache & `pf pull` (canonical HF hub layout, native Swift)

**Decision:** write the **canonical `huggingface_hub` cache** so the model sits alongside all the
user's other models and is visible to `hf`/python (true unification). HubApi's flat
`~/Documents/huggingface/models/...` layout was rejected.

- **Base:** `$HF_HUB_CACHE` ▸ `$HF_HOME/hub` ▸ `~/.cache/huggingface/hub`.
- **`pf pull` (URLSession, no SH/PY/HubApi):** for each file (`config.json`,
  `model.safetensors`, `tokenizer.json` at root + `q4-8emb/*`) GET
  `https://huggingface.co/beshkenadze/privacy-filter-mlx/resolve/main/<path>`; from response
  headers take `X-Repo-Commit` (commit) and `X-Linked-Etag` ▸ `ETag` (blob name — SHA256 for LFS,
  git-SHA1 for small, **exactly like huggingface_hub** so blobs dedup with the python cache).
  Write `blobs/<etag>` (atomic temp→rename, skip if present), `snapshots/<commit>/<path>` as a
  **relative symlink** to the blob, `refs/main` = commit. Auth `Authorization: Bearer` from
  `~/.cache/huggingface/token` if present. Progress on stderr; resume via Content-Length.
- **Resolve:** `pf`/`pf serve` read `refs/main` → `snapshots/<commit>/q4-8emb/`. Missing →
  fail-closed `model missing: run \`pf pull\``. `--model <dir>` overrides to any local dir.

---

## 6. Distribution (two tracks — validated against Homebrew docs)

Homebrew's own terminology: **formula = built from source; cask = pre-compiled binaries built
and signed by upstream; bottle = Homebrew-CI prebuilt from a source formula.** Policy: "Binary-
only formulae should go in homebrew/cask." So:

**Phase 1 — formula, build-from-source (ship first, fastest):** in `beshkenadze/tap`.
`depends_on xcode: [:build]` (metallib needs full Xcode), `arch: :arm64`, `macos: ">= :sonoma"`.
`install`: `xcodebuild -scheme pf -configuration Release …`; install `pf` + `default.metallib` +
dylibs into `libexec`; `bin.install_symlink`. `service do … pf serve …`. caveats: `pf pull`.
Local arm64 build is ad-hoc signed → runs. Cons: needs Xcode + multi-min build.

**Phase 2 — cask, Developer ID + notarized (best UX; we have a Developer ID):**
GH Actions (macos arm64): `xcodebuild` Release → bundle `pf`+`default.metallib`+dylibs with
**`@loader_path` rpaths** (relocatable, no path rewriting) → `codesign --options runtime -s
"Developer ID Application: …"` → `notarytool submit --wait` → `stapler staple` → tar.gz →
GitHub Release + sha256. Cask installs the notarized binary as-is (`binary` stanza), no build.

**Bottles dropped:** Homebrew relocates + ad-hoc re-signs bottles → would strip our Developer ID
signature/notarization. With a Developer ID, the cask is the correct prebuilt channel.

Tag releases `pf-vX.Y.Z` (versioned tarball) for both tracks.

---

## 7. Testing

**Swift unit (`swift test`, Metal-free):** `Frame` (encode/decode, status, oversize, partial
reads); `Lock` (stale-vs-live decision on a fake socket/pid); Viterbi (already 24 PFCore tests).

**Dev harnesses (`reference/`, not shipped):**
- **serve ≡ spawn:** start serve, send a frame, assert byte-identical to one-shot `pf`.
- **fairness:** big payload on conn A + tiny on conn B → B not starved.
- **lifecycle:** double-start → `already running`; `kill -9` → auto-reclaim; `--force` displaces.
- **pull:** canonical layout written (`blobs/<etag>`, snapshot symlinks, `refs/main`); `hf`/python
  sees it as cached (unification cross-check).
- Existing `pf-parity` / `leak_rate.sh` / `eval_prf.py` cover serve via the byte-identity invariant.

---

## 8. agentvault client (Go)

- Thin client to `~/.pf/pf.sock` (or `PF_SOCK`): write `[len][text]`, read `[status][len][redacted]`.
- **Insertion point:** in `av run`'s output-masking writer — **exact masker first**
  (`{{AV:NAME}}`), **then** pipe through the pf client. Gated by `agentvault.yaml: redact: ml` /
  `--redact-pii` (default off).
- **One connection per `av run`** → per-connection Redactor = stable tokens within that command.
- **Fail-closed:** client error/timeout or `status != ok` → withhold/drop, never raw. If
  `redact: ml` is explicitly requested but the socket is absent → loud fail, not silent passthrough.
- agentvault **does not own** `pf`: `pf serve` is its own brew service; agentvault just connects.

---

## Open questions / future
- Tune `ViterbiBias` (spanEntry/stayOutside) against `eval_prf` to recover the ~1–2 pt recall dip.
- Micro-batching (§3 C) — implement after the fair-interleave MVP if profiling shows GPU idle.
- Scale-out beyond one GPU: process-isolation (N `pf serve` workers) — MLX-blessed, composes with
  the standalone-daemon model. Not needed now.
- `pf pull` variant flag (`--variant bf16|q4-8emb`), default `q4-8emb`.
