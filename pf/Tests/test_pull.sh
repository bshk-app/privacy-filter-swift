#!/usr/bin/env bash
#
# Dev harness for `pf pull` (Task 5) — NOT shipped. Proves `pf pull` writes the CANONICAL
# huggingface_hub cache layout (blobs/snapshots/refs) so the model dedups with — and is
# visible to — python `huggingface_hub`.
#
# Metadata-ONLY pull (no 870 MB safetensors): we restrict the download with --include so the
# whole test moves ~27 MB (config.json + tokenizer.json), exercising both a small git-SHA1
# blob (config.json) and an LFS SHA256 blob (tokenizer.json) — the two etag paths.
#
# Asserts, under an ISOLATED cache (--cache /tmp/…):
#   1. blobs/<etag> exists (content-addressed),
#   2. snapshots/<commit>/q4-8emb/config.json is a SYMLINK that resolves to a real file,
#   3. refs/main contains the 40-char commit sha,
#   4. CROSS-COMPAT: python huggingface_hub's try_to_load_from_cache() sees the file as CACHED
#      (this is the real proof the layout IS the huggingface_hub format — python is allowed
#       here because this is a dev harness, NOT the shipped code path).
#
# Prints `PULL OK` / `PULL FAIL`, exits 0/1, cleans up the temp cache.

set -euo pipefail
cd "$(dirname "$0")/.."   # pf/

REPO="beshkenadze/privacy-filter-mlx"
REPO_DIR_NAME="models--beshkenadze--privacy-filter-mlx"
CACHE="/tmp/pf-pulltest-$$"
CONFIG="${CONFIG:-Debug}"
fails=0
note() { printf '  %-5s %s\n' "$1" "$2"; }
fail() { note "FAIL" "$1"; fails=$((fails+1)); }
ok()   { note "ok" "$1"; }

cleanup() { rm -rf "$CACHE"; }
trap cleanup EXIT

# ── Build once (xcodebuild → Metal), then run the BUILT binary directly. ─────────────────
echo "building pf ($CONFIG) via run.sh…" >&2
BIN="$PWD/.build/xcode/Build/Products/$CONFIG/pf"
# run.sh builds then execs; we just want the build, so build with a harmless --help invocation.
CONFIG="$CONFIG" ./run.sh pf --help >/dev/null 2>&1 || true
[ -x "$BIN" ] || { echo "PULL FAIL (binary not built at $BIN)"; exit 1; }

# ── Metadata-only pull into an isolated cache. ───────────────────────────────────────────
echo "pulling metadata (config.json + tokenizer.json) → $CACHE …" >&2
if ! "$BIN" pull --variant q4-8emb \
        --include config.json --include tokenizer.json \
        --cache "$CACHE" 2>&1; then
    echo "PULL FAIL (pf pull exited non-zero)"; exit 1
fi

REPO_DIR="$CACHE/$REPO_DIR_NAME"

# ── 1. blobs/<etag> exists. ──────────────────────────────────────────────────────────────
blob_count=$(find "$REPO_DIR/blobs" -type f 2>/dev/null | wc -l | tr -d ' ')
if [ "$blob_count" -ge 1 ]; then ok "blobs/ has $blob_count content-addressed blob(s)";
else fail "no blobs written under $REPO_DIR/blobs"; fi

# ── 2. snapshots/<commit>/q4-8emb/config.json is a symlink resolving to a real file. ─────
COMMIT="$(cat "$REPO_DIR/refs/main" 2>/dev/null || true)"
SNAP_CONFIG="$REPO_DIR/snapshots/$COMMIT/q4-8emb/config.json"
if [ -L "$SNAP_CONFIG" ]; then
    ok "snapshots/$COMMIT/q4-8emb/config.json is a symlink ($(readlink "$SNAP_CONFIG"))"
    if [ -f "$SNAP_CONFIG" ]; then ok "  symlink resolves to a real file ($(wc -c < "$SNAP_CONFIG" | tr -d ' ') bytes)";
    else fail "symlink does NOT resolve to a real file (dangling)"; fi
else
    fail "snapshots config.json is not a symlink"
fi

# ── 3. refs/main holds a 40-char commit sha. ─────────────────────────────────────────────
if [ "${#COMMIT}" -eq 40 ]; then ok "refs/main = $COMMIT (40 chars)";
else fail "refs/main is not a 40-char sha (got '${COMMIT}')"; fi

# ── 4. CROSS-COMPAT: python huggingface_hub sees the file as CACHED. ─────────────────────
echo "cross-compat: asking python huggingface_hub if it sees the cache…" >&2
HF_STATUS="$(C="$CACHE" uv run --with huggingface_hub python -c \
  "from huggingface_hub import try_to_load_from_cache; import os; \
p=try_to_load_from_cache('$REPO','q4-8emb/config.json', cache_dir=os.environ['C']); \
print('CACHED' if p else 'MISS')" 2>/dev/null || echo "ERROR")"
if [ "$HF_STATUS" = "CACHED" ]; then ok "python huggingface_hub: CACHED (layout is canonical)";
else fail "python huggingface_hub: $HF_STATUS (layout NOT recognised as cached)"; fi

echo "---"
if [ "$fails" -eq 0 ]; then echo "PULL OK"; exit 0
else echo "PULL FAIL ($fails check(s) failed)"; exit 1; fi
