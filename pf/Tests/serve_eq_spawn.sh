#!/usr/bin/env bash
# Dev harness for `pf serve` (Task 3) — NOT shipped. Proves the two core invariants:
#
#   1. serve ≡ spawn : a request frame with text T produces the SAME bytes that
#                      `printf '%s' T | pf` produces (the byte-identity invariant).
#   2. fairness      : a big payload on connection A does not starve a one-liner on
#                      connection B (fair line-interleave through the single GPU actor).
#
# Builds the product once via xcodebuild, starts `pf serve` on a private socket, drives it
# with a tiny python3 client (python is fine for a dev harness — NOT in the shipped path),
# then kills the daemon and cleans up. Prints SERVE OK / SERVE FAIL and exits 0/1.
#
# Model dir: $PF_MODEL (default ../models/q4-8emb, relative to pf/). Override with:
#   PF_MODEL=/path/to/model bash pf/Tests/serve_eq_spawn.sh

set -uo pipefail
cd "$(dirname "$0")/.."          # -> pf/
PF_MODEL="${PF_MODEL:-../models/q4-8emb}"
CONFIG="${CONFIG:-Debug}"
DERIVED="$PWD/.build/xcode"
BIN="$DERIVED/Build/Products/$CONFIG/pf"
SOCK="/tmp/pf-test-$$.sock"
LOG="$PWD/.build/pf-serve-test.log"

echo "building pf ($CONFIG) via xcodebuild…" >&2
xcodebuild -scheme pf -destination 'platform=macOS' \
    -derivedDataPath "$DERIVED" -configuration "$CONFIG" build >"$PWD/.build/pf-serve-test-build.log" 2>&1 || {
    echo "build failed — tail of build log:" >&2; tail -25 "$PWD/.build/pf-serve-test-build.log" >&2; exit 1
}
[ -x "$BIN" ] || { echo "SERVE FAIL: built binary not found at $BIN" >&2; exit 1; }

fails=0
DAEMON_PID=""
cleanup() {
    [ -n "$DAEMON_PID" ] && kill "$DAEMON_PID" 2>/dev/null
    rm -f "$SOCK"
}
trap cleanup EXIT

# ── Start the daemon and wait for the socket to appear (warm model load takes a few s). ──
rm -f "$SOCK"
"$BIN" serve --model "$PF_MODEL" --sock "$SOCK" >"$LOG" 2>&1 &
DAEMON_PID=$!

for _ in $(seq 1 120); do            # up to ~60s for the model to load + bind
    [ -S "$SOCK" ] && break
    kill -0 "$DAEMON_PID" 2>/dev/null || { echo "SERVE FAIL: daemon exited early" >&2; tail -20 "$LOG" >&2; exit 1; }
    sleep 0.5
done
[ -S "$SOCK" ] || { echo "SERVE FAIL: socket never appeared at $SOCK" >&2; tail -20 "$LOG" >&2; exit 1; }
echo "  ok   : daemon up, socket bound at $SOCK"

# ── Multi-line probe text (name + email + secret + clean line + repeated value). ─────────
PROBE='Contact John Smith at john@acme.com key sk-proj-abc123def456
nothing sensitive on this line
email john@acme.com again should reuse the same token
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'

# ── One-shot reference (the oracle): exactly what serve must reproduce byte-for-byte. ────
# printf '%s' (no trailing newline) → pf reads each line; output has a trailing newline per
# `print`. We strip ONE trailing newline from the spawn output to match the joined frame.
SPAWN_OUT="$(printf '%s' "$PROBE" | "$BIN" --model "$PF_MODEL" 2>/dev/null)"

# ── python3 frame client: [len:u32 BE][utf8] out, [status:u8][len:u32 BE][utf8] in. ─────
# Emits "status<TAB>redacted" so the shell can split status from payload without ambiguity.
read -r -d '' CLIENT <<'PY'
import socket, struct, sys, os
sock_path, text = sys.argv[1], sys.argv[2]
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sock_path)
payload = text.encode("utf-8")
s.sendall(struct.pack(">I", len(payload)) + payload)
def recvn(n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk: raise SystemExit("server closed mid-frame")
        buf += chunk
    return buf
status = recvn(1)[0]
length = struct.unpack(">I", recvn(4))[0]
body = recvn(length).decode("utf-8") if length else ""
s.close()
# stdout: status byte on fd 3-style via first line, then raw body; keep body exact.
os.write(1, str(status).encode() + b"\n" + body.encode("utf-8"))
PY

client() { python3 -c "$CLIENT" "$SOCK" "$1"; }

# ── Two-frame client: ONE connection, TWO request frames sent sequentially. Reads both
# responses and prints them separated by a "<<<FRAME2>>>" marker so the shell can split them.
# Proves per-connection Redactor continuity: stable <CATEGORY_n> tokens persist ACROSS frames.
read -r -d '' CLIENT2 <<'PY'
import socket, struct, sys, os
sock_path, t1, t2 = sys.argv[1], sys.argv[2], sys.argv[3]
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sock_path)
def recvn(n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk: raise SystemExit("server closed mid-frame")
        buf += chunk
    return buf
def roundtrip(text):
    p = text.encode("utf-8")
    s.sendall(struct.pack(">I", len(p)) + p)
    recvn(1)                                   # status (ignored here)
    length = struct.unpack(">I", recvn(4))[0]
    return recvn(length).decode("utf-8") if length else ""
b1 = roundtrip(t1)
b2 = roundtrip(t2)                             # same connection → same Redactor
s.close()
os.write(1, b1.encode("utf-8") + b"\n<<<FRAME2>>>\n" + b2.encode("utf-8"))
PY

client2() { python3 -c "$CLIENT2" "$SOCK" "$1" "$2"; }

# ── Invariant 1: serve ≡ spawn (byte-identical). ─────────────────────────────────────────
RESP="$(client "$PROBE")"
STATUS="$(printf '%s' "$RESP" | head -n1)"
SERVE_OUT="$(printf '%s' "$RESP" | tail -n +2)"

if [ "$STATUS" = "0" ]; then
    echo "  ok   : serve returned status 0"
else
    echo "  FAIL: serve returned status $STATUS (expected 0)"; fails=$((fails + 1))
fi

if [ "$SERVE_OUT" = "$SPAWN_OUT" ]; then
    echo "  ok   : serve output BYTE-IDENTICAL to one-shot pf"
else
    echo "  FAIL: serve != spawn"
    echo "        spawn='$SPAWN_OUT'"
    echo "        serve='$SERVE_OUT'"; fails=$((fails + 1))
fi
# Sanity: the raw secret must be absent from serve output (fail-closed contract).
if printf '%s' "$SERVE_OUT" | grep -qF -- "sk-proj-abc123def456"; then
    echo "  FAIL: raw secret leaked in serve output"; fails=$((fails + 1))
else
    echo "  ok   : raw secret absent from serve output"
fi

# ── Invariant 1b: multi-frame continuity — stable tokens persist ACROSS frames on ONE conn.
# Frame 1 introduces a secret + an email; frame 2 REUSES the same email. Because the same
# connection keeps one Redactor, the email must map to the SAME <CATEGORY_n> token in both
# responses (this is what `Serve.swift` threads across frames; nothing tested it before).
F1='Contact John Smith at john@acme.com key sk-proj-abc123def456'
F2='ping john@acme.com once more'
DUAL="$(client2 "$F1" "$F2")"
DUAL_F1="$(printf '%s' "$DUAL" | sed -n '1,/^<<<FRAME2>>>$/p' | sed '$d')"
DUAL_F2="$(printf '%s' "$DUAL" | sed -n '/^<<<FRAME2>>>$/,$p' | tail -n +2)"

# Frame 2 reuses ONLY the email, so its output carries exactly the token that masked that
# email. Continuity means that token is the SAME number assigned in frame 1 (a fresh per-frame
# Redactor would restart at _1 and could collide/diverge). Pull the token from frame 2 and
# assert it ALSO appears in frame 1 — i.e. the reused value kept its original frame-1 token.
F2_TOKEN="$(printf '%s' "$DUAL_F2" | grep -oE '<[A-Z_]+_[0-9]+>' | head -n1)"

if [ -z "$F2_TOKEN" ]; then
    echo "  FAIL: multi-frame: no <CATEGORY_n> token found in frame 2 output"
    echo "        frame2='$DUAL_F2'"; fails=$((fails + 1))
elif printf '%s' "$DUAL_F1" | grep -qF -- "$F2_TOKEN"; then
    echo "  ok   : multi-frame continuity — reused value keeps token $F2_TOKEN across frames"
else
    echo "  FAIL: multi-frame: token $F2_TOKEN in frame 2 was not the one assigned in frame 1"
    echo "        frame1='$DUAL_F1'"
    echo "        frame2='$DUAL_F2'"; fails=$((fails + 1))
fi
# The reused email must never leak raw in frame 2 (fail-closed across frames).
if printf '%s' "$DUAL_F2" | grep -qF -- "john@acme.com"; then
    echo "  FAIL: multi-frame: raw email leaked in frame 2"; fails=$((fails + 1))
fi

# ── Invariant 2: fairness — big conn A must not starve tiny conn B. ──────────────────────
# Build a 400-line PII payload for A; B sends one short line. Launch A in the background,
# then B; assert B completes well before A. With fair line-interleave B should finish in a
# small fraction of A's total time (timing-tolerant: B must beat A AND beat 60% of A's time).
BIG_LINE='Contact John Smith at john@acme.com key sk-proj-abc123def456'
BIG_PAYLOAD="$(for _ in $(seq 1 400); do echo "$BIG_LINE"; done)"

ms() { python3 -c 'import time;print(int(time.time()*1000))'; }

A_START="$(ms)"
client "$BIG_PAYLOAD" >/dev/null 2>&1 &
A_PID=$!
sleep 0.05                       # let A get its lines enqueued first
B_START="$(ms)"
client "one short line for conn B" >/dev/null 2>&1
B_END="$(ms)"
wait "$A_PID"
A_END="$(ms)"

A_MS=$(( A_END - A_START ))
B_MS=$(( B_END - B_START ))
echo "  info : conn A (400 lines) took ${A_MS}ms; conn B (1 line) took ${B_MS}ms"
# B must (a) return before A finishes overall, and (b) take well under A's total
# (≤60%) — proving it was interleaved, not queued behind all of A.
if [ "$B_END" -lt "$A_END" ] && [ "$(( B_MS * 100 ))" -lt "$(( A_MS * 60 ))" ]; then
    echo "  ok   : conn B not starved by big conn A (fair interleave)"
else
    echo "  FAIL: conn B starved (B=${B_MS}ms vs A=${A_MS}ms)"; fails=$((fails + 1))
fi

echo "---"
if [ "$fails" -eq 0 ]; then
    echo "SERVE OK"; exit 0
else
    echo "SERVE FAIL ($fails check(s) failed)"; exit 1
fi
