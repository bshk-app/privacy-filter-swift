#!/usr/bin/env bash
# Dev harness for `pf serve` startup lifecycle (Task 4, design §4) — NOT shipped. Proves the
# self-healing daemon lifecycle on a private /tmp socket:
#
#   1. already-running : daemon A is up; starting B on the SAME sock → B exits non-zero with
#                        "already running" (fail-closed: never two daemons on one socket).
#   2. auto-reclaim    : kill -9 A (leaves a stale socket file, nobody answering) → starting C
#                        on the same sock AUTO-RECLAIMS (unlink + bind) and serves — no --force,
#                        no manual cleanup (this is the crashed-instance self-heal).
#   3. --force displace: daemon D is live; starting E with --force on the same sock SIGTERMs D
#                        and takes over — E serves (the guaranteed hammer).
#   4. corrupt pidfile : daemon F is live but pf.pid is poisoned with a bogus-but-live foreign pid;
#                        --force must displace the KERNEL-ATTESTED owner (LOCAL_PEERPID) F WITHOUT
#                        killing the innocent foreign process (proves the pidfile isn't trusted for
#                        signalling — C1/I2 regression guard).
#
# Builds the product once via xcodebuild, then drives daemons with a tiny python3 frame client
# (python is fine for a dev harness — NOT in the shipped path). Prints LIFECYCLE OK / LIFECYCLE
# FAIL and exits 0/1.
#
# Model dir: $PF_MODEL (default ../models/q4-8emb, relative to pf/). Override with:
#   PF_MODEL=/path/to/model bash pf/Tests/lifecycle.sh

set -uo pipefail
cd "$(dirname "$0")/.."          # -> pf/
PF_MODEL="${PF_MODEL:-../models/q4-8emb}"
CONFIG="${CONFIG:-Debug}"
DERIVED="$PWD/.build/xcode"
BIN="$DERIVED/Build/Products/$CONFIG/pf"
SOCK="/tmp/pf-life-$$.sock"
PIDFILE="$(dirname "$SOCK")/pf.pid"   # pf writes <sockdir>/pf.pid (see Serve.swift acquireListener)
LOGDIR="$PWD/.build"
mkdir -p "$LOGDIR"

echo "building pf ($CONFIG) via xcodebuild…" >&2
xcodebuild -scheme pf -destination 'platform=macOS' \
    -derivedDataPath "$DERIVED" -configuration "$CONFIG" build >"$LOGDIR/pf-life-build.log" 2>&1 || {
    echo "build failed — tail of build log:" >&2; tail -25 "$LOGDIR/pf-life-build.log" >&2; exit 1
}
[ -x "$BIN" ] || { echo "LIFECYCLE FAIL: built binary not found at $BIN" >&2; exit 1; }

fails=0
PIDS=()
SLEEP_PID=""   # a harmless foreign process whose pid we plant into the pidfile (case 4)
trap 'for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done; kill -9 "${PIDS[@]:-}" 2>/dev/null; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; rm -f "$SOCK" "$PIDFILE"' EXIT

# ── python3 one-shot frame client: send text, return redacted body (or "ERR" on failure). ─────
read -r -d '' CLIENT <<'PY'
import socket, struct, sys
sock_path, text = sys.argv[1], sys.argv[2]
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(10); s.connect(sock_path)
    p = text.encode(); s.sendall(struct.pack(">I", len(p)) + p)
    def recvn(n):
        b=b""
        while len(b)<n:
            c=s.recv(n-len(b))
            if not c: raise SystemExit("closed")
            b+=c
        return b
    recvn(1); ln=struct.unpack(">I",recvn(4))[0]
    sys.stdout.write(recvn(ln).decode() if ln else "")
except Exception as e:
    sys.stderr.write(str(e)); sys.stdout.write("ERR")
PY
client() { python3 -c "$CLIENT" "$SOCK" "$1"; }

# A daemon serves iff a probe round-trips a redacted (secret-free) body.
serves() {
    local out; out="$(client 'key sk-proj-abc123def456 here')"
    [ "$out" != "ERR" ] && [ -n "$out" ] && ! printf '%s' "$out" | grep -qF -- "sk-proj-abc123def456"
}
# Start a daemon ($1 = log path, rest = extra flags); echo its pid; wait until it actually
# SERVES (a probe round-trips), not merely until the socket file exists — when reclaiming a
# stale socket the file is already present, so a probe is the only race-free readiness signal.
start_daemon() {
    local log="$1"; shift
    "$BIN" serve --model "$PF_MODEL" --sock "$SOCK" "$@" >"$log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 120); do          # up to ~60s for warm model load + bind
        if [ -S "$SOCK" ] && serves; then echo "$pid"; return 0; fi
        kill -0 "$pid" 2>/dev/null || { echo ""; return 1; }   # exited before serving
        sleep 0.5
    done
    echo ""; return 1
}

# ── Case 1: already-running — second start on the same sock must refuse. ──────────────────────
echo "── case 1: already-running ──" >&2
A_PID="$(start_daemon "$LOGDIR/pf-life-A.log")"
if [ -z "$A_PID" ]; then echo "  FAIL: daemon A never came up"; fails=$((fails+1)); fi
if [ -n "$A_PID" ] && serves; then echo "  ok   : daemon A up and serving (pid $A_PID)"; else echo "  FAIL: daemon A not serving"; fails=$((fails+1)); fi

# Start B on the SAME socket WITHOUT --force — must exit non-zero and say "already running".
"$BIN" serve --model "$PF_MODEL" --sock "$SOCK" >"$LOGDIR/pf-life-B.log" 2>&1
B_RC=$?
if [ "$B_RC" -ne 0 ] && grep -qi "already running" "$LOGDIR/pf-life-B.log"; then
    echo "  ok   : daemon B refused (rc=$B_RC, 'already running')"
else
    echo "  FAIL: daemon B should have refused (rc=$B_RC); log:"; sed 's/^/        /' "$LOGDIR/pf-life-B.log"; fails=$((fails+1))
fi
# A must STILL be serving (B's refusal didn't disturb it — never stomped a live daemon).
if serves; then echo "  ok   : daemon A still serving after B's refusal"; else echo "  FAIL: daemon A disturbed by B"; fails=$((fails+1)); fi

# ── Case 2: auto-reclaim — kill -9 A leaves a stale socket; C must self-heal. ─────────────────
echo "── case 2: auto-reclaim after kill -9 ──" >&2
[ -n "$A_PID" ] && kill -9 "$A_PID" 2>/dev/null
sleep 0.5
[ -e "$SOCK" ] && echo "  info : stale socket file present after kill -9 (as expected)" || echo "  info : socket already gone"
C_PID="$(start_daemon "$LOGDIR/pf-life-C.log")"   # NO --force — must auto-reclaim
if [ -z "$C_PID" ]; then
    echo "  FAIL: daemon C failed to auto-reclaim stale socket; log:"; sed 's/^/        /' "$LOGDIR/pf-life-C.log"; fails=$((fails+1))
elif serves; then
    echo "  ok   : daemon C auto-reclaimed stale socket and serves (pid $C_PID)"
    grep -qi "reclaim" "$LOGDIR/pf-life-C.log" && echo "  ok   : C logged the reclaim" || echo "  info : (no reclaim log line — bind on absent socket is also fine)"
else
    echo "  FAIL: daemon C up but not serving"; fails=$((fails+1))
fi

# ── Case 3: --force displace — E displaces the live D. ───────────────────────────────────────
echo "── case 3: --force displace ──" >&2
# D = the currently-live daemon (C). Confirm it's live, then start E --force on the same sock.
# NOTE: while E loads its model, the OLD D keeps answering probes — so "socket serves" alone
# does NOT mean E is up. The race-free readiness signal for a DISPLACE is "old D pid dead AND
# socket still serves" (E has taken over). We poll for exactly that.
D_PID="$C_PID"
if [ -n "$D_PID" ] && kill -0 "$D_PID" 2>/dev/null && serves; then
    echo "  ok   : daemon D live before displace (pid $D_PID)"
else
    echo "  FAIL: no live daemon D to displace"; fails=$((fails+1))
fi

"$BIN" serve --model "$PF_MODEL" --sock "$SOCK" --force >"$LOGDIR/pf-life-E.log" 2>&1 &
E_PID=$!
displaced=0
for _ in $(seq 1 120); do                        # up to ~60s: wait for D to die AND E to serve
    kill -0 "$E_PID" 2>/dev/null || break        # E exited before taking over → fail below
    if ! kill -0 "$D_PID" 2>/dev/null && serves; then displaced=1; break; fi
    sleep 0.5
done

if [ "$displaced" -eq 1 ] && [ "$E_PID" != "$D_PID" ]; then
    echo "  ok   : daemon E displaced D and serves (E=$E_PID, old D=$D_PID terminated)"
else
    echo "  FAIL: --force displace did not complete (E=$E_PID up=$(kill -0 "$E_PID" 2>/dev/null && echo y || echo n), D=$D_PID alive=$(kill -0 "$D_PID" 2>/dev/null && echo y || echo n))"
    sed 's/^/        /' "$LOGDIR/pf-life-E.log"; fails=$((fails+1))
fi
PIDS=("$E_PID")   # only E should remain; trap cleans it up

# ── Case 4: corrupt-pidfile — --force must displace the KERNEL-ATTESTED owner, not the pidfile. ─
# Regression for C1/I2: --force used to SIGTERM the pid read from pf.pid. If that pid is stale and
# the OS recycled it onto an innocent process, --force killed the wrong process (C1) while the real
# daemon survived → two daemons (I2). We now signal the LOCAL_PEERPID owner, so a poisoned pidfile
# must NOT cause a foreign kill, AND the real daemon must still be displaced.
echo "── case 4: corrupt pidfile (LOCAL_PEERPID is authoritative) ──" >&2
# F = the currently-live daemon (E from case 3). Confirm it's live.
F_PID="$E_PID"
if [ -n "$F_PID" ] && kill -0 "$F_PID" 2>/dev/null && serves; then
    echo "  ok   : daemon F live before corrupt-pidfile displace (pid $F_PID)"
else
    echo "  FAIL: no live daemon F to displace"; fails=$((fails+1))
fi

# Spawn a harmless long-lived foreign process and poison the pidfile with ITS pid. If --force
# trusted the pidfile it would SIGTERM this innocent process; LOCAL_PEERPID must ignore it.
sleep 600 &
SLEEP_PID=$!
echo "  info : planted foreign pid $SLEEP_PID (a live 'sleep 600') into $PIDFILE" >&2
if ! kill -0 "$SLEEP_PID" 2>/dev/null; then echo "  FAIL: foreign sleep process didn't start"; fails=$((fails+1)); fi
printf '%s\n' "$SLEEP_PID" > "$PIDFILE"   # corrupt the pidfile: bogus-but-live pid (NOT the daemon)
echo "  info : pidfile now reads '$(cat "$PIDFILE")' (daemon F is actually pid $F_PID)" >&2

# Now --force on the same socket. It must SIGTERM the attested owner (F), not the pidfile pid.
"$BIN" serve --model "$PF_MODEL" --sock "$SOCK" --force >"$LOGDIR/pf-life-G.log" 2>&1 &
G_PID=$!
displaced4=0
for _ in $(seq 1 120); do                        # up to ~60s: wait for F to die AND G to serve
    kill -0 "$G_PID" 2>/dev/null || break        # G exited before taking over → fail below
    if ! kill -0 "$F_PID" 2>/dev/null && serves; then displaced4=1; break; fi
    sleep 0.5
done

# (a) The foreign process must SURVIVE — proves the pidfile pid was NOT signalled.
if kill -0 "$SLEEP_PID" 2>/dev/null; then
    echo "  ok   : foreign pid $SLEEP_PID SURVIVED (pidfile pid was NOT killed — LOCAL_PEERPID authoritative)"
else
    echo "  FAIL: foreign pid $SLEEP_PID was killed — --force trusted the pidfile (C1 regression)"; fails=$((fails+1))
fi
# (b) The real daemon F must be displaced and G must serve.
if [ "$displaced4" -eq 1 ] && [ "$G_PID" != "$F_PID" ]; then
    echo "  ok   : daemon G displaced the REAL owner F and serves (G=$G_PID, old F=$F_PID terminated)"
else
    echo "  FAIL: --force did not displace the real daemon (G=$G_PID up=$(kill -0 "$G_PID" 2>/dev/null && echo y || echo n), F=$F_PID alive=$(kill -0 "$F_PID" 2>/dev/null && echo y || echo n))"
    sed 's/^/        /' "$LOGDIR/pf-life-G.log"; fails=$((fails+1))
fi
# Tidy the planted foreign process now (trap also covers it).
[ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; SLEEP_PID=""
PIDS=("$G_PID")   # only G should remain; trap cleans it up

echo "---"
if [ "$fails" -eq 0 ]; then
    echo "LIFECYCLE OK"; exit 0
else
    echo "LIFECYCLE FAIL ($fails check(s) failed)"; exit 1
fi
