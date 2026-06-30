// Startup lifecycle I/O for `pf serve` (design §4) — the impure half of the decision in
// PFCore/LockDecision.swift. Everything here touches the filesystem / sockets / signals:
//
//   • flock(LOCK_EX) on ~/.pf/pf.lock around the whole start sequence — serializes two
//     simultaneous starts so the probe→decide→bind window can't race (the TOCTOU §4 calls out).
//   • Liveness probe: connect() to the existing socket. Success ⇒ a daemon is accepting (live);
//     ECONNREFUSED / ENOENT ⇒ stale-or-absent (reclaimable). Optionally strengthened with an
//     empty request-frame round-trip so a wedged-but-bound daemon (accepts, never replies) reads
//     as NOT responsive and gets displaced instead of blocking us forever.
//   • decideStart() (pure) maps {responsive, force} → alreadyRunning / displace / reclaim.
//   • SIGTERM target on displace = the KERNEL-ATTESTED socket owner (LOCAL_PEERPID), NOT the
//     pidfile pid — the pidfile can be stale/recycled/forged, so trusting it for signalling risks
//     killing an innocent process or letting the real daemon survive a "displace".
//   • Pidfile ~/.pf/pf.pid (0600): written after a successful bind; now COSMETIC — read only to
//     NAME the running instance ("already running (pid N)"). Never trusted for signalling.
//
// Fail-closed posture: we NEVER end up with two daemons on one socket, and the user is never
// left wedged — a stale socket self-heals (reclaim) and --force always wins (displace).

import Darwin
import Foundation
import PFCore

/// Holds the flock fd for the lifetime of the start sequence. flock is advisory and released
/// automatically when the fd closes (or the process dies), but we hold the handle explicitly so
/// the lock spans the entire probe→decide→bind critical section.
struct StartLock {
    let lockPath: String
    let pidPath: String
    private let lockFD: Int32

    /// Acquire an exclusive flock on `~/.pf/pf.lock` (created 0600). Blocks until granted, so a
    /// second `pf serve` racing the first waits here rather than interleaving probe/bind.
    init(runtimeDir: String) throws {
        self.lockPath = "\(runtimeDir)/pf.lock"
        self.pidPath = "\(runtimeDir)/pf.pid"
        let fd = open(lockPath, O_CREAT | O_RDWR, 0o600)
        guard fd >= 0 else {
            throw RuntimeError("open(\(lockPath)) for lock failed: \(String(cString: strerror(errno)))")
        }
        if flock(fd, LOCK_EX) != 0 {
            let e = errno
            close(fd)
            throw RuntimeError("flock(\(lockPath)) failed: \(String(cString: strerror(e)))")
        }
        self.lockFD = fd
    }

    /// Release the flock. Called once `acquireListener` has bound the socket — i.e. the listener
    /// is in the kernel's listen backlog. From that point a racing start's probe `connect()`
    /// succeeds (the backlog accepts the connection even before this daemon posts `accept()`), so
    /// the bound-socket-in-listen state — not the flock — is the serialization boundary that keeps
    /// two daemons off one socket. The flock only guards the earlier probe→decide→bind window.
    func release() { close(lockFD) }

    // MARK: pidfile

    /// The pid recorded in the pidfile, or nil if absent/garbage. COSMETIC only: used to NAME the
    /// running instance ("already running (pid N)") as a fallback when the kernel-attested owner is
    /// unavailable. NEVER the SIGTERM target — displace signals the LOCAL_PEERPID owner instead.
    func readPid() -> Int32? {
        guard let s = try? String(contentsOfFile: pidPath, encoding: .utf8) else { return nil }
        return Int32(s.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Write our pid to the pidfile, 0600 (created restrictively from the start — same posture
    /// as the socket and the --map file). Best-effort: a pidfile-write failure must not abort a
    /// daemon that already bound successfully; we only log it.
    func writePid() {
        let fm = FileManager.default
        if fm.fileExists(atPath: pidPath) { try? fm.removeItem(atPath: pidPath) }
        let data = Data("\(getpid())\n".utf8)
        if !fm.createFile(atPath: pidPath, contents: data, attributes: [.posixPermissions: 0o600]) {
            logStderr("pf serve: could not write pidfile \(pidPath) — continuing")
        }
    }
}

// MARK: - Liveness probe

// LOCAL_PEERPID lets the kernel attest WHO is listening on a connected unix socket, independent of
// any pidfile. SOL_LOCAL (0) / LOCAL_PEERPID (0x002) ARE surfaced by Swift's Darwin module today
// (imported from <sys/un.h>), so we use them directly below. If a future SDK ever stops exporting
// them, build with -DPF_DEFINE_LOCAL_PEER_CONSTS to splice in these ABI-stable literals — the
// values are fixed by the kernel ABI and never change.
#if PF_DEFINE_LOCAL_PEER_CONSTS
private let SOL_LOCAL: CInt = 0
private let LOCAL_PEERPID: CInt = 0x002
#endif

/// Outcome of a liveness probe: is the socket answering, and — if a daemon is bound there — WHO is
/// it (the kernel-attested owner pid via LOCAL_PEERPID)? `ownerPID` is the authoritative SIGTERM
/// target on displace: it is the pid the kernel says owns the listening socket, NOT the pidfile pid
/// (which can be stale or point at a recycled/foreign process). It is nil when nothing is bound or
/// when the kernel couldn't attest the peer.
struct ProbeResult {
    let responsive: Bool
    let ownerPID: pid_t?
}

/// Ask the kernel which process owns the listening end of a connected unix socket. Returns nil if
/// the attestation fails (e.g. peer already gone) or yields a non-positive pid. The returned pid is
/// trustworthy in a way `readPid()` is not: it cannot be stale, recycled, or forged via the pidfile.
private func peerPID(of fd: Int32) -> pid_t? {
    var pid: pid_t = 0
    var len = socklen_t(MemoryLayout<pid_t>.size)
    guard getsockopt(fd, SOL_LOCAL, LOCAL_PEERPID, &pid, &len) == 0 else { return nil }
    return pid > 1 ? pid : nil   // pid must be a real, signallable process (see kill guard, M2)
}

/// Probe an existing socket path for a *responsive* daemon (design §4 "socket PING") AND attest who
/// owns it (LOCAL_PEERPID). Returns both signals so `--force` can SIGTERM the kernel-attested owner
/// rather than trusting the pidfile.
///
/// Primary signal: a `connect()` to the unix socket succeeds → something is accepting there
/// (live). ECONNREFUSED (crashed daemon's lingering socket) / ENOENT (no socket) → not
/// responsive → reclaimable.
///
/// Nice-to-have strengthening (on by default): once connected, send an EMPTY request frame and
/// await ANY response within `timeoutMs`. A daemon that accepts but never replies (wedged) then
/// reads as NOT responsive, so --force isn't required to displace it — it self-heals via reclaim.
/// If the round-trip is inconclusive we trust "connect succeeded ⇒ responsive" (never a false
/// "dead", so we never stomp a daemon that's merely slow to answer the probe).
func socketProbe(at sockPath: String, timeoutMs: Int = 500) -> ProbeResult {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { return ProbeResult(responsive: false, ownerPID: nil) }
    defer { close(fd) }

    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let cap = MemoryLayout.size(ofValue: addr.sun_path) - 1
    guard sockPath.utf8.count <= cap else { return ProbeResult(responsive: false, ownerPID: nil) }
    _ = withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
        sockPath.withCString { src in
            strncpy(UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: CChar.self), src, cap)
        }
    }
    let len = socklen_t(MemoryLayout<sockaddr_un>.size)
    let connected = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, len) }
    }
    guard connected == 0 else { return ProbeResult(responsive: false, ownerPID: nil) }   // ECONNREFUSED/ENOENT ⇒ stale or absent

    // Connected → ask the kernel who is on the other end. This is the authoritative owner pid.
    let owner = peerPID(of: fd)

    // Strengthen with an empty-frame round-trip. Bound the wait so a wedged daemon can't hang us.
    // (tv_sec is Int, tv_usec is Int32 on Darwin → cast the microseconds component.)
    var tv = timeval(tv_sec: timeoutMs / 1000, tv_usec: Int32((timeoutMs % 1000) * 1000))
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

    let req = encodeRequest("")
    let sent = req.withUnsafeBytes { raw -> Int in
        guard let base = raw.baseAddress else { return -1 }
        return write(fd, base, raw.count)
    }
    if sent != req.count { return ProbeResult(responsive: true, ownerPID: owner) }  // couldn't probe further → connect() said live

    var byte: UInt8 = 0
    let r = read(fd, &byte, 1)
    // r > 0  → daemon answered (the status byte) → definitely live.
    // r == 0 → clean EOF: it accepted then closed without replying — treat as wedged/dead.
    // r < 0  → timeout/error: inconclusive → trust the successful connect() (responsive).
    if r == 0 { return ProbeResult(responsive: false, ownerPID: owner) }
    return ProbeResult(responsive: true, ownerPID: owner)
}

/// Convenience: just the boolean responsiveness (used by the displace poll loop, which only needs
/// to know whether the old daemon is still answering — not who it is).
func socketResponsive(at sockPath: String, timeoutMs: Int = 500) -> Bool {
    socketProbe(at: sockPath, timeoutMs: timeoutMs).responsive
}

// MARK: - Displace a live daemon (--force)

/// SIGTERM the KERNEL-ATTESTED socket owner (`pid`, from LOCAL_PEERPID — NOT the pidfile) and wait
/// (poll) until the socket stops responding, up to `timeoutMs`. Returns true if the socket went
/// quiet (safe to reclaim+bind), false if it's still answering after the timeout (the caller
/// fails closed — refuses to bind over a live daemon). A nil pid is treated as "already gone".
///
/// Signalling the attested owner (not `readPid()`) closes two bugs at once: a stale pidfile whose
/// pid the OS recycled can no longer cause us to SIGTERM an innocent process (C1), and we always
/// kill the REAL daemon even when the pidfile is missing/wrong, so we never end up rebinding while
/// the old daemon is still alive (I2).
func displaceDaemon(pid: pid_t?, sockPath: String, timeoutMs: Int = 3000) -> Bool {
    if let p = pid {
        // `pid > 1` is LOAD-BEARING (M2): a non-positive pid makes kill() target a process GROUP —
        // kill(0, …) hits our whole group, kill(-pgid, …) hits another group — which must never
        // happen. peerPID() already filters ≤1, but we re-assert it at the syscall as defence in
        // depth. Also skip our own pid (we are about to become the new owner; never SIGTERM self).
        if p > 1 && p != getpid() {
            _ = kill(p, SIGTERM)
        }
    }
    // Poll the socket until it goes silent (the old daemon's cleanup unlinks it / stops accepting).
    let deadline = Date().addingTimeInterval(Double(timeoutMs) / 1000)
    while Date() < deadline {
        if !socketResponsive(at: sockPath, timeoutMs: 200) { return true }
        usleep(100_000)  // 100 ms between polls
    }
    return !socketResponsive(at: sockPath, timeoutMs: 200)
}
