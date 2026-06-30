// Startup lifecycle I/O for `pf serve` (design §4) — the impure half of the decision in
// PFCore/LockDecision.swift. Everything here touches the filesystem / sockets / signals:
//
//   • flock(LOCK_EX) on ~/.pf/pf.lock around the whole start sequence — serializes two
//     simultaneous starts so the probe→decide→bind window can't race (the TOCTOU §4 calls out).
//   • Liveness probe: connect() to the existing socket. Success ⇒ a daemon is accepting (live);
//     ECONNREFUSED / ENOENT ⇒ stale-or-absent (reclaimable). Optionally strengthened with an
//     empty request-frame round-trip so a wedged-but-bound daemon (accepts, never replies) reads
//     as NOT responsive and gets displaced instead of blocking us forever.
//   • decideStart() (pure) maps {responsive, force} → bind / alreadyRunning / displace / reclaim.
//   • Pidfile ~/.pf/pf.pid (0600): written after a successful bind, read to identify whom to
//     SIGTERM on displace / whom to name in the "already running" message.
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

    /// Release the flock. Called once the daemon is bound and serving (the start sequence is
    /// over — concurrent re-starts from here on are handled by the socket probe, not the lock).
    func release() { close(lockFD) }

    // MARK: pidfile

    /// The pid recorded in the pidfile, or nil if absent/garbage. Used to name the running
    /// instance ("already running (pid N)") and to pick the SIGTERM target on displace.
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

/// Probe an existing socket path for a *responsive* daemon (design §4 "socket PING").
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
func socketResponsive(at sockPath: String, timeoutMs: Int = 500) -> Bool {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { return false }
    defer { close(fd) }

    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let cap = MemoryLayout.size(ofValue: addr.sun_path) - 1
    guard sockPath.utf8.count <= cap else { return false }
    _ = withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
        sockPath.withCString { src in
            strncpy(UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: CChar.self), src, cap)
        }
    }
    let len = socklen_t(MemoryLayout<sockaddr_un>.size)
    let connected = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, len) }
    }
    guard connected == 0 else { return false }   // ECONNREFUSED/ENOENT ⇒ stale or absent

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
    if sent != req.count { return true }          // couldn't probe further → connect() said live

    var byte: UInt8 = 0
    let r = read(fd, &byte, 1)
    // r > 0  → daemon answered (the status byte) → definitely live.
    // r == 0 → clean EOF: it accepted then closed without replying — treat as wedged/dead.
    // r < 0  → timeout/error: inconclusive → trust the successful connect() (responsive).
    if r == 0 { return false }
    return true
}

// MARK: - Displace a live daemon (--force)

/// SIGTERM the recorded pid and wait (poll) until the socket stops responding, up to `timeoutMs`.
/// Returns true if the socket went quiet (safe to reclaim+bind), false if it's still answering
/// after the timeout (the caller escalates). A nil/dead pid is treated as "already gone".
func displaceDaemon(pid: Int32?, sockPath: String, timeoutMs: Int = 3000) -> Bool {
    if let p = pid, p > 1 {
        _ = kill(p, SIGTERM)
    }
    // Poll the socket until it goes silent (the old daemon's cleanup unlinks it / stops accepting).
    let deadline = Date().addingTimeInterval(Double(timeoutMs) / 1000)
    while Date() < deadline {
        if !socketResponsive(at: sockPath, timeoutMs: 200) { return true }
        usleep(100_000)  // 100 ms between polls
    }
    return !socketResponsive(at: sockPath, timeoutMs: 200)
}
