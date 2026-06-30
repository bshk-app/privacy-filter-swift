// `pf serve` — resident daemon (design §2 protocol, §3 concurrency; Task 3 MVP).
//
// Loads the MLX model ONCE and serves many concurrent clients over a unix socket. A frame
// payload is treated as a complete stdin batch: the response equals one-shot `pf` output for
// the same input WITH THE TRAILING NEWLINE TRIMMED (one-shot `print`s a trailing newline per
// line; serve joins lines with "\n" and emits no trailing newline). This is the serve≡spawn
// invariant. It is pure transport around the unchanged redaction core (RedactPipeline).
//
// ── DESIGN (Task 3 scope) ─────────────────────────────────────────────────────────────
//   • Unix domain socket at $PF_SOCK ▸ ~/.pf/pf.sock (dir 0700, socket 0600). Low-level
//     POSIX (Foundation/Network have no clean unix-domain *server* API).
//   • Model + tokenizer load ONCE before bind/accept — fail-closed: a load error exits
//     non-zero before serving (a daemon that cannot redact must never accept clients).
//   • Single GPU executor = an `actor` (GPUExecutor) serializing ALL forwards — by choice,
//     not necessity: a single GPU yields no concurrency speedup, so serializing keeps the
//     executor simple. (MLX 0.31.2+ is in fact thread-safe for independent computation — see
//     design §3.) Connections await ONE line at a time, so the actor interleaves lines fairly:
//     a 1000-line frame on conn A never starves a 1-line frame on conn B.
//   • Each accepted connection = its own Task with its OWN fresh Redactor (stable
//     <SECRET_n> tokens are per-connection; NEVER shared across connections).
//   • Per-line fail-closed EXACTLY like one-shot: a line that can't be processed becomes
//     the placeholder (never emitted raw). --fail-open is out of scope for the serve MVP.
//
// Startup lifecycle (Task 4, design §4) lives in Lock.swift + acquireListener(): an flock around
// the start, a liveness probe of any existing socket, decideStart() (PFCore) → bind / already-
// running / displace / reclaim, a pidfile, and --force. A stale socket self-heals; --force always
// wins; two daemons never share one socket. Micro-batching (design §3 C) is deferred.
//
// SHUTDOWN: SIGINT/SIGTERM hard-exit WITHOUT draining in-flight connections (the handler only
// unlinks the socket + pidfile and _exit()s — async-signal-safe). Graceful drain (design §4)
// is deferred.

import ArgumentParser
import Darwin
import Foundation
import PFCore
import PFModel

/// The one GPU executor. An `actor` so all MLX forwards run serialized — serialized by choice
/// (single GPU → no concurrency speedup; keeps the executor simple), not because MLX is unsafe
/// (MLX 0.31.2+ is thread-safe for independent computation, see design §3) — AND so connections
/// fairly interleave: each connection awaits a single line, releasing the actor between lines so
/// other connections' lines can run.
actor GPUExecutor {
    private let pipeline: RedactPipeline

    init(pipeline: RedactPipeline) {
        self.pipeline = pipeline
    }

    /// Redact one line through the shared pipeline. Per-line fail-closed: on ANY error the
    /// line is withheld (placeholder, identical to one-shot) — never emitted raw. The
    /// caller's per-connection Redactor is updated and returned (value type → carry forward).
    ///
    /// STATUS CONTRACT (design §2): a per-line failure here yields the placeholder INSIDE a
    /// `status 0` frame — it mirrors one-shot `pf`, which also withholds the failing line and
    /// keeps producing output. `status 1` (whole-request failure) is unused in the MVP (the
    /// caller never emits it; that path is reserved for future request-level errors), and
    /// `status 2` is reserved for protocol/frame errors. So the placeholder does NOT escalate
    /// the frame status — that is intentional, not a missing error path.
    func process(_ line: String, _ redactor: Redactor) -> (out: String, redactor: Redactor) {
        var r = redactor
        let out = (try? pipeline.redactLine(line, into: &r)) ?? lineRedactedPlaceholder
        return (out, r)
    }
}

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Run a resident daemon: load the MLX model once and redact for many clients over a unix socket.",
        discussion: """
            Binds a unix domain socket (default $PF_SOCK ▸ ~/.pf/pf.sock, mode 0600). Each
            request frame ([len:u32 BE][utf8]) is processed exactly as one-shot `pf` would
            process that text on stdin and answered with [status:u8][len:u32 BE][utf8].
            Stable tokens are per-connection. Fails closed: a load error exits before binding;
            a per-line error withholds that line (never raw).
            """)

    // Shared with one-shot `pf` (SSOT): --model/--only/--except/--decoder. Splatted via
    // @OptionGroup so serve's flags parse correctly alongside the root command's options.
    @OptionGroup var opts: RedactionOptions

    @Option(name: .long, help: "Unix socket path (default $PF_SOCK ▸ ~/.pf/pf.sock).")
    var sock: String?

    @Flag(name: .long, help: "Displace a live/wedged daemon already on the socket (SIGTERM it, take over).")
    var force = false

    mutating func run() async throws {
        let sockPath = Self.resolveSockPath(sock)

        // ── Load model + tokenizer ONCE, before binding. Fail-closed: a load failure exits
        //    non-zero here, BEFORE the socket exists, so no client can ever connect to a
        //    daemon that cannot redact. ───────────────────────────────────────────────────
        let pfModel: Model
        let tok: PFTokenizer
        do {
            pfModel = try Model(modelDir: URL(fileURLWithPath: opts.model), qbits: 4, qgroup: 64, qembed: 8)
            tok = try await PFTokenizer(modelDir: URL(fileURLWithPath: opts.model))
        } catch {
            throw RuntimeError("failed to load model/tokenizer from \(opts.model): \(error)")
        }
        let pipeline = RedactPipeline(tok: tok, model: pfModel, labels: pfModel.hp.labels, decoder: opts.decoder)
        let executor = GPUExecutor(pipeline: pipeline)

        // ── Install the cleanup handler BEFORE bind so a Ctrl-C during startup can't orphan
        //    the socket file. Unlinking a not-yet-created path is harmless (the handler only
        //    unlink()s + _exit()s — async-signal-safe). The pidfile path is registered too so
        //    a clean shutdown removes sock + pid (design §4). ──────────────────────────────────
        let runtimeDir = (sockPath as NSString).deletingLastPathComponent
        installSignalCleanup(sockPath: sockPath, pidPath: "\(runtimeDir)/pf.pid")

        // ── Start lifecycle (design §4): flock, probe the existing socket, decide bind /
        //    already-running / displace / reclaim, then bind + write the pidfile. A stale
        //    socket self-heals; --force always wins; two daemons never share one socket. ──────
        let listenFD = try Self.acquireListener(sockPath: sockPath, force: force)
        logStderr("pf serve: listening on \(sockPath) (pid \(getpid()))")

        // ── Accept loop. Each connection → its own Task with a fresh Redactor. ─────────────
        let only = opts.only, except = opts.except
        await withTaskGroup(of: Void.self) { group in
            while true {
                let connFD = accept(listenFD, nil, nil)
                if connFD < 0 {
                    let e = errno
                    // A single client must NEVER be able to kill the listener: treat per-connection
                    // and resource-exhaustion errors as transient (log + continue). Only a truly
                    // broken listener fd (EBADF/EINVAL) is fatal.
                    switch e {
                    case EINTR:                       // interrupted by a signal → retry
                        continue
                    case ECONNABORTED:                // client aborted before accept → skip it
                        logStderr("pf serve: accept() ECONNABORTED — skipping")
                        continue
                    case EMFILE, ENFILE:              // fd table full → back off briefly, don't busy-spin
                        logStderr("pf serve: accept() \(String(cString: strerror(e))) — fds exhausted, backing off")
                        usleep(20_000)                // ~20 ms while fds free up
                        continue
                    case EBADF, EINVAL:               // listener fd is broken → fatal, stop accepting
                        logStderr("pf serve: accept() \(String(cString: strerror(e))) — listener unusable, stopping")
                    default:                          // unexpected → treat as transient, keep serving
                        logStderr("pf serve: accept() \(String(cString: strerror(e))) — transient, continuing")
                        continue
                    }
                    break
                }
                group.addTask {
                    await Self.serveConnection(connFD, executor: executor, only: only, except: except)
                }
            }
        }
    }

    // MARK: - Per-connection handler

    /// Drive one client connection to EOF. Owns a fresh per-connection Redactor (stable
    /// tokens live only within this stream; the Redactor is destroyed on EOF). Reads framed
    /// requests, redacts line-by-line through the shared executor (awaiting one line at a
    /// time → fair interleave), and writes framed responses.
    private static func serveConnection(
        _ fd: Int32, executor: GPUExecutor, only: [String], except: [String]
    ) async {
        defer { close(fd) }
        var redactor = Redactor(only: only.isEmpty ? nil : only, except: except)
        var reader = RequestFrameReader()
        var buf = [UInt8](repeating: 0, count: 64 * 1024)

        while true {
            let n = read(fd, &buf, buf.count)
            if n < 0 { if errno == EINTR { continue }; return }  // read error → drop conn
            if n == 0 { return }                                  // EOF → client gone
            reader.append(Data(buf[0..<n]))

            // Drain every complete frame now buffered (reads may coalesce many frames).
            while true {
                let payload: String?
                do {
                    payload = try reader.next()
                } catch {
                    // Frame-level / protocol error (e.g. oversize header) → status 2, then drop.
                    _ = Self.writeAll(fd, encodeResponse(status: 2, ""))
                    return
                }
                guard let text = payload else { break }  // need more bytes

                // Process exactly as one-shot stdin would: split into lines (keeping empties),
                // redact each through the executor (one await per line → fair interleave),
                // rejoin with "\n". This is the serve≡spawn invariant.
                let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
                var outLines = [String](); outLines.reserveCapacity(lines.count)
                for line in lines {
                    let (out, r) = await executor.process(String(line), redactor)
                    redactor = r                       // carry per-connection stable-token state
                    outLines.append(out)
                }
                let joined = outLines.joined(separator: "\n")
                if !Self.writeAll(fd, encodeResponse(status: 0, joined)) { return }
            }
        }
    }

    // MARK: - Socket setup

    /// $PF_SOCK ▸ ~/.pf/pf.sock (or an explicit --sock).
    static func resolveSockPath(_ flag: String?) -> String {
        if let flag, !flag.isEmpty { return flag }
        if let env = ProcessInfo.processInfo.environment["PF_SOCK"], !env.isEmpty { return env }
        return "\(NSHomeDirectory())/.pf/pf.sock"
    }

    /// Thrown by `bindOnly` when bind() fails with EADDRINUSE specifically — a zombie still holds
    /// the path after the unlink. The caller distinguishes this (escalate to --force / error) from
    /// any other bind failure (a hard, non-recoverable error).
    struct AddrInUse: Error {}

    /// Ensure the runtime dir exists (0700). Called before any probe/bind so the lockfile,
    /// socket, and pidfile have a home with owner-only perms (design §4: `~/.pf/` dir 0700).
    /// For an explicit --sock in a SHARED dir we don't own (/tmp, XDG runtime), tightening is
    /// best-effort — not pf's dir to lock down, and the socket itself is still 0600.
    static func prepareRuntimeDir(for sockPath: String) throws -> String {
        let dir = (sockPath as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        let defaultDir = "\(NSHomeDirectory())/.pf"
        if chmod(dir, 0o700) != 0 {
            let e = errno
            if dir == defaultDir {
                throw RuntimeError("chmod(\(dir), 0700) failed: \(String(cString: strerror(e)))")
            }
            logStderr("pf serve: chmod(\(dir), 0700) failed (\(String(cString: strerror(e)))) — not the pf runtime dir, continuing")
        }
        return dir
    }

    /// Create a unix socket at `sockPath` (0600), bind it, and listen. Returns the listening fd.
    /// Assumes the path is free (the caller has already unlinked any stale socket). A bind that
    /// fails with EADDRINUSE throws `AddrInUse` so the caller can escalate; any other failure is
    /// a hard `RuntimeError`. The umask window keeps the socket 0600 from creation (no world-
    /// connectable window), and chmod 0600 is enforced belt-and-braces (design §2).
    static func bindOnly(sockPath: String) throws -> Int32 {
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        // sun_path is a fixed C array; reserve one byte for the NUL terminator. Derive the cap
        // from the type (not a literal) so it tracks the platform's sockaddr_un definition.
        let cap = MemoryLayout.size(ofValue: addr.sun_path) - 1
        guard sockPath.utf8.count <= cap else {
            throw RuntimeError("socket path too long (\(sockPath.utf8.count) bytes, max \(cap)): \(sockPath)")
        }

        let oldMask = umask(0o077)
        defer { umask(oldMask) }

        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw RuntimeError("socket() failed: \(String(cString: strerror(errno)))") }

        _ = withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            sockPath.withCString { src in
                strncpy(UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: CChar.self), src, cap)
            }
        }
        let len = socklen_t(MemoryLayout<sockaddr_un>.size)
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(fd, $0, len) }
        }
        guard bound == 0 else {
            let e = errno
            close(fd)
            if e == EADDRINUSE { throw AddrInUse() }
            throw RuntimeError("bind(\(sockPath)) failed: \(String(cString: strerror(e)))")
        }
        guard chmod(sockPath, 0o600) == 0 else {
            close(fd); unlink(sockPath)
            throw RuntimeError("chmod(\(sockPath), 0600) failed: \(String(cString: strerror(errno)))")
        }
        guard listen(fd, 64) == 0 else {
            close(fd); unlink(sockPath)
            throw RuntimeError("listen() failed: \(String(cString: strerror(errno)))")
        }
        return fd
    }

    /// Run the full start lifecycle (design §4) under an flock and return the listening fd:
    ///   1. flock ~/.pf/pf.lock (serialize two simultaneous starts; released on exit).
    ///   2. Probe the existing socket → decideStart({responsive, force}).
    ///   3. .alreadyRunning → print + exit non-zero. .displace → SIGTERM the pid, wait, reclaim.
    ///      .reclaim/.bind → unlink any stale socket, bind. EADDRINUSE after unlink: require
    ///      --force (else error); with --force, displace then retry.
    /// On success the pidfile is written and the lock released. Throws (fail-closed) on any error.
    static func acquireListener(sockPath: String, force: Bool) throws -> Int32 {
        let runtimeDir = try prepareRuntimeDir(for: sockPath)
        let lock = try StartLock(runtimeDir: runtimeDir)
        defer { lock.release() }  // hold the lock across probe→decide→bind, release once bound

        let responsive = socketResponsive(at: sockPath)
        switch decideStart(socketResponsive: responsive, force: force) {
        case .alreadyRunning:
            let pidStr = lock.readPid().map(String.init) ?? "unknown"
            logStderr("pf serve: already running (pid \(pidStr)) — pass --force to displace")
            throw ExitCode.failure

        case .displace:
            logStderr("pf serve: --force — displacing live daemon (pid \(lock.readPid().map(String.init) ?? "unknown")) on \(sockPath)")
            if !displaceDaemon(pid: lock.readPid(), sockPath: sockPath) {
                logStderr("pf serve: WARNING — daemon still responding after SIGTERM; reclaiming anyway")
            }
            fallthrough

        case .reclaim, .bind:
            // Nobody's answering (stale/crashed or never existed) → unlink any lingering socket
            // and bind. This self-heals the crashed-daemon case with no --force needed.
            if FileManager.default.fileExists(atPath: sockPath) {
                if responsive {
                    logStderr("pf serve: reclaiming socket \(sockPath)")
                } else {
                    logStderr("pf serve: stale socket at \(sockPath) — auto-reclaiming")
                }
                unlink(sockPath)
            }
            let fd: Int32
            do {
                fd = try bindOnly(sockPath: sockPath)
            } catch is AddrInUse {
                // A zombie still holds the path even after unlink. Without --force we must not
                // guess; with --force, escalate to displace then retry once.
                guard force else {
                    throw RuntimeError("socket busy and unresponsive — rerun with --force")
                }
                _ = displaceDaemon(pid: lock.readPid(), sockPath: sockPath)
                unlink(sockPath)
                fd = try bindOnly(sockPath: sockPath)
            }
            lock.writePid()
            return fd
        }
    }

    // MARK: - I/O + lifecycle helpers

    /// Write `data` fully (handles partial writes / EINTR). Returns false on a hard error.
    private static func writeAll(_ fd: Int32, _ data: Data) -> Bool {
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Bool in
            guard let base = raw.baseAddress else { return true }
            var off = 0
            while off < raw.count {
                let w = write(fd, base + off, raw.count - off)
                if w < 0 { if errno == EINTR { continue }; return false }
                off += w
            }
            return true
        }
    }

    /// Install SIGINT/SIGTERM handlers that unlink the socket + pidfile and exit cleanly. Uses
    /// process-global paths (a C signal handler can't capture Swift state). exit() inside a
    /// handler is acceptable here: shutdown only needs to remove the socket + pidfile.
    private func installSignalCleanup(sockPath: String, pidPath: String) {
        signalSockPath = strdup(sockPath)
        signalPidPath = strdup(pidPath)
        signal(SIGINT, pfServeSignalHandler)
        signal(SIGTERM, pfServeSignalHandler)
    }
}

// MARK: - Signal handling (C-compatible, file-scope)

/// The socket path to unlink on signal. Set once at startup; read only inside the handler.
private nonisolated(unsafe) var signalSockPath: UnsafeMutablePointer<CChar>?

/// The pidfile path to unlink on signal (design §4: clean shutdown removes sock + pid). Set
/// once the pidfile is written; nil before then. Read only inside the handler.
private nonisolated(unsafe) var signalPidPath: UnsafeMutablePointer<CChar>?

/// async-signal-safe cleanup: unlink the socket + pidfile and exit. Only calls unlink()/_exit(),
/// both of which are async-signal-safe.
private func pfServeSignalHandler(_ sig: Int32) {
    if let p = signalSockPath { unlink(p) }
    if let p = signalPidPath { unlink(p) }
    _exit(0)
}
