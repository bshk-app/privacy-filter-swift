// Start-of-`pf serve` decision (design §4 lifecycle) — the pure, unit-testable core.
//
// The daemon's "is another instance already here?" judgement is factored out of the socket
// I/O so `swift test` (Metal-free PFCore) can exercise every branch. The single source of
// truth is the *socket*, NOT file presence: a crashed daemon leaves a dangling socket file
// that no longer answers, so `socketResponsive` (a successful connect — see `Lock.swift`) is
// what distinguishes a live instance from stale junk.

/// What `pf serve` should do when (re)starting, given a liveness probe of the existing socket.
public enum StartDecision: Equatable {
    case bind            // unreachable in practice from decideStart (folded into .reclaim); kept for clarity/future
    case alreadyRunning  // a live daemon answered and no --force → refuse (exit non-zero)
    case displace        // a live daemon answered and --force → SIGTERM it, then take over
    case reclaim         // socket silent (stale) OR absent → unlink-if-present + bind (self-heal)
}

/// Decide how to start. Inputs are deliberately minimal (design §4 "pure-testable piece"):
///   - `socketResponsive`: did a connect()/PING to the existing socket succeed? (false ⇒ no
///     socket at all OR a crashed daemon's lingering-but-dead socket — both reclaimable).
///   - `force`: was `--force` passed (the guaranteed hammer to displace a live/wedged daemon)?
///
/// Rules (design §4):
///   responsive && !force → .alreadyRunning   (someone's there; refuse without --force)
///   responsive &&  force → .displace          (someone's there; --force says take over)
///   !responsive          → .reclaim           (nobody's answering; unlink stale + bind, self-heal)
public func decideStart(socketResponsive: Bool, force: Bool) -> StartDecision {
    if socketResponsive {
        return force ? .displace : .alreadyRunning
    }
    return .reclaim
}
