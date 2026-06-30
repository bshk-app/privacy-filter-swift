import XCTest
@testable import PFCore

// Exhaustive decision matrix for `decideStart` (design §4 lifecycle). The pure core is what
// keeps the socket-I/O in Serve/Lock thin: every start branch is proven here, Metal-free.
final class LockDecisionTests: XCTestCase {
    // responsive + no force → refuse: a live daemon owns the socket.
    func test_responsive_noForce_alreadyRunning() {
        XCTAssertEqual(decideStart(socketResponsive: true, force: false), .alreadyRunning)
    }

    // responsive + force → displace: --force is the hammer to take over a live daemon.
    func test_responsive_force_displace() {
        XCTAssertEqual(decideStart(socketResponsive: true, force: true), .displace)
    }

    // not responsive (no socket OR stale/crashed) + no force → reclaim: nobody's there, self-heal.
    func test_unresponsive_noForce_reclaim() {
        XCTAssertEqual(decideStart(socketResponsive: false, force: false), .reclaim)
    }

    // not responsive + force → still reclaim: --force changes nothing when no one answers.
    func test_unresponsive_force_reclaim() {
        XCTAssertEqual(decideStart(socketResponsive: false, force: true), .reclaim)
    }
}
