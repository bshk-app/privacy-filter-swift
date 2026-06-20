import XCTest
@testable import PFCore

final class ViterbiTests: XCTestCase {
    // 9-label toy taxonomy: O + BIES×{secret,email}.
    let labels = ["O", "B-secret", "I-secret", "E-secret", "S-secret",
                  "B-email", "I-email", "E-email", "S-email"]
    // column indices
    let O = 0, Bs = 1, Is = 2, Es = 3, Ss = 4, Be = 5, Ie = 6, Ee = 7, Se = 8

    private func row(_ pairs: [(Int, Float)]) -> [Float] {
        var r = [Float](repeating: 0, count: 9)
        for (i, v) in pairs { r[i] = v }
        return r
    }
    private func decode(_ rows: [[Float]]) -> [String] {
        viterbiLabels(rows.flatMap { $0 }, nCls: 9, labels: labels)
    }

    func test_empty_input() {
        XCTAssertEqual(viterbiLabels([], nCls: 0, labels: []), [])
        XCTAssertEqual(viterbiLabels([], nCls: 9, labels: labels), [])
    }

    func test_legal_sequence_preserved() {
        // argmax is already a legal BIOES path -> Viterbi must reproduce it exactly.
        let out = decode([
            row([(O, 5)]),
            row([(Bs, 5), (Ss, 1)]),
            row([(Es, 5)]),
            row([(O, 5)]),
        ])
        XCTAssertEqual(out, ["O", "B-secret", "E-secret", "O"])
    }

    func test_illegal_O_to_I_is_repaired() {
        // token1 argmax = I-secret, illegal after O. Best LEGAL continuation is S-secret.
        let out = decode([
            row([(O, 5)]),
            row([(Is, 5), (Ss, 4)]),
        ])
        XCTAssertEqual(out, ["O", "S-secret"])
        XCTAssertNotEqual(out[1], "I-secret")
    }

    func test_mid_span_type_switch_is_repaired() {
        // argmax = [B-email, I-secret, E-email] -> fragments. Viterbi forces same type:
        // B-email must continue in email, so token1 becomes I-email.
        let out = decode([
            row([(Be, 5)]),
            row([(Is, 5), (Ie, 4)]),
            row([(Ee, 5)]),
        ])
        XCTAssertEqual(out, ["B-email", "I-email", "E-email"])
    }

    func test_unclosed_span_at_end_is_repaired() {
        // single token argmax = B-secret (a span that never closes). Legal single-token span
        // is S-secret; B is not a valid end (nor a valid lone start).
        let out = decode([row([(Bs, 5), (Ss, 4)])])
        XCTAssertEqual(out, ["S-secret"])
    }

    func test_span_entry_bias_creates_span() {
        // Borderline token: O just edges out S-secret. A positive spanEntry bias should flip
        // it into a span (recall lever).
        let flat = [row([(O, 1.0), (Ss, 0.9)])].flatMap { $0 }
        XCTAssertEqual(viterbiLabels(flat, nCls: 9, labels: labels), ["O"])
        let biased = viterbiLabels(flat, nCls: 9, labels: labels,
                                   bias: ViterbiBias(spanEntry: 0.5))
        XCTAssertEqual(biased, ["S-secret"])
    }
}
