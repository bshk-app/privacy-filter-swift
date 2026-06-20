// Constrained Viterbi BIOES decoder.
//
// Per-token argmax decodes each token independently, so it can emit BIOES sequences that are
// structurally illegal — `O → I-x` (inside with no begin), `B-email → I-person` (a span that
// changes type mid-way), or a span left unclosed at end of line. The lenient `bioesToSpans`
// papers over some of these, but a type switch mid-span still fragments one entity into
// several, which shows up as spurious detections / split spans (precision loss).
//
// This decodes the single highest-scoring LEGAL BIOES path with a linear-chain Viterbi:
// emission = the per-token logits; transitions = a hard legality mask (illegal = forbidden),
// plus optional additive bias knobs (default 0 → pure constraint, i.e. the best legal path).
// Legal transitions, mirroring the model's BIOES taxonomy:
//   • O / E-* / S-*  →  O, B-*, S-*           (outside, or close-then-start-anew)
//   • B-* / I-*      →  I-x, E-x  (SAME type)  (a span must continue in its own type & close)
//   • first token starts in {O, B-*, S-*}; last token ends in {O, E-*, S-*} (no open span).
// Output is the same per-token label-string array `bioesToSpans` already consumes, so the
// span-building/redaction path is unchanged — only the label sequence is made coherent.
//
// Pure Swift (no MLX): runs on the flat [n*nCls] logits, so it lives in PFCore and is unit
// tested via `swift test` (Metal-free).

/// Additive transition bonuses (log-space). All default to 0 → the decoder returns the best
/// LEGAL path with no preference beyond the model's own emissions. Tunable later to trade
/// precision↔recall the way the upstream sequence decoder does (we don't ship its learned
/// values, so the principled default is pure constraint enforcement).
public struct ViterbiBias {
    public var spanEntry: Float   // bonus to enter a span (… → B-*/S-*); ↑ = more spans (recall)
    public var stayOutside: Float // bonus to stay background (O → O);     ↑ = fewer spans (precision)
    public init(spanEntry: Float = 0, stayOutside: Float = 0) {
        self.spanEntry = spanEntry
        self.stayOutside = stayOutside
    }
}

private enum Kind: UInt8 { case o, b, i, e, s }

private func kindOf(_ label: String) -> Kind {
    if label == "O" { return .o }
    // "B-secret" etc.: a single-char BIES prefix + '-'. Anything else is treated as Outside.
    if label.count > 2 {
        let chars = Array(label)
        if chars[1] == "-" {
            switch chars[0] {
            case "B": return .b
            case "I": return .i
            case "E": return .e
            case "S": return .s
            default: break
            }
        }
    }
    return .o
}

/// Decode the per-token label sequence with a constrained linear-chain Viterbi.
/// `flat` is the row-major [n * nCls] logits; `labels[c]` is the string for column `c`.
/// Returns one label string per token (length n). Falls back to a safe path if n or nCls is 0.
public func viterbiLabels(_ flat: [Float], nCls: Int, labels: [String],
                          bias: ViterbiBias = ViterbiBias()) -> [String] {
    guard nCls > 0 else { return [] }
    let n = flat.count / nCls
    guard n > 0 else { return [] }

    let kind = labels.map(kindOf)
    let type = labels.map(entityType)   // "" for O; e.g. "secret" for B-secret (shared with Spans.swift)
    let neg = -Float.greatestFiniteMagnitude

    let startOK = (0..<nCls).map { kind[$0] == .o || kind[$0] == .b || kind[$0] == .s }
    let endOK = (0..<nCls).map { kind[$0] == .o || kind[$0] == .e || kind[$0] == .s }

    // legal[p][c] and additive bonus[p][c] (bias only; legality is separate).
    var legal = [[Bool]](repeating: [Bool](repeating: false, count: nCls), count: nCls)
    var bonus = [[Float]](repeating: [Float](repeating: 0, count: nCls), count: nCls)
    for p in 0..<nCls {
        for c in 0..<nCls {
            let ok: Bool
            switch kind[p] {
            case .o, .e, .s:
                ok = (kind[c] == .o || kind[c] == .b || kind[c] == .s)
            case .b, .i:
                ok = (kind[c] == .i || kind[c] == .e) && type[c] == type[p]
            }
            legal[p][c] = ok
            if ok {
                if kind[c] == .b || kind[c] == .s { bonus[p][c] += bias.spanEntry }
                if kind[p] == .o && kind[c] == .o { bonus[p][c] += bias.stayOutside }
            }
        }
    }

    func emit(_ i: Int, _ c: Int) -> Float { flat[i * nCls + c] }

    var dp = [Float](repeating: neg, count: nCls)
    for c in 0..<nCls where startOK[c] {
        // Start term: apply the same entry/background bias to the first token as the
        // transition step would, so single-token lines respond to the bias too.
        var s = emit(0, c)
        if kind[c] == .b || kind[c] == .s { s += bias.spanEntry }
        if kind[c] == .o { s += bias.stayOutside }
        dp[c] = s
    }
    var back = [[Int]](repeating: [Int](repeating: -1, count: nCls), count: n)

    for i in 1..<n {
        var ndp = [Float](repeating: neg, count: nCls)
        for c in 0..<nCls {
            var bestP = -1
            var bestScore = neg
            for p in 0..<nCls where dp[p] > neg && legal[p][c] {
                let s = dp[p] + bonus[p][c]
                if s > bestScore { bestScore = s; bestP = p }
            }
            if bestP >= 0 {
                ndp[c] = bestScore + emit(i, c)
                back[i][c] = bestP
            }
        }
        dp = ndp
    }

    // Best final state among legal end states; fall back to overall best if (impossibly) none.
    var bestC = -1
    var bestScore = neg
    for c in 0..<nCls where endOK[c] && dp[c] > bestScore { bestScore = dp[c]; bestC = c }
    if bestC < 0 {
        for c in 0..<nCls where dp[c] > bestScore { bestScore = dp[c]; bestC = c }
    }
    if bestC < 0 { bestC = 0 }  // degenerate guard (all-neg): emit Outside-equivalent column 0

    var seq = [Int](repeating: 0, count: n)
    seq[n - 1] = bestC
    var i = n - 1
    while i > 0 {
        let prev = back[i][seq[i]]
        seq[i - 1] = prev >= 0 ? prev : 0
        i -= 1
    }
    return seq.map { labels[$0] }
}
