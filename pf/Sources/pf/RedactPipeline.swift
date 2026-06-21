// RedactPipeline — the single per-line redaction core shared by the one-shot `pf`
// (PF.swift) and the resident `pf serve` (Task 3). Built ONCE with the loaded model,
// tokenizer, and derived labels/decoder choice; each line runs the SAME path:
//   tokenize → empty-ids guard → model.logits → decoder (viterbi/argmax)
//   → label/offset-count guard → bioesToSpans → Redactor.redact.
//
// Stateless across lines except through the caller's `Redactor` (passed inout), so the
// caller owns stable-token state (same value → same token) and the --only/--except/--map
// policy. Throws on any failure so the caller applies its own fail-closed policy — a line
// is never emitted raw from here.

import Foundation
import MLX
import PFCore
import PFModel

struct RedactPipeline {
    private let tok: PFTokenizer
    private let model: Model
    private let labels: [String]
    private let nCls: Int
    private let decoder: String

    init(tok: PFTokenizer, model: Model, labels: [String], decoder: String) {
        self.tok = tok
        self.model = model
        self.labels = labels
        self.nCls = labels.count
        self.decoder = decoder
    }

    /// Process a single line into its redacted form. Throws on any failure so the caller
    /// can apply the fail-closed policy (the line is never emitted raw from here).
    func redactLine(_ line: String, into redactor: inout Redactor) throws -> String {
        let (ids, offsets) = try tok.encode(line)
        // An empty/whitespace-only line tokenizes to no ids; it carries no entities and
        // running the forward on a zero-length sequence is undefined. Pass it through the
        // redactor with no spans (returns the line unchanged) — safe by construction.
        guard !ids.isEmpty else {
            return redactor.redact(line, spans: [])
        }
        let flat = model.logits(ids).asType(.float32).asArray(Float.self)   // [n*C], row-major
        // Constrained Viterbi (default) decodes the best LEGAL BIOES path → coherent, same-type
        // spans; 'argmax' is the per-token baseline (kept for A/B measurement via eval_prf.py).
        let lineLabels = decoder == "argmax"
            ? argmaxLabels(flat, nCls: nCls, labels: labels)
            : viterbiLabels(flat, nCls: nCls, labels: labels)
        // Belt-and-suspenders: labels (one per token row) and offsets (one per id) must be
        // 1:1. A model/tokenizer desync would otherwise let bioesToSpans index past offsets
        // and trap (uncatchable). Throwing here turns it into a CAUGHT fail-closed line.
        guard lineLabels.count == offsets.count else {
            throw RuntimeError("internal: \(lineLabels.count) labels vs \(offsets.count) offsets")
        }
        let spans = bioesToSpans(labels: lineLabels, offsets: offsets)
        return redactor.redact(line, spans: spans)
    }
}

/// Argmax each row of the n×C logits (flat, row-major) and map to its label.
/// Mirrors pf-parity's argmax.
func argmaxLabels(_ flat: [Float], nCls: Int, labels: [String]) -> [String] {
    guard nCls > 0 else { return [] }
    let n = flat.count / nCls
    var result = [String](); result.reserveCapacity(n)
    for i in 0..<n {
        var best = 0
        let base = i * nCls
        for c in 1..<nCls where flat[base + c] > flat[base + best] { best = c }
        result.append(labels[best])
    }
    return result
}
