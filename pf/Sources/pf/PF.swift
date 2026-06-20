// `pf` — streaming secret/PII redactor (Tasks C2–C4).
//
// Reads stdin line-by-line, runs the on-device MLX privacy-filter over each line,
// turns BIOES labels into typed spans, and replaces each hit with a stable token
// (`<SECRET_1>`). Non-span text is preserved byte-exact. Same value → same token
// within a run (one Redactor is reused for the whole stream).
//
// ── FAIL-CLOSED CONTRACT (the whole point of a redactor) ────────────────────────────
//   • Model/tokenizer load failure → exit non-zero BEFORE any stdin is read. We never
//     fall back to passthrough; a redactor that cannot redact must not emit input.
//   • A line that cannot be processed (e.g. PFTokenizerError.cannotAnchor, or any other
//     error) is NOT emitted raw. By default we print a placeholder that contains none of
//     the original line (`⟦pf:line-redacted⟧`) and log the reason to stderr — one weird
//     line never leaks and never aborts the stream. `--fail-open` opts into raw passthrough
//     for non-sensitive contexts only.
//   • `--map` writes the token→value table (RAW secrets) to a SEPARATE file, chmod 0600.
//     It is NEVER written to stdout.
//
// MLX (via PFModel) → build/run with run.sh (xcodebuild), not `swift run`.
// (@main lives here, NOT in a file literally named main.swift — Swift forbids that combo.)

import ArgumentParser
import Foundation
import MLX
import PFCore
import PFModel

/// Placeholder emitted for a line that could not be processed. Contains NONE of the
/// original line's content, so a failed line leaks nothing.
let lineRedactedPlaceholder = "\u{27E6}pf:line-redacted\u{27E7}"  // ⟦pf:line-redacted⟧

@main
struct PF: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pf",
        abstract: "Stream stdin→stdout, redacting secrets & PII with the on-device MLX model.",
        discussion: """
            Each detected secret/PII span is replaced with a stable typed token
            (e.g. <SECRET_1>); the same value maps to the same token for the whole stream.
            Fails closed: on any per-line error the original line is withheld (a placeholder
            is emitted instead) unless --fail-open is given; if the model or tokenizer cannot
            load, pf exits non-zero before reading any input.
            """)

    @Option(name: .long, help: "Directory holding model.safetensors + config.json + tokenizer.json.")
    var model = "\(NSHomeDirectory())/.pf/model"

    @Option(parsing: .upToNextOption, help: "Only redact these categories (e.g. secret private_email).")
    var only: [String] = []

    @Option(parsing: .upToNextOption, help: "Redact everything except these categories.")
    var except: [String] = []

    @Option(name: .long, help: "Dump the token→value map (RAW secrets) as JSON to this file (chmod 0600).")
    var map: String?

    @Flag(name: .long, help: "Emit the raw line on a processing error instead of withholding it (UNSAFE; opt-in).")
    var failOpen = false

    mutating func run() async throws {
        let modelDir = URL(fileURLWithPath: model)

        // ── Load model + tokenizer ONCE, before any stdin is read. ──────────────────
        // If either throws, the error propagates out of run(): ArgumentParser prints it
        // to stderr and exits non-zero. No stdout has been written → no passthrough leak.
        let pfModel: Model
        let tok: PFTokenizer
        do {
            // Quantized default (4-bit MoE + 8-bit embed, group 64) → ~870 MB shipped
            // footprint. Redaction quality is preserved (cosine ≥ 0.995 / argmax ≥ 99%
            // vs fp32; see pf-parity [q4/8] block).
            pfModel = try Model(modelDir: modelDir, qbits: 4, qgroup: 64, qembed: 8)
            tok = try await PFTokenizer(modelDir: modelDir)
        } catch {
            throw RuntimeError("failed to load model/tokenizer from \(modelDir.path): \(error)")
        }
        let labels = pfModel.hp.labels
        let nCls = labels.count

        // ── ONE Redactor for the whole stream. ──────────────────────────────────────
        // Stable tokens must persist across lines (same value → same token), so it is
        // created here and reused — never recreated per line.
        var redactor = Redactor(only: only.isEmpty ? nil : only, except: except)

        // ── Streaming loop. ─────────────────────────────────────────────────────────
        while let line = readLine(strippingNewline: true) {
            let out: String
            do {
                out = try redactLine(line, tok: tok, model: pfModel, labels: labels, nCls: nCls, &redactor)
            } catch {
                // Fail-closed: the line could not be processed. Do NOT emit it raw.
                logStderr("pf: line withheld (\(error))")
                out = failOpen ? line : lineRedactedPlaceholder
            }
            print(out)
            fflush(stdout)  // stream live: one line out per line in, not block-buffered.
        }

        // ── EOF: optionally dump the value map to a SEPARATE 0600 file. ──────────────
        if let mapPath = map {
            try writeMap(redactor.map, to: mapPath)
        }
    }

    /// Process a single line into its redacted form. Throws on any failure so the caller
    /// can apply the fail-closed policy (the line is never emitted raw from here).
    private func redactLine(
        _ line: String, tok: PFTokenizer, model: Model,
        labels: [String], nCls: Int, _ redactor: inout Redactor
    ) throws -> String {
        let (ids, offsets) = try tok.encode(line)
        // An empty/whitespace-only line tokenizes to no ids; it carries no entities and
        // running the forward on a zero-length sequence is undefined. Pass it through the
        // redactor with no spans (returns the line unchanged) — safe by construction.
        guard !ids.isEmpty else {
            return redactor.redact(line, spans: [])
        }
        let flat = model.logits(ids).asType(.float32).asArray(Float.self)   // [n*C], row-major
        let lineLabels = argmaxLabels(flat, nCls: nCls, labels: labels)
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

/// A simple error with a clean message (no Swift type noise) for the fail-closed exit path.
struct RuntimeError: Error, CustomStringConvertible {
    let message: String
    init(_ message: String) { self.message = message }
    var description: String { message }
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

/// Write the token→value map as pretty JSON to `path` with owner-only (0600) permissions.
/// The map holds RAW secret/PII values, so it goes to its own file (never stdout) and is
/// created 0600 from the start: we remove any stale file first, then `createFile` with the
/// restrictive mode in its attributes — so the secrets never momentarily sit at a looser
/// default (an atomic write + later chmod would leave that race window open).
func writeMap(_ map: [String: String], to path: String) throws {
    let data = try JSONSerialization.data(withJSONObject: map, options: [.prettyPrinted, .sortedKeys])
    let fm = FileManager.default
    if fm.fileExists(atPath: path) { try fm.removeItem(atPath: path) }
    guard fm.createFile(atPath: path, contents: data, attributes: [.posixPermissions: 0o600]) else {
        throw RuntimeError("could not write --map file at \(path)")
    }
}

func logStderr(_ s: String) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
}
