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
import PFCore
import PFModel

/// Placeholder emitted for a line that could not be processed. Contains NONE of the
/// original line's content, so a failed line leaks nothing.
let lineRedactedPlaceholder = "\u{27E6}pf:line-redacted\u{27E7}"  // ⟦pf:line-redacted⟧

/// Redaction options shared by one-shot `pf` and `pf serve` (SSOT). Declared ONCE and
/// splatted into each command via `@OptionGroup` — this is the idiomatic ArgumentParser
/// way to share flags across a root command and its subcommands without the option-shadowing
/// that arises when both declare identically-named options independently.
struct RedactionOptions: ParsableArguments {
    @Option(name: .long, help: "Directory holding model.safetensors + config.json + tokenizer.json.")
    var model = "\(NSHomeDirectory())/.pf/model"

    @Option(parsing: .upToNextOption, help: "Only redact these categories (e.g. secret private_email).")
    var only: [String] = []

    @Option(parsing: .upToNextOption, help: "Redact everything except these categories.")
    var except: [String] = []

    @Option(name: .long, help: "Label decoder: 'viterbi' (constrained BIOES, default) or 'argmax' (per-token).")
    var decoder = "viterbi"
}

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

            Run `pf serve` for a resident daemon (warm model, unix socket, many clients);
            with no subcommand `pf` is the one-shot stdin→stdout filter described above.
            """,
        // `serve` is a subcommand; with NO subcommand the root's own run() (one-shot
        // stdin→stdout) executes — so existing `printf … | pf` behaviour is unchanged.
        subcommands: [Serve.self])

    @OptionGroup var opts: RedactionOptions

    @Option(name: .long, help: "Dump the token→value map (RAW secrets) as JSON to this file (chmod 0600).")
    var map: String?

    @Flag(name: .long, help: "Emit the raw line on a processing error instead of withholding it (UNSAFE; opt-in).")
    var failOpen = false

    mutating func run() async throws {
        let modelDir = URL(fileURLWithPath: opts.model)

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

        // ── ONE pipeline + ONE Redactor for the whole stream. ───────────────────────
        // The pipeline is the per-line core shared with `pf serve`; the Redactor carries
        // stable-token state (same value → same token) across lines, so both are created
        // here and reused — never recreated per line.
        let pipeline = RedactPipeline(tok: tok, model: pfModel, labels: labels, decoder: opts.decoder)
        var redactor = Redactor(only: opts.only.isEmpty ? nil : opts.only, except: opts.except)

        // ── Streaming loop. ─────────────────────────────────────────────────────────
        while let line = readLine(strippingNewline: true) {
            let out: String
            do {
                out = try pipeline.redactLine(line, into: &redactor)
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
}

/// A simple error with a clean message (no Swift type noise) for the fail-closed exit path.
struct RuntimeError: Error, CustomStringConvertible {
    let message: String
    init(_ message: String) { self.message = message }
    var description: String { message }
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
