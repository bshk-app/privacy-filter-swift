// PFTokenizer — the single source of tokenization for `pf`.
//
// Promoted from the Task A2 `tok-check` spike, hardened per code review (A3):
//   I1 — offsets are reconstructed from the tokenizer's OWN freshly-emitted ids, never
//        from a fixture (see `encode(_:)`: it calls `encode` then walks those exact ids).
//   I2 — offsets are indexed by CODEPOINTS (`text.unicodeScalars`), not by Character
//        (extended grapheme clusters). The Python `tokenizers` library reports offsets as
//        Python `str` indices, which are codepoints; a grapheme walk drifts by the number
//        of combining marks. The fixture's multi-byte case ("café ❤️ résumé 日本語 …",
//        28 codepoints / 24 graphemes) exercises exactly this.
//   M3 — the decode-walk FAILS CLOSED. If a decoded token cannot be anchored at/after the
//        cursor, we throw `PFTokenizerError.cannotAnchor` instead of forward-scanning,
//        guessing, or emitting a zero-width span. A wrong offset in a redactor is a leak.
//
// ── OFFSET UNIT (load-bearing; the Task B3 redactor MUST slice by the SAME unit) ────────
//   Offsets are CODEPOINT spans: (start, end) are indices into `Array(text.unicodeScalars)`
//   — i.e. Unicode scalar values, equivalent to Python `str` indices. Empirically confirmed
//   against the Python oracle: for "café ❤️ résumé 日本語 x@y.io" the final offset end is 28,
//   which equals the codepoint count (28) and NOT the UTF-8 byte count (41) or the grapheme
//   count (24). To slice the original text by an offset (s, e), use:
//       let scalars = Array(text.unicodeScalars); String(String.UnicodeScalarView(scalars[s..<e]))
//   Do NOT use `Array(text)[s..<e]` (that is grapheme-based and will corrupt multi-byte text).
//
// swift-transformers 1.3.x exposes no native per-token offsets (the `Tokenizer` protocol
// has only tokenize/encode/decode/convert*), so offsets are reconstructed by decode-walk.

import Foundation
import Tokenizers
import Hub

public enum PFTokenizerError: Error, CustomStringConvertible {
    /// A decoded token could not be located at/after the running cursor while
    /// reconstructing offsets. We refuse to guess (fail closed) — a misplaced span
    /// in a redactor leaks the very data we are meant to remove. The associated values
    /// help diagnose which token/position failed.
    case cannotAnchor(tokenIndex: Int, id: Int, piece: String, cursor: Int)

    public var description: String {
        switch self {
        case let .cannotAnchor(tokenIndex, id, piece, cursor):
            return "PFTokenizer: cannot anchor token #\(tokenIndex) (id=\(id), "
                + "decoded=\(String(reflecting: piece))) at/after codepoint \(cursor); "
                + "refusing to guess offset (fail closed)"
        }
    }
}

public struct PFTokenizer {
    private let tokenizer: Tokenizer

    /// Load the tokenizer from a model directory containing `tokenizer.json`
    /// (and optionally `config.json` / `tokenizer_config.json`).
    public init(modelDir: URL) async throws {
        self.tokenizer = try await Self.load(modelDir: modelDir)
    }

    /// Encode `text` to ids and reconstruct each token's CODEPOINT offsets in `text`.
    ///
    /// - ids come from `encode(text:addSpecialTokens:false)`.
    /// - offsets are reconstructed from THOSE ids (I1) by walking `text.unicodeScalars`
    ///   (I2) with a moving cursor, throwing `PFTokenizerError.cannotAnchor` on any token
    ///   that cannot be placed at/after the cursor (M3).
    public func encode(_ text: String) throws -> (ids: [Int], offsets: [(Int, Int)]) {
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        let offsets = try self.offsets(for: ids, in: text)
        return (ids, offsets)
    }

    // MARK: - Offset reconstruction (decode-walk over codepoints)

    /// Reconstruct codepoint offsets for `ids` within `text`. Fails closed.
    ///
    /// o200k is byte-level BPE: a token typically decodes to an exact substring of the
    /// source (leading-space tokens decode WITH their leading space, e.g. id -> " John").
    /// We decode each token individually and anchor it at the running cursor. The fixture
    /// offsets include any leading space, so an anchored substring match reproduces them.
    private func offsets(for ids: [Int], in text: String) throws -> [(Int, Int)] {
        // Index by Unicode scalar (codepoint) == Python `str` index. NOT Character.
        let scalars = Array(text.unicodeScalars)
        var result: [(Int, Int)] = []
        result.reserveCapacity(ids.count)
        var cursor = 0

        for (tokenIndex, id) in ids.enumerated() {
            let piece = tokenizer.decode(tokens: [id], skipSpecialTokens: false)
            let needle = Array(piece.unicodeScalars)

            // A token that decodes to nothing carries no source extent. We cannot prove
            // where it sits, so rather than fabricate a (possibly wrong) zero-width span
            // we fail closed — same principle as a non-anchorable piece.
            if needle.isEmpty {
                throw PFTokenizerError.cannotAnchor(
                    tokenIndex: tokenIndex, id: id, piece: piece, cursor: cursor)
            }

            guard let (start, end) = Self.anchor(needle, in: scalars, from: cursor) else {
                // Could not place this token at/after the cursor. DO NOT forward-scan or
                // emit a zero-width guess — a wrong offset here is a leak. Fail closed.
                throw PFTokenizerError.cannotAnchor(
                    tokenIndex: tokenIndex, id: id, piece: piece, cursor: cursor)
            }
            result.append((start, end))
            cursor = end
        }
        return result
    }

    /// Anchor `needle` in `hay` at/after `from`, returning its (start, end) codepoint span,
    /// or `nil` if it cannot be placed.
    ///
    /// Deliberately conservative (M3): we only accept a match (1) exactly at the cursor, or
    /// (2) after skipping leading whitespace that BPE may have folded into the token, or
    /// (3) when the token's own leading space was dropped by the source. We do NOT scan
    /// arbitrarily far forward looking for "some" occurrence — that is the guessing behavior
    /// the review flagged. If none of the anchored attempts match, return nil → caller throws.
    private static func anchor(
        _ needle: [Unicode.Scalar], in hay: [Unicode.Scalar], from: Int
    ) -> (Int, Int)? {
        func matchAt(_ i: Int, _ n: ArraySlice<Unicode.Scalar>) -> Bool {
            guard i >= 0, i + n.count <= hay.count else { return false }
            var hi = i
            for c in n {
                if hay[hi] != c { return false }
                hi += 1
            }
            return true
        }

        // 1) exact, anchored at the cursor (the overwhelmingly common case).
        if matchAt(from, needle[...]) { return (from, from + needle.count) }

        // 2) the source has leading whitespace before this token's content; skip it and
        //    anchor there (still no arbitrary forward scan — only whitespace is skipped).
        var i = from
        while i < hay.count, isWhitespace(hay[i]) { i += 1 }
        if i != from, matchAt(i, needle[...]) { return (i, i + needle.count) }

        // 3) the decoded token itself starts with a space the source did not have; drop
        //    that single leading space and re-anchor at the cursor (and past whitespace).
        if needle.first == " " {
            let trimmed = needle.dropFirst()
            if !trimmed.isEmpty {
                if matchAt(from, trimmed) { return (from, from + trimmed.count) }
                if i != from, matchAt(i, trimmed) { return (i, i + trimmed.count) }
            }
        }
        return nil
    }

    private static func isWhitespace(_ s: Unicode.Scalar) -> Bool {
        s == " " || s == "\t" || s == "\n" || s == "\r"
    }

    // MARK: - Load (ported from the A2 spike)

    /// `AutoTokenizer.from(modelFolder:)` reads tokenizer_config.json (it can synthesize a
    /// fallback from config.json's model_type). This model dir has no tokenizer_config.json,
    /// so try the convenience path first; on failure, build `PreTrainedTokenizer` directly
    /// from tokenizer.json, seeding config.json and injecting a generic fast tokenizer class.
    private static func load(modelDir: URL) async throws -> Tokenizer {
        do {
            return try await AutoTokenizer.from(modelFolder: modelDir)
        } catch {
            let msg = "PFTokenizer: AutoTokenizer.from(modelFolder:) failed (\(error)); "
                + "loading tokenizer.json directly\n"
            FileHandle.standardError.write(Data(msg.utf8))
            let tokenizerData = Config(
                try loadDict(modelDir.appendingPathComponent("tokenizer.json")))
            var cfgDict = (try? loadDict(modelDir.appendingPathComponent("config.json"))) ?? [:]
            cfgDict["tokenizer_class"] = "PreTrainedTokenizerFast"
            return try PreTrainedTokenizer(
                tokenizerConfig: Config(cfgDict), tokenizerData: tokenizerData, strict: false)
        }
    }

    private static func loadDict(_ url: URL) throws -> [NSString: Any] {
        let data = try Data(contentsOf: url)
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [NSString: Any] else {
            throw NSError(domain: "PFTokenizer", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "not a JSON object: \(url.lastPathComponent)"])
        }
        return dict
    }
}
