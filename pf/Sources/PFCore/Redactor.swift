// Stable-token redactor. Replaces each span with a typed token (`<SECRET_1>`), giving
// the SAME value the SAME token within a run so downstream readers can correlate hits
// without ever seeing the original value. Non-span text is passed through byte-exact.
//
// ── OFFSET UNIT (load-bearing) ──────────────────────────────────────────────────────
// Spans carry CODEPOINT offsets (Unicode scalar indices == Python `str` indices), the
// same unit `PFTokenizer` emits (see Sources/pf/Tokenizer.swift). We therefore index the
// line by `Array(line.unicodeScalars)` and slice with `String.UnicodeScalarView`.
// Indexing by `Array(line)` (extended grapheme clusters) would drift by the number of
// combining marks on multi-byte input and slice the wrong substring — a redactor leak.
// All stored properties are value types (Dictionary/Set/String/optional), so a Redactor is
// safely Sendable — it can cross into the `pf serve` GPU executor actor and back by value.
public struct Redactor: Sendable {
    /// Token → original value. SENSITIVE: holds raw plaintext values; never serialize to the
    /// same sink as redacted output (e.g. the `--map` task must write this to a separate file).
    public private(set) var map: [String: String] = [:]   // token -> original value
    private var tokenFor: [String: String] = [:]          // "category\u{0}value" -> token
    private var counts: [String: Int] = [:]               // category -> next n
    private let only: Set<String>?
    private let except: Set<String>

    public init(only: [String]? = nil, except: [String] = []) {
        self.only = only.map(Set.init)
        self.except = Set(except)
    }

    private func enabled(_ category: String) -> Bool {
        if let only { return only.contains(category) }
        return !except.contains(category)
    }

    public mutating func redact(_ line: String, spans: [Span]) -> String {
        let scalars = Array(line.unicodeScalars)            // index by codepoint offset
        var out = ""
        var cursor = 0
        // Fail-CLOSED on malformed spans: clamp every enabled span into [cursor, count] so its
        // protected codepoints are ALWAYS redacted. Over-redacting on overlap is acceptable;
        // emitting protected bytes raw (the old `continue` did) is a leak — the worst redactor bug.
        for span in spans.sorted(by: { $0.start < $1.start }) where enabled(span.category) {
            let start = max(cursor, span.start)        // clamp overlap + negative start
            let end   = min(span.end, scalars.count)   // clamp OOB end
            guard start < end else { continue }         // degenerate after clamp → emit nothing raw
            out += String(String.UnicodeScalarView(scalars[cursor..<start]))
            let value = String(String.UnicodeScalarView(scalars[start..<end]))
            out += token(for: span.category, value: value)
            cursor = end
        }
        out += String(String.UnicodeScalarView(scalars[cursor...]))
        return out
    }

    private mutating func token(for category: String, value: String) -> String {
        let key = category + "\u{0}" + value
        if let t = tokenFor[key] { return t }
        let n = (counts[category] ?? 0) + 1
        counts[category] = n
        let t = "<\(category.uppercased())_\(n)>"
        tokenFor[key] = t
        map[t] = value
        return t
    }
}
