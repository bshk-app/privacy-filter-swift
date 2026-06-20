// Span — a contiguous redaction target in a single line of input.
//
// Offsets are CODEPOINT indices (Unicode scalar indices == Python `str` indices),
// matching the unit emitted by `PFTokenizer` (see Sources/pf/Tokenizer.swift, the
// "OFFSET UNIT" note). The Task B3 redactor slices the source line with these same
// codepoint indices — never grapheme (`Array(line)`) indices, which drift on
// combining marks and corrupt multi-byte text.
public struct Span: Equatable {
    public let start: Int       // codepoint offset (inclusive)
    public var end: Int         // codepoint offset (exclusive)
    public let category: String // entity type without BIES prefix, e.g. "secret"

    public init(start: Int, end: Int, category: String) {
        self.start = start
        self.end = end
        self.category = category
    }
}
