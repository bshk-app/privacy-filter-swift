import XCTest
@testable import PFCore

final class RedactorTests: XCTestCase {
    func test_replaces_span_with_typed_token() {
        var r = Redactor()
        let out = r.redact("hi john@x.com", spans: [Span(start: 3, end: 13, category: "private_email")])
        XCTAssertEqual(out, "hi <PRIVATE_EMAIL_1>")
    }

    func test_same_value_same_token() {
        var r = Redactor()
        // "AAAAA xxxx AAAAA" — the two "AAAAA" runs are at codepoints [0,5) and [11,16)
        // (the plan draft's [10,15) selected " AAAA" and leaked a trailing "A"). Same
        // value twice → same token.
        let spans = [Span(start: 0, end: 5, category: "secret"), Span(start: 11, end: 16, category: "secret")]
        XCTAssertEqual(r.redact("AAAAA xxxx AAAAA", spans: spans), "<SECRET_1> xxxx <SECRET_1>")
    }

    func test_distinct_values_increment() {
        var r = Redactor()
        let spans = [Span(start: 0, end: 5, category: "secret"), Span(start: 6, end: 11, category: "secret")]
        XCTAssertEqual(r.redact("AAAAA BBBBB", spans: spans), "<SECRET_1> <SECRET_2>")
    }

    func test_non_span_text_byte_exact() {
        var r = Redactor()
        let out = r.redact("a\tjohn@x.com\n", spans: [Span(start: 2, end: 12, category: "private_email")])
        XCTAssertEqual(out, "a\t<PRIVATE_EMAIL_1>\n")
    }

    func test_map_records_value_to_token() {
        var r = Redactor()
        _ = r.redact("AAAAA", spans: [Span(start: 0, end: 5, category: "secret")])
        XCTAssertEqual(r.map["<SECRET_1>"], "AAAAA")
    }

    func test_only_filter_skips_categories() {
        var r = Redactor(only: ["secret"])
        let out = r.redact("john@x.com", spans: [Span(start: 0, end: 10, category: "private_email")])
        XCTAssertEqual(out, "john@x.com")   // email not in `only` → untouched
    }

    // Correction 2: codepoint slicing must be correct on non-ASCII.
    // "café x@y.io" — é is 1 codepoint here (precomposed), so total = 11 codepoints.
    // The email "x@y.io" (6 codepoints) starts at codepoint 5, spanning [5, 11).
    // A grapheme-indexed (`Array(line)`) redactor would also count 11 here, but the
    // offset UNIT must be codepoints to match PFTokenizer (see Tokenizer.swift) — this
    // asserts the redactor slices by unicodeScalars and reproduces the email exactly.
    func test_multibyte_codepoint_slicing() {
        var r = Redactor()
        let line = "café x@y.io"
        XCTAssertEqual(line.unicodeScalars.count, 11)   // precomposed é ⇒ 11 codepoints
        let out = r.redact(line, spans: [Span(start: 5, end: 11, category: "private_email")])
        XCTAssertEqual(out, "café <PRIVATE_EMAIL_1>")
    }

    // ── Fail-closed on MALFORMED spans of ENABLED categories ────────────────────────────
    // A redactor must NEVER emit a protected codepoint raw. The old loop did `continue` on
    // overlap / OOB / negative-start, leaking the skipped span's bytes via the next prefix
    // slice or final flush. These prove every enabled span is clamped-and-redacted instead.
    // In-span content is distinctive ("leak…"/"zone") so any raw appearance is detectable;
    // it must not collide with the uppercased token text ("<SECRET_n>").

    func test_failclosed_partial_overlap() {
        var r = Redactor()
        // 12-char ASCII; spans [2,6) & [4,9) overlap at [4,6). Combined protected region [2,9)
        // = "leakzon" must be fully redacted; old bug leaked the tail [6,9) = "zon".
        let out = r.redact("ableakzoneXX",
                           spans: [Span(start: 2, end: 6, category: "secret"),
                                   Span(start: 4, end: 9, category: "secret")])
        XCTAssertEqual(out, "ab<SECRET_1><SECRET_2>eXX")
        XCTAssertFalse(out.contains("leak"))   // head not leaked
        XCTAssertFalse(out.contains("zon"))    // overlap tail not leaked
    }

    func test_failclosed_oob_end() {
        var r = Redactor()
        // 8-char string; span end 99 is OOB → clamps to count(8). Everything from 2 redacted.
        let out = r.redact("ableakzz", spans: [Span(start: 2, end: 99, category: "secret")])
        XCTAssertEqual(out, "ab<SECRET_1>")
        XCTAssertFalse(out.contains("leakzz"))
    }

    func test_failclosed_oob_then_valid() {
        var r = Redactor()
        // OOB span [2,99) clamps to [2,8) and consumes the whole tail; the later valid [4,8)
        // is fully inside it → degenerate after clamp (start==cursor==end) → emits nothing raw.
        let out = r.redact("ableakzz",
                           spans: [Span(start: 2, end: 99, category: "secret"),
                                   Span(start: 4, end: 8, category: "secret")])
        XCTAssertEqual(out, "ab<SECRET_1>")
        XCTAssertFalse(out.contains("leakzz"))
    }

    func test_failclosed_negative_start() {
        var r = Redactor()
        // start -3 clamps to 0 → [0,4) = "leak" redacted (head must not leak).
        let out = r.redact("leakxyz", spans: [Span(start: -3, end: 4, category: "secret")])
        XCTAssertEqual(out, "<SECRET_1>xyz")
        XCTAssertFalse(out.contains("leak"))
    }

    func test_failclosed_unsorted_input() {
        var r = Redactor()
        // Spans arrive out of order [5,9) before [0,4); must be sorted and BOTH redacted.
        let out = r.redact("leakxzone",
                           spans: [Span(start: 5, end: 9, category: "secret"),
                                   Span(start: 0, end: 4, category: "secret")])
        XCTAssertEqual(out, "<SECRET_1>x<SECRET_2>")
        XCTAssertFalse(out.contains("leak"))   // first span not leaked
        XCTAssertFalse(out.contains("zone"))   // second span not leaked
    }
}
