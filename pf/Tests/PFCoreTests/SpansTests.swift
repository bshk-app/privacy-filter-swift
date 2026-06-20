import XCTest
@testable import PFCore

final class SpansTests: XCTestCase {
    // labels stripped of BIES prefix elsewhere; here we pass full labels + offsets
    func test_single_entity_span() {
        let spans = bioesToSpans(labels: ["O", "B-private_person", "E-private_person", "O"],
                                 offsets: [(0, 1), (1, 5), (5, 11), (11, 12)])
        XCTAssertEqual(spans, [Span(start: 1, end: 11, category: "private_person")])
    }

    func test_S_singleton() {
        let spans = bioesToSpans(labels: ["S-secret"], offsets: [(0, 8)])
        XCTAssertEqual(spans, [Span(start: 0, end: 8, category: "secret")])
    }

    func test_distinct_adjacent_types_not_merged() {
        let spans = bioesToSpans(labels: ["B-private_email", "I-private_phone"],
                                 offsets: [(0, 5), (5, 10)])
        XCTAssertEqual(spans.count, 2)
    }

    func test_contiguous_same_type_merges() {
        let spans = bioesToSpans(labels: ["I-secret", "I-secret", "I-secret"],
                                 offsets: [(0, 3), (3, 6), (6, 9)])
        XCTAssertEqual(spans, [Span(start: 0, end: 9, category: "secret")])
    }

    func test_small_gap_same_type_merges() { // gap ≤ GAP closes splits
        let spans = bioesToSpans(labels: ["I-secret", "O", "I-secret"],
                                 offsets: [(0, 3), (3, 4), (4, 7)])
        XCTAssertEqual(spans, [Span(start: 0, end: 7, category: "secret")])
    }

    func test_O_only_no_spans() {
        XCTAssertEqual(bioesToSpans(labels: ["O", "O"], offsets: [(0, 1), (1, 2)]), [])
    }
}
