// BIOES → spans. The ML head emits one BIES+O label per token; this collapses the
// per-token label sequence into contiguous typed spans (codepoint offsets), merging
// same-type runs and closing small gaps the model sometimes leaves between adjacent
// pieces of one entity.

// Merge same-type spans separated by ≤ this many codepoints (closes ML splits).
let SPAN_GAP_MERGE = 1

/// Strip a leading single-char BIES prefix + hyphen ("B-secret" -> "secret").
/// Labels without that exact shape (e.g. "O", or a type that itself contains "-")
/// are returned unchanged.
public func entityType(_ label: String) -> String {
    if label.count > 2,
       let i = label.firstIndex(of: "-"),
       label.distance(from: label.startIndex, to: i) == 1 {
        return String(label[label.index(after: i)...])
    }
    return label
}

public func bioesToSpans(labels: [String], offsets: [(Int, Int)]) -> [Span] {
    var spans: [Span] = []
    for (i, label) in labels.enumerated() where label != "O" {
        let ent = entityType(label)
        let (s, e) = offsets[i]
        if var last = spans.last, last.category == ent, s <= last.end + SPAN_GAP_MERGE {
            last.end = max(last.end, e)
            spans[spans.count - 1] = last
        } else {
            spans.append(Span(start: s, end: e, category: ent))
        }
    }
    return spans
}
