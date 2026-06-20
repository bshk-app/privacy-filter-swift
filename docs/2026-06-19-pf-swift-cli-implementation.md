# `pf` Swift CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship `pf`, a native Swift CLI that streams stdin→stdout and redacts dev secrets + PII using the on-device MLX privacy-filter, replacing each hit with a stable typed token (`<SECRET_1>`), failing closed.

**Architecture:** A `PFCore` pure-Swift library (no MLX) holds the testable logic — BIOES→spans and the stable-token redactor. The `pf` executable wires the MLX model (forward already ported & parity-proven in `pf-parity`) + a swift-transformers tokenizer (with char offsets) + a line-streaming loop on top of `PFCore`. Pure logic is unit-tested with `swift test`; anything touching MLX is built/run via Xcode (`run.sh`) because SwiftPM CLI can't compile Metal shaders.

**Tech Stack:** Swift 6, mlx-swift 0.31.4 (MLX, MLXFast), swift-transformers (Tokenizers), swift-argument-parser, XCTest. Build via `xcodebuild` (see `apple/pf/run.sh`).

---

## Current state (already done — do NOT redo)

- **M0 ✅** `apple/pf/` package builds; `pf-spike` loads `model.safetensors` on Metal. Build recipe: `apple/pf/run.sh <product> [args]` (xcodebuild; `swift run` cannot build the metallib — mlx-swift README §"SwiftPM (command line) cannot build the Metal shaders").
- **M2 ✅** `Sources/pf-parity/main.swift` ports the full forward (RMSNorm, attn+sinks+bidirectional SWA mask, YaRN RoPE, dense attention, unsorted `gatherMM` MoE, 33-label head). Verified **cosine 1.0, argmax 20/20** vs `apple/pf/parity-fixture.json` (exported from `pf_mlx.py`). `pf_mlx.py` is the numerical oracle.
- Design: `docs/plans/2026-06-19-swift-redaction-cli-design.md`. Model facts: bf16 weights, vocab 200064, d=640, 8 layers, 14/2 heads ×64, I=640, 128 experts top-4, SWA radius 128, 33 BIES+O labels.

**Build/run reminders for every task below:**
- Pure `PFCore` tests (no MLX): `swift test` from `apple/pf` — fast, no Metal.
- Anything importing MLX (model, e2e): `apple/pf/run.sh <product> …` or `xcodebuild test -scheme pf-Package -destination 'platform=macOS' -derivedDataPath .build/xcode`.
- Work on a branch (repo is on `master`): `git switch -c feat/pf-swift-cli` before Task 1. Commit after every task.

---

## Phase A — Tokenizer (M1)

The model needs token ids; the redactor needs each token's **char offsets** in the original line. swift-transformers may or may not expose offsets — verify before building on it (Task A1 is the de-risk).

### Task A1: Export a tokenizer fixture from Python (oracle)

**Files:**
- Create: `apple/pf/Tests/fixtures/tok-fixture.json`
- Create: `apple/scripts/make_tok_fixture.py`

**Step 1: Write the exporter**

```python
# apple/scripts/make_tok_fixture.py
import json, sys; from pathlib import Path
from tokenizers import Tokenizer
md = Path(sys.argv[1]); out = Path(sys.argv[2])
tok = Tokenizer.from_file(str(md / "tokenizer.json"))
cases = ["Contact John Smith at john@acme.com.",
         "key sk-proj-abc123 and AKIAIOSFODNN7EXAMPLE",
         "", "   ", "no pii here at all"]
data = [{"text": t, "ids": (e := tok.encode(t)).ids, "offsets": [list(o) for o in e.offsets]} for t in cases]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=0))
print(f"wrote {out}: {len(data)} cases")
```

**Step 2: Run it**

Run: `uv run --with tokenizers python apple/scripts/make_tok_fixture.py apple/models/privacy-filter apple/pf/Tests/fixtures/tok-fixture.json`
Expected: `wrote …: 5 cases`, and the JSON has `ids` + `offsets` (char spans) per case.

**Step 3: Commit**

```bash
git add apple/scripts/make_tok_fixture.py apple/pf/Tests/fixtures/tok-fixture.json
git commit -m "test: tokenizer parity fixture (ids+offsets) from Python"
```

### Task A2: Add swift-transformers; verify it produces matching ids + offsets

**Files:**
- Modify: `apple/pf/Package.swift` (add dependency + a `tok-check` executable)
- Create: `apple/pf/Sources/tok-check/main.swift`

**Step 1: Add the dependency and a check target to `Package.swift`**

```swift
// in dependencies:
.package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.20"),
// new target (bump version if SwiftPM resolves differently):
.executableTarget(
    name: "tok-check",
    dependencies: [.product(name: "Transformers", package: "swift-transformers")],
    path: "Sources/tok-check"),
```

**Step 2: Write the check (decode fixture, compare ids + offsets)**

```swift
// apple/pf/Sources/tok-check/main.swift
import Foundation
import Tokenizers
import Hub

struct Case: Decodable { let text: String; let ids: [Int]; let offsets: [[Int]] }
let args = CommandLine.arguments            // <tokenizer.json dir> <fixture.json>
let tok = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: args[1]))
let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
var idsOK = true, offOK = true
for c in cases {
    let ids = tok.encode(text: c.text)
    if ids != c.ids { idsOK = false; print("IDS MISMATCH: \(c.text)\n  swift=\(ids)\n  py=\(c.ids)") }
    // offsets: swift-transformers offset support is unknown → probe API here
}
print("ids parity: \(idsOK ? "OK" : "FAIL")")
print("offsets: <fill in once API confirmed>")
```

**Step 3: Build + run**

Run: `apple/pf/run.sh tok-check ../models/privacy-filter Tests/fixtures/tok-fixture.json`
Expected: `ids parity: OK`. **Investigate the offsets API**: try `tok.encode` variants / `Tokenizer` protocol for an offset-returning call. Record findings inline.

**Step 4: Decision gate (offsets)**
- **If swift-transformers returns char offsets** → use them; proceed to A3.
- **If NOT** → fallback: reconstruct offsets by decoding each token id to its string and walking the original text (works for byte-level BPE; o200k tokens decode to exact substrings). Implement `offsets(for ids: [Int], in text: String)` in the tokenizer wrapper and test it against `tok-fixture.json` offsets.

**Step 5: Commit**

```bash
git add apple/pf/Package.swift apple/pf/Sources/tok-check
git commit -m "test: verify swift-transformers id parity; probe offsets API"
```

### Task A3: Tokenizer wrapper

**Files:**
- Create: `apple/pf/Sources/pf/Tokenizer.swift`
- Test: covered by A2's `tok-check` (promote it to assert offsets too)

**Step 1: Implement** a `PFTokenizer` with `func encode(_ text: String) -> (ids: [Int], offsets: [(Int,Int)])` using the A2-confirmed path (native offsets or the decode-walk fallback). Keep it the single source of tokenization.

**Step 2: Run** `tok-check` again, now asserting `offsets == fixture.offsets` for all cases. Expected: `offsets: OK`.

**Step 3: Commit** `git commit -m "feat: PFTokenizer (ids + char offsets)"`

---

## Phase B — Spans + Redactor (M3, pure Swift, strict TDD)

This is where redaction bugs live. Pure logic, no MLX → real `swift test` TDD.

### Task B1: PFCore library + Category/Span types

**Files:**
- Modify: `apple/pf/Package.swift` (add `PFCore` library target + `PFCoreTests`)
- Create: `apple/pf/Sources/PFCore/Span.swift`

**Step 1: Add targets to `Package.swift`**

```swift
.target(name: "PFCore", path: "Sources/PFCore"),
.testTarget(name: "PFCoreTests", dependencies: ["PFCore"], path: "Tests/PFCoreTests"),
```

**Step 2: Write types**

```swift
// apple/pf/Sources/PFCore/Span.swift
public struct Span: Equatable {
    public let start: Int      // char offset (inclusive)
    public var end: Int        // char offset (exclusive)
    public let category: String // entity type w/o BIES prefix, e.g. "secret"
    public init(start: Int, end: Int, category: String) {
        self.start = start; self.end = end; self.category = category
    }
}
```

**Step 3: Commit** `git commit -m "feat(PFCore): Span type"`

### Task B2: BIOES → spans (TDD)

**Files:**
- Create: `apple/pf/Sources/PFCore/Spans.swift`
- Test: `apple/pf/Tests/PFCoreTests/SpansTests.swift`

**Step 1: Write the failing tests**

```swift
// apple/pf/Tests/PFCoreTests/SpansTests.swift
import XCTest
@testable import PFCore

final class SpansTests: XCTestCase {
    // labels stripped of BIES prefix elsewhere; here we pass full labels + offsets
    func test_single_entity_span() {
        let spans = bioesToSpans(labels: ["O","B-private_person","E-private_person","O"],
                                 offsets: [(0,1),(1,5),(5,11),(11,12)])
        XCTAssertEqual(spans, [Span(start: 1, end: 11, category: "private_person")])
    }
    func test_S_singleton() {
        let spans = bioesToSpans(labels: ["S-secret"], offsets: [(0,8)])
        XCTAssertEqual(spans, [Span(start: 0, end: 8, category: "secret")])
    }
    func test_distinct_adjacent_types_not_merged() {
        let spans = bioesToSpans(labels: ["B-private_email","I-private_phone"],
                                 offsets: [(0,5),(5,10)])
        XCTAssertEqual(spans.count, 2)
    }
    func test_contiguous_same_type_merges() {
        let spans = bioesToSpans(labels: ["I-secret","I-secret","I-secret"],
                                 offsets: [(0,3),(3,6),(6,9)])
        XCTAssertEqual(spans, [Span(start: 0, end: 9, category: "secret")])
    }
    func test_small_gap_same_type_merges() { // gap ≤ GAP closes splits
        let spans = bioesToSpans(labels: ["I-secret","O","I-secret"],
                                 offsets: [(0,3),(3,4),(4,7)])
        XCTAssertEqual(spans, [Span(start: 0, end: 7, category: "secret")])
    }
    func test_O_only_no_spans() {
        XCTAssertEqual(bioesToSpans(labels: ["O","O"], offsets: [(0,1),(1,2)]), [])
    }
}
```

**Step 2: Run to verify it fails**

Run: `cd apple/pf && swift test --filter SpansTests`
Expected: FAIL — `bioesToSpans` not defined.

**Step 3: Implement**

```swift
// apple/pf/Sources/PFCore/Spans.swift
let SPAN_GAP_MERGE = 1   // merge same-type spans separated by ≤ this many chars (closes ML splits)

public func entityType(_ label: String) -> String {
    // strip a leading single-char BIES prefix + hyphen
    if label.count > 2, let i = label.firstIndex(of: "-"), label.distance(from: label.startIndex, to: i) == 1 {
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
            last.end = max(last.end, e); spans[spans.count - 1] = last
        } else {
            spans.append(Span(start: s, end: e, category: ent))
        }
    }
    return spans
}
```

**Step 4: Run to verify pass**

Run: `swift test --filter SpansTests`
Expected: PASS (6 tests).

**Step 5: Commit** `git commit -m "feat(PFCore): BIOES→spans with gap-merge (TDD)"`

### Task B3: Stable-token redactor (TDD)

**Files:**
- Create: `apple/pf/Sources/PFCore/Redactor.swift`
- Test: `apple/pf/Tests/PFCoreTests/RedactorTests.swift`

**Step 1: Write the failing tests**

```swift
// apple/pf/Tests/PFCoreTests/RedactorTests.swift
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
        let spans = [Span(start: 0, end: 5, category: "secret"), Span(start: 10, end: 15, category: "secret")]
        // "AAAAA xxxx AAAAA" — same value twice → same token
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
}
```

**Step 2: Run to verify fail**

Run: `swift test --filter RedactorTests`
Expected: FAIL — `Redactor` not defined.

**Step 3: Implement**

```swift
// apple/pf/Sources/PFCore/Redactor.swift
public struct Redactor {
    public private(set) var map: [String: String] = [:]   // token -> original value
    private var tokenFor: [String: String] = [:]          // "category\u{0}value" -> token
    private var counts: [String: Int] = [:]               // category -> next n
    private let only: Set<String>?
    private let except: Set<String>

    public init(only: [String]? = nil, except: [String] = []) {
        self.only = only.map(Set.init); self.except = Set(except)
    }

    private func enabled(_ category: String) -> Bool {
        if let only { return only.contains(category) }
        return !except.contains(category)
    }

    public mutating func redact(_ line: String, spans: [Span]) -> String {
        let chars = Array(line)                              // index by char offset
        var out = "", cursor = 0
        for span in spans.sorted(by: { $0.start < $1.start }) where enabled(span.category) {
            if span.start < cursor || span.end > chars.count { continue }  // skip overlaps/OOB
            out += String(chars[cursor..<span.start])
            let value = String(chars[span.start..<span.end])
            out += token(for: span.category, value: value)
            cursor = span.end
        }
        out += String(chars[cursor...])
        return out
    }

    private mutating func token(for category: String, value: String) -> String {
        let key = category + "\u{0}" + value
        if let t = tokenFor[key] { return t }
        let n = (counts[category] ?? 0) + 1; counts[category] = n
        let t = "<\(category.uppercased())_\(n)>"
        tokenFor[key] = t; map[t] = value
        return t
    }
}
```

**Step 4: Run to verify pass**

Run: `swift test --filter RedactorTests`
Expected: PASS (6 tests).

**Step 5: Commit** `git commit -m "feat(PFCore): stable-token redactor (TDD)"`

---

## Phase C — Model wiring + Streaming CLI (M4)

### Task C1: Refactor the forward into a reusable `Model`

**Files:**
- Create: `apple/pf/Sources/pf/Model.swift` (move forward + HP + weight-loading out of `pf-parity/main.swift`)
- Modify: `apple/pf/Package.swift` (`pf-parity` and new `pf` both depend on this code; simplest: a small `PFModel` target importing MLX, or share a file)
- Keep `pf-parity` green.

**Step 1:** Extract `HP`, `loadHP`, `yarnInvFreq`, `Layer`, `loadModel`, `forward` into `Model.swift` as a `Model` type with `func logits(ids: [Int]) -> [[Float]]` (n×C). Reuse the exact M2 code (already parity-proven — do not change the math).

**Step 2:** Re-point `pf-parity` to call `Model`. Run: `apple/pf/run.sh pf-parity ../models/privacy-filter parity-fixture.json`
Expected: still `PARITY OK` (refactor must not change numbers).

**Step 3: Commit** `git commit -m "refactor: extract Model from pf-parity (parity still 1.0)"`

### Task C2: `pf` executable — CLI flags

**Files:**
- Modify: `apple/pf/Package.swift` (add `pf` executable: MLX, MLXFast, Transformers, PFCore, ArgumentParser)
- Create: `apple/pf/Sources/pf/main.swift`

**Step 1:** Define the CLI with swift-argument-parser:

```swift
import ArgumentParser
struct PF: ParsableCommand {
    @Option var model = "\(NSHomeDirectory())/.pf/model"
    @Option(parsing: .upToNextOption) var only: [String] = []
    @Option(parsing: .upToNextOption) var except: [String] = []
    @Option var map: String?            // dump token->value JSON (0600)
    @Flag var failOpen = false
    func run() throws { /* C3 */ }
}
PF.main()
```

**Step 2:** Build: `apple/pf/run.sh pf --help`. Expected: usage prints. **Commit.**

### Task C3: Streaming loop + fail-closed

**Files:** Modify `apple/pf/Sources/pf/main.swift`

**Step 1:** Implement `run()`: load `Model` + `PFTokenizer` once; then read stdin line-by-line; per line: `(ids, offsets) = tok.encode(line)`; `labels = model.logits(ids).map { argmax → hp.labels[$0] }`; `spans = bioesToSpans(labels, offsets)`; `out = redactor.redact(line, spans:)`; print + flush. Wrap per-line work in `do/catch`; on error emit `⟦pf:error⟧` (unless `--fail-open`), never the raw line. Exit non-zero before streaming if model/tokenizer load fails. On EOF, if `--map`, write `redactor.map` JSON with `chmod 0600`.

**Step 2 (end-to-end test):**

```bash
printf 'Contact John Smith at john@acme.com key sk-proj-abc123\n' \
  | apple/pf/run.sh pf --model ../models/privacy-filter
```
Expected: a line with `<PRIVATE_PERSON_1>`, `<PRIVATE_EMAIL_1>`, `<SECRET_1>` and no raw secret/email/name.

**Step 3 (fail-closed test):** feed input with a forced error path (e.g., `--model /nonexistent` → non-zero exit, no stdout passthrough). Expected: exits non-zero before emitting input.

**Step 4: Commit** `git commit -m "feat: streaming redaction CLI (fail-closed)"`

### Task C4: End-to-end golden test (scripted)

**Files:** Create `apple/pf/Tests/e2e.sh`

**Step 1:** Script that pipes the probe inputs (AWS / `sk-` / JWT / `postgres` / name+email / SSN) through `pf` and greps that **no** raw secret/PII remains and the expected token types appear. Run via `bash apple/pf/Tests/e2e.sh`. Expected: `E2E OK`. **Commit.**

---

## Phase D — Long-context, speed, footprint, leak-rate (M5)

Optimizations; the model is already correct. Each is parity/quality-gated.

### Task D1: Windowed attention (long inputs)
Port `pf_mlx.py._attn` (blocked, 3-neighbour) into `Model`. Test: extend `parity-fixture.json` with a **> 256-token** case (regenerate via the M2 exporter) and assert `pf-parity` cosine ≥ 0.999 at that length. Commit.

### Task D2: Sorted MoE
Port the sort/unsort `gatherMM(lhsIndices:rhsIndices:sortedIndices:true)` path (`pf_mlx.py._switch`/`_moe`). Parity must stay 1.0. Commit.

### Task D3: Quantized load (4-bit MoE + 8-bit embed = 870 MB)
`quantize(...)` the expert + embedding weights at load; `gatherQMM` in `_switch`; dequantize embedding rows on lookup. Gate: argmax-agree ≥ 99% / cosine ≥ 0.995 vs `pf_mlx.py` on a multi-text fixture. Commit.

### Task D4: Leak-rate test
Reuse `apple/eval.npz` texts: run them through `pf`, assert fraction of known secret/PII spans left visible ≤ target. Commit.

---

## Testing & build summary

| what | how |
|---|---|
| `PFCore` pure logic (spans, redactor) | `cd apple/pf && swift test` (fast, no Metal) |
| model parity | `apple/pf/run.sh pf-parity …` (xcodebuild) |
| end-to-end CLI | `bash apple/pf/Tests/e2e.sh` |
| anything importing MLX in XCTest | `xcodebuild test -scheme pf-Package -destination 'platform=macOS' -derivedDataPath .build/xcode` |

**Cleanup at the end:** delete `pf-spike` (M0 throwaway); keep `pf`, `pf-parity`, `PFCore`, tests.

## Definition of done
`echo "... sk-proj-... john@x.com ..." | pf` emits stable typed tokens, no raw secrets/PII; `swift test` green; `pf-parity` cosine 1.0; e2e + leak-rate pass; README updated; built reproducibly via `run.sh`.
