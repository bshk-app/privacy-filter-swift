import XCTest
@testable import PFCore

final class FrameTests: XCTestCase {
    // --- request round-trip -------------------------------------------------
    func test_request_round_trip() throws {
        var reader = RequestFrameReader()
        reader.append(encodeRequest("hello world"))
        XCTAssertEqual(try reader.next(), "hello world")
        XCTAssertNil(try reader.next()) // buffer drained
    }

    // --- response round-trip incl. status byte ------------------------------
    func test_response_round_trip_preserves_status() throws {
        var reader = ResponseFrameReader()
        reader.append(encodeResponse(status: 1, "withheld"))
        let frame = try reader.next()
        XCTAssertEqual(frame?.status, 1)
        XCTAssertEqual(frame?.text, "withheld")
        XCTAssertNil(try reader.next())
    }

    // --- empty payload ------------------------------------------------------
    func test_empty_payload_round_trips() throws {
        var req = RequestFrameReader()
        req.append(encodeRequest(""))
        XCTAssertEqual(try req.next(), "")

        var resp = ResponseFrameReader()
        resp.append(encodeResponse(status: 0, ""))
        let frame = try resp.next()
        XCTAssertEqual(frame?.status, 0)
        XCTAssertEqual(frame?.text, "")
    }

    // --- partial feed: one frame split across 3 appends ---------------------
    func test_partial_feed_yields_one_frame_only_when_complete() throws {
        let frame = encodeRequest("partial payload")
        let third = frame.count / 3
        let a = frame.prefix(third)
        let b = frame.prefix(2 * third).suffix(from: third)
        let c = frame.suffix(from: 2 * third)

        var reader = RequestFrameReader()
        reader.append(Data(a))
        XCTAssertNil(try reader.next())
        reader.append(Data(b))
        XCTAssertNil(try reader.next())
        reader.append(Data(c))
        XCTAssertEqual(try reader.next(), "partial payload")
        XCTAssertNil(try reader.next())
    }

    // --- coalesced: two frames in one append --------------------------------
    func test_coalesced_frames_yield_both() throws {
        var buf = encodeRequest("first")
        buf.append(encodeRequest("second"))

        var reader = RequestFrameReader()
        reader.append(buf)
        XCTAssertEqual(try reader.next(), "first")
        XCTAssertEqual(try reader.next(), "second")
        XCTAssertNil(try reader.next())
    }

    // --- leftover partial bytes after a complete frame stay buffered --------
    func test_leftover_partial_bytes_remain_buffered() throws {
        var buf = encodeRequest("complete")
        let next = encodeRequest("incomplete-tail")
        buf.append(next.prefix(3)) // header fragment only

        var reader = RequestFrameReader()
        reader.append(buf)
        XCTAssertEqual(try reader.next(), "complete")
        XCTAssertNil(try reader.next()) // tail not yet a full frame, not garbage

        reader.append(next.suffix(from: 3))
        XCTAssertEqual(try reader.next(), "incomplete-tail")
    }

    // --- oversize header throws ---------------------------------------------
    func test_oversize_header_throws() {
        // Craft a request header claiming 100 bytes against a 16-byte cap.
        var bytes = Data()
        let len = UInt32(100).bigEndian
        withUnsafeBytes(of: len) { bytes.append(contentsOf: $0) }

        var reader = RequestFrameReader(maxFrameSize: 16)
        reader.append(bytes)
        XCTAssertThrowsError(try reader.next()) { error in
            XCTAssertEqual(error as? FrameError, .oversize)
        }
    }

    func test_oversize_response_header_throws() {
        var bytes = Data([7]) // status byte
        let len = UInt32(100).bigEndian
        withUnsafeBytes(of: len) { bytes.append(contentsOf: $0) }

        var reader = ResponseFrameReader(maxFrameSize: 16)
        reader.append(bytes)
        XCTAssertThrowsError(try reader.next()) { error in
            XCTAssertEqual(error as? FrameError, .oversize)
        }
    }

    // --- UTF-8 multi-byte round-trips ---------------------------------------
    func test_utf8_multibyte_round_trips() throws {
        let s = "café — 日本語 — 🔒"
        var reader = RequestFrameReader()
        reader.append(encodeRequest(s))
        XCTAssertEqual(try reader.next(), s)
    }
}
