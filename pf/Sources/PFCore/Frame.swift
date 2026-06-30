// Length-prefixed frame codec for `pf serve` (design §2 — the wire-protocol SSOT).
//
//   REQUEST  [len: UInt32 BE][payload: len bytes UTF-8]
//   RESPONSE [status: UInt8][len: UInt32 BE][payload: len bytes UTF-8]
//     status: 0 = ok, 1 = request-fail (withhold), 2 = proto-error.
//
// Unix/TCP streams do NOT preserve message boundaries, so the readers accumulate
// bytes and only yield a frame once it is fully buffered (`next()` -> nil means
// "need more bytes", not an error). A length header above `maxFrameSize` is an
// OOM guard and throws `FrameError.oversize`.
import Foundation

/// Default frame-size cap (16 MiB). A length header exceeding this throws.
public let defaultMaxFrameSize = 16 * 1024 * 1024

public enum FrameError: Error {
    case oversize
}

/// Encode a request frame: `[len: UInt32 BE][utf8]`.
///
/// Precondition: the UTF-8 payload byte count must be `<= defaultMaxFrameSize`.
/// Unreachable in practice — payloads are single lines of redacted text, far
/// under 16 MiB. The decode path guards oversize symmetrically; encode documents
/// the precondition rather than adding a throwing branch (keeps the API non-throwing).
public func encodeRequest(_ text: String) -> Data {
    let payload = Data(text.utf8)
    var frame = Data(capacity: 4 + payload.count)
    appendLength(UInt32(payload.count), to: &frame)
    frame.append(payload)
    return frame
}

/// Encode a response frame: `[status: UInt8][len: UInt32 BE][utf8]`.
///
/// Precondition: the UTF-8 payload byte count must be `<= defaultMaxFrameSize`.
/// Unreachable in practice — payloads are single lines of redacted text, far
/// under 16 MiB. The decode path guards oversize symmetrically; encode documents
/// the precondition rather than adding a throwing branch (keeps the API non-throwing).
public func encodeResponse(status: UInt8, _ text: String) -> Data {
    let payload = Data(text.utf8)
    var frame = Data(capacity: 5 + payload.count)
    frame.append(status)
    appendLength(UInt32(payload.count), to: &frame)
    frame.append(payload)
    return frame
}

/// Streaming reader for request frames. Handles partial and coalesced reads.
public struct RequestFrameReader {
    private var buffer = FrameBuffer()
    private let maxFrameSize: Int

    public init(maxFrameSize: Int = defaultMaxFrameSize) {
        self.maxFrameSize = maxFrameSize
    }

    /// Feed freshly read bytes; may complete zero or more frames.
    public mutating func append(_ data: Data) { buffer.append(data) }

    /// Return the next complete frame's payload (consuming its bytes), or `nil`
    /// if a full frame is not yet buffered. Throws `.oversize` on a bad header.
    public mutating func next() throws -> String? {
        try buffer.next(headerExtra: 0, maxFrameSize: maxFrameSize).map { $0.text }
    }
}

/// Streaming reader for response frames. Handles partial and coalesced reads.
public struct ResponseFrameReader {
    private var buffer = FrameBuffer()
    private let maxFrameSize: Int

    public init(maxFrameSize: Int = defaultMaxFrameSize) {
        self.maxFrameSize = maxFrameSize
    }

    /// Feed freshly read bytes; may complete zero or more frames.
    public mutating func append(_ data: Data) { buffer.append(data) }

    /// Return the next complete frame `(status, text)` (consuming its bytes), or
    /// `nil` if not yet buffered. Throws `.oversize` on a bad header.
    public mutating func next() throws -> (status: UInt8, text: String)? {
        try buffer.next(headerExtra: 1, maxFrameSize: maxFrameSize)
    }
}

// MARK: - internals

/// Append a UInt32 in big-endian order.
private func appendLength(_ value: UInt32, to data: inout Data) {
    withUnsafeBytes(of: value.bigEndian) { data.append(contentsOf: $0) }
}

/// Shared accumulate-and-consume buffer for both frame directions. The only
/// difference between them is an optional leading status byte (`headerExtra`).
private struct FrameBuffer {
    private var bytes = Data()

    mutating func append(_ data: Data) { bytes.append(data) }

    /// Decode one frame if fully buffered. `headerExtra` is the count of fixed
    /// bytes preceding the UInt32 length (0 for requests, 1 status byte for
    /// responses). Consumes the frame's bytes on success.
    mutating func next(headerExtra: Int, maxFrameSize: Int) throws -> (status: UInt8, text: String)? {
        let headerSize = headerExtra + 4
        guard bytes.count >= headerSize else { return nil }

        // Re-base to 0 — slices from prior consumes carry a non-zero startIndex.
        let base = bytes.startIndex
        // Raw status passthrough: the codec does NOT range-validate status (0/1/2);
        // interpreting/validating it is the daemon/client's job (Serve/agentvault).
        let status: UInt8 = headerExtra == 1 ? bytes[base] : 0
        let lenStart = base + headerExtra
        let length = UInt32(
            bigEndian: bytes[lenStart ..< lenStart + 4].withUnsafeBytes {
                $0.loadUnaligned(as: UInt32.self) // Data slices aren't 4-byte aligned
            }
        )
        guard Int(length) <= maxFrameSize else { throw FrameError.oversize }

        let total = headerSize + Int(length)
        guard bytes.count >= total else { return nil } // need more bytes

        let payloadStart = base + headerSize
        let payload = bytes[payloadStart ..< base + total]
        // Lossy decode (invalid bytes → U+FFFD) is INTENTIONAL: the daemon redacts
        // whatever text it can rather than dropping a connection; strict rejection
        // is not the contract.
        let text = String(decoding: payload, as: UTF8.self)
        bytes.removeSubrange(base ..< base + total)
        return (status, text)
    }
}
