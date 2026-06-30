// `pf pull` — native HF download into the canonical huggingface_hub cache (design §5, Task 5).
//
// Writes the SAME on-disk layout the python `hf`/`huggingface_hub` CLI uses, so the model
// dedups with — and is visible to — the user's existing cache:
//
//     <base>/models--beshkenadze--privacy-filter-mlx/
//       ├── blobs/<etag>                          (content-addressed; written via <etag>.incomplete,
//       │                                          then atomic same-volume rename → final)
//       ├── snapshots/<commit>/<path>             (RELATIVE symlink into ../…/blobs/<etag>)
//       └── refs/main                             (40-char commit sha, NO trailing newline)
//
// ── WHY WE INTERCEPT THE REDIRECT (the load-bearing detail) ─────────────────────────────
//   The blob name MUST be huggingface_hub's etag (git-SHA1 for small files, SHA256 for LFS)
//   so blobs are byte-identical with the python cache. That value lives in `X-Linked-Etag`
//   on the FIRST response from huggingface.co (the 302/307 to the CDN). If you follow the
//   redirect to the LFS CDN, both `X-Repo-Commit` and `X-Linked-Etag` are GONE — the CDN
//   only returns its own (different) `ETag`. So we capture the commit + linked-etag from the
//   redirect response, THEN let the redirect proceed to stream the body. (huggingface_hub
//   does the equivalent: a metadata HEAD on the hub URL + a separate body GET.)
//
// Native Swift only — URLSession/Foundation, no shelling out to hf/python/curl.
// `@main` and the subcommand wiring live in PF.swift; this file adds the `Pull` command,
// the cache-base/etag helpers, and `resolveModelDir` (used by `pf`/`pf serve`).

import ArgumentParser
import Foundation

/// The repository `pf pull` downloads from (public).
let pfRepoId = "beshkenadze/privacy-filter-mlx"

/// huggingface_hub's per-repo folder name: `models--<owner>--<name>` (slashes → `--`).
func hubRepoFolder(_ repoId: String) -> String {
    "models--" + repoId.replacingOccurrences(of: "/", with: "--")
}

/// Canonical hub cache BASE, in huggingface_hub's own precedence order, with a `--cache`
/// override on top (for power users AND isolated tests):
///   `--cache` ▸ `$HF_HUB_CACHE` ▸ `$HF_HOME/hub` ▸ `~/.cache/huggingface/hub`.
func resolveHubCacheBase(_ override: String?) -> String {
    if let override, !override.isEmpty { return (override as NSString).expandingTildeInPath }
    let env = ProcessInfo.processInfo.environment
    if let c = env["HF_HUB_CACHE"], !c.isEmpty { return (c as NSString).expandingTildeInPath }
    if let h = env["HF_HOME"], !h.isEmpty { return (h as NSString).expandingTildeInPath + "/hub" }
    return "\(NSHomeDirectory())/.cache/huggingface/hub"
}

/// Files that make up each variant, as repo-relative paths (`<dir>/<file>` or root `<file>`).
/// q4-8emb is the shipped default (~870 MB quantized); bf16 is the full-precision variant.
func variantFiles(_ variant: String) -> [String] {
    switch variant {
    case "bf16":
        return ["config.json", "model.safetensors", "tokenizer.json"]
    case "q4-8emb":
        return ["q4-8emb/config.json", "q4-8emb/model.safetensors", "q4-8emb/tokenizer.json"]
    default:
        return []
    }
}

/// Strip the surrounding double-quotes huggingface returns on etag headers (`"abc"` → `abc`),
/// and the weak-validator `W/` prefix if present. Matches huggingface_hub's `_normalize_etag`.
func unquoteEtag(_ raw: String) -> String {
    var s = raw.trimmingCharacters(in: .whitespaces)
    if s.hasPrefix("W/") { s.removeFirst(2) }
    if s.hasPrefix("\"") && s.hasSuffix("\"") && s.count >= 2 { s = String(s.dropFirst().dropLast()) }
    return s
}

/// Relative symlink target from `snapshots/<commit>/<relPath>` to `blobs/<etag>`.
///
/// The link lives at `snapshots/<commit>/<relPath>`; from ITS OWN directory we must climb to
/// the repo root, then descend into `blobs`. Depth from the link's directory to the repo root
/// = 2 (snapshots, <commit>) + (number of intermediate dirs in relPath). Concretely:
///   `snapshots/<c>/config.json`          → `../../blobs/<etag>`        (2 ups)
///   `snapshots/<c>/q4-8emb/config.json`  → `../../../blobs/<etag>`     (3 ups)
func blobSymlinkTarget(relPath: String, etag: String) -> String {
    let intermediateDirs = relPath.split(separator: "/", omittingEmptySubsequences: true).count - 1
    let ups = 2 + intermediateDirs
    return String(repeating: "../", count: ups) + "blobs/" + etag
}

/// Metadata captured from the FIRST (un-redirected) hub response for one file.
private struct FileMeta {
    let commit: String
    let etag: String
}

/// URLSession delegate that captures `X-Repo-Commit` + (`X-Linked-Etag` ▸ `ETag`) from the
/// redirect response, then ALLOWS the redirect so the body streams from the CDN in the same
/// task. If the file is served inline (no redirect), the headers are read off the final
/// response in `parseMeta` instead.
private final class RedirectCapturingDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    var captured: FileMeta?

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        if captured == nil, let m = Self.parseMeta(response) { captured = m }
        // SECURITY: the hub 302/307-redirects `resolve/main` to a DIFFERENT LFS-CDN host.
        // URLSession re-sends `Authorization: Bearer <hf-token>` on the new request by default,
        // leaking the token to that CDN. Strip it whenever the redirect target host differs from
        // the ORIGINAL request host (compare against the original, not the previous hop), mirroring
        // huggingface_hub which only carries auth on same-host/relative redirects.
        var req = request
        if req.url?.host != task.originalRequest?.url?.host {
            req.setValue(nil, forHTTPHeaderField: "Authorization")
        }
        completionHandler(req)  // follow the redirect to fetch the body (auth stripped cross-host)
    }

    /// Pull commit + etag out of a hub response. Returns nil if the commit is absent (e.g. a
    /// CDN response, which must never be the source of truth for the blob name).
    static func parseMeta(_ response: HTTPURLResponse) -> FileMeta? {
        guard let commit = header(response, "X-Repo-Commit"), !commit.isEmpty else { return nil }
        let rawEtag = header(response, "X-Linked-Etag") ?? header(response, "ETag")
        guard let rawEtag, !rawEtag.isEmpty else { return nil }
        return FileMeta(commit: commit, etag: unquoteEtag(rawEtag))
    }

    /// Case-insensitive header lookup (HTTP headers are case-insensitive; URLSession preserves
    /// server casing, which varies between the hub and the CDN). `value(forHTTPHeaderField:)`
    /// is case-insensitive and available on the macOS .v14 deployment target, so no fallback.
    static func header(_ response: HTTPURLResponse, _ name: String) -> String? {
        response.value(forHTTPHeaderField: name)
    }
}

struct Pull: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download the model into the canonical huggingface_hub cache (blobs/snapshots/refs).",
        discussion: """
            Writes the same blobs/snapshots/refs layout the python `hf` CLI uses, so the model
            dedups with and is visible to huggingface_hub. Resolves the cache base from
            $HF_HUB_CACHE ▸ $HF_HOME/hub ▸ ~/.cache/huggingface/hub (override with --cache).
            Blobs are content-addressed (etag): each is written to <etag>.incomplete then atomically
            renamed into place on the cache volume, so a partial download never looks complete. An
            already-present blob is skipped (resume is blob-granularity, not HTTP Range). `pf` /
            `pf serve` then find the model in that cache.
            """)

    @Option(name: .long, help: "Variant to pull: 'q4-8emb' (default, ~870 MB quantized) or 'bf16'.")
    var variant = "q4-8emb"

    @Option(name: .long, help: "Override the hub cache base dir ($HF_HUB_CACHE ▸ $HF_HOME/hub ▸ ~/.cache/huggingface/hub).")
    var cache: String?

    @Option(name: .long, parsing: .singleValue,
            help: "Only download files whose repo path contains this substring (repeatable; e.g. config.json).")
    var include: [String] = []

    mutating func run() async throws {
        var files = variantFiles(variant)
        guard !files.isEmpty else {
            throw RuntimeError("unknown --variant '\(variant)' (expected 'q4-8emb' or 'bf16')")
        }
        if !include.isEmpty {
            files = files.filter { path in include.contains { path.contains($0) } }
            guard !files.isEmpty else {
                throw RuntimeError("no files in variant '\(variant)' match --include \(include)")
            }
        }

        let base = resolveHubCacheBase(cache)
        let repoDir = "\(base)/\(hubRepoFolder(pfRepoId))"
        let fm = FileManager.default
        for sub in ["blobs", "snapshots", "refs"] {
            try fm.createDirectory(atPath: "\(repoDir)/\(sub)", withIntermediateDirectories: true)
        }

        let token = loadHubToken()
        let session = URLSession(configuration: .ephemeral)

        logStderr("pf pull: \(pfRepoId) [\(variant)] → \(repoDir)")
        var commit: String?
        for path in files {
            let c = try await downloadFile(path: path, repoDir: repoDir, token: token, session: session)
            commit = c
        }

        // refs/main = commit sha, NO trailing newline (byte-identical to huggingface_hub).
        // Only advertise a complete commit on an UNFILTERED pull: a metadata-only `--include`
        // pull downloads a SUBSET of the variant, so writing refs/main would make an incomplete
        // snapshot resolve as present (the model load would then fail on a missing weight). With
        // no refs/main, resolveModelDirFromCache fails closed cleanly until a full `pf pull`.
        if include.isEmpty, let commit {
            let refsMain = "\(repoDir)/refs/main"
            try Data(commit.utf8).write(to: URL(fileURLWithPath: refsMain), options: .atomic)
        }
        logStderr("pf pull: done.")
    }

    /// Download ONE repo file into the cache. Returns the commit sha (for `refs/main`).
    /// Skips the body download when the target blob already exists (resume/dedup).
    private func downloadFile(path: String, repoDir: String, token: String?,
                              session: URLSession) async throws -> String {
        let url = URL(string: "https://huggingface.co/\(pfRepoId)/resolve/main/\(path)")!
        var request = URLRequest(url: url)
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }

        let delegate = RedirectCapturingDelegate()
        let (tmpURL, response) = try await session.download(for: request, delegate: delegate)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        guard let http = response as? HTTPURLResponse else {
            throw RuntimeError("pull \(path): non-HTTP response")
        }
        guard http.statusCode == 200 else {
            // One-line hint for the common auth/missing cases. No blob/refs written on error.
            let hint: String
            switch http.statusCode {
            case 401, 403: hint = " (check ~/.cache/huggingface/token)"
            case 404:      hint = " (repo/file missing or private)"
            default:       hint = ""
            }
            throw RuntimeError("pull \(path): HTTP \(http.statusCode)\(hint)")
        }
        // Prefer the metadata captured pre-redirect; fall back to the final response for files
        // served inline by the hub (no CDN redirect → commit+etag are on the 200 itself).
        guard let meta = delegate.captured ?? RedirectCapturingDelegate.parseMeta(http) else {
            throw RuntimeError("pull \(path): missing X-Repo-Commit / etag headers")
        }

        let fm = FileManager.default
        let blobPath = "\(repoDir)/blobs/\(meta.etag)"
        if fm.fileExists(atPath: blobPath) {
            logStderr("  skip  \(path) (blob present)")
        } else {
            // Atomic blob write — never leave a partial blob at the content-addressed path.
            // The URLSession temp may live on a different volume than the cache (e.g. cache on
            // /Volumes/DATA), so `moveItem` there is a non-atomic copy-then-delete: an interruption
            // mid-copy would leave a PARTIAL file at blobs/<etag> that the fileExists skip then
            // trusts forever (corrupt weights). So we copy to blobs/<etag>.incomplete first — a name
            // that can NEVER satisfy the fileExists(blobs/<etag>) skip — and only after the full body
            // is in place do a SAME-FILESYSTEM atomic rename .incomplete → final. A partial download
            // can thus never become a "complete" blob. (download(for:) already streamed the full body
            // to tmpURL; one move + one rename, no re-read.)
            let incompletePath = blobPath + ".incomplete"
            let incompleteURL = URL(fileURLWithPath: incompletePath)
            try? fm.removeItem(at: incompleteURL)  // overwrite any stale pre-existing .incomplete
            try fm.moveItem(at: tmpURL, to: incompleteURL)
            // Atomic rename within the cache volume: .incomplete → final blob path.
            try fm.moveItem(at: incompleteURL, to: URL(fileURLWithPath: blobPath))
            logStderr("  pull  \(path) → blobs/\(String(meta.etag.prefix(12)))…")
        }

        // snapshots/<commit>/<path> as a RELATIVE symlink into blobs/<etag>.
        let snapPath = "\(repoDir)/snapshots/\(meta.commit)/\(path)"
        try fm.createDirectory(atPath: (snapPath as NSString).deletingLastPathComponent,
                               withIntermediateDirectories: true)
        // Clear ANY pre-existing link first (live OR dangling) so re-pull is idempotent.
        // attributesOfItem follows the link, so it can't see a dangling one — removeItem on the
        // path is a no-follow unlink that clears either, avoiding an EEXIST from createSymbolicLink.
        try? fm.removeItem(atPath: snapPath)
        let target = blobSymlinkTarget(relPath: path, etag: meta.etag)
        try fm.createSymbolicLink(atPath: snapPath, withDestinationPath: target)

        return meta.commit
    }
}

/// Read the HF auth token from `~/.cache/huggingface/token` if present (our repo is public,
/// but support gated/private use). Trimmed of surrounding whitespace/newlines.
func loadHubToken() -> String? {
    let env = ProcessInfo.processInfo.environment
    let tokenPath: String
    if let h = env["HF_HOME"], !h.isEmpty {
        tokenPath = (h as NSString).expandingTildeInPath + "/token"
    } else {
        tokenPath = "\(NSHomeDirectory())/.cache/huggingface/token"
    }
    guard let raw = try? String(contentsOfFile: tokenPath, encoding: .utf8) else { return nil }
    let t = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    return t.isEmpty ? nil : t
}

// ── Model resolution (used by `pf` and `pf serve`) ──────────────────────────────────────

/// The files every variant must contain to be loadable (config + weights + tokenizer).
private let requiredModelFiles = ["config.json", "model.safetensors", "tokenizer.json"]

/// Resolve the model directory for a variant from the canonical hub cache:
///   read `refs/main` → `snapshots/<commit>/<variant>/`.
/// Returns nil if the cache, ref, or snapshot dir is absent, OR if the snapshot is INCOMPLETE —
/// any required file missing, or a snapshot symlink that does not resolve to a real file (a
/// metadata-only/interrupted pull). The caller then fails closed with `run \`pf pull\`` rather
/// than surfacing an opaque model-load error later.
func resolveModelDirFromCache(variant: String, cacheBase override: String? = nil) -> String? {
    let base = resolveHubCacheBase(override)
    let repoDir = "\(base)/\(hubRepoFolder(pfRepoId))"
    let refsMain = "\(repoDir)/refs/main"
    guard let raw = try? String(contentsOfFile: refsMain, encoding: .utf8) else { return nil }
    let commit = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !commit.isEmpty else { return nil }
    // bf16 lives at the snapshot root; q4-8emb in its own subdir.
    let modelDir = variant == "bf16"
        ? "\(repoDir)/snapshots/\(commit)"
        : "\(repoDir)/snapshots/\(commit)/\(variant)"
    let fm = FileManager.default
    var isDir: ObjCBool = false
    guard fm.fileExists(atPath: modelDir, isDirectory: &isDir), isDir.boolValue else {
        return nil
    }
    // Each required file must exist AND its snapshot symlink must resolve to a real file.
    // `fileExists` follows symlinks, so a dangling link reports false here → incomplete.
    for file in requiredModelFiles {
        let p = "\(modelDir)/\(file)"
        guard fm.fileExists(atPath: p) else { return nil }
    }
    return modelDir
}

/// The model directory a command should load. `--model <dir>` (an explicit, non-default value)
/// ALWAYS wins, so dev invocations like `--model ../models/q4-8emb` keep working. When `--model`
/// was left at its default, resolve from the canonical hub cache; if the model is not cached,
/// fail closed with a clear instruction — never silently read stdin against a missing model.
func resolveModelDir(modelFlag: String, defaultFlag: String, variant: String = "q4-8emb") throws -> String {
    if modelFlag != defaultFlag {
        return modelFlag  // explicit override → any local dir
    }
    if let cached = resolveModelDirFromCache(variant: variant) {
        return cached
    }
    throw RuntimeError("model not found in HF cache — run `pf pull`")
}
