# Phase 1 — build-from-source formula for the `bshk-app/tap` Homebrew tap.
#
# This file is the source of truth; ship it by copying to
#   homebrew-tap/Formula/pf.rb
# (Homebrew formula = "built from source" per docs.brew.sh — see design §6.)
# Phase 2 (a Developer ID-notarized prebuilt) ships as a CASK, not a bottle
# (Homebrew re-signs relocated bottles ad-hoc, which would strip notarization).
#
# DISTRIBUTION: privacy-filter-swift is PUBLIC, so the source-tarball `url` below works
# once a release is tagged. To ship: push `main`, create a tag `pf-vX.Y.Z`, then fill
# `url` + `sha256` (`shasum -a 256` of `.../archive/refs/tags/pf-vX.Y.Z.tar.gz`) and copy
# this file to `Formula/pf.rb` in the bshk-app/homebrew-tap repo.

class Pf < Formula
  desc "On-device PII/secret redactor: stdin-stdout filter plus resident daemon"
  homepage "https://github.com/bshk-app/privacy-filter-swift"
  url "https://github.com/bshk-app/privacy-filter-swift/archive/refs/tags/pf-v0.1.0.tar.gz"
  sha256 "0" * 64 # TODO: fill at release (`shasum -a 256` of the tagged tarball)
  license "Apache-2.0"

  # MLX's Metal kernels need the FULL Xcode toolchain (`swift build` can't compile the
  # metallib); MLX/Metal is arm64-only; the package targets .macOS(.v14).
  depends_on xcode: ["16.0", :build]
  depends_on arch: :arm64
  depends_on macos: ">= :sonoma"

  def install
    # Build via xcodebuild so `default.metallib` is produced alongside the product
    # (a plain `swift build` yields a binary that crashes at runtime: "Failed to load
    # the default metallib").
    xcodebuild "-scheme", "pf",
               "-configuration", "Release",
               "-derivedDataPath", "build",
               "-destination", "platform=macOS",
               "SKIP_INSTALL=NO",
               "build"

    products = "build/Build/Products/Release"
    # Keep the binary next to its metallib + any MLX dylibs in libexec; symlink into bin.
    # A symlink (not a copy) preserves the executable's real path so MLX finds the
    # metallib sitting beside it in libexec.
    libexec.install Dir["#{products}/pf", "#{products}/default.metallib", "#{products}/*.dylib"]
    bin.install_symlink libexec/"pf"
  end

  # `brew services start pf` runs the resident daemon. Restart on crash, but NOT on a
  # clean error exit (so "model missing → run `pf pull`" doesn't hot-loop). After a crash
  # the daemon auto-reclaims its stale socket on restart (see design §4).
  service do
    run [opt_bin/"pf", "serve"]
    keep_alive crashed: true
    log_path var/"log/pf.log"
    error_log_path var/"log/pf.log"
  end

  def caveats
    <<~EOS
      Fetch the model once (~870 MB) into the canonical Hugging Face cache:
        pf pull
      Use it as a filter:
        cat app.log | pf
      Or run the resident daemon (warm model, unix socket at ~/.pf/pf.sock):
        brew services start pf
    EOS
  end

  test do
    # Smoke: the binary loads (metallib resolves) and prints usage. A redaction test
    # needs the model (`pf pull`), so we only assert the CLI is wired here.
    assert_match "redact", shell_output("#{bin}/pf --help 2>&1")
  end
end
