# Packaging `pf` for Homebrew

Two distribution tracks, per the design (`docs/plans/2026-06-21-pf-serve-and-brew-design.md` §6)
and Homebrew's own policy (formula = built from source; cask = pre-compiled binary built **and
signed** by upstream).

## Phase 1 — source formula (this dir: `homebrew/pf.rb`)

`brew install beshkenadze/tap/pf` → builds from source via `xcodebuild` (MLX's Metal kernels
need the full Xcode toolchain; `swift build` can't compile the metallib). Ships the binary +
`default.metallib` + dylibs into `libexec`, symlinks `bin/pf`, and wires `brew services` to run
`pf serve`.

**To ship it:** copy `homebrew/pf.rb` to `Formula/pf.rb` in the `beshkenadze/homebrew-tap` repo,
fill `url`/`sha256` for a tagged source tarball, and `brew install beshkenadze/tap/pf`.

> **⚠️ Distribution decision (unresolved):** `privacy-filter-swift` is **private**, so a public
> tap can't fetch its source tarball. Pick one:
> - **(a)** make the repo public, then `url` the tagged `…/archive/refs/tags/pf-vX.Y.Z.tar.gz`;
> - **(b)** keep it private, use a **private tap**, and `brew install` with a `HOMEBREW_GITHUB_API_TOKEN`;
> - **(c)** attach a public **source-tarball Release asset** and `url` that.

### `brew style` note
Running `brew style homebrew/pf.rb` **standalone** reports 4 offenses (Sorbet sigils,
frozen-string-literal, class-doc). These are **formula-exempt** in Homebrew's rubocop config and
do NOT fire once the file lives in a tap under `Formula/`. The substantive cops (`desc` length,
`xcodebuild` DSL helper) pass.

## Phase 2 — notarized cask (we have a Developer ID)

`brew install --cask beshkenadze/tap/pf` → downloads a Developer ID-signed + notarized prebuilt,
no build. Outline (a future GitHub Actions job):

1. `xcodebuild` Release on a `macos-14` arm64 runner.
2. Bundle `pf` + `default.metallib` + dylibs with **`@loader_path`** rpaths (relocatable, so no
   path rewriting that would invalidate the signature).
3. `codesign --options runtime -s "Developer ID Application: …"` → `notarytool submit --wait` →
   `stapler staple` → tar.gz → attach to the GitHub Release + sha256.
4. A `Casks/pf.rb` with the `binary` stanza installs the notarized binary as-is.

**Why a cask, not a bottle:** Homebrew relocates + ad-hoc re-signs bottles on pour, which would
strip the Developer ID signature/notarization. A cask installs the artifact unmodified, so the
signature stays valid (hence the `@loader_path` rpaths — nothing gets rewritten).

## After install (both tracks)

```sh
pf pull                       # fetch the model (~870 MB) into the canonical HF cache
cat app.log | pf              # one-shot filter
brew services start pf        # resident daemon (warm model, ~/.pf/pf.sock)
```
