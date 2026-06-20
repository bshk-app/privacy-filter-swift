#!/usr/bin/env bash
# Build (via Xcode, so MLX's Metal kernels compile into default.metallib) + run an
# executable from the `pf` package. Plain `swift run` cannot build the Metal shaders.
#
#   ./run.sh pf-parity ../models/privacy-filter parity-fixture.json
#
# CONFIG=Release ./run.sh ...   for an optimized build.
set -euo pipefail
cd "$(dirname "$0")"
PRODUCT="${1:?usage: run.sh <product> [args...]}"; shift || true
CONFIG="${CONFIG:-Debug}"
DERIVED="$PWD/.build/xcode"
LOG=/tmp/pf-xcodebuild.log

echo "building $PRODUCT ($CONFIG) via xcodebuild…" >&2
if ! xcodebuild -scheme "$PRODUCT" -destination 'platform=macOS' \
        -derivedDataPath "$DERIVED" -configuration "$CONFIG" build >"$LOG" 2>&1; then
    echo "build failed — tail of $LOG:" >&2; tail -25 "$LOG" >&2; exit 1
fi
exec "$DERIVED/Build/Products/$CONFIG/$PRODUCT" "$@"
