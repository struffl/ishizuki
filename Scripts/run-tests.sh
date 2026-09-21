#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-debug}"

STALE=$(find .build -path "*IshizukiKitTests.xctest/Contents/MacOS/mlx.metallib" 2>/dev/null || true)
[ -n "$STALE" ] && rm -f $STALE

swift build --build-tests -c "$CONFIG"

BUNDLE=$(find .build -name "IshizukiKitTests.xctest" -maxdepth 5 -type d | head -1)
[ -n "$BUNDLE" ] || { echo "test bundle not found; did the build succeed?" >&2; exit 1; }

METALLIB=$(find "$BUNDLE" -name "default.metallib" | head -1)
[ -n "$METALLIB" ] || { echo "no default.metallib inside $BUNDLE" >&2; exit 1; }

cp "$METALLIB" "$BUNDLE/Contents/MacOS/mlx.metallib"
codesign -f -s - "$BUNDLE" >/dev/null 2>&1 || true

STATUS=0
xcrun xctest "$BUNDLE" || STATUS=$?

LINK=$(find .build -name "IshizukiLinkTests.xctest" -maxdepth 5 -type d | head -1)
if [ -n "$LINK" ]; then
    codesign -f -s - "$LINK" >/dev/null 2>&1 || true
    xcrun xctest "$LINK" || STATUS=$?
fi

exit "$STATUS"
