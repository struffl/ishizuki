#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Build a signed + notarized loose tarball: the binary and the metallib it needs, nothing else.
set -euo pipefail

SWIFT="${SWIFT:-swift}"
NOTARY_PROFILE="${NOTARY_PROFILE:-AC_NOTARY}"
VERSION="${1:-$(git describe --tags --always --dirty 2>/dev/null || echo 0.0.0)}"
ARCH="$(uname -m)"

: "${APP_IDENTITY:?set APP_IDENTITY to your 'Developer ID Application: ...' identity}"

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"

echo "==> Building release"
"$SWIFT" build -c release

BIN=".build/release/ishizuki"
BUNDLE=".build/release/mlx-swift_Cmlx.bundle"
[ -x "$BIN" ]    || { echo "missing $BIN"; exit 1; }
[ -d "$BUNDLE" ] || { echo "missing $BUNDLE"; exit 1; }

NAME="ishizuki-$VERSION-macos-$ARCH"
TAR="$NAME.tar.gz"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
payload="$STAGE/$NAME"
mkdir -p "$payload"

echo "==> Staging payload"
cp "$BIN" "$payload/ishizuki"
cp "$BUNDLE/Contents/Resources/default.metallib" "$payload/mlx.metallib"
cp LICENSE "$payload/LICENSE"

echo "==> Code signing"
codesign --force --timestamp --options runtime --sign "$APP_IDENTITY" "$payload/ishizuki"
codesign --verify --strict --verbose=2 "$payload/ishizuki"
xattr -cr "$payload"

echo "==> Notarizing"
ZIP="$STAGE/$NAME.zip"
ditto -c -k --keepParent "$payload" "$ZIP"
if ! xcrun notarytool submit "$ZIP" --keychain-profile "$NOTARY_PROFILE" --wait; then
  echo "Notarization failed. Inspect with:"
  echo "  xcrun notarytool history --keychain-profile $NOTARY_PROFILE"
  echo "  xcrun notarytool log <submission-id> --keychain-profile $NOTARY_PROFILE"
  exit 1
fi

# A loose executable cannot carry a stapled ticket the way a .pkg can; Gatekeeper resolves the
# notarization for this binary's cdhash online instead.
rm -f "$root_dir/$TAR"
tar -czf "$root_dir/$TAR" -C "$STAGE" "$NAME"

echo
echo "Done: $TAR"
echo "  tar -xzf $TAR && $NAME/ishizuki serve"
