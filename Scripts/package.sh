#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Build a signed + notarized rootless .pkg for ishizuki (installs to ~/.local, no admin password).
set -euo pipefail

SWIFT="${SWIFT:-swift}"
PKG_ID="${PKG_ID:-studio.ishizuki}"
NOTARY_PROFILE="${NOTARY_PROFILE:-AC_NOTARY}"
MIN_OS="${MIN_OS:-15.0}"
VERSION="${1:-$(git describe --tags --always --dirty 2>/dev/null || echo 0.0.0)}"

: "${APP_IDENTITY:?set APP_IDENTITY to your 'Developer ID Application: ...' identity}"
: "${PKG_IDENTITY:?set PKG_IDENTITY to your 'Developer ID Installer: ...' identity}"

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"

echo "==> Building release"
"$SWIFT" build -c release

BIN=".build/release/ishizuki"
BUNDLE=".build/release/mlx-swift_Cmlx.bundle"
[ -x "$BIN" ]    || { echo "missing $BIN"; exit 1; }
[ -d "$BUNDLE" ] || { echo "missing $BUNDLE"; exit 1; }

echo "==> Compiling app icon"
ICON_OUT="$(mktemp -d)"
actool "assets/Ishizuki.icon" --compile "$ICON_OUT" --app-icon Ishizuki \
  --platform macosx --minimum-deployment-target 26 \
  --output-partial-info-plist "$ICON_OUT/partial.plist" --errors --warnings >/dev/null
ICNS="$ICON_OUT/Ishizuki.icns"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE" "$ICON_OUT"' EXIT
libexec="$STAGE/root/.local/libexec/ishizuki"
bindir="$STAGE/root/.local/bin"
mkdir -p "$libexec" "$bindir"

echo "==> Staging payload"
cp "$BIN" "$libexec/ishizuki"
cp -R "$BUNDLE" "$libexec/"
cp "$BUNDLE/Contents/Resources/default.metallib" "$libexec/mlx.metallib"
cp "$ICNS" "$libexec/ishizuki.icns"
install -m 0755 Scripts/uninstall.sh "$libexec/uninstall.sh"
cat > "$bindir/ishizuki" <<'SHIM'
#!/bin/sh
# MLX finds its metallib next to the real binary (dladdr, symlinks unresolved),
# so exec the real path rather than symlinking to it.
exec "$(dirname "$0")/../libexec/ishizuki/ishizuki" "$@"
SHIM
chmod 0755 "$bindir/ishizuki"

echo "==> Code signing"
codesign --force --timestamp --sign "$APP_IDENTITY" "$libexec/mlx-swift_Cmlx.bundle"
codesign --force --timestamp --options runtime --sign "$APP_IDENTITY" "$libexec/ishizuki"
codesign --verify --strict --verbose=2 "$libexec/ishizuki"

xattr -cr "$STAGE/root"

SCRIPTS="$STAGE/scripts"
mkdir -p "$SCRIPTS"
cat > "$SCRIPTS/postinstall" <<'SH'
#!/bin/bash
BINDIR="$HOME/.local/bin"
RC="$HOME/.zshrc"
case ":$PATH:" in *":$BINDIR:"*) exit 0 ;; esac
grep -qF "# >>> ishizuki PATH >>>" "$RC" 2>/dev/null && exit 0
[ -f "$RC" ] && [ -n "$(tail -c1 "$RC" 2>/dev/null)" ] && printf '\n' >> "$RC"
printf '%s\n' '# >>> ishizuki PATH >>>' 'export PATH="$HOME/.local/bin:$PATH"' '# <<< ishizuki PATH <<<' >> "$RC"
exit 0
SH
chmod +x "$SCRIPTS/postinstall"

echo "==> Building component package"
pkgbuild --root "$STAGE/root" --identifier "$PKG_ID" --version "$VERSION" \
  --scripts "$SCRIPTS" --install-location "/" "$STAGE/component.pkg"

echo "==> Building product installer"
RES="$STAGE/resources"
mkdir -p "$RES"
cp "assets/ishizuki-icon.png" "$RES/background.png"
cat > "$RES/conclusion.html" <<'HTML'
<!DOCTYPE html><html><body style="font-family:-apple-system,Helvetica,sans-serif;font-size:13px;color:#111;margin:16px">
<p><b>Ishizuki is installed</b> in your home folder. Open a new Terminal and run <code>ishizuki --help</code>.</p>
<p style="margin-top:16px">To uninstall (no password needed):</p>
<p><code>~/.local/libexec/ishizuki/uninstall.sh</code></p>
<p style="color:#666">Add <code>--purge</code> to also delete downloaded models.</p>
</body></html>
HTML
cat > "$STAGE/distribution.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>Ishizuki</title>
    <background file="background.png" alignment="bottomleft" scaling="proportional"/>
    <conclusion file="conclusion.html"/>
    <options customize="never" require-scripts="false" hostArchitectures="arm64"/>
    <domains enable_anywhere="false" enable_currentUserHome="true" enable_localSystem="false"/>
    <volume-check>
        <allowed-os-versions><os-version min="$MIN_OS"/></allowed-os-versions>
    </volume-check>
    <choices-outline>
        <line choice="default"><line choice="$PKG_ID"/></line>
    </choices-outline>
    <choice id="default"/>
    <choice id="$PKG_ID" visible="false"><pkg-ref id="$PKG_ID"/></choice>
    <pkg-ref id="$PKG_ID" version="$VERSION" onConclusion="none">component.pkg</pkg-ref>
</installer-gui-script>
XML

PKG="ishizuki-${VERSION}.pkg"
productbuild --distribution "$STAGE/distribution.xml" --package-path "$STAGE" \
  --resources "$RES" --sign "$PKG_IDENTITY" "$PKG"

echo "==> Notarizing"
if ! xcrun notarytool submit "$PKG" --keychain-profile "$NOTARY_PROFILE" --wait; then
  echo "Notarization failed. Inspect with:"
  echo "  xcrun notarytool history --keychain-profile $NOTARY_PROFILE"
  echo "  xcrun notarytool log <submission-id> --keychain-profile $NOTARY_PROFILE"
  exit 1
fi

echo "==> Stapling"
xcrun stapler staple "$PKG"
xcrun stapler validate "$PKG"

echo
echo "Done: $PKG  (double-click installs to ~/.local, no admin password)"
