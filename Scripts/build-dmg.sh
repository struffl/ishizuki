#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Archive the menu bar app for Developer ID, notarize it, and package it
# into a drag-to-Applications DMG. Uses the existing notarytool keychain
# profile; override its name with NOTARYTOOL_PROFILE.
set -euo pipefail

VERSION="${1:-$(git describe --tags --always --dirty 2>/dev/null || echo 0.0.0)}"
VERSION="${VERSION#v}"
BUILD="${BUILD:-$(date +%Y%m%d%H%M)}"
TEAM_ID="${TEAM_ID:-2WTR6KH74L}"
NOTARY_PROFILE="${NOTARYTOOL_PROFILE:-AC_PASSWORD}"

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir/App"

identity="$(security find-identity -v -p codesigning \
  | grep '"Developer ID Application' | head -1 \
  | sed -E 's/.*\) ([0-9A-F]{40}) .*/\1/')"
if [[ -z "$identity" ]]; then
  cat >&2 <<'MISSING'
No "Developer ID Application" certificate in the keychain.

Make one in Xcode (Settings > Accounts > Manage Certificates > +) or at
developer.apple.com under Certificates, Identifiers & Profiles.
MISSING
  exit 1
fi

archive="$root_dir/.build/Ishizuki-dmg.xcarchive"
export_dir="$root_dir/.build/developer-id"
rm -rf "$archive" "$export_dir"

echo "==> Archiving $VERSION ($BUILD)"
xcodegen generate >/dev/null
xcodebuild archive \
  -project Ishizuki.xcodeproj \
  -scheme Ishizuki \
  -destination 'platform=macOS,arch=arm64' \
  -configuration Release \
  -archivePath "$archive" \
  -skipPackagePluginValidation -skipMacroValidation \
  MARKETING_VERSION="$VERSION" \
  CURRENT_PROJECT_VERSION="$BUILD" \
  CODE_SIGN_STYLE=Manual \
  CODE_SIGN_IDENTITY="$identity" \
  ARCHS=arm64 ONLY_ACTIVE_ARCH=NO \
  | grep -E "error:|warning: .*deprecat|ARCHIVE" || true

options="$root_dir/.build/export-options-dmg.plist"
cat > "$options" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>developer-id</string>
    <key>teamID</key>
    <string>$TEAM_ID</string>
    <key>signingCertificate</key>
    <string>Developer ID Application</string>
    <key>destination</key>
    <string>export</string>
</dict>
</plist>
PLIST

echo "==> Exporting"
xcodebuild -exportArchive \
  -archivePath "$archive" \
  -exportOptionsPlist "$options" \
  -exportPath "$export_dir"

app="$export_dir/Ishizuki.app"
[[ -d "$app" ]] || { echo "error: export did not produce $app" >&2; exit 1; }

codesign --verify --deep --strict "$app"
signature="$(codesign --display --verbose=2 "$app" 2>&1)"
[[ "$signature" == *"flags="*"runtime"* ]] || {
    echo "error: the hardened runtime did not stick" >&2
    exit 1
}

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
cp -R "$app" "$staging/"
ln -s /Applications "$staging/Applications"

dmg="$root_dir/.build/Ishizuki.dmg"
writable_dmg="$root_dir/.build/Ishizuki-writable.dmg"
rm -f "$dmg" "$writable_dmg"

echo "==> Building DMG"
diskutil image create from "$staging" --format RAW --volumeName Ishizuki "$writable_dmg"
mount="$(diskutil image attach --plist "$writable_dmg" | python3 -c 'import plistlib, sys; print(next(e["mount-point"] for e in plistlib.loads(sys.stdin.buffer.read())["system-entities"] if "mount-point" in e))')"
for attempt in 1 2 3; do
    osascript \
    -e 'on run argv' \
    -e 'set dmgFolder to POSIX file (item 1 of argv) as alias' \
    -e 'tell application "Finder"' \
    -e 'set dmgWindow to container window of dmgFolder' \
    -e 'open dmgWindow' \
    -e 'set current view of dmgWindow to icon view' \
    -e 'set toolbar visible of dmgWindow to false' \
    -e 'set statusbar visible of dmgWindow to false' \
    -e 'set bounds of dmgWindow to {100, 100, 580, 370}' \
    -e 'set theViewOptions to the icon view options of dmgWindow' \
    -e 'set arrangement of theViewOptions to not arranged' \
    -e 'set icon size of theViewOptions to 64' \
    -e 'set position of item "Ishizuki.app" of dmgFolder to {120, 178}' \
    -e 'set position of item "Applications" of dmgFolder to {380, 178}' \
    -e 'update dmgFolder without registering applications' \
    -e 'delay 2' \
    -e 'close dmgWindow' \
    -e 'end tell' \
    -e 'end run' "$mount"
    for _ in $(seq 15); do if [ -f "$mount/.DS_Store" ]; then break; fi; sleep 1; done
    if [ -f "$mount/.DS_Store" ]; then break; fi
    echo "Finder has not written the window layout yet; retrying ($attempt)" >&2
done
test -f "$mount/.DS_Store" || { echo "Finder never wrote the window layout" >&2; exit 1; }
diskutil eject "$mount" >/dev/null
diskutil image create from "$writable_dmg" --format UDZO "$dmg"
rm -f "$writable_dmg"

echo "==> Notarizing"
xcrun notarytool submit "$dmg" --keychain-profile "$NOTARY_PROFILE" --wait
xcrun stapler staple "$dmg"
xcrun stapler validate "$dmg"

echo ""
echo "wrote $dmg"
