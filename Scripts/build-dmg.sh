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
sips -z 816 1456 "$root_dir/assets/ishizuki.jpg" -s format png --out "$staging/background@2x.png" >/dev/null
sips -z 408 728 "$root_dir/assets/ishizuki.jpg" -s format png --out "$staging/background.png" >/dev/null

dmg="$root_dir/.build/Ishizuki.dmg"
rm -f "$dmg"

echo "==> Building DMG"
"${DMGBUILD:-dmgbuild}" -s "$root_dir/Scripts/dmg-settings.py" \
  -D app="$app" -D background="$staging/background.png" \
  Ishizuki "$dmg"
codesign --sign "$identity" --timestamp "$dmg"

echo "==> Notarizing"
xcrun notarytool submit "$dmg" --keychain-profile "$NOTARY_PROFILE" --wait
xcrun stapler staple "$dmg"
xcrun stapler validate "$dmg"

echo ""
echo "wrote $dmg"
