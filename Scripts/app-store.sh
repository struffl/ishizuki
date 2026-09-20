#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Archive the menu bar app and export a Mac App Store package, ready to upload.
set -euo pipefail

VERSION="${1:-$(git describe --tags --always 2>/dev/null || echo 0.0.0)}"
VERSION="${VERSION#v}"
BUILD="${BUILD:-$(date +%Y%m%d%H%M)}"
TEAM_ID="${TEAM_ID:-2WTR6KH74L}"

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir/App"

if ! security find-identity -v -p codesigning | grep -q "Apple Distribution"; then
  cat >&2 <<'MISSING'
No "Apple Distribution" certificate in the keychain.

A Mac App Store build needs two you do not have yet:
  Apple Distribution            signs the .app
  3rd Party Mac Developer Installer   signs the .pkg

Make them in Xcode (Settings > Accounts > Manage Certificates > +), and register
an App Store provisioning profile for studio.ishizuki.app in the developer portal.
MISSING
  exit 1
fi

archive="$root_dir/.build/Ishizuki.xcarchive"
export_dir="$root_dir/.build/app-store"
rm -rf "$archive" "$export_dir"

echo "==> Archiving $VERSION ($BUILD)"
xcodegen generate >/dev/null
xcodebuild archive \
  -project Ishizuki.xcodeproj \
  -scheme Ishizuki \
  -configuration Release \
  -archivePath "$archive" \
  -skipPackagePluginValidation -skipMacroValidation \
  MARKETING_VERSION="$VERSION" \
  CURRENT_PROJECT_VERSION="$BUILD" \
  | grep -E "error:|warning: .*deprecat|ARCHIVE" || true

options="$root_dir/.build/export-options.plist"
cat > "$options" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>app-store-connect</string>
    <key>teamID</key>
    <string>$TEAM_ID</string>
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

pkg="$(find "$export_dir" -name "*.pkg" | head -1)"
echo ""
echo "wrote $pkg"
echo ""
echo "Upload it with an App Store Connect API key:"
echo "  xcrun altool --upload-app -f \"$pkg\" -t macos \\"
echo "    --apiKey \$ASC_KEY_ID --apiIssuer \$ASC_ISSUER_ID"
