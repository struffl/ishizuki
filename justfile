# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Task runner for ishizuki — run `just` to list recipes. https://github.com/casey/just

set dotenv-load := true

swift := "swift"
swift_format := "xcrun swift-format"
version := `git describe --tags --always --dirty 2>/dev/null || echo 0.0.0`
# Everything builds arm64 only: Apple Silicon machines, and x86_64 is never
# useful here (mlx-swift has no Float16 for it, so a universal build fails).
# ARCHS=arm64 as an override is the reliable lever — the SwiftPM package
# targets don't inherit ARCHS from the app target's build settings.
xcodebuild := "xcodebuild -project Ishizuki.xcodeproj -scheme Ishizuki -destination 'platform=macOS' -derivedDataPath .build -skipPackagePluginValidation -skipMacroValidation ARCHS=arm64"

# list recipes
default:
    @just --list

# build the engine library on its own
build:
    {{ swift }} build

# regenerate the Xcode project from App/project.yml
app-project:
    cd App && xcodegen generate

# build the menu bar app in Debug target
app-debug: app-project
    cd App && {{ xcodebuild }} -configuration Debug build

# build the menu bar app
app: app-project
    cd App && {{ xcodebuild }} -configuration Release build

# build and launch the menu bar app
app-run: app
    open App/.build/Build/Products/Release/Ishizuki.app

# build and launch the menu bar app in Debug target
app-run-debug: app-debug
    open App/.build/Build/Products/Debug/Ishizuki.app

# build the iPhone companion for the simulator
phone: app-project
    cd App && xcodebuild -project Ishizuki.xcodeproj -scheme IshizukiPhone -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -derivedDataPath .build -skipPackagePluginValidation -skipMacroValidation ARCHS=arm64 -configuration Debug build

# build and run the iPhone companion in the simulator
phone-run: phone
    open -a Simulator
    xcrun simctl install booted App/.build/Build/Products/Debug-iphonesimulator/IshizukiPhone.app
    xcrun simctl launch booted studio.ishizuki.phone

# build the iPhone companion for a device
phone-device: app-project
    cd App && xcodebuild -project Ishizuki.xcodeproj -scheme IshizukiPhone -destination 'generic/platform=iOS' -derivedDataPath .build -skipPackagePluginValidation -skipMacroValidation ARCHS=arm64 -configuration Release build

# archive and export a Mac App Store upload (needs Apple Distribution + installer identities)
app-store: app-project
    ./Scripts/app-store.sh {{ version }}

# signed, notarized drag-to-Applications DMG (needs a Developer ID Application cert)
build-dmg: app-project
    ./Scripts/build-dmg.sh {{ version }}

# unit tests
test:
    ./Scripts/run-tests.sh

# format Swift in place
fmt:
    {{ swift_format }} format --in-place --recursive --parallel Sources Tests App/Ishizuki App/IshizukiPhone App/Shared Package.swift

# check formatting (nonzero on findings; for CI)
fmt-check:
    {{ swift_format }} lint --strict --recursive Sources Tests App/Ishizuki App/IshizukiPhone App/Shared Package.swift

# compile the app icon to .icns
icon:
    rm -rf .build/icon && mkdir -p .build/icon
    actool assets/Ishizuki.icon --compile .build/icon --app-icon Ishizuki --platform macosx --minimum-deployment-target 26 --output-partial-info-plist .build/icon/partial.plist --errors --warnings >/dev/null
    cp .build/icon/Ishizuki.icns assets/ishizuki.icns
    @echo "wrote assets/ishizuki.icns"

# remove build artifacts
clean:
    rm -rf .build App/.build
