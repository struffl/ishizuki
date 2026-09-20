# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Task runner for ishizuki — run `just` to list recipes. https://github.com/casey/just

set dotenv-load := true

swift := "swift"
swift_format := "xcrun swift-format"
version := `git describe --tags --always --dirty 2>/dev/null || echo 0.0.0`
xcodebuild := "xcodebuild -project Ishizuki.xcodeproj -scheme Ishizuki -derivedDataPath .build -skipPackagePluginValidation -skipMacroValidation"

# list recipes
default:
    @just --list

# build the engine library on its own
build:
    {{ swift }} build

# regenerate the Xcode project from App/project.yml
app-project:
    cd App && xcodegen generate

# build the menu bar app
app: app-project
    cd App && {{ xcodebuild }} -configuration Debug build

# build and launch the menu bar app
app-run: app
    open App/.build/Build/Products/Debug/Ishizuki.app

# archive and export a Mac App Store upload (needs Apple Distribution + installer identities)
app-store: app-project
    ./Scripts/app-store.sh {{ version }}

# unit tests
test:
    ./Scripts/run-tests.sh

# format Swift in place
fmt:
    {{ swift_format }} format --in-place --recursive --parallel Sources Tests App/Ishizuki Package.swift

# check formatting (nonzero on findings; for CI)
fmt-check:
    {{ swift_format }} lint --strict --recursive Sources Tests App/Ishizuki Package.swift

# compile the app icon to .icns
icon:
    rm -rf .build/icon && mkdir -p .build/icon
    actool assets/Ishizuki.icon --compile .build/icon --app-icon Ishizuki --platform macosx --minimum-deployment-target 26 --output-partial-info-plist .build/icon/partial.plist --errors --warnings >/dev/null
    cp .build/icon/Ishizuki.icns assets/ishizuki.icns
    @echo "wrote assets/ishizuki.icns"

# remove build artifacts
clean:
    rm -rf .build App/.build
