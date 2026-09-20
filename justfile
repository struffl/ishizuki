# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Task runner for ishizuki — run `just` to list recipes. https://github.com/casey/just

set dotenv-load := true

swift := "swift"
swift_format := "xcrun swift-format"
prefix := env_var_or_default("PREFIX", home_directory() / ".local")
version := `git describe --tags --always --dirty 2>/dev/null || echo 0.0.0`
bin := ".build/release/ishizuki"
bundle := ".build/release/mlx-swift_Cmlx.bundle"

# list recipes
default:
    @just --list

# build the release binary, with the metallib embedded in it
release:
    {{ swift }} build -c release
    @mkdir -p .build/embed
    @cp {{ bundle }}/Contents/Resources/default.metallib .build/embed/mlx.metallib
    ISHIZUKI_METALLIB="{{ justfile_directory() }}/.build/embed/mlx.metallib" {{ swift }} build -c release

# run the server: OpenAI + Anthropic on one port (default :8128). e.g. just serve --kv-bits 3.5
serve *args: release
    {{ bin }} serve {{ args }}

# talk to the model in the terminal, the quick demo. e.g. just demo --fold-thinking
demo *args: release
    {{ bin }} demo {{ args }}

# download / repair the model pack from HuggingFace (uses $HF_TOKEN if set)
pull *args: release
    {{ bin }} pull {{ args }}

# sampler unit tests
test:
    ./Scripts/run-tests.sh

# format Swift in place
fmt:
    {{ swift_format }} format --in-place --recursive --parallel Sources Tests App/Ishizuki Package.swift

# check formatting (nonzero on findings; for CI)
fmt-check:
    {{ swift_format }} lint --strict --recursive Sources Tests App/Ishizuki Package.swift

# numerics + kernel checks
verify: release
    {{ bin }} verify
    {{ bin }} verify-logits
    {{ bin }} kernel-check

# benchmarks
bench: release
    {{ bin }} context-bench --kv-bits 16 --lengths 512,2048
    {{ bin }} kv-bench
    {{ bin }} spec-bench
    {{ bin }} batch-check --batch 4

# compile the app icon to .icns
icon:
    rm -rf .build/icon && mkdir -p .build/icon
    actool assets/Ishizuki.icon --compile .build/icon --app-icon Ishizuki --platform macosx --minimum-deployment-target 26 --output-partial-info-plist .build/icon/partial.plist --errors --warnings >/dev/null
    cp .build/icon/Ishizuki.icns assets/ishizuki.icns
    @echo "wrote assets/ishizuki.icns"

# install to ~/.local, rootless (override with PREFIX=...)
install: icon
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'git checkout -- Sources/IshizukiKit/BuildInfo.swift 2>/dev/null || true' EXIT
    ./Scripts/stamp-version.sh "{{ version }}"
    {{ swift }} build -c release
    mkdir -p .build/embed
    cp {{ bundle }}/Contents/Resources/default.metallib .build/embed/mlx.metallib
    ISHIZUKI_METALLIB="{{ justfile_directory() }}/.build/embed/mlx.metallib" {{ swift }} build -c release
    prefix="{{ prefix }}"
    install -d "$prefix/libexec/ishizuki" "$prefix/bin"
    install -m 0755 {{ bin }} "$prefix/libexec/ishizuki/ishizuki"
    rm -rf "$prefix/libexec/ishizuki/mlx-swift_Cmlx.bundle"
    cp assets/ishizuki.icns "$prefix/libexec/ishizuki/"
    install -m 0755 Scripts/uninstall.sh "$prefix/libexec/ishizuki/uninstall.sh"
    printf '#!/bin/sh\nexec "$(dirname "$0")/../libexec/ishizuki/ishizuki" "$@"\n' > "$prefix/bin/ishizuki"
    chmod 0755 "$prefix/bin/ishizuki"
    echo "installed -> $prefix/bin/ishizuki"
    case ":$PATH:" in
      *":$prefix/bin:"*) : ;;
      *) echo "  add to PATH:  echo 'export PATH=\"$prefix/bin:\$PATH\"' >> ~/.zshrc" ;;
    esac
    echo "  uninstall: $prefix/libexec/ishizuki/uninstall.sh"

# uninstall, rootless
uninstall:
    #!/usr/bin/env bash
    set -euo pipefail
    prefix="{{ prefix }}"
    if [ -x "$prefix/libexec/ishizuki/uninstall.sh" ]; then
      "$prefix/libexec/ishizuki/uninstall.sh"
    else
      rm -f "$prefix/bin/ishizuki"
      rm -rf "$prefix/libexec/ishizuki"
      echo "uninstalled (models kept)"
    fi

# regenerate the Xcode project for the menu bar app from App/project.yml
app-project:
    cd App && xcodegen generate

# build the menu bar app (Debug) into App/.build
app: app-project
    cd App && xcodebuild -project Ishizuki.xcodeproj -scheme Ishizuki -configuration Debug \
      -derivedDataPath .build -skipPackagePluginValidation -skipMacroValidation build

# build and launch the menu bar app
app-run: app
    open App/.build/Build/Products/Debug/Ishizuki.app

# archive and export a Mac App Store upload (needs Apple Distribution + installer identities)
app-store: app-project
    ./Scripts/app-store.sh {{ version }}

# build a signed, notarized, stapled .pkg (needs identities from .env)
package: release
    ./Scripts/package.sh {{ version }}

# build a signed, notarized loose tarball (one self-contained binary)
tarball: release
    ./Scripts/tarball.sh {{ version }}

# both release artifacts
dist: package tarball

# remove build artifacts
clean:
    rm -rf .build
