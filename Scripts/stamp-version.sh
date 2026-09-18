#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Write the build version into BuildInfo.swift. Callers restore it afterwards:
#   git checkout -- Sources/IshizukiKit/BuildInfo.swift
set -euo pipefail

VERSION="${1:?usage: stamp-version.sh <version>}"
root_dir="$(cd "$(dirname "$0")/.." && pwd)"
target="$root_dir/Sources/IshizukiKit/BuildInfo.swift"

cat > "$target" <<SWIFT
// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

public enum BuildInfo {
  // Stamped by Scripts/stamp-version.sh during a release build; "dev" otherwise.
  public static let version = "$VERSION"
}
SWIFT
