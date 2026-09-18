#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Uninstall ishizuki (self-locating; rootless for a ~/.local install). --purge also deletes models.
set -euo pipefail

PURGE=0
[ "${1:-}" = "--purge" ] && PURGE=1

self_dir="$(cd "$(dirname "$0")" && pwd)"
libexec="$(dirname "$self_dir")"
PREFIX="$(dirname "$libexec")"
BIN="$PREFIX/bin/ishizuki"
PKG_ID="studio.ishizuki"
AGENT_LABEL="studio.ishizuki.server"

if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
  as_user="$SUDO_USER"; user_home="$(eval echo "~$SUDO_USER")"; run_as=(sudo -u "$SUDO_USER")
else
  as_user="$(id -un)"; user_home="$HOME"; run_as=()
fi

if [ ! -w "$PREFIX/bin" ] || [ ! -w "$libexec" ]; then
  echo "No write access to $PREFIX — re-run with sudo:"
  echo "  sudo $0 ${*:-}"
  exit 1
fi

echo "Uninstalling ishizuki from $PREFIX ..."

plist="$user_home/Library/LaunchAgents/${AGENT_LABEL}.plist"
if [ -f "$plist" ]; then
  uid="$(id -u "$as_user")"
  "${run_as[@]}" launchctl bootout "gui/${uid}/${AGENT_LABEL}" 2>/dev/null || true
  rm -f "$plist"
  echo "  • removed launchd agent"
fi

rc="$user_home/.zshrc"
if [ -f "$rc" ] && grep -qF "# >>> ishizuki PATH >>>" "$rc" 2>/dev/null; then
  tmp="$(mktemp)"
  sed '/# >>> ishizuki PATH >>>/,/# <<< ishizuki PATH <<</d' "$rc" > "$tmp" && cat "$tmp" > "$rc"
  rm -f "$tmp"
  echo "  • removed PATH line from ~/.zshrc"
fi

rm -f "$BIN"
rm -rf "$self_dir"
echo "  • removed $BIN and $self_dir"

pkgutil --forget "$PKG_ID" >/dev/null 2>&1 || "${run_as[@]}" pkgutil --forget "$PKG_ID" >/dev/null 2>&1 || true

data="$user_home/Library/Application Support/Ishizuki"
if [ -d "$data" ]; then
  if [ "$PURGE" -eq 1 ]; then
    rm -rf "$data"
    echo "  • purged models at $data"
  else
    size="$(du -sh "$data" 2>/dev/null | cut -f1 || echo '?')"
    echo "  • kept models at $data ($size) — remove with: rm -rf \"$data\""
  fi
fi

echo "Done."
