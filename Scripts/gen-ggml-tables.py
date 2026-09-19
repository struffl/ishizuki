#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Lifts the i-quant codebooks out of llama.cpp's ggml-common.h into Swift.

import re
import sys
from pathlib import Path

TABLES = [
    ("kmask_iq2xs", "UInt8"),
    ("ksigns_iq2xs", "UInt8"),
    ("iq2xxs_grid", "UInt64"),
    ("iq2xs_grid", "UInt64"),
    ("iq2s_grid", "UInt64"),
    ("iq3xxs_grid", "UInt32"),
    ("iq3s_grid", "UInt32"),
    ("iq1s_grid", "UInt64"),
    ("kvalues_iq4nl", "Int8"),
]

PER_LINE = {"UInt64": 4, "UInt32": 6, "UInt8": 16, "Int8": 16}


def extract(source: str, name: str) -> list[str]:
    pattern = re.compile(
        r"GGML_TABLE_BEGIN\(\s*\w+\s*,\s*" + name + r"\s*,[^)]*\)(.*?)GGML_TABLE_END\(\)",
        re.S,
    )
    match = pattern.search(source)
    if not match:
        sys.exit(f"table {name} not found")
    body = re.sub(r"//[^\n]*", "", match.group(1))
    return [token.strip() for token in body.split(",") if token.strip()]


def literal(token: str, swift_type: str) -> str:
    if token.lower().startswith("0x"):
        return token
    value = int(token)
    if swift_type == "Int8" and value > 127:
        value -= 256
    return str(value)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: gen-ggml-tables.py <ggml-common.h> <output.swift>")
    source = Path(sys.argv[1]).read_text()

    out = [
        "// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>",
        "// SPDX-FileCopyrightText: 2023-2026 The ggml authors",
        "// SPDX-License-Identifier: AGPL-3.0-or-later",
        "//",
        "// The i-quant codebooks, generated from llama.cpp's ggml-common.h by",
        "// Scripts/gen-ggml-tables.py. Do not edit by hand.",
        "",
        "/// Lookup tables the GGML i-quants index into. A block stores grid indices and sign",
        "/// bits rather than quantized values, so these are part of the format, not a cache.",
        "public enum GGMLTables {",
    ]

    for name, swift_type in TABLES:
        values = extract(source, name)
        out.append(f"  public static let {name}: [{swift_type}] = [")
        width = PER_LINE[swift_type]
        for start in range(0, len(values), width):
            chunk = values[start : start + width]
            out.append("    " + ", ".join(literal(v, swift_type) for v in chunk) + ",")
        out.append("  ]")
        out.append("")
        print(f"{name}: {len(values)} entries", file=sys.stderr)

    out.append("}")
    Path(sys.argv[2]).write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
