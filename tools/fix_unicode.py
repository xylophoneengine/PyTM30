#!/usr/bin/env python3
"""Transliterate non-ASCII characters in source and docs to ASCII, in place.

Rationale: the repo is C++20 + Python with no locale assumptions; comments and
docstrings carrying "smart" typography (em dashes, curly quotes, box-drawing
banners, maths glyphs) render inconsistently across terminals, diff viewers and
editors, and are easy to introduce accidentally by copy-paste. This script
rewrites them to ASCII equivalents so every tracked text file stays 7-bit.

Single exception: U+00A7 SECTION SIGN. Spec citations are mandated in the form
`// TM-30-20 §x.y` and `tools/check_constants.py` matches that literal glyph in
its citation regexes (CITATION_LINE / CITATION_BLOCK). Rewriting it here would
silently break the spec-citation gate, so it is preserved.

Usage:
    python3 tools/fix_unicode.py                 # fix every in-scope tracked file
    python3 tools/fix_unicode.py FILE...         # fix the given files
    python3 tools/fix_unicode.py --check [FILE]  # report only, never write

Exit status:
    0  nothing to do
    1  files were rewritten (--check: would be rewritten), or an unmappable
       character was found and left in place

Pre-commit convention: a non-zero exit aborts the commit. Re-stage the rewritten
files (`git add`) and commit again.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------

# Text we author by hand. Everything else (CSV data tables, golden fixtures,
# notebooks, images, generated benchmark reports) is machine-written or upstream
# and is not ours to reformat.
INCLUDE_SUFFIXES = {
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".h",
    ".py",
    ".pyi",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".cmake",
    ".toml",
    ".cff",
}

# Paths excluded even when the suffix matches.
EXCLUDE_PATTERNS = (
    re.compile(r"^data/"),
    re.compile(r"^tests/fixtures/"),
    re.compile(r"^build/"),
    re.compile(r"^benchmarks/out_"),
    re.compile(r"^benchmarks/.*report.*\.txt$"),
    re.compile(r"^LICENSE$"),
)

# The one glyph we keep; see module docstring.
ALLOWED = {"§"}

# --------------------------------------------------------------------------
# Transliteration
# --------------------------------------------------------------------------

# Applied before the per-character table, where a plain 1:1 substitution would
# glue a word-shaped replacement onto the following token ("√n" -> "sqrtn",
# "∫St" -> "integralSt").
PRE_SUBS = (
    # A base letter followed by COMBINING MACRON is the CMF bar notation
    # (x̄, z̄): must become "xbar"/"zbar", not "x"/"z".
    (re.compile(r"([A-Za-z])̄"), r"\1bar"),
    # Radicand written without brackets: √n -> sqrt(n). Bracketed forms (√[...])
    # fall through to the plain table entry.
    (re.compile(r"√([A-Za-z0-9_]+)"), r"sqrt(\1)"),
    # Operators whose ASCII name needs separating from the expression that
    # follows. Subscripted forms (Σ_i, ∏_i) keep the underscore and need no space.
    (re.compile(r"∫\s*(?=\S)"), "integral "),
    (re.compile(r"Σ(?=[A-Za-z0-9(\[])"), "sum "),
    (re.compile(r"∑(?=[A-Za-z0-9(\[])"), "sum "),
    (re.compile(r"∏(?=[A-Za-z0-9(\[])"), "prod "),
)

# Domain shorthand that beats the generic Greek/symbol names below. These are
# chosen to read the way the surrounding prose already does: ΔE' is written dE',
# a summation is written sum, and so on.
OVERRIDES = {
    "Δ": "d",  # GREEK CAPITAL DELTA -> difference operator, e.g. dE', dlambda
    "Σ": "sum",  # GREEK CAPITAL SIGMA
    "∫": "integral",  # INTEGRAL
    "µ": "u",  # MICRO SIGN
    "μ": "u",  # GREEK SMALL MU (also used as micro)
    "ȳ": "ybar",  # LATIN SMALL Y WITH MACRON (precomposed CMF bar)
    "°": "-deg",  # DEGREE SIGN -- reads as the adjective it always is here
    #                              ("2° XYZ" -> "2-deg XYZ")
}

GREEK = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "ν": "nu",
    "ξ": "xi",
    "ο": "o",
    "π": "pi",
    "ρ": "rho",
    "ς": "sigma",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Α": "Alpha",
    "Β": "Beta",
    "Γ": "Gamma",
    "Ε": "Epsilon",
    "Ζ": "Zeta",
    "Η": "Eta",
    "Θ": "Theta",
    "Ι": "Iota",
    "Κ": "Kappa",
    "Λ": "Lambda",
    "Μ": "Mu",
    "Ν": "Nu",
    "Ξ": "Xi",
    "Ο": "Omicron",
    "Π": "Pi",
    "Ρ": "Rho",
    "Τ": "Tau",
    "Υ": "Upsilon",
    "Φ": "Phi",
    "Χ": "Chi",
    "Ψ": "Psi",
    "Ω": "Omega",
}

PUNCTUATION = {
    "—": "--",  # EM DASH
    "–": "-",  # EN DASH
    "−": "-",  # MINUS SIGN
    "‐": "-",  # HYPHEN
    "‑": "-",  # NON-BREAKING HYPHEN
    "‘": "'",  # LEFT SINGLE QUOTE
    "’": "'",  # RIGHT SINGLE QUOTE
    "‚": "'",  # SINGLE LOW-9 QUOTE
    "‛": "'",
    "′": "'",  # PRIME (J'a'b' notation)
    "″": "''",  # DOUBLE PRIME
    "“": '"',  # LEFT DOUBLE QUOTE
    "”": '"',  # RIGHT DOUBLE QUOTE
    "„": '"',  # DOUBLE LOW-9 QUOTE
    "«": "<<",
    "»": ">>",
    "…": "...",  # HORIZONTAL ELLIPSIS
    "•": "*",  # BULLET
    "·": "*",  # MIDDLE DOT (used as multiplication)
    "‧": "*",
    "†": "+",  # DAGGER
    "‡": "++",  # DOUBLE DAGGER
    " ": " ",  # NO-BREAK SPACE
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    "​": "",  # ZERO WIDTH SPACE
    "‌": "",
    "‍": "",
    "﻿": "",  # BOM / ZERO WIDTH NO-BREAK SPACE
    "­": "",  # SOFT HYPHEN
}

MATH = {
    "×": "x",  # MULTIPLICATION SIGN
    "÷": "/",  # DIVISION SIGN
    "±": "+/-",  # PLUS-MINUS
    "≈": "~=",  # ALMOST EQUAL TO
    "≃": "~=",
    "≅": "~=",
    "≤": "<=",  # LESS-THAN OR EQUAL
    "≥": ">=",  # GREATER-THAN OR EQUAL
    "≠": "!=",  # NOT EQUAL
    "≡": "==",  # IDENTICAL TO
    "√": "sqrt",  # SQUARE ROOT
    "∞": "inf",  # INFINITY
    "∂": "d",  # PARTIAL DIFFERENTIAL
    "∇": "grad",  # NABLA
    "∈": " in ",  # ELEMENT OF
    "∑": "sum",  # N-ARY SUMMATION
    "∏": "prod",  # N-ARY PRODUCT
    "½": "1/2",
    "¼": "1/4",
    "¾": "3/4",
    "⅓": "1/3",
    "⅔": "2/3",
    "‰": "o/oo",  # PER MILLE
}

ARROWS = {
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "⇒": "=>",
    "⇐": "<=",
    "⇔": "<=>",
    "↑": "^",
    "↓": "v",
}

# Banner separators and ASCII-art trees. Single-cell replacements keep the
# original column widths, so aligned banners stay aligned.
BOX = {
    "─": "-",
    "━": "-",
    "┄": "-",
    "┅": "-",
    "┈": "-",
    "┉": "-",
    "═": "=",
    "│": "|",
    "┃": "|",
    "┆": "|",
    "┊": "|",
    "║": "|",
    "┌": "+",
    "┏": "+",
    "╔": "+",
    "┐": "+",
    "┓": "+",
    "╗": "+",
    "└": "`",
    "┗": "`",
    "╚": "+",
    "┘": "+",
    "┛": "+",
    "╝": "+",
    "├": "|",
    "┣": "|",
    "╠": "|",
    "┤": "+",
    "┫": "+",
    "╣": "+",
    "┬": "+",
    "┴": "+",
    "┼": "+",
    "█": "#",
    "░": ".",
    "▒": ":",
    "▓": "=",
    "■": "#",
    "●": "*",
    "○": "o",
}

STATUS = {
    "❌": "[x]",  # CROSS MARK
    "✅": "[+]",  # WHITE HEAVY CHECK MARK
    "✓": "[+]",  # CHECK MARK
    "✔": "[+]",
    "✗": "[x]",  # BALLOT X
    "✘": "[x]",
    "⚠": "[!]",  # WARNING SIGN
    "️": "",  # VARIATION SELECTOR-16 (emoji presentation)
}

SUPERSCRIPTS = {
    "⁰": "^0",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
    "⁺": "^+",
    "⁻": "^-",
    "ⁿ": "^n",
}

SUBSCRIPTS = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₊": "+",
    "₋": "-",
}

# Later dicts win on key collisions; OVERRIDES is applied last so its domain
# shorthand beats the generic tables.
TABLE: dict[str, str] = {}
for _part in (
    GREEK,
    PUNCTUATION,
    MATH,
    ARROWS,
    BOX,
    STATUS,
    SUPERSCRIPTS,
    SUBSCRIPTS,
    OVERRIDES,
):
    TABLE.update(_part)


def transliterate(text: str) -> tuple[str, list[str]]:
    """Return (ascii_text, unmapped) — unmapped lists chars left non-ASCII."""
    for pattern, repl in PRE_SUBS:
        text = pattern.sub(repl, text)

    out: list[str] = []
    unmapped: list[str] = []
    for ch in text:
        if ord(ch) < 128 or ch in ALLOWED:
            out.append(ch)
            continue
        mapped = TABLE.get(ch)
        if mapped is None:
            # Last resort: NFKD compatibility decomposition drops accents and
            # expands ligatures. Only accepted if it lands entirely in ASCII.
            decomposed = unicodedata.normalize("NFKD", ch)
            stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
            mapped = stripped if stripped.isascii() and stripped else None
        if mapped is None:
            out.append(ch)
            unmapped.append(ch)
            continue
        out.append(mapped)
    return "".join(out), unmapped


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def in_scope(relpath: str) -> bool:
    """relpath is repo-relative; the exclude patterns are anchored to that."""
    if any(p.search(relpath) for p in EXCLUDE_PATTERNS):
        return False
    if Path(relpath).name == "CMakeLists.txt":
        return True
    return Path(relpath).suffix in INCLUDE_SUFFIXES


def path_in_scope(path: Path) -> bool:
    """Same check for an arbitrary path, inside the repo or not."""
    if path.is_relative_to(REPO_ROOT):
        return in_scope(str(path.relative_to(REPO_ROOT)))
    # Outside the repo the exclude patterns cannot apply; go by filename alone.
    return in_scope(path.name)


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [REPO_ROOT / p for p in out.split("\0") if p and in_scope(p)]


def describe(ch: str) -> str:
    return f"U+{ord(ch):04X} {unicodedata.name(ch, 'UNNAMED')}"


def display(path: Path) -> Path:
    """Repo-relative path where possible; absolute for files outside the repo."""
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def process(path: Path, check_only: bool) -> tuple[bool, list[str]]:
    """Return (changed, unmapped_report_lines)."""
    try:
        original = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False, []

    if original.isascii():
        return False, []

    fixed, unmapped = transliterate(original)

    report: list[str] = []
    if unmapped:
        rel = display(path)
        seen: set[str] = set()
        for lineno, line in enumerate(fixed.splitlines(), start=1):
            for ch in line:
                if ch in unmapped and (key := f"{lineno}:{ch}") not in seen:
                    seen.add(key)
                    report.append(f"  {rel}:{lineno}  {describe(ch)}  {ch!r}")

    changed = fixed != original
    if changed and not check_only:
        path.write_text(fixed, encoding="utf-8")
    return changed, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="files to fix (default: tracked files)")
    ap.add_argument(
        "--check",
        action="store_true",
        help="report without writing",
    )
    args = ap.parse_args()

    if args.files:
        paths = [
            p
            for p in (Path(f).resolve() for f in args.files)
            if p.is_file() and path_in_scope(p)
        ]
    else:
        paths = tracked_files()

    changed: list[Path] = []
    unmapped_report: list[str] = []
    for path in paths:
        did_change, report = process(path, args.check)
        if did_change:
            changed.append(path)
        unmapped_report.extend(report)

    verb = "would rewrite" if args.check else "rewrote"
    for path in changed:
        print(f"{verb}: {display(path)}")

    if unmapped_report:
        print(
            f"\nERROR: {len(unmapped_report)} non-ASCII character(s) with no ASCII "
            f"mapping. Replace by hand, or add a mapping in {Path(__file__).name}:",
            file=sys.stderr,
        )
        for line in unmapped_report:
            print(line, file=sys.stderr)

    if changed and not args.check:
        print(
            f"\n{len(changed)} file(s) rewritten to ASCII. Re-stage and commit again.",
            file=sys.stderr,
        )

    return 1 if (changed or unmapped_report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
