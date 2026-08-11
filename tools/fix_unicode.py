#!/usr/bin/env python3
"""Transliterate non-ASCII characters in source and docs to ASCII, in place.

Rationale: the repo is C++20 + Python with no locale assumptions; comments and
docstrings carrying "smart" typography (em dashes, curly quotes, box-drawing
banners, maths glyphs) render inconsistently across terminals, diff viewers and
editors, and are easy to introduce accidentally by copy-paste. This script
rewrites them to ASCII equivalents so every tracked text file stays 7-bit.

Single exception: U+00A7 SECTION SIGN. Spec citations are mandated in the form
`// TM-30-20` + U+00A7 + `x.y` and `tools/check_constants.py` matches that
literal glyph in its citation regexes (CITATION_LINE / CITATION_BLOCK).
Rewriting it here would silently break the spec-citation gate, so it is kept.

This file is in its own scope (`.py`, not excluded), so it must never contain a
literal instance of a glyph it maps -- running the hook over the tree would
rewrite the mapping tables into `"d": "d"` and destroy the tool. Every key and
pattern below is therefore written as a `\\uXXXX` escape with the Unicode name in
a trailing comment, and comments name glyphs rather than showing them. That
keeps the file a fixed point of its own transformation; `tools/test_fix_unicode.py`
asserts it. Note that `re` interprets `\\uXXXX` inside a raw pattern string, so
the PRE_SUBS patterns stay raw for their regex backslashes.

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
ALLOWED = {"\u00a7"}  # SECTION SIGN

# --------------------------------------------------------------------------
# Transliteration
# --------------------------------------------------------------------------

# Applied before the per-character table, where a plain 1:1 substitution would
# glue a word-shaped replacement onto the following token (SQUARE ROOT + "n"
# would become "sqrtn"; INTEGRAL + "St" would become "integralSt").
PRE_SUBS = (
    # A base letter followed by COMBINING MACRON is the CMF bar notation
    # (x-bar, z-bar): must become "xbar"/"zbar", not "x"/"z".
    (re.compile(r"([A-Za-z])\u0304"), r"\1bar"),  # COMBINING MACRON
    # Radicand written without brackets: SQUARE ROOT + n -> sqrt(n). Bracketed
    # forms (SQUARE ROOT followed by "[") fall through to the plain table entry.
    (re.compile(r"\u221a([A-Za-z0-9_]+)"), r"sqrt(\1)"),  # SQUARE ROOT
    # Operators whose ASCII name needs separating from the expression that
    # follows. Subscripted forms (SIGMA or N-ARY PRODUCT followed by "_i") keep
    # the underscore and need no space.
    (re.compile(r"\u222b\s*(?=\S)"), "integral "),  # INTEGRAL
    (re.compile(r"\u03a3(?=[A-Za-z0-9(\[])"), "sum "),  # GREEK CAPITAL SIGMA
    (re.compile(r"\u2211(?=[A-Za-z0-9(\[])"), "sum "),  # N-ARY SUMMATION
    (re.compile(r"\u220f(?=[A-Za-z0-9(\[])"), "prod "),  # N-ARY PRODUCT
)

# Domain shorthand that beats the generic Greek/symbol names below. These are
# chosen to read the way the surrounding prose already does: CAPITAL DELTA + E'
# is written dE', a summation is written sum, and so on.
OVERRIDES = {
    "\u0394": "d",  # GREEK CAPITAL DELTA -> difference operator: dE', dlambda
    "\u03a3": "sum",  # GREEK CAPITAL SIGMA
    "\u222b": "integral",  # INTEGRAL
    "\u00b5": "u",  # MICRO SIGN
    "\u03bc": "u",  # GREEK SMALL MU (also used as micro)
    "\u0233": "ybar",  # LATIN SMALL Y WITH MACRON (precomposed CMF bar)
    "\u00b0": "-deg",  # DEGREE SIGN -- reads as the adjective it always is here
    #                    ("2" + DEGREE SIGN + " XYZ" -> "2-deg XYZ")
}

GREEK = {
    "\u03b1": "alpha",  # GREEK SMALL ALPHA
    "\u03b2": "beta",  # GREEK SMALL BETA
    "\u03b3": "gamma",  # GREEK SMALL GAMMA
    "\u03b4": "delta",  # GREEK SMALL DELTA
    "\u03b5": "epsilon",  # GREEK SMALL EPSILON
    "\u03b6": "zeta",  # GREEK SMALL ZETA
    "\u03b7": "eta",  # GREEK SMALL ETA
    "\u03b8": "theta",  # GREEK SMALL THETA
    "\u03b9": "iota",  # GREEK SMALL IOTA
    "\u03ba": "kappa",  # GREEK SMALL KAPPA
    "\u03bb": "lambda",  # GREEK SMALL LAMDA
    "\u03bd": "nu",  # GREEK SMALL NU
    "\u03be": "xi",  # GREEK SMALL XI
    "\u03bf": "o",  # GREEK SMALL OMICRON
    "\u03c0": "pi",  # GREEK SMALL PI
    "\u03c1": "rho",  # GREEK SMALL RHO
    "\u03c2": "sigma",  # GREEK SMALL FINAL SIGMA
    "\u03c3": "sigma",  # GREEK SMALL SIGMA
    "\u03c4": "tau",  # GREEK SMALL TAU
    "\u03c5": "upsilon",  # GREEK SMALL UPSILON
    "\u03c6": "phi",  # GREEK SMALL PHI
    "\u03c7": "chi",  # GREEK SMALL CHI
    "\u03c8": "psi",  # GREEK SMALL PSI
    "\u03c9": "omega",  # GREEK SMALL OMEGA
    "\u0391": "Alpha",  # GREEK CAPITAL ALPHA
    "\u0392": "Beta",  # GREEK CAPITAL BETA
    "\u0393": "Gamma",  # GREEK CAPITAL GAMMA
    "\u0395": "Epsilon",  # GREEK CAPITAL EPSILON
    "\u0396": "Zeta",  # GREEK CAPITAL ZETA
    "\u0397": "Eta",  # GREEK CAPITAL ETA
    "\u0398": "Theta",  # GREEK CAPITAL THETA
    "\u0399": "Iota",  # GREEK CAPITAL IOTA
    "\u039a": "Kappa",  # GREEK CAPITAL KAPPA
    "\u039b": "Lambda",  # GREEK CAPITAL LAMDA
    "\u039c": "Mu",  # GREEK CAPITAL MU
    "\u039d": "Nu",  # GREEK CAPITAL NU
    "\u039e": "Xi",  # GREEK CAPITAL XI
    "\u039f": "Omicron",  # GREEK CAPITAL OMICRON
    "\u03a0": "Pi",  # GREEK CAPITAL PI
    "\u03a1": "Rho",  # GREEK CAPITAL RHO
    "\u03a4": "Tau",  # GREEK CAPITAL TAU
    "\u03a5": "Upsilon",  # GREEK CAPITAL UPSILON
    "\u03a6": "Phi",  # GREEK CAPITAL PHI
    "\u03a7": "Chi",  # GREEK CAPITAL CHI
    "\u03a8": "Psi",  # GREEK CAPITAL PSI
    "\u03a9": "Omega",  # GREEK CAPITAL OMEGA
}

PUNCTUATION = {
    "\u2014": "--",  # EM DASH
    "\u2013": "-",  # EN DASH
    "\u2212": "-",  # MINUS SIGN
    "\u2010": "-",  # HYPHEN
    "\u2011": "-",  # NON-BREAKING HYPHEN
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u201a": "'",  # SINGLE LOW-9 QUOTATION MARK
    "\u201b": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2032": "'",  # PRIME (J'a'b' notation)
    "\u2033": "''",  # DOUBLE PRIME
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
    "\u201e": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "\u00ab": "<<",  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    "\u00bb": ">>",  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    "\u2026": "...",  # HORIZONTAL ELLIPSIS
    "\u2022": "*",  # BULLET
    "\u00b7": "*",  # MIDDLE DOT (used as multiplication)
    "\u2027": "*",  # HYPHENATION POINT
    "\u2020": "+",  # DAGGER
    "\u2021": "++",  # DOUBLE DAGGER
    "\u00a0": " ",  # NO-BREAK SPACE
    "\u2007": " ",  # FIGURE SPACE
    "\u2009": " ",  # THIN SPACE
    "\u202f": " ",  # NARROW NO-BREAK SPACE
    "\u200a": " ",  # HAIR SPACE
    "\u2002": " ",  # EN SPACE
    "\u2003": " ",  # EM SPACE
    "\u200b": "",  # ZERO WIDTH SPACE
    "\u200c": "",  # ZERO WIDTH NON-JOINER
    "\u200d": "",  # ZERO WIDTH JOINER
    "\ufeff": "",  # BOM / ZERO WIDTH NO-BREAK SPACE
    "\u00ad": "",  # SOFT HYPHEN
}

MATH = {
    "\u00d7": "x",  # MULTIPLICATION SIGN
    "\u00f7": "/",  # DIVISION SIGN
    "\u00b1": "+/-",  # PLUS-MINUS SIGN
    "\u2248": "~=",  # ALMOST EQUAL TO
    "\u2243": "~=",  # ASYMPTOTICALLY EQUAL TO
    "\u2245": "~=",  # APPROXIMATELY EQUAL TO
    "\u2264": "<=",  # LESS-THAN OR EQUAL TO
    "\u2265": ">=",  # GREATER-THAN OR EQUAL TO
    "\u2260": "!=",  # NOT EQUAL TO
    "\u2261": "==",  # IDENTICAL TO
    "\u221a": "sqrt",  # SQUARE ROOT
    "\u221e": "inf",  # INFINITY
    "\u2202": "d",  # PARTIAL DIFFERENTIAL
    "\u2207": "grad",  # NABLA
    "\u2208": " in ",  # ELEMENT OF
    "\u2211": "sum",  # N-ARY SUMMATION
    "\u220f": "prod",  # N-ARY PRODUCT
    "\u00bd": "1/2",  # VULGAR FRACTION ONE HALF
    "\u00bc": "1/4",  # VULGAR FRACTION ONE QUARTER
    "\u00be": "3/4",  # VULGAR FRACTION THREE QUARTERS
    "\u2153": "1/3",  # VULGAR FRACTION ONE THIRD
    "\u2154": "2/3",  # VULGAR FRACTION TWO THIRDS
    "\u2030": "o/oo",  # PER MILLE SIGN
}

ARROWS = {
    "\u2192": "->",  # RIGHTWARDS ARROW
    "\u2190": "<-",  # LEFTWARDS ARROW
    "\u2194": "<->",  # LEFT RIGHT ARROW
    "\u21d2": "=>",  # RIGHTWARDS DOUBLE ARROW
    "\u21d0": "<=",  # LEFTWARDS DOUBLE ARROW
    "\u21d4": "<=>",  # LEFT RIGHT DOUBLE ARROW
    "\u2191": "^",  # UPWARDS ARROW
    "\u2193": "v",  # DOWNWARDS ARROW
}

# Banner separators and ASCII-art trees. Single-cell replacements keep the
# original column widths, so aligned banners stay aligned.
BOX = {
    "\u2500": "-",  # BOX DRAWINGS LIGHT HORIZONTAL
    "\u2501": "-",  # BOX DRAWINGS HEAVY HORIZONTAL
    "\u2504": "-",  # BOX DRAWINGS LIGHT TRIPLE DASH HORIZONTAL
    "\u2505": "-",  # BOX DRAWINGS HEAVY TRIPLE DASH HORIZONTAL
    "\u2508": "-",  # BOX DRAWINGS LIGHT QUADRUPLE DASH HORIZONTAL
    "\u2509": "-",  # BOX DRAWINGS HEAVY QUADRUPLE DASH HORIZONTAL
    "\u2550": "=",  # BOX DRAWINGS DOUBLE HORIZONTAL
    "\u2502": "|",  # BOX DRAWINGS LIGHT VERTICAL
    "\u2503": "|",  # BOX DRAWINGS HEAVY VERTICAL
    "\u2506": "|",  # BOX DRAWINGS LIGHT TRIPLE DASH VERTICAL
    "\u250a": "|",  # BOX DRAWINGS LIGHT QUADRUPLE DASH VERTICAL
    "\u2551": "|",  # BOX DRAWINGS DOUBLE VERTICAL
    "\u250c": "+",  # BOX DRAWINGS LIGHT DOWN AND RIGHT
    "\u250f": "+",  # BOX DRAWINGS HEAVY DOWN AND RIGHT
    "\u2554": "+",  # BOX DRAWINGS DOUBLE DOWN AND RIGHT
    "\u2510": "+",  # BOX DRAWINGS LIGHT DOWN AND LEFT
    "\u2513": "+",  # BOX DRAWINGS HEAVY DOWN AND LEFT
    "\u2557": "+",  # BOX DRAWINGS DOUBLE DOWN AND LEFT
    "\u2514": "`",  # BOX DRAWINGS LIGHT UP AND RIGHT
    "\u2517": "`",  # BOX DRAWINGS HEAVY UP AND RIGHT
    "\u255a": "+",  # BOX DRAWINGS DOUBLE UP AND RIGHT
    "\u2518": "+",  # BOX DRAWINGS LIGHT UP AND LEFT
    "\u251b": "+",  # BOX DRAWINGS HEAVY UP AND LEFT
    "\u255d": "+",  # BOX DRAWINGS DOUBLE UP AND LEFT
    "\u251c": "|",  # BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    "\u2523": "|",  # BOX DRAWINGS HEAVY VERTICAL AND RIGHT
    "\u2560": "|",  # BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    "\u2524": "+",  # BOX DRAWINGS LIGHT VERTICAL AND LEFT
    "\u252b": "+",  # BOX DRAWINGS HEAVY VERTICAL AND LEFT
    "\u2563": "+",  # BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    "\u252c": "+",  # BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    "\u2534": "+",  # BOX DRAWINGS LIGHT UP AND HORIZONTAL
    "\u253c": "+",  # BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    "\u2588": "#",  # FULL BLOCK
    "\u2591": ".",  # LIGHT SHADE
    "\u2592": ":",  # MEDIUM SHADE
    "\u2593": "=",  # DARK SHADE
    "\u25a0": "#",  # BLACK SQUARE
    "\u25cf": "*",  # BLACK CIRCLE
    "\u25cb": "o",  # WHITE CIRCLE
}

STATUS = {
    "\u274c": "[x]",  # CROSS MARK
    "\u2705": "[+]",  # WHITE HEAVY CHECK MARK
    "\u2713": "[+]",  # CHECK MARK
    "\u2714": "[+]",  # HEAVY CHECK MARK
    "\u2717": "[x]",  # BALLOT X
    "\u2718": "[x]",  # HEAVY BALLOT X
    "\u26a0": "[!]",  # WARNING SIGN
    "\ufe0f": "",  # VARIATION SELECTOR-16 (emoji presentation)
}

SUPERSCRIPTS = {
    "\u2070": "^0",  # SUPERSCRIPT ZERO
    "\u00b9": "^1",  # SUPERSCRIPT ONE
    "\u00b2": "^2",  # SUPERSCRIPT TWO
    "\u00b3": "^3",  # SUPERSCRIPT THREE
    "\u2074": "^4",  # SUPERSCRIPT FOUR
    "\u2075": "^5",  # SUPERSCRIPT FIVE
    "\u2076": "^6",  # SUPERSCRIPT SIX
    "\u2077": "^7",  # SUPERSCRIPT SEVEN
    "\u2078": "^8",  # SUPERSCRIPT EIGHT
    "\u2079": "^9",  # SUPERSCRIPT NINE
    "\u207a": "^+",  # SUPERSCRIPT PLUS SIGN
    "\u207b": "^-",  # SUPERSCRIPT MINUS
    "\u207f": "^n",  # SUPERSCRIPT LATIN SMALL N
}

SUBSCRIPTS = {
    "\u2080": "0",  # SUBSCRIPT ZERO
    "\u2081": "1",  # SUBSCRIPT ONE
    "\u2082": "2",  # SUBSCRIPT TWO
    "\u2083": "3",  # SUBSCRIPT THREE
    "\u2084": "4",  # SUBSCRIPT FOUR
    "\u2085": "5",  # SUBSCRIPT FIVE
    "\u2086": "6",  # SUBSCRIPT SIX
    "\u2087": "7",  # SUBSCRIPT SEVEN
    "\u2088": "8",  # SUBSCRIPT EIGHT
    "\u2089": "9",  # SUBSCRIPT NINE
    "\u208a": "+",  # SUBSCRIPT PLUS SIGN
    "\u208b": "-",  # SUBSCRIPT MINUS
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
    """Return (ascii_text, unmapped) -- unmapped lists chars left non-ASCII."""
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
