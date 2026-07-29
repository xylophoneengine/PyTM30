#!/usr/bin/env python3
"""check_constants.py - CI gate: every float literal in src/ and include/ must cite a spec section.

Fails on any float literal in the monitored directories that lacks a
`// TM-30-20 §x.y` (or `/* TM-30-20 §x.y */`) citation on or above its line.

Usage:
    python3 tools/check_constants.py              # check all float literals
    python3 tools/check_constants.py --verbose    # show passing citations too
    python3 tools/check_constants.py --whitelist  # show whitelist contents

Whitelist: tools/check_constants_whitelist.txt (maintainer-controlled)
One entry per line: "filepath:lineno" - float literals exempted by the maintainer.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = [
    REPO_ROOT / "src",
    REPO_ROOT / "include",
]
WHITELIST_FILE = Path(__file__).resolve().parent / "check_constants_whitelist.txt"

# Float literal patterns to detect
# Matches: 3.14, 0.5, 1.0e-5, .5, 1e5, 1.0f, 2.0e-3f, 42 (integers used as doubles in array init)
# We don't flag integers (42, 0, 1) unless they appear in a floating-point context
FLOAT_PATTERN = re.compile(
    r"(?<![a-zA-Z_0-9])"  # not part of an identifier
    r"(?:"
    r"\d+\.\d*(?:[eE][+-]?\d+)?[fFlL]?"  # 3.14, 0.5, 1e5
    r"|"
    r"\.\d+(?:[eE][+-]?\d+)?[fFlL]?"  # .5
    r"|"
    r"\d+[eE][+-]?\d+[fFlL]?"  # 1e5
    r")"
    r"(?![a-zA-Z_0-9.])"  # not part of another number
)

# Citation patterns
CITATION_LINE = re.compile(r"//\s*TM-30-20\s+§\s*[\d.]+[a-z]?")
CITATION_BLOCK = re.compile(r"/\*\s*TM-30-20\s+§\s*[\d.]+[a-z]?\s*\*/")

# Lines to ignore (comments, preprocessor, string literals - though we skip those separately)
IGNORE_LINE = re.compile(r"^\s*//|^\s*#|^\s*\*")


def load_whitelist():
    """Load whitelisted (file, lineno) pairs."""
    whitelist = set()
    if WHITELIST_FILE.exists():
        with open(WHITELIST_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.rsplit(":", 1)
                    if len(parts) == 2:
                        whitelist.add((parts[0], int(parts[1])))
    return whitelist


def find_citation(lines, idx, window=5):
    """Look backward from idx for a citation comment."""
    start = max(0, idx - window)
    for i in range(idx, start - 1, -1):
        line = lines[i]
        if CITATION_LINE.search(line) or CITATION_BLOCK.search(line):
            return True, line.strip()
    return False, None


def check_file(filepath, whitelist, verbose=False):
    """Check one source file for uncited float literals. Returns list of violations."""
    violations = []
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except Exception as e:
        return [(str(filepath), 0, f"ERROR reading file: {e}", "")]

    relpath = str(filepath.relative_to(REPO_ROOT))

    for lineno, line in enumerate(lines, start=1):
        # Skip pure comment lines and preprocessor directives
        if IGNORE_LINE.match(line):
            continue

        # Remove inline comments and string literals for float detection
        code_part = line
        # Strip // comments
        comment_pos = code_part.find("//")
        if comment_pos >= 0:
            # But check if the comment itself contains a citation
            comment_text = code_part[comment_pos:]
            code_part = code_part[:comment_pos]

        # Skip string literals (rough)
        code_part = re.sub(r'".*?"', '""', code_part)
        code_part = re.sub(r"'.*?'", "''", code_part)

        # Find all float literals
        for match in FLOAT_PATTERN.finditer(code_part):
            value = match.group()

            # Check whitelist
            if (relpath, lineno) in whitelist:
                if verbose:
                    print(f"  [WHITELISTED] {relpath}:{lineno}  {value!r}")
                continue

            # Check for citation on this line or recent previous lines
            has_citation, cite_line = find_citation(lines, lineno - 1)

            if has_citation:
                if verbose:
                    print(f"  [OK] {relpath}:{lineno}  {value!r}  ← cited")
            else:
                violations.append((relpath, lineno, value, line.rstrip()))

    return violations


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    show_whitelist = "--whitelist" in sys.argv

    whitelist = load_whitelist()

    if show_whitelist:
        if whitelist:
            print("Whitelist entries (maintainer-controlled):")
            for path, lineno in sorted(whitelist):
                print(f"  {path}:{lineno}")
        else:
            print("Whitelist is empty.")
        return 0

    if not SOURCE_DIRS[0].exists():
        print(f"Source directory not found: {SOURCE_DIRS[0]}")
        print("(This is expected before any slices are implemented.)")
        return 0

    # Collect all source files
    source_files = []
    for sdir in SOURCE_DIRS:
        if sdir.exists():
            for ext in [".hpp", ".h", ".cpp", ".cc", ".cxx"]:
                source_files.extend(sdir.rglob(f"*{ext}"))

    if not source_files:
        print("No source files found in src/ or include/.")
        print("(This is expected before any slices are implemented.)")
        return 0

    source_files.sort()

    all_violations = []
    files_checked = 0
    constants_checked = 0

    for filepath in source_files:
        violations = check_file(filepath, whitelist, verbose)
        all_violations.extend(violations)
        files_checked += 1

        if verbose:
            print(f"\n{filepath.relative_to(REPO_ROOT)}:")
            if not violations:
                print("  [PASS] all constants cited")

    # Report
    print(f"\n{'=' * 60}")
    print("check_constants.py - Float Literal Citation Audit")
    print(f"{'=' * 60}")
    print(f"Files checked: {files_checked}")
    print(f"Violations:    {len(all_violations)}")

    if all_violations:
        print(f"\n❌ FAIL: {len(all_violations)} uncited float literal(s) found.\n")

        # Group by file
        by_file = defaultdict(list)
        for path, lineno, value, line in all_violations:
            by_file[path].append((lineno, value, line))

        for path in sorted(by_file):
            print(f"\n{path}:")
            for lineno, value, line in by_file[path]:
                print(f"  line {lineno}: {value!r}")
                print(f"    {line[:100]}")

        print("\nEach float literal in src/ or include/ must have a citation")
        print("in a comment on or above its line, e.g.:")
        print("  // TM-30-20 §3.7.1")
        print("  double surround_f = 0.69;")
        print("\nAdd the citation, or if the constant genuinely doesn't need one,")
        print(f"the maintainer can whitelist it in {WHITELIST_FILE}")
        return 1
    else:
        print("\n✅ PASS: All float constants cite a TM-30-20 section.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
