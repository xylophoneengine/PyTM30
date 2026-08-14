#!/usr/bin/env python3
"""check_constants.py - CI gate: every float literal in src/ and include/ must cite a spec section.

Fails on any float literal in the monitored directories that lacks a
`// TM-30-20 §x.y` (or `/* TM-30-20 §x.y */`) citation on or above its line.

Also validates the citation tags themselves against
tools/tm30_clause_equations.txt: a `TM-30-20 §x.y` tag must name a known
clause, and any `Eq. (nn)` on the same line must fall inside a clause
cited on that line. Clause entries marked 'unverified' in the map file
produce warnings rather than failures until the maintainer confirms them
against the standard.

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

# --- Clause-reference validation ---
CLAUSE_MAP_FILE = Path(__file__).resolve().parent / "tm30_clause_equations.txt"
# Clause list attached to a TM-30-20 token ("TM-30-20 §4.6, §4.7" or
# "§4.6-§4.8"); a "§" belonging to another document (e.g. "CIE 15:2004
# §8.2.1" on the same line) is not captured.
TM30_CLAUSES = re.compile(r"TM-30-20\s*(§\s*[\d.]+\d(?:\s*[-,/]\s*§\s*[\d.]+\d)*)")
CLAUSE_NUM = re.compile(r"§\s*([\d.]+\d)")
EQ_REF = re.compile(r"Eqs?\.?\s*\(?(\d+)\)?(?:\s*[-,]\s*\(?(\d+)\)?)?")


def load_clause_map():
    """Load {clause: (eq_lo, eq_hi, verified)} with (None, None, v) for 'none'."""
    clause_map = {}
    if not CLAUSE_MAP_FILE.exists():
        return None
    with open(CLAUSE_MAP_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            clause, _, spec = line.partition(":")
            # Strip the trailing '# ...' clause-title comment before
            # tokenising, so titles never affect range or 'unverified'
            # parsing.
            spec = spec.partition("#")[0]
            parts = spec.split()
            if not parts:
                continue
            verified = "unverified" not in parts
            rng = parts[0]
            if rng == "none":
                lo = hi = None
            elif "-" in rng:
                lo, hi = (int(x) for x in rng.split("-", 1))
            else:
                lo = hi = int(rng)
            clause_map[clause.strip()] = (lo, hi, verified)
    return clause_map


def check_clause_refs(filepath, clause_map):
    """Validate every 'TM-30-20 §x.y' tag (and Eq. numbers on its line).

    Returns (violations, warnings): lists of (relpath, lineno, message).
    A violation is an unknown clause, or an equation number outside the
    range of every clause cited on that line when all those clauses are
    verified. If any clause involved is unverified, the finding is
    downgraded to a warning.
    """
    violations = []
    warnings = []
    relpath = str(filepath.relative_to(REPO_ROOT))
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except Exception as e:
        return [(relpath, 0, f"ERROR reading file: {e}")], []

    for lineno, line in enumerate(lines, start=1):
        if "TM-30-20" not in line:
            continue
        clauses = []
        for group in TM30_CLAUSES.findall(line):
            clauses.extend(CLAUSE_NUM.findall(group))
        if not clauses:
            continue

        line_verified = True
        known_clauses = []
        for clause in clauses:
            if clause not in clause_map:
                violations.append(
                    (relpath, lineno, f"unknown clause §{clause}")
                )
                continue
            known_clauses.append(clause)
            if not clause_map[clause][2]:
                line_verified = False

        if not known_clauses:
            continue

        for m in EQ_REF.finditer(line):
            eq_nums = [int(m.group(1))]
            if m.group(2):
                eq_nums.append(int(m.group(2)))
            for eq in eq_nums:
                in_range = any(
                    clause_map[c][0] is not None
                    and clause_map[c][0] <= eq <= clause_map[c][1]
                    for c in known_clauses
                )
                if not in_range:
                    msg = (
                        f"Eq. ({eq}) is outside every clause cited on this "
                        f"line ({', '.join('§' + c for c in known_clauses)})"
                    )
                    if line_verified:
                        violations.append((relpath, lineno, msg))
                    else:
                        warnings.append((relpath, lineno, msg + " [unverified clause]"))
    return violations, warnings


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
                    print(f"  [OK] {relpath}:{lineno}  {value!r}  <- cited")
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

    clause_map = load_clause_map()
    clause_violations = []
    clause_warnings = []
    unverified_count = (
        sum(1 for v in clause_map.values() if not v[2]) if clause_map else 0
    )

    for filepath in source_files:
        violations = check_file(filepath, whitelist, verbose)
        all_violations.extend(violations)
        files_checked += 1

        if clause_map is not None:
            cv, cw = check_clause_refs(filepath, clause_map)
            clause_violations.extend(cv)
            clause_warnings.extend(cw)

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

    if clause_map is None:
        print("\n[!] Clause map tools/tm30_clause_equations.txt not found;")
        print("    clause-reference validation skipped.")
    else:
        print(f"Clause-reference violations: {len(clause_violations)}"
              f"  (warnings: {len(clause_warnings)};"
              f" unverified map entries: {unverified_count})")
        for relpath, lineno, msg in clause_warnings:
            print(f"  [warn] {relpath}:{lineno}: {msg}")
        if clause_violations:
            print("\n[x] FAIL: invalid TM-30-20 clause reference(s):")
            for relpath, lineno, msg in clause_violations:
                print(f"  {relpath}:{lineno}: {msg}")
            print("\nEither the citation tag is wrong, or the clause map")
            print(f"({CLAUSE_MAP_FILE}) is incomplete -- verify against the")
            print("standard before editing the map.")
            return 1

    if all_violations:
        print(f"\n[x] FAIL: {len(all_violations)} uncited float literal(s) found.\n")

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
        print("\n[+] PASS: All float constants cite a TM-30-20 section.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
