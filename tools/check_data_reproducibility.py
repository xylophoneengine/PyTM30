#!/usr/bin/env python3
"""
check_data_reproducibility.py -- Numeric-equivalence gate for data/*.csv.

Compares a committed snapshot of the data tables against a freshly
regenerated set. Byte-identity across platforms is not attainable for
these tables: the generators go through numpy and libm, whose last-ULP
rounding differs between architectures (the shipped tables are
byte-reproducible on the generating platform, macOS arm64 -- see
PROVENANCE.md). What IS platform-independent is the numeric content, so
this gate enforces:

  - identical file sets, headers, and row counts,
  - integer wavelength grids exactly equal,
  - every other value within MAX_ULP units in the last place, or within
    ABS_TOL absolutely (for near-zero values produced by cancellation,
    e.g. the daylight-basis S1/S2 columns at 560 nm, where a handful of
    1e-16-magnitude results make ULP distance meaningless).

Bounds are calibrated from an observed Linux-x86_64-vs-macOS-arm64
regeneration: max 17 ULP and ~4.4e-16 absolute near zero. MAX_ULP = 64
and ABS_TOL = 1e-12 give ~4x and ~2000x headroom while still sitting
many orders of magnitude below any algorithmically meaningful change
(the smallest real table value is ~5e-2).

Usage: check_data_reproducibility.py <committed_dir> <regenerated_dir>
Exit 0 when equivalent, 1 with a per-cell report otherwise.
"""
import csv
import math
import struct
import sys
from pathlib import Path

MAX_ULP = 64
ABS_TOL = 1e-12


def ordered_int(x: float) -> int:
    """Map a float64 to an integer that is monotonic in the float order,
    so ULP distance is a plain integer difference."""
    i = struct.unpack("<q", struct.pack("<d", x))[0]
    return i if i >= 0 else -(1 << 63) - i


def ulp_distance(a: float, b: float) -> int:
    return abs(ordered_int(a) - ordered_int(b))


def load(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    committed_dir, regenerated_dir = Path(sys.argv[1]), Path(sys.argv[2])

    committed_files = sorted(p.name for p in committed_dir.glob("*.csv"))
    regenerated_files = sorted(p.name for p in regenerated_dir.glob("*.csv"))
    if committed_files != regenerated_files:
        print(f"file sets differ:\n  committed:   {committed_files}\n"
              f"  regenerated: {regenerated_files}")
        return 1
    if not committed_files:
        print(f"no CSV files found in {committed_dir}")
        return 1

    failures = 0
    for name in committed_files:
        header_a, rows_a = load(committed_dir / name)
        header_b, rows_b = load(regenerated_dir / name)
        if header_a != header_b:
            print(f"{name}: header differs: {header_a} vs {header_b}")
            failures += 1
            continue
        if len(rows_a) != len(rows_b):
            print(f"{name}: row count differs: {len(rows_a)} vs {len(rows_b)}")
            failures += 1
            continue
        max_ulp_seen = 0
        atol_cells = 0
        file_failures_before = failures
        for r, (row_a, row_b) in enumerate(zip(rows_a, rows_b), start=2):
            for c, (cell_a, cell_b) in enumerate(zip(row_a, row_b)):
                if cell_a == cell_b:
                    continue
                # Integer wavelength grids must match exactly; only cells
                # that already differ textually reach the float path.
                try:
                    va, vb = float(cell_a), float(cell_b)
                except ValueError:
                    print(f"{name}:{r} col {c}: non-numeric mismatch "
                          f"{cell_a!r} vs {cell_b!r}")
                    failures += 1
                    continue
                if math.isnan(va) or math.isnan(vb) or math.isinf(va) \
                        or math.isinf(vb):
                    print(f"{name}:{r} col {c}: non-finite value "
                          f"{cell_a} vs {cell_b}")
                    failures += 1
                    continue
                if abs(va - vb) <= ABS_TOL:
                    atol_cells += 1
                    continue
                d = ulp_distance(va, vb)
                max_ulp_seen = max(max_ulp_seen, d)
                if d > MAX_ULP:
                    print(f"{name}:{r} col {c}: {cell_a} vs {cell_b} "
                          f"differs by {d} ULP (limit {MAX_ULP})")
                    failures += 1
        if failures == file_failures_before:
            print(f"[{name}] OK (max {max_ulp_seen} ULP, "
                  f"{atol_cells} cells within ABS_TOL)")
        else:
            print(f"[{name}] FAILED")
    if failures:
        print(f"\n{failures} discrepancies exceed the reproducibility "
              f"bounds (MAX_ULP={MAX_ULP}, ABS_TOL={ABS_TOL}).")
        return 1
    print(f"\nAll {len(committed_files)} tables numerically equivalent "
          f"(MAX_ULP={MAX_ULP}, ABS_TOL={ABS_TOL}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
