#!/usr/bin/env python3
"""Parallel-evaluation benchmark harness for TM30Calc.

Measures three workloads:

A. Single-SPD eval throughput  -- the sequential per-SPD path.
    Protocol: 50 reps x 19 SPDs = 950 timed calls (per-SPD timing).
B. True batch eval             -- all 19 SPDs in one call.
    Protocol: 100 reps of the 19x401 batch matrix.
C. Repeated-call throughput    -- Phase 2's persistent_workers scenario:
    N repeated eval() calls of the same batch on one long-lived instance.

Each section reports mean +/- std, median, and max in ms (per-SPD for A,
per-call total for B/C), so a reported regression can be judged against
the spread rather than just the mean.

Usage:
    python3 benchmarks/benchmark_parallel.py                # n_workers=1
    python3 benchmarks/benchmark_parallel.py --n-workers 4
    python3 benchmarks/benchmark_parallel.py --n-workers 4 --persistent
    python3 benchmarks/benchmark_parallel.py --json out.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tm30_calc import TM30Calc

import _common as com

COMMON_WL = np.arange(380.0, 781.0, 1.0)

# {name: (wl, spd)} for the bundled real standard-illuminant SPDs. The name
# list lives in data/illuminant_corpus.txt, the one file every consumer of
# the corpus reads.
load_corpus = com.load_corpus


def stats_ms(times_s):
    t = np.asarray(times_s) * 1000.0
    return {
        "mean": float(t.mean()),
        "std": float(t.std()),
        "median": float(np.median(t)),
        "max": float(t.max()),
    }


def fmt(tag, s, unit):
    print(
        f"  {tag:>8s}: {s['mean']:.4f} +/- {s['std']:.4f} ms {unit} "
        f"(median {s['median']:.4f}, max {s['max']:.4f})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--persistent", action="store_true")
    ap.add_argument("--reps-single", type=int, default=50)
    ap.add_argument("--reps-batch", type=int, default=100)
    ap.add_argument("--reps-repeated", type=int, default=100)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    corpus = load_corpus()
    # Pre-change compat: n_workers/persistent_workers don't exist yet, so
    # the baseline run falls back to the plain constructor (sequential).
    try:
        calc = TM30Calc(n_workers=args.n_workers, persistent_workers=args.persistent)
    except TypeError:
        calc = TM30Calc()
    batch_matrix = np.array(
        [np.interp(COMMON_WL, wl, spd) for wl, spd in corpus.values()]
    )

    print(f"Machine: {sys.platform}, nproc=4, numpy {np.__version__}")
    print(f"Config: n_workers={args.n_workers} persistent_workers={args.persistent}")
    print(f"Corpus: {len(corpus)} SPDs, batch matrix {batch_matrix.shape}")

    # -- A. Single-SPD eval --------------------------------------------
    calc.eval(batch_matrix)  # warm tables
    single_s = []
    for _ in range(args.reps_single):
        for wl, spd in corpus.values():
            t0 = time.perf_counter()
            calc.eval(spd, wl)
            single_s.append(time.perf_counter() - t0)
    single = stats_ms(single_s)

    # -- B. True batch -------------------------------------------------
    calc.eval(batch_matrix, COMMON_WL)  # warm-up
    batch_s = []
    for _ in range(args.reps_batch):
        t0 = time.perf_counter()
        calc.eval(batch_matrix, COMMON_WL)
        batch_s.append(time.perf_counter() - t0)
    batch = stats_ms(batch_s)

    # -- C. Repeated-call throughput (Phase 2 scenario) ----------------
    calc.eval(batch_matrix, COMMON_WL)  # warm-up
    repeated_s = []
    for _ in range(args.reps_repeated):
        t0 = time.perf_counter()
        calc.eval(batch_matrix, COMMON_WL)
        repeated_s.append(time.perf_counter() - t0)
    repeated = stats_ms(repeated_s)

    print(
        f"\nA) single eval, {args.reps_single} reps x {len(corpus)} SPDs = {len(single_s)} calls:"
    )

    fmt("ms/SPD", single, "/SPD")
    print(f"\nB) true batch, {len(corpus)} SPDs/call, {args.reps_batch} reps:")
    fmt("ms/call", batch, "total")
    print(f"      per-SPD: {batch['mean'] / len(corpus):.4f} ms/SPD")
    print(f"\nC) repeated batch call, {args.reps_repeated} reps (same instance):")
    fmt("ms/call", repeated, "total")

    if args.json:
        out = {
            "n_workers": args.n_workers,
            "persistent": args.persistent,
            "single": single,
            "batch": batch,
            "repeated": repeated,
            "n_spds": len(corpus),
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()
