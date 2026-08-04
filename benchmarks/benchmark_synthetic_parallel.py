#!/usr/bin/env python3
"""Synthetic timing benchmark: 100 reps x batches of 1000 SPDs.

Four evaluation strategies on the same synthetic corpus (1000 SPDs as
random convex blends of the bundled real illuminants - two corpus SPDs
picked at random, S_new = spdA*x + spdB*(1-x), x ~ U[0,1] - resampled
to 380-780 nm @ 1 nm):

    a) Strict sequential  - every SPD evaluated by itself (single-SPD path)
    b) Batch, no workers  - TM30Calc(n_workers=1): sequential batch loop
    c) Batch, threads     - TM30Calc(n_workers=4): spawn-per-call threads
    d) Batch, persistent  - TM30Calc(n_workers=4, persistent_workers=True)

Per-repetition wall-clock times (time.perf_counter) are collected for
each case and reported as histograms + Gaussian KDEs, with summary
statistics. Raw times are saved as Apache Feather (benchmark best
practice in this project).

Outputs land in benchmarks/out_synthetic_parallel/:
    times.feather   - per-repetition timings (long format)
    histograms.png  - 2x2 per-case histogram + KDE overlay
    kde_overlay.png - all four KDEs on one axis
    report.txt      - summary statistics

Run from the repo root (bindings built in python/build):
    PYTHONPATH=python/build .venv/bin/python benchmarks/benchmark_synthetic_parallel.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tm30_calc import TM30Calc

OUT_DIR = Path(__file__).resolve().parent / "out_synthetic_parallel"
N_SPDS = 1000
N_REPS = 100
N_WORKERS = 4  # matches this machine's 4 hardware threads

# The bundled real illuminant corpus (same set benchmarks use).
CORPUS_NAMES = (
    ["d65_1nm"]
    + [f"fl{i}_1nm" for i in range(1, 13)]
    + [f"hp{i}_1nm" for i in range(1, 6)]
    + ["illuminant_a_1nm"]
)


def make_corpus(n_spds: int = N_SPDS, seed: int = 42):
    """n_spds synthetic SPDs as random convex blends of the bundled real
    illuminant corpus: pick two corpus SPDs at random and form
    S_new = spdA*x + spdB*(1-x) with x ~ U[0,1] (a convex combination,
    so every value stays non-negative and the 380-780 nm range holds).
    All corpus SPDs are resampled to the common 1 nm grid first.
    """
    wl = np.arange(380.0, 781.0, 1.0)
    data_dir = Path(__file__).resolve().parent.parent / "data"
    base = []
    for n in CORPUS_NAMES:
        arr = np.loadtxt(data_dir / f"{n}.csv", delimiter=",", skiprows=1)
        base.append(np.interp(wl, arr[:, 0], arr[:, 1]))
    base = np.ascontiguousarray(np.array(base))  # (19, 401)

    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(base), n_spds)
    idx_b = rng.integers(0, len(base), n_spds)
    x = rng.uniform(0.0, 1.0, n_spds)
    spds = base[idx_a] * x[:, None] + base[idx_b] * (1.0 - x[:, None])
    return wl, np.ascontiguousarray(spds)


def stats_ms(t_s: np.ndarray) -> str:
    t = np.asarray(t_s) * 1000.0
    return (
        f"mean {t.mean():7.2f} ± {t.std():6.2f} ms | "
        f"median {np.median(t):7.2f} | min {t.min():7.2f} | "
        f"max {t.max():7.2f} | ms/SPD {t.mean() / N_SPDS * 1000:.4f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_REPS)
    ap.add_argument("--n-spds", type=int, default=N_SPDS)
    ap.add_argument("--n-workers", type=int, default=N_WORKERS)
    args = ap.parse_args()

    reps, n_spds, nw = args.reps, args.n_spds, args.n_workers
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Synthetic corpus: {n_spds} Planckian SPDs, 380-780 nm @ 1 nm")
    print(f"Protocol: {reps} repetitions per case; n_workers={nw} (c, d)\n")

    wl, matrix = make_corpus(n_spds)
    calc_seq = TM30Calc()  # case a (single-SPD)
    calc_b = TM30Calc(n_workers=1)  # case b
    calc_c = TM30Calc(n_workers=nw)  # case c
    calc_d = TM30Calc(n_workers=nw, persistent_workers=True)  # case d

    # Correctness cross-check up front: all four cases must agree
    # bit-for-bit on a small batch.
    small = matrix[:8]
    ra = np.array([calc_seq.eval(s).rf for s in small])
    rb = np.array(calc_b.eval(small).rf)
    rc = np.array(calc_c.eval(small).rf)
    rd = np.array(calc_d.eval(small).rf)
    assert (
        np.array_equal(ra, rb) and np.array_equal(ra, rc) and np.array_equal(ra, rd)
    ), "cases disagree bit-for-bit - aborting benchmark"
    print("Bit-identity cross-check (a=b=c=d on 8 SPDs): OK\n")

    times = {}  # case -> np.array(reps) seconds

    # ── Case a: strict sequential, one SPD per call ──────────────────
    print(f"a) strict sequential ({n_spds} single evals/rep) ...", flush=True)
    calc_seq.eval(matrix[:50])  # warm tables
    ta = np.empty(reps)
    for r in range(reps):
        t0 = time.perf_counter()
        for s in matrix:
            calc_seq.eval(s)
        ta[r] = time.perf_counter() - t0
    times["a_sequential"] = ta
    print(f"   {stats_ms(ta)}\n")

    # ── Case b: batch, n_workers=1 ───────────────────────────────────
    print(f"b) batch n_workers=1 ({n_spds} SPDs/call) ...", flush=True)
    calc_b.eval(matrix[:50])
    tb = np.empty(reps)
    for r in range(reps):
        t0 = time.perf_counter()
        calc_b.eval(matrix)
        tb[r] = time.perf_counter() - t0
    times["b_batch_n1"] = tb
    print(f"   {stats_ms(tb)}\n")

    # ── Case c: batch, spawn-per-call threads ────────────────────────
    print(f"c) batch n_workers={nw} (spawn per call) ...", flush=True)
    calc_c.eval(matrix[:50])
    tc = np.empty(reps)
    for r in range(reps):
        t0 = time.perf_counter()
        calc_c.eval(matrix)
        tc[r] = time.perf_counter() - t0
    times["c_batch_threads"] = tc
    print(f"   {stats_ms(tc)}\n")

    # ── Case d: batch, persistent threads ────────────────────────────
    print(f"d) batch n_workers={nw} persistent ...", flush=True)
    calc_d.eval(matrix[:50])
    td = np.empty(reps)
    for r in range(reps):
        t0 = time.perf_counter()
        calc_d.eval(matrix)
        td[r] = time.perf_counter() - t0
    times["d_batch_persistent"] = td
    print(f"   {stats_ms(td)}\n")

    # ── Save raw times (Feather) ─────────────────────────────────────
    rows = []
    for case, t in times.items():
        for r, v in enumerate(t):
            rows.append({"case": case, "rep": r, "time_ms": v * 1000.0})
    df = pd.DataFrame(rows)
    feather_path = OUT_DIR / "times.feather"
    df.to_feather(feather_path)

    # ── Plots ────────────────────────────────────────────────────────
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    colors = {
        "a_sequential": "#dc2626",
        "b_batch_n1": "#2563eb",
        "c_batch_threads": "#059669",
        "d_batch_persistent": "#7c3aed",
    }
    labels = {
        "a_sequential": "a) sequential (1 SPD/call)",
        "b_batch_n1": "b) batch n_workers=1",
        "c_batch_threads": f"c) batch n_workers={nw}",
        "d_batch_persistent": f"d) batch n_workers={nw}, persistent",
    }

    # 2x2: per-case histogram + KDE overlay
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    all_t = {k: v * 1000.0 for k, v in times.items()}
    for ax, case in zip(axes.flat, all_t):
        t = all_t[case]
        ax.hist(
            t, bins=24, color=colors[case], alpha=0.45, edgecolor="white", density=True
        )
        xs = np.linspace(t.min() * 0.9, t.max() * 1.1, 300)
        kde = gaussian_kde(t)
        ax.plot(
            xs,
            kde(xs),
            color=colors[case],
            lw=2,
            label=f"KDE (med {np.median(t):.1f} ms)",
        )
        ax.set_title(f"{labels[case]}", fontsize=11)
        ax.set_xlabel("time per repetition (ms)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"1000 SPDs/repetition, {reps} reps, n_workers={nw} - "
        f"execution time histograms + KDEs",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    hist_path = OUT_DIR / "histograms.png"
    fig.savefig(hist_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Overlay of all four KDEs
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for case in all_t:
        t = all_t[case]
        xs = np.linspace(t.min() * 0.9, t.max() * 1.1, 400)
        kde = gaussian_kde(t)
        ax2.plot(
            xs,
            kde(xs),
            color=colors[case],
            lw=2.2,
            label=f"{labels[case]} - med {np.median(t):.1f} ms",
        )
    ax2.set_xlabel("time per repetition (ms)")
    ax2.set_ylabel("density")
    ax2.set_title(
        f"KDE overlay - {n_spds} SPDs/repetition, {reps} reps (n_workers={nw})"
    )
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    kde_path = OUT_DIR / "kde_overlay.png"
    fig2.savefig(kde_path, dpi=130, bbox_inches="tight")
    plt.close(fig2)

    # ── Report ───────────────────────────────────────────────────────
    med_b = np.median(times["b_batch_n1"])
    med_a = np.median(times["a_sequential"])
    med_c = np.median(times["c_batch_threads"])
    med_d = np.median(times["d_batch_persistent"])
    lines = [
        f"Synthetic timing benchmark - {n_spds} SPDs/rep, {reps} reps, n_workers={nw}",
        f"machine: {sys.platform}, corpus: convex blends of "
        f"{len(CORPUS_NAMES)} bundled illuminants, 380-780 nm @ 1 nm",
        "",
        "per-repetition wall time (ms):",
    ]
    for case in all_t:
        lines.append(f"  {labels[case]:<34s} {stats_ms(times[case])}")
    lines += [
        "",
        "median speedups:",
        f"  b vs a (batching vs single-SPD loop):   {med_a / med_b:5.2f}x",
        f"  c vs b (threads vs sequential batch):   {med_b / med_c:5.2f}x",
        f"  d vs c (persistent vs spawn):           {med_c / med_d:5.2f}x",
        f"  d vs a (persistent vs single-SPD loop): {med_a / med_d:5.2f}x",
        "",
        f"raw times: {feather_path.relative_to(OUT_DIR.parent)}",
        f"plots:     {hist_path.relative_to(OUT_DIR.parent)}, "
        f"{kde_path.relative_to(OUT_DIR.parent)}",
    ]
    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "report.txt").write_text(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
