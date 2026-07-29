#!/usr/bin/env python3
"""Rudimentary benchmark for TM30Calc.eval() (Rf, Rg, CCT, Duv, ...).

Three sections: (1) test setup, (2) speed, (3) accuracy. Uses only the
real, bundled standard-illuminant SPDs in PyTM30/data/ -- no synthetic
spectra. colour-science is an optional dependency, used only for the
accuracy section as an independent cross-check.

Console output is also saved to benchmark_tm30_report.txt alongside this
script. Timing plots (histograms, not just mean/std) are saved next to it
too.

Run from the PyTM30/ directory:
    python3 benchmarks/benchmark_tm30.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from tm30_calc import TM30Calc

try:
    import colour
    HAVE_COLOUR = True
except ImportError:
    HAVE_COLOUR = False

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BENCH_DIR = Path(__file__).resolve().parent


class _Tee:
    """Writes to both the real console and a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


log_path = BENCH_DIR / "benchmark_tm30_report.txt"
log_file = open(log_path, "w")
sys.stdout = _Tee(sys.__stdout__, log_file)

# The ~19 real standard-illuminant SPDs bundled with the package.
SPD_NAMES = (
    ["d65_1nm"]
    + [f"fl{i}_1nm" for i in range(1, 13)]
    + [f"hp{i}_1nm" for i in range(1, 6)]
    + ["illuminant_a_1nm"]
)


def load_spd(name):
    """Load a bundled 2-column (wavelength, value) CSV from data/."""
    arr = np.loadtxt(DATA_DIR / f"{name}.csv", delimiter=",", skiprows=1)
    return arr[:, 0], arr[:, 1]


def hist_panel(ax, data, color, xlabel, title):
    """Plain histogram (no scipy KDE dependency) showing the actual
    distribution shape -- a mean+/-std number alone hides skew, outliers,
    and multi-modality."""
    ax.hist(data, bins=min(30, max(10, len(data) // 10)), color=color,
             edgecolor="white", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=9)


# ═══════════════════════════════════════════════════════════════════
# 1) TEST SETUP
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("1) TEST SETUP")
print("=" * 70)

corpus = {name: load_spd(name) for name in SPD_NAMES}

print(f"SPD corpus: {len(corpus)} real standard-illuminant spectra from data/*.csv")
for name, (wl, _) in corpus.items():
    step = wl[1] - wl[0]
    print(f"  {name:<20s} {len(wl):4d} pts   {wl[0]:.0f}-{wl[-1]:.0f} nm   step={step:g} nm")

calc = TM30Calc()
print(f"\nCalculator: {calc!r}")

# ═══════════════════════════════════════════════════════════════════
# 2) SPEED -- pytm30 (TM30Calc.eval), and colour-science side by side
#    when available, so it's unambiguous which numbers/plots belong to
#    which implementation.
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2) SPEED")
print("=" * 70)

PYTM30_COLOR = "#10b981"   # green -- pytm30, used consistently below
COLOUR_COLOR = "#2563eb"   # blue  -- colour-science, used consistently below

N_SINGLE_REPS = 50
N_REPS = 100
wl0, spd0 = next(iter(corpus.values()))


def cs_fidelity_index(wl, spd):
    """colour-science's equivalent single-SPD call -- SD construction is
    included in the timed region, same as pytm30's call, for a fair
    'given raw wavelength+value arrays' comparison."""
    sd = colour.SpectralDistribution(dict(zip(wl, spd)))
    return colour.colour_fidelity_index(sd, additional_data=True, method="ANSI/IES TM-30-18")


# -- Single eval: repeated over the corpus to build a real distribution,
#    not a single one-off sample per SPD --
_ = calc.eval(spd0, wl0)  # warm-up (pays any one-time setup cost)

single_times = []
for _ in range(N_SINGLE_REPS):
    for wl, spd in corpus.values():
        t0 = time.perf_counter()
        calc.eval(spd, wl)
        single_times.append(time.perf_counter() - t0)
single_ms = np.array(single_times) * 1000.0

print(f"pytm30 single eval, {N_SINGLE_REPS} repetitions x {len(corpus)} SPDs "
      f"= {len(single_ms)} timed calls:")
print(f"  {single_ms.mean():.4f} +/- {single_ms.std():.4f} ms/SPD "
      f"(median {np.median(single_ms):.4f}, max {single_ms.max():.4f})")

if HAVE_COLOUR:
    _ = cs_fidelity_index(wl0, spd0)  # warm-up
    cs_single_times = []
    for _ in range(N_SINGLE_REPS):
        for wl, spd in corpus.values():
            t0 = time.perf_counter()
            cs_fidelity_index(wl, spd)
            cs_single_times.append(time.perf_counter() - t0)
    cs_single_ms = np.array(cs_single_times) * 1000.0
    print(f"\ncolour-science single eval, {N_SINGLE_REPS} repetitions x {len(corpus)} SPDs "
          f"= {len(cs_single_ms)} timed calls:")
    print(f"  {cs_single_ms.mean():.4f} +/- {cs_single_ms.std():.4f} ms/SPD "
          f"(median {np.median(cs_single_ms):.4f}, max {cs_single_ms.max():.4f})")
    print(f"\npytm30 is {cs_single_ms.mean() / single_ms.mean():.1f}x faster, single eval")

# -- True batch: whole corpus stacked into one 2-D array --
# HP1-5 are natively tabulated at 5 nm; linearly resample them onto the
# common 1 nm grid so every row of the batch matrix has the same width
# (this is just a resampling of real data for the batch API's fixed-width
# matrix, not synthetic spectrum generation).
common_wl = np.arange(380.0, 781.0, 1.0)
batch_matrix = np.array([np.interp(common_wl, wl, spd) for wl, spd in corpus.values()])

_ = calc.eval(batch_matrix, common_wl)  # warm-up

batch_s = np.empty(N_REPS)
for i in range(N_REPS):
    t0 = time.perf_counter()
    calc.eval(batch_matrix, common_wl)
    batch_s[i] = time.perf_counter() - t0
batch_ms_total = batch_s * 1000.0

print(f"\npytm30 true batch, all {len(corpus)} SPDs in one call, {N_REPS} repetitions:")
print(f"  {batch_ms_total.mean():.4f} +/- {batch_ms_total.std():.4f} ms total "
      f"({batch_ms_total.mean() / len(corpus):.4f} ms/SPD)")

if HAVE_COLOUR:
    # colour-science has no native batch API -- its "batch" is a full loop
    # over the corpus, repeated N_REPS times (a real distribution of totals,
    # not a single one-off loop timing).
    for wl, spd in corpus.values():
        cs_fidelity_index(wl, spd)  # warm-up
    cs_batch_s = np.empty(N_REPS)
    for i in range(N_REPS):
        t0 = time.perf_counter()
        for wl, spd in corpus.values():
            cs_fidelity_index(wl, spd)
        cs_batch_s[i] = time.perf_counter() - t0
    cs_batch_ms_total = cs_batch_s * 1000.0
    print(f"\ncolour-science pseudo-batch (loop over {len(corpus)} SPDs, "
          f"no native batch API), {N_REPS} repetitions:")
    print(f"  {cs_batch_ms_total.mean():.4f} +/- {cs_batch_ms_total.std():.4f} ms total "
          f"({cs_batch_ms_total.mean() / len(corpus):.4f} ms/SPD)")
    print(f"\npytm30 is {cs_batch_ms_total.mean() / batch_ms_total.mean():.1f}x faster, batch")

# Timing distribution plots -- a mean+/-std pair alone hides the shape.
# Every panel is titled with which implementation it is.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if HAVE_COLOUR:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    hist_panel(
        axes[0, 0], single_ms, PYTM30_COLOR, "Time per SPD (ms)",
        f"pytm30 single eval -- mean={single_ms.mean():.4f} ms  "
        f"std={single_ms.std():.4f} ms  n={len(single_ms)}")
    hist_panel(
        axes[0, 1], cs_single_ms, COLOUR_COLOR, "Time per SPD (ms)",
        f"colour-science single eval -- mean={cs_single_ms.mean():.4f} ms  "
        f"std={cs_single_ms.std():.4f} ms  n={len(cs_single_ms)}")
    hist_panel(
        axes[1, 0], batch_ms_total, PYTM30_COLOR, f"Total time for {len(corpus)} SPDs (ms)",
        f"pytm30 true batch ({N_REPS} reps) -- mean={batch_ms_total.mean():.4f} ms  "
        f"std={batch_ms_total.std():.4f} ms")
    hist_panel(
        axes[1, 1], cs_batch_ms_total, COLOUR_COLOR, f"Total time for {len(corpus)} SPDs (ms)",
        f"colour-science pseudo-batch ({N_REPS} reps) -- "
        f"mean={cs_batch_ms_total.mean():.4f} ms  std={cs_batch_ms_total.std():.4f} ms")
    fig.suptitle("TM30Calc.eval() vs colour-science -- Timing Distributions "
                 "(green=pytm30, blue=colour-science)", fontsize=12)
else:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    hist_panel(
        axes[0], single_ms, PYTM30_COLOR, "Time per SPD (ms)",
        f"pytm30 single eval -- mean={single_ms.mean():.4f} ms  "
        f"std={single_ms.std():.4f} ms  n={len(single_ms)}")
    hist_panel(
        axes[1], batch_ms_total, PYTM30_COLOR, f"Total time for {len(corpus)} SPDs (ms)",
        f"pytm30 true batch ({N_REPS} reps) -- mean={batch_ms_total.mean():.4f} ms  "
        f"std={batch_ms_total.std():.4f} ms")
    fig.suptitle("TM30Calc.eval() Timing Distributions "
                 "(colour-science unavailable -- pytm30 only)", fontsize=12)
fig.tight_layout()
timing_plot_path = BENCH_DIR / "benchmark_tm30_timing.png"
fig.savefig(timing_plot_path, dpi=150)
plt.close(fig)
print(f"\nTiming plot saved: {timing_plot_path}")

# ═══════════════════════════════════════════════════════════════════
# 3) ACCURACY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3) ACCURACY")
print("=" * 70)

if not HAVE_COLOUR:
    print("pip install colour-science to run the accuracy comparison")
else:
    # Independent reference: colour-science's own ANSI/IES TM-30-18
    # implementation, fed the exact same SPD arrays as TM30Calc (so any
    # residual reflects differing algorithms, not differing input data).
    deltas = {"Rf": [], "Rg": [], "CCT": [], "Duv": []}
    for name, (wl, spd) in corpus.items():
        r = calc.eval(spd, wl)
        sd = colour.SpectralDistribution(dict(zip(wl, spd)))
        spec = colour.colour_fidelity_index(
            sd, additional_data=True, method="ANSI/IES TM-30-18")
        deltas["Rf"].append(r.rf - spec.R_f)
        deltas["Rg"].append(r.rg - spec.R_g)
        deltas["CCT"].append(r.cct - spec.CCT)
        deltas["Duv"].append(r.duv - spec.D_uv)

    print(f"pytm30 vs colour-science {colour.__version__} "
          f"('ANSI/IES TM-30-18'), {len(corpus)} SPDs:\n")
    print(f"  {'Metric':<8s} {'Mean':>12s} {'Std':>12s} {'Max|Delta|':>12s}")
    for metric, fmt in [("Rf", "{:.4f}"), ("Rg", "{:.4f}"),
                         ("CCT", "{:.3f}"), ("Duv", "{:.2e}")]:
        v = np.asarray(deltas[metric])
        print(f"  {metric:<8s} {fmt.format(v.mean()):>12s} "
              f"{fmt.format(v.std()):>12s} {fmt.format(np.abs(v).max()):>12s}")

    # NOTE: FL1-FL12 in data/ were already resampled from colour-science's
    # native 5 nm tables to 1 nm (see tools/generate_data_colour_science.py);
    # since both implementations here are fed that same bundled 1 nm data,
    # the fluorescent-lamp-specific ~5 K CCT/data-resolution difference
    # documented elsewhere in this project does not apply to this
    # same-input comparison -- deviations above are algorithmic noise only.

    # Simple plot: per-SPD Rf deviation, to see at a glance that agreement
    # is uniformly tight across every real illuminant in the corpus.
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(range(len(corpus)), deltas["Rf"], color="#2563eb")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(corpus)))
    ax.set_xticklabels(list(corpus.keys()), rotation=90, fontsize=7)
    ax.set_ylabel("Rf (pytm30 - colour-science)")
    ax.set_title("TM30Calc.eval() Rf accuracy vs colour-science, per SPD")
    fig.tight_layout()
    out_path = BENCH_DIR / "benchmark_tm30_accuracy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

print("\nDone.")
print(f"Report saved: {log_path}")

sys.stdout = sys.__stdout__
log_file.close()
