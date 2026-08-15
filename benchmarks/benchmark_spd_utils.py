#!/usr/bin/env python3
"""Rudimentary benchmark for TM30Calc.spd_to_xyz() / .spd_to_Yuv().

Four sections: (0) environment, (1) test setup, (2) speed, (3) accuracy.
Uses only the real, bundled standard-illuminant SPDs in PyTM30/data/ -- no
synthetic spectra. colour-science is an optional dependency, used only for
the accuracy section as an independent cross-check.

Console output is also saved to benchmark_spd_utils_report.txt alongside
this script. Timing plots (histograms, not just mean/std) are saved next
to it too. Corpus list, environment block and plotting helpers are shared
with benchmark_tm30.py through _common.py.

Run from the PyTM30/ directory:
    python3 benchmarks/benchmark_spd_utils.py
"""

import sys
import time

import numpy as np
from tm30_calc import TM30Calc

import _common as com

try:
    import colour
    HAVE_COLOUR = True
except ImportError:
    HAVE_COLOUR = False

BENCH_DIR = com.BENCH_DIR

log_path = BENCH_DIR / "benchmark_spd_utils_report.txt"
log_file = com.tee_stdout(log_path)

# ===================================================================
# 0) ENVIRONMENT -- same block benchmark_tm30.py prints, for the same
#    reason: without it, a shift in this report's numbers between two
#    commits is indistinguishable from a performance change.
# ===================================================================
env = com.print_environment(colour if HAVE_COLOUR else None)

# ===================================================================
# 1) TEST SETUP
# ===================================================================
print("\n" + "=" * 70)
print("1) TEST SETUP")
print("=" * 70)

corpus = com.load_corpus()
GRID_STEPS = com.grid_steps(corpus)

print(f"SPD corpus: {len(corpus)} real standard-illuminant spectra from data/*.csv")
for name, (wl, _) in corpus.items():
    step = wl[1] - wl[0]
    print(f"  {name:<20s} {len(wl):4d} pts   {wl[0]:.0f}-{wl[-1]:.0f} nm   step={step:g} nm")

calc = TM30Calc()
print(f"\nCalculator: {calc!r}  (spd_to_xyz/spd_to_Yuv use the CIE 1964 10-degree observer)")

# ===================================================================
# 2) SPEED -- pytm30 (spd_to_xyz/spd_to_Yuv), and colour-science side by
#    side when available, so it's unambiguous which numbers/plots belong
#    to which implementation.
# ===================================================================
print("\n" + "=" * 70)
print("2) SPEED")
print("=" * 70)

N_SINGLE_REPS = 50
N_REPS = 100
wl0, spd0 = next(iter(corpus.values()))

if HAVE_COLOUR:
    _cmfs_10deg = colour.MSDS_CMFS["CIE 1964 10 Degree Standard Observer"].copy()


def cs_spd_to_xyz(wl, spd):
    """colour-science's equivalent of spd_to_xyz -- SD/CMF construction is
    included in the timed region, same as pytm30's call, for a fair
    'given raw wavelength+value arrays' comparison."""
    shape = colour.SpectralShape(wl[0], wl[-1], wl[1] - wl[0])
    cmfs_aligned = _cmfs_10deg.copy().align(shape)
    xyz_raw = colour.sd_to_XYZ(spd, cmfs=cmfs_aligned, shape=shape, method="Integration")
    return xyz_raw * (100.0 / xyz_raw[1])  # match pytm30's Y=100 convention


def cs_spd_to_Yuv(wl, spd):
    """colour-science's equivalent of spd_to_Yuv (chains through XYZ)."""
    xyz = cs_spd_to_xyz(wl, spd)
    u, v = colour.UCS_to_uv(colour.XYZ_to_UCS(xyz))
    return np.array([xyz[1], u, 1.5 * v])


PT_FNS = {"spd_to_xyz": calc.spd_to_xyz, "spd_to_Yuv": calc.spd_to_Yuv}
CS_FNS = {"spd_to_xyz": cs_spd_to_xyz, "spd_to_Yuv": cs_spd_to_Yuv} if HAVE_COLOUR else {}

_ = calc.spd_to_xyz(spd0, wl0)  # warm-up
_ = calc.spd_to_Yuv(spd0, wl0)

# Per-call grid lengths are appended inside each timing loop, next to the
# timing they belong to, rather than precomputed once from the loop bounds:
# a precomputed array stays the same length when a loop is reordered or an
# SPD is skipped, so a desync would silently relabel the timing modes.
single_ms = {}
single_npts = {}
for fn_name, fn in PT_FNS.items():
    times = []
    npts = []
    for _ in range(N_SINGLE_REPS):
        for wl, spd in corpus.values():
            t0 = time.perf_counter()
            fn(spd, wl)
            times.append(time.perf_counter() - t0)
            npts.append(len(wl))
    single_ms[fn_name] = np.array(times) * 1000.0
    single_npts[fn_name] = np.array(npts)
    print(f"pytm30 {fn_name} single eval, {N_SINGLE_REPS} repetitions x {len(corpus)} SPDs "
          f"= {len(times)} timed calls:")
    print(f"  {single_ms[fn_name].mean():.4f} +/- {single_ms[fn_name].std():.4f} ms/SPD "
          f"(median {np.median(single_ms[fn_name]):.4f}, max {single_ms[fn_name].max():.4f})")

cs_single_ms = {}
cs_single_npts = {}
if HAVE_COLOUR:
    _ = cs_spd_to_xyz(wl0, spd0)  # warm-up
    _ = cs_spd_to_Yuv(wl0, spd0)
    print()
    for fn_name, fn in CS_FNS.items():
        times = []
        npts = []
        for _ in range(N_SINGLE_REPS):
            for wl, spd in corpus.values():
                t0 = time.perf_counter()
                fn(wl, spd)
                times.append(time.perf_counter() - t0)
                npts.append(len(wl))
        cs_single_ms[fn_name] = np.array(times) * 1000.0
        cs_single_npts[fn_name] = np.array(npts)
        print(f"colour-science {fn_name}-equivalent single eval, {N_SINGLE_REPS} reps x "
              f"{len(corpus)} SPDs = {len(times)} timed calls:")
        print(f"  {cs_single_ms[fn_name].mean():.4f} +/- {cs_single_ms[fn_name].std():.4f} ms/SPD "
              f"(median {np.median(cs_single_ms[fn_name]):.4f}, max {cs_single_ms[fn_name].max():.4f})")
        print(f"  pytm30 is {cs_single_ms[fn_name].mean() / single_ms[fn_name].mean():.1f}x faster")

# True batch: whole corpus stacked into one 2-D array. HP1-5 are natively
# tabulated at 5 nm; linearly resample them onto the common 1 nm grid so
# every row of the batch matrix has the same width (a resampling of real
# data for the batch API's fixed-width matrix, not synthetic generation).
# It happens here, once, outside the timed region -- the batch timings
# below cover the call on an already-uniform matrix, and every published
# description of them has to say so.
common_wl = np.arange(380.0, 781.0, 1.0)
batch_matrix = np.array([np.interp(common_wl, wl, spd) for wl, spd in corpus.values()])

batch_ms_total = {}
print()
for fn_name, fn in PT_FNS.items():
    _ = fn(batch_matrix, common_wl)  # warm-up
    batch_s = np.empty(N_REPS)
    for i in range(N_REPS):
        t0 = time.perf_counter()
        fn(batch_matrix, common_wl)
        batch_s[i] = time.perf_counter() - t0
    batch_ms_total[fn_name] = batch_s * 1000.0
    print(f"pytm30 {fn_name} true batch, all {len(corpus)} SPDs in one call, {N_REPS} repetitions:")
    print(f"  {batch_ms_total[fn_name].mean():.4f} +/- {batch_ms_total[fn_name].std():.4f} ms total "
          f"({batch_ms_total[fn_name].mean() / len(corpus):.4f} ms/SPD)")

cs_batch_ms_total = {}
if HAVE_COLOUR:
    # colour-science has no native batch API -- its "batch" is a full loop
    # over the corpus, repeated N_REPS times (a real distribution of
    # totals, not a single one-off loop timing).
    print()
    for fn_name, fn in CS_FNS.items():
        for wl, spd in corpus.values():
            fn(wl, spd)  # warm-up
        batch_s = np.empty(N_REPS)
        for i in range(N_REPS):
            t0 = time.perf_counter()
            for wl, spd in corpus.values():
                fn(wl, spd)
            batch_s[i] = time.perf_counter() - t0
        cs_batch_ms_total[fn_name] = batch_s * 1000.0
        print(f"colour-science {fn_name}-equivalent pseudo-batch (loop over {len(corpus)} "
              f"SPDs, no native batch API), {N_REPS} repetitions:")
        print(f"  {cs_batch_ms_total[fn_name].mean():.4f} +/- {cs_batch_ms_total[fn_name].std():.4f} "
              f"ms total ({cs_batch_ms_total[fn_name].mean() / len(corpus):.4f} ms/SPD)")
        print(f"  pytm30 is {cs_batch_ms_total[fn_name].mean() / batch_ms_total[fn_name].mean():.1f}x faster")

# Timing distribution plots -- a mean+/-std pair alone hides the shape.
# Every panel is titled with which implementation it is; green=pytm30,
# blue=colour-science, consistently.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The single-eval histograms are bimodal because the corpus mixes two
# wavelength grids and per-call cost scales with grid length -- split them
# by grid so the modes are attributable, and say so in a caption on the
# figure itself. The caption is built from the same predicate as the split,
# so it is None (and no caption is drawn) whenever the split falls back.
GRID_PANELS = [(single_npts[fn_name], com.PYTM30_GRID_SHADES) for fn_name in PT_FNS]
GRID_PANELS += [(cs_single_npts[fn_name], com.COLOUR_GRID_SHADES) for fn_name in CS_FNS]
GRID_CAPTION = com.grid_caption(
    GRID_PANELS,
    "CMF resampling and the XYZ integration run on the input grid",
    GRID_STEPS)

if HAVE_COLOUR:
    fig, axes = plt.subplots(2, 4, figsize=(19, 7.8))
    for col, fn_name in enumerate(["spd_to_xyz", "spd_to_Yuv"]):
        com.hist_panel(
            axes[0, col * 2], single_ms[fn_name], com.PYTM30_COLOR, "Time per SPD (ms)",
            f"pytm30 {fn_name} single eval -- mean={single_ms[fn_name].mean():.4f} ms  "
            f"n={len(single_ms[fn_name])}",
            groups=com.grid_split(single_ms[fn_name], single_npts[fn_name],
                                  com.PYTM30_GRID_SHADES))
        com.hist_panel(
            axes[0, col * 2 + 1], cs_single_ms[fn_name], com.COLOUR_COLOR, "Time per SPD (ms)",
            f"colour-science {fn_name}-eq single eval -- "
            f"mean={cs_single_ms[fn_name].mean():.4f} ms  n={len(cs_single_ms[fn_name])}",
            groups=com.grid_split(cs_single_ms[fn_name], cs_single_npts[fn_name],
                                  com.COLOUR_GRID_SHADES))
        com.hist_panel(
            axes[1, col * 2], batch_ms_total[fn_name], com.PYTM30_COLOR,
            f"Total time for {len(corpus)} SPDs (ms)",
            f"pytm30 {fn_name} true batch ({N_REPS} reps) -- "
            f"mean={batch_ms_total[fn_name].mean():.4f} ms")
        com.hist_panel(
            axes[1, col * 2 + 1], cs_batch_ms_total[fn_name], com.COLOUR_COLOR,
            f"Total time for {len(corpus)} SPDs (ms)",
            f"colour-science {fn_name}-eq pseudo-batch ({N_REPS} reps) -- "
            f"mean={cs_batch_ms_total[fn_name].mean():.4f} ms")
    fig.suptitle("spd_to_xyz / spd_to_Yuv vs colour-science -- Timing Distributions "
                 "(green=pytm30, blue=colour-science)", fontsize=12)
else:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    for col, fn_name in enumerate(["spd_to_xyz", "spd_to_Yuv"]):
        com.hist_panel(
            axes[0, col], single_ms[fn_name], com.PYTM30_COLOR, "Time per SPD (ms)",
            f"pytm30 {fn_name} single eval -- mean={single_ms[fn_name].mean():.4f} ms  "
            f"n={len(single_ms[fn_name])}",
            groups=com.grid_split(single_ms[fn_name], single_npts[fn_name],
                                  com.PYTM30_GRID_SHADES))
        com.hist_panel(
            axes[1, col], batch_ms_total[fn_name], com.PYTM30_COLOR,
            f"Total time for {len(corpus)} SPDs (ms)",
            f"pytm30 {fn_name} true batch ({N_REPS} reps) -- "
            f"mean={batch_ms_total[fn_name].mean():.4f} ms")
    fig.suptitle("spd_to_xyz / spd_to_Yuv Timing Distributions "
                 "(colour-science unavailable -- pytm30 only)", fontsize=12)
com.add_caption(fig, GRID_CAPTION)
timing_plot_path = BENCH_DIR / "benchmark_spd_utils_timing.png"
fig.savefig(timing_plot_path, dpi=150)
plt.close(fig)
print(f"\nTiming plot saved: {timing_plot_path}")

# ===================================================================
# 3) ACCURACY
# ===================================================================
print("\n" + "=" * 70)
print("3) ACCURACY")
print("=" * 70)

if not HAVE_COLOUR:
    print("pip install colour-science to run the accuracy comparison")
else:
    # Independent reference: colour-science's sd_to_XYZ with the CIE 1964
    # 10-degree observer explicitly (NOT colour-science's 2-degree default --
    # pytm30's spd_to_xyz/spd_to_Yuv use the 10-degree observer, so comparing
    # against the 2-degree default would produce a large, meaningless
    # discrepancy driven by the wrong CMF, not a real accuracy issue). Reuses
    # the exact same cs_spd_to_xyz/cs_spd_to_Yuv helpers timed in section 2,
    # so speed and accuracy sections stay consistent with each other.
    xyz_dev, yuv_dev = [], []
    for name, (wl, spd) in corpus.items():
        xyz_pt = calc.spd_to_xyz(spd, wl)          # K=None -> Y normalised to 100
        yuv_pt = calc.spd_to_Yuv(spd, wl)

        xyz_cs = cs_spd_to_xyz(wl, spd)
        yuv_cs = cs_spd_to_Yuv(wl, spd)

        xyz_dev.append(xyz_pt - xyz_cs)
        yuv_dev.append(yuv_pt - yuv_cs)

    xyz_dev = np.abs(np.array(xyz_dev))  # (N, 3): X, Y, Z
    yuv_dev = np.abs(np.array(yuv_dev))  # (N, 3): Y, u', v'

    print(f"pytm30 vs colour-science {colour.__version__} "
          f"(CIE 1964 10-degree observer, Integration, Y=100), {len(corpus)} SPDs:\n")
    print(f"  {'Field':<10s} {'Mean|Delta|':>12s} {'Std|Delta|':>12s} {'Max|Delta|':>12s}")
    for i, label in enumerate(["X", "Y", "Z"]):
        v = xyz_dev[:, i]
        print(f"  {label:<10s} {v.mean():>12.2e} {v.std():>12.2e} {v.max():>12.2e}")
    for i, label in enumerate(["Y (Yuv)", "u'", "v'"]):
        v = yuv_dev[:, i]
        print(f"  {label:<10s} {v.mean():>12.2e} {v.std():>12.2e} {v.max():>12.2e}")

    # Simple plot: per-SPD max|XYZ deviation|, to see at a glance that
    # agreement is at the numerical-noise level across every real illuminant.
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(range(len(corpus)), xyz_dev.max(axis=1), color="#059669")
    ax.set_xticks(range(len(corpus)))
    ax.set_xticklabels(list(corpus.keys()), rotation=90, fontsize=7)
    ax.set_ylabel("max|XYZ delta| (pytm30 vs colour-science)")
    ax.set_title("spd_to_xyz() accuracy vs colour-science, per SPD")
    fig.tight_layout()
    out_path = BENCH_DIR / "benchmark_spd_utils_accuracy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

print("\nDone.")
print(f"Report saved: {log_path}")

sys.stdout = sys.__stdout__
log_file.close()
