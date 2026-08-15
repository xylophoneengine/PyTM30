#!/usr/bin/env python3
"""Rudimentary benchmark for TM30Calc.eval() (Rf, Rg, CCT, Duv, ...).

Three sections: (1) test setup, (2) speed, (3) accuracy. Uses only the
real, bundled standard-illuminant SPDs in PyTM30/data/ -- no synthetic
spectra. colour-science is an optional dependency, used only for the
accuracy section as an independent cross-check.

Console output is also saved to benchmark_tm30_report.txt alongside this
script. Timing plots (histograms, not just mean/std) are saved next to it
too. Corpus list, environment block and plotting helpers are shared with
benchmark_spd_utils.py through _common.py.

Run from the PyTM30/ directory:
    python3 benchmarks/benchmark_tm30.py
"""

import platform
import sys
import textwrap
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

log_path = BENCH_DIR / "benchmark_tm30_report.txt"
log_file = com.tee_stdout(log_path)

# ===================================================================
# 0) ENVIRONMENT -- timings are meaningless without the machine and the
#    library versions they were taken on, and byte-level reproducibility
#    of results is only claimed for pinned versions.
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
print(f"\nCalculator: {calc!r}")

# ===================================================================
# 2) SPEED -- pytm30 (TM30Calc.eval), and colour-science side by side
#    when available, so it's unambiguous which numbers/plots belong to
#    which implementation.
# ===================================================================
print("\n" + "=" * 70)
print("2) SPEED")
print("=" * 70)

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
single_npts = []  # per-call grid length, to attribute timing modes to grids
for _ in range(N_SINGLE_REPS):
    for wl, spd in corpus.values():
        t0 = time.perf_counter()
        calc.eval(spd, wl)
        single_times.append(time.perf_counter() - t0)
        single_npts.append(len(wl))
single_ms = np.array(single_times) * 1000.0

print(f"pytm30 single eval, {N_SINGLE_REPS} repetitions x {len(corpus)} SPDs "
      f"= {len(single_ms)} timed calls:")
print(f"  {single_ms.mean():.4f} +/- {single_ms.std():.4f} ms/SPD "
      f"(median {np.median(single_ms):.4f}, max {single_ms.max():.4f})")

if HAVE_COLOUR:
    _ = cs_fidelity_index(wl0, spd0)  # warm-up
    cs_single_times = []
    cs_single_npts = []
    for _ in range(N_SINGLE_REPS):
        for wl, spd in corpus.values():
            t0 = time.perf_counter()
            cs_fidelity_index(wl, spd)
            cs_single_times.append(time.perf_counter() - t0)
            cs_single_npts.append(len(wl))
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
# matrix, not synthetic spectrum generation). It happens here, once,
# outside the timed region -- the batch timings below are for the eval
# call on an already-uniform matrix, and every published description of
# them has to say so.
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

# The single-eval histograms are bimodal because the corpus mixes two
# wavelength grids and per-eval cost scales with grid length -- split them
# by grid so the modes are attributable, and say so in a caption on the
# figure itself. The caption is built from the same predicate as the split,
# so it is None (and no caption is drawn) whenever the split falls back.
GRID_PANELS = [(single_npts, com.PYTM30_GRID_SHADES)]
if HAVE_COLOUR:
    GRID_PANELS.append((cs_single_npts, com.COLOUR_GRID_SHADES))
GRID_CAPTION = com.grid_caption(
    GRID_PANELS,
    "CES/CMF resampling and tristimulus integration run on the input grid",
    GRID_STEPS)

if HAVE_COLOUR:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    com.hist_panel(
        axes[0, 0], single_ms, com.PYTM30_COLOR, "Time per SPD (ms)",
        f"pytm30 single eval -- mean={single_ms.mean():.4f} ms  "
        f"std={single_ms.std():.4f} ms  n={len(single_ms)}",
        groups=com.grid_split(single_ms, single_npts, com.PYTM30_GRID_SHADES))
    com.hist_panel(
        axes[0, 1], cs_single_ms, com.COLOUR_COLOR, "Time per SPD (ms)",
        f"colour-science single eval -- mean={cs_single_ms.mean():.4f} ms  "
        f"std={cs_single_ms.std():.4f} ms  n={len(cs_single_ms)}",
        groups=com.grid_split(cs_single_ms, cs_single_npts, com.COLOUR_GRID_SHADES))
    com.hist_panel(
        axes[1, 0], batch_ms_total, com.PYTM30_COLOR, f"Total time for {len(corpus)} SPDs (ms)",
        f"pytm30 true batch ({N_REPS} reps) -- mean={batch_ms_total.mean():.4f} ms  "
        f"std={batch_ms_total.std():.4f} ms")
    com.hist_panel(
        axes[1, 1], cs_batch_ms_total, com.COLOUR_COLOR, f"Total time for {len(corpus)} SPDs (ms)",
        f"colour-science pseudo-batch ({N_REPS} reps) -- "
        f"mean={cs_batch_ms_total.mean():.4f} ms  std={cs_batch_ms_total.std():.4f} ms")
    fig.suptitle("TM30Calc.eval() vs colour-science -- Timing Distributions "
                 "(green=pytm30, blue=colour-science)", fontsize=12)
else:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    com.hist_panel(
        axes[0], single_ms, com.PYTM30_COLOR, "Time per SPD (ms)",
        f"pytm30 single eval -- mean={single_ms.mean():.4f} ms  "
        f"std={single_ms.std():.4f} ms  n={len(single_ms)}",
        groups=com.grid_split(single_ms, single_npts, com.PYTM30_GRID_SHADES))
    com.hist_panel(
        axes[1], batch_ms_total, com.PYTM30_COLOR, f"Total time for {len(corpus)} SPDs (ms)",
        f"pytm30 true batch ({N_REPS} reps) -- mean={batch_ms_total.mean():.4f} ms  "
        f"std={batch_ms_total.std():.4f} ms")
    fig.suptitle("TM30Calc.eval() Timing Distributions "
                 "(colour-science unavailable -- pytm30 only)", fontsize=12)
com.add_caption(fig, GRID_CAPTION)
timing_plot_path = BENCH_DIR / "benchmark_tm30_timing.png"
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

# ===================================================================
# 4) README SYNC -- rewrite the measured-claims block in README.md so
#    the published numbers can never go stale: rerunning this benchmark
#    IS the edit. Comparative claims, so only runs when colour-science
#    was available -- and never for a run taken in macOS low-power mode,
#    whose timings are not comparable to normal-mode ones and so must not
#    become the published headline numbers.
# ===================================================================
if HAVE_COLOUR and env.power_mode == com.LOW_POWER_MODE:
    print(f"\nREADME claims block NOT updated: this run was taken in macOS "
          f"low-power mode (pmset powermode {env.power_mode}). Those timings "
          f"are not comparable to normal-mode ones. Turn low-power mode off "
          f"and rerun to refresh the published numbers.")
elif HAVE_COLOUR:
    readme_path = BENCH_DIR.parent / "README.md"
    begin = "<!-- benchmark-results:begin -->"
    end = "<!-- benchmark-results:end -->"
    n = len(corpus)
    single_speedup = cs_single_ms.mean() / single_ms.mean()
    batch_speedup = cs_batch_ms_total.mean() / batch_ms_total.mean()
    rf_max = float(np.abs(np.asarray(deltas["Rf"])).max())
    rg_max = float(np.abs(np.asarray(deltas["Rg"])).max())
    cct_max = float(np.abs(np.asarray(deltas["CCT"])).max())
    # Same predicate as the figure's split and caption: no grid paragraph
    # when the panels were not split by grid.
    sizes = com.grid_sizes(single_npts, com.PYTM30_GRID_SHADES)
    if sizes is None:
        grid_note = ""
    else:
        listed = " and ".join(f"{s}-pt {GRID_STEPS[s]:g} nm" for s in sizes)
        grid_note = "\n" + textwrap.fill(
            f"The modes in the single-eval histograms are a corpus property, not timing "
            f"noise: the bundled illuminants mix wavelength grids ({listed}), and per-eval "
            f"cost scales with grid length because CES/CMF resampling and tristimulus "
            f"integration run on the input grid. The benchmark script resamples the batch "
            f"matrix onto one common 1 nm grid before the timed call -- that resampling is "
            f"not part of the batch timing -- hence the batch panels' single mode.",
            width=72) + "\n"
    block = f"""{begin}
<!-- Auto-written by benchmarks/benchmark_tm30.py; do not edit by hand.
     Rerun the benchmark to refresh numbers, plot, and environment. -->
The payoff, measured against colour-science on the bundled illuminant
corpus (`benchmarks/benchmark_tm30.py`):

| Path | colour-science | pytm30 | Speedup |
|---|---|---|---|
| Single eval | {cs_single_ms.mean():.3f} ms/SPD | {single_ms.mean():.3f} ms/SPD | **{single_speedup:.1f}x** |
| Batch ({n} SPDs per call) | {cs_batch_ms_total.mean() / n:.3f} ms/SPD | {batch_ms_total.mean() / n:.3f} ms/SPD | **{batch_speedup:.1f}x** |

Accuracy on the same corpus: Rf within {rf_max:.3f}, Rg within
{rg_max:.3f}, and CCT within {cct_max:.2f} K of colour-science's own
values.

Measured on: {env.cpu}, Python {platform.python_version()},
numpy {np.__version__}, colour-science {colour.__version__}\
{com.power_mode_note(env.power_mode)} -- full
environment and distributions in `benchmarks/benchmark_tm30_report.txt`.

![Timing distributions, pytm30 vs colour-science](benchmarks/benchmark_tm30_timing.png)
{grid_note}{end}"""
    text = readme_path.read_text()
    if begin in text and end in text:
        pre, rest = text.split(begin, 1)
        _, post = rest.split(end, 1)
        readme_path.write_text(pre + block + post)
        print(f"\nREADME measured-claims block refreshed: {readme_path}")
    else:
        print(f"\nREADME markers not found; claims block NOT updated. "
              f"Add '{begin}' ... '{end}' to README.md to enable the sync.")

print("\nDone.")
print(f"Report saved: {log_path}")

sys.stdout = sys.__stdout__
log_file.close()
