"""Shared pieces of the benchmark scripts in this directory.

benchmark_tm30.py and benchmark_spd_utils.py publish figures, reports and
(for the former) README claims that have to agree with each other. Keeping
the corpus list, the environment block, the timing-plot helpers and the
grid-split caption here is what makes that agreement structural instead of
a matter of keeping near-identical copies in sync by hand.

Imported by the scripts as `import _common as com` -- they run as
`python3 benchmarks/<script>.py`, so this directory is already sys.path[0].
"""

import platform
import subprocess
import sys
import textwrap
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR.parent / "data"
CORPUS_MANIFEST = DATA_DIR / "illuminant_corpus.txt"


def corpus_names():
    """Basenames of the bundled illuminant corpus, read from the manifest in
    data/ -- the same file the C++ timing test and tools/bench_cpp_baseline
    read, so renaming an SPD is one edit rather than seven."""
    names = []
    for line in CORPUS_MANIFEST.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


SPD_NAMES = corpus_names()


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


def tee_stdout(log_path):
    """Send stdout to both the console and log_path. Returns the log file;
    the caller restores sys.stdout and closes it at the end of the run."""
    log_file = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    return log_file


def load_spd(name):
    """Load a bundled 2-column (wavelength, value) CSV from data/."""
    arr = np.loadtxt(DATA_DIR / f"{name}.csv", delimiter=",", skiprows=1)
    return arr[:, 0], arr[:, 1]


def load_corpus():
    """{name: (wavelengths, values)} for the whole bundled corpus."""
    return {name: load_spd(name) for name in SPD_NAMES}


def grid_steps(corpus):
    """{grid length: wavelength step} over a corpus, for labelling the grid
    sizes a timing histogram was split by."""
    return {len(wl): wl[1] - wl[0] for wl, _ in corpus.values()}


# ===================================================================
# Environment
# ===================================================================

Environment = namedtuple("Environment", "cpu power_mode")


def print_environment(colour_module=None):
    """Print the ENVIRONMENT section: timings are meaningless without the
    machine and the library versions they were taken on, and byte-level
    reproducibility of results is only claimed for pinned versions. Both
    benchmark scripts print it, so a shift in either report's numbers can
    always be attributed to the run it came from.

    Returns Environment(cpu, power_mode); power_mode is the raw macOS pmset
    value as a string, or None off macOS / when pmset is unavailable."""
    print("=" * 70)
    print("0) ENVIRONMENT")
    print("=" * 70)

    cpu_label = platform.machine()
    power_mode = None
    print(f"Date (UTC):     {time.strftime('%Y-%m-%d %H:%M', time.gmtime())}")
    print(f"Python:         {platform.python_version()} ({platform.python_implementation()})")
    print(f"OS / machine:   {platform.platform()} ({platform.machine()})")
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if cpu:
                cpu_label = cpu
                print(f"CPU:            {cpu}")
            # Timings taken in low-power mode are not comparable to normal
            # ones; record the mode so a skewed run is recognisable later,
            # and so the README sync can refuse to publish one.
            pm = subprocess.run(["pmset", "-g"], capture_output=True, text=True,
                                timeout=5).stdout
            for line in pm.splitlines():
                if "powermode" in line:
                    power_mode = line.split()[-1]
                    print(f"Power mode:     {power_mode} "
                          "(macOS pmset powermode; 1 = low power)")
        except Exception:
            pass  # environment report is best-effort, never fails the run
    print(f"numpy:          {np.__version__}")
    if colour_module is not None:
        print(f"colour-science: {colour_module.__version__}")
    try:
        from importlib.metadata import version as _pkg_version
        print(f"pytm30:         {_pkg_version('pytm30')}")
    except Exception:
        print("pytm30:         (not installed as a package -- run from source tree)")
    return Environment(cpu_label, power_mode)


LOW_POWER_MODE = "1"
# macOS `pmset -g powermode`: 0 automatic, 1 low power, 2 high power. Only
# mode 1 throttles, but all three belong in a published claims block -- high
# power is no more "the machine as normally configured" than low power is.
POWER_MODE_LABELS = {"0": "automatic", LOW_POWER_MODE: "low power", "2": "high power"}


def power_mode_note(power_mode):
    """Power-mode clause for a published claims block, or "" when the mode is
    unknown -- published timings have to say how the machine was configured."""
    if power_mode is None:
        return ""
    label = POWER_MODE_LABELS.get(power_mode)
    return (f", macOS power mode {power_mode} ({label})" if label
            else f", macOS power mode {power_mode}")


# ===================================================================
# Timing plots
# ===================================================================

PYTM30_COLOR = "#10b981"   # green -- pytm30, used consistently below
COLOUR_COLOR = "#2563eb"   # blue  -- colour-science, used consistently below
# Two-tone shades of each implementation colour, used to split the single-eval
# histograms by wavelength-grid size (finest grid gets the first shade). Pairs
# validated for CVD separation, lightness band and chroma on a white surface.
# The green pair starts from PYTM30_COLOR itself; the blue pair was validated
# as a set that does not include COLOUR_COLOR, so it is deliberately not
# written as ``(COLOUR_COLOR, ...)``.
PYTM30_GRID_SHADES = (PYTM30_COLOR, "#047857")
COLOUR_GRID_SHADES = ("#60a5fa", "#1d4ed8")


def hist_panel(ax, data, color, xlabel, title, groups=None):
    """Plain histogram (no scipy KDE dependency) showing the actual
    distribution shape -- a mean+/-std number alone hides skew, outliers,
    and multi-modality.

    `groups` is an optional list of (subset, color, label) that splits the
    histogram into labelled series over shared bins. Used for the single-eval
    panels, whose bimodality is a corpus property (two wavelength-grid sizes,
    per-call cost scales with grid length) that an unlabelled histogram
    presents as a mystery. The series are stacked rather than drawn over each
    other: overlaid bars hide whichever series is drawn first in every shared
    bin, so bar heights would stop reading as counts."""
    bins = np.histogram_bin_edges(data, bins=min(30, max(10, len(data) // 10)))
    if groups:
        ax.hist([g_data for g_data, _, _ in groups], bins=bins, stacked=True,
                color=[g_color for _, g_color, _ in groups], edgecolor="white",
                alpha=0.85,
                label=[f"{g_label} (n={len(g_data)})"
                       for g_data, _, g_label in groups])
        ax.legend(fontsize=7)
    else:
        ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=9)


def grid_sizes(npts, shades):
    """Distinct wavelength-grid lengths in `npts`, finest first, or None when
    the split cannot be drawn (one grid only, or more grids than there are
    shades). Single predicate behind both the split and everything that
    describes it, so a figure can never carry a caption for a split that was
    not made."""
    sizes = sorted(set(np.asarray(npts).tolist()), reverse=True)
    if len(sizes) < 2 or len(sizes) > len(shades):
        return None
    return sizes


def grid_split(times_ms, npts, shades):
    """Split per-call timings into one histogram series per wavelength-grid
    size (finest grid first), pairing each with a shade of the panel's
    implementation colour. Returns None when the split cannot be drawn, so
    the caller falls back to a plain unsplit histogram."""
    sizes = grid_sizes(npts, shades)
    if sizes is None:
        return None
    npts = np.asarray(npts)
    return [(times_ms[npts == n], shade, f"{n}-pt grid")
            for n, shade in zip(sizes, shades)]


CAPTION_FONTSIZE = 8
CAPTION_WIDTH_CHARS = 100
# tight_layout() lays out axes only and ignores fig.text, so the rect it is
# given is the caption's only protection against the panels' x-axis labels
# landing on top of it. Reserve the band in inches -- what the caption
# actually occupies -- and convert to the figure fraction tight_layout wants,
# rather than hand-tuning one fraction per figure height.
CAPTION_LINE_IN = 0.16
CAPTION_PAD_IN = 0.06


def grid_caption(panels, cost_clause, steps):
    """Caption explaining the split single-eval panels, or None when no panel
    was actually split (single-grid corpus, more grids than shades, or panels
    that split differently) -- in which case the figure gets no caption
    instead of one describing a split it does not show.

    `panels` is the list of (npts, shades) pairs handed to grid_split, and
    `steps` the {grid length: wavelength step} map from grid_steps()."""
    sizes = None
    for npts, shades in panels:
        panel_sizes = grid_sizes(npts, shades)
        if panel_sizes is None or (sizes is not None and panel_sizes != sizes):
            return None
        sizes = panel_sizes
    listed = " and ".join(f"{n}-pt ({steps[n]:g} nm)" for n in sizes)
    return textwrap.fill(
        f"Single-eval panels are split by SPD wavelength grid: per-call cost scales with grid "
        f"length ({cost_clause}), so the {listed} illuminants form {len(sizes)} timing modes. "
        f"The batch matrix is resampled onto one common 1 nm grid by this benchmark script "
        f"before the timed call -- that resampling is not part of the batch timing -- hence "
        f"the batch panels' single mode.",
        width=CAPTION_WIDTH_CHARS)


def add_caption(fig, caption):
    """Draw `caption` in a band reserved at the bottom of `fig`, then lay the
    panels out above it. Plain tight_layout when there is no caption."""
    if not caption:
        fig.tight_layout()
        return
    band_in = CAPTION_LINE_IN * (caption.count("\n") + 1) + CAPTION_PAD_IN
    fig.text(0.5, 0.005, caption, ha="center", va="bottom",
             fontsize=CAPTION_FONTSIZE, color="#555555")
    fig.tight_layout(rect=(0.0, band_in / fig.get_figheight(), 1.0, 1.0))
