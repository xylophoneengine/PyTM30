# pytm30 - TM-30-20 Colour Rendition in C++20

Welcome! ANSI/IES TM-30-20 colour fidelity & gamut metrics implemented from
the spec in C++20 with zero runtime dependencies, and Python bindings via
nanobind.

VIBECODE-ALERT!!! Opencode and Claude did help me.

> **What is TM-30-20?** ANSI/IES TM-30-20 is the modern replacement for CRI
> (Ra) for evaluating how a light source renders colour. It scores a source
> on two independent axes - **Rf** (fidelity: how close colours look to a
> reference illuminant) and **Rg** (gamut: whether colours look more or less
> saturated than the reference) - plus 16 hue-angle bins for a finer-grained
> profile than a single number can give.

## Why pytm30?

The two existing Python implementations of TM-30,
[luxpy](https://github.com/ksmet1977/luxpy) and
[colour-science](https://github.com/colour-science/colour-science), are both
mature, well-validated references - but neither is built for **mass
evaluation**. Their pipelines are geared toward one spectrum (or a modest
handful) at a time; once you're scoring tens of thousands of SPDs, per-call
Python overhead and repeated table resampling dominate the runtime.

That was exactly the workload I needed: evaluate huge batches of spectra,
fast. So I designed a different architecture for it - a wavelength grid
resampled and cached once instead of per call, a true contiguous-array batch
API instead of a Python loop over single-SPD calls, a zero-heap-allocation
hot path, domain validity (out-of-range CCT/Duv) modeled as result data
rather than exceptions - and had it implemented from the TM-30-20 spec,
slice by slice, by AI coding agents, verifying every stage against
colour-science (and, in the original internal version, luxpy) as an
accuracy oracle.

The payoff, measured against colour-science on the bundled illuminant
corpus (`benchmarks/benchmark_tm30.py`): pytm30 runs **~6.4x faster** on
single-SPD evaluation and **~5x faster** in batch, with Rf/Rg/CCT/Duv within
floating-point noise of colour-science's own values.

---

## Contents

- [pytm30 - TM-30-20 Colour Rendition in C++20](#pytm30---tm-30-20-colour-rendition-in-c20)
  - [Why pytm30?](#why-pytm30)
  - [Contents](#contents)
  - [Quick Start (Python)](#quick-start-python)
    - [The full result set - `extras=True`](#the-full-result-set---extrastrue)
    - [Convenience: SPD → XYZ / Yuv](#convenience-spd--xyz--yuv)
    - [Configure CMF Observer](#configure-cmf-observer)
    - [Configure Integration Range](#configure-integration-range)
  - [Quick Start (C++)](#quick-start-c)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Python](#python)
    - [C++ only](#c-only)
    - [Troubleshooting](#troubleshooting)
  - [Architecture](#architecture)
  - [Performance](#performance)
  - [Data Files \& Provenance](#data-files--provenance)
  - [Tests](#tests)
  - [Design Principles](#design-principles)
  - [Project Structure](#project-structure)
  - [Author](#author)
  - [License](#license)

---

## Quick Start (Python)

```python
import numpy as np
from tm30_calc import TM30Calc, Cmf

calc = TM30Calc()
spd  = np.loadtxt("my_spectrum.csv")          # 380-780 nm, 1 nm step (401 values)
res  = calc.eval(spd)

print(f"Rf      = {res.rf:.1f}")
print(f"Rg      = {res.rg:.1f}")
print(f"CCT     = {res.cct:.0f} K")
print(f"Duv     = {res.duv:.6f}")
print(f"Rf,skin = {res.rf_skin:.1f}")

# Batch: pass a 2-D array
spd_matrix = np.loadtxt("many_spectra.csv")    # shape (N, 401)
results    = calc.eval(spd_matrix)
```

`TM30Calc` loads all reference tables once at construction; `eval()` is the
hot path and never re-reads a file. Batch evaluation never raises per-SPD -
failures simply drop out of the results list, so one malformed spectrum in a
1000-row batch won't kill the run.

### The full result set - `extras=True`

By default `eval()` returns the 9 headline fields (`rf`, `rg`, `cct`, `duv`,
`delta_e_avg`, `rf_skin`, `rf_cesi`, `rcs_hj`, `rhs_hj`). Passing
`extras=True` additionally unlocks local per-bin fidelity, the Color Vector
Graphic plot geometry, and per-sample colorimetry - everything needed to
build your own TM-30 report or plot:

```python
res = calc.eval(spd, extras=True)

res.rf_hj, res.de_hj                       # local fidelity / mean ΔE′, 16 values each
res.cvg_x_test, res.cvg_y_test             # CVG test-vector plot coordinates, 16 values each
res.cvg_x_ref,  res.cvg_y_ref              # CVG reference-circle plot coordinates
res.reference_spd                          # resampled reference-illuminant SPD
res.xyz_test_ces,  res.xyz_ref_ces         # per-sample XYZ, shape (99, 3) each
res.jab_test_ces,  res.jab_ref_ces         # per-sample CAM02-UCS J'a'b', shape (99, 3) each
res.hue_bin_index                          # which of the 16 hue bins each sample fell into
```

`extras` defaults to `False` because these add roughly 1,100 numbers per
SPD - negligible for one spectrum, real memory for a 100,000-row batch. See
`notebooks/pytm30_tutorial.ipynb` for a full walkthrough including plotting
the CVG "flower" graphic from these fields.

Going the other direction, `bins=False` and/or `samples=False` opt *out* of
`rcs_hj`/`rhs_hj` and `rf_cesi` respectively, skipping the corresponding
array allocation/copy entirely - a genuine memory/bandwidth win for large
batches when you only need the scalar fields (`rf`, `rg`, `cct`, `duv`,
`delta_e_avg`, `rf_skin`).

### Convenience: SPD → XYZ / Yuv

```python
xyz = calc.spd_to_xyz(spd)      # → np.array([X, Y, Z]),    Y=100 (auto)
yuv = calc.spd_to_Yuv(spd)      # → np.array([Y, u′, v′]),  Y=100 (auto)
xyz_raw = calc.spd_to_xyz(spd, K=1.0)   # raw integrals
```

### Configure CMF Observer

```python
from tm30_calc import Cmf

calc = TM30Calc(cmf=Cmf.CIE_1964_10)   # default, enum tab-complete
calc = TM30Calc(cmf='1931_2')          # string, case-insensitive
calc = TM30Calc(cmf='data/my_cmf.csv') # custom CSV path
calc = TM30Calc(cmf_2deg=Cmf.CIE_2015_2)  # separate 2° observer for CCT
```

Available: `CIE_1931_2`, `CIE_1964_10`, `CIE_2006_2`, `CIE_2006_10`,
`CIE_2015_2`, `CIE_2015_10`.

### Configure Integration Range

```python
calc = TM30Calc(lambda_min=360, lambda_max=830)  # wider range
calc = TM30Calc(lambda_min=380, lambda_max=780)  # TM-30-20 standard
```

---

## Quick Start (C++)

pytm30 is, underneath, a self-contained C++20 library - the Python bindings
are a thin wrapper, not where the functionality lives. Everything above is
also directly available with zero Python involvement:

```cpp
#include "tm30/tm30.hpp"
#include "tm30/csv_loader.hpp"

// Load data tables once
auto cmf2  = load_cmf("data/cie_1931_2.csv");
auto cmf10 = load_cmf("data/cmf_1964_10.csv");
auto ces   = load_ces("data/ces.csv");
auto basis = load_daylight_basis("data/daylight_basis.csv");
auto lut   = load_planckian_lut("data/planckian_uv.csv");

// Evaluate one SPD
std::vector<double> wl(401), spd(401);  // … populate …
tm30::Spd myspd(std::move(wl), std::move(spd));
tm30::Tm30 m(myspd, cmf2, cmf10, ces, basis, lut);

double rf  = m.rf();     // runs pipeline, caches
double rg  = m.rg();     // reuses cache
double cct = m.cct();

// Everything extras=True exposes in Python is a plain method here too:
const auto& cvg = m.cvg();                          // CVG plot coordinates
const auto& local = m.local_chroma_shift();          // Rf,hj / Rcs,hj / Rhs,hj / DE_hj
const auto& colorimetry = m.colorimetry_result();    // full raw result
```

---

## Installation

### Prerequisites

| Tool           | Version              | Notes                                                                                                                                            |
| -------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| C++20 compiler | GCC ≥11 or Clang ≥14 | Includes Apple Clang on macOS. No compiler-specific extensions - should build clean under `-std=c++20` on Linux, macOS, and Windows, x86_64 and arm64. |
| CMake          | ≥3.20                | `pip install cmake` works fine if you'd rather not touch your system package manager.                                                            |
| Python         | ≥3.10                | Only needed for the Python bindings.                                                                                                             |

> **Platform status:** the code is written to be portable (standard C++20,
> no platform-specific extensions or intrinsics), but it has so far only
> been built and tested on an Apple MacBook (M4, Apple Silicon). Other
> platforms/architectures should work but haven't been verified yet -
> reports (or PRs) welcome.

### Python

```bash
git clone https://github.com/xylophoneengine/PyTM30.git
cd pytm30

pip install .
```

That's it - `pip install .` builds the C++ core and the Python bindings in
one step via [scikit-build-core](https://github.com/scikit-build/scikit-build-core),
and installs a real, importable `tm30_calc` package. Verify:

```bash
python3 -c "
import numpy as np
from tm30_calc import TM30Calc
print(TM30Calc().eval(np.full(401, 100.0)))
"
```

For development (editable install, rebuilds on source change require
re-running the install):

```bash
pip install --no-build-isolation -e .
```

### C++ only

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build -j$(nproc)
./build/tm30-tests
```

There's no `cmake --install` target yet - link directly against the build
output: `build/libtm30-core.a`, with `include/` on your include path.

### Troubleshooting

- **`cmake: command not found`** - `pip install cmake` into the environment
  you're installing into, or install it via your system package manager.
- **`import nanobind` fails during build** - build isolation should handle
  this automatically; if you used `--no-build-isolation`, run
  `pip install nanobind` into the same environment first.

---

## Architecture

```
spd → [validate] → [resample CES + CMF] → [XYZ] → [CCT + Reference]
    → [CIECAM02 J'a'b'] → [ΔE′] → [Rf] → [Hue Bins] → [Gamut Rg + Local + CVG]
```

- **Zero heap in the hot path** - `std::array<double,99>` and `std::array<double,16>` throughout the per-SPD pipeline
- **Lazy memoized handle** - construction validates, first access computes, subsequent accesses return cache
- **Batch API** - per-SPD failures return `std::nullopt` / drop from the results list, never throws
- **Core has no Python dependency** - bindings are a separate CMake target in `python/`

## Performance

~40-80 µs per SPD (single-threaded, full TM-30 pipeline). Batch evaluation
amortizes table-loading overhead - see `benchmarks/benchmark_tm30.py`.

## Data Files & Provenance

| File                                                                                             | Description                                                    | Range                |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- | -------------------- |
| `ces.csv` / `ces_5nm.csv`                                                                        | 99 CES reflectance spectra                                     | 380-780 nm           |
| `cmf_1964_10.csv`                                                                                | CIE 1964 10° CMFs (default observer)                           | 360-830 nm, 1 nm     |
| `cmf_1931_2.csv` / `cie_1931_2.csv`                                                              | CIE 1931 2° CMFs (general / CCT-default)                       | 360-830 / 380-780 nm |
| `cmf_2006_2.csv` / `cmf_2006_10.csv`                                                             | CIE 2006 physiologically-based CMFs                            | 360-830 nm           |
| `cmf_2015_2.csv` / `cmf_2015_10.csv`                                                             | CIE 2015 CMFs                                                  | 360-830 nm           |
| `daylight_basis.csv`                                                                             | CIE daylight vectors S₀, S₁, S₂                                | 300-830 nm, 5 nm     |
| `planckian_uv.csv`                                                                               | Planckian locus LUT (u,v)                                      | 1000-25000 K         |
| `d65_1nm.csv`, `fl1_1nm.csv`…`fl12_1nm.csv`, `hp1_1nm.csv`…`hp5_1nm.csv`, `illuminant_a_1nm.csv` | Standard illuminant/lamp spectra, used in tests and benchmarks | 380-780 nm           |

All data tables are sourced from **[colour-science](https://github.com/colour-science/colour-science)**
(BSD-3-Clause), not hand-derived - every one is a universal, published CIE
standard quantity (color-matching functions, the CIE 2017/TM-30 test-colour
samples, the daylight-locus basis functions), verified to match to
floating-point noise against independent computation. `tools/generate_data_colour_science.py`
and `tools/generate_planckian_lut.py` regenerate every file from scratch and
document exactly which colour-science API produced each one. `tools/generate_fixtures.py`
uses the same oracle to regenerate the golden test fixtures in `tests/fixtures/`.

## Tests

```bash
./build/tm30-tests                  # 191 tests, 5966 assertions
python3 tools/check_constants.py    # 0 uncited float literals
```

Every numeric constant in `src/`/`include/` cites the TM-30-20 spec section
it comes from; `check_constants.py` enforces this mechanically.

---

## Design Principles

- **No magic numbers** - every constant cites its TM-30-20 section
- **Domain validity ≠ errors** - out-of-range CCT/Duv are result flags, not exceptions
- **SPD is never interpolated** - resample CES and CMFs to the test SPD's grid
- **Linear interpolation** per TM-30-20 §3.5

---

## Project Structure

```
pytm30/
├── include/tm30/        # Public headers (standalone-compilable)
├── src/tm30/            # Implementation
├── tests/               # Catch2 test suite + golden fixtures
├── data/                # CSV data tables (CMFs, CES, daylight basis, Planckian LUT, illuminants)
├── python/              # nanobind bindings + TM30Calc convenience wrapper
│   ├── tm30_calc.py
│   └── tm30_bindings.cpp
├── tools/               # check_constants.py, data/fixture generation scripts
├── notebooks/           # pytm30_tutorial.ipynb - full walkthrough
├── benchmarks/          # speed + accuracy benchmarks vs. colour-science
├── pyproject.toml       # pip-installable via scikit-build-core
└── CMakeLists.txt       # C++ build
```

## Author

**Adrian Zwenger** - [ALSVV, Technical University of Darmstadt](https://www.lichttechnik.tu-darmstadt.de/)
(Laboratory of Adaptive Lighting Systems and Visual Processing)

Questions or issues - please use [GitHub Issues](../../issues).

If you use pytm30 in published work, please cite it - see
[CITATION.cff](CITATION.cff) (also picked up by GitHub's "Cite this
repository" button).

## License

MIT - see [LICENSE](LICENSE).
