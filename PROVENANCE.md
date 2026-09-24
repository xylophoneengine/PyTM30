# Provenance

PyTM30 implements ANSI/IES TM-30-20 from the published standard:

> ANSI/IES TM-30-20, *IES Method for Evaluating Light Source Color
> Rendition*, An American National Standard, Illuminating Engineering
> Society, New York, 2020, incorporating Errata 1 (2021).
> ISBN 978-0-87995-379-9. (IES standards are issued without DOIs.)

## Data

The spectral data tables in `data/` are CIE datasets, published by the
International Commission on Illumination (CIE) under CC BY-SA 4.0 and
distributed here under the same licence, separately from the MIT code.
`data/README.md` lists each file's source DOI and the changes made. They were
extracted using colour-science (BSD-3-Clause) as a tool:

> Mansencal, T., Mauderer, M., Parsons, M., et al. (2025). *Colour 0.4.7*
> [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.17837391>
> Repository: <https://github.com/colour-science/colour>
> Website: <https://www.colour-science.org>
Per-table provenance is recorded in tools/generate_data_colour_science.py. Every
shipped table is byte-reproducible by running that script and
tools/generate_planckian_lut.py on the generating platform, macOS arm64,
against colour-science 0.4.7 (verified byte-identical there under both
numpy 2.3.3 and numpy 2.5.2; last verified 2026-08-14, all tables, zero byte
differences). Byte-identity does NOT transfer across architectures: numpy and
libm round differently in the last ULP (observed max 17 ULP regenerating on
Linux x86_64), so the data-reproducibility CI job regenerates the tables on
Linux with the same pins and enforces numeric equivalence instead -- identical
structure and every value within 64 ULP (or 1e-12 absolutely, for near-zero
cancellation results) via tools/check_data_reproducibility.py. Byte-level
reproducibility is claimed only for the pinned combination on macOS arm64;
other versions of colour-science or numpy have not been tested.
The CES reflectance data and the TM-30-20 method originate with the IES and CIE.

## Validation

Numerical accuracy is cross-validated against colour-science (BSD-3-Clause).
The golden test fixtures are produced by tools/generate_fixtures.py from
colour-science primitives plus the standard's own equations; see
docs/divergences.md for exactly which quantities are independently validated
and by what.

## Development history

PyTM30 was first developed outside this repository, with luxpy (GPL-3.0) as
its main accuracy oracle: the C++ implementation and its test suite were
validated against fixtures generated with luxpy. Before the first commit here
(35848d4), the data tables and fixtures were regenerated from CIE data using
colour-science, so that commit already contains a finished implementation.

Until 2026-08-14, some sites followed luxpy's conventions where luxpy departs
from TM-30-20: the Rcs/Rhs scaling, the CVG display scale and reference hue
angle, and a shifted-triangular CCT blend that was always on. A remediation
series merged that day (3505b16) changed those sites to follow the standard,
regenerated the fixtures, recomputed the test literals that were still luxpy
output values (e49e69f), and replaced luxpy-based justifications in comments
with citations to TM-30-20 and its normative references. Most clause
citations were attached to existing code in that series: they record which
clause each site implements, not the order in which the code was written.

Where the standard is silent, two choices still follow luxpy's TM-30
configuration: the extent of the Planckian LUT (tools/generate_planckian_lut.py)
and the optional shifted-triangular blend, now off by default. The Python
layer's result names follow luxpy's spd_to_tm30() for compatibility. Other
implementation choices the standard does not dictate are listed in
docs/divergences.md.

luxpy is still used from time to time as a comparison oracle during
development. It is not a dependency, and no luxpy code is present in this
repository or its history.

## Documented divergences

Where TM-30-20 and existing implementations differ, this library follows the
standard. See docs/divergences.md.
