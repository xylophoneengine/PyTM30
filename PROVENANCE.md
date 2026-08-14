# Provenance

PyTM30 implements ANSI/IES TM-30-20 from the published standard:

> ANSI/IES TM-30-20, *IES Method for Evaluating Light Source Color
> Rendition*, An American National Standard, Illuminating Engineering
> Society, New York, 2020, incorporating Errata 1 (2021).
> ISBN 978-0-87995-379-9. (IES standards are issued without DOIs.)

## Data

Spectral data is sourced from colour-science (BSD-3-Clause) and CIE publications:

> Mansencal, T., Mauderer, M., Parsons, M., et al. (2025). *Colour 0.4.7*
> [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.17837391>
> Repository: <https://github.com/colour-science/colour>
> Website: <https://www.colour-science.org>
Per-table provenance is recorded in tools/generate_data_colour_science.py. Every
shipped table is byte-reproducible by running that script and
tools/generate_planckian_lut.py against colour-science 0.4.7 with numpy 2.5.2.
Last verified 2026-08-14 on macOS arm64 (Python 3.12.13, all 30 tables, zero
byte differences); the data-reproducibility CI job in .github/workflows/ci.yml
repeats the check on Linux with both versions pinned. Byte-level
reproducibility is claimed only for that pinned combination -- other versions
of colour-science or numpy may produce numerically equivalent but
byte-different tables and have not been tested.
The CES reflectance data and the TM-30-20 method originate with the IES and CIE.

## Validation

Numerical accuracy is cross-validated against colour-science (BSD-3-Clause).
The golden test fixtures are produced by tools/generate_fixtures.py from
colour-science primitives plus the standard's own equations; see
docs/divergences.md for exactly which quantities are independently validated
and by what.

## Development history

During early development, luxpy (GPL-3.0) was used privately as an accuracy oracle
alongside colour-science. No luxpy code or data is present in this repository; all
data tables were regenerated from colour-science and CIE sources, and all
implementation decisions are derived from the TM-30-20 text with the relevant clause
cited at each site.

## Documented divergences

Where TM-30-20 and existing implementations differ, this library follows the
standard. See docs/divergences.md.
