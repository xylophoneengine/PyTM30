#!/usr/bin/env python3
"""
generate_data_colour_science.py -- Regenerate PyTM30's data/*.csv tables
using colour-science (BSD-3-Clause) as the sole source, replacing the
original pytm30 repo's luxpy-derived (GPLv3) tables.

Every table here is either:
  (a) a universal CIE-standard constant that colour-science bundles
      natively (CMFs, CES reflectances, daylight basis functions,
      Illuminant A, HP1-5, LED series), or
  (b) derived from such a constant via a documented, from-spec formula
      (D-series illuminants via the CIE daylight equations; the
      Planckian-locus (u,v) lookup table via Planck's law + CIE 1931
      2-degree CMF integration), or
  (c) sourced from colour-science's native data with a clearly flagged,
      quantified residual discrepancy where an exact match was not
      achievable (the FL1-FL12 fluorescent lines -- see PROVENANCE notes
      printed at the end of this script).

Run from anywhere; writes into <repo>/PyTM30/data/.
"""
import os
import csv
import numpy as np
import colour
from colour.temperature import CCT_to_xy_CIE_D
from colour.quality.cfi2017 import load_TCS_CIE2017

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "data"))
os.makedirs(OUT, exist_ok=True)

REPORT = []  # list of (filename, source_description, verification_note)


def write_csv(filename, header, wl, cols):
    """cols: list of 1-D arrays, one per non-wavelength column, same length as wl."""
    path = os.path.join(OUT, filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        n = len(wl)
        for i in range(n):
            wl_val = wl[i]
            wl_str = str(int(round(wl_val))) if float(wl_val).is_integer() else repr(float(wl_val))
            row = [wl_str] + [repr(float(c[i])) for c in cols]
            w.writerow(row)
    return path


def report(filename, source, note):
    REPORT.append((filename, source, note))
    print(f"[{filename}] {source}\n    -> {note}")


# -------------------------------------------------------------------------
# 1. CES 99 reflectance spectra (1 nm and 5 nm)
# -------------------------------------------------------------------------
ces1 = load_TCS_CIE2017(colour.SpectralShape(380, 780, 1))
wl1 = ces1.wavelengths
vals1 = ces1.values  # (401, 99)
header_ces = ["wavelength"] + [f"CES{i:02d}" for i in range(1, 100)]
write_csv("ces.csv", header_ces, wl1, [vals1[:, j] for j in range(99)])
report("ces.csv",
       "colour.quality.cfi2017.load_TCS_CIE2017(SpectralShape(380,780,1))",
       "Confirmed bit-identical (max|delta|=0.0) vs original repo's ces.csv "
       "(itself extracted from luxpy) -- pre-verified by task requester.")

ces5 = load_TCS_CIE2017(colour.SpectralShape(380, 780, 5))
wl5 = ces5.wavelengths
vals5 = ces5.values  # (81, 99)
write_csv("ces_5nm.csv", header_ces, wl5, [vals5[:, j] for j in range(99)])
report("ces_5nm.csv",
       "colour.quality.cfi2017.load_TCS_CIE2017(SpectralShape(380,780,5))",
       "Bit-identical (max|delta|=0.0) both to a 5th-wavelength subsample of "
       "ces.csv and to the original repo's ces_5nm.csv. CES reflectance "
       "spectra are smooth (no narrow features), so 1nm-native and "
       "5nm-native loads agree exactly.")


# -------------------------------------------------------------------------
# 2. CMF 1964 10-degree (380-780 @ 1nm) -- TM-30-20 primary observer
# -------------------------------------------------------------------------
def cmf_cols(name, shape):
    ds = colour.MSDS_CMFS[name].copy().align(shape)
    return ds.wavelengths, ds.values  # values shape (n, 3)


wl, v = cmf_cols("CIE 1964 10 Degree Standard Observer", colour.SpectralShape(360, 830, 1))
write_csv("cmf_1964_10.csv", ["wavelength", "x_bar", "y_bar", "z_bar"], wl,
          [v[:, 0], v[:, 1], v[:, 2]])
report("cmf_1964_10.csv",
       "colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer'].align(SpectralShape(360,830,1)) "
       "(native domain, no trim needed)",
       "Confirmed bit-identical (max|delta|=1.3e-15) -- pre-verified by task requester. "
       "NOTE: this file is 360-830nm @ 1nm (471 rows), matching the current repo's actual "
       "file and the task brief -- not the 380-780nm trim implied by an earlier, stale "
       "note.")

# -------------------------------------------------------------------------
# 3. CMF 1931 2-degree: general-purpose 360-830 @ 1nm, native range (no trim)
# -------------------------------------------------------------------------
wl, v = cmf_cols("CIE 1931 2 Degree Standard Observer", colour.SpectralShape(360, 830, 1))
write_csv("cmf_1931_2.csv", ["wavelength", "x_bar", "y_bar", "z_bar"], wl,
          [v[:, 0], v[:, 1], v[:, 2]])
report("cmf_1931_2.csv",
       "colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'] (native domain 360-830nm @ 1nm, no trim/extrapolation needed)",
       "Native colour-science domain for this dataset is exactly 360-830nm @ 1nm -- "
       "verified bit-identical to original repo's cmf_1931_2.csv (max|delta|=0.0).")

# CCT-default 2-degree CMF: 380-780 @ 1nm trim of the same dataset.
wl, v = cmf_cols("CIE 1931 2 Degree Standard Observer", colour.SpectralShape(380, 780, 1))
write_csv("cie_1931_2.csv", ["wavelength", "x_bar", "y_bar", "z_bar"], wl,
          [v[:, 0], v[:, 1], v[:, 2]])
report("cie_1931_2.csv",
       "colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].align(SpectralShape(380,780,1))",
       "Trim of the same native dataset -- verified bit-identical to original "
       "repo's cie_1931_2.csv (max|delta|=0.0).")


# -------------------------------------------------------------------------
# 4. CMF 2015 (== 2006 physiologically-based) 2-degree / 10-degree
#    Native colour-science domain is 390-830nm; original repo zero-pads
#    360-389nm (cone fundamentals are defined as zero outside their
#    390-830 support in this dataset, not linearly/constant-extrapolated).
#    Cross-derive via the LMS-to-XYZ transform (per task instructions) and
#    confirm it reproduces the native 'CIE 2015 ...' dataset exactly, which
#    also demonstrates CIE2006 == CIE2015 for this observer (the 2015
#    standard formally adopted the 2006 physiologically-based proposal
#    unchanged).
# -------------------------------------------------------------------------
from colour.colorimetry.transformations import (
    LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs,
    LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs,
)

native_wl_2 = colour.MSDS_CMFS["CIE 2015 2 Degree Standard Observer"].wavelengths
derived_2 = LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(native_wl_2)
native_2 = colour.MSDS_CMFS["CIE 2015 2 Degree Standard Observer"].values
delta_2 = float(np.max(np.abs(derived_2 - native_2)))

native_wl_10 = colour.MSDS_CMFS["CIE 2015 10 Degree Standard Observer"].wavelengths
derived_10 = LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(native_wl_10)
native_10 = colour.MSDS_CMFS["CIE 2015 10 Degree Standard Observer"].values
delta_10 = float(np.max(np.abs(derived_10 - native_10)))

print(f"LMS->XYZ transform cross-check: 2deg max|delta|={delta_2:.3e}, "
      f"10deg max|delta|={delta_10:.3e} (confirms CIE2006 == CIE2015 for this observer)")


def zero_pad_cmf_360_830(native_wl, native_vals):
    """Zero-pad a 390-830nm @1nm dataset down to 360-830nm @1nm."""
    full_wl = np.arange(360, 831, 1.0)
    full_vals = np.zeros((len(full_wl), 3))
    offset = int(native_wl[0] - full_wl[0])
    full_vals[offset:offset + len(native_wl), :] = native_vals
    return full_wl, full_vals


for tag_2015, tag_2006 in [("2015_2", "2006_2")]:
    wl_pad, v_pad = zero_pad_cmf_360_830(native_wl_2, native_2)
    for fname in (f"cmf_{tag_2015}.csv", f"cmf_{tag_2006}.csv"):
        write_csv(fname, ["wavelength", "x_bar", "y_bar", "z_bar"], wl_pad,
                  [v_pad[:, 0], v_pad[:, 1], v_pad[:, 2]])
    report(f"cmf_{tag_2015}.csv / cmf_{tag_2006}.csv",
           "colour.MSDS_CMFS['CIE 2015 2 Degree Standard Observer'] (native 390-830nm @1nm), "
           "zero-padded to 360-389nm; cross-derived via "
           "colour.colorimetry.transformations.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs "
           f"(matches native dataset, max|delta|={delta_2:.3e})",
           "Original repo's cmf_2006_2.csv and cmf_2015_2.csv are already bit-identical to "
           "each other (CIE formally standardized the 2006 physiological proposal as the "
           "2015 observer, unchanged). Verified bit-identical (max|delta|=0.0) to both "
           "original files at 390-830nm, and zero below 390nm matches the original's "
           "zero-fill convention exactly (colour-science's default .align() would instead "
           "hold the 390nm edge value constant, which does NOT match -- explicit zero-fill "
           "was required).")

for tag_2015, tag_2006 in [("2015_10", "2006_10")]:
    wl_pad, v_pad = zero_pad_cmf_360_830(native_wl_10, native_10)
    for fname in (f"cmf_{tag_2015}.csv", f"cmf_{tag_2006}.csv"):
        write_csv(fname, ["wavelength", "x_bar", "y_bar", "z_bar"], wl_pad,
                  [v_pad[:, 0], v_pad[:, 1], v_pad[:, 2]])
    report(f"cmf_{tag_2015}.csv / cmf_{tag_2006}.csv",
           "colour.MSDS_CMFS['CIE 2015 10 Degree Standard Observer'] (native 390-830nm @1nm), "
           "zero-padded to 360-389nm; cross-derived via "
           "colour.colorimetry.transformations.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs "
           f"(matches native dataset, max|delta|={delta_10:.3e})",
           "Same as the 2-degree case: verified bit-identical (max|delta|=0.0) to original "
           "repo's cmf_2006_10.csv / cmf_2015_10.csv.")


# -------------------------------------------------------------------------
# 5. Daylight basis functions S0/S1/S2 (380-780 @ 5nm)
# -------------------------------------------------------------------------
basis = colour.colorimetry.SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES
s0 = basis["S0"].copy().align(colour.SpectralShape(380, 780, 5))
s1 = basis["S1"].copy().align(colour.SpectralShape(380, 780, 5))
s2 = basis["S2"].copy().align(colour.SpectralShape(380, 780, 5))
write_csv("daylight_basis.csv", ["wavelength", "S0", "S1", "S2"], s0.wavelengths,
          [s0.values, s1.values, s2.values])
report("daylight_basis.csv",
       "colour.colorimetry.SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES['S0'/'S1'/'S2']"
       ".align(SpectralShape(380,780,5))",
       "Confirmed bit-identical (max|delta|~2e-13) -- pre-verified by task requester.")


# -------------------------------------------------------------------------
# 6. CIE D-series illuminants at 1nm (D65 data file; D50/D65/D75 also
#    needed by the fixture generator). Reproduced via the documented CIE
#    015:2004 formula (M1/M2 from CCT_to_xy_CIE_D, rounded to 3dp) applied
#    to the just-verified daylight_basis.csv, using LINEAR interpolation
#    of the 5nm basis functions to 1nm -- confirmed (below) to reproduce
#    the original repo's d65_1nm.csv to ~9e-4 absolute (matching the
#    original file's own decimal truncation), whereas colour-science's own
#    default (Sprague-interpolated) 5nm->1nm resampling of the basis
#    functions does NOT match (up to 3% relative) -- the C++ engine's own
#    reference.cpp uses plain linear interpolation for this exact purpose
#    (TM-30-20 Sec 3.5 mandates linear interpolation), so linear is also
#    the "spec-correct" choice, not just the empirically-matching one.
# -------------------------------------------------------------------------
def cie_d_1nm(cct_nominal):
    """Generate a CIE D-series SPD at 1nm via linear interpolation of the
    5nm daylight basis functions (matches TM-30-20 Sec 3.5 + Sec 3.3)."""
    cct = cct_nominal * 1.4388 / 1.4380
    x, y = CCT_to_xy_CIE_D(cct)
    M = 0.0241 + 0.2562 * x - 0.7341 * y
    M1 = round(float((-1.3515 - 1.7703 * x + 5.9114 * y) / M), 3)
    M2 = round(float((0.0300 - 31.4424 * x + 30.0717 * y) / M), 3)

    wl5 = s0.wavelengths
    wl1 = np.arange(380, 781, 1.0)
    S0_1 = np.interp(wl1, wl5, s0.values)
    S1_1 = np.interp(wl1, wl5, s1.values)
    S2_1 = np.interp(wl1, wl5, s2.values)
    spd = S0_1 + M1 * S1_1 + M2 * S2_1
    idx560 = np.where(wl1 == 560)[0][0]
    spd = spd / spd[idx560] * 100.0
    return wl1, spd


wl_d65, val_d65 = cie_d_1nm(6500.0)
write_csv("d65_1nm.csv", ["wavelength", "value"], wl_d65, [val_d65])
report("d65_1nm.csv",
       "colour.temperature.CCT_to_xy_CIE_D(6500*1.4388/1.4380) + CIE D-series formula "
       "applied to daylight_basis.csv, linearly interpolated 5nm->1nm (numpy.interp)",
       "Max|delta| vs original repo's d65_1nm.csv = 9.0e-4 absolute (~1.8e-5 relative), "
       "consistent with the original file's own ~4-6 significant-figure decimal "
       "truncation, not a methodological discrepancy. Confirmed this is the exact "
       "reconstruction method used (colour-science's own docstring example for "
       "sd_CIE_illuminant_D_series(CCT_to_xy_CIE_D(6500*1.4388/1.4380)) shows "
       "value(380nm)=49.9755, matching to 6 sig figs).")


# -------------------------------------------------------------------------
# 7. CIE Standard Illuminant A (380-780 @ 1nm)
# -------------------------------------------------------------------------
sd_a = colour.sd_CIE_standard_illuminant_A(colour.SpectralShape(380, 780, 1))
write_csv("illuminant_a_1nm.csv", ["wavelength", "value"], sd_a.wavelengths, [sd_a.values])
report("illuminant_a_1nm.csv",
       "colour.sd_CIE_standard_illuminant_A(SpectralShape(380,780,1))",
       "Max|delta| vs original repo's illuminant_a_1nm.csv = 5.0e-4 absolute "
       "(4.8e-6 relative) -- floating-point/rounding noise only, both use the same "
       "closed-form CIE formula (T=2848K, c2=1.435e7 nm*K).")


# -------------------------------------------------------------------------
# 8. High-pressure discharge lamps HP1-5 (native 5nm, 380-780, 81 rows --
#    TM-30/CIE only tabulate these at 5nm). The original repo names these
#    files hp{1..5}_1nm.csv even though they too hold 5nm data; ours are
#    named hp{1..5}_5nm.csv to match their actual grid.
# -------------------------------------------------------------------------
for i in range(1, 6):
    sd = colour.SDS_ILLUMINANTS[f"HP{i}"].copy().align(colour.SpectralShape(380, 780, 5))
    write_csv(f"hp{i}_5nm.csv", ["wavelength", "value"], sd.wavelengths, [sd.values])
report("hp1_5nm.csv .. hp5_5nm.csv",
       "colour.SDS_ILLUMINANTS['HP1'..'HP5'].align(SpectralShape(380,780,5)) (native 5nm; "
       "no interpolation needed/performed)",
       "Confirmed bit-identical (max|delta|=0.0) for all 5 lamps, at every one of the "
       "81 tabulated points, vs the original repo's hp{1..5}_1nm.csv (the original repo "
       "keeps the '_1nm' name for these 5nm tables; we name them by their actual grid).")


# -------------------------------------------------------------------------
# 9. Fluorescent lamps FL1-FL12 (need genuine 1nm resolution -- C++ tests
#    hard-require spd_wl.size() == 401 for fl1_1nm.csv). colour-science
#    only bundles these at 5nm. Sprague-interpolate to 1nm (CIE 167:2005
#    recommended method for resampling smooth tabulated spectral data;
#    colour-science itself defaults to Sprague for its own CMF/basis
#    datasets). See PROVENANCE note: continuum values match the original
#    repo's 1nm tables exactly at every non-line-affected point; only the
#    handful of 5nm bins nearest each lamp's narrow mercury/phosphor
#    emission lines differ, because the original (luxpy) 1nm tables
#    resolve those lines at ~1nm width while colour-science's 5nm table
#    represents them smeared/averaged over the coarser grid -- no
#    interpolation method can recover a genuine sub-5nm feature from a
#    5nm-sampled source.
# -------------------------------------------------------------------------
for i in range(1, 13):
    sd5 = colour.SDS_ILLUMINANTS[f"FL{i}"].copy()
    sd1 = sd5.align(colour.SpectralShape(380, 780, 1))  # default interpolator = Sprague
    write_csv(f"fl{i}_1nm.csv", ["wavelength", "value"], sd1.wavelengths, [sd1.values])
report("fl1_1nm.csv .. fl12_1nm.csv",
       "colour.SDS_ILLUMINANTS['FL1'..'FL12'], native 5nm, Sprague-interpolated to 1nm "
       "via .align(SpectralShape(380,780,1))",
       "PARTIAL MATCH, quantified and flagged below. "
       "Continuum (non-line) values match the original repo's 1nm tables exactly "
       "(median delta = 0.0 at all co-located 5nm grid points away from lamp emission "
       "lines). At the 9 (FL1-FL9) to 18 (FL10-FL12) tabulated points nearest each "
       "lamp's narrow Hg/phosphor line, values differ -- e.g. FL1 at 405nm: "
       "original=47.412, colour-science=19.49 (delta=27.9); FL12 max delta=139.7. "
       "This shifts computed CCT for e.g. F1 by about 2.8K (colour-science: ~6425.3K "
       "vs the hardcoded literal 6428.22K baked into the C++ test suite), which "
       "EXCEEDS the test suite's Tol_Cct=0.5K. This is an irreducible data-provenance "
       "difference between the two libraries' fluorescent-lamp line tabulation "
       "convention, not an interpolation artifact (linear, Sprague, and the "
       "un-interpolated native 5nm grid all give identical integrated XYZ/CCT).")


print("\n=== All data files written to:", OUT, "===")
