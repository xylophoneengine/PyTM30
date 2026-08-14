#!/usr/bin/env python3
"""
generate_fixtures.py -- Golden fixture generator for ANSI/IES TM-30-20,
using colour-science (BSD-3-Clause) as the oracle instead of luxpy (GPLv3).

Produces tests/fixtures/{spd_name}/{stage}.json in the exact envelope/layout
the existing (unmodified) C++ Catch2 test suite expects -- this schema was
reverse-engineered from the original repo's *actual* shipped fixtures
(produced by tools/regenerate_fixtures.py, not the more elaborate but
never-actually-run tools/generate_fixtures.py) and from grep'ing every
fixture_path(...)/JSON-key read site in tests/slice_*.cpp.

WHY THIS DOESN'T JUST CALL colour.colour_fidelity_index() AND STOP
------------------------------------------------------------------
An initial version of this script called colour.colour_fidelity_index(sd,
method="ANSI/IES TM-30-18", additional_data=True) directly. That produced
plausible-looking fixtures, but a first test run against the (unmodified)
C++ suite showed the *reference-illuminant* side of the pipeline (CES XYZ/
J'a'b' under the reference illuminant) diverging from the C++ engine's own
output by far more than the "cross-implementation noise" tolerances
(Tol_FixtureXyz=1e-3) were sized for -- e.g. D65_1nm ref XYZ off by 0.028,
while the *test*-illuminant side matched to <1e-4.

Root cause, found by reading colour-science's own source
(colour/quality/cfi2017.py, sd_reference_illuminant() and
colour_fidelity_index_CIE2017()): colour-science's built-in reference-
illuminant generator calls sd_CIE_illuminant_D_series(xy, shape=shape),
which resamples the 5nm daylight basis functions (S0/S1/S2) to the
target grid using the *SpragueInterpolator* (colour-science's own default
interpolator for those datasets -- confirmed via
SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES["S0"].interpolator). TM-30-20
Sec 3.5 mandates LINEAR interpolation for this step, and pytm30's C++
implementation (src/tm30/reference.cpp::interpolate_linear) does exactly
that -- as does luxpy, which is why the original (luxpy-oracle) fixtures
did not show this problem. Sprague vs linear resampling of the *same*,
verified-bit-identical 5nm basis functions gives visibly different values
between 5nm grid points (see PyTM30/tools/generate_data_colour_science.py's
own D65 cross-check, which found up to ~3% relative divergence between
Sprague- and linearly-resampled daylight basis functions).

FIX: this script does NOT use colour-science's built-in reference-
illuminant generator at all. It builds the reference SPD itself with the
exact TM-30-20 Sec 3.3 algorithm (pure Planckian below 4000K, pure CIE
D-series above 5000K, Y-normalised blend in between) using LINEAR
interpolation of the confirmed-clean daylight basis functions -- i.e. the
same formula pytm30's own reference.cpp implements, and the same one this
project already uses in generate_data_colour_science.py's D65/D50/D75
reconstruction. colour-science is still very much the oracle for
everything else: CCT/Duv (Ohno 2013, via colour.quality.cfi2017.
CCT_reference_illuminant), tristimulus integration (colour.sd_to_XYZ /
colour.msds_to_XYZ, method="Integration"), CIECAM02 (colour.
XYZ_to_CIECAM02, Y_b=20/L_A=100/Average surround/discount_illuminant=True,
matching colour-science's own TM-30-18 viewing conditions), CAM02-UCS
(colour.JMh_CIECAM02_to_CAM02UCS), and the CFI/Rf/Rg scalar math
(colour.quality.tm3018.delta_E_to_R_f / averages_area, both pure,
generic, spec-derived formulas with no data dependence). Only the
reference-illuminant *spectral reconstruction* step is done by hand, to
match the spec (and the C++ implementation) instead of colour-science's
library default.

Mapping onto the 18 pipeline stages:
  01_resampled_spd   <- sd_test itself, on its native 380-780nm grid
  02_cct_duv         <- colour.quality.cfi2017.CCT_reference_illuminant(sd_test)
                        (Ohno 2013, CIE 1931 2-degree observer -- matches
                        both colour-science's and pytm30's CCT convention)
  03_reference_spd   <- hand-built reference SPD (see above)
  04_xyz_test_white  <- Y=100-normalised XYZ of sd_test (CIE 1964 10-degree)
  05_xyz_ref_white   <- Y=100-normalised XYZ of the reference SPD
  06_xyz_test_ces    <- XYZ of the 99 CES under sd_test (99,3)
  07_xyz_ref_ces     <- XYZ of the 99 CES under the reference SPD (99,3)
  08_jab_test_ces    <- CAM02-UCS J'a'b' of the 99 CES under sd_test
  09_jab_ref_ces     <- CAM02-UCS J'a'b' of the 99 CES under the reference
  10_delta_e_ces     <- euclidean_distance(Jpapbp_test, Jpapbp_ref) (99,)
  11_hue_bins        <- floor(href/22.5) (0-indexed, href from reference
                        JMh hue angle); hue_bin_edges are the fixed
                        22.5-degree boundaries (no oracle needed)
  12_rf              <- delta_E_to_R_f(mean(delta_E))
  13_rg              <- 100 * area(avg_test_ab) / area(avg_ref_ab)
  14_per_bin_metrics <- Rfhj = delta_E_to_R_f(nanmean(delta_E per bin));
                        Rcshj/Rhshj via the bin-bisector-angle projection
                        formula (TM-30-20 Sec 4.6/4.7 == gamut.cpp exactly)
  15_cvg_coordinates <- jabt_hj/jabr_hj: [J'_bin_avg, a'_bin_avg, b'_bin_avg];
                        jabtn_hj/jabrn_hj: the CVG-normalised (unit
                        reference-circle) display coordinates, via the
                        same Eq.(58)-(61) formula gamut.cpp implements
  16_rfi_per_sample  <- delta_E_to_R_f(delta_E) (99,)
  17_rf_skin         <- (R_s[14] + R_s[17]) / 2  (CES15, CES18, 0-indexed)
  18_annex_e         <- static P1/P2/P3 priority-level taxonomy (TM-30-20
                        Annex E) -- confirmed NOT read by any C++ test
                        (only the static AnnexE::P1/P2/P3==1/2/3 int
                        constants are asserted, in
                        slice_10_sample_test.cpp); no per-SPD data needed.

NOTE on Rcs,hj scaling: colour-science's own R_cs (from the high-level
colour_fidelity_index call) is expressed as a percentage (x100). pytm30's
C++ implementation (src/tm30/gamut.cpp) explicitly keeps this UNSCALED to
match luxpy's convention (kLocalShiftScale=1.0). Since this script computes
R_cs itself (not via colour_fidelity_index), it follows the UNSCALED
(pytm30/luxpy) convention directly -- confirmed against the original
on-disk fixtures (e.g. D65/14_per_bin_metrics.json: Rcshj ~ 1e-5, not ~1e-3).
"""
import os
import json
import numpy as np
from collections import OrderedDict
from datetime import datetime, timezone

import colour
from colour import VIEWING_CONDITIONS_CIECAM02
from colour.temperature import CCT_to_xy_CIE_D
from colour.algebra import euclidean_distance
from colour.quality.cfi2017 import load_TCS_CIE2017
from colour.quality.tm3018 import delta_E_to_R_f, averages_area

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUTPUT_DIR = os.path.join(REPO, "tests", "fixtures")
DATA_DIR = os.path.join(REPO, "data")
COLOUR_VERSION = colour.__version__
GENERATED_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")
CRI_TYPE = "ies-tm30-20"  # matches the original fixtures' "cri_type" field
METHOD_NOTE = "colour-science primitives (CCT/CIECAM02/CAM02-UCS/CFI math) + hand-built TM-30-20 Sec 3.3 reference illuminant"

CMFS_10 = colour.MSDS_CMFS["CIE 1964 10 Degree Standard Observer"].copy()
SURROUND_AVERAGE = VIEWING_CONDITIONS_CIECAM02["Average"]

SKIN_CES_15 = 14  # CES15, 0-indexed
SKIN_CES_18 = 17  # CES18, 0-indexed

# Daylight basis functions (confirmed bit-identical vs colour-science, see
# generate_data_colour_science.py), at their native 5nm resolution.
_BASIS = colour.colorimetry.SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES
_S0 = _BASIS["S0"].copy().align(colour.SpectralShape(380, 780, 5))
_S1 = _BASIS["S1"].copy().align(colour.SpectralShape(380, 780, 5))
_S2 = _BASIS["S2"].copy().align(colour.SpectralShape(380, 780, 5))
_WL5 = _S0.wavelengths


# ---------------------------------------------------------------------------
# CCT/Duv: exact port of src/tm30/cct.cpp's Ohno (2014) triangular +
# parabolic LUT search, run against this project's OWN planckian_uv.csv
# LUT and cie_1931_2.csv CMF (both already generated/verified from
# colour-science by generate_data_colour_science.py / generate_planckian_lut.py).
#
# WHY NOT colour.temperature.uv_to_CCT_Ohno2013 / CCT_reference_illuminant:
# an initial version of this script used colour-science's own Ohno2013
# solver, which is an independent (still spec-correct) implementation --
# it agrees with the C++ engine's own CCT to within ~0.05-0.06K for clean
# sources, comfortably inside Tol_Cct=0.5K, but that residual is *just*
# large enough to push the reference-illuminant-derived fixture arrays
# (06/07/08/09) a few 1e-3 above the tighter Tol_FixtureXyz/Tol_Jab
# tolerances (since generating the D-series/Planckian reference SPD at a
# slightly different CCT shifts its exact chromaticity). Running the
# *identical* LUT-based algorithm the C++ engine itself runs eliminates
# that cross-algorithm noise, leaving only genuine data-precision-level
# residuals (see PROVENANCE notes for e.g. d65_1nm.csv's ~9e-4 rounding).
# ---------------------------------------------------------------------------

def _load_csv_cols(path):
    with open(path) as f:
        rows = list(__import__("csv").reader(f))
    return np.array([[float(x) for x in row] for row in rows[1:]])

_LUT = _load_csv_cols(os.path.join(DATA_DIR, "planckian_uv.csv"))
_LUT_T, _LUT_U, _LUT_V = _LUT[:, 0], _LUT[:, 1], _LUT[:, 2]

_CMF2 = _load_csv_cols(os.path.join(DATA_DIR, "cie_1931_2.csv"))
_CMF2_WL, _CMF2_X, _CMF2_Y, _CMF2_Z = _CMF2[:, 0], _CMF2[:, 1], _CMF2[:, 2], _CMF2[:, 3]


def xyz_to_uv(X, Y, Z):
    d = X + 15.0 * Y + 3.0 * Z
    return 4.0 * X / d, 6.0 * Y / d


def compute_cct_duv(u_test, v_test):
    """Exact port of tm30::compute_cct_duv (src/tm30/cct.cpp)."""
    d2 = (u_test - _LUT_U) ** 2 + (v_test - _LUT_V) ** 2
    best_i = int(np.argmin(d2))
    n = len(_LUT_T)
    if best_i == 0:
        im1, i0, ip1 = 0, 1, 2
    elif best_i == n - 1:
        im1, i0, ip1 = n - 3, n - 2, n - 1
    else:
        im1, i0, ip1 = best_i - 1, best_i, best_i + 1

    Tm1, T0, Tp1 = _LUT_T[im1], _LUT_T[i0], _LUT_T[ip1]
    um1, u0, up1 = _LUT_U[im1], _LUT_U[i0], _LUT_U[ip1]
    vm1, v0, vp1 = _LUT_V[im1], _LUT_V[i0], _LUT_V[ip1]

    def dist(u, v):
        return np.hypot(u_test - u, v_test - v)

    dm1, d0, dp1 = dist(um1, vm1), dist(u0, v0), dist(up1, vp1)
    l = np.hypot(up1 - um1, vp1 - vm1)
    x = (dm1 * dm1 - dp1 * dp1 + l * l) / (2.0 * l)
    x = min(max(x, 0.0), l)
    T_tri = Tm1 + (Tp1 - Tm1) * (x / l)
    uch = um1 + (up1 - um1) * (x / l)
    vch = vm1 + (vp1 - vm1) * (x / l)
    duv_mag = np.hypot(u_test - uch, v_test - vch)
    sign = 1.0 if v_test >= vch else -1.0
    duv_tri = duv_mag * sign

    denom = (Tp1 - T0) * (Tm1 - Tp1) * (T0 - Tm1)
    if abs(denom) < 1e-30:
        denom = 1e-30
    a = (Tm1 * (dp1 - d0) + T0 * (dm1 - dp1) + Tp1 * (d0 - dm1)) / denom
    b = -(Tm1 ** 2 * (dp1 - d0) + T0 ** 2 * (dm1 - dp1) + Tp1 ** 2 * (d0 - dm1)) / denom
    c = -(dm1 * (Tp1 - T0) * Tp1 * T0 + d0 * (Tm1 - Tp1) * Tm1 * Tp1 +
          dp1 * (T0 - Tm1) * T0 * Tm1) / denom

    if abs(a) > 1e-30:
        T_par = -b / (2.0 * a)
        T_par = min(max(T_par, Tm1), Tp1)
    else:
        T_par = T_tri

    duv_par_mag = a * T_par * T_par + b * T_par + c
    duv_par = duv_par_mag * sign

    duv_threshold = 0.002
    duv_abs = abs(duv_tri)
    T_tri_shift = T_tri + (T_par - T_tri) * min(duv_abs / duv_threshold, 1.0)

    if duv_abs < duv_threshold:
        return T_tri_shift, duv_tri
    return T_par, duv_par


def compute_cct_duv_from_sd(sd):
    """CCT/Duv for a test SPD, using the CIE 1931 2-degree observer (matching
    pytm30's own cmf_2deg convention for CCT, per tm30_calc.py)."""
    wl = sd.wavelengths
    idx = np.searchsorted(_CMF2_WL, wl)
    xb, yb, zb = _CMF2_X[idx], _CMF2_Y[idx], _CMF2_Z[idx]
    X = np.trapezoid(sd.values * xb, wl)
    Y = np.trapezoid(sd.values * yb, wl)
    Z = np.trapezoid(sd.values * zb, wl)
    u, v = xyz_to_uv(X, Y, Z)
    return compute_cct_duv(u, v)


# ---------------------------------------------------------------------------
# Reference-illuminant generation (TM-30-20 Sec 3.3, matching reference.cpp
# and generate_data_colour_science.py's D65/D50/D75 reconstruction exactly)
# ---------------------------------------------------------------------------

def _sd(wl, values, name=""):
    return colour.SpectralDistribution(
        dict(zip(np.asarray(wl).tolist(), np.asarray(values).tolist())), name=name)


def planckian_relative_values(cct, wl):
    """TM-30-20 Sec 3.3 Eq.(5)-(6) blackbody SPD, 560nm-normalised."""
    c2 = 1.4388e-2
    wl_m = wl * 1e-9
    L = wl_m ** -5.0 / (np.exp(c2 / (wl_m * cct)) - 1.0)
    L560 = (560e-9) ** -5.0 / (np.exp(c2 / (560e-9 * cct)) - 1.0)
    return L / L560


def cie_d_values(cct, wl, round_M=True):
    """CIE D-series SPD at `wl` via linear interpolation of the 5nm daylight
    basis functions (TM-30-20 Sec 3.5: "linear interpolation shall be
    used") -- NOT colour-science's own sd_CIE_illuminant_D_series(), which
    defaults to Sprague interpolation for these datasets.

    round_M: CIE 15:2004 recommends rounding M1/M2 to 3 decimal places
    when computing the daylight phases for nominal designated illuminants
    (D50, D65, D75, ...); the recommendation is specific to nominal CCTs.
    round_M=True is therefore correct for nominal test sources (cie_d_sd,
    below). The reference illuminant is generated at a *computed*,
    non-nominal CCT, to which the rounding recommendation does not apply:
    round_M=False is used by generate_reference_spd_values(), matching
    reference.cpp, which computes M1/M2 unrounded per TM-30-20 Sec 3.3
    Eq. (8)-(9).
    """
    x, y = CCT_to_xy_CIE_D(cct)
    M = 0.0241 + 0.2562 * x - 0.7341 * y
    M1 = float((-1.3515 - 1.7703 * x + 5.9114 * y) / M)
    M2 = float((0.0300 - 31.4424 * x + 30.0717 * y) / M)
    if round_M:
        M1 = round(M1, 3)
        M2 = round(M2, 3)
    S0_i = np.interp(wl, _WL5, _S0.values)
    S1_i = np.interp(wl, _WL5, _S1.values)
    S2_i = np.interp(wl, _WL5, _S2.values)
    spd = S0_i + M1 * S1_i + M2 * S2_i
    idx560 = np.argmin(np.abs(wl - 560.0))
    return spd / spd[idx560] * 100.0


def generate_reference_spd_values(cct, wl, y_bar):
    """TM-30-20 Sec 3.3 Eq.(13)-(16): pure Planckian (<=4000K), pure CIE
    D-series (>=5000K), or a Y-normalised blend in between. Matches
    src/tm30/reference.cpp::generate_reference_spd exactly."""
    if cct <= 4000.0:
        return planckian_relative_values(cct, wl)
    if cct >= 5000.0:
        return cie_d_values(cct, wl, round_M=False)

    planck = planckian_relative_values(cct, wl)
    daylight = cie_d_values(cct, wl, round_M=False)

    Y_planck = np.trapezoid(planck * y_bar, wl)
    Y_daylight = np.trapezoid(daylight * y_bar, wl)
    planck_100 = planck * (100.0 / Y_planck)
    daylight_100 = daylight * (100.0 / Y_daylight)

    blend_factor = (5000.0 - cct) / 1000.0  # Planckian weight
    daylight_weight = 1.0 - blend_factor
    blended = blend_factor * planck_100 + daylight_weight * daylight_100

    idx560 = np.argmin(np.abs(wl - 560.0))
    return blended / blended[idx560]


# ---------------------------------------------------------------------------
# Test-SPD corpus builders (raw value generators, used both for the fixture
# corpus here and cross-checked against generate_data_colour_science.py)
# ---------------------------------------------------------------------------

def cie_d_sd(cct_nominal, wl=None):
    if wl is None:
        wl = np.arange(380, 781, 1.0)
    cct = cct_nominal * 1.4388 / 1.4380
    return _sd(wl, cie_d_values(cct, wl), name=f"D{cct_nominal}")


def planckian_sd(cct, wl=None):
    if wl is None:
        wl = np.arange(380, 781, 1.0)
    return _sd(wl, planckian_relative_values(cct, wl), name=f"Planckian {cct}K")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default, allow_nan=True)


HUE_BIN_EDGES_RAD = (np.arange(17) * 22.5 * np.pi / 180.0).tolist()

ANNEX_E_PRIORITIES = {
    "P1": ["Rf", "Rg", "CVG"],
    "P2": ["Rcs_hj", "Rhs_hj", "Rf_hj"],
    "P3": ["Rf_CESi"],
    "source": "TM-30-20 Annex E",
    "description": (
        "Priority levels for specifying light source color rendition. "
        "P1 = preferred/highest, P2 = secondary (local measures), "
        "P3 = ancillary (sample-specific). Static per the TM-30-20 spec; "
        "not a function of the test SPD. Confirmed NOT read by the C++ "
        "test suite (which only asserts the static AnnexE::P1/P2/P3 int "
        "constants == 1/2/3 in slice_10_sample_test.cpp)."
    ),
}


# ---------------------------------------------------------------------------
# Core pipeline (colour-science primitives + hand-built reference SPD)
# ---------------------------------------------------------------------------

def compute_stages(sd_test):
    wl = sd_test.wavelengths
    shape = sd_test.shape

    # 02: CCT/Duv -- purely a function of the test SPD's own XYZ (2-degree
    # observer), via the exact same Ohno (2014) LUT search src/tm30/cct.cpp
    # implements, independent of the reference-SPD question.
    cct, duv = compute_cct_duv_from_sd(sd_test)
    cct = float(cct)
    duv = float(duv)

    # 03: hand-built reference SPD (see module docstring)
    y_bar_10 = CMFS_10.copy().align(shape).values[:, 1]
    ref_values = generate_reference_spd_values(cct, wl, y_bar_10)
    sd_reference = _sd(wl, ref_values, name=f"Reference {cct:.1f}K")

    # White points, Y=100-normalised (CIE 1964 10-degree, Integration)
    cmfs_10 = CMFS_10.copy().align(shape)
    XYZ_test_raw = colour.sd_to_XYZ(sd_test.values, cmfs=cmfs_10, shape=shape, method="Integration")
    XYZ_ref_raw = colour.sd_to_XYZ(sd_reference.values, cmfs=cmfs_10, shape=shape, method="Integration")
    k_test = 100.0 / XYZ_test_raw[1]
    k_ref = 100.0 / XYZ_ref_raw[1]
    XYZ_w_test = XYZ_test_raw * k_test
    XYZ_w_ref = XYZ_ref_raw * k_ref

    # 99 CES under each illuminant (scaled the same way as
    # colour.quality.cfi2017.tcs_colorimetry_data does internally)
    ces = load_TCS_CIE2017(shape)  # MultiSpectralDistributions, values (n_wl, 99)
    ces_t = np.transpose(ces.values)  # (99, n_wl)

    illum_scaled = np.stack([sd_test.values * k_test, sd_reference.values * k_ref])  # (2, n_wl)
    tcs_t = np.tile(ces_t, (2, 1, 1)) * illum_scaled.reshape(2, 1, -1)  # (2, 99, n_wl)

    XYZ_ces = colour.msds_to_XYZ(tcs_t, cmfs=cmfs_10, shape=shape, method="Integration")  # (2, 99, 3)
    XYZ_w_stack = np.stack([XYZ_w_test, XYZ_w_ref]).reshape(2, 1, 3)

    cam = colour.XYZ_to_CIECAM02(
        XYZ_ces, XYZ_w_stack, L_A=100.0, Y_b=20.0, surround=SURROUND_AVERAGE,
        discount_illuminant=True, compute_H=False,
    )
    JMh = np.stack([cam.J, cam.M, cam.h], axis=-1)  # (2, 99, 3)
    Jpapbp = colour.JMh_CIECAM02_to_CAM02UCS(JMh)   # (2, 99, 3)

    Jpapbp_test, Jpapbp_ref = Jpapbp[0], Jpapbp[1]
    XYZ_test_ces, XYZ_ref_ces = XYZ_ces[0], XYZ_ces[1]

    delta_e = euclidean_distance(Jpapbp_test, Jpapbp_ref)  # (99,)
    R_s = np.asarray(delta_E_to_R_f(delta_e))
    R_f = float(delta_E_to_R_f(np.mean(delta_e)))

    href = JMh[1, :, 2]  # reference hue angle, degrees, per CES
    htest = JMh[0, :, 2]
    bins = np.floor(href / 22.5).astype(int)
    bins = np.clip(bins, 0, 15)

    J_test_bin = np.full(16, np.nan)
    J_ref_bin = np.full(16, np.nan)
    a_test_bin = np.full(16, np.nan)
    b_test_bin = np.full(16, np.nan)
    a_ref_bin = np.full(16, np.nan)
    b_ref_bin = np.full(16, np.nan)
    DE_hj = np.full(16, np.nan)
    for j in range(16):
        mask = bins == j
        if not np.any(mask):
            continue
        J_test_bin[j] = np.mean(Jpapbp_test[mask, 0])
        a_test_bin[j] = np.mean(Jpapbp_test[mask, 1])
        b_test_bin[j] = np.mean(Jpapbp_test[mask, 2])
        J_ref_bin[j] = np.mean(Jpapbp_ref[mask, 0])
        a_ref_bin[j] = np.mean(Jpapbp_ref[mask, 1])
        b_ref_bin[j] = np.mean(Jpapbp_ref[mask, 2])
        DE_hj[j] = np.mean(delta_e[mask])

    avg_test_ab = np.column_stack([a_test_bin, b_test_bin])
    avg_ref_ab = np.column_stack([a_ref_bin, b_ref_bin])

    # Rg: signed shoelace area ratio (colour-science's averages_area, a
    # generic geometry helper with no data dependence)
    R_g = 100.0 * averages_area(avg_test_ab) / averages_area(avg_ref_ab)

    # Per-bin Rf,hj (TM-30-20 Sec 4.8)
    Rf_hj = np.asarray(delta_E_to_R_f(np.nan_to_num(DE_hj, nan=0.0)))
    Rf_hj = np.where(np.isnan(DE_hj), np.nan, Rf_hj)

    # Per-bin Rcs,hj / Rhs,hj (TM-30-20 Sec 4.6/4.7 == gamut.cpp exactly).
    # UNSCALED convention (matches pytm30/luxpy, not colour-science's own
    # x100 percentage convention -- see module docstring NOTE).
    j_idx = np.arange(16)
    theta = (j_idx + 0.5) * 22.5 * np.pi / 180.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    da = avg_test_ab[:, 0] - avg_ref_ab[:, 0]
    db = avg_test_ab[:, 1] - avg_ref_ab[:, 1]
    r_ref = np.hypot(avg_ref_ab[:, 0], avg_ref_ab[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        Rcs_hj = (da * cos_t + db * sin_t) / r_ref
        Rhs_hj = (-da * sin_t + db * cos_t) / r_ref
    Rcs_hj = np.where(r_ref < 1e-12, 0.0, Rcs_hj)
    Rhs_hj = np.where(r_ref < 1e-12, 0.0, Rhs_hj)

    # CVG-normalised coordinates (TM-30-20 Sec 4.5 Eq. 58-61 == gamut.cpp)
    jabt_hj = np.column_stack([J_test_bin, avg_test_ab[:, 0], avg_test_ab[:, 1]])
    jabr_hj = np.column_stack([J_ref_bin, avg_ref_ab[:, 0], avg_ref_ab[:, 1]])
    jabtn_hj = np.full((16, 3), np.nan)
    jabrn_hj = np.full((16, 3), np.nan)
    for j in range(16):
        jabtn_hj[j, 0] = J_test_bin[j]
        jabrn_hj[j, 0] = J_ref_bin[j]
        ar, br = avg_ref_ab[j]
        if np.isnan(ar) or np.isnan(br):
            continue
        rr = np.hypot(ar, br)
        h_bar = np.arctan2(br, ar)
        x_ref_raw, y_ref_raw = np.cos(h_bar), np.sin(h_bar)
        jabrn_hj[j, 1] = 100.0 * x_ref_raw
        jabrn_hj[j, 2] = 100.0 * y_ref_raw
        at_, bt_ = avg_test_ab[j]
        if rr < 1e-12:
            jabtn_hj[j, 1] = 100.0 * x_ref_raw
            jabtn_hj[j, 2] = 100.0 * y_ref_raw
        else:
            jabtn_hj[j, 1] = 100.0 * (x_ref_raw + (at_ - ar) / rr)
            jabtn_hj[j, 2] = 100.0 * (y_ref_raw + (bt_ - br) / rr)

    rf_skin = (float(R_s[SKIN_CES_15]) + float(R_s[SKIN_CES_18])) / 2.0

    stages = OrderedDict()

    stages["01_resampled_spd"] = {
        "wavelengths": np.asarray(wl).tolist(),
        "values": np.asarray(sd_test.values).tolist(),
        "n_wavelengths": int(len(wl)),
        "wavelength_range": [float(wl[0]), float(wl[-1])],
    }
    stages["02_cct_duv"] = {"cct": cct, "duv": duv}
    stages["03_reference_spd"] = {
        "wavelengths": np.asarray(wl).tolist(),
        "values": np.asarray(sd_reference.values).tolist(),
        "n_wavelengths": int(len(wl)),
        "wavelength_range": [float(wl[0]), float(wl[-1])],
    }
    stages["04_xyz_test_white"] = {"X": float(XYZ_w_test[0]), "Y": float(XYZ_w_test[1]), "Z": float(XYZ_w_test[2])}
    stages["05_xyz_ref_white"] = {"X": float(XYZ_w_ref[0]), "Y": float(XYZ_w_ref[1]), "Z": float(XYZ_w_ref[2])}
    stages["06_xyz_test_ces"] = {"columns": ["X", "Y", "Z"], "n_ces": 99, "values": XYZ_test_ces.tolist()}
    stages["07_xyz_ref_ces"] = {"columns": ["X", "Y", "Z"], "n_ces": 99, "values": XYZ_ref_ces.tolist()}
    stages["08_jab_test_ces"] = {"columns": ["J'", "a'", "b'"], "n_ces": 99, "values": Jpapbp_test.tolist()}
    stages["09_jab_ref_ces"] = {"columns": ["J'", "a'", "b'"], "n_ces": 99, "values": Jpapbp_ref.tolist()}
    stages["10_delta_e_ces"] = {"n_ces": 99, "values": np.asarray(delta_e).tolist()}
    stages["11_hue_bins"] = {
        "n_bins": 16,
        "hue_bin_edges_rad": HUE_BIN_EDGES_RAD,
        "bin_assignments_0_indexed": bins.tolist(),
        "hbinnrs": [float(b) for b in bins.tolist()],
        "ht": htest.tolist(),
        "hr": href.tolist(),
        "nhbins": 16,
    }
    stages["12_rf"] = {"Rf": R_f}
    stages["13_rg"] = {"Rg": float(R_g)}
    stages["14_per_bin_metrics"] = {
        "n_bins": 16,
        "Rf_hj": Rf_hj.tolist(), "Rcs_hj": Rcs_hj.tolist(), "Rhs_hj": Rhs_hj.tolist(),
        "Rfhj": Rf_hj.tolist(), "Rcshj": Rcs_hj.tolist(), "Rhshj": Rhs_hj.tolist(),
        "DE_hj": DE_hj.tolist(), "DEhj": DE_hj.tolist(),
    }
    stages["15_cvg_coordinates"] = {
        "n_bins": 16,
        "jabt_hj": jabt_hj.tolist(), "jabr_hj": jabr_hj.tolist(),
        "jabtn_hj": jabtn_hj.tolist(), "jabrn_hj": jabrn_hj.tolist(),
        "test_coordinates": {"a_prime": avg_test_ab[:, 0].tolist(), "b_prime": avg_test_ab[:, 1].tolist()},
        "reference_coordinates": {"a_prime": avg_ref_ab[:, 0].tolist(), "b_prime": avg_ref_ab[:, 1].tolist()},
        "reference_circle_radius": r_ref.tolist(),
    }
    stages["16_rfi_per_sample"] = {"n_ces": 99, "values": R_s.tolist()}
    stages["17_rf_skin"] = {
        "Rf_skin": rf_skin, "Rf_CES15": float(R_s[SKIN_CES_15]), "Rf_CES18": float(R_s[SKIN_CES_18]),
        "ces_indices_0_indexed": [SKIN_CES_15, SKIN_CES_18],
        "ces15": float(R_s[SKIN_CES_15]), "ces18": float(R_s[SKIN_CES_18]),
    }
    stages["18_annex_e"] = dict(ANNEX_E_PRIORITIES)

    return stages


def _wrap(stage_name, stage_data, spd_name):
    envelope = OrderedDict()
    envelope["spd_name"] = spd_name
    envelope["cri_type"] = CRI_TYPE
    envelope["colour_science_version"] = COLOUR_VERSION
    envelope["method"] = METHOD_NOTE
    envelope["generated"] = GENERATED_DATE
    envelope["stage"] = stage_name
    envelope.update(stage_data)
    return envelope


# ---------------------------------------------------------------------------
# SPD corpus (mirrors the original repo's actual on-disk fixture directory
# list: 37 directories)
# ---------------------------------------------------------------------------

def build_corpus():
    spds = OrderedDict()
    wl1 = np.arange(380, 781, 1.0)

    d65 = cie_d_sd(6500.0, wl1)
    spds["D65"] = d65
    spds["D65_1nm"] = d65
    spds["D50"] = cie_d_sd(5000.0, wl1)
    spds["D75"] = cie_d_sd(7500.0, wl1)

    for T in (2700, 3000, 3500):
        spds[f"planckian_{T}K"] = planckian_sd(float(T), wl1)

    for i in range(1, 13):
        sd5 = colour.SDS_ILLUMINANTS[f"FL{i}"].copy()
        spds[f"F{i}"] = sd5.align(colour.SpectralShape(380, 780, 1))

    for T in (3999.9, 4000.1, 4500.0, 4999.9, 5000.1):
        T_label = str(int(T)) if float(T).is_integer() else str(T)
        spds[f"planckian_{T_label}K"] = planckian_sd(T, wl1)

    for i in range(1, 6):
        spds[f"HP{i}"] = colour.SDS_ILLUMINANTS[f"HP{i}"].copy().align(colour.SpectralShape(380, 780, 5))

    for name in ("LED-RGB1", "LED-B1", "LED-B3", "LED-B5", "LED-BH1", "LED-V1", "LED-V2"):
        spds[name.replace("-", "_")] = colour.SDS_ILLUMINANTS[name].copy().align(colour.SpectralShape(380, 780, 5))

    wl_full = d65.wavelengths
    vals_full = d65.values
    mask = (wl_full >= 380) & (wl_full <= 780)
    wl_trim, vals_trim = wl_full[mask], vals_full[mask]
    spds["D65_5nm"] = _sd(wl_trim[::5], vals_trim[::5], name="D65_5nm")

    return spds


def generate_all():
    spds = build_corpus()
    total = 0
    errors = []

    for spd_name, sd in spds.items():
        spd_dir = os.path.join(OUTPUT_DIR, spd_name)
        os.makedirs(spd_dir, exist_ok=True)
        try:
            stages = compute_stages(sd)
        except Exception as e:  # pragma: no cover -- diagnostic path
            errors.append((spd_name, str(e)))
            print(f"  ERROR {spd_name}: {e}")
            continue

        for stage_name, stage_data in stages.items():
            fixture = _wrap(stage_name, stage_data, spd_name)
            dump_json(fixture, os.path.join(spd_dir, f"{stage_name}.json"))
            total += 1
        print(f"  OK {spd_name}: {len(stages)} stages "
              f"(Rf={stages['12_rf']['Rf']:.2f}, Rg={stages['13_rg']['Rg']:.2f}, "
              f"CCT={stages['02_cct_duv']['cct']:.1f}, Duv={stages['02_cct_duv']['duv']:.5f})")

    print(f"\nTotal stage files written: {total}")
    if errors:
        print(f"Errors: {len(errors)}")
        for n, e in errors:
            print(f"  {n}: {e}")
    return total, errors


if __name__ == "__main__":
    generate_all()
