#!/usr/bin/env python3
"""TM30Calc - cached-table TM-30 calculator for repeated evaluation.

Wraps tm30_core.BatchContext to load reference tables once and evaluate
many SPDs without re-I/O.  The constructor also binds a fixed wavelength
grid (`wavelengths=`, default 380-780 nm @ 1 nm) and pre-resamples the CES/
CMF/daylight-basis tables to it once, so eval() calls for SPDs sharing that
grid -- the common case -- skip resampling entirely.

Usage:
    import numpy as np
    from tm30_calc import TM30Calc

    calc = TM30Calc()
    result = calc.eval(spd)          # single SPD (1-D array) -> Tm30Result
    print(result.rf, result.rg)      # scalars

    result = calc.eval(spd_matrix)   # batch (2-D array) -> Tm30BatchResult
    print(result.rf)                 # shape (N,) -- one value per input row
    print(result.rf[3], result.rg[3])  # index into it like any array
    print(result.valid)              # shape (N,) bool -- which rows succeeded

DataFrame support:
    df = result.to_dataframe()       # single -> 1-row DataFrame
    df = batch_result.to_dataframe() # batch  -> N-row DataFrame

    # Multi-level columns, backwards-compatible with the luxpy
    # tm30_dict_to_dataframe() convention (lx_util.py): per-CES arrays get
    # a ('field', 'CES_01') .. ('field', 'CES_99') second column level,
    # per-bin arrays get ('field', 'bin_01') .. ('field', 'bin_16'), and
    # per-CES/bin [X,Y,Z] or [J,a,b] triples get a third level.
    df = batch_result.to_dataframe(multiindex=True)
    df['rf_cesi']['CES_05']          # per-CES fidelity, sample 5
    df['xyz_test_ces']['CES_05']['X']  # per-CES XYZ (extras=True), sample 5

Extras (opt-in, extras=True on eval()):
    Additional per-bin/per-CES fields not included by default because they
    add ~1100+ floats/SPD - rf_hj, de_hj (16 each), CVG coordinates
    (cvg_j_test/x_test/y_test/j_ref/x_ref/y_ref, 16 each), reference_spd
    (resampled reference-illuminant SPD), and xyz_test_ces/xyz_ref_ces
    (99x3 per-CES XYZ under test/reference illuminant).
"""

from __future__ import annotations

import numpy as np
import tm30_core
from enum import Enum
import os


# -- CMF Observer Enum --------------------------------------------------


class Cmf(Enum):
    """CIE Color Matching Function observer.

    String values are accepted interchangeably (case-insensitive).
    Examples: Cmf.CIE_1964_10, Cmf.CIE_2015_2, Cmf('1931_2'), Cmf('2015_10').

    -- Standard observers --
    CIE_1931_2      - CIE 1931 2-deg (luxpy default)
    CIE_1964_10     - CIE 1964 10-deg (TM-30-20 standard, pytm30 default)
    CIE_2006_2      - CIE 2006 2-deg
    CIE_2006_10     - CIE 2006 10-deg
    CIE_2015_2      - CIE 2015 2-deg
    CIE_2015_10     - CIE 2015 10-deg

    -- Usage --
        calc = TM30Calc(cmf=Cmf.CIE_1964_10)
        calc = TM30Calc(cmf='1931_2')         # string also works
        calc = TM30Calc(cmf='cmf_1931_2.csv') # custom CSV path
    """

    CIE_1931_2 = "1931_2"
    CIE_1964_10 = "1964_10"
    CIE_2006_2 = "2006_2"
    CIE_2006_10 = "2006_10"
    CIE_2015_2 = "2015_2"
    CIE_2015_10 = "2015_10"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value):
        """Allow case-insensitive string lookup."""
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


# -- Internal: resolve cmf to CSV path ---------------------------------


def _resolve_cmf(
    cmf: Cmf | str | os.PathLike | None, data_dir: str, suffix: str = ""
) -> str:
    """Resolve a CMF specifier to a CSV file path.

    Parameters
    ----------
    cmf : Cmf, str, Path, or None
        - Cmf enum member -> 'data/cmf_1931_2.csv'
        - str matching an enum value -> same
        - str containing '/' or '.csv' -> treated as file path
        - None -> defaults to 'data/cmf_1964_10.csv' (for 10-deg) or
                 'data/cie_1931_2.csv' (for 2-deg, if suffix='2deg')
    data_dir : str
        Path to the data directory.
    suffix : str
        '2deg' -> default cmf_2deg path. Empty -> default cmf_10deg path.

    Returns
    -------
    str - absolute path to the CSV file.
    """
    if cmf is None:
        if suffix == "2deg":
            path = os.path.join(data_dir, "cie_1931_2.csv")
        else:
            path = os.path.join(data_dir, "cmf_1964_10.csv")
        return path

    # Handle Cmf enum or string matching an enum value
    if isinstance(cmf, Cmf):
        key = cmf.value
    elif isinstance(cmf, str):
        # Try as enum key first
        try:
            key = Cmf(cmf).value
        except (ValueError, KeyError):
            # Treat as path: if it looks like a file path, resolve relative to CWD
            if "/" in cmf or ".csv" in cmf:
                return cmf if os.path.isabs(cmf) else os.path.abspath(cmf)
            raise ValueError(
                f"Unknown CMF: '{cmf}'. Valid values: {', '.join(m.value for m in Cmf)}"
            )
    else:
        return str(cmf)

    return os.path.join(data_dir, f"cmf_{key}.csv")


# -- Public re-exports --------------------------------------------------

__all__ = ["TM30Calc", "Tm30Result", "Tm30BatchResult", "to_dataframe", "Cmf"]


# -- Multi-level column support ------------------------------------------
#
# Backwards-compatible with the author's internal lx_util.py wrapper.
# Mirrors the column-naming convention of its tm30_dict_to_dataframe():
# per-CES arrays (99 values/row) -> 'CES_01'..'CES_99', per-bin arrays (16
# values/row) -> 'bin_01'..'bin_16', per-CES/bin [X,Y,Z]/[J,a,b] triples get
# a third column level, and scalar fields are single-level (padded with ''
# to match the deepest column's tuple length, same as its padded_cols).
#
# Top-level column names follow luxpy's spd_to_tm30() dict vocabulary so
# existing downstream code keeps working -- e.g. `df['Rcshj']` works
# exactly like it did against a luxpy-dict-derived DataFrame. Fields with
# no direct top-level equivalent in that vocabulary keep pytm30's own
# name -- see the comment below.
_COMPAT_KEY_ALIASES: dict[str, str] = {
    "rf": "Rf",
    "rg": "Rg",
    # cct, duv: luxpy uses the same lowercase names already -- no alias needed.
    "delta_e_avg": "DEa",
    "rf_cesi": "Rfi",
    "rcs_hj": "Rcshj",
    "rhs_hj": "Rhshj",
    "rf_hj": "Rfhj",
    "de_hj": "DEhj",
    "reference_spd": "Sr",
    "xyz_test_ces": "xyzt",
    "xyz_ref_ces": "xyzr",
    "jab_test_ces": "jabt",
    "jab_ref_ces": "jabr",
    # rf_skin: no luxpy equivalent -- not part of luxpy's spd_to_tm30() dict.
    # cvg_j/x/y_test, cvg_j/x/y_ref, hue_bin_index: luxpy has no top-level
    # dict key for these either -- the closest luxpy data lives nested
    # inside 'hue_bin_data' (e.g. 'jabtn_hj' + a division by
    # 'normalized_chroma_ref' for the CVG plot coordinates, 'hbinnrs' for
    # the bin index), computed differently enough that there's no 1:1
    # top-level key to alias to. These keep pytm30's own descriptive names.
}


def _field_column_tuples(
    key: str,
    per_row_shape: tuple[int, ...],
    wavelengths: np.ndarray | None,
) -> list[tuple[str, ...]]:
    """Build luxpy-style column-name tuples for one field."""
    label = _COMPAT_KEY_ALIASES.get(key, key)

    if per_row_shape == ():
        return [(label,)]

    if len(per_row_shape) == 1:
        n = per_row_shape[0]
        if key == "reference_spd":
            if wavelengths is not None and len(wavelengths) == n:
                return [(label, f"wvl_{w:g}") for w in wavelengths]
            return [(label, f"wvl_{i}") for i in range(n)]
        prefix = "CES" if n == 99 else "bin" if n == 16 else "v"
        return [(label, f"{prefix}_{i + 1:02d}") for i in range(n)]

    if len(per_row_shape) == 2:
        n, width = per_row_shape
        prefix = "CES" if n == 99 else "bin" if n == 16 else "row"
        key_lower = key.lower()
        if "xyz" in key_lower:
            coords = ["X", "Y", "Z"]
        elif "jab" in key_lower:
            coords = ["J", "a", "b"]
        else:
            coords = [str(c) for c in range(width)]
        return [(label, f"{prefix}_{i + 1:02d}", c) for i in range(n) for c in coords]

    raise ValueError(
        f"Cannot build MultiIndex columns for field '{key}' with "
        f"per-row shape {per_row_shape}"
    )


def _multiindex_dataframe(
    fields: dict[str, np.ndarray],
    n_rows: int,
    wavelengths: np.ndarray | None,
):
    """Stack `fields` (each shaped (n_rows, ...)) into one luxpy-style
    MultiIndex-columned DataFrame, matching lx_util.py's
    tm30_dict_to_dataframe() final hstack + pd.MultiIndex.from_tuples()."""
    import pandas as pd

    col_tuples: list[tuple] = []
    data_blocks: list[np.ndarray] = []
    for key, value in fields.items():
        value = np.asarray(value)
        per_row_shape = value.shape[1:]
        col_tuples.extend(_field_column_tuples(key, per_row_shape, wavelengths))
        data_blocks.append(value.reshape(n_rows, -1))

    if not col_tuples:
        return pd.DataFrame(index=range(n_rows))

    max_depth = max(len(t) for t in col_tuples)
    padded = [t + ("",) * (max_depth - len(t)) for t in col_tuples]
    final_data = np.hstack(data_blocks)
    return pd.DataFrame(final_data, columns=pd.MultiIndex.from_tuples(padded))


class Tm30Result:
    """Lightweight attribute-access wrapper around a batch-evaluate dict."""

    __slots__ = ("_d", "_wavelengths")

    # -- Scalar property names in display order --
    _SCALAR_KEYS = ("rf", "rg", "cct", "duv", "delta_e_avg", "rf_skin")
    # samples=True-only / bins=True-only array keys; absent otherwise.
    _SAMPLES_ARRAY_KEYS = ("rf_cesi",)
    _BINS_ARRAY_KEYS = ("rcs_hj", "rhs_hj")
    # extras=True-only array keys (see eval(extras=...)); absent otherwise.
    _EXTRAS_ARRAY_KEYS = (
        "rf_hj",
        "de_hj",
        "cvg_j_test",
        "cvg_x_test",
        "cvg_y_test",
        "cvg_j_ref",
        "cvg_x_ref",
        "cvg_y_ref",
        "reference_spd",
        "xyz_test_ces",
        "xyz_ref_ces",
        "jab_test_ces",
        "jab_ref_ces",
        "hue_bin_index",
    )

    def __init__(self, d: dict, wavelengths: np.ndarray | None = None) -> None:
        self._d = d
        self._wavelengths = wavelengths

    def _present_keys(self) -> tuple[str, ...]:
        """All field names actually present on this result (scalars, the
        default arrays, and any extras=True fields)."""
        return (
            self._SCALAR_KEYS
            + tuple(k for k in self._SAMPLES_ARRAY_KEYS if k in self._d)
            + tuple(k for k in self._BINS_ARRAY_KEYS if k in self._d)
            + tuple(k for k in self._EXTRAS_ARRAY_KEYS if k in self._d)
        )

    @property
    def rf(self) -> float:
        return float(self._d["rf"])

    @property
    def rg(self) -> float:
        return float(self._d["rg"])

    @property
    def cct(self) -> float:
        return float(self._d["cct"])

    @property
    def duv(self) -> float:
        return float(self._d["duv"])

    @property
    def delta_e_avg(self) -> float:
        return float(self._d["delta_e_avg"])

    @property
    def rf_skin(self) -> float:
        return float(self._d["rf_skin"])

    @property
    def rf_cesi(self) -> np.ndarray:
        if "rf_cesi" not in self._d:
            raise AttributeError(
                "rf_cesi is not available on this result -- call "
                "eval(..., samples=True) to include it (samples defaults "
                "to False)."
            )
        return np.asarray(self._d["rf_cesi"])

    @property
    def rcs_hj(self) -> np.ndarray:
        if "rcs_hj" not in self._d:
            raise AttributeError(
                "rcs_hj is not available on this result -- call "
                "eval(..., bins=True) to include it (bins defaults "
                "to True; this result came from an explicit bins=False)."
            )
        return np.asarray(self._d["rcs_hj"])

    @property
    def rhs_hj(self) -> np.ndarray:
        if "rhs_hj" not in self._d:
            raise AttributeError(
                "rhs_hj is not available on this result -- call "
                "eval(..., bins=True) to include it (bins defaults "
                "to True; this result came from an explicit bins=False)."
            )
        return np.asarray(self._d["rhs_hj"])

    # -- Extras (only present if eval(..., extras=True)) --

    @property
    def rf_hj(self) -> np.ndarray:
        """Per-bin local fidelity Rf,hj - 16 values.  TM-30-20 S4.8."""
        return np.asarray(self._d["rf_hj"])

    @property
    def de_hj(self) -> np.ndarray:
        """Per-bin mean dE', DE_hj - 16 values.  TM-30-20 S4.8."""
        return np.asarray(self._d["de_hj"])

    @property
    def cvg_j_test(self) -> np.ndarray:
        """Test-vector J' - 16 values.  pytm30 extension of the S4.4 bin
        averages (S4.4 explicitly discards J'; S4.5 CVG geometry is 2-D in
        (a',b') only, per Eqs. 58-61)."""
        return np.asarray(self._d["cvg_j_test"])

    @property
    def cvg_x_test(self) -> np.ndarray:
        """CVG test-vector x - 16 values.  TM-30-20 S4.5 Eq. (60)."""
        return np.asarray(self._d["cvg_x_test"])

    @property
    def cvg_y_test(self) -> np.ndarray:
        """CVG test-vector y - 16 values.  TM-30-20 S4.5 Eq. (61)."""
        return np.asarray(self._d["cvg_y_test"])

    @property
    def cvg_j_ref(self) -> np.ndarray:
        """Reference-circle J' - 16 values.  pytm30 extension of the S4.4
        bin averages (S4.4 discards J'; S4.5 CVG geometry is 2-D in (a',b')
        only, per Eqs. 58-61)."""
        return np.asarray(self._d["cvg_j_ref"])

    @property
    def cvg_x_ref(self) -> np.ndarray:
        """CVG reference-circle x - 16 values.  TM-30-20 S4.5 Eq. (58)."""
        return np.asarray(self._d["cvg_x_ref"])

    @property
    def cvg_y_ref(self) -> np.ndarray:
        """CVG reference-circle y - 16 values.  TM-30-20 S4.5 Eq. (59)."""
        return np.asarray(self._d["cvg_y_ref"])

    @property
    def reference_spd(self) -> np.ndarray:
        """Reference-illuminant SPD, resampled to the input wavelength grid.

        TM-30-20 S3.3 Eq. (13)-(16).
        """
        return np.asarray(self._d["reference_spd"])

    @property
    def xyz_test_ces(self) -> np.ndarray:
        """Per-CES XYZ under the test source - shape (99, 3).

        TM-30-20 S3.6 Eq. (21)-(23).
        """
        return np.asarray(self._d["xyz_test_ces"])

    @property
    def xyz_ref_ces(self) -> np.ndarray:
        """Per-CES XYZ under the reference illuminant - shape (99, 3).

        TM-30-20 S3.6 Eq. (25)-(27).
        """
        return np.asarray(self._d["xyz_ref_ces"])

    @property
    def jab_test_ces(self) -> np.ndarray:
        """Per-CES CAM02-UCS [J', a', b'] under the test source - shape (99, 3).

        TM-30-20 S3.7.1 Eq. (48)-(50).
        """
        return np.asarray(self._d["jab_test_ces"])

    @property
    def jab_ref_ces(self) -> np.ndarray:
        """Per-CES CAM02-UCS [J', a', b'] under the reference illuminant - shape (99, 3).

        TM-30-20 S3.7.1 Eq. (48)-(50).
        """
        return np.asarray(self._d["jab_ref_ces"])

    @property
    def hue_bin_index(self) -> np.ndarray:
        """Per-CES hue-angle bin index (0-15), 99 values.

        Assigned from the reference hue angle hr = atan2(b'r, a'r); the same
        assignment is reused for both the test and reference bin averages.
        TM-30-20 S4.3.
        """
        return np.asarray(self._d["hue_bin_index"])

    def __repr__(self) -> str:
        return (
            f"Tm30Result(Rf={self.rf:.1f}, Rg={self.rg:.1f}, "
            f"CCT={self.cct:.0f} K, Duv={self.duv:.6f})"
        )

    # -- DataFrame / dict support ----------------------------------

    def to_dict(self, *, arrays: bool = False) -> dict[str, float | list[float]]:
        """Return a flat dict of this result.

        Parameters
        ----------
        arrays : bool
            If True, include rf_cesi, rcs_hj, rhs_hj as list columns.
            Default False - scalars only.
        """
        d = {k: getattr(self, k) for k in self._SCALAR_KEYS}
        if arrays:
            d["rf_cesi"] = list(self.rf_cesi)
            d["rcs_hj"] = list(self.rcs_hj)
            d["rhs_hj"] = list(self.rhs_hj)
        return d

    def to_dataframe(self, *, arrays: bool = False, multiindex: bool = False):
        """Return a 1-row pandas DataFrame.

        Parameters
        ----------
        arrays : bool
            If True, include array columns as lists.  Default False.
            Ignored if multiindex=True.
        multiindex : bool
            If True, return a DataFrame with a multi-level pd.MultiIndex
            for its columns, backwards-compatible with lx_util.py's
            tm30_dict_to_dataframe() convention: per-CES arrays (rf_cesi,
            hue_bin_index, ...) get a ('field', 'CES_01') .. ('field',
            'CES_99') second level, per-bin arrays (rcs_hj, rhs_hj, rf_hj,
            de_hj, cvg_*, ...) get ('field', 'bin_01') .. ('field',
            'bin_16'), per-CES/bin [X,Y,Z] or [J,a,b] triples (xyz_*_ces,
            jab_*_ces) get a third level, and scalar fields are single-
            level (padded with '' to match the deepest column).  Includes
            every field present on this result -- scalars, rf_cesi,
            rcs_hj, rhs_hj, and any extras=True fields.  Default False.
        """
        import pandas as pd

        if multiindex:
            fields = {}
            for k in self._present_keys():
                arr = np.asarray(getattr(self, k))
                fields[k] = arr.reshape(1, *arr.shape)
            return _multiindex_dataframe(
                fields, n_rows=1, wavelengths=self._wavelengths
            )
        return pd.DataFrame([self.to_dict(arrays=arrays)])


class Tm30BatchResult:
    """Structure-of-arrays result for calc.eval() on a batch (2-D) input.

    Every field is a numpy array with the batch as its leading axis:
    `.rf` has shape (N,), `.rf_cesi` has shape (N, 99), `.xyz_test_ces`
    has shape (N, 99, 3), and so on - directly usable for downstream
    numerical work without a list comprehension over per-SPD objects.

    Rows where the input SPD failed validation are NaN across every field
    for that row; `.valid` (shape (N,), bool) tells you which rows those
    are, and always has length N regardless of how many rows failed - so
    `result.rf[i]` always corresponds to input row i. This is different
    from a legitimate per-value NaN (e.g. an empty hue bin for an
    otherwise-valid SPD): check `.valid` first if you need to distinguish
    "no result at all" from "a valid result with some NaNs already in it".
    """

    __slots__ = ("_d", "_n", "valid", "_wavelengths")

    _SCALAR_KEYS = ("rf", "rg", "cct", "duv", "delta_e_avg", "rf_skin")
    _ARRAY_SHAPES = {
        "rf_cesi": (99,),
        "rcs_hj": (16,),
        "rhs_hj": (16,),
        "rf_hj": (16,),
        "de_hj": (16,),
        "cvg_j_test": (16,),
        "cvg_x_test": (16,),
        "cvg_y_test": (16,),
        "cvg_j_ref": (16,),
        "cvg_x_ref": (16,),
        "cvg_y_ref": (16,),
        "xyz_test_ces": (99, 3),
        "xyz_ref_ces": (99, 3),
        "jab_test_ces": (99, 3),
        "jab_ref_ces": (99, 3),
        "hue_bin_index": (99,),
    }

    def __init__(self, raw: list, wavelengths: np.ndarray) -> None:
        n = len(raw)
        self._n = n
        self._wavelengths = wavelengths
        n_wavelengths = len(wavelengths)
        self.valid = np.array([d is not None for d in raw], dtype=bool)

        # Which optional keys are actually present (depends on extras=...);
        # every successful row has the same key set, so any one will do.
        present_keys = next((d.keys() for d in raw if d is not None), ())

        self._d = {}
        for key in self._SCALAR_KEYS:
            arr = np.full(n, np.nan)
            for i, d in enumerate(raw):
                if d is not None:
                    arr[i] = d[key]
            self._d[key] = arr

        for key, shape in self._ARRAY_SHAPES.items():
            if key not in present_keys:
                continue
            arr = np.full((n,) + shape, np.nan)
            for i, d in enumerate(raw):
                if d is not None:
                    arr[i] = d[key]
            self._d[key] = arr

        if "reference_spd" in present_keys:
            arr = np.full((n, n_wavelengths), np.nan)
            for i, d in enumerate(raw):
                if d is not None:
                    arr[i] = d["reference_spd"]
            self._d["reference_spd"] = arr

    def __len__(self) -> int:
        return self._n

    def _present_keys(self) -> tuple[str, ...]:
        """All field names actually present on this result (scalars, the
        default arrays, and any extras=True fields).

        `reference_spd` is handled separately here since its per-row shape
        (n_wavelengths,) is variable and isn't in `_ARRAY_SHAPES`.
        """
        keys = self._SCALAR_KEYS + tuple(k for k in self._ARRAY_SHAPES if k in self._d)
        if "reference_spd" in self._d:
            keys = keys + ("reference_spd",)
        return keys

    @property
    def rf(self) -> np.ndarray:
        """Fidelity index Rf - shape (N,).  TM-30-20 S4.1."""
        return self._d["rf"]

    @property
    def rg(self) -> np.ndarray:
        """Gamut index Rg - shape (N,).  TM-30-20 S4.4."""
        return self._d["rg"]

    @property
    def cct(self) -> np.ndarray:
        """Correlated Color Temperature (K) - shape (N,).  TM-30-20 S3.3."""
        return self._d["cct"]

    @property
    def duv(self) -> np.ndarray:
        """Distance from Planckian locus - shape (N,).  TM-30-20 S3.3."""
        return self._d["duv"]

    @property
    def delta_e_avg(self) -> np.ndarray:
        """Average dE' across 99 CES - shape (N,).  TM-30-20 S4.1."""
        return self._d["delta_e_avg"]

    @property
    def rf_skin(self) -> np.ndarray:
        """Skin fidelity Rf,skin - shape (N,).

        PyTM30 research extension informed by TM-30-20 S4.2 (mean of the
        CES15 and CES18 fidelity values); not a standardised TM-30
        measure (S1.2, S4.0).
        """
        return self._d["rf_skin"]

    @property
    def rf_cesi(self) -> np.ndarray:
        """Per-sample fidelity Rf,CESi - shape (N, 99).  TM-30-20 S4.2."""
        if "rf_cesi" not in self._d:
            raise AttributeError(
                "rf_cesi is not available on this result -- call "
                "eval(..., samples=True) to include it (samples defaults "
                "to False)."
            )
        return self._d["rf_cesi"]

    @property
    def rcs_hj(self) -> np.ndarray:
        """Per-bin chroma shift Rcs,hj, in percent - shape (N, 16).

        TM-30-20 S4.6: Eq. (62) computes a ratio; S4.6 requires percentage
        representation, applied once in the core library.
        """
        if "rcs_hj" not in self._d:
            raise AttributeError(
                "rcs_hj is not available on this result -- call "
                "eval(..., bins=True) to include it (bins defaults "
                "to True; this result came from an explicit bins=False)."
            )
        return self._d["rcs_hj"]

    @property
    def rhs_hj(self) -> np.ndarray:
        """Per-bin hue shift Rhs,hj, dimensionless ratio - shape (N, 16).

        TM-30-20 S4.7: Eq. (63) is ratio-valued and S4.7 states no
        percentage requirement; reported unscaled.
        """
        if "rhs_hj" not in self._d:
            raise AttributeError(
                "rhs_hj is not available on this result -- call "
                "eval(..., bins=True) to include it (bins defaults "
                "to True; this result came from an explicit bins=False)."
            )
        return self._d["rhs_hj"]

    # -- Extras (only present if eval(..., extras=True)) --

    @property
    def rf_hj(self) -> np.ndarray:
        """Per-bin local fidelity Rf,hj - shape (N, 16).  TM-30-20 S4.8."""
        return self._d["rf_hj"]

    @property
    def de_hj(self) -> np.ndarray:
        """Per-bin mean dE', DE_hj - shape (N, 16).  TM-30-20 S4.8."""
        return self._d["de_hj"]

    @property
    def cvg_j_test(self) -> np.ndarray:
        """Test-vector J' - shape (N, 16).  pytm30 extension of the S4.4
        bin averages (S4.4 discards J'; S4.5 CVG geometry is 2-D in (a',b')
        only, per Eqs. 58-61)."""
        return self._d["cvg_j_test"]

    @property
    def cvg_x_test(self) -> np.ndarray:
        """CVG test-vector x - shape (N, 16).  TM-30-20 S4.5 Eq. (60)."""
        return self._d["cvg_x_test"]

    @property
    def cvg_y_test(self) -> np.ndarray:
        """CVG test-vector y - shape (N, 16).  TM-30-20 S4.5 Eq. (61)."""
        return self._d["cvg_y_test"]

    @property
    def cvg_j_ref(self) -> np.ndarray:
        """Reference-circle J' - shape (N, 16).  pytm30 extension of the
        S4.4 bin averages (S4.4 discards J'; S4.5 CVG geometry is 2-D in
        (a',b') only, per Eqs. 58-61)."""
        return self._d["cvg_j_ref"]

    @property
    def cvg_x_ref(self) -> np.ndarray:
        """CVG reference-circle x - shape (N, 16).  TM-30-20 S4.5 Eq. (58)."""
        return self._d["cvg_x_ref"]

    @property
    def cvg_y_ref(self) -> np.ndarray:
        """CVG reference-circle y - shape (N, 16).  TM-30-20 S4.5 Eq. (59)."""
        return self._d["cvg_y_ref"]

    @property
    def reference_spd(self) -> np.ndarray:
        """Reference-illuminant SPD, resampled to the input wavelength grid.

        Shape (N, N_wl).  TM-30-20 S3.3 Eq. (13)-(16).
        """
        return self._d["reference_spd"]

    @property
    def xyz_test_ces(self) -> np.ndarray:
        """Per-CES XYZ under the test source - shape (N, 99, 3).

        TM-30-20 S3.6 Eq. (21)-(23).
        """
        return self._d["xyz_test_ces"]

    @property
    def xyz_ref_ces(self) -> np.ndarray:
        """Per-CES XYZ under the reference illuminant - shape (N, 99, 3).

        TM-30-20 S3.6 Eq. (25)-(27).
        """
        return self._d["xyz_ref_ces"]

    @property
    def jab_test_ces(self) -> np.ndarray:
        """Per-CES CAM02-UCS [J', a', b'] under the test source - shape (N, 99, 3).

        TM-30-20 S3.7.1 Eq. (48)-(50).
        """
        return self._d["jab_test_ces"]

    @property
    def jab_ref_ces(self) -> np.ndarray:
        """Per-CES CAM02-UCS [J', a', b'] under the reference illuminant - shape (N, 99, 3).

        TM-30-20 S3.7.1 Eq. (48)-(50).
        """
        return self._d["jab_ref_ces"]

    @property
    def hue_bin_index(self) -> np.ndarray:
        """Per-CES hue-angle bin index (0-15) - shape (N, 99).

        TM-30-20 S4.3.
        """
        return self._d["hue_bin_index"]

    def __repr__(self) -> str:
        n_valid = int(self.valid.sum())
        return (
            f"Tm30BatchResult(n={self._n}, valid={n_valid}/{self._n}, "
            f"Rf mean={np.nanmean(self.rf):.1f}, Rg mean={np.nanmean(self.rg):.1f})"
        )

    def to_dataframe(
        self,
        *,
        arrays: bool = False,
        expand_arrays: bool = False,
        multiindex: bool = False,
    ):
        """Return an N-row pandas DataFrame - one row per input SPD.

        Parameters
        ----------
        arrays : bool
            If True, include rf_cesi, rcs_hj, rhs_hj as list-of-float columns.
            Ignored if multiindex=True.
        expand_arrays : bool
            If True, expand rf_cesi into 99 individual columns
            (rf_cesi_1 ... rf_cesi_99).  Implies arrays=True.  Ignored if
            multiindex=True.
        multiindex : bool
            If True, return a DataFrame with a multi-level pd.MultiIndex
            for its columns, backwards-compatible with lx_util.py's
            tm30_dict_to_dataframe() convention: per-CES arrays (rf_cesi,
            hue_bin_index, ...) get a ('field', 'CES_01') .. ('field',
            'CES_99') second level, per-bin arrays (rcs_hj, rhs_hj, rf_hj,
            de_hj, cvg_*, ...) get ('field', 'bin_01') .. ('field',
            'bin_16'), per-CES/bin [X,Y,Z] or [J,a,b] triples (xyz_*_ces,
            jab_*_ces) get a third level, and scalar fields are single-
            level (padded with '' to match the deepest column).  Includes
            every field present on this result -- scalars, valid,
            rf_cesi, rcs_hj, rhs_hj, and any extras=True fields.
            Default False.
        """
        import pandas as pd

        if multiindex:
            fields = {k: self._d[k] for k in self._present_keys()}
            fields["valid"] = self.valid
            return _multiindex_dataframe(
                fields, n_rows=self._n, wavelengths=self._wavelengths
            )

        data: dict = {k: self._d[k] for k in self._SCALAR_KEYS}
        data["valid"] = self.valid

        if arrays or expand_arrays:
            data["rcs_hj"] = list(self.rcs_hj)
            data["rhs_hj"] = list(self.rhs_hj)
        if expand_arrays:
            for i in range(99):
                data[f"rf_cesi_{i + 1}"] = self.rf_cesi[:, i]
        elif arrays:
            data["rf_cesi"] = list(self.rf_cesi)

        return pd.DataFrame(data)


def to_dataframe(
    results: list[Tm30Result],
    *,
    arrays: bool = False,
    expand_arrays: bool = False,
    multiindex: bool = False,
):
    """Convert a list of individually-collected Tm30Result objects to a
    pandas DataFrame - e.g. if you called calc.eval() once per SPD yourself
    and gathered the single-SPD results into your own list.

    calc.eval(spd_matrix) (2-D input) returns a Tm30BatchResult instead,
    which has its own .to_dataframe() method - use that one for actual
    batch calls; this free function is for a list you assembled yourself.

    Uses numpy arrays internally for speed - ~1 us/SPD overhead.

    Parameters
    ----------
    results : list[Tm30Result]
        Individually-collected single-SPD results.
    arrays : bool
        If True, include rf_cesi, rcs_hj, rhs_hj as list-of-float columns.
        Ignored if multiindex=True.
    expand_arrays : bool
        If True, expand rf_cesi into 99 individual columns
        (rf_cesi_1 ... rf_cesi_99).  Implies arrays=True.  Ignored if
        multiindex=True.
    multiindex : bool
        If True, return a DataFrame with a multi-level pd.MultiIndex for
        its columns, backwards-compatible with lx_util.py's
        tm30_dict_to_dataframe() convention -- see Tm30BatchResult.to_dataframe()
        for the exact layout.  Assumes every result has the same fields
        present (i.e. all were produced with the same extras= setting) and
        the same wavelength grid; both are taken from results[0].
    """
    import pandas as pd

    n = len(results)
    if n == 0:
        return pd.DataFrame()

    if multiindex:
        keys = results[0]._present_keys()
        fields = {
            key: np.array([getattr(r, key) for r in results], dtype=np.float64)
            for key in keys
        }
        return _multiindex_dataframe(
            fields, n_rows=n, wavelengths=results[0]._wavelengths
        )

    # Scalar columns via numpy (fastest path)
    data: dict[str, np.ndarray] = {}
    for key in Tm30Result._SCALAR_KEYS:
        data[key] = np.fromiter(
            (getattr(r, key) for r in results), dtype=np.float64, count=n
        )

    if arrays or expand_arrays:
        data["rcs_hj"] = [list(r.rcs_hj) for r in results]
        data["rhs_hj"] = [list(r.rhs_hj) for r in results]

    if expand_arrays:
        # rf_cesi -> 99 individual columns
        cesi_matrix = np.array([r.rf_cesi for r in results], dtype=np.float64)
        for i in range(99):
            data[f"rf_cesi_{i + 1}"] = cesi_matrix[:, i]
    elif arrays:
        data["rf_cesi"] = [list(r.rf_cesi) for r in results]

    return pd.DataFrame(data)


class TM30Calc:
    """Pre-loaded TM-30 calculator.

    Loads reference tables (CMFs, CES, daylight basis, Planckian LUT)
    once at construction.  Call eval() for single or batch SPDs.

    Parameters
    ----------
    data_dir : str or None
        Directory containing the TM-30 data CSV files.
        Defaults to the built-in path.
    cmf : Cmf, str, Path, or None
        CIE observer for tristimulus integration (10-deg CMF).
        - Cmf.CIE_1964_10 (default) - TM-30-20 standard
        - Cmf.CIE_1931_2, Cmf.CIE_2006_10, Cmf.CIE_2015_2, Cmf.CIE_2015_10
        - '1931_2' (string lookup, case-insensitive)
        - '/path/to/my_cmf.csv' (custom CSV)
        Default: CIE 1964 10-deg (cmf='1964_10').
    cmf_2deg : Cmf, str, Path, or None
        CIE observer for CCT computation (2-deg CMF).
        Same format as `cmf`.  Default: CIE 1931 2-deg (cmf_2deg='1931_2').
    wavelengths : np.ndarray or None
        Fixed wavelength grid (nm) this calculator is bound to.  Defaults to
        380-780 nm at 1 nm steps (401 points, the implicit default used
        throughout this project) if None.  At construction, the 99 CES
        reflectance curves, the 2 deg/10 deg CMF curves, and the daylight
        basis are resampled to this grid once and cached, so that eval()
        calls for SPDs sharing this grid (the common case) skip resampling
        entirely.  This grid never changes after construction -- pass an
        explicit `wavelengths=` to eval() for a one-off different grid
        without replacing it.
    n_workers : int
        Number of worker threads for batch evaluation.  1 (default) is the
        pure sequential path with zero thread overhead; values > 1 spawn
        min(n_workers, batch_size) threads across the batch (results are
        bit-identical to the sequential path).  Values < 1 raise ValueError.
    persistent_workers : bool
        Keep the worker threads alive across eval() calls instead of
        spawning them per call (Phase 2).  Only meaningful when
        n_workers > 1: with n_workers <= 1 it is SILENTLY INERT (no
        pool is created, behavior is exactly the sequential path) - this
        is a documented decision, not an error.  Use for long-lived
        calculators making many repeated eval() calls; threads are
        created eagerly at construction and shut down when the
        calculator is garbage-collected.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        *,
        cmf: Cmf | str | os.PathLike | None = None,
        cmf_2deg: Cmf | str | os.PathLike | None = None,
        wavelengths: np.ndarray | None = None,
        n_workers: int = 1,
        persistent_workers: bool = False,
    ) -> None:
        if n_workers < 1:
            raise ValueError(
                f"n_workers must be >= 1 (got {n_workers}); "
                "n_workers=0/-1 auto-detection is not implemented"
            )
        self._n_workers = n_workers
        self._persistent_workers = persistent_workers
        if data_dir is None:
            # Resolve compiled-in TM30_DATA_DIR from the existing Tm30 class.
            # Tm30.__init__.__doc__ contains the default data_dir path.
            import re

            doc = tm30_core.Tm30.__init__.__doc__ or ""
            m = re.search(r"'([^']+)'", doc)
            data_dir = m.group(1) if m else ""
            if not data_dir:
                raise RuntimeError(
                    "Cannot determine default data_dir. Pass data_dir= explicitly."
                )

        # Resolve CMF paths
        path_2deg = _resolve_cmf(cmf_2deg, data_dir, suffix="2deg")
        path_10deg = _resolve_cmf(cmf, data_dir)

        # Use the explicit-CMF constructor (n_workers/persistent_workers
        # handled inside BatchContext: pool created eagerly when
        # persistent_workers && n_workers > 1, else inert).
        self._ctx = tm30_core.BatchContext(
            data_dir, path_2deg, path_10deg, n_workers, persistent_workers
        )
        self._data_dir = data_dir
        self._cmf = cmf
        self._cmf_2deg = cmf_2deg

        # -- Fixed wavelength grid: normalize, cache, precompute tables --
        if wavelengths is None:
            wavelengths = np.arange(380.0, 781.0, 1.0)
        wavelengths = np.asarray(wavelengths)
        if wavelengths.dtype != np.float64:
            wavelengths = wavelengths.astype(np.float64)
        wavelengths = np.ascontiguousarray(wavelengths)
        self._wavelengths = wavelengths
        self._ctx.set_fixed_grid(self._wavelengths)

    @property
    def cmf(self) -> Cmf | str | os.PathLike | None:
        """Configured 10-deg CMF observer (tristimulus integration)."""
        return self._cmf

    @property
    def cmf_2deg(self) -> Cmf | str | os.PathLike | None:
        """Configured 2-deg CMF observer (CCT computation)."""
        return self._cmf_2deg

    @property
    def wavelengths(self) -> np.ndarray:
        """The fixed wavelength grid (nm) this calculator is bound to."""
        return self._wavelengths

    def eval(
        self,
        spd: np.ndarray,
        wavelengths: np.ndarray | None = None,
        *,
        bins: bool = True,
        samples: bool = False,
        extras: bool = False,
    ) -> Tm30Result | Tm30BatchResult:
        """Evaluate TM-30 for one or many SPDs.

        Parameters
        ----------
        spd : np.ndarray, shape (N_wl,) or (N_spds, N_wl)
            1-D -> single SPD -> single Tm30Result (scalar fields)
            2-D -> batch of N_spds -> single Tm30BatchResult (every field is
            a numpy array with the batch as its leading axis, e.g. `.rf`
            has shape (N_spds,) - see Tm30BatchResult's docstring)
        wavelengths : np.ndarray or None
            Wavelength grid (nm) for this call.

            None (the common case): use the wavelength grid this calculator
            was constructed with (see the `wavelengths` parameter of
            ``TM30Calc.__init__``) - `spd`'s wavelength-axis length must
            match it exactly, or a ValueError is raised naming both
            lengths.  This reuses the cached, pre-resampled CES/CMF/
            daylight-basis tables - no resampling happens.

            Explicit array: a one-off different grid for this call only.
            Resampling is redone on the fly; the calculator's cached fixed
            grid is unaffected and unchanged by this.
        bins : bool
            Include per-bin metrics (Rcs,hj, Rhs,hj).  Default True - pass
            False to skip allocating/copying these arrays as a batch-size
            memory/bandwidth optimization.
        samples : bool
            Include per-sample fidelity Rf,CESi.  Default False (matching
            the C++ Tm30Request default) - pass True to include the
            99-element per-sample array; leaving it off skips allocating/
            copying that array as a batch-size memory/bandwidth
            optimization.
        extras : bool
            Include additional fields: rf_hj, de_hj (16 each), CVG
            coordinates (cvg_j_test/x_test/y_test/j_ref/x_ref/y_ref, 16
            each), reference_spd, and xyz_test_ces/xyz_ref_ces (99x3 each).
            Default False - these add ~1100+ floats/SPD, so they're opt-in.

        Returns
        -------
        Tm30Result (single SPD) or Tm30BatchResult (batch of SPDs)
        """
        single = spd.ndim == 1
        matrix = spd[np.newaxis, :] if single else spd
        if matrix.dtype != np.float64:
            matrix = matrix.astype(np.float64)
        matrix = np.ascontiguousarray(matrix)

        if wavelengths is None:
            if matrix.shape[1] != len(self._wavelengths):
                raise ValueError(
                    f"SPD wavelength-axis length ({matrix.shape[1]}) does "
                    f"not match this TM30Calc's fixed wavelength grid "
                    f"length ({len(self._wavelengths)}). Pass an explicit "
                    f"wavelengths= to eval() for a different grid, or "
                    f"construct TM30Calc(wavelengths=...) matching your "
                    f"data."
                )
            self._ctx.prepare_batch(matrix, self._wavelengths)
            raw = self._ctx.evaluate_cached(
                bins=bins,
                samples=samples,
                extras=extras,
                n_workers=self._n_workers,
            )
            used_wavelengths = self._wavelengths
        else:
            if wavelengths.dtype != np.float64:
                wavelengths = wavelengths.astype(np.float64)
            # A column slice (e.g. `csv[:, 0]`), transpose, or reversed view
            # is not contiguous - the C++ layer requires it to be.
            wavelengths = np.ascontiguousarray(wavelengths)
            self._ctx.prepare_batch(matrix, wavelengths)
            raw = self._ctx.evaluate(
                bins=bins,
                samples=samples,
                extras=extras,
                n_workers=self._n_workers,
            )
            used_wavelengths = wavelengths

        if single:
            results = [
                Tm30Result(d, wavelengths=used_wavelengths)
                for d in raw
                if d is not None
            ]
            return results[0]
        return Tm30BatchResult(raw, wavelengths=used_wavelengths)

    def __repr__(self) -> str:
        parts = ["tables loaded"]
        if self._cmf is not None:
            cmf_str = self._cmf.value if isinstance(self._cmf, Cmf) else str(self._cmf)
            parts.append(f"cmf={cmf_str}")
        if self._cmf_2deg is not None:
            cmf2_str = (
                self._cmf_2deg.value
                if isinstance(self._cmf_2deg, Cmf)
                else str(self._cmf_2deg)
            )
            if cmf2_str != "1931_2":  # only show if non-default
                parts.append(f"cmf_2deg={cmf2_str}")
        return f"TM30Calc({', '.join(parts)})"

    # -- Convenience: SPD -> XYZ / Yuv --------------------------------

    def spd_to_xyz(
        self,
        spd: np.ndarray,
        wavelengths: np.ndarray | None = None,
        *,
        K: float | None = None,
        cmf: Cmf | str | os.PathLike | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
    ) -> np.ndarray:
        """Compute source XYZ for one or many SPDs.

        Uses CIE 1964 10-deg CMFs.

        Parameters
        ----------
        spd : np.ndarray, shape (N_wl,) or (N_spds, N_wl)
            1-D -> single SPD -> returns (3,) array [X, Y, Z]
            2-D -> batch of N_spds -> returns (N_spds, 3) array
        wavelengths : np.ndarray or None
            Wavelength grid (nm).  None (the common case) -> use this
            calculator's fixed wavelength grid (see the `wavelengths`
            parameter of ``TM30Calc.__init__``).  Explicit array -> a
            one-off different grid for this call only.
        K : float or None
            Normalisation constant.  None (default): auto-compute
            k = 100/integral St*ybar dlambda -> Y = 100 (TM-30-20 S3.2 Eq. 4).
            K = 1.0: raw tristimulus integrals.
            K = 683.0: photometric absolute -- Km, the maximum luminous
            efficacy of radiation at 555 nm (CIE/SI definition of the
            candela); not a TM-30-20 quantity.
        cmf : Cmf, str, Path, or None
            CIE observer for this call. None (default): use this
            calculator's bound CMF. Explicit value: load+resample a
            different CMF for this call only.
        lambda_min, lambda_max : float or None
            Per-call override of integration bounds.  None -> integrate
            over the full wavelength grid.

        Returns
        -------
        np.ndarray, shape (3,) or (N_spds, 3)
        """
        lo = lambda_min
        hi = lambda_max

        single = spd.ndim == 1
        matrix = spd[np.newaxis, :] if single else spd
        if matrix.dtype != np.float64:
            matrix = matrix.astype(np.float64)
        matrix = np.ascontiguousarray(matrix)

        if wavelengths is None:
            wavelengths = self._wavelengths
        else:
            if wavelengths.dtype != np.float64:
                wavelengths = wavelengths.astype(np.float64)
            # A column slice (e.g. `csv[:, 0]`), transpose, or reversed view
            # is not contiguous - the C++ layer requires it to be.
            wavelengths = np.ascontiguousarray(wavelengths)

        cmf_path = None if cmf is None else _resolve_cmf(cmf, self._data_dir)
        result = self._ctx.spd_to_xyz(matrix, wavelengths, K, lo, hi, cmf_path)
        return result[0] if single else result

    def spd_to_Yuv(
        self,
        spd: np.ndarray,
        wavelengths: np.ndarray | None = None,
        *,
        K: float | None = None,
        cmf: Cmf | str | os.PathLike | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
    ) -> np.ndarray:
        """Compute CIE 1976 Y,u',v' for one or many SPDs.

        Chains spd_to_xyz -> xyz_to_Yuv (CIE 15:2004 S8.2.1).

        Parameters
        ----------
        spd : np.ndarray, shape (N_wl,) or (N_spds, N_wl)
            1-D -> single SPD -> returns (3,) array [Y, u', v']
            2-D -> batch of N_spds -> returns (N_spds, 3) array
        wavelengths : np.ndarray or None
            Wavelength grid (nm).  None (the common case) -> use this
            calculator's fixed wavelength grid (see the `wavelengths`
            parameter of ``TM30Calc.__init__``).  Explicit array -> a
            one-off different grid for this call only.
        K : float or None
            Normalisation constant passed to spd_to_xyz.
            None (default): auto-normalise Y = 100.
            K = 1.0: raw integrals. K = 683.0: photometric absolute.
        cmf : Cmf, str, Path, or None
            CIE observer for this call. None (default): use this
            calculator's bound CMF. Explicit value: load+resample a
            different CMF for this call only.
        lambda_min, lambda_max : float or None
            Per-call override of integration bounds.  None -> integrate
            over the full wavelength grid.

        Returns
        -------
        np.ndarray, shape (3,) or (N_spds, 3)
        """
        lo = lambda_min
        hi = lambda_max

        single = spd.ndim == 1
        matrix = spd[np.newaxis, :] if single else spd
        if matrix.dtype != np.float64:
            matrix = matrix.astype(np.float64)
        matrix = np.ascontiguousarray(matrix)

        if wavelengths is None:
            wavelengths = self._wavelengths
        else:
            if wavelengths.dtype != np.float64:
                wavelengths = wavelengths.astype(np.float64)
            # A column slice (e.g. `csv[:, 0]`), transpose, or reversed view
            # is not contiguous - the C++ layer requires it to be.
            wavelengths = np.ascontiguousarray(wavelengths)

        cmf_path = None if cmf is None else _resolve_cmf(cmf, self._data_dir)
        result = self._ctx.spd_to_Yuv(matrix, wavelengths, K, lo, hi, cmf_path)
        return result[0] if single else result

    def xyz_to_Yuv(self, xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ tristimulus values to CIE 1976 Y,u',v'.

        Pure coordinate transform - no CMF or wavelength grid is involved,
        since the CMF's job (turning a spectrum into XYZ) is already done
        by the time XYZ reaches this function.

        Parameters
        ----------
        xyz : np.ndarray, shape (3,) or (N, 3)
            1-D -> single XYZ triple -> returns (3,) array [Y, u', v']
            2-D -> batch of N triples -> returns (N, 3) array

        Returns
        -------
        np.ndarray, shape (3,) or (N, 3)
        """
        single = xyz.ndim == 1
        matrix = xyz[np.newaxis, :] if single else xyz
        if matrix.dtype != np.float64:
            matrix = matrix.astype(np.float64)
        matrix = np.ascontiguousarray(matrix)

        result = self._ctx.xyz_to_Yuv(matrix)
        return result[0] if single else result

    def cct_to_xyz(
        self,
        cct,
        *,
        cmf: Cmf | str | os.PathLike | None = None,
        K: float | None = None,
    ) -> np.ndarray:
        """Compute XYZ for the TM-30-20 reference illuminant at a given CCT.

        Always uses this calculator's fixed wavelength grid (see the
        `wavelengths` parameter of ``TM30Calc.__init__``) - there's no
        per-call wavelengths override, since (unlike spd_to_xyz) there's no
        input SPD to carry an implicit grid.

        Parameters
        ----------
        cct : float or np.ndarray, shape (N,)
            Correlated color temperature(s) (K).
            scalar -> returns (3,) array [X, Y, Z]
            (N,) array -> returns (N, 3) array
        cmf : Cmf, str, Path, or None
            CIE observer for this call. None (default): use this
            calculator's bound CMF (see the `cmf` parameter of
            ``TM30Calc.__init__``). Explicit value: load+resample a
            different CMF for this call only (same format as the
            constructor's `cmf=`).
        K : float or None
            Normalisation constant. None (default): auto-normalise Y=100.
            K=1.0: raw integrals. K=683.0: photometric absolute.

        Returns
        -------
        np.ndarray, shape (3,) or (N, 3)
        """
        single = np.ndim(cct) == 0
        cct_arr = np.ascontiguousarray(np.atleast_1d(np.asarray(cct, dtype=np.float64)))

        cmf_path = None if cmf is None else _resolve_cmf(cmf, self._data_dir)

        result = self._ctx.cct_to_xyz(cct_arr, cmf_path, K)
        return result[0] if single else result

    def spd_to_power(
        self,
        spd: np.ndarray,
        wavelengths: np.ndarray | None = None,
        *,
        cmf: Cmf | str | os.PathLike | None = None,
        photometric: bool = False,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
    ):
        """Integrate one or many SPDs to a single power value each.

        Parameters
        ----------
        spd : np.ndarray, shape (N_wl,) or (N_spds, N_wl)
            1-D -> single SPD -> returns a float
            2-D -> batch of N_spds -> returns (N_spds,) array
        wavelengths : np.ndarray or None
            Wavelength grid (nm). None (common case): use this
            calculator's fixed grid - `spd`'s wavelength-axis length must
            match it exactly. Explicit array: a one-off grid for this call
            only.
        cmf : Cmf, str, Path, or None
            CIE observer for the photometric weighting. None (default):
            use this calculator's bound CMF. Has no effect when
            photometric=False (radiometric power has no CMF dependency).
        photometric : bool
            False (default): radiometric power (W), an unweighted spectral
            integral. True: photometric power (lm), a Km=683.0 x
            ybar-weighted integral.
        lambda_min, lambda_max : float or None
            Integration bounds (nm). None: full range of `wavelengths`.

        Returns
        -------
        float (single SPD) or np.ndarray, shape (N_spds,) (batch)
        """
        single = spd.ndim == 1
        matrix = spd[np.newaxis, :] if single else spd
        if matrix.dtype != np.float64:
            matrix = matrix.astype(np.float64)
        matrix = np.ascontiguousarray(matrix)

        if wavelengths is None:
            if matrix.shape[1] != len(self._wavelengths):
                raise ValueError(
                    f"SPD wavelength-axis length ({matrix.shape[1]}) does "
                    f"not match this TM30Calc's fixed wavelength grid "
                    f"length ({len(self._wavelengths)}). Pass an explicit "
                    f"wavelengths= for a different grid."
                )
            wl_arg = self._wavelengths
        else:
            if wavelengths.dtype != np.float64:
                wavelengths = wavelengths.astype(np.float64)
            wl_arg = np.ascontiguousarray(wavelengths)

        cmf_path = None if cmf is None else _resolve_cmf(cmf, self._data_dir)

        result = self._ctx.spd_to_power(
            matrix, wl_arg, cmf_path, photometric, lambda_min, lambda_max
        )
        return float(result[0]) if single else result

    def spd_to_cct(
        self,
        spd: np.ndarray,
        wavelengths: np.ndarray | None = None,
        *,
        cmf: Cmf | str | os.PathLike | None = None,
    ) -> np.ndarray:
        """Compute CCT and Duv for one or many SPDs.

        Uses the CIE 1931 2-deg CMFs and the Ohno 2014 method (TM-30-20
        S3.1 exception, S3.3) -- this is the one calculation in TM-30-20
        that uses the 2-deg, not 10-deg, observer.

        Parameters
        ----------
        spd : np.ndarray, shape (N_wl,) or (N_spds, N_wl)
            1-D -> single SPD -> returns shape (2,) array [cct, duv]
            2-D -> batch of N_spds -> returns shape (2, N_spds) array,
            row 0 = cct, row 1 = duv
        wavelengths : np.ndarray or None
            Wavelength grid (nm). None (common case): use this
            calculator's fixed wavelength grid. Explicit array: a
            one-off different grid for this call only.
        cmf : Cmf, str, Path, or None
            2-deg CIE observer for this call. None (default): use this
            calculator's bound cmf_2deg. Explicit value: load+resample a
            different 2-deg CMF for this call only.

        Returns
        -------
        np.ndarray, shape (2,) or (2, N_spds)
            Row/index 0 is cct (K), row/index 1 is duv. Unpacks
            naturally: ``cct, duv = calc.spd_to_cct(spd_matrix)``. Index
            [0] to keep only cct and discard duv.
        """
        single = spd.ndim == 1
        matrix = spd[np.newaxis, :] if single else spd
        if matrix.dtype != np.float64:
            matrix = matrix.astype(np.float64)
        matrix = np.ascontiguousarray(matrix)

        if wavelengths is None:
            wavelengths = self._wavelengths
        else:
            if wavelengths.dtype != np.float64:
                wavelengths = wavelengths.astype(np.float64)
            # A column slice (e.g. `csv[:, 0]`), transpose, or reversed view
            # is not contiguous - the C++ layer requires it to be.
            wavelengths = np.ascontiguousarray(wavelengths)

        cmf_path = (
            None if cmf is None else _resolve_cmf(cmf, self._data_dir, suffix="2deg")
        )
        result = self._ctx.spd_to_cct(matrix, wavelengths, cmf_path)
        return result[:, 0] if single else result
