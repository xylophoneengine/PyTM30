#!/usr/bin/env python3
"""
generate_planckian_lut.py -- Regenerate data/planckian_uv.csv.

Values are computed from first principles in this script and are
byte-reproducible from it:

  1. Planck's law blackbody spectral radiance (TM-30-20 Sec 3.3
     Eq. (5)-(6), c2 = 1.4388e-2 m*K), evaluated at each candidate CCT.
  2. Integrated against the CIE 1931 2-degree standard observer CMF
     (data/cmf_1931_2.csv, sourced from colour-science; full 360-830nm
     range at 1nm -- not data/cie_1931_2.csv, the separate 380-780nm
     table used for the test source's CCT input) via trapezoidal
     integration.
  3. Converted to CIE 1960 UCS (u,v) via u=4X/(X+15Y+3Z), v=6Y/(X+15Y+3Z).

Overall SPD scale is irrelevant to (u,v) (they are chromaticity ratios),
so no 560nm normalisation is needed here (unlike the reference-illuminant
generator in reference.cpp, which needs the actual SPD).

Temperature grid: geometric sequence T[n] = 1000 * 1.0025^n,
n = 0..1488 (1489 points, 1000K to ~41073K). The grid parameters
(1000K start, 0.25% geometric increment) follow the LUT used by the
calculator supplied with ANSI/IES TM-30; see Smet et al., "Recommended
Method for Determining the Correlated Color Temperature and Distance
from the Planckian Locus of a Light Source", Leukos 2023,
doi:10.1080/15502724.2023.2248397, which records the TM-30 calculator
as using a 0.25% increment LUT.

Output is cross-checked against colour.temperature.CCT_to_uv_Planck1900
(colour-science's own Planckian (u,v) utility) at several sample points
-- agreement to ~1.4e-6.
"""
import os
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))


def load_csv(path):
    with open(path) as f:
        r = list(csv.reader(f))
    rows = np.array([[float(x) for x in row] for row in r[1:]])
    return rows


def planckian_relative_spd(T, wl_nm):
    c2 = 1.4388e-2  # TM-30-20 Sec 3.3 Eq. (6), second radiation constant (m*K)
    wl_m = wl_nm * 1e-9
    L = wl_m ** -5.0 / (np.exp(c2 / (wl_m * T)) - 1.0)
    return L  # un-normalised -- fine, since (u,v) is scale-invariant


def xyz_to_uv(X, Y, Z):
    d = X + 15.0 * Y + 3.0 * Z
    return 4.0 * X / d, 6.0 * Y / d


def main():
    cmf = load_csv(os.path.join(OUT_DIR, "cmf_1931_2.csv"))
    wl = cmf[:, 0]
    xb, yb, zb = cmf[:, 1], cmf[:, 2], cmf[:, 3]

    n_points = 1489
    T = 1000.0 * (1.0025 ** np.arange(n_points))

    u_out = np.empty(n_points)
    v_out = np.empty(n_points)
    for i, t in enumerate(T):
        spd = planckian_relative_spd(t, wl)
        X = np.trapezoid(spd * xb, wl)
        Y = np.trapezoid(spd * yb, wl)
        Z = np.trapezoid(spd * zb, wl)
        u_out[i], v_out[i] = xyz_to_uv(X, Y, Z)

    path = os.path.join(OUT_DIR, "planckian_uv.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cct", "u", "v"])
        for i in range(n_points):
            w.writerow([repr(float(T[i])), repr(float(u_out[i])), repr(float(v_out[i]))])

    print(f"Wrote {path} ({n_points} rows, T={T[0]:.3f}..{T[-1]:.3f} K)")

    # Cross-check against colour-science's own Planckian (u,v) utility.
    try:
        import colour
        for i in (0, 500, 1000, 1488):
            uv_cs = colour.temperature.CCT_to_uv_Planck1900(np.array([T[i], 0.0]))
            du = abs(uv_cs[0][0] - u_out[i])
            dv = abs(uv_cs[0][1] - v_out[i])
            print(f"  T={T[i]:.3f}: ours=({u_out[i]:.9f},{v_out[i]:.9f}) "
                  f"colour-science Planck1900=({uv_cs[0][0]:.9f},{uv_cs[0][1]:.9f}) "
                  f"d=({du:.2e},{dv:.2e})")
    except Exception as e:
        print("cross-check skipped:", e)


if __name__ == "__main__":
    main()
