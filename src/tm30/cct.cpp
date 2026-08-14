// CCT and Duv via the Ohno triangular + parabolic search.
//
// TM-30-20 §3.3 prints no CCT algorithm; it incorporates one by normative
// reference: Ohno, Y. "Practical Use and Calculation of CCT and Duv."
// Leukos 10(1):47-55, doi:10.1080/15502724.2014.839020. (TM-30-20's
// reference list dates the article 2013, online-first; Smet et al. 2023
// cite it as 2014 -- same article, so the DOI is the stable citation.)
// Per TM-30-20 §3.1, CCT determination uses the CIE 1931 2-deg observer,
// unlike the 1964 10-deg observer used everywhere else.

#include "tm30/cct.hpp"
#include "tm30/chromaticity.hpp"
#include "tm30/csv_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace tm30 {

// -------------------------------------------------------------------------
// LUT loading
// -------------------------------------------------------------------------

PlanckianLut load_planckian_lut(const std::string &filepath) {
  CsvTable table = load_csv(filepath);

  // CSV columns: cct, u, v
  if (table.headers.size() < 3) {
    throw std::runtime_error(
        "Planckian LUT CSV must have at least 3 columns (cct, u, v): " +
        filepath);
  }

  PlanckianLut lut;
  lut.T.reserve(table.rows.size());
  lut.u.reserve(table.rows.size());
  lut.v.reserve(table.rows.size());

  for (const auto &row : table.rows) {
    lut.T.push_back(row[0]);
    lut.u.push_back(row[1]);
    lut.v.push_back(row[2]);
  }

  if (lut.T.size() < 3) {
    throw std::runtime_error("Planckian LUT must contain at least 3 points: " +
                             filepath);
  }

  return lut;
}

// -------------------------------------------------------------------------
// Ohno 2014 triangular + parabolic CCT solver
// -------------------------------------------------------------------------

CctDuvResult compute_cct_duv(double u_test, double v_test,
                             const PlanckianLut &lut) {
  const std::size_t n = lut.T.size();

  // --- Step 1: Find closest point in LUT by Euclidean distance in (u,v) ---

  std::size_t best_i = 0;
  double best_dist2 = std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < n; ++i) {
    const double du = u_test - lut.u[i];
    const double dv = v_test - lut.v[i];
    const double d2 = du * du + dv * dv;
    if (d2 < best_dist2) {
      best_dist2 = d2;
      best_i = i;
    }
  }

  // --- Step 2: Select three adjacent LUT points for triangular/parabolic fit
  // ---

  // The three points are (m1, 0, p1) where 0 is the closest.
  // If best_i is at endpoint, shift inward.
  std::size_t im1, i0, ip1;

  if (best_i == 0) {
    im1 = 0;
    i0 = 1;
    ip1 = 2;
  } else if (best_i == n - 1) {
    im1 = n - 3;
    i0 = n - 2;
    ip1 = n - 1;
  } else {
    im1 = best_i - 1;
    i0 = best_i;
    ip1 = best_i + 1;
  }

  const double Tm1 = lut.T[im1];
  const double T0 = lut.T[i0];
  const double Tp1 = lut.T[ip1];

  const double um1 = lut.u[im1];
  const double u0 = lut.u[i0];
  const double up1 = lut.u[ip1];

  const double vm1 = lut.v[im1];
  const double v0 = lut.v[i0];
  const double vp1 = lut.v[ip1];

  // --- Step 3: Compute distances from test point to the three LUT points ---

  auto dist = [&](double u, double v) -> double {
    const double du = u_test - u;
    const double dv = v_test - v;
    return std::sqrt(du * du + dv * dv);
  };

  const double dm1 = dist(um1, vm1);
  const double d0 = dist(u0, v0);
  const double dp1 = dist(up1, vp1);

  // --- Step 4: Triangular solution ---

  // Length of segment from m1 to p1
  const double l =
      std::sqrt((up1 - um1) * (up1 - um1) + (vp1 - vm1) * (vp1 - vm1));

  // Projected distance along segment: x = (dm1^2 - dp1^2 + l^2) / (2*l)
  // TM-30-20 §3.3 incorporation: Ohno (2014) triangular geometry. The
  // [0, l] clamp below is an implementation robustness extension, not
  // part of the published method.
  double x = (dm1 * dm1 - dp1 * dp1 + l * l) / (2.0 * l);
  if (x < 0.0)
    x = 0.0;
  if (x > l)
    x = l;

  // Triangular CCT estimate: linear interpolation along the segment
  const double T_tri = Tm1 + (Tp1 - Tm1) * (x / l);

  // Triangular Duv: signed distance from test to the chord
  // Chord point at x: (uch, vch)
  const double uch = um1 + (up1 - um1) * (x / l);
  const double vch = vm1 + (vp1 - vm1) * (x / l);

  // Duv magnitude
  const double du_ch = u_test - uch;
  const double dv_ch = v_test - vch;
  const double duv_mag = std::sqrt(du_ch * du_ch + dv_ch * dv_ch);

  // Sign: positive if test is "above" the chord (toward green/yellow)
  // We determine sign by comparing v coordinates: v_test > v_ch -> positive
  // (In CIE 1960 UCS, the Planckian locus generally has negative slope,
  //  so being "above" means higher v at the same approximate u.)
  // Duv sign convention per Ohno (2014); method incorporated by
  // TM-30-20 §3.3 by normative reference.
  const double sign = (v_test >= vch) ? 1.0 : -1.0;
  const double duv_tri = duv_mag * sign;

  // --- Step 5: Parabolic solution ---

  // Fit a parabola T -> distance for the three points
  // a*T^2 + b*T + c = distance
  // Using the formula from Ohno 2014 (equivalent to Lagrange interpolation)

  // Denominator for the quadratic coefficients
  // TM-30-20 §3.3 incorporation: Ohno (2014) parabolic fit. The zero
  // guard below is an implementation robustness extension, not part of
  // the published method.
  double denom = (Tp1 - T0) * (Tm1 - Tp1) * (T0 - Tm1);
  if (std::abs(denom) < 1e-30)
    denom = 1e-30;

  // a = (Tm1*(dp1-d0) + T0*(dm1-dp1) + Tp1*(d0-dm1)) / denom
  const double a =
      (Tm1 * (dp1 - d0) + T0 * (dm1 - dp1) + Tp1 * (d0 - dm1)) / denom;

  // b = -(Tm1^2*(dp1-d0) + T0^2*(dm1-dp1) + Tp1^2*(d0-dm1)) / denom
  const double b = -(Tm1 * Tm1 * (dp1 - d0) + T0 * T0 * (dm1 - dp1) +
                     Tp1 * Tp1 * (d0 - dm1)) /
                   denom;

  // c = -(dm1*(Tp1-T0)*Tp1*T0 + d0*(Tm1-Tp1)*Tm1*Tp1 + dp1*(T0-Tm1)*T0*Tm1) /
  // denom
  const double c =
      -(dm1 * (Tp1 - T0) * Tp1 * T0 + d0 * (Tm1 - Tp1) * Tm1 * Tp1 +
        dp1 * (T0 - Tm1) * T0 * Tm1) /
      denom;

  // Vertex of parabola: T_par = -b / (2a)
  // Ohno (2014) parabolic vertex formula; method incorporated by
  // TM-30-20 §3.3 by normative reference. The [Tm1, Tp1] clamp and the
  // triangular fallback below are implementation robustness extensions.
  double T_par;
  if (std::abs(a) > 1e-30) {
    T_par = -b / (2.0 * a);
    // Clamp to the local range [Tm1, Tp1]
    if (T_par < Tm1)
      T_par = Tm1;
    if (T_par > Tp1)
      T_par = Tp1;
  } else {
    T_par = T_tri; // fall back to triangular
  }

  // Parabolic Duv
  const double duv_par_mag = a * T_par * T_par + b * T_par + c;
  // Sign follows the sign of duv_tri (same side of locus)
  const double duv_par = duv_par_mag * sign;

  // --- Step 6: Select between triangular and parabolic ---

  // The triangular/parabolic selection threshold is from Ohno (2014),
  // the method incorporated by normative reference; the TM-30-20 text
  // itself prints no threshold. When |Duv| < 0.002 the triangular
  // solution is used; otherwise the parabolic one.
  // TM-30-20 §3.3 incorporation; threshold value from Ohno (2014).
  constexpr double duv_threshold = 0.002;

  // Apply linear shift to triangular solution (as in luxpy/TM-30)
  // T_tri_shift = T_tri + (T_par - T_tri) * |duv_tri| / threshold
  // TM-30-20 §3.3 (blend region clamp)
  const double duv_abs = std::abs(duv_tri);
  const double T_tri_shift =
      T_tri + (T_par - T_tri) * std::min(duv_abs / duv_threshold, 1.0);

  CctDuvResult result;
  if (duv_abs < duv_threshold) {
    result.cct = T_tri_shift;
    result.duv = duv_tri;
  } else {
    result.cct = T_par;
    result.duv = duv_par;
  }

  return result;
}

// -------------------------------------------------------------------------
// Convenience: from XYZ
// -------------------------------------------------------------------------

CctDuvResult compute_cct_duv_from_xyz(double X, double Y, double Z,
                                      const PlanckianLut &lut) {
  const UvCoord uv = xyz_to_uv(X, Y, Z);
  return compute_cct_duv(uv.u, uv.v, lut);
}

} // namespace tm30
