// TM-30-20: Gamut Area Index (Rg), per-bin local metrics, and CVG coordinates.
//
// TM-30-20 §4.4: Gamut Index (Rg)
// TM-30-20 §4.5: Color Vector Graphic (CVG)
// TM-30-20 §4.6: Local Chroma Shift (Rcs,hj)
// TM-30-20 §4.7: Local Hue Shift (Rhs,hj)
// TM-30-20 §4.8: Local Color Fidelity (Rf,hj)
#include "tm30/gamut.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers> // std::numbers::pi

namespace tm30 {

// --- Constants ------------------------------------------------------------

// TM-30-20 §4.3: 16 bins of 22.5-deg each.
static constexpr int kNumBins = 16; // TM-30-20 §4.3

// TM-30-20 §4.3: Bin width in radians (22.5-deg = pi/8).
static constexpr double kBinWidthDeg = 22.5; // TM-30-20 §4.3

// TM-30-20 §4.1: Rf scaling factor.
static constexpr double kRfScale = 6.73; // TM-30-20 §4.1 Eq. (53)

// TM-30-20 §4.1: Log rescale divisor.
static constexpr double kLogRescale = 10.0; // TM-30-20 §4.1 Eq. (54)

// TM-30-20 §4.6, §4.7: Local shift scaling.
// Note: Eqs. (62)/(63) as printed are ratio-valued and carry NO x100
// factor. §4.6/§4.7 only require Rcs,hj/Rhs,hj to be "represented as a
// percentage" -- a reporting instruction, not equation content. luxpy
// stores raw ratios; we match luxpy (compute raw, scale at report time).
static constexpr double kLocalShiftScale = 1.0; // TM-30-20 §4.6, §4.7

// TM-30-20 §4.5: reference coordinates are normalised to a unit circle
// (radius 1); the prescribed plot axis limits are +/-1.5 and the optional
// guide circles are radii 0.8/0.9/1.1/1.2, all dimensionless multiples of
// that unit circle. Eqs. (58)-(61) carry no scale factor. Any display
// scaling belongs in a plotting layer, not here.
static constexpr double kCvgScale = 1.0; // TM-30-20 §4.5

// --- Bin Averages ---------------------------------------------------------

BinAverages bin_average(const std::array<Cam02Ucs, 99> &jab_ces,
                        const HueBins &bins) {
  // TM-30-20 §4.4
  BinAverages avg{};

  for (int j = 0; j < kNumBins; ++j) {
    // TM-30-20 §4.4
    const auto &bin = bins[j];
    const std::size_t m = bin.size();

    if (m == 0) {
      // Empty bin: mark as NaN
      // TM-30-20 §4.4 edge case
      avg.J_prime[j] = std::numeric_limits<double>::quiet_NaN();
      avg.a_prime[j] = std::numeric_limits<double>::quiet_NaN();
      avg.b_prime[j] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    // TM-30-20 §4.4: bin average J, a', b' - initialize accumulators
    double sum_J = 0.0, sum_a = 0.0, sum_b = 0.0;
    for (int idx : bin) {
      sum_J += jab_ces[idx].J_prime;
      sum_a += jab_ces[idx].a_prime;
      sum_b += jab_ces[idx].b_prime;
    }
    // TM-30-20 §4.4: arithmetic mean
    avg.J_prime[j] = sum_J / static_cast<double>(m);
    avg.a_prime[j] = sum_a / static_cast<double>(m);
    avg.b_prime[j] = sum_b / static_cast<double>(m);
  }

  return avg;
}

// --- Polygon Area (Shoelace) ----------------------------------------------

double polygon_area(const BinAverages &avg) {
  // TM-30-20 §4.4: Shoelace formula in (a', b') plane

  // Collect non-NaN vertices in bin order (0->15).
  // TM-30-20 §4.4: empty bins are skipped.
  struct Vertex {
    double a;
    double b;
  };

  // Stack-allocate up to 16 vertices
  Vertex verts[16];
  int n = 0;
  for (int j = 0; j < kNumBins; ++j) {
    // TM-30-20 §4.4 edge case: skip empty bins
    if (std::isnan(avg.a_prime[j]))
      continue;
    verts[n].a = avg.a_prime[j]; // TM-30-20 §4.4
    verts[n].b = avg.b_prime[j]; // TM-30-20 §4.4
    ++n;
  }

  if (n < 3) {
    // Degenerate polygon: fewer than 3 vertices.
    // TM-30-20 §4.4 edge case
    return 0.0;
  }

  // Shoelace: A = 0.5 * |sum (x_i * y_{i+1} - x_{i+1} * y_i)|
  // TM-30-20 §4.4
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    const int next = (i + 1) % n; // wrap-around
    sum += verts[i].a * verts[next].b - verts[next].a * verts[i].b;
  }

  return 0.5 * std::abs(sum); // TM-30-20 §4.4
}

// --- Rg -------------------------------------------------------------------

double compute_rg(const BinAverages &test_avg, const BinAverages &ref_avg) {
  // TM-30-20 §4.4 Eq. (57)
  const double A_test = polygon_area(test_avg); // TM-30-20 §4.4
  const double A_ref = polygon_area(ref_avg);   // TM-30-20 §4.4
  return 100.0 * A_test / A_ref;                // TM-30-20 §4.4 Eq. (57)
}

// --- Local Bin Metrics ----------------------------------------------------

LocalBinMetrics compute_local_bin_metrics(const BinAverages &test_avg,
                                          const BinAverages &ref_avg,
                                          const std::array<double, 99> &delta_e,
                                          const HueBins &bins) {

  // TM-30-20 §4.6, §4.7, §4.8
  LocalBinMetrics metrics{};

  for (int j = 0; j < kNumBins; ++j) {
    // TM-30-20 §4.6, §4.7, §4.8
    const auto &bin = bins[j];
    const std::size_t m = bin.size();

    if (m == 0 || std::isnan(ref_avg.a_prime[j])) {
      // Empty bin: set metrics to NaN
      // TM-30-20 §4.6-§4.8 edge case
      metrics.Rf_hj[j] = std::numeric_limits<double>::quiet_NaN();
      metrics.Rcs_hj[j] = std::numeric_limits<double>::quiet_NaN();
      metrics.Rhs_hj[j] = std::numeric_limits<double>::quiet_NaN();
      metrics.DE_hj[j] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    // -- Mean dE' per bin ---------------------------------------
    // TM-30-20 §4.8
    double sum_de = 0.0;
    for (int idx : bin) {
      sum_de += delta_e[idx];
    }
    const double DE_hj = sum_de / static_cast<double>(m); // TM-30-20 §4.8
    metrics.DE_hj[j] = DE_hj;

    // -- Rf,hj --------------------------------------------------
    // TM-30-20 §4.8 Eq. (64), (65)
    const double Rf_hj_prime =
        100.0 - kRfScale * DE_hj; // TM-30-20 §4.8 Eq. (64)
    metrics.Rf_hj[j] =
        kLogRescale * std::log( // TM-30-20 §4.8 Eq. (65)
                          std::exp(Rf_hj_prime / kLogRescale) + 1.0);

    // -- Rcs,hj and Rhs,hj --------------------------------------
    // TM-30-20 §4.6, §4.7
    const double da = test_avg.a_prime[j] - ref_avg.a_prime[j]; // TM-30-20 §4.6
    const double db = test_avg.b_prime[j] - ref_avg.b_prime[j]; // TM-30-20 §4.6

    // Reference radial distance
    const double r_ref =
        std::sqrt(ref_avg.a_prime[j] * ref_avg.a_prime[j] +
                  ref_avg.b_prime[j] * ref_avg.b_prime[j]); // TM-30-20 §4.6

    if (r_ref < 1e-12) {
      // Degenerate: reference at origin - shifts undefined.
      // TM-30-20 §4.6 edge case
      metrics.Rcs_hj[j] = 0.0;
      metrics.Rhs_hj[j] = 0.0;
      continue;
    }

    // Bin bisector angle: thetaj = (j + 0.5) x 22.5-deg (0-indexed)
    // TM-30-20 §4.6: thetaj is the bisector angle of bin j
    const double theta_deg = (static_cast<double>(j) + 0.5) * kBinWidthDeg;
    const double theta = theta_deg * std::numbers::pi / 180.0; // TM-30-20 §4.6

    const double cos_t = std::cos(theta);
    const double sin_t = std::sin(theta);

    // Local chroma shift: Eq. (62)
    // TM-30-20 §4.6 Eq. (62)
    metrics.Rcs_hj[j] = kLocalShiftScale * (da * cos_t + db * sin_t) / r_ref;

    // Local hue shift: Eq. (63) - note leading negative on first term
    // TM-30-20 §4.7 Eq. (63)
    metrics.Rhs_hj[j] = kLocalShiftScale * (-da * sin_t + db * cos_t) / r_ref;
  }

  return metrics;
}

// --- CVG Coordinates ------------------------------------------------------

CvgCoordinates
compute_cvg_coordinates(const BinAverages &test_avg, const BinAverages &ref_avg,
                        const std::array<Cam02Ucs, 99> &jab_ref,
                        const HueBins &bins) {

  // TM-30-20 §4.5
  CvgCoordinates cvg{};

  for (int j = 0; j < kNumBins; ++j) {
    // Pass J' through as a pytm30 extension of the §4.4 bin averages.
    // NOT a §4.5 quantity -- §4.4 explicitly discards J' ("so that the
    // (a', b') coordinates each form a polygon"), and §4.5 Eqs. (58)-(61)
    // are strictly 2-D in (a', b'). Carrying J' is a convenience for
    // downstream consumers; the spec does not.
    cvg.J_test[j] = test_avg.J_prime[j];
    cvg.J_ref[j] = ref_avg.J_prime[j];

    if (std::isnan(ref_avg.a_prime[j])) {
      // Empty bin
      cvg.x_test[j] = std::numeric_limits<double>::quiet_NaN();
      cvg.y_test[j] = std::numeric_limits<double>::quiet_NaN();
      cvg.x_ref[j] = std::numeric_limits<double>::quiet_NaN();
      cvg.y_ref[j] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    // TM-30-20 §4.5 Eqs. (58)-(59): the reference circle position for bin j
    // derives from the arithmetic mean of the individual CES hue angles in
    // the bin, each taken from that sample's own reference-illuminant
    // (a', b'). This is distinct from the hue angle of the bin-averaged
    // coordinates, which would weight samples by chroma; Eqs. (58)-(59)
    // weight every sample in the bin equally.
    //
    // std::atan2 returns [-pi, pi], so samples with negative b' (bins 9-16)
    // come back negative and a raw mean would be meaningless; normalise
    // each angle to [0, 2*pi) before averaging. After normalisation a plain arithmetic
    // mean suffices: §4.3 places 0 deg on the positive a' axis with 16 bins
    // of 22.5 deg increasing counterclockwise, and §4.6 confirms 0 deg is
    // the boundary between bins 1 and 16, so no bin straddles the wrap
    // point and circular-mean machinery is unnecessary.
    double sum_h = 0.0; // TM-30-20 §4.5 Eqs. (58)-(59) accumulator
    for (int idx : bins[j]) {
      double h = std::atan2(jab_ref[idx].b_prime, jab_ref[idx].a_prime);
      if (h < 0.0) {                 // TM-30-20 §4.3: hue angles in [0, 360)
        h += 2.0 * std::numbers::pi; // TM-30-20 §4.3: full turn, [0, 2*pi)
      }
      sum_h += h;
    }
    const double h_bar = sum_h / static_cast<double>(bins[j].size());

    // Reference radial distance (for test coordinate offset)
    const double r_ref =
        std::sqrt(ref_avg.a_prime[j] * ref_avg.a_prime[j] +
                  ref_avg.b_prime[j] * ref_avg.b_prime[j]); // TM-30-20 §4.5

    // Reference circle coordinates: Eq. (58), (59)
    // TM-30-20 §4.5 Eq. (58), (59)
    const double x_ref_raw = std::cos(h_bar);
    const double y_ref_raw = std::sin(h_bar);

    cvg.x_ref[j] = kCvgScale * x_ref_raw; // TM-30-20 §4.5 Eq. (58)
    cvg.y_ref[j] = kCvgScale * y_ref_raw; // TM-30-20 §4.5 Eq. (59)

    // Test vector endpoints: Eq. (60), (61)
    // TM-30-20 §4.5 Eq. (60), (61)
    const double da = test_avg.a_prime[j] - ref_avg.a_prime[j];
    const double db = test_avg.b_prime[j] - ref_avg.b_prime[j];

    if (r_ref < 1e-12) {
      // Degenerate: reference at origin
      cvg.x_test[j] = kCvgScale * x_ref_raw;
      cvg.y_test[j] = kCvgScale * y_ref_raw;
    } else {
      cvg.x_test[j] =
          kCvgScale * (x_ref_raw + da / r_ref); // TM-30-20 §4.5 Eq. (60)
      cvg.y_test[j] =
          kCvgScale * (y_ref_raw + db / r_ref); // TM-30-20 §4.5 Eq. (61)
    }
  }

  return cvg;
}

// --- Main compute_gamut ---------------------------------------------------

GamutResult compute_gamut(const std::array<Cam02Ucs, 99> &jab_test,
                          const std::array<Cam02Ucs, 99> &jab_ref,
                          const std::array<double, 99> &delta_e,
                          const HueBins &bins) {

  // TM-30-20 §4.4-§4.8
  GamutResult result{};

  // Step 1: Bin-averaged J'a'b'
  // TM-30-20 §4.4
  result.test_avg = bin_average(jab_test, bins);
  result.ref_avg = bin_average(jab_ref, bins);

  // Step 2: Rg
  // TM-30-20 §4.4 Eq. (57)
  result.Rg = compute_rg(result.test_avg, result.ref_avg);

  // Step 3: Per-bin local metrics
  // TM-30-20 §4.6-§4.8
  result.local =
      compute_local_bin_metrics(result.test_avg, result.ref_avg, delta_e, bins);

  // Step 4: CVG coordinates
  // TM-30-20 §4.5
  result.cvg =
      compute_cvg_coordinates(result.test_avg, result.ref_avg, jab_ref, bins);

  return result;
}

} // namespace tm30
