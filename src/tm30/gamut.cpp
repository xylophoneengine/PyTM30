// TM-30-20: Gamut Area Index (Rg), per-bin local metrics, and CVG coordinates.
//
// TM-30-20 §4.4: Gamut Index (Rg)
// TM-30-20 §4.5: Color Vector Graphic (CVG)
// TM-30-20 §4.6: Local Chroma Shift (Rcs,hj)
// TM-30-20 §4.7: Local Hue Shift (Rhs,hj)
// TM-30-20 §4.8: Local Color Fidelity (Rf,hj)
#include "tm30/gamut.hpp"

#include <cmath>
#include <limits>

namespace tm30 {

// ─── Constants ────────────────────────────────────────────────────────────

// TM-30-20 §4.3: 16 bins of 22.5° each.
static constexpr int kNumBins = 16; // TM-30-20 §4.3

// TM-30-20 §4.3: Bin width in radians (22.5° = π/8).
static constexpr double kBinWidthDeg = 22.5; // TM-30-20 §4.3

// TM-30-20 §4.1: Rf scaling factor.
static constexpr double kRfScale = 6.73; // TM-30-20 §4.1 Eq. (53)

// TM-30-20 §4.1: Log rescale divisor.
static constexpr double kLogRescale = 10.0; // TM-30-20 §4.1 Eq. (54)

// TM-30-20 §4.6, §4.7: Local shift scaling.
// Note: Eq. (62) and (63) in the spec include a factor of 100 to convert
// to percentage, but luxpy (the oracle) stores unscaled values.
// We match the oracle convention.
static constexpr double kLocalShiftScale = 1.0; // TM-30-20 §4.6, §4.7

// TM-30-20 §4.5: CVG display scaling factor.
static constexpr double kCvgScale = 100.0; // TM-30-20 §4.5

// ─── Bin Averages ─────────────────────────────────────────────────────────

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

// ─── Polygon Area (Shoelace) ──────────────────────────────────────────────

double polygon_area(const BinAverages &avg) {
  // TM-30-20 §4.4: Shoelace formula in (a', b') plane

  // Collect non-NaN vertices in bin order (0→15).
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

  // Shoelace: A = 0.5 * |Σ (x_i * y_{i+1} - x_{i+1} * y_i)|
  // TM-30-20 §4.4
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    const int next = (i + 1) % n; // wrap-around
    sum += verts[i].a * verts[next].b - verts[next].a * verts[i].b;
  }

  return 0.5 * std::abs(sum); // TM-30-20 §4.4
}

// ─── Rg ───────────────────────────────────────────────────────────────────

double compute_rg(const BinAverages &test_avg, const BinAverages &ref_avg) {
  // TM-30-20 §4.4 Eq. (57)
  const double A_test = polygon_area(test_avg); // TM-30-20 §4.4
  const double A_ref = polygon_area(ref_avg);   // TM-30-20 §4.4
  return 100.0 * A_test / A_ref;                // TM-30-20 §4.4 Eq. (57)
}

// ─── Local Bin Metrics ────────────────────────────────────────────────────

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
      // TM-30-20 §4.6–§4.8 edge case
      metrics.Rf_hj[j] = std::numeric_limits<double>::quiet_NaN();
      metrics.Rcs_hj[j] = std::numeric_limits<double>::quiet_NaN();
      metrics.Rhs_hj[j] = std::numeric_limits<double>::quiet_NaN();
      metrics.DE_hj[j] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    // ── Mean ΔE′ per bin ───────────────────────────────────────
    // TM-30-20 §4.8
    double sum_de = 0.0;
    for (int idx : bin) {
      sum_de += delta_e[idx];
    }
    const double DE_hj = sum_de / static_cast<double>(m); // TM-30-20 §4.8
    metrics.DE_hj[j] = DE_hj;

    // ── Rf,hj ──────────────────────────────────────────────────
    // TM-30-20 §4.8 Eq. (64), (65)
    const double Rf_hj_prime =
        100.0 - kRfScale * DE_hj; // TM-30-20 §4.8 Eq. (64)
    metrics.Rf_hj[j] =
        kLogRescale * std::log( // TM-30-20 §4.8 Eq. (65)
                          std::exp(Rf_hj_prime / kLogRescale) + 1.0);

    // ── Rcs,hj and Rhs,hj ──────────────────────────────────────
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

    // Bin bisector angle: θj = (j + 0.5) × 22.5° (0-indexed)
    // TM-30-20 §4.6: θj is the bisector angle of bin j
    const double theta_deg = (static_cast<double>(j) + 0.5) * kBinWidthDeg;
    const double theta = theta_deg * M_PI / 180.0; // TM-30-20 §4.6

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

// ─── CVG Coordinates ──────────────────────────────────────────────────────

CvgCoordinates
compute_cvg_coordinates(const BinAverages &test_avg, const BinAverages &ref_avg,
                        const std::array<Cam02Ucs, 99> & /*jab_ref*/,
                        const HueBins & /*bins*/) {

  // TM-30-20 §4.5
  CvgCoordinates cvg{};

  for (int j = 0; j < kNumBins; ++j) {
    // TM-30-20 §4.5: J' passes through unchanged
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

    // Reference hue angle from bin-averaged a', b'
    // TM-30-20 §4.5: h̄_ref,j = atan2(b̄'r,j, ā'r,j)
    // The spec calls for mean of individual hue angles, but the
    // oracle (luxpy) uses the bin-averaged coordinates directly.
    const double h_bar = std::atan2(ref_avg.b_prime[j], ref_avg.a_prime[j]);

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

// ─── Main compute_gamut ───────────────────────────────────────────────────

GamutResult compute_gamut(const std::array<Cam02Ucs, 99> &jab_test,
                          const std::array<Cam02Ucs, 99> &jab_ref,
                          const std::array<double, 99> &delta_e,
                          const HueBins &bins) {

  // TM-30-20 §4.4–§4.8
  GamutResult result{};

  // Step 1: Bin-averaged J'a'b'
  // TM-30-20 §4.4
  result.test_avg = bin_average(jab_test, bins);
  result.ref_avg = bin_average(jab_ref, bins);

  // Step 2: Rg
  // TM-30-20 §4.4 Eq. (57)
  result.Rg = compute_rg(result.test_avg, result.ref_avg);

  // Step 3: Per-bin local metrics
  // TM-30-20 §4.6–§4.8
  result.local =
      compute_local_bin_metrics(result.test_avg, result.ref_avg, delta_e, bins);

  // Step 4: CVG coordinates
  // TM-30-20 §4.5
  result.cvg =
      compute_cvg_coordinates(result.test_avg, result.ref_avg, jab_ref, bins);

  return result;
}

} // namespace tm30
