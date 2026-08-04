// TM-30-20 §4.3: Hue-Angle Bin assignment.
// Divides the 99 CES into 16 equal hue-angle bins (22.5° each)
// based on reference CAM02-UCS hue angle hr = atan2(b'r, a'r).
//
// TM-30-20 §4.3, Annex B
#include "tm30/hue_bins.hpp"

#include <cmath> // atan2, M_PI
#include <cstdint>

namespace tm30 {

// TM-30-20 §4.3: 16 bins of 22.5° each in the a'-b' plane.
// Bin width in radians: 22.5° = π/8 rad.
static constexpr double kBinWidth = 0.39269908169872414; // TM-30-20 §4.3

HueBins bin_by_hue(const std::array<Cam02Ucs, 99> &jab_ref) {
  HueBins bins;

  for (int i = 0; i < 99; ++i) {
    // TM-30-20 §4.3: hr = atan2(b'r,i, a'r,i) in [-π, π]
    double h = std::atan2(jab_ref[i].b_prime, jab_ref[i].a_prime);

    // Normalize to [0, 2π)
    // TM-30-20 §4.3 edge case: atan2 returns [-π, π]; map to [0, 2π)
    if (h < 0.0) {
      h += 2.0 * M_PI;
    }

    // Assign to bin. TM-30-20 §4.3:
    // Bin j (0-indexed) spans [j × 22.5°, (j+1) × 22.5°).
    // Half-open intervals: boundary value goes to the bin starting at that
    // angle (higher bin index). Bin 15 spans [337.5°, 360.0°] inclusive.
    int bin = static_cast<int>(h / kBinWidth); // TM-30-20 §4.3
    if (std::isnan(h) || bin < 0) {
      // Degenerate input (e.g. an all-zero SPD) propagates NaN through
      // atan2; static_cast<int>(NaN) is UB and yields INT_MIN on x86-64,
      // which would index bins[] out of bounds. Clamp to bin 0 - the
      // result is garbage-in/garbage-out anyway (the pipeline flags such
      // SPDs via Validity).
      bin = 0;
    } else if (bin >= 16) {
      bin = 15; // h ≈ 2π → bin 15 (0-indexed)  // TM-30-20 §4.3
    }

    bins[bin].push_back(i);
  }

  return bins;
}

} // namespace tm30
