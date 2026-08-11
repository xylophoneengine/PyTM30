// TM-30-20 §4.3: Hue-Angle Bin assignment.
// Divides the 99 CES into 16 equal hue-angle bins (22.5-deg each)
// based on reference CAM02-UCS hue angle hr = atan2(b'r, a'r).
//
// TM-30-20 §4.3 + Figure 3 (Annex B is graphics colors only -- RGB tables
// for bar charts / CVG vectors -- and contains no bin-assignment content).
#include "tm30/hue_bins.hpp"

#include <array>
#include <cmath> // atan2
#include <cstdint>
#include <numbers> // std::numbers::pi

namespace tm30 {

// TM-30-20 §4.3: 16 bins of 22.5-deg each in the a'-b' plane.
// Bin width in radians: 22.5-deg = pi/8 rad.
static constexpr double kBinWidth = 0.39269908169872414; // TM-30-20 §4.3

HueBins bin_by_hue(const std::array<Cam02Ucs, 99> &jab_ref) {
  HueBins bins;

  for (int i = 0; i < 99; ++i) {
    // TM-30-20 §4.3: hr = atan2(b'r,i, a'r,i) in [-pi, pi]
    double h = std::atan2(jab_ref[i].b_prime, jab_ref[i].a_prime);

    // Normalize to [0, 2pi)
    // TM-30-20 §4.3 edge case: atan2 returns [-pi, pi]; map to [0, 2pi)
    if (h < 0.0) {
      h += 2.0 * std::numbers::pi;
    }

    // Assign to bin. TM-30-20 §4.3:
    // Bin j (0-indexed) spans [j x 22.5-deg, (j+1) x 22.5-deg).
    // Half-open intervals: boundary value goes to the bin starting at that
    // angle (higher bin index). Bin 15 spans [337.5-deg, 360.0-deg] inclusive.
    int bin = static_cast<int>(h / kBinWidth); // TM-30-20 §4.3
    if (std::isnan(h) || bin < 0) {
      // Degenerate input (e.g. an all-zero SPD) propagates NaN through
      // atan2; static_cast<int>(NaN) is UB and yields INT_MIN on x86-64,
      // which would index bins[] out of bounds. Clamp to bin 0 - the
      // result is garbage-in/garbage-out anyway (the pipeline flags such
      // SPDs via Validity).
      bin = 0;
    } else if (bin >= 16) {
      bin = 15; // h ~= 2pi -> bin 15 (0-indexed)  // TM-30-20 §4.3
    }

    bins[bin].push_back(i);
  }

  return bins;
}

} // namespace tm30
