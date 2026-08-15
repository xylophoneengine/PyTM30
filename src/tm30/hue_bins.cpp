// TM-30-20 S4.3: Hue-Angle Bin assignment.
// Divides the 99 CES into 16 equal hue-angle bins (22.5-deg each)
// based on reference CAM02-UCS hue angle hr = atan2(b'r, a'r).
//
// TM-30-20 S4.3 + Figure 3 (Annex B is graphics colors only -- RGB tables
// for bar charts / CVG vectors -- and contains no bin-assignment content).
#include "tm30/hue_bins.hpp"

#include <array>
#include <cmath> // isnan
#include <cstdint>

namespace tm30 {

// TM-30-20 S4.3: 16 bins of 22.5-deg each in the a'-b' plane.
// Bin width in radians: 22.5-deg = pi/8 rad.
static constexpr double kBinWidth = 0.39269908169872414; // TM-30-20 S4.3

HueBins bin_by_hue(const std::array<Cam02Ucs, 99> &jab_ref,
                   HueAngles *out_hue_angles) {
  HueBins bins;

  for (int i = 0; i < 99; ++i) {
    // TM-30-20 S4.3: hr = atan2(b'r,i, a'r,i) in [-pi, pi], normalized to
    // [0, 2pi). reference_hue_angle() (hue_bins.hpp) is the single
    // definition of that expression; compute_cvg_coordinates() consumes
    // the very same values for S4.5 Eqs. (58)-(59).
    const double h = reference_hue_angle(jab_ref[i]);

    // Hand the angle back if the caller asked for it, so the CVG step does
    // not have to recompute all 99 atan2 calls. Storing and reloading a
    // double is exact, so the two paths agree bit-for-bit - asserted in
    // tests/slice_09_rg_local_cvg_test.cpp.
    if (out_hue_angles != nullptr) {
      (*out_hue_angles)[i] = h;
    }

    // Assign to bin. TM-30-20 S4.3:
    // Bin j (0-indexed) spans [j x 22.5-deg, (j+1) x 22.5-deg).
    // Half-open intervals: boundary value goes to the bin starting at that
    // angle (higher bin index). Bin 15 spans [337.5-deg, 360.0-deg] inclusive.
    int bin = static_cast<int>(h / kBinWidth); // TM-30-20 S4.3
    if (std::isnan(h) || bin < 0) {
      // Degenerate input (e.g. an all-zero SPD) propagates NaN through
      // atan2; static_cast<int>(NaN) is UB and yields INT_MIN on x86-64,
      // which would index bins[] out of bounds. Clamp to bin 0 - the
      // result is garbage-in/garbage-out anyway (the pipeline flags such
      // SPDs via Validity).
      bin = 0;
    } else if (bin >= 16) {
      bin = 15; // h ~= 2pi -> bin 15 (0-indexed)  // TM-30-20 S4.3
    }

    bins[bin].push_back(i);
  }

  return bins;
}

} // namespace tm30
