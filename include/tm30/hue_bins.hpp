#pragma once

/// @file hue_bins.hpp
/// TM-30-20 S4.3: Hue-Angle Bin assignment for 99 CES.
///
/// Divides the a'-b' plane into 16 equal hue-angle bins of 22.5-deg each,
/// starting at 0-deg (positive a' axis). Each CES is assigned to a bin
/// based on its reference hue angle hr = atan2(b'r, a'r).

#include <array>
#include <cmath>   // atan2
#include <numbers> // std::numbers::pi
#include <vector>

#include "tm30/ciecam02.hpp" // Cam02Ucs

namespace tm30 {

/// 16 hue-angle bins, each containing the CES indices assigned to it.
///
/// Bin j (0-indexed, j = 0...15) spans [(j x 22.5-deg), ((j+1) x 22.5-deg)) in
/// the a'-b' plane. Bin 15 spans [337.5-deg, 360.0-deg] inclusive.
///
/// TM-30-20 S4.3, Figure 3
using HueBins = std::array<std::vector<int>, 16>;

/// Reference hue angles hr for the 99 CES, in radians, normalized to
/// [0, 2pi) - one per CES, indexed by 0-based CES index.
///
/// TM-30-20 S4.3
using HueAngles = std::array<double, 99>;

/// The reference hue angle hr = atan2(b'r, a'r) of one sample, normalized
/// to [0, 2pi).
///
/// This is the ONLY definition of that angle in the library. Both
/// bin_by_hue() (TM-30-20 S4.3, to pick the bin) and
/// compute_cvg_coordinates() (TM-30-20 S4.5 Eqs. (58)-(59), to average the
/// angles within a bin) go through it, which is what makes reusing a
/// binning-time angle in the CVG step bit-identical to recomputing it
/// there rather than merely close.
///
/// NaN in: atan2 propagates NaN and the comparison below is false, so NaN
/// comes back out unchanged - the caller decides what that means (see
/// bin_by_hue's degenerate-input clamp).
///
/// TM-30-20 S4.3
inline double reference_hue_angle(const Cam02Ucs &jab) {
  // TM-30-20 S4.3: hr = atan2(b'r,i, a'r,i) in [-pi, pi]
  double h = std::atan2(jab.b_prime, jab.a_prime);

  // Normalize to [0, 2pi)
  // TM-30-20 S4.3 edge case: atan2 returns [-pi, pi]; map to [0, 2pi)
  if (h < 0.0) {
    h += 2.0 * std::numbers::pi; // TM-30-20 S4.3: full turn
  }
  return h;
}

/// Assign 99 CES to 16 hue-angle bins based on reference hue angle.
///
/// For each CES i (0-indexed):
///   hr = atan2(b'r,i, a'r,i)    // in radians, normalized to [0, 2pi)
///   bin = floor(hr / (pi/8))     // 22.5-deg = pi/8 rad per bin
///   Clamp to 15 to handle hr -> 2pi (== 0-deg)
///
/// Boundary tie-break per TM-30-20 S4.3:
///   Half-open intervals [start, end) - the boundary value belongs to
///   the bin starting at that angle (higher bin index).
///   Bin 16 (0-indexed: 15) spans [337.5-deg, 360.0-deg] inclusive.
///
/// @param jab_ref  CAM02-UCS J'a'b' coordinates under reference illuminant (99
/// CES).
/// @param out_hue_angles
///                 Optional output: the normalized hue angles hr this
///                 function computes on its way to the bin assignment,
///                 one per CES. Supplying it costs one store per CES and
///                 lets compute_cvg_coordinates() (S4.5 Eqs. (58)-(59))
///                 skip recomputing the identical 99 atan2 calls. Purely
///                 a performance affordance with no effect on the bins.
///                 Defaults to null, preserving existing behavior for all
///                 current callers.
/// @return 16 vectors of 0-based CES indices, one per hue-angle bin.
///
/// TM-30-20 S4.3 + Figure 3
HueBins bin_by_hue(const std::array<Cam02Ucs, 99> &jab_ref,
                   HueAngles *out_hue_angles = nullptr);

} // namespace tm30
