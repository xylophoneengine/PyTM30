#pragma once

/// @file hue_bins.hpp
/// TM-30-20 S4.3: Hue-Angle Bin assignment for 99 CES.
///
/// Divides the a'-b' plane into 16 equal hue-angle bins of 22.5-deg each,
/// starting at 0-deg (positive a' axis). Each CES is assigned to a bin
/// based on its reference hue angle hr = atan2(b'r, a'r).

#include <array>
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
/// @return 16 vectors of 0-based CES indices, one per hue-angle bin.
///
/// TM-30-20 S4.3 + Figure 3
HueBins bin_by_hue(const std::array<Cam02Ucs, 99> &jab_ref);

} // namespace tm30
