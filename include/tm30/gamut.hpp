#pragma once

/// @file gamut.hpp
/// TM-30-20 Gamut Area Index (Rg), local per-bin metrics, and CVG coordinates.
///
/// TM-30-20 §4.3: hue-angle bins and bin-averaged (a', b') coordinates
/// TM-30-20 §4.4: Gamut Index (Rg)
/// TM-30-20 §4.5: Color Vector Graphic (CVG)
/// TM-30-20 §4.6: Local Chroma Shift (Rcs,hj)
/// TM-30-20 §4.7: Local Hue Shift (Rhs,hj)
/// TM-30-20 §4.8: Local Color Fidelity (Rf,hj)

#include <array>
#include <cstddef>

#include "tm30/ciecam02.hpp" // Cam02Ucs
#include "tm30/hue_bins.hpp" // HueBins

namespace tm30 {

/// Bin-averaged J'a'b' coordinates (16 bins).
/// For each bin j, the arithmetic mean of the CAM02-UCS coordinates
/// of all CES assigned to that bin.
///
/// TM-30-20 §4.3: the closing paragraph specifies the per-bin arithmetic
/// mean of (a', b') for both conditions; §4.4 consumes the averages for
/// Rg. J' is a PyTM30 extension -- §4.3 specifies only a' and b'.
struct BinAverages {
  std::array<double, 16> J_prime; // Average J' per bin  // PyTM30 extension
  std::array<double, 16> a_prime; // Average a' per bin  // TM-30-20 §4.3
  std::array<double, 16> b_prime; // Average b' per bin  // TM-30-20 §4.3
};

/// Per-bin local metrics.
///
/// TM-30-20 §4.6, §4.7, §4.8
struct LocalBinMetrics {
  std::array<double, 16> Rf_hj;  // Local fidelity per bin   // TM-30-20 §4.8
  // Local chroma shift per bin, as a PERCENTAGE. Eq. (62) computes a
  // ratio; §4.6 requires percentage representation (Table E-1 range is
  // roughly -100% to 100%). The x100 is applied exactly once, at struct
  // fill in compute_local_bin_metrics. TM-30-20 §4.6
  std::array<double, 16> Rcs_hj_percent;
  // Local hue shift per bin, dimensionless RATIO. §4.7 states no
  // percentage requirement; Table E-1 range is roughly -1 to 1 and the
  // Annex D templates print bare decimals. TM-30-20 §4.7
  std::array<double, 16> Rhs_hj;
  std::array<double, 16> DE_hj;  // Mean dE' per bin        // TM-30-20 §4.8
};

/// CVG-normalized bin-average coordinates.
///
/// The reference polygon is normalized to a unit circle.
/// Test vector endpoints are offset from the reference circle by the
/// (a',b')-plane displacement, scaled by the reference radial distance.
///
/// TM-30-20 §4.5
struct CvgCoordinates {
  /// Test CES bin-averaged J'a'b', normalized per CVG §4.5.
  std::array<double, 16> J_test; // J' unchanged               // TM-30-20 §4.5
  std::array<double, 16>
      x_test; // CVG x (horizontal axis)    // TM-30-20 §4.5 Eq. (60)
  std::array<double, 16>
      y_test; // CVG y (vertical axis)      // TM-30-20 §4.5 Eq. (61)

  /// Reference CES bin-averaged J'a'b', normalized per CVG §4.5.
  std::array<double, 16> J_ref; // J' unchanged               // TM-30-20 §4.5
  std::array<double, 16>
      x_ref; // Reference circle x (cos)   // TM-30-20 §4.5 Eq. (58)
  std::array<double, 16>
      y_ref; // Reference circle y (sin)   // TM-30-20 §4.5 Eq. (59)
};

/// Complete gamut result: Rg, per-bin local metrics, bin averages, and CVG.
///
/// TM-30-20 §4.3-§4.8
struct GamutResult {
  double Rg;             // Gamut area index         // TM-30-20 §4.4 Eq. (57)
  BinAverages test_avg;  // Test bin averages       // TM-30-20 §4.3
  BinAverages ref_avg;   // Reference bin averages  // TM-30-20 §4.3
  LocalBinMetrics local; // Per-bin local metrics   // TM-30-20 §4.6-§4.8
  CvgCoordinates cvg;    // CVG coordinates         // TM-30-20 §4.5
};

/// Compute bin-averaged J'a'b' for all 16 bins.
///
/// For each bin j, compute the arithmetic mean of all CES assigned to it.
/// Skips empty bins (stores NaN for averages).
///
/// @param jab_ces  CAM02-UCS coordinates for all 99 CES.
/// @param bins     16 hue-angle bins with CES indices.
/// @return         BinAverages with per-bin average J', a', b'.
///
/// TM-30-20 §4.3 (closing paragraph); §4.4 consumes the averages for Rg.
BinAverages bin_average(const std::array<Cam02Ucs, 99> &jab_ces,
                        const HueBins &bins);

/// Compute the polygon area in the (a', b') plane using the shoelace formula.
///
/// Vertices are the 16 bin-averaged (a', b') points in counterclockwise
/// bin order (0->15). Empty bins are skipped.
///
/// @param avg  Bin-averaged a', b' coordinates.
/// @return     Polygon area (positive).
///
/// TM-30-20 §4.4
double polygon_area(const BinAverages &avg);

/// Compute the Gamut Area Index Rg.
///
/// Rg = 100 x Area_test / Area_ref
///
/// @param test_avg  Bin-averaged test coordinates.
/// @param ref_avg   Bin-averaged reference coordinates.
/// @return          Rg value.
///
/// TM-30-20 §4.4 Eq. (57)
double compute_rg(const BinAverages &test_avg, const BinAverages &ref_avg);

/// Compute per-bin local metrics: Rf,hj, Rcs,hj, Rhs,hj, DE_hj.
///
/// Requires the bin-averaged coordinates (test and reference), the per-CES
/// dE' array, and the hue bin assignments.
///
/// @param test_avg     Bin-averaged test coordinates.
/// @param ref_avg      Bin-averaged reference coordinates.
/// @param delta_e      Per-CES dE' values (99).
/// @param bins         Hue bin assignments.
/// @return             LocalBinMetrics for all 16 bins.
///
/// TM-30-20 §4.6, §4.7, §4.8
LocalBinMetrics compute_local_bin_metrics(const BinAverages &test_avg,
                                          const BinAverages &ref_avg,
                                          const std::array<double, 99> &delta_e,
                                          const HueBins &bins);

/// Compute CVG-normalized bin-average coordinates.
///
/// Reference polygon is normalized to a unit circle centered at origin.
/// The reference circle position uses hbar_ref,j = mean of individual CES
/// hue angles per bin (§4.5). Test coordinates are offset from reference
/// circle points using the bin-averaged (a',b') displacement scaled by
/// the reference radial distance.
///
/// @param test_avg  Bin-averaged test coordinates.
/// @param ref_avg   Bin-averaged reference coordinates.
/// @param jab_ref   CAM02-UCS for reference illuminant (for mean hue angles).
/// @param bins      Hue-angle bin assignments.
/// @return          CVG coordinates for test and reference.
///
/// TM-30-20 §4.5 Eq. (58)-(61)
CvgCoordinates compute_cvg_coordinates(const BinAverages &test_avg,
                                       const BinAverages &ref_avg,
                                       const std::array<Cam02Ucs, 99> &jab_ref,
                                       const HueBins &bins);

/// Compute all gamut metrics: Rg, per-bin local metrics, and CVG.
///
/// This is the main entry point for Slice 9, combining all sub-computations.
///
/// @param jab_test    CAM02-UCS J'a'b' for test source (99 CES).
/// @param jab_ref     CAM02-UCS J'a'b' for reference illuminant (99 CES).
/// @param delta_e     Per-CES dE' values (99).
/// @param bins        Hue-angle bin assignments (16 bins).
/// @return            Complete GamutResult.
///
/// TM-30-20 §4.3-§4.8
GamutResult compute_gamut(const std::array<Cam02Ucs, 99> &jab_test,
                          const std::array<Cam02Ucs, 99> &jab_ref,
                          const std::array<double, 99> &delta_e,
                          const HueBins &bins);

} // namespace tm30
