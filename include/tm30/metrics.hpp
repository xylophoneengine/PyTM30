#pragma once

/// @file metrics.hpp
/// TM-30-20 color difference (dE') and fidelity index (Rf).
///
/// TM-30-20 §3.8: Color Difference Formula
/// TM-30-20 §4.1: Fidelity Index (Rf)

#include <array>
#include <cstddef>

#include "tm30/ciecam02.hpp" // Cam02Ucs

namespace tm30 {

/// Result of the Rf computation.
///
/// TM-30-20 §4.1
struct RfResult {
  double Rf_prime; // Raw fidelity before log rescale  // TM-30-20 §4.1 Eq. (53)
  double Rf;       // Final fidelity (0 to 100)        // TM-30-20 §4.1 Eq. (54)
  double delta_e_avg; // Average color difference         // TM-30-20 §4.1
};

/// Compute the CAM02-UCS color difference dE' for each CES.
///
/// For each CES i:
///   dE'_i = sqrt[(J't,i - J'r,i)^2 + (a't,i - a'r,i)^2 + (b't,i - b'r,i)^2]
///
/// @param jab_test  Test-source CAM02-UCS coordinates (99 samples).
/// @param jab_ref   Reference-source CAM02-UCS coordinates (99 samples).
/// @return          Array of 99 dE' values.
///
/// TM-30-20 §3.8 Eq. (52)
std::array<double, 99> compute_delta_e(const std::array<Cam02Ucs, 99> &jab_test,
                                       const std::array<Cam02Ucs, 99> &jab_ref);

/// Compute the Fidelity Index Rf from the array of dE' values.
///
/// Steps:
///   1. dE_avg = mean of 99 dE' values                    // TM-30-20 §4.1
///   2. Rf' = 100 - 6.73 * dE_avg                         // TM-30-20 §4.1 Eq.
///   (53)
///   3. Rf = 10 * ln(exp(Rf' / 10) + 1)                   // TM-30-20 §4.1 Eq.
///   (54)
///
/// @param delta_e_array  Array of 99 dE' values.
/// @return               RfResult with Rf_prime, Rf, and delta_e_avg.
///
/// TM-30-20 §4.1 Eq. (53), (54)
RfResult compute_rf(const std::array<double, 99> &delta_e_array);

/// Compute per-sample color fidelity Rf,CESi for each CES.
///
/// For each CES i:
///   Rf,CESi' = 100 - 6.73 * dE'_i              // TM-30-20 §4.2 Eq. (55)
///   Rf,CESi  = 10 * ln(exp(Rf,CESi' / 10) + 1) // TM-30-20 §4.2 Eq. (56)
///
/// TM-30-20 §4.2
std::array<double, 99>
compute_rf_cesi(const std::array<double, 99> &delta_e_array);

/// Compute skin fidelity Rf,skin from per-sample Rf,CESi values.
///
/// Rf,skin = (Rf,CES15 + Rf,CES18) / 2   (0-indexed: indices 14 and 17)
///
/// PyTM30 research extension informed by TM-30-20 §4.2 (the CES15/CES18
/// mean correlates with the broader skin-sample set). Not one of the
/// standard's 50 measures (§1.2); per §4.0 not identified as part of
/// the TM-30 method.
double compute_rf_skin(const std::array<double, 99> &rf_cesi);

/// Annex E priority levels for specifying light source color rendition.
///
/// P1 (Preferred): Rf, Rg, CVG
/// P2 (Secondary): Rcs,hj, Rhs,hj, Rf,hj
/// P3 (Ancillary): Rf,CESi (sample-specific)
///
/// TM-30-20 Annex E
struct AnnexE {
  static constexpr int P1 = 1; // TM-30-20 Annex E
  static constexpr int P2 = 2; // TM-30-20 Annex E
  static constexpr int P3 = 3; // TM-30-20 Annex E
};

} // namespace tm30
