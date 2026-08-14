#pragma once
#include <cmath>

// Tolerances for TM-30-20 implementation verification.
// All values justified against oracle precision and spec requirements.
// DO NOT MODIFY without maintainer approval.
//
// This is the ONLY file containing tolerance epsilon values for the project.
// All tests #include this header and never define their own tolerances.

namespace tm30::test {

// -------------------------------------------------------------------------
// XYZ tristimulus: published tables (e.g. CIE 15, ASTM E308) go to
// 6 significant figures.  Absolute tolerance derived from rounding error
// at the 6th significant digit (half-ULP ~= 5e-5 for values around 100).
inline constexpr double Tol_Xyz = 5e-5; // absolute

// -------------------------------------------------------------------------
// CCT / Duv: Ohno 2013 validation tables quote CCT to 0.1 K and Duv to
// 0.0001.  Tolerance set to allow for one unit of reported precision
// plus solver noise.
inline constexpr double Tol_Cct = 0.5;  // absolute (K)
inline constexpr double Tol_Duv = 5e-4; // absolute

// -------------------------------------------------------------------------
// CIECAM02 J'a'b': CIE TC8-01 worked examples provide values to 2-3
// decimal places.  Tolerance set to accommodate the rounding resolution
// of those examples across all three dimensions.
inline constexpr double Tol_Jab = 1e-3; // absolute

// -------------------------------------------------------------------------
// Delta E' and Rf/Rg: TM-30 rounds Rf and Rg to integer in its reporting
// format; the oracle (luxpy) provides 1 decimal place.  Tolerance is
// set to 0.15 units - tight enough to detect pipeline drift, loose enough
// to absorb legitimate floating-point differences between implementations.
inline constexpr double Tol_DeltaE = 5e-3; // absolute per-single dE'
inline constexpr double Tol_Rf = 0.15;     // absolute
inline constexpr double Tol_Rg = 0.15;     // absolute

// -------------------------------------------------------------------------
// Local chroma/hue shifts. Rcs,hj is a percentage (TM-30-20 §4.6 requires
// percentage representation of the Eq. (62) ratio); tolerance matches its
// 1-decimal reporting resolution. Rhs,hj stays a ratio (§4.7 states no
// percentage requirement) and is two orders of magnitude smaller in
// scale, so its tolerance is Tol_LocalShift at ratio scale -- both checks
// equally tight relative to their units.
inline constexpr double Tol_LocalShift = 0.15;      // absolute (percent units)
inline constexpr double Tol_HueShiftRatio = 1.5e-3; // absolute (ratio units)

} // namespace tm30::test
