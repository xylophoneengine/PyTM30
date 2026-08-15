#pragma once

/// @file ciecam02.hpp
/// CIECAM02 forward transform -> CAM02-UCS J'a'b' coordinates.
///
/// Implements the 12-step CIECAM02 forward model with fixed TM-30-20
/// viewing conditions, producing CAM02-UCS coordinates for color
/// difference computation.
///
/// TM-30-20 S3.7: Color Space and Chromatic Adaptation Transformation

#include <array>
#include <cstddef>

#include "tm30/xyz.hpp" // XyzTriple

namespace tm30 {

/// Fixed viewing condition parameters for all TM-30-20 calculations.
///
/// TM-30-20 S3.7
struct ViewingConditions {
  // TM-30-20 S3.7
  double Yb = 20.0; // Background luminance (cd/m^2)

  // TM-30-20 S3.7
  double F = 1.0; // Surround parameter

  // TM-30-20 S3.7
  double Nc = 1.0; // Surround parameter

  // TM-30-20 S3.7
  double c = 0.69; // Surround parameter

  // TM-30-20 S3.7
  double LA = 100.0; // Luminance of adapting field (cd/m^2)

  // TM-30-20 S3.7
  double D = 1.0; // Degree of adaptation

  // TM-30-20 S3.7
  double Yw = 100.0; // Luminous reflectance of white
};

/// CAM02-UCS coordinates: J' (lightness), a' (red-green), b' (yellow-blue).
///
/// TM-30-20 S3.7.1 Eq. (48)-(50)
struct Cam02Ucs {
  double J_prime; // TM-30-20 S3.7.1 Eq. (48)
  double a_prime; // TM-30-20 S3.7.1 Eq. (49)
  double b_prime; // TM-30-20 S3.7.1 Eq. (50)
};

/// Run the CIECAM02 forward transform for a set of sample XYZ values
/// under a given adapting white point.
///
/// Computes the full 12-step CIECAM02 pipeline (TM-30-20 S3.7.1):
///   1. XYZ -> RGB via MCAT02
///   2. Chromatic adaptation (D=1)
///   3. Convert to HPE cone space
///   4. Luminance-level adaptation
///   5. Opponent channels a, b
///   6. Achromatic response A
///   7. Lightness J
///   8. Hue angle h
///   9. Eccentricity et
///  10. Temporary t
///  11. Chroma C, Colorfulness M
///  12. CAM02-UCS J', a', b'
///
/// @param xyz_white   Adapting white point XYZ (10-deg observer).
/// @param xyz_samples Array of 99 CES sample XYZ values.
///
/// @return Array of 99 Cam02Ucs values.
///
/// TM-30-20 S3.7.1
std::array<Cam02Ucs, 99>
ciecam02_forward(const XyzTriple &xyz_white,
                 const std::array<XyzTriple, 99> &xyz_samples);

/// Compute the achromatic response Aw for a given white point.
///
/// This is needed separately when computing J (step 7), since J depends
/// on the ratio A / Aw.
///
/// @param xyz_white  Adapting white point XYZ.
/// @return Achromatic response Aw.
///
/// TM-30-20 S3.7.1 Eq. (41), (44)
double ciecam02_compute_aw(const XyzTriple &xyz_white);

} // namespace tm30
