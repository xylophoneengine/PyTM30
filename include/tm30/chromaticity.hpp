#pragma once

/// @file chromaticity.hpp
/// CIE 1960 UCS (u, v) and CIE 1976 Y,u',v' chromaticity coordinates.
///
/// TM-30-20 §3.3: CCT determination uses CIE 1960 UCS (u, v) diagram.
/// The CIE 1960 UCS is also referred to as the Judd 1960 UCS.
///
/// CIE 1976 UCS (u', v') is the chromaticity space of CIELUV,
/// defined in CIE 15:2004 §8.2.1.

namespace tm30 {

/// CIE 1960 UCS (u, v) chromaticity coordinates.
struct UvCoord {
  double u; // CIE 1960 UCS u
  double v; // CIE 1960 UCS v
};

/// CIE 1976 Y,u',v' chromaticity coordinates.
///
/// Y is the CIE 1931 luminance (cd/m^2 for absolute, or
/// luminance factor Y/Yn for relative).
struct YuvTriple {
  double Y;       // CIE 1931 Y (luminance or luminance factor)
  double u_prime; // CIE 1976 UCS u'
  double v_prime; // CIE 1976 UCS v'
};

/// Convert CIE XYZ tristimulus values to CIE 1960 UCS (u, v).
///
/// TM-30-20 §3.3:
///   u = 4X / (X + 15Y + 3Z)
///   v = 6Y / (X + 15Y + 3Z)
///
/// @param X  CIE XYZ X tristimulus value.
/// @param Y  CIE XYZ Y tristimulus value.
/// @param Z  CIE XYZ Z tristimulus value.
/// @return   UvCoord with u, v chromaticity coordinates.
UvCoord xyz_to_uv(double X, double Y, double Z);

/// Convert CIE XYZ tristimulus values to CIE 1976 Y,u',v'.
///
/// CIE 15:2004 §8.2.1:
///   u' = 4X / (X + 15Y + 3Z)
///   v' = 9Y / (X + 15Y + 3Z)
///
/// @param X  CIE XYZ X tristimulus value.
/// @param Y  CIE XYZ Y tristimulus value.
/// @param Z  CIE XYZ Z tristimulus value.
/// @return   YuvTriple with Y, u', v'.
YuvTriple xyz_to_Yuv(double X, double Y, double Z);

} // namespace tm30
