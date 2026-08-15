// CIE 1960 UCS (u, v) and CIE 1976 Y,u',v' from XYZ.
// TM-30-20 S3.3: CCT determination
// CIE 15:2004 S8.2.1: CIE 1976 UCS
#include "tm30/chromaticity.hpp"

namespace tm30 {

UvCoord xyz_to_uv(double X, double Y, double Z) {
  // TM-30-20 S3.3: CIE 1960 UCS transformation
  // u = 4X / (X + 15Y + 3Z)
  // v = 6Y / (X + 15Y + 3Z)
  const double denom = X + 15.0 * Y + 3.0 * Z;
  return UvCoord{
      4.0 * X / denom, // TM-30-20 S3.3: u = 4X / (X + 15Y + 3Z)
      6.0 * Y / denom  // TM-30-20 S3.3: v = 6Y / (X + 15Y + 3Z)
  };
}

YuvTriple xyz_to_Yuv(double X, double Y, double Z) {
  // TM-30-20 S3.3: denominator X + 15Y + 3Z (shared with CIE 1960 UCS).
  // CIE 15:2004 S8.2.1: CIE 1976 UCS uses same u' = u but v' = 9Y/(X+15Y+3Z).
  const double denom = X + 15.0 * Y + 3.0 * Z; // TM-30-20 S3.3 denominator
  return YuvTriple{
      Y, // Y passes through unchanged
      4.0 * X /
          denom, // TM-30-20 S3.3: u' = 4X / (X + 15Y + 3Z) [same as CIE 1960 u]
      9.0 * Y /
          denom // TM-30-20 S3.3 / CIE 15:2004 S8.2.1: v' = 9Y / (X + 15Y + 3Z)
  };
}

} // namespace tm30
