// CIECAM02 forward transform -> CAM02-UCS J'a'b' coordinates.
//
// Implements the 12-step CIECAM02 forward model with fixed
// TM-30-20 viewing conditions (TM-30-20 §3.7).
//
// TM-30-20 §3.7: Color Space and Chromatic Adaptation Transformation
// TM-30-20 §3.7.1: Procedure

#include "tm30/ciecam02.hpp"

#include <algorithm>
#include <cmath>
#include <numbers> // std::numbers::pi

namespace tm30 {
namespace {

// -------------------------------------------------------------------------
// Fixed viewing condition parameters (TM-30-20 §3.7)
// -------------------------------------------------------------------------

// TM-30-20 §3.7
inline constexpr double kYb = 20.0;

// TM-30-20 §3.7
inline constexpr double kF = 1.0;

// TM-30-20 §3.7
inline constexpr double kNc = 1.0;

// TM-30-20 §3.7
inline constexpr double kc = 0.69;

// TM-30-20 §3.7
inline constexpr double kLA = 100.0;

// TM-30-20 §3.7
inline constexpr double kYw = 100.0;

// -------------------------------------------------------------------------
// Derived constants (TM-30-20 §3.7)
// -------------------------------------------------------------------------

// k = 1 / (5*LA + 1)
// TM-30-20 §3.7
inline constexpr double kK = 1.0 / (5.0 * kLA + 1.0);
// = 0.0020 (approx)

// n = Yb / Yw
// TM-30-20 §3.7
inline constexpr double kN = kYb / kYw;
// = 0.2000

// Nbb = Ncb = 0.725 * n^(-0.2)
// TM-30-20 §3.7
inline const double kNbb =
    0.725 *
    std::pow(
        kN,
        -0.2); // not constexpr: std::pow isn't constant-evaluable everywhere
// = 1.0003 (approx)

// z = 1.48 + sqrt(n)
// TM-30-20 §3.7
inline const double kZ = 1.48 + std::sqrt(kN); // not constexpr: see kNbb above
// = 1.9272 (approx)

// FL = (1/5)*k^4*(5*LA) + (1/10)*(1 - k^4)^2 * (5*LA)^(1/3)
// TM-30-20 §3.7
inline double compute_FL() { // not constexpr: see kNbb above
  const double k4 = kK * kK * kK * kK;
  const double five_LA = 5.0 * kLA;
  const double term1 = 0.2 * k4 * five_LA;
  const double term2 = 0.1 * (1.0 - k4) * (1.0 - k4) * std::cbrt(five_LA);
  return term1 + term2;
}
inline const double kFL = compute_FL();
// = 0.7937 (approx)

// -------------------------------------------------------------------------
// CAT02 matrix MCAT02 (Eq. 30, TM-30-20 §3.7.1)
// -------------------------------------------------------------------------

// Row 0: [0.7328,  0.4296, -0.1624]
// TM-30-20 §3.7.1 Eq. (30)
inline constexpr double kMCAT02_00 = 0.7328;
// TM-30-20 §3.7.1 Eq. (30)
inline constexpr double kMCAT02_01 = 0.4296;
// TM-30-20 §3.7.1 Eq. (30)
inline constexpr double kMCAT02_02 = -0.1624;

// Row 1: [-0.7036, 1.6975,  0.0061]
// TM-30-20 §3.7.1 Eq. (30)
inline constexpr double kMCAT02_10 = -0.7036;
// TM-30-20 §3.7.1 Eq. (30)
inline constexpr double kMCAT02_11 = 1.6975;
// TM-30-20 §3.7.1 Eq. (30)
inline constexpr double kMCAT02_12 = 0.0061;

// Row 2: [0.0030,  0.0136,  0.9834]
// TM-30-20 §3.7.1 Eq. (30)
inline constexpr double kMCAT02_20 = 0.0030;
// TM-30-20 §3.7.1 Eq. (30)
inline constexpr double kMCAT02_21 = 0.0136;
// TM-30-20 §3.7.1 Eq. (30)
inline constexpr double kMCAT02_22 = 0.9834;

// -------------------------------------------------------------------------
// Combined matrix M = MHPE * MCAT02^(-1)  (Eq. 34-35, TM-30-20 §3.7.1)
//
// Precomputed from:
//   MHPE = [[0.38971, 0.68898, -0.07868],   // Eq. (35)
//           [-0.22981, 1.18340, 0.04641],
//           [0.0, 0.0, 1.0]]
//   MCAT02 = [[0.7328, 0.4296, -0.1624],    // Eq. (30)
//             [-0.7036, 1.6975, 0.0061],
//             [0.0030, 0.0136, 0.9834]]
// -------------------------------------------------------------------------

// Row 0: MHPE * MCAT02^(-1)
// TM-30-20 §3.7.1 Eq. (34)-(35)
inline constexpr double kCOMBINED_00 = 0.7409790970135308;
// TM-30-20 §3.7.1 Eq. (34)-(35)
inline constexpr double kCOMBINED_01 = 0.2180251556757357;
// TM-30-20 §3.7.1 Eq. (34)-(35)
inline constexpr double kCOMBINED_02 = 0.0410057473107336;

// Row 1
// TM-30-20 §3.7.1 Eq. (34)-(35)
inline constexpr double kCOMBINED_10 = 0.2853532916858802;
// TM-30-20 §3.7.1 Eq. (34)-(35)
inline constexpr double kCOMBINED_11 = 0.6242015741188158;
// TM-30-20 §3.7.1 Eq. (34)-(35)
inline constexpr double kCOMBINED_12 = 0.0904451341953042;

// Row 2
// TM-30-20 §3.7.1 Eq. (34)-(35)
inline constexpr double kCOMBINED_20 = -0.0096276087384294;
// TM-30-20 §3.7.1 Eq. (34)-(35)
inline constexpr double kCOMBINED_21 = -0.0056980312161134;
// TM-30-20 §3.7.1 Eq. (34)-(35)
inline constexpr double kCOMBINED_22 = 1.0153256399545427;

// -------------------------------------------------------------------------
// Step 4 constants (TM-30-20 §3.7.1 Eq. (36)-(38))
// -------------------------------------------------------------------------

// TM-30-20 §3.7.1 Eq. (36)-(38)
inline constexpr double kLumScale = 400.0;

// TM-30-20 §3.7.1 Eq. (36)-(38)
inline constexpr double kLumExp = 0.42;

// TM-30-20 §3.7.1 Eq. (36)-(38)
inline constexpr double kLumAdd = 27.13;

// TM-30-20 §3.7.1 Eq. (36)-(38)
inline constexpr double kLumOffset = 0.1;

// -------------------------------------------------------------------------
// Step 6 constant (TM-30-20 §3.7.1 Eq. (44))
// -------------------------------------------------------------------------

// TM-30-20 §3.7.1 Eq. (44)
inline constexpr double kAOffset = 0.305;

// -------------------------------------------------------------------------
// Step 7 constants (TM-30-20 §3.7.1 Eq. (41))
// -------------------------------------------------------------------------

// TM-30-20 §3.7.1 Eq. (41)
inline constexpr double kJScale = 100.0;

// -------------------------------------------------------------------------
// Step 8: atan2 -> degrees
// -------------------------------------------------------------------------

// TM-30-20 §3.7.1 Eq. (45)
inline constexpr double kRadToDeg = 180.0 / std::numbers::pi;

// -------------------------------------------------------------------------
// Step 9 constants (TM-30-20 §3.7.1 Eq. (47))
// -------------------------------------------------------------------------

// TM-30-20 §3.7.1 Eq. (47)
inline constexpr double kEtScale = 0.25;

// TM-30-20 §3.7.1 Eq. (47)
inline constexpr double kEtPhase = 2.0;

// TM-30-20 §3.7.1 Eq. (47)
inline constexpr double kEtBaseline = 3.8;

// -------------------------------------------------------------------------
// Step 10 constant (TM-30-20 §3.7.1 Eq. (46))
// -------------------------------------------------------------------------

// TM-30-20 §3.7.1 Eq. (46)
inline constexpr double kTFactor = 50000.0 / 13.0;

// -------------------------------------------------------------------------
// Step 11 constants (TM-30-20 §3.7.1 Eq. (42)-(43))
// -------------------------------------------------------------------------

// TM-30-20 §3.7.1 Eq. (42)
inline constexpr double kCExpT = 0.9;

// TM-30-20 §3.7.1 Eq. (42)
inline constexpr double kCParenA = 1.64;

// TM-30-20 §3.7.1 Eq. (42)
inline constexpr double kCParenB = 0.29;

// TM-30-20 §3.7.1 Eq. (42)
inline constexpr double kCExpOuter = 0.73;

// TM-30-20 §3.7.1 Eq. (43) - FL exponent for M
inline constexpr double kMExpFL = 0.25;

// -------------------------------------------------------------------------
// Step 12 constants (TM-30-20 §3.7.1 Eq. (48), (51))
// -------------------------------------------------------------------------

// TM-30-20 §3.7.1 Eq. (48)
inline constexpr double kJCompress = 0.007;

// TM-30-20 §3.7.1 Eq. (51)
inline constexpr double kMCompress = 0.0228;

// -------------------------------------------------------------------------
// Helper: apply 3x3 matrix to [x, y, z]
// -------------------------------------------------------------------------

inline void mat3_mul(double m00, double m01, double m02, double m10, double m11,
                     double m12, double m20, double m21, double m22, double x,
                     double y, double z, double &rx, double &ry, double &rz) {
  rx = m00 * x + m01 * y + m02 * z;
  ry = m10 * x + m11 * y + m12 * z;
  rz = m20 * x + m21 * y + m22 * z;
}

// -------------------------------------------------------------------------
// Step 4: luminance-level adaptation (Eq. 36-38, TM-30-20 §3.7.1)
//
// R'a = 400*(FL*R'/100)^0.42 / (27.13 + (FL*R'/100)^0.42) + 0.1
//
// Handles negative values with sign-preserving power.
// -------------------------------------------------------------------------

inline double luminance_adapt(double x) {
  // TM-30-20 §3.7.1 Eq. (36)-(38)
  const double base = kFL * x / kJScale; // FL * R' / 100
  if (base < 0.0) {
    // Edge case: sign-preserving power for negative values
    // TM-30-20 §3.7.1 (edge case note)
    const double pos_pow = std::pow(-base, kLumExp);
    return -kLumScale * pos_pow / (kLumAdd + pos_pow) + kLumOffset;
  }
  const double pow_val = std::pow(base, kLumExp);
  // TM-30-20 §3.7.1 Eq. (36)-(38)
  return kLumScale * pow_val / (kLumAdd + pow_val) + kLumOffset;
}

// -------------------------------------------------------------------------
// Precomputed (1.64 - 0.29^n)^0.73  (Eq. 42, TM-30-20 §3.7.1)
// -------------------------------------------------------------------------
inline double compute_c_paren() { // not constexpr: see kNbb above
  const double inner = kCParenA - std::pow(kCParenB, kN);
  return std::pow(inner, kCExpOuter);
}
// TM-30-20 §3.7.1 Eq. (42)
inline const double kCParen = compute_c_paren();

} // anonymous namespace

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

double ciecam02_compute_aw(const XyzTriple &xyz_white) {
  // Step 1: XYZ -> RGB (CAT02) for the white point
  // TM-30-20 §3.7.1 Eq. (29)-(30)
  double Rw, Gw, Bw;
  mat3_mul(kMCAT02_00, kMCAT02_01, kMCAT02_02, kMCAT02_10, kMCAT02_11,
           kMCAT02_12, kMCAT02_20, kMCAT02_21, kMCAT02_22, xyz_white.X,
           xyz_white.Y, xyz_white.Z, Rw, Gw, Bw);
  // TM-30-20 §3.7.1 Eq. (29)-(30)

  // Step 2: For the white point itself, chromatic adaptation gives
  // RC = 100, GC = 100, BC = 100 (since we divide by Rw and multiply by 100)
  // TM-30-20 §3.7.1 Eq. (31)-(33)
  const double RC = kJScale; // 100.0
  const double GC = kJScale;
  const double BC = kJScale;

  // Step 3: Convert to HPE cone space
  // TM-30-20 §3.7.1 Eq. (34)-(35)
  double Rp, Gp, Bp;
  mat3_mul(kCOMBINED_00, kCOMBINED_01, kCOMBINED_02, kCOMBINED_10, kCOMBINED_11,
           kCOMBINED_12, kCOMBINED_20, kCOMBINED_21, kCOMBINED_22, RC, GC, BC,
           Rp, Gp, Bp);
  // TM-30-20 §3.7.1 Eq. (34)-(35)

  // Step 4: Luminance-level adaptation
  // TM-30-20 §3.7.1 Eq. (36)-(38)
  const double Ra = luminance_adapt(Rp);
  const double Ga = luminance_adapt(Gp);
  const double Ba = luminance_adapt(Bp);

  // Step 6: Achromatic response
  // TM-30-20 §3.7.1 Eq. (44)
  return (2.0 * Ra + Ga + (1.0 / 20.0) * Ba - kAOffset) * kNbb;
}

std::array<Cam02Ucs, 99>
ciecam02_forward(const XyzTriple &xyz_white,
                 const std::array<XyzTriple, 99> &xyz_samples) {

  // -- Precompute white-point values ----------------------------------

  // Step 1 for white: RGB_w
  // TM-30-20 §3.7.1 Eq. (29)-(30)
  double Rw, Gw, Bw;
  mat3_mul(kMCAT02_00, kMCAT02_01, kMCAT02_02, kMCAT02_10, kMCAT02_11,
           kMCAT02_12, kMCAT02_20, kMCAT02_21, kMCAT02_22, xyz_white.X,
           xyz_white.Y, xyz_white.Z, Rw, Gw, Bw);
  // TM-30-20 §3.7.1 Eq. (29)-(30)

  // Precompute Aw (achromatic response of white)
  // TM-30-20 §3.7.1 Eq. (41), (44)
  const double Aw = ciecam02_compute_aw(xyz_white);

  // c*z exponent product
  // TM-30-20 §3.7.1 Eq. (41)
  const double cz = kc * kZ;

  // FL^0.25 for M (Eq. 43)
  // TM-30-20 §3.7.1 Eq. (43)
  const double FL_025 = std::pow(kFL, kMExpFL);

  // -- Process each CES sample ----------------------------------------

  std::array<Cam02Ucs, 99> result;

  for (std::size_t i = 0; i < 99; ++i) {
    const auto &xyz = xyz_samples[i];

    // -- Step 1: XYZ -> RGB (CAT02) ----------------------------------
    // TM-30-20 §3.7.1 Eq. (29)-(30)
    double Rs, Gs, Bs;
    mat3_mul(kMCAT02_00, kMCAT02_01, kMCAT02_02, kMCAT02_10, kMCAT02_11,
             kMCAT02_12, kMCAT02_20, kMCAT02_21, kMCAT02_22, xyz.X, xyz.Y,
             xyz.Z, Rs, Gs, Bs);
    // TM-30-20 §3.7.1 Eq. (29)-(30)

    // -- Step 2: Chromatic adaptation (D=1) -------------------------
    // TM-30-20 §3.7.1 Eq. (31)-(33)
    const double RC = kJScale * Rs / Rw;
    const double GC = kJScale * Gs / Gw;
    const double BC = kJScale * Bs / Bw;

    // -- Step 3: Convert to HPE cone space --------------------------
    // TM-30-20 §3.7.1 Eq. (34)-(35)
    double Rp, Gp, Bp;
    mat3_mul(kCOMBINED_00, kCOMBINED_01, kCOMBINED_02, kCOMBINED_10,
             kCOMBINED_11, kCOMBINED_12, kCOMBINED_20, kCOMBINED_21,
             kCOMBINED_22, RC, GC, BC, Rp, Gp, Bp);
    // TM-30-20 §3.7.1 Eq. (34)-(35)

    // -- Step 4: Luminance-level adaptation -------------------------
    // TM-30-20 §3.7.1 Eq. (36)-(38)
    const double Ra = luminance_adapt(Rp);
    const double Ga = luminance_adapt(Gp);
    const double Ba = luminance_adapt(Bp);

    // -- Step 5: Opponent channels ----------------------------------
    // TM-30-20 §3.7.1 Eq. (39)-(40)
    const double a = Ra - (12.0 / 11.0) * Ga + (1.0 / 11.0) * Ba;
    const double b = (1.0 / 9.0) * (Ra + Ga - 2.0 * Ba);

    // -- Step 6: Achromatic response --------------------------------
    // TM-30-20 §3.7.1 Eq. (44)
    const double A = (2.0 * Ra + Ga + (1.0 / 20.0) * Ba - kAOffset) * kNbb;

    // -- Step 7: Lightness J ----------------------------------------
    // TM-30-20 §3.7.1 Eq. (41)
    // Clamp A/Aw to >= 0: for unphysical SPDs with negative XYZ,
    // A can become negative, making pow() produce NaN.
    const double A_ratio = std::fmax(0.0, A / Aw);
    const double J = kJScale * std::pow(A_ratio, cz);

    // -- Step 8: Hue angle h ----------------------------------------
    // TM-30-20 §3.7.1 Eq. (45)
    double h = std::atan2(b, a) * kRadToDeg;
    if (h < 0.0) {
      h += 360.0;
    }

    // -- Step 9: Eccentricity et ------------------------------------
    // TM-30-20 §3.7.1 Eq. (47)
    // cos((pi/180)*h + 2) where the "+ 2" is in radians
    const double h_rad = h * std::numbers::pi / 180.0;
    const double et = kEtScale * (std::cos(h_rad + kEtPhase) + kEtBaseline);

    // -- Step 10: Temporary t ---------------------------------------
    // TM-30-20 §3.7.1 Eq. (46)
    const double denom = Ra + Ga + (21.0 / 20.0) * Ba;
    const double t =
        kTFactor * kNc * kNbb * et * std::sqrt(a * a + b * b) / denom;

    // -- Step 11: Chroma C and Colorfulness M -----------------------
    // TM-30-20 §3.7.1 Eq. (42)
    // Use fabs(t): for physically unrealizable SPDs (negative XYZ),
    // the denominator can become negative, making t negative.
    // pow(negative, 0.9) produces NaN; fabs preserves magnitude.
    const double C = std::pow(std::fabs(t), kCExpT) *
                     std::sqrt(std::fmax(0.0, J) / kJScale) * kCParen;
    // TM-30-20 §3.7.1 Eq. (43)
    const double M = C * FL_025;

    // -- Step 12: CAM02-UCS coordinates -----------------------------
    // TM-30-20 §3.7.1 Eq. (48)
    const double J_prime =
        (1.0 + kJScale * kJCompress) * J / (1.0 + kJCompress * J);
    // TM-30-20 §3.7.1 Eq. (51)
    const double M_prime = (1.0 / kMCompress) * std::log(1.0 + kMCompress * M);
    // TM-30-20 §3.7.1 Eq. (49)-(50)
    const double cos_h = std::cos(h_rad);
    const double sin_h = std::sin(h_rad);
    const double a_prime = M_prime * cos_h;
    const double b_prime = M_prime * sin_h;

    result[i] = Cam02Ucs{J_prime, a_prime, b_prime};
  }

  return result;
}

} // namespace tm30
