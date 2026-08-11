// Reference illuminant generation per TM-30-20 §3.3.
// Planckian radiation, CIE D-series daylight, and the 4000-5000 K blend.
#include "tm30/reference.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/integrate.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tm30 {

// -------------------------------------------------------------------------
// load_daylight_basis
// -------------------------------------------------------------------------

DaylightBasis load_daylight_basis(const std::string &filepath) {
  CsvTable table = load_csv(filepath);

  // Verify expected column count
  if (table.headers.size() != 4) {
    throw std::runtime_error("Daylight basis CSV must have 4 columns "
                             "(wavelength, S0, S1, S2); got " +
                             std::to_string(table.headers.size()));
  }

  DaylightBasis basis;
  for (const auto &row : table.rows) {
    basis.wavelengths.push_back(row[0]);
    basis.S0.push_back(row[1]);
    basis.S1.push_back(row[2]);
    basis.S2.push_back(row[3]);
  }
  return basis;
}

// -------------------------------------------------------------------------
// Linear interpolation helper for spectral vectors
// -------------------------------------------------------------------------

namespace {

/// Linearly interpolate (or flat-extrapolate) source values to target
/// wavelengths.  Source and target wavelengths must be monotonically
/// increasing.
///
/// TM-30-20 §3.5: "Linear interpolation shall be used."
/// TM-30-20 §1.3 (Errata): flat extrapolation.
std::vector<double> interpolate_linear(const std::vector<double> &target_wl,
                                       const std::vector<double> &source_wl,
                                       const std::vector<double> &source_vals) {

  std::vector<double> result;
  result.reserve(target_wl.size());

  std::size_t j = 0;
  for (double tw : target_wl) {
    if (tw <= source_wl.front()) {
      result.push_back(source_vals.front());
      continue;
    }
    if (tw >= source_wl.back()) {
      result.push_back(source_vals.back());
      continue;
    }
    while (j + 1 < source_wl.size() && source_wl[j + 1] <= tw) {
      ++j;
    }
    if (j + 1 >= source_wl.size() || source_wl[j] > tw) {
      j = 0;
      while (j + 1 < source_wl.size() && source_wl[j + 1] <= tw) {
        ++j;
      }
    }
    const double w0 = source_wl[j];
    const double w1 = source_wl[j + 1];
    const double v0 = source_vals[j];
    const double v1 = source_vals[j + 1];
    const double t = (tw - w0) / (w1 - w0);
    result.push_back(v0 + t * (v1 - v0));
  }
  return result;
}

} // anonymous namespace

// -------------------------------------------------------------------------
// resample_daylight_basis
// -------------------------------------------------------------------------

DaylightBasis
resample_daylight_basis(const std::vector<double> &target_wavelengths,
                        const DaylightBasis &basis) {
  DaylightBasis result;
  result.wavelengths = target_wavelengths;
  result.S0 =
      interpolate_linear(target_wavelengths, basis.wavelengths, basis.S0);
  result.S1 =
      interpolate_linear(target_wavelengths, basis.wavelengths, basis.S1);
  result.S2 =
      interpolate_linear(target_wavelengths, basis.wavelengths, basis.S2);
  return result;
}

// -------------------------------------------------------------------------
// generate_planckian
// -------------------------------------------------------------------------

std::vector<double> generate_planckian(double cct,
                                       const std::vector<double> &wavelengths) {
  // TM-30-20 §3.3, Eq. (6): second radiation constant
  constexpr double c2 = 1.4388e-2; // TM-30-20 §3.3 Eq. (6)

  const std::size_t n = wavelengths.size();
  std::vector<double> spd(n);

  // Pre-compute normalisation denominator at 560 nm
  // lambda = 560 nm = 5.60e-7 m
  constexpr double lambda_560_m = 560.0e-9; // TM-30-20 §3.3 Eq. (5)
  // TM-30-20 §3.3 Eq. (6):
  //   Le,lambda(lambda, T) = lambda^(-5) / (exp(c2/(lambda*T)) - 1)
  const double L_560 =
      std::pow(lambda_560_m, -5.0) /
      (std::exp(c2 / (lambda_560_m * cct)) - 1.0); // TM-30-20 §3.3 Eq. (6)

  for (std::size_t i = 0; i < n; ++i) {
    // Convert nm to m: 1 nm = 1e-9 m
    const double lambda_m = wavelengths[i] * 1.0e-9; // TM-30-20 §3.3

    // TM-30-20 §3.3 Eq. (6)
    const double L_lambda =
        std::pow(lambda_m, -5.0) /
        (std::exp(c2 / (lambda_m * cct)) - 1.0); // TM-30-20 §3.3 Eq. (6)

    // TM-30-20 §3.3 Eq. (5): normalise at 560 nm
    spd[i] = L_lambda / L_560; // TM-30-20 §3.3 Eq. (5)
  }

  return spd;
}

// -------------------------------------------------------------------------
// generate_cie_d
// -------------------------------------------------------------------------

std::vector<double> generate_cie_d(double cct,
                                   const std::vector<double> &wavelengths,
                                   const DaylightBasis &basis,
                                   bool already_resampled) {
  const std::size_t n = wavelengths.size();

  // Interpolate daylight basis vectors to the requested wavelength grid.
  // The basis is at 5 nm; the target grid may be 1 nm, 5 nm, etc.
  // If the caller already resampled `basis` to `wavelengths` (e.g. via
  // resample_daylight_basis()/prepare_resampled_tables()), skip the
  // interpolation - it would just be a costly no-op.
  std::vector<double> S0, S1, S2;
  if (already_resampled) {
    S0 = basis.S0;
    S1 = basis.S1;
    S2 = basis.S2;
  } else {
    S0 = interpolate_linear(wavelengths, basis.wavelengths, basis.S0);
    S1 = interpolate_linear(wavelengths, basis.wavelengths, basis.S1);
    S2 = interpolate_linear(wavelengths, basis.wavelengths, basis.S2);
  }

  // TM-30-20 §3.3: xD chromaticity coordinate
  // The reciprocal temperature
  const double Tr = cct; // TM-30-20 §3.3: Tr = Tt for TM-30 reference

  // TM-30-20 §3.3 Eq. (10): Tt <= 7000 K branch
  // TM-30-20 §3.3 Eq. (11): Tt > 7000 K branch
  double xD;
  if (Tr <= 7000.0) {
    // TM-30-20 §3.3 Eq. (10)
    xD = -4.6070e9 / (Tr * Tr * Tr) // TM-30-20 §3.3 Eq. (10)
         + 2.9678e6 / (Tr * Tr)     // TM-30-20 §3.3 Eq. (10)
         + 0.09911e3 / Tr           // TM-30-20 §3.3 Eq. (10)
         + 0.244063;                // TM-30-20 §3.3 Eq. (10)
  } else {
    // TM-30-20 §3.3 Eq. (11)
    xD = -2.0064e9 / (Tr * Tr * Tr) // TM-30-20 §3.3 Eq. (11)
         + 1.9018e6 / (Tr * Tr)     // TM-30-20 §3.3 Eq. (11)
         + 0.24748e3 / Tr           // TM-30-20 §3.3 Eq. (11)
         + 0.237040;                // TM-30-20 §3.3 Eq. (11)
  }

  // TM-30-20 §3.3 Eq. (12): yD from xD
  const double yD =
      -3.000 * xD * xD + 2.870 * xD - 0.275; // TM-30-20 §3.3 Eq. (12)

  // TM-30-20 §3.3 Eq. (8)-(9): M1, M2 multipliers
  const double denom = 0.0241 + 0.2562 * xD -
                       0.7341 * yD; // TM-30-20 §3.3 Eq. (8)-(9) denominator

  const double M1 =
      (-1.3515 - 1.7703 * xD + 5.9114 * yD) / denom; // TM-30-20 §3.3 Eq. (8)
  const double M2 =
      (0.0300 - 31.4424 * xD + 30.0717 * yD) / denom; // TM-30-20 §3.3 Eq. (9)
  // NOTE: 0.0300 per TM-30-20 §3.3 Eq. (9) main body (TM-30-15 printed
  // 0.030). Errata 1 covers only §3.7.1 + Annex F.3.2/F.3.3/F.3.6 (plus
  // figure replacements F-2/F-3/F-5/F-9) and does NOT touch Eq. (9).

  // TM-30-20 §3.3 Eq. (7): S(lambda) = S0(lambda) + M1*S1(lambda) +
  // M2*S2(lambda)
  std::vector<double> spd(n);
  for (std::size_t i = 0; i < n; ++i) {
    spd[i] = S0[i] + M1 * S1[i] + M2 * S2[i]; // TM-30-20 §3.3 Eq. (7)
  }

  // Normalise at 560 nm.
  // The basis vectors give S0(560) = 100, S1(560) = 0, S2(560) = 0,
  // so S(560) = 100.  Divide by 100 (or more generally, by the value
  // at whichever wavelength is closest to 560 nm).
  // Find the index closest to 560 nm
  // TM-30-20 §3.3 Eq. (5): normalisation at 560 nm
  std::size_t idx_560 = 0;
  double min_dist = 1e9;
  for (std::size_t i = 0; i < n; ++i) {
    double d = std::abs(wavelengths[i] - 560.0);
    if (d < min_dist) {
      min_dist = d;
      idx_560 = i;
    }
  }

  const double scale = spd[idx_560];
  for (std::size_t i = 0; i < n; ++i) {
    spd[i] /= scale;
  }

  return spd;
}

// -------------------------------------------------------------------------
// generate_reference_spd
// -------------------------------------------------------------------------

std::vector<double>
generate_reference_spd(double cct, const std::vector<double> &wavelengths,
                       const DaylightBasis &basis,
                       const std::vector<double> &cmf_y_bar,
                       bool already_resampled) {
  const std::size_t n = wavelengths.size();

  // TM-30-20 §3.3 Eq. (14): Tt <= 4000 K -> pure Planckian
  if (cct <= 4000.0) { // TM-30-20 §3.3 Eq. (14)
    return generate_planckian(cct, wavelengths);
  }

  // TM-30-20 §3.3 Eq. (16): Tt >= 5000 K -> pure D-series
  if (cct >= 5000.0) { // TM-30-20 §3.3 Eq. (16)
    return generate_cie_d(cct, wavelengths, basis, already_resampled);
  }

  // TM-30-20 §3.3 Eq. (15): 4000 K < Tt < 5000 K -> proportional blend
  //
  // Blend procedure:
  // 1. Generate Planckian and D-series SPDs, both 560-nm normalised.
  // 2. Y-normalise each to 100 using CIE 1964 10-deg ybar10(lambda).
  // 3. Blend linearly with factor (5000 - Tt) / 1000.
  // 4. Re-normalise at 560 nm.
  //
  // TM-30-20 §3.3 Eq. (13) and normative text

  std::vector<double> planck =
      generate_planckian(cct, wavelengths); // TM-30-20 §3.3 Eq. (5)
  std::vector<double> daylight = generate_cie_d(
      cct, wavelengths, basis, already_resampled); // TM-30-20 §3.3 Eq. (7)

  // Compute Y for each component via trapezoidal integration.
  // TM-30-20 §3.6: Y = integral SPD(lambda) * ybar10(lambda) dlambda
  auto compute_Y = [&](const std::vector<double> &spd) -> double {
    std::vector<double> integrand(n);
    for (std::size_t i = 0; i < n; ++i) {
      integrand[i] = spd[i] * cmf_y_bar[i];
    }
    return trapezoidal_integrate(wavelengths, integrand); // TM-30-20 §3.6
  };

  const double Y_planck = compute_Y(planck);
  const double Y_daylight = compute_Y(daylight);

  // Y-normalise each to 100
  std::vector<double> planck_Y100(n);
  std::vector<double> daylight_Y100(n);
  const double k_planck = 100.0 / Y_planck; // TM-30-20 §3.3 blend normalisation
  const double k_daylight =
      100.0 / Y_daylight; // TM-30-20 §3.3 blend normalisation
  for (std::size_t i = 0; i < n; ++i) {
    planck_Y100[i] = planck[i] * k_planck;
    daylight_Y100[i] = daylight[i] * k_daylight;
  }

  // TM-30-20 §3.3 Eq. (13): blend factor
  const double blend_factor = (5000.0 - cct) / 1000.0; // TM-30-20 §3.3 Eq. (13)
  const double daylight_weight = 1.0 - blend_factor;   // TM-30-20 §3.3 Eq. (13)

  std::vector<double> blended(n);
  for (std::size_t i = 0; i < n; ++i) {
    blended[i] = blend_factor * planck_Y100[i] +
                 daylight_weight * daylight_Y100[i]; // TM-30-20 §3.3 Eq. (13)
  }

  // Re-normalise at 560 nm
  // TM-30-20 §3.3 Eq. (5): normalisation at 560 nm
  std::size_t idx_560 = 0;
  double min_dist = 1e9;
  for (std::size_t i = 0; i < n; ++i) {
    double d = std::abs(wavelengths[i] - 560.0);
    if (d < min_dist) {
      min_dist = d;
      idx_560 = i;
    }
  }

  const double scale_560 = blended[idx_560];
  for (std::size_t i = 0; i < n; ++i) {
    blended[i] /= scale_560;
  }

  return blended;
}

} // namespace tm30
