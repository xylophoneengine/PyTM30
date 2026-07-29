#pragma once

/// @file resample.hpp
/// CIE 15:2018 spectral resampling: linear interpolation + flat extrapolation.
///
/// Resamples CES reflectance functions and CIE 1964 10° CMFs to match
/// the test SPD's wavelength grid. The test SPD itself is never interpolated.
///
/// TM-30-20 §3.5: Interpolation Rules

#include <vector>
#include <cstddef>

namespace tm30 {

/// CES reflectance data: one wavelength column + 99 CES sample columns.
struct CesData {
  std::vector<double> wavelengths;            // N source wavelengths
  std::vector<std::vector<double>> samples;   // 99 vectors, each size N
};

/// CIE 1964 10° CMF data: wavelength + x̄₁₀, ȳ₁₀, z̄₁₀.
/// TM-30-20 §3.1: CIE 1964 10° standard colorimetric observer.
struct CmfData {
  std::vector<double> wavelengths;  // N source wavelengths
  std::vector<double> x_bar;        // x̄₁₀(λ)
  std::vector<double> y_bar;        // ȳ₁₀(λ)
  std::vector<double> z_bar;        // z̄₁₀(λ)
};

/// Linearly interpolate CES reflectance data to target_wavelengths.
///
/// Flat extrapolation per TM-30-20 Errata:
///   λ < first CES λ  → first CES value
///   λ > last CES λ   → last CES value
///
/// TM-30-20 §3.5: "Linear interpolation shall be used."
CesData resample_ces(const std::vector<double>& target_wavelengths,
                     const CesData& source);

/// Linearly interpolate CMF data to target_wavelengths.
///
/// Same linear interpolation + flat extrapolation as resample_ces.
CmfData resample_cmf(const std::vector<double>& target_wavelengths,
                     const CmfData& source);

}  // namespace tm30
