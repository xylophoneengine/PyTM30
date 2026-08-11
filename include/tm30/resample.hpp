#pragma once

/// @file resample.hpp
/// CIE 15:2018 spectral resampling: linear interpolation + flat extrapolation.
///
/// Resamples CES reflectance functions and CIE 1964 10-deg CMFs to match
/// the test SPD's wavelength grid. The test SPD itself is never interpolated.
///
/// TM-30-20 §3.5: Interpolation Rules

#include <cstddef>
#include <vector>

namespace tm30 {

/// CES reflectance data: one wavelength column + 99 CES sample columns.
struct CesData {
  std::vector<double> wavelengths;          // N source wavelengths
  std::vector<std::vector<double>> samples; // 99 vectors, each size N
};

/// CIE 1964 10-deg CMF data: wavelength + xbar10, ybar10, zbar10.
/// TM-30-20 §3.1: CIE 1964 10-deg standard colorimetric observer.
struct CmfData {
  std::vector<double> wavelengths; // N source wavelengths
  std::vector<double> x_bar;       // xbar10(lambda)
  std::vector<double> y_bar;       // ybar10(lambda)
  std::vector<double> z_bar;       // zbar10(lambda)
};

/// Linearly interpolate CES reflectance data to target_wavelengths.
///
/// Flat extrapolation per TM-30-20 Errata:
///   lambda < first CES lambda  -> first CES value
///   lambda > last CES lambda   -> last CES value
///
/// TM-30-20 §3.5: "Linear interpolation shall be used."
CesData resample_ces(const std::vector<double> &target_wavelengths,
                     const CesData &source);

/// Linearly interpolate CMF data to target_wavelengths.
///
/// Same linear interpolation + flat extrapolation as resample_ces.
CmfData resample_cmf(const std::vector<double> &target_wavelengths,
                     const CmfData &source);

} // namespace tm30
