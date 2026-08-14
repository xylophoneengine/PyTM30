// CIE 15:2018 resampling - linear interpolation + flat extrapolation.
// TM-30-20 §3.5: Interpolation Rules
#include "tm30/resample.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string> // std::to_string
#include <vector>

namespace tm30 {

namespace {

/// Linearly interpolate (or flat-extrapolate) a single spectral vector
/// from source wavelengths/values to target_wavelengths.
///
/// TM-30-20 §3.5 requires linear interpolation.
/// TM-30-20 §1.3 (Errata): flat extrapolation replaces logarithm-based.
std::vector<double> lerp_vector(const std::vector<double> &target_wl,
                                const std::vector<double> &source_wl,
                                const std::vector<double> &source_vals) {

  if (source_wl.empty() || source_vals.empty()) {
    throw std::invalid_argument("Source data is empty");
  }

  std::vector<double> result;
  result.reserve(target_wl.size());

  // Two-pointer approach: source_wl and target_wl are both sorted ascending.
  std::size_t j =
      0; // index into source_wl such that source_wl[j] <= target_wl[i]

  for (double tw : target_wl) {
    // Flat extrapolation - low side
    // TM-30-20 §1.3: flat extrapolation for lambda < first CES lambda
    if (tw <= source_wl.front()) {
      result.push_back(source_vals.front());
      continue;
    }

    // Flat extrapolation - high side
    // TM-30-20 §1.3: flat extrapolation for lambda > last CES lambda
    if (tw >= source_wl.back()) {
      result.push_back(source_vals.back());
      continue;
    }

    // Advance j until source_wl[j] <= tw < source_wl[j+1]
    while (j + 1 < source_wl.size() && source_wl[j + 1] <= tw) {
      ++j;
    }
    // Ensure we haven't overshot
    if (j + 1 >= source_wl.size() || source_wl[j] > tw) {
      j = 0;
      while (j + 1 < source_wl.size() && source_wl[j + 1] <= tw) {
        ++j;
      }
    }

    // Linear interpolation between indices j and j+1
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

CesData resample_ces(const std::vector<double> &target_wavelengths,
                     const CesData &source) {
  CesData result;
  result.wavelengths = target_wavelengths;

  const std::size_t n_ces = source.samples.size();

  if (n_ces != 99) {
    throw std::invalid_argument("Expected 99 CES samples, got " +
                                std::to_string(n_ces));
  }

  result.samples.resize(n_ces);

  for (std::size_t i = 0; i < n_ces; ++i) {
    result.samples[i] =
        lerp_vector(target_wavelengths, source.wavelengths, source.samples[i]);
  }

  return result;
}

CmfData resample_cmf(const std::vector<double> &target_wavelengths,
                     const CmfData &source) {
  CmfData result;
  result.wavelengths = target_wavelengths;

  result.x_bar =
      lerp_vector(target_wavelengths, source.wavelengths, source.x_bar);
  result.y_bar =
      lerp_vector(target_wavelengths, source.wavelengths, source.y_bar);
  result.z_bar =
      lerp_vector(target_wavelengths, source.wavelengths, source.z_bar);

  return result;
}

} // namespace tm30
