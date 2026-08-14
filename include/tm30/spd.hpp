#pragma once

/// @file spd.hpp
/// Spectral Power Distribution container.
/// Holds a wavelength grid and corresponding spectral values,
/// with construction-time validation per TM-30-20 requirements.
///
/// TM-30-20 §3.5: Range and Interpolation of Data
/// TM-30-20 §3.2: Test Source

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <vector>

#include "tm30/errors.hpp"

namespace tm30 {

/// Spectral Power Distribution.
///
/// Construction validates:
/// - non-empty
/// - wavelengths cover at least 400-700 nm (minimum required range per §3.5)
/// - wavelengths are strictly monotonic increasing
/// - no wavelength step exceeds 5 nm (§3.5)
/// - all spectral values are non-negative
///
/// and then normalises the stored data to the §3.5 calculation range:
/// - samples outside 380-780 nm are dropped (§3.5: values outside the
///   range are dropped from the calculation)
/// - if the input covers less than 380-780 nm, the missing edge values
///   are zero-filled (§3.5: missing values are replaced by zeros),
///   extending the grid outward at the input's native edge step with the
///   final point clamped to exactly 380/780 nm. Grid alignment of the
///   fill is an implementation choice; the standard does not specify it.
///
/// The stored data is therefore the §3.5-conformed input, not the raw
/// input; the raw input range remains available via
/// input_min_wavelength()/input_max_wavelength().
///
/// Throws InvalidSpd on validation failure.
class Spd {
public:
  /// Construct from parallel wavelength and value arrays.
  /// @throws InvalidSpd if validation fails.
  Spd(std::vector<double> wavelengths, std::vector<double> values);

  const std::vector<double> &wavelengths() const noexcept {
    return wavelengths_;
  }
  const std::vector<double> &values() const noexcept { return values_; }

  std::size_t size() const noexcept { return wavelengths_.size(); }

  double min_wavelength() const noexcept { return wavelengths_.front(); }
  double max_wavelength() const noexcept { return wavelengths_.back(); }

  /// Raw input range before §3.5 normalisation (drop/zero-fill).
  double input_min_wavelength() const noexcept {
    return input_min_wavelength_;
  }
  double input_max_wavelength() const noexcept {
    return input_max_wavelength_;
  }

  /// True if zero-fill extended the grid to cover 380-780 nm (§3.5).
  bool zero_filled() const noexcept { return zero_filled_; }

  /// Return the uniform step size, or 0.0 if the grid is non-uniform.
  /// TM-30-20 §3.5: increments not greater than 5 nm.
  double step() const noexcept { return step_; }

private:
  void validate();
  void normalize();
  void detect_step();

  std::vector<double> wavelengths_;
  std::vector<double> values_;
  double input_min_wavelength_{};
  double input_max_wavelength_{};
  bool zero_filled_{false};
  double step_{}; // 0.0 when non-uniform; set in detect_step()
};

} // namespace tm30
