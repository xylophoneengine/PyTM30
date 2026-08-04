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
/// - all spectral values are non-negative
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

  /// Return the uniform step size, or 0.0 if the grid is non-uniform.
  /// TM-30-20 §3.5: increments not greater than 5 nm.
  double step() const noexcept { return step_; }

private:
  void validate();

  std::vector<double> wavelengths_;
  std::vector<double> values_;
  double step_{}; // 0.0 when non-uniform; set in validate()
};

} // namespace tm30
