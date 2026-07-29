// SPD container - construction-time validation per TM-30-20 §3.5.
#include "tm30/spd.hpp"

namespace tm30 {

// TM-30-20 §3.5: "Calculations shall be performed over the range 380 to 780 nm"
// TM-30-20 §3.5: "Minimum required range: 400 to 700 nm"
static constexpr double kMinFullRange = 380.0;  // TM-30-20 §3.5
static constexpr double kMaxFullRange = 780.0;  // TM-30-20 §3.5
static constexpr double kMinRequiredLo = 400.0; // TM-30-20 §3.5
static constexpr double kMinRequiredHi = 700.0; // TM-30-20 §3.5

Spd::Spd(std::vector<double> wavelengths, std::vector<double> values)
    : wavelengths_(std::move(wavelengths)), values_(std::move(values)) {
  validate();
}

void Spd::validate() {
  // 1. Non-empty
  if (wavelengths_.empty()) {
    throw InvalidSpd(
        "SPD is empty; at least one wavelength–value pair required");
  }

  if (wavelengths_.size() != values_.size()) {
    throw InvalidSpd("Wavelength and value arrays must have the same size");
  }

  // 2. Non-negative values - TM-30-20 §3.2 (spectral power is never negative)
  for (std::size_t i = 0; i < values_.size(); ++i) {
    if (values_[i] < 0) {
      std::ostringstream oss;
      oss << "Negative SPD value at index " << i << " (wavelength "
          << wavelengths_[i] << " nm)";
      throw InvalidSpd(oss.str());
    }
  }

  // 3. Strictly monotonic increasing wavelengths
  for (std::size_t i = 1; i < wavelengths_.size(); ++i) {
    if (wavelengths_[i] <= wavelengths_[i - 1]) {
      std::ostringstream oss;
      oss << "Wavelengths must be strictly monotonic increasing; "
          << "found " << wavelengths_[i - 1] << " >= " << wavelengths_[i]
          << " at index " << i;
      throw InvalidSpd(oss.str());
    }
  }

  // 4. Minimum required range: at least 400–700 nm - TM-30-20 §3.5
  const double lo = wavelengths_.front();
  const double hi = wavelengths_.back();
  if (lo > kMinRequiredLo || hi < kMinRequiredHi) {
    std::ostringstream oss;
    oss << "SPD wavelength range [" << lo << ", " << hi
        << "] does not cover the minimum required range [" << kMinRequiredLo
        << ", " << kMinRequiredHi << "] nm";
    throw InvalidSpd(oss.str());
  }

  // 5. Compute step size (uniformity check)
  if (wavelengths_.size() >= 2) {
    const double first_step = wavelengths_[1] - wavelengths_[0];
    bool uniform = true;

    // Use a tolerance relative to the step size to detect non-uniform grids.
    // Derived from machine epsilon; no hard-coded magic number.
    const double tol = std::numeric_limits<double>::epsilon() * first_step * 10;

    for (std::size_t i = 2; i < wavelengths_.size(); ++i) {
      const double s = wavelengths_[i] - wavelengths_[i - 1];
      if (std::abs(s - first_step) > tol) {
        uniform = false;
        break;
      }
    }
    if (uniform) {
      step_ = first_step;
    }
    // else step_ stays at its default-initialized value (0.0, from member init)
  }
}

} // namespace tm30
