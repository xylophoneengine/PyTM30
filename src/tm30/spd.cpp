// SPD container - construction-time validation per TM-30-20 §3.5.
#include "tm30/spd.hpp"

#include <cstddef>
#include <limits>
#include <sstream>
#include <utility> // std::move
#include <vector>

namespace tm30 {

// TM-30-20 §3.5: calculations run over 380-780 nm; the minimum input
// range is 400-700 nm; increments above 5 nm are not permitted.
static constexpr double kMinFullRange = 380.0;  // TM-30-20 §3.5
static constexpr double kMaxFullRange = 780.0;  // TM-30-20 §3.5
static constexpr double kMinRequiredLo = 400.0; // TM-30-20 §3.5
static constexpr double kMinRequiredHi = 700.0; // TM-30-20 §3.5
static constexpr double kMaxStepNm = 5.0;       // TM-30-20 §3.5

Spd::Spd(std::vector<double> wavelengths, std::vector<double> values)
    : wavelengths_(std::move(wavelengths)), values_(std::move(values)) {
  validate();
  normalize();
  detect_step();
}

void Spd::validate() {
  // 1. Non-empty
  if (wavelengths_.empty()) {
    throw InvalidSpd(
        "SPD is empty; at least one wavelength-value pair required");
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

  // 3. Strictly monotonic increasing wavelengths, with no step above
  //    5 nm (TM-30-20 §3.5: larger increments are not permitted).
  for (std::size_t i = 1; i < wavelengths_.size(); ++i) {
    if (wavelengths_[i] <= wavelengths_[i - 1]) {
      std::ostringstream oss;
      oss << "Wavelengths must be strictly monotonic increasing; "
          << "found " << wavelengths_[i - 1] << " >= " << wavelengths_[i]
          << " at index " << i;
      throw InvalidSpd(oss.str());
    }
    const double gap = wavelengths_[i] - wavelengths_[i - 1];
    if (gap > kMaxStepNm) {
      std::ostringstream oss;
      oss << "Wavelength step of " << gap << " nm between "
          << wavelengths_[i - 1] << " nm and " << wavelengths_[i]
          << " nm exceeds the 5 nm maximum increment permitted by "
          << "TM-30-20 §3.5";
      throw InvalidSpd(oss.str());
    }
  }

  // 4. Minimum required range: at least 400-700 nm - TM-30-20 §3.5
  const double lo = wavelengths_.front();
  const double hi = wavelengths_.back();
  if (lo > kMinRequiredLo || hi < kMinRequiredHi) {
    std::ostringstream oss;
    oss << "SPD wavelength range [" << lo << ", " << hi
        << "] does not cover the minimum required range [" << kMinRequiredLo
        << ", " << kMinRequiredHi << "] nm";
    throw InvalidSpd(oss.str());
  }

}

void Spd::normalize() {
  input_min_wavelength_ = wavelengths_.front();
  input_max_wavelength_ = wavelengths_.back();

  // TM-30-20 §3.5: samples outside the 380-780 nm calculation range are
  // dropped.
  if (input_min_wavelength_ < kMinFullRange ||
      input_max_wavelength_ > kMaxFullRange) {
    std::size_t first = 0;
    while (first < wavelengths_.size() &&
           wavelengths_[first] < kMinFullRange) {
      ++first;
    }
    std::size_t last = wavelengths_.size();
    while (last > first && wavelengths_[last - 1] > kMaxFullRange) {
      --last;
    }
    wavelengths_.assign(wavelengths_.begin() + first,
                        wavelengths_.begin() + last);
    values_.assign(values_.begin() + first, values_.begin() + last);
  }

  // TM-30-20 §3.5: missing values within 380-780 nm are replaced by
  // zeros. The grid is extended outward at the input's native edge step;
  // the final point is clamped to exactly 380/780 nm (grid alignment of
  // the fill is an implementation choice the standard does not specify),
  // so the last filled interval may be shorter than the native step. The
  // trapezoidal integrator handles non-uniform grids.
  if (wavelengths_.front() > kMinFullRange) {
    const double step = wavelengths_[1] - wavelengths_[0];
    std::vector<double> fill;
    double w = wavelengths_.front();
    while (w - step > kMinFullRange) {
      w -= step;
      fill.push_back(w);
    }
    fill.push_back(kMinFullRange); // TM-30-20 §3.5 lower calculation bound
    wavelengths_.insert(wavelengths_.begin(), fill.rbegin(), fill.rend());
    values_.insert(values_.begin(), fill.size(), 0.0); // TM-30-20 §3.5 zeros
    zero_filled_ = true;
  }
  if (wavelengths_.back() < kMaxFullRange) {
    const std::size_t n = wavelengths_.size();
    const double step = wavelengths_[n - 1] - wavelengths_[n - 2];
    double w = wavelengths_.back();
    while (w + step < kMaxFullRange) {
      w += step;
      wavelengths_.push_back(w);
      values_.push_back(0.0); // TM-30-20 §3.5 zeros
    }
    wavelengths_.push_back(kMaxFullRange); // TM-30-20 §3.5 upper bound
    values_.push_back(0.0);                // TM-30-20 §3.5 zeros
    zero_filled_ = true;
  }
}

void Spd::detect_step() {
  // Compute step size (uniformity check)
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
