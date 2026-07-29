// Trapezoidal integration over wavelength grids.
// TM-30-20 §3.6: Calculation of Tristimulus Values
#include "tm30/integrate.hpp"

#include <stdexcept>
#include <string>

namespace tm30 {

double trapezoidal_integrate(const std::vector<double>& wavelengths,
                             const std::vector<double>& integrand) {
  const std::size_t n = wavelengths.size();

  if (n < 2) {
    throw std::invalid_argument(
        "trapezoidal_integrate requires at least 2 wavelength points, got " +
        std::to_string(n));
  }

  if (n != integrand.size()) {
    throw std::invalid_argument(
        "trapezoidal_integrate: wavelengths and integrand must have the same size");
  }

  // TM-30-20 §3.6: trapezoidal integration for tristimulus values
  double integral = 0.0;

  for (std::size_t i = 0; i < n - 1; ++i) {
    const double dw = wavelengths[i + 1] - wavelengths[i];  // TM-30-20 §3.6: Δλ per segment
    const double avg = 0.5 * (integrand[i] + integrand[i + 1]);
    integral += avg * dw;
  }

  return integral;
}

}  // namespace tm30
