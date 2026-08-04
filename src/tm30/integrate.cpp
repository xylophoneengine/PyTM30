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

std::vector<double> trapezoidal_weights(const std::vector<double>& wavelengths) {
  const std::size_t n = wavelengths.size();

  if (n < 2) {
    throw std::invalid_argument(
        "trapezoidal_weights requires at least 2 wavelength points, got " +
        std::to_string(n));
  }

  std::vector<double> weights(n);

  // First point: w[0] = 0.5 * (λ[1] - λ[0])
  weights[0] = 0.5 * (wavelengths[1] - wavelengths[0]);  // TM-30-20 §3.6

  // Interior points: w[i] = 0.5 * ((λ[i] - λ[i-1]) + (λ[i+1] - λ[i]))
  for (std::size_t i = 1; i < n - 1; ++i) {
    const double left_width = wavelengths[i] - wavelengths[i - 1];      // TM-30-20 §3.6
    const double right_width = wavelengths[i + 1] - wavelengths[i];     // TM-30-20 §3.6
    weights[i] = 0.5 * (left_width + right_width);  // TM-30-20 §3.6
  }

  // Last point: w[n-1] = 0.5 * (λ[n-1] - λ[n-2])
  weights[n - 1] = 0.5 * (wavelengths[n - 1] - wavelengths[n - 2]);  // TM-30-20 §3.6

  return weights;
}

}  // namespace tm30
