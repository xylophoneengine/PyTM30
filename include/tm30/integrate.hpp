#pragma once

/// @file integrate.hpp
/// Trapezoidal integration over wavelength grids.
///
/// Used for computing tristimulus integrals (XYZ) from spectral data.
/// Handles non-uniform wavelength grids correctly.
///
/// TM-30-20 §3.6: Calculation of Tristimulus Values

#include <vector>
#include <cstddef>

namespace tm30 {

/// Trapezoidal-rule integration over a wavelength grid.
///
/// For N wavelength points with values f(λ_i):
///   ∫ f(λ) dλ ≈ Σ_{i=0}^{N-2} 0.5 · (f_i + f_{i+1}) · (λ_{i+1} - λ_i)
///
/// Handles non-uniform wavelength grids correctly by using Δλ per segment.
///
/// @param wavelengths  Monotonically increasing wavelength values (nm).
/// @param integrand    Function values at each wavelength.
/// @return             The definite integral over the given range.
///
/// @throws std::invalid_argument if wavelengths and integrand sizes differ,
///         or if fewer than 2 points are provided.
double trapezoidal_integrate(const std::vector<double>& wavelengths,
                             const std::vector<double>& integrand);

}  // namespace tm30
