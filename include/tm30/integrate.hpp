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

/// Per-point trapezoidal weights for a wavelength grid.
///
/// Returns a vector of weights such that Σ_i weights[i] * f[i] is
/// mathematically equivalent to trapezoidal_integrate(wavelengths, f) for
/// any integrand f. This allows precomputing weights once for a wavelength
/// grid and reusing them across many integrands.
///
/// Handles non-uniform wavelength grids correctly by computing Δλ per
/// segment. Each interior point's weight is half the sum of its two adjacent
/// segment widths; each endpoint's weight is half of its one adjacent segment.
///
/// @param wavelengths  Monotonically increasing wavelength values (nm).
/// @return             A vector of per-point weights (same size as wavelengths).
///
/// @throws std::invalid_argument if fewer than 2 points are provided.
std::vector<double> trapezoidal_weights(const std::vector<double>& wavelengths);

}  // namespace tm30
