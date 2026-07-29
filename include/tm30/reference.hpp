#pragma once

/// @file reference.hpp
/// Reference illuminant generation per TM-30-20 §3.3.
///
/// Provides Planckian radiation (Tt ≤ 4000 K), CIE D-series daylight
/// (Tt ≥ 5000 K), and the proportional blend (4000 K < Tt < 5000 K).

#include <string>
#include <vector>

namespace tm30 {

/// Daylight basis vectors S₀(λ), S₁(λ), S₂(λ) from CIE 15:2004 Table T.2.
///
/// Values are at 5 nm intervals from 380 to 780 nm.
/// TM-30-20 §3.3: CIE Daylight D Series
struct DaylightBasis {
  std::vector<double> wavelengths; // 380, 385, …, 780 nm
  std::vector<double> S0;
  std::vector<double> S1;
  std::vector<double> S2;
};

/// Load daylight basis vectors from a CSV file.
///
/// Expected columns: wavelength, S0, S1, S2.
/// @throws std::runtime_error on file-not-found or parse failure.
DaylightBasis load_daylight_basis(const std::string &filepath);

/// Generate a Planckian radiator SPD at the given CCT.
///
/// Normalized at 560 nm so that Sr,P(560 nm) = 1.0.
///
/// @param cct         Correlated color temperature (K).
/// @param wavelengths Wavelength grid (nm), monotonically increasing.
/// @return            SPD values at each wavelength, normalized at 560 nm.
///
/// TM-30-20 §3.3 Eq. (5)-(6)
std::vector<double> generate_planckian(double cct,
                                       const std::vector<double> &wavelengths);

/// Resample daylight basis vectors (S₀, S₁, S₂) to a target wavelength
/// grid, using the same linear-interpolation + flat-extrapolation rule
/// that generate_cie_d() applies internally.
///
/// Exposed so callers who share one wavelength grid across many calls can
/// resample the basis once and pass the result to generate_cie_d() /
/// generate_reference_spd() with already_resampled=true, instead of
/// re-interpolating it on every call.
///
/// TM-30-20 §3.5: "Linear interpolation shall be used."
DaylightBasis
resample_daylight_basis(const std::vector<double> &target_wavelengths,
                        const DaylightBasis &basis);

/// Generate a CIE D-series daylight SPD at the given CCT.
///
/// Uses the daylight basis vectors and the M₁, M₂ multiplier formulas.
/// Normalized at 560 nm so that Sr,D(560 nm) = 1.0.
///
/// @param cct         Correlated color temperature (K) - used as Tr.
/// @param wavelengths Wavelength grid (nm), monotonically increasing.
/// @param basis       Daylight basis vectors (S₀, S₁, S₂). Interpolated to
///                    the requested wavelength grid internally, unless
///                    already_resampled is true.
/// @param already_resampled
///                    If true, `basis` is assumed to already be resampled
///                    to `wavelengths` (e.g. via resample_daylight_basis()
///                    or prepare_resampled_tables()), and the internal
///                    interpolation step is skipped - a pure performance
///                    optimization with no effect on the result when the
///                    assumption holds. Defaults to false, preserving
///                    existing behavior for all current callers.
/// @return            SPD values at each wavelength, normalized at 560 nm.
///
/// TM-30-20 §3.3 Eq. (7)-(12)
std::vector<double> generate_cie_d(double cct,
                                   const std::vector<double> &wavelengths,
                                   const DaylightBasis &basis,
                                   bool already_resampled = false);

/// Generate the reference illuminant SPD for a test source with the given CCT.
///
/// Selection rules (TM-30-20 §3.3):
///   - Tt ≤ 4000 K  → pure Planckian               // Eq. (14)
///   - 4000 K < Tt < 5000 K → proportional blend    // Eq. (15)
///   - Tt ≥ 5000 K  → pure CIE D-series             // Eq. (16)
///
/// For the blend, each component is Y-normalized to 100 (CIE 1964 10°)
/// before blending per the normative requirement in §3.3.
/// The returned SPD is re-normalized at 560 nm (value = 1.0).
///
/// @param cct         Correlated color temperature of the test source (K).
/// @param wavelengths Wavelength grid (nm), monotonically increasing.
/// @param basis       Daylight basis vectors resampled to the wavelength grid.
/// @param cmf_y_bar   CIE 1964 10° ȳ₁₀(λ), resampled to match wavelengths.
/// @param already_resampled
///                    Forwarded to generate_cie_d() - see its docs.
///                    Defaults to false, preserving existing behavior.
/// @return            Reference illuminant SPD, normalized at 560 nm.
///
/// TM-30-20 §3.3 Eq. (13)-(16)
std::vector<double>
generate_reference_spd(double cct, const std::vector<double> &wavelengths,
                       const DaylightBasis &basis,
                       const std::vector<double> &cmf_y_bar,
                       bool already_resampled = false);

} // namespace tm30
