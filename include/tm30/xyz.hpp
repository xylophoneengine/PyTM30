#pragma once

/// @file xyz.hpp
/// CIE 1964 10° tristimulus value computation.
///
/// Computes XYZ tristimulus values for sources and for the 99 CES samples
/// under a given source SPD, using the CIE 1964 10° standard colorimetric
/// observer.
///
/// TM-30-20 §3.1: Colorimetric Observer - all color rendition calculations
///                use the CIE 1964 10° standard colorimetric observer.
/// TM-30-20 §3.2: Test Source - tristimulus values for the source itself.
/// TM-30-20 §3.6: Calculation of Tristimulus Values - CES tristimulus values.

#include <array>
#include <cstddef>
#include <vector>

#include "tm30/chromaticity.hpp" // YuvTriple
#include "tm30/resample.hpp"     // CesData, CmfData

#include <optional>

namespace tm30 {

/// CIE XYZ tristimulus triple.
struct XyzTriple {
  double X;
  double Y;
  double Z;
};

/// Result of compute_source_xyz: the source's own tristimulus values
/// and the normalisation constant k.
///
/// Y should equal 100.0 exactly (within floating-point precision)
/// due to the normalisation.
///
/// TM-30-20 §3.2: Eq. (1)-(4)
struct SourceXyz {
  // TM-30-20 §3.2: Eq. (1)-(3) - source tristimulus values
  double X; // TM-30-20 §3.2 Eq. (1)
  double Y; // TM-30-20 §3.2 Eq. (2)
  double Z; // TM-30-20 §3.2 Eq. (3)

  // TM-30-20 §3.2 Eq. (4): kt = 100 / ∫ St(λ) · ȳ₁₀(λ) dλ
  double k; // TM-30-20 §3.2 Eq. (4)
};

/// Compute the source's own tristimulus values and normalisation constant.
///
/// @param spd_wavelengths  The SPD wavelength grid (nm), monotonically
/// increasing.
/// @param spd_values       The SPD spectral power values St(λ).
/// @param cmf_x_bar        CIE 1964 10° x̄₁₀(λ) values, resampled to match
/// spd_wavelengths.
/// @param cmf_y_bar        CIE 1964 10° ȳ₁₀(λ) values, resampled to match
/// spd_wavelengths.
/// @param cmf_z_bar        CIE 1964 10° z̄₁₀(λ) values, resampled to match
/// spd_wavelengths.
///
/// @return SourceXyz with X, Y, Z (Y = 100.0) and normalisation constant k.
///
/// TM-30-20 §3.2: Eq. (1)-(4)
SourceXyz compute_source_xyz(const std::vector<double> &spd_wavelengths,
                             const std::vector<double> &spd_values,
                             const std::vector<double> &cmf_x_bar,
                             const std::vector<double> &cmf_y_bar,
                             const std::vector<double> &cmf_z_bar);

/// Compute tristimulus values for all 99 CES samples under a source SPD.
///
/// Uses the source's pre-computed normalisation constant k.
/// The CES reflectance data must be pre-resampled to match spd_wavelengths.
/// The CMF data must also be pre-resampled to match spd_wavelengths.
///
/// @param spd_wavelengths  The SPD wavelength grid (nm).
/// @param spd_values       The SPD spectral power values St(λ).
/// @param ces_data         CES reflectance data, resampled to spd_wavelengths.
///                         Must contain exactly 99 CES samples.
/// @param cmf_x_bar        CIE 1964 10° x̄₁₀(λ), resampled.
/// @param cmf_y_bar        CIE 1964 10° ȳ₁₀(λ), resampled.
/// @param cmf_z_bar        CIE 1964 10° z̄₁₀(λ), resampled.
/// @param k                Source normalisation constant from
/// compute_source_xyz.
///
/// @return Array of 99 XyzTriple values.
///
/// TM-30-20 §3.6: Eq. (21)-(24)
std::array<XyzTriple, 99>
compute_ces_xyz(const std::vector<double> &spd_wavelengths,
                const std::vector<double> &spd_values, const CesData &ces_data,
                const std::vector<double> &cmf_x_bar,
                const std::vector<double> &cmf_y_bar,
                const std::vector<double> &cmf_z_bar, double k);

// ── Convenience functions (handle CMF resampling internally) ──────────

/// Compute source XYZ from an SPD, with automatic CMF resampling.
///
/// This is the convenience entry point for single-SPD XYZ computation.
/// It resamples the CMF to match the SPD wavelength grid, then integrates.
///
/// @param spd_wavelengths  SPD wavelength grid (nm).
/// @param spd_values       SPD spectral power values.
/// @param cmf_data         CIE 1964 10° CMF data (will be resampled to
/// spd_wavelengths).
/// @param K                Normalisation constant.
///                         If std::nullopt (default): auto-compute k =
///                         100/∫St·ȳ dλ
///                           → Y = 100 (TM-30-20 §3.2 Eq. 4).
///                         If a value is provided, it is used directly as the
///                           multiplier for the raw tristimulus integrals.
///                           K = 1.0 returns raw integrals.
///                           K = 683.0 matches luxpy's photometric absolute
///                           mode.
/// @param lambda_min       Lower integration bound (nm).  If std::nullopt,
///                         integrates from the first wavelength in the SPD.
/// @param lambda_max       Upper integration bound (nm).  If std::nullopt,
///                         integrates to the last wavelength in the SPD.
/// @return                 XyzTriple with X, Y, Z.
XyzTriple spd_to_xyz(const std::vector<double> &spd_wavelengths,
                     const std::vector<double> &spd_values,
                     const CmfData &cmf_data,
                     std::optional<double> K = std::nullopt,
                     std::optional<double> lambda_min = std::nullopt,
                     std::optional<double> lambda_max = std::nullopt);

/// Compute source XYZ for multiple SPDs sharing the same wavelength grid.
///
/// @param spd_wavelengths  SPD wavelength grid (nm), shared by all SPDs.
/// @param spd_matrix       Vector of SPD value vectors (one per SPD).
/// @param cmf_data         CIE 1964 10° CMF data.
/// @param K                Normalisation constant (see single-SPD version).
/// @param lambda_min       Lower integration bound (nm).
/// @param lambda_max       Upper integration bound (nm).
/// @return                 Vector of XyzTriple (one per SPD).
std::vector<XyzTriple>
spd_to_xyz_batch(const std::vector<double> &spd_wavelengths,
                 const std::vector<std::vector<double>> &spd_matrix,
                 const CmfData &cmf_data,
                 std::optional<double> K = std::nullopt,
                 std::optional<double> lambda_min = std::nullopt,
                 std::optional<double> lambda_max = std::nullopt);

/// Compute CIE 1976 Y,u′,v′ from an SPD.
///
/// Chains spd_to_xyz → xyz_to_Yuv.
///
/// @param spd_wavelengths  SPD wavelength grid (nm).
/// @param spd_values       SPD spectral power values.
/// @param cmf_data         CIE 1964 10° CMF data.
/// @param K                Normalisation constant (see spd_to_xyz).
/// @param lambda_min       Lower integration bound (nm).
/// @param lambda_max       Upper integration bound (nm).
/// @return                 YuvTriple with Y, u′, v′.
YuvTriple spd_to_Yuv(const std::vector<double> &spd_wavelengths,
                     const std::vector<double> &spd_values,
                     const CmfData &cmf_data,
                     std::optional<double> K = std::nullopt,
                     std::optional<double> lambda_min = std::nullopt,
                     std::optional<double> lambda_max = std::nullopt);

/// Compute CIE 1976 Y,u′,v′ for multiple SPDs sharing the same wavelength grid.
///
/// @param spd_wavelengths  SPD wavelength grid (nm).
/// @param spd_matrix       Vector of SPD value vectors.
/// @param cmf_data         CIE 1964 10° CMF data.
/// @param K                Normalisation constant (see spd_to_xyz).
/// @param lambda_min       Lower integration bound (nm).
/// @param lambda_max       Upper integration bound (nm).
/// @return                 Vector of YuvTriple (one per SPD).
std::vector<YuvTriple>
spd_to_Yuv_batch(const std::vector<double> &spd_wavelengths,
                 const std::vector<std::vector<double>> &spd_matrix,
                 const CmfData &cmf_data,
                 std::optional<double> K = std::nullopt,
                 std::optional<double> lambda_min = std::nullopt,
                 std::optional<double> lambda_max = std::nullopt);

} // namespace tm30
