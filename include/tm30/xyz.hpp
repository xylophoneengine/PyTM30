#pragma once

/// @file xyz.hpp
/// CIE 1964 10-deg tristimulus value computation.
///
/// Computes XYZ tristimulus values for sources and for the 99 CES samples
/// under a given source SPD, using the CIE 1964 10-deg standard colorimetric
/// observer.
///
/// TM-30-20 §3.1: Colorimetric Observer - all color rendition calculations
///                use the CIE 1964 10-deg standard colorimetric observer.
/// TM-30-20 §3.2: Test Source - tristimulus values for the source itself.
/// TM-30-20 §3.6: Calculation of Tristimulus Values - CES tristimulus values.

#include <array>
#include <cstddef>
#include <vector>

#include "tm30/cct.hpp"          // PlanckianLut, CctDuvResult
#include "tm30/chromaticity.hpp" // YuvTriple
#include "tm30/reference.hpp"    // DaylightBasis, generate_reference_spd
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

  // TM-30-20 §3.2 Eq. (4): kt = 100 / integral St(lambda) * ybar10(lambda)
  // dlambda
  double k; // TM-30-20 §3.2 Eq. (4)
};

/// Compute the source's own tristimulus values and normalisation constant.
///
/// @param spd_wavelengths  The SPD wavelength grid (nm), monotonically
/// increasing.
/// @param spd_values       The SPD spectral power values St(lambda).
/// @param cmf_x_bar        CIE 1964 10-deg xbar10(lambda) values, resampled to
/// match spd_wavelengths.
/// @param cmf_y_bar        CIE 1964 10-deg ybar10(lambda) values, resampled to
/// match spd_wavelengths.
/// @param cmf_z_bar        CIE 1964 10-deg zbar10(lambda) values, resampled to
/// match spd_wavelengths.
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
/// @param spd_values       The SPD spectral power values St(lambda).
/// @param ces_data         CES reflectance data, resampled to spd_wavelengths.
///                         Must contain exactly 99 CES samples.
/// @param cmf_x_bar        CIE 1964 10-deg xbar10(lambda), resampled.
/// @param cmf_y_bar        CIE 1964 10-deg ybar10(lambda), resampled.
/// @param cmf_z_bar        CIE 1964 10-deg zbar10(lambda), resampled.
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

// -- Convenience functions (handle CMF resampling internally) ----------

/// Compute source XYZ from an SPD, with automatic CMF resampling.
///
/// This is the convenience entry point for single-SPD XYZ computation.
/// It resamples the CMF to match the SPD wavelength grid, then integrates.
///
/// @param spd_wavelengths  SPD wavelength grid (nm).
/// @param spd_values       SPD spectral power values.
/// @param cmf_data         CIE 1964 10-deg CMF data (will be resampled to
/// spd_wavelengths).
/// @param K                Normalisation constant.
///                         If std::nullopt (default): auto-compute k =
///                         100/integral St*ybar dlambda
///                           -> Y = 100 (TM-30-20 §3.2 Eq. 4).
///                         If a value is provided, it is used directly as the
///                           multiplier for the raw tristimulus integrals.
///                           K = 1.0 returns raw integrals.
///                           K = 683.0 gives absolute photometric quantities:
///                           683 lm/W is Km, the maximum luminous efficacy of
///                           radiation at 555 nm (CIE/SI definition of the
///                           candela), not a TM-30-20 quantity.
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
/// @param cmf_data         CIE 1964 10-deg CMF data.
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

/// spd_to_xyz_batch() with the CMF already resampled to `spd_wavelengths`
/// and no integration-range clipping, for callers that cache the
/// resample per grid (e.g. the Python bindings). A length mismatch
/// between `cmf_resampled` and the grid throws std::invalid_argument.
std::vector<XyzTriple>
spd_to_xyz_batch_prepared(const std::vector<double> &spd_wavelengths,
                          const std::vector<std::vector<double>> &spd_matrix,
                          const CmfData &cmf_resampled,
                          std::optional<double> K = std::nullopt);

/// Compute CIE 1976 Y,u',v' from an SPD.
///
/// Chains spd_to_xyz -> xyz_to_Yuv.
///
/// @param spd_wavelengths  SPD wavelength grid (nm).
/// @param spd_values       SPD spectral power values.
/// @param cmf_data         CIE 1964 10-deg CMF data.
/// @param K                Normalisation constant (see spd_to_xyz).
/// @param lambda_min       Lower integration bound (nm).
/// @param lambda_max       Upper integration bound (nm).
/// @return                 YuvTriple with Y, u', v'.
YuvTriple spd_to_Yuv(const std::vector<double> &spd_wavelengths,
                     const std::vector<double> &spd_values,
                     const CmfData &cmf_data,
                     std::optional<double> K = std::nullopt,
                     std::optional<double> lambda_min = std::nullopt,
                     std::optional<double> lambda_max = std::nullopt);

/// Compute CIE 1976 Y,u',v' for multiple SPDs sharing the same wavelength grid.
///
/// @param spd_wavelengths  SPD wavelength grid (nm).
/// @param spd_matrix       Vector of SPD value vectors.
/// @param cmf_data         CIE 1964 10-deg CMF data.
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

/// spd_to_Yuv_batch() with the CMF already resampled -- same contract as
/// spd_to_xyz_batch_prepared().
std::vector<YuvTriple>
spd_to_Yuv_batch_prepared(const std::vector<double> &spd_wavelengths,
                          const std::vector<std::vector<double>> &spd_matrix,
                          const CmfData &cmf_resampled,
                          std::optional<double> K = std::nullopt);

/// Convert multiple XYZ tristimulus triples to CIE 1976 Y,u',v'.
///
/// A plain per-row loop over the existing xyz_to_Yuv() scalar function -
/// there's no CMF or wavelength dependency to resample or cache here, so
/// this exists purely so a batch of XYZ triples can cross the Python/C++
/// boundary once instead of once per row.
///
/// @param xyzs  XYZ tristimulus triples to convert.
/// @return      Vector of YuvTriple (one per input triple).
std::vector<YuvTriple> xyz_to_Yuv_batch(const std::vector<XyzTriple> &xyzs);

/// Compute XYZ tristimulus values for the TM-30-20 §3.3 reference
/// illuminant at a given CCT.
///
/// Chains generate_reference_spd() (TM-30-20 §3.3 Eq. (13)-(16):
/// Planckian for Tt<=4000K, CIE D-series for Tt>=5000K, proportional blend
/// between) with the same tristimulus-integration logic spd_to_xyz() uses.
/// The same cmf_data is used both for the reference SPD's internal
/// Y-normalization blend step and the final XYZ integration, resampled
/// exactly once.
///
/// There is no lambda_min/lambda_max here (unlike spd_to_xyz) - the
/// reference SPD is synthetic data over the full requested grid, not a
/// real measured spectrum with a sensor-limited valid band, so
/// integration-bound clipping doesn't have the same motivating use case.
///
/// @param cct          Correlated color temperature (K).
/// @param wavelengths  Wavelength grid (nm) to generate the reference SPD
///                     on and integrate over.
/// @param basis        Daylight basis vectors (S0, S1, S2).
/// @param cmf_data     CMF data (will be resampled to `wavelengths`).
/// @param K            Normalisation constant - see spd_to_xyz(). nullopt
///                     (default): auto-normalize Y=100.
/// @return             XyzTriple for the reference illuminant at this CCT.
XyzTriple cct_to_xyz(double cct, const std::vector<double> &wavelengths,
                     const DaylightBasis &basis, const CmfData &cmf_data,
                     std::optional<double> K = std::nullopt);

/// Compute XYZ for the reference illuminant at each of several CCTs,
/// sharing one wavelength grid and one resampled CMF across the batch.
///
/// @param ccts         Correlated color temperatures (K).
/// @param wavelengths  Wavelength grid shared by every CCT in the batch.
/// @param basis        Daylight basis vectors.
/// @param cmf_data     CMF data (resampled once, reused for every CCT).
/// @param K            Normalisation constant - see spd_to_xyz().
/// @return             Vector of XyzTriple, one per input CCT.
std::vector<XyzTriple> cct_to_xyz_batch(const std::vector<double> &ccts,
                                        const std::vector<double> &wavelengths,
                                        const DaylightBasis &basis,
                                        const CmfData &cmf_data,
                                        std::optional<double> K = std::nullopt);

/// cct_to_xyz_batch() with the CMF already resampled to `wavelengths`,
/// for callers that already hold it (e.g. the bindings' fixed-grid
/// tables). A length mismatch throws std::invalid_argument.
std::vector<XyzTriple> cct_to_xyz_batch_prepared(
    const std::vector<double> &ccts, const std::vector<double> &wavelengths,
    const DaylightBasis &basis, const CmfData &cmf_resampled,
    std::optional<double> K = std::nullopt);

/// Compute CCT and Duv from an SPD, with automatic CMF resampling.
///
/// TM-30-20 §3.1 exception: CCT determination uses the CIE 1931 2-deg
/// observer, not the 1964 10-deg observer used everywhere else in this
/// library - `cmf_data` here must be 2-deg. Chains compute_source_xyz
/// (2-deg XYZ) -> compute_cct_duv_from_xyz (Ohno 2014, TM-30-20 §3.3).
///
/// No K/lambda_min/lambda_max: CCT/Duv come from (u,v) chromaticity, which
/// is scale-invariant, so K has no effect; lambda_min/max have no
/// motivating truncation use case (same reasoning as cct_to_xyz above).
///
/// The SPD is §3.5-conformed before integration (Spd construction:
/// samples outside 380-780 nm dropped, missing edges zero-filled, steps
/// > 5 nm rejected) -- identical treatment to the full pipeline.
///
/// @param spd_wavelengths SPD wavelength grid (nm).
/// @param spd_values      SPD spectral power values.
/// @param cmf_data        CIE 1931 2-deg CMF data (resampled internally).
/// @param planckian_lut   Pre-computed Planckian locus LUT.
/// @return                CctDuvResult with cct (K) and duv.
/// @throws InvalidSpd     if the SPD fails §3.5 validation.
CctDuvResult spd_to_cct(const std::vector<double> &spd_wavelengths,
                        const std::vector<double> &spd_values,
                        const CmfData &cmf_data,
                        const PlanckianLut &planckian_lut);

/// Compute spd_to_cct() for multiple SPDs sharing one wavelength grid.
///
/// Resamples the 2-deg CMF once (against the §3.5-conformed shared
/// grid), reused for every SPD in the batch - same
/// resample-once-loop-many pattern as spd_to_xyz_batch. Each row is
/// §3.5-conformed exactly like the single-SPD overload.
///
/// @param spd_wavelengths SPD wavelength grid (nm), shared by all SPDs.
/// @param spd_matrix      Vector of SPD value vectors (one per SPD).
/// @param cmf_data        CIE 1931 2-deg CMF data.
/// @param planckian_lut   Pre-computed Planckian locus LUT.
/// @return                Vector of CctDuvResult (one per SPD).
/// @throws InvalidSpd     if the grid or any row fails §3.5 validation.
std::vector<CctDuvResult>
spd_to_cct_batch(const std::vector<double> &spd_wavelengths,
                 const std::vector<std::vector<double>> &spd_matrix,
                 const CmfData &cmf_data, const PlanckianLut &planckian_lut);

/// spd_to_cct_batch() with the per-grid work already done, for callers
/// that evaluate many batches on one grid (e.g. the Python bindings'
/// per-grid cache): `cmf_resampled` must be the 2-deg CMF resampled to
/// the §3.5-conformed form of `raw_wavelengths` (i.e. to
/// `Spd(raw_wavelengths, ...).wavelengths()`). Each row is still
/// §3.5-validated and conformed via Spd; a size mismatch between the
/// conformed grid and `cmf_resampled` throws std::invalid_argument.
///
/// @throws InvalidSpd if the grid or any row fails §3.5 validation.
std::vector<CctDuvResult>
spd_to_cct_batch_prepared(const std::vector<double> &raw_wavelengths,
                          const std::vector<std::vector<double>> &spd_matrix,
                          const CmfData &cmf_resampled,
                          const PlanckianLut &planckian_lut);

/// Compute the total power of an SPD - radiometric (unweighted) or
/// photometric (CIE luminous-efficiency-weighted).
///
/// Radiometric (photometric=false): integral_st = integral S(lambda) dlambda
/// over [lambda_min, lambda_max] - no CMF weighting at all; units follow
/// whatever units the SPD's own values are in (e.g. W if S is in W/nm).
///
/// Photometric (photometric=true): Km * integral S(lambda) * ybar(lambda)
/// dlambda, where ybar is cmf_data's y-bar (resampled to the clipped grid) and
/// Km = 683.0 lm/W is the CIE/SI maximum luminous efficacy constant (matches
/// this codebase's existing K=683.0 "photometric absolute" convention on
/// spd_to_xyz).
///
/// @param wavelengths  SPD wavelength grid (nm).
/// @param values       SPD spectral power values.
/// @param cmf_data     CMF data (only used, and resampled, when
///                     photometric=true).
/// @param photometric  false: radiometric (W). true: photometric (lm).
/// @param lambda_min   Lower integration bound (nm). nullopt: from the
///                     first wavelength in the SPD.
/// @param lambda_max   Upper integration bound (nm). nullopt: to the last
///                     wavelength in the SPD.
/// @return             Total power (W or lm depending on `photometric`).
double spd_to_power(const std::vector<double> &wavelengths,
                    const std::vector<double> &values, const CmfData &cmf_data,
                    bool photometric,
                    std::optional<double> lambda_min = std::nullopt,
                    std::optional<double> lambda_max = std::nullopt);

/// Compute spd_to_power() for multiple SPDs sharing one wavelength grid.
///
/// @return  Vector of power values (W or lm), one per input SPD - unlike
///          every other *_batch function in this file, this returns a
///          flat scalar per row, not an XyzTriple: power is a single
///          number, not a tristimulus triple.
std::vector<double>
spd_to_power_batch(const std::vector<double> &wavelengths,
                   const std::vector<std::vector<double>> &spd_matrix,
                   const CmfData &cmf_data, bool photometric,
                   std::optional<double> lambda_min = std::nullopt,
                   std::optional<double> lambda_max = std::nullopt);

/// spd_to_power_batch() with the CMF already resampled to `wavelengths`
/// and no integration-range clipping -- same contract as
/// spd_to_xyz_batch_prepared(). The CMF is only consulted (and only
/// size-checked) when photometric is true.
std::vector<double>
spd_to_power_batch_prepared(const std::vector<double> &wavelengths,
                            const std::vector<std::vector<double>> &spd_matrix,
                            const CmfData &cmf_resampled, bool photometric);

} // namespace tm30
