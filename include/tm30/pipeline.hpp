#pragma once

/// @file pipeline.hpp
/// TM-30-20 CES colorimetry pipeline - end-to-end integration of Slices 1-4.
///
/// Orchestrates the full sequence:
///   1. Resample CES to SPD grid (Slice 1)
///   2. Compute 2-deg XYZ for test source -> CCT/Duv (Slice 3)
///   3. Generate reference SPD from CCT + basis + 10-deg CMF y_bar (Slice 4)
///   4. Compute source XYZ for test -> normalisation constant kt (Slice 2)
///   5. Compute CES XYZ under test source (Slice 2)
///   6. Compute source XYZ for reference -> normalisation kr
///   7. Compute CES XYZ under reference illuminant
///
/// TM-30-20 S3.4: Color Evaluation Samples
/// TM-30-20 S3.6: Calculation of Tristimulus Values

#include <array>
#include <vector>

#include "tm30/cct.hpp" // PlanckianLut, CctDuvResult, compute_cct_duv_from_xyz
#include "tm30/ciecam02.hpp"  // Cam02Ucs, ciecam02_forward
#include "tm30/gamut.hpp"     // GamutResult, compute_gamut
#include "tm30/hue_bins.hpp"  // HueBins, bin_by_hue
#include "tm30/metrics.hpp"   // RfResult, compute_delta_e, compute_rf
#include "tm30/reference.hpp" // DaylightBasis, generate_reference_spd
#include "tm30/resample.hpp"  // CesData, CmfData, resample_ces, resample_cmf
#include "tm30/xyz.hpp" // XyzTriple, SourceXyz, compute_source_xyz, compute_ces_xyz

namespace tm30 {

/// Complete result of the CES colorimetry pipeline for a single test SPD.
///
/// TM-30-20 S3.3: CCT, Duv
/// TM-30-20 S3.3: Reference illuminant SPD
/// TM-30-20 S3.6: Eq. (21)-(23) - CES tristimulus values under test source
/// TM-30-20 S3.6: Eq. (25)-(27) - CES tristimulus values under reference
struct CesColorimetryResult {
  double cct;                               // TM-30-20 S3.3
  double duv;                               // TM-30-20 S3.3
  std::vector<double> reference_spd_values; // Reference illuminant SPD values
                                            // TM-30-20 S3.3 Eq. (13)-(16)
  std::array<XyzTriple, 99> xyz_test_ces;   // TM-30-20 S3.6 Eq. (21)-(23)
  std::array<XyzTriple, 99> xyz_ref_ces;    // TM-30-20 S3.6 Eq. (25)-(27)
  std::array<Cam02Ucs, 99> jab_test_ces;    // TM-30-20 S3.7.1 (test adaptation)
  std::array<Cam02Ucs, 99>
      jab_ref_ces;    // TM-30-20 S3.7.1 (reference adaptation)
  HueBins hue_bins;   // TM-30-20 S4.3 (16 hue-angle bins)
  double delta_e_avg; // TM-30-20 S4.1 - mean of 99 dE' values
  double Rf;          // TM-30-20 S4.1 Eq. (54) - fidelity index
  GamutResult gamut;  // TM-30-20 S4.4-S4.8 - gamut, local metrics, CVG
  std::array<double, 99>
      rf_cesi;    // TM-30-20 S4.2 Eq. (56) - per-sample fidelity
  double rf_skin; // TM-30-20 S4.2 - skin fidelity (CES15+18 avg)
};

/// Run the full CES colorimetry pipeline for a test SPD.
///
/// Steps (TM-30-20 S3.4, S3.6):
///   1. Resample CES reflectance data to the SPD wavelength grid.
///   2. Resample both 2-deg and 10-deg CMF data to the SPD wavelength grid.
///   3. Compute the test source's own 2-deg XYZ -> CCT and Duv.
///   4. Generate the reference illuminant SPD from the CCT.
///   5. Compute the test source's 10-deg XYZ -> normalisation constant kt.
///   6. Compute CES XYZ under the test source (with kt).
///   7. Compute the reference source's 10-deg XYZ -> normalisation kr.
///   8. Compute CES XYZ under the reference illuminant (with kr).
///   9. Compute CIECAM02 J'a'b' under test source adaptation.
///  10. Compute CIECAM02 J'a'b' under reference illuminant adaptation.
///  11. Compute dE' color differences and Rf fidelity index.
///
/// @param spd_wavelengths  Test SPD wavelength grid (nm), monotonically
/// increasing.
/// @param spd_values       Test SPD spectral power values St(lambda).
/// @param cmf_2deg         CIE 1931 2-deg CMF data (for CCT computation).
/// @param cmf_10deg        CIE 1964 10-deg CMF data (for tristimulus
/// integration).
/// @param ces_data         CES reflectance data (99 samples, 1-nm native).
/// @param daylight_basis   Daylight basis vectors (S0, S1, S2).
/// @param planckian_lut    Planckian locus LUT (CIE 1931 2-deg observer).
///
/// @return CesColorimetryResult with CCT, Duv, reference SPD, and all CES XYZ
/// values.
///
/// TM-30-20 S3.4, S3.6
CesColorimetryResult compute_ces_colorimetry(
    const std::vector<double> &spd_wavelengths,
    const std::vector<double> &spd_values, const CmfData &cmf_2deg,
    const CmfData &cmf_10deg, const CesData &ces_data,
    const DaylightBasis &daylight_basis, const PlanckianLut &planckian_lut);

// ==========================================================================
//  Grid-fixed caching - precompute once, reuse across many SPDs that share
//  the same wavelength grid (the overwhelmingly common case in practice).
// ==========================================================================

/// Wavelength-grid-dependent tables, precomputed once and reused across
/// many SPD evaluations that share the same grid.
///
/// Building this is the expensive part (99 CES + 3 CMF curves resampled,
/// daylight basis resampled) - evaluating with it
/// (compute_ces_colorimetry_cached) skips all of that work entirely.
struct ResampledTables {
  std::vector<double> wavelengths; // Target grid these tables are resampled to.
  CesData ces;       // CES reflectance data, resampled to `wavelengths`.
  CmfData cmf_2deg;  // CIE 1931 2-deg CMF, resampled to `wavelengths`.
  CmfData cmf_10deg; // CIE 1964 10-deg CMF, resampled to `wavelengths`.
  DaylightBasis
      daylight_basis; // Daylight basis (S0,S1,S2), resampled to `wavelengths`.
  std::vector<double>
      lambda_pow_m5; // lambda^(-5), lambda in metres, one per grid point.
                     // The grid-fixed half of Planck's law, which
                     // generate_planckian() would otherwise recompute for
                     // every SPD that reaches the Planckian or blend
                     // branch. See planckian_lambda_pow_table().
                     // TM-30-20 S3.3 Eq. (6)
  std::vector<double>
      trapezoidal_w; // Per-point trapezoidal weights for `wavelengths`,
                     // which compute_ces_xyz() would otherwise rebuild on
                     // each of its two calls per SPD. See
                     // trapezoidal_weights().
                     // TM-30-20 S3.6 Eq. (21)-(23)
};

/// Precompute (once) all wavelength-grid-dependent resampled tables for a
/// target grid, from the raw (native-grid) source tables.
///
/// @param target_wavelengths  The wavelength grid to resample everything to.
/// @param cmf_2deg_src        Raw (native-grid) CIE 1931 2-deg CMF data.
/// @param cmf_10deg_src       Raw (native-grid) CIE 1964 10-deg CMF data.
/// @param ces_src             Raw (native-grid) CES reflectance data (99
/// samples).
/// @param daylight_basis_src  Raw (native-grid, 5 nm) daylight basis vectors.
///
/// @return ResampledTables holding all four tables resampled to
/// target_wavelengths.
ResampledTables
prepare_resampled_tables(const std::vector<double> &target_wavelengths,
                         const CmfData &cmf_2deg_src,
                         const CmfData &cmf_10deg_src, const CesData &ces_src,
                         const DaylightBasis &daylight_basis_src);

/// Run the full CES colorimetry pipeline for a test SPD using pre-resampled
/// tables (see prepare_resampled_tables()).
///
/// Identical to compute_ces_colorimetry() except steps 1-2 (CES/CMF
/// resampling) are skipped entirely - the CES/CMF data is read directly
/// off `tables`, and the pre-resampled daylight basis is threaded through
/// to reference-SPD generation. The caller is responsible for ensuring
/// `spd_values` is aligned with `tables.wavelengths` (same length, same
/// grid); this function does not re-check that invariant.
///
/// @param spd_values    Test SPD spectral power values St(lambda), aligned with
///                      tables.wavelengths.
/// @param tables        Pre-resampled CES/CMF/daylight-basis tables (see
///                      prepare_resampled_tables()).
/// @param planckian_lut Planckian locus LUT (CIE 1931 2-deg observer).
///
/// @return CesColorimetryResult, numerically identical to what
///         compute_ces_colorimetry() would produce for the same SPD on
///         the same grid.
CesColorimetryResult
compute_ces_colorimetry_cached(const std::vector<double> &spd_values,
                               const ResampledTables &tables,
                               const PlanckianLut &planckian_lut);

/// Reference-side colorimetry for a single CCT: reference illuminant SPD,
/// its 10-deg white point, CES tristimulus values, and CAM02-UCS J'a'b'.
///
/// This is exactly the reference-side subset of CesColorimetryResult
/// (steps 4, 7, 8, 10 of compute_ces_colorimetry_cached) - the part that
/// depends on the CCT alone, not on any test SPD. Callers that need the
/// reference side for a CCT without a test SPD (e.g. one already obtained
/// from some other CCT/Duv computation) can call
/// compute_reference_colorimetry() directly instead of running the whole
/// pipeline.
///
/// TM-30-20 S3.3 Eq. (13)-(16): reference illuminant SPD.
/// TM-30-20 S3.6 Eq. (25)-(27): reference CES tristimulus values.
/// TM-30-20 S3.7.1: reference-adapted CAM02-UCS J'a'b'.
struct ReferenceColorimetry {
  std::vector<double> spd;           // Reference illuminant SPD values.
  XyzTriple white;                   // Reference 10-deg white point XYZ.
  std::array<XyzTriple, 99> xyz_ces; // Reference CES tristimulus values.
  std::array<Cam02Ucs, 99> jab_ces;  // Reference-adapted CAM02-UCS J'a'b'.
};

/// Compute the reference-side colorimetry for `cct` using pre-resampled
/// `tables` (see prepare_resampled_tables()).
///
/// Body is the same statements as steps 4, 7, 8, 10 of
/// compute_ces_colorimetry_cached(), extracted so both that function and
/// standalone callers can share them; compute_ces_colorimetry_cached()
/// calls this and copies its fields into its own result.
///
/// @param cct     Correlated color temperature (Kelvin) to generate the
///                reference illuminant for.
/// @param tables  Pre-resampled CES/CMF/daylight-basis tables (see
///                prepare_resampled_tables()), on the same grid `cct` was
///                derived from.
///
/// @return ReferenceColorimetry for `cct` on `tables`' grid.
ReferenceColorimetry
compute_reference_colorimetry(double cct, const ResampledTables &tables);

// ==========================================================================
//  Linear tristimulus maps - unnormalised integrands on an input grid.
// ==========================================================================

/// Linear maps from an SPD's samples to its (unnormalised) source and CES
/// tristimulus integrands, on the caller's own (unconformed) input
/// wavelength grid.
///
/// For an SPD `s` sampled on `input_wavelengths`: `source` and `ces` are
/// built so that, treating `source`/`ces` as (3, n_in) / (99, 3, n_in)
/// row-major matrices W / M, `k = 100 / (W[1] . s)` and
/// `k * (W . s)` / `k * (M[i] . s)` reproduce compute_source_xyz's and
/// compute_ces_xyz's X/Y/Z exactly (TM-30-20 S3.2 Eq. (1)-(4), S3.6 Eq.
/// (21)-(23)) for any `s` aligned with `input_wavelengths`.
struct XyzLinearMaps {
  std::vector<double> ces;    // 99*3*n_in, row-major (i, c, lambda)
  std::vector<double> source; // 3*n_in, row-major (c, lambda)
  std::size_t n_in;           // = input_wavelengths.size()
};

/// Build the linear source/CES tristimulus maps for `input_wavelengths`.
///
/// `tables` must be prepared (via prepare_resampled_tables()) on
/// `Spd(input_wavelengths, ...).wavelengths()` - the S3.5-conformed form
/// of `input_wavelengths` - so its trapezoidal weights, CES reflectance,
/// and 10-deg CMF line up with the conformed grid this function maps back
/// onto the caller's input columns.
///
/// Each conformed-grid point is matched (by exact double comparison) to
/// the input column holding the same wavelength; TM-30-20 S3.5 keeps an
/// in-range input sample's value unchanged in the conformed grid (see
/// Spd::normalize), so every conformed point either equals one input
/// point exactly or is a synthetic zero-fill edge point that matches no
/// input column and is skipped. Columns for input wavelengths outside
/// 380-780 nm, and any zero-fill point, are left at zero.
///
/// @param input_wavelengths  The caller's own wavelength grid (nm), the
///                           same grid `tables` was prepared from (after
///                           S3.5 conforming).
/// @param tables             Pre-resampled tables for
///                           Spd(input_wavelengths, ...).wavelengths()
///                           (see prepare_resampled_tables()).
/// @return XyzLinearMaps with `source` (3, n_in) and `ces` (99, 3, n_in).
XyzLinearMaps xyz_linear_maps(const std::vector<double> &input_wavelengths,
                              const ResampledTables &tables);

} // namespace tm30
