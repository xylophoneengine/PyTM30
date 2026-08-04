#pragma once

/// @file pipeline.hpp
/// TM-30-20 CES colorimetry pipeline - end-to-end integration of Slices 1-4.
///
/// Orchestrates the full sequence:
///   1. Resample CES to SPD grid (Slice 1)
///   2. Compute 2° XYZ for test source → CCT/Duv (Slice 3)
///   3. Generate reference SPD from CCT + basis + 10° CMF y_bar (Slice 4)
///   4. Compute source XYZ for test → normalisation constant kt (Slice 2)
///   5. Compute CES XYZ under test source (Slice 2)
///   6. Compute source XYZ for reference → normalisation kr
///   7. Compute CES XYZ under reference illuminant
///
/// TM-30-20 §3.4: Color Evaluation Samples
/// TM-30-20 §3.6: Calculation of Tristimulus Values

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
/// TM-30-20 §3.3: CCT, Duv
/// TM-30-20 §3.3: Reference illuminant SPD
/// TM-30-20 §3.6: Eq. (21)-(23) - CES tristimulus values under test source
/// TM-30-20 §3.6: Eq. (25)-(27) - CES tristimulus values under reference
struct CesColorimetryResult {
  double cct;                               // TM-30-20 §3.3
  double duv;                               // TM-30-20 §3.3
  std::vector<double> reference_spd_values; // Reference illuminant SPD values
                                            // TM-30-20 §3.3 Eq. (13)-(16)
  std::array<XyzTriple, 99> xyz_test_ces;   // TM-30-20 §3.6 Eq. (21)-(23)
  std::array<XyzTriple, 99> xyz_ref_ces;    // TM-30-20 §3.6 Eq. (25)-(27)
  std::array<Cam02Ucs, 99> jab_test_ces;    // TM-30-20 §3.7.1 (test adaptation)
  std::array<Cam02Ucs, 99>
      jab_ref_ces;    // TM-30-20 §3.7.1 (reference adaptation)
  HueBins hue_bins;   // TM-30-20 §4.3 (16 hue-angle bins)
  double delta_e_avg; // TM-30-20 §4.1 - mean of 99 ΔE′ values
  double Rf;          // TM-30-20 §4.1 Eq. (54) - fidelity index
  GamutResult gamut;  // TM-30-20 §4.4-§4.8 - gamut, local metrics, CVG
  std::array<double, 99>
      rf_cesi;    // TM-30-20 §4.2 Eq. (56) - per-sample fidelity
  double rf_skin; // TM-30-20 §4.2 - skin fidelity (CES15+18 avg)
};

/// Run the full CES colorimetry pipeline for a test SPD.
///
/// Steps (TM-30-20 §3.4, §3.6):
///   1. Resample CES reflectance data to the SPD wavelength grid.
///   2. Resample both 2° and 10° CMF data to the SPD wavelength grid.
///   3. Compute the test source's own 2° XYZ → CCT and Duv.
///   4. Generate the reference illuminant SPD from the CCT.
///   5. Compute the test source's 10° XYZ → normalisation constant kt.
///   6. Compute CES XYZ under the test source (with kt).
///   7. Compute the reference source's 10° XYZ → normalisation kr.
///   8. Compute CES XYZ under the reference illuminant (with kr).
///   9. Compute CIECAM02 J'a'b' under test source adaptation.
///  10. Compute CIECAM02 J'a'b' under reference illuminant adaptation.
///  11. Compute ΔE′ color differences and Rf fidelity index.
///
/// @param spd_wavelengths  Test SPD wavelength grid (nm), monotonically
/// increasing.
/// @param spd_values       Test SPD spectral power values St(λ).
/// @param cmf_2deg         CIE 1931 2° CMF data (for CCT computation).
/// @param cmf_10deg        CIE 1964 10° CMF data (for tristimulus integration).
/// @param ces_data         CES reflectance data (99 samples, 1-nm native).
/// @param daylight_basis   Daylight basis vectors (S₀, S₁, S₂).
/// @param planckian_lut    Planckian locus LUT (CIE 1931 2° observer).
///
/// @return CesColorimetryResult with CCT, Duv, reference SPD, and all CES XYZ
/// values.
///
/// TM-30-20 §3.4, §3.6
CesColorimetryResult compute_ces_colorimetry(
    const std::vector<double> &spd_wavelengths,
    const std::vector<double> &spd_values, const CmfData &cmf_2deg,
    const CmfData &cmf_10deg, const CesData &ces_data,
    const DaylightBasis &daylight_basis, const PlanckianLut &planckian_lut);

// ══════════════════════════════════════════════════════════════════════════
//  Grid-fixed caching - precompute once, reuse across many SPDs that share
//  the same wavelength grid (the overwhelmingly common case in practice).
// ══════════════════════════════════════════════════════════════════════════

/// Wavelength-grid-dependent tables, precomputed once and reused across
/// many SPD evaluations that share the same grid.
///
/// Building this is the expensive part (99 CES + 3 CMF curves resampled,
/// daylight basis resampled) - evaluating with it
/// (compute_ces_colorimetry_cached) skips all of that work entirely.
struct ResampledTables {
  std::vector<double> wavelengths; // Target grid these tables are resampled to.
  CesData ces;       // CES reflectance data, resampled to `wavelengths`.
  CmfData cmf_2deg;  // CIE 1931 2° CMF, resampled to `wavelengths`.
  CmfData cmf_10deg; // CIE 1964 10° CMF, resampled to `wavelengths`.
  DaylightBasis
      daylight_basis; // Daylight basis (S0,S1,S2), resampled to `wavelengths`.
};

/// Precompute (once) all wavelength-grid-dependent resampled tables for a
/// target grid, from the raw (native-grid) source tables.
///
/// @param target_wavelengths  The wavelength grid to resample everything to.
/// @param cmf_2deg_src        Raw (native-grid) CIE 1931 2° CMF data.
/// @param cmf_10deg_src       Raw (native-grid) CIE 1964 10° CMF data.
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
/// @param spd_values    Test SPD spectral power values St(λ), aligned with
///                      tables.wavelengths.
/// @param tables        Pre-resampled CES/CMF/daylight-basis tables (see
///                      prepare_resampled_tables()).
/// @param planckian_lut Planckian locus LUT (CIE 1931 2° observer).
///
/// @return CesColorimetryResult, numerically identical to what
///         compute_ces_colorimetry() would produce for the same SPD on
///         the same grid.
CesColorimetryResult
compute_ces_colorimetry_cached(const std::vector<double> &spd_values,
                               const ResampledTables &tables,
                               const PlanckianLut &planckian_lut);

} // namespace tm30
