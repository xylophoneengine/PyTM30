// TM-30-20 CES colorimetry pipeline - end-to-end integration.
// Orchestrates resampling, CCT computation, reference generation,
// and CES tristimulus integration.
//
// TM-30-20 §3.3: CCT and Reference Illuminant
// TM-30-20 §3.4: Color Evaluation Samples
// TM-30-20 §3.6: Calculation of Tristimulus Values
#include "tm30/pipeline.hpp"

#include "tm30/ciecam02.hpp"
#include "tm30/gamut.hpp"
#include "tm30/hue_bins.hpp"
#include "tm30/integrate.hpp"
#include "tm30/metrics.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace tm30 {

CesColorimetryResult compute_ces_colorimetry(
    const std::vector<double> &spd_wavelengths,
    const std::vector<double> &spd_values, const CmfData &cmf_2deg,
    const CmfData &cmf_10deg, const CesData &ces_data,
    const DaylightBasis &daylight_basis, const PlanckianLut &planckian_lut) {

  // -- Step 1: Resample CES reflectance data to SPD wavelength grid ------
  // TM-30-20 §3.5: "Linear interpolation shall be used."
  const CesData ces_resampled = resample_ces(spd_wavelengths, ces_data);
  // TM-30-20 §3.5

  // -- Step 2: Resample CMF data to SPD wavelength grid ------------------
  // TM-30-20 §3.1: CIE 1964 10-deg observer for tristimulus integration
  // TM-30-20 §3.3: CIE 1931 2-deg observer for CCT determination
  const CmfData cmf2 = resample_cmf(spd_wavelengths, cmf_2deg);
  const CmfData cmf10 = resample_cmf(spd_wavelengths, cmf_10deg);
  // TM-30-20 §3.5

  // -- Step 3: Compute 2-deg XYZ -> CCT / Duv --------------------------------
  // TM-30-20 §3.3: CCT determined from CIE 1931 2-deg XYZ via Ohno 2014 method
  const SourceXyz src_2deg = compute_source_xyz(
      spd_wavelengths, spd_values, cmf2.x_bar, cmf2.y_bar, cmf2.z_bar);
  // TM-30-20 §3.2 Eq. (1)-(4) with 2-deg observer
  const CctDuvResult cct_duv = compute_cct_duv_from_xyz(
      src_2deg.X, src_2deg.Y, src_2deg.Z, planckian_lut);
  // TM-30-20 §3.3

  // -- Step 4: Generate reference illuminant SPD -------------------------
  // TM-30-20 §3.3 Eq. (13)-(16)
  const std::vector<double> ref_spd = generate_reference_spd(
      cct_duv.cct, spd_wavelengths, daylight_basis, cmf10.y_bar);
  // TM-30-20 §3.3

  // -- Step 5: Compute test source 10-deg XYZ -> normalisation constant kt ---
  // TM-30-20 §3.2 Eq. (4): kt = 100 / integral St(lambda) * ybar10(lambda)
  // dlambda
  const SourceXyz test_10deg = compute_source_xyz(
      spd_wavelengths, spd_values, cmf10.x_bar, cmf10.y_bar, cmf10.z_bar);
  // TM-30-20 §3.2 Eq. (1)-(4) with 10-deg observer

  // -- Step 6: Compute CES XYZ under test source -------------------------
  // TM-30-20 §3.6 Eq. (21)-(23)
  const auto xyz_test =
      compute_ces_xyz(spd_wavelengths, spd_values, ces_resampled, cmf10.x_bar,
                      cmf10.y_bar, cmf10.z_bar, test_10deg.k);
  // TM-30-20 §3.6

  // -- Step 7: Compute reference source 10-deg XYZ -> normalisation kr -------
  // TM-30-20 §3.6: Reference illuminant uses same normalisation formula
  const SourceXyz ref_10deg = compute_source_xyz(
      spd_wavelengths, ref_spd, cmf10.x_bar, cmf10.y_bar, cmf10.z_bar);
  // TM-30-20 §3.6

  // -- Step 8: Compute CES XYZ under reference illuminant ----------------
  // TM-30-20 §3.6 Eq. (25)-(27)
  const auto xyz_ref =
      compute_ces_xyz(spd_wavelengths, ref_spd, ces_resampled, cmf10.x_bar,
                      cmf10.y_bar, cmf10.z_bar, ref_10deg.k);
  // TM-30-20 §3.6

  // -- Assemble result ---------------------------------------------------
  CesColorimetryResult result;
  result.cct = cct_duv.cct;              // TM-30-20 §3.3
  result.duv = cct_duv.duv;              // TM-30-20 §3.3
  result.reference_spd_values = ref_spd; // TM-30-20 §3.3
  result.xyz_test_ces = xyz_test;        // TM-30-20 §3.6 Eq. (21)-(23)
  result.xyz_ref_ces = xyz_ref;          // TM-30-20 §3.6 Eq. (25)-(27)

  // -- Step 9: CIECAM02 J'a'b' under test source adaptation --------------
  // TM-30-20 §3.7.1: Adapting to test source white point (10-deg XYZ)
  {
    const XyzTriple test_white{test_10deg.X, test_10deg.Y, test_10deg.Z};
    result.jab_test_ces = ciecam02_forward(test_white, xyz_test);
  }
  // TM-30-20 §3.7.1

  // -- Step 10: CIECAM02 J'a'b' under reference illuminant adaptation ----
  // TM-30-20 §3.7.1: Adapting to reference illuminant white point (10-deg XYZ)
  {
    const XyzTriple ref_white{ref_10deg.X, ref_10deg.Y, ref_10deg.Z};
    result.jab_ref_ces = ciecam02_forward(ref_white, xyz_ref);
  }
  // TM-30-20 §3.7.1

  // -- Step 11: Compute dE' and Rf ---------------------------------------
  // TM-30-20 §3.8 Eq. (52): dE' for each CES
  const auto delta_e_array =
      compute_delta_e(result.jab_test_ces, result.jab_ref_ces);
  // TM-30-20 §3.8

  // TM-30-20 §4.1 Eq. (53), (54): Rf fidelity index
  const RfResult rf_result = compute_rf(delta_e_array);
  result.delta_e_avg = rf_result.delta_e_avg; // TM-30-20 §4.1
  result.Rf = rf_result.Rf;                   // TM-30-20 §4.1 Eq. (54)

  // -- Step 12: Hue-angle binning -----------------------------------------
  // TM-30-20 §4.3: Assign 99 CES to 16 hue-angle bins based on reference hr
  result.hue_bins = bin_by_hue(result.jab_ref_ces);

  // -- Step 13: Gamut metrics (Rg, local per-bin, CVG) --------------------
  // TM-30-20 §4.4-§4.8
  result.gamut = compute_gamut(result.jab_test_ces, result.jab_ref_ces,
                               delta_e_array, result.hue_bins);

  // -- Step 14: Per-sample fidelity and skin fidelity -----------------
  // TM-30-20 §4.2 Eq. (55)-(56): Rf,CESi for each CES
  result.rf_cesi = compute_rf_cesi(delta_e_array);
  // Rf,skin = (Rf,CES15 + Rf,CES18) / 2 -- PyTM30 research extension
  // informed by TM-30-20 §4.2; not a standardised measure (§1.2, §4.0).
  result.rf_skin = compute_rf_skin(result.rf_cesi);

  return result;
}

// ==========================================================================
//  Grid-fixed caching
// ==========================================================================

ResampledTables
prepare_resampled_tables(const std::vector<double> &target_wavelengths,
                         const CmfData &cmf_2deg_src,
                         const CmfData &cmf_10deg_src, const CesData &ces_src,
                         const DaylightBasis &daylight_basis_src) {

  ResampledTables tables;
  tables.wavelengths = target_wavelengths;
  tables.ces = resample_ces(target_wavelengths, ces_src);
  tables.cmf_2deg = resample_cmf(target_wavelengths, cmf_2deg_src);
  tables.cmf_10deg = resample_cmf(target_wavelengths, cmf_10deg_src);
  tables.daylight_basis =
      resample_daylight_basis(target_wavelengths, daylight_basis_src);
  return tables;
}

CesColorimetryResult
compute_ces_colorimetry_cached(const std::vector<double> &spd_values,
                               const ResampledTables &tables,
                               const PlanckianLut &planckian_lut) {

  const std::vector<double> &spd_wavelengths = tables.wavelengths;
  const CesData &ces_resampled = tables.ces;
  const CmfData &cmf2 = tables.cmf_2deg;
  const CmfData &cmf10 = tables.cmf_10deg;

  // -- Step 3: Compute 2-deg XYZ -> CCT / Duv --------------------------------
  // TM-30-20 §3.3: CCT determined from CIE 1931 2-deg XYZ via Ohno 2014 method
  const SourceXyz src_2deg = compute_source_xyz(
      spd_wavelengths, spd_values, cmf2.x_bar, cmf2.y_bar, cmf2.z_bar);
  // TM-30-20 §3.2 Eq. (1)-(4) with 2-deg observer
  const CctDuvResult cct_duv = compute_cct_duv_from_xyz(
      src_2deg.X, src_2deg.Y, src_2deg.Z, planckian_lut);
  // TM-30-20 §3.3

  // -- Step 4: Generate reference illuminant SPD -------------------------
  // TM-30-20 §3.3 Eq. (13)-(16). Daylight basis is already resampled to
  // spd_wavelengths (tables.daylight_basis), so skip the internal
  // interpolation that generate_cie_d() would otherwise redo.
  const std::vector<double> ref_spd = generate_reference_spd(
      cct_duv.cct, spd_wavelengths, tables.daylight_basis, cmf10.y_bar,
      /*already_resampled=*/true);
  // TM-30-20 §3.3

  // -- Step 5: Compute test source 10-deg XYZ -> normalisation constant kt ---
  // TM-30-20 §3.2 Eq. (4): kt = 100 / integral St(lambda) * ybar10(lambda)
  // dlambda
  const SourceXyz test_10deg = compute_source_xyz(
      spd_wavelengths, spd_values, cmf10.x_bar, cmf10.y_bar, cmf10.z_bar);
  // TM-30-20 §3.2 Eq. (1)-(4) with 10-deg observer

  // -- Step 6: Compute CES XYZ under test source -------------------------
  // TM-30-20 §3.6 Eq. (21)-(23)
  const auto xyz_test =
      compute_ces_xyz(spd_wavelengths, spd_values, ces_resampled, cmf10.x_bar,
                      cmf10.y_bar, cmf10.z_bar, test_10deg.k);
  // TM-30-20 §3.6

  // -- Step 7: Compute reference source 10-deg XYZ -> normalisation kr -------
  // TM-30-20 §3.6: Reference illuminant uses same normalisation formula
  const SourceXyz ref_10deg = compute_source_xyz(
      spd_wavelengths, ref_spd, cmf10.x_bar, cmf10.y_bar, cmf10.z_bar);
  // TM-30-20 §3.6

  // -- Step 8: Compute CES XYZ under reference illuminant ----------------
  // TM-30-20 §3.6 Eq. (25)-(27)
  const auto xyz_ref =
      compute_ces_xyz(spd_wavelengths, ref_spd, ces_resampled, cmf10.x_bar,
                      cmf10.y_bar, cmf10.z_bar, ref_10deg.k);
  // TM-30-20 §3.6

  // -- Assemble result ---------------------------------------------------
  CesColorimetryResult result;
  result.cct = cct_duv.cct;              // TM-30-20 §3.3
  result.duv = cct_duv.duv;              // TM-30-20 §3.3
  result.reference_spd_values = ref_spd; // TM-30-20 §3.3
  result.xyz_test_ces = xyz_test;        // TM-30-20 §3.6 Eq. (21)-(23)
  result.xyz_ref_ces = xyz_ref;          // TM-30-20 §3.6 Eq. (25)-(27)

  // -- Step 9: CIECAM02 J'a'b' under test source adaptation --------------
  // TM-30-20 §3.7.1: Adapting to test source white point (10-deg XYZ)
  {
    const XyzTriple test_white{test_10deg.X, test_10deg.Y, test_10deg.Z};
    result.jab_test_ces = ciecam02_forward(test_white, xyz_test);
  }
  // TM-30-20 §3.7.1

  // -- Step 10: CIECAM02 J'a'b' under reference illuminant adaptation ----
  // TM-30-20 §3.7.1: Adapting to reference illuminant white point (10-deg XYZ)
  {
    const XyzTriple ref_white{ref_10deg.X, ref_10deg.Y, ref_10deg.Z};
    result.jab_ref_ces = ciecam02_forward(ref_white, xyz_ref);
  }
  // TM-30-20 §3.7.1

  // -- Step 11: Compute dE' and Rf ---------------------------------------
  // TM-30-20 §3.8 Eq. (52): dE' for each CES
  const auto delta_e_array =
      compute_delta_e(result.jab_test_ces, result.jab_ref_ces);
  // TM-30-20 §3.8

  // TM-30-20 §4.1 Eq. (53), (54): Rf fidelity index
  const RfResult rf_result = compute_rf(delta_e_array);
  result.delta_e_avg = rf_result.delta_e_avg; // TM-30-20 §4.1
  result.Rf = rf_result.Rf;                   // TM-30-20 §4.1 Eq. (54)

  // -- Step 12: Hue-angle binning -----------------------------------------
  // TM-30-20 §4.3: Assign 99 CES to 16 hue-angle bins based on reference hr
  result.hue_bins = bin_by_hue(result.jab_ref_ces);

  // -- Step 13: Gamut metrics (Rg, local per-bin, CVG) --------------------
  // TM-30-20 §4.4-§4.8
  result.gamut = compute_gamut(result.jab_test_ces, result.jab_ref_ces,
                               delta_e_array, result.hue_bins);

  // -- Step 14: Per-sample fidelity and skin fidelity -----------------
  // TM-30-20 §4.2 Eq. (55)-(56): Rf,CESi for each CES
  result.rf_cesi = compute_rf_cesi(delta_e_array);
  // Rf,skin = (Rf,CES15 + Rf,CES18) / 2 -- PyTM30 research extension
  // informed by TM-30-20 §4.2; not a standardised measure (§1.2, §4.0).
  result.rf_skin = compute_rf_skin(result.rf_cesi);

  return result;
}

} // namespace tm30
