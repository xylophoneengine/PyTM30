// TM-30-20 CES colorimetry pipeline - end-to-end integration.
// Orchestrates resampling, CCT computation, reference generation,
// and CES tristimulus integration.
//
// TM-30-20 S3.3: CCT and Reference Illuminant
// TM-30-20 S3.4: Color Evaluation Samples
// TM-30-20 S3.6: Calculation of Tristimulus Values
#include "tm30/pipeline.hpp"

#include "tm30/ciecam02.hpp"
#include "tm30/gamut.hpp"
#include "tm30/hue_bins.hpp"
#include "tm30/integrate.hpp"
#include "tm30/metrics.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tm30 {

CesColorimetryResult compute_ces_colorimetry(
    const std::vector<double> &spd_wavelengths,
    const std::vector<double> &spd_values, const CmfData &cmf_2deg,
    const CmfData &cmf_10deg, const CesData &ces_data,
    const DaylightBasis &daylight_basis, const PlanckianLut &planckian_lut) {

  // -- Step 1: Resample CES reflectance data to SPD wavelength grid ------
  // TM-30-20 S3.5 requires linear interpolation.
  const CesData ces_resampled = resample_ces(spd_wavelengths, ces_data);
  // TM-30-20 S3.5

  // -- Step 2: Resample CMF data to SPD wavelength grid ------------------
  // TM-30-20 S3.1: CIE 1964 10-deg observer for tristimulus integration
  // TM-30-20 S3.3: CIE 1931 2-deg observer for CCT determination
  const CmfData cmf2 = resample_cmf(spd_wavelengths, cmf_2deg);
  const CmfData cmf10 = resample_cmf(spd_wavelengths, cmf_10deg);
  // TM-30-20 S3.5

  // -- Step 3: Compute 2-deg XYZ -> CCT / Duv --------------------------------
  // TM-30-20 S3.3: CCT determined from CIE 1931 2-deg XYZ via Ohno 2014 method
  const SourceXyz src_2deg = compute_source_xyz(
      spd_wavelengths, spd_values, cmf2.x_bar, cmf2.y_bar, cmf2.z_bar);
  // TM-30-20 S3.2 Eq. (1)-(4) with 2-deg observer
  const CctDuvResult cct_duv = compute_cct_duv_from_xyz(
      src_2deg.X, src_2deg.Y, src_2deg.Z, planckian_lut);
  // TM-30-20 S3.3

  // -- Step 4: Generate reference illuminant SPD -------------------------
  // TM-30-20 S3.3 Eq. (13)-(16)
  const std::vector<double> ref_spd = generate_reference_spd(
      cct_duv.cct, spd_wavelengths, daylight_basis, cmf10.y_bar);
  // TM-30-20 S3.3

  // -- Step 5: Compute test source 10-deg XYZ -> normalisation constant kt ---
  // TM-30-20 S3.2 Eq. (4): kt = 100 / integral St(lambda) * ybar10(lambda)
  // dlambda
  const SourceXyz test_10deg = compute_source_xyz(
      spd_wavelengths, spd_values, cmf10.x_bar, cmf10.y_bar, cmf10.z_bar);
  // TM-30-20 S3.2 Eq. (1)-(4) with 10-deg observer

  // -- Step 6: Compute CES XYZ under test source -------------------------
  // TM-30-20 S3.6 Eq. (21)-(23)
  const auto xyz_test =
      compute_ces_xyz(spd_wavelengths, spd_values, ces_resampled, cmf10.x_bar,
                      cmf10.y_bar, cmf10.z_bar, test_10deg.k);
  // TM-30-20 S3.6

  // -- Step 7: Compute reference source 10-deg XYZ -> normalisation kr -------
  // TM-30-20 S3.6: Reference illuminant uses same normalisation formula
  const SourceXyz ref_10deg = compute_source_xyz(
      spd_wavelengths, ref_spd, cmf10.x_bar, cmf10.y_bar, cmf10.z_bar);
  // TM-30-20 S3.6

  // -- Step 8: Compute CES XYZ under reference illuminant ----------------
  // TM-30-20 S3.6 Eq. (25)-(27)
  const auto xyz_ref =
      compute_ces_xyz(spd_wavelengths, ref_spd, ces_resampled, cmf10.x_bar,
                      cmf10.y_bar, cmf10.z_bar, ref_10deg.k);
  // TM-30-20 S3.6

  // -- Assemble result ---------------------------------------------------
  CesColorimetryResult result;
  result.cct = cct_duv.cct;              // TM-30-20 S3.3
  result.duv = cct_duv.duv;              // TM-30-20 S3.3
  result.reference_spd_values = ref_spd; // TM-30-20 S3.3
  result.xyz_test_ces = xyz_test;        // TM-30-20 S3.6 Eq. (21)-(23)
  result.xyz_ref_ces = xyz_ref;          // TM-30-20 S3.6 Eq. (25)-(27)

  // -- Step 9: CIECAM02 J'a'b' under test source adaptation --------------
  // TM-30-20 S3.7.1: Adapting to test source white point (10-deg XYZ)
  {
    const XyzTriple test_white{test_10deg.X, test_10deg.Y, test_10deg.Z};
    result.jab_test_ces = ciecam02_forward(test_white, xyz_test);
  }
  // TM-30-20 S3.7.1

  // -- Step 10: CIECAM02 J'a'b' under reference illuminant adaptation ----
  // TM-30-20 S3.7.1: Adapting to reference illuminant white point (10-deg XYZ)
  {
    const XyzTriple ref_white{ref_10deg.X, ref_10deg.Y, ref_10deg.Z};
    result.jab_ref_ces = ciecam02_forward(ref_white, xyz_ref);
  }
  // TM-30-20 S3.7.1

  // -- Step 11: Compute dE' and Rf ---------------------------------------
  // TM-30-20 S3.8 Eq. (52): dE' for each CES
  const auto delta_e_array =
      compute_delta_e(result.jab_test_ces, result.jab_ref_ces);
  // TM-30-20 S3.8

  // TM-30-20 S4.1 Eq. (53), (54): Rf fidelity index
  const RfResult rf_result = compute_rf(delta_e_array);
  result.delta_e_avg = rf_result.delta_e_avg; // TM-30-20 S4.1
  result.Rf = rf_result.Rf;                   // TM-30-20 S4.1 Eq. (54)

  // -- Step 12: Hue-angle binning -----------------------------------------
  // TM-30-20 S4.3: Assign 99 CES to 16 hue-angle bins based on reference hr.
  // The normalised hue angles are kept rather than discarded - step 13's
  // CVG sub-step needs the identical 99 values.
  HueAngles hue_angles{};
  result.hue_bins = bin_by_hue(result.jab_ref_ces, &hue_angles);

  // -- Step 13: Gamut metrics (Rg, local per-bin, CVG) --------------------
  // TM-30-20 S4.4-S4.8. (The hue angles argument is unused since the CVG
  // reference points became bin centres; see compute_cvg_coordinates().)
  result.gamut = compute_gamut(result.jab_test_ces, result.jab_ref_ces,
                               delta_e_array, result.hue_bins, &hue_angles);

  // -- Step 14: Per-sample fidelity and skin fidelity -----------------
  // TM-30-20 S4.2 Eq. (55)-(56): Rf,CESi for each CES
  result.rf_cesi = compute_rf_cesi(delta_e_array);
  // Rf,skin = (Rf,CES15 + Rf,CES18) / 2 -- PyTM30 research extension
  // informed by TM-30-20 S4.2; not a standardised measure (S1.2, S4.0).
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
  // TM-30-20 S3.3 Eq. (6): grid-fixed lambda^(-5), so generate_planckian()
  // does not repeat 401 std::pow calls per SPD.
  tables.lambda_pow_m5 = planckian_lambda_pow_table(target_wavelengths);
  // TM-30-20 S3.6 Eq. (21)-(23): grid-fixed trapezoidal weights, so
  // compute_ces_xyz() does not rebuild them twice per SPD.
  tables.trapezoidal_w = trapezoidal_weights(target_wavelengths);
  return tables;
}

ReferenceColorimetry
compute_reference_colorimetry(double cct, const ResampledTables &tables) {
  const std::vector<double> &spd_wavelengths = tables.wavelengths;
  const CesData &ces_resampled = tables.ces;
  const CmfData &cmf10 = tables.cmf_10deg;

  ReferenceColorimetry result;

  // -- Step 4: Generate reference illuminant SPD -------------------------
  // TM-30-20 S3.3 Eq. (13)-(16). Daylight basis is already resampled to
  // spd_wavelengths (tables.daylight_basis), so skip the internal
  // interpolation that generate_cie_d() would otherwise redo. Likewise the
  // grid-fixed lambda^(-5) factor of Eq. (6) is precomputed on the tables,
  // so generate_planckian() does not rebuild it per SPD.
  result.spd = generate_reference_spd(
      cct, spd_wavelengths, tables.daylight_basis, cmf10.y_bar,
      /*already_resampled=*/true, &tables.lambda_pow_m5);
  // TM-30-20 S3.3

  // -- Step 7: Compute reference source 10-deg XYZ -> normalisation kr -------
  // TM-30-20 S3.6: Reference illuminant uses same normalisation formula
  const SourceXyz ref_10deg = compute_source_xyz(
      spd_wavelengths, result.spd, cmf10.x_bar, cmf10.y_bar, cmf10.z_bar);
  // TM-30-20 S3.6

  // -- Step 8: Compute CES XYZ under reference illuminant ----------------
  // TM-30-20 S3.6 Eq. (25)-(27)
  const auto xyz_ref = compute_ces_xyz(
      spd_wavelengths, result.spd, ces_resampled, cmf10.x_bar, cmf10.y_bar,
      cmf10.z_bar, ref_10deg.k, &tables.trapezoidal_w);
  // TM-30-20 S3.6

  result.white = XyzTriple{ref_10deg.X, ref_10deg.Y, ref_10deg.Z};
  result.xyz_ces = xyz_ref; // TM-30-20 S3.6 Eq. (25)-(27)

  // -- Step 10: CIECAM02 J'a'b' under reference illuminant adaptation ----
  // TM-30-20 S3.7.1: Adapting to reference illuminant white point (10-deg XYZ)
  result.jab_ces = ciecam02_forward(result.white, xyz_ref);
  // TM-30-20 S3.7.1

  return result;
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
  // TM-30-20 S3.3: CCT determined from CIE 1931 2-deg XYZ via Ohno 2014 method
  const SourceXyz src_2deg = compute_source_xyz(
      spd_wavelengths, spd_values, cmf2.x_bar, cmf2.y_bar, cmf2.z_bar);
  // TM-30-20 S3.2 Eq. (1)-(4) with 2-deg observer
  const CctDuvResult cct_duv = compute_cct_duv_from_xyz(
      src_2deg.X, src_2deg.Y, src_2deg.Z, planckian_lut);
  // TM-30-20 S3.3

  // -- Steps 4, 7, 8, 10: Reference-side colorimetry ----------------------
  // (reference SPD, reference source XYZ, reference CES XYZ, reference
  // CAM02-UCS J'a'b') - see compute_reference_colorimetry().
  ReferenceColorimetry ref = compute_reference_colorimetry(cct_duv.cct, tables);

  // -- Step 5: Compute test source 10-deg XYZ -> normalisation constant kt ---
  // TM-30-20 S3.2 Eq. (4): kt = 100 / integral St(lambda) * ybar10(lambda)
  // dlambda
  const SourceXyz test_10deg = compute_source_xyz(
      spd_wavelengths, spd_values, cmf10.x_bar, cmf10.y_bar, cmf10.z_bar);
  // TM-30-20 S3.2 Eq. (1)-(4) with 10-deg observer

  // -- Step 6: Compute CES XYZ under test source -------------------------
  // TM-30-20 S3.6 Eq. (21)-(23). The trapezoidal weights are grid-fixed,
  // so they come off the tables instead of being rebuilt per SPD.
  const auto xyz_test = compute_ces_xyz(
      spd_wavelengths, spd_values, ces_resampled, cmf10.x_bar, cmf10.y_bar,
      cmf10.z_bar, test_10deg.k, &tables.trapezoidal_w);
  // TM-30-20 S3.6

  // -- Assemble result ---------------------------------------------------
  CesColorimetryResult result;
  result.cct = cct_duv.cct;                         // TM-30-20 S3.3
  result.duv = cct_duv.duv;                         // TM-30-20 S3.3
  result.reference_spd_values = std::move(ref.spd); // TM-30-20 S3.3
  result.xyz_test_ces = xyz_test;   // TM-30-20 S3.6 Eq. (21)-(23)
  result.xyz_ref_ces = ref.xyz_ces; // TM-30-20 S3.6 Eq. (25)-(27)

  // -- Step 9: CIECAM02 J'a'b' under test source adaptation --------------
  // TM-30-20 S3.7.1: Adapting to test source white point (10-deg XYZ)
  {
    const XyzTriple test_white{test_10deg.X, test_10deg.Y, test_10deg.Z};
    result.jab_test_ces = ciecam02_forward(test_white, xyz_test);
  }
  // TM-30-20 S3.7.1

  // Step 10 (reference-adapted CAM02-UCS J'a'b') is already computed by
  // compute_reference_colorimetry() above.
  result.jab_ref_ces = ref.jab_ces;

  // -- Step 11: Compute dE' and Rf ---------------------------------------
  // TM-30-20 S3.8 Eq. (52): dE' for each CES
  const auto delta_e_array =
      compute_delta_e(result.jab_test_ces, result.jab_ref_ces);
  // TM-30-20 S3.8

  // TM-30-20 S4.1 Eq. (53), (54): Rf fidelity index
  const RfResult rf_result = compute_rf(delta_e_array);
  result.delta_e_avg = rf_result.delta_e_avg; // TM-30-20 S4.1
  result.Rf = rf_result.Rf;                   // TM-30-20 S4.1 Eq. (54)

  // -- Step 12: Hue-angle binning -----------------------------------------
  // TM-30-20 S4.3: Assign 99 CES to 16 hue-angle bins based on reference hr.
  // The normalised hue angles are kept rather than discarded - step 13's
  // CVG sub-step needs the identical 99 values.
  HueAngles hue_angles{};
  result.hue_bins = bin_by_hue(result.jab_ref_ces, &hue_angles);

  // -- Step 13: Gamut metrics (Rg, local per-bin, CVG) --------------------
  // TM-30-20 S4.4-S4.8. (The hue angles argument is unused since the CVG
  // reference points became bin centres; see compute_cvg_coordinates().)
  result.gamut = compute_gamut(result.jab_test_ces, result.jab_ref_ces,
                               delta_e_array, result.hue_bins, &hue_angles);

  // -- Step 14: Per-sample fidelity and skin fidelity -----------------
  // TM-30-20 S4.2 Eq. (55)-(56): Rf,CESi for each CES
  result.rf_cesi = compute_rf_cesi(delta_e_array);
  // Rf,skin = (Rf,CES15 + Rf,CES18) / 2 -- PyTM30 research extension
  // informed by TM-30-20 S4.2; not a standardised measure (S1.2, S4.0).
  result.rf_skin = compute_rf_skin(result.rf_cesi);

  return result;
}

// ==========================================================================
//  Linear tristimulus maps
// ==========================================================================

XyzLinearMaps xyz_linear_maps(const std::vector<double> &input_wavelengths,
                              const ResampledTables &tables) {
  if (tables.ces.samples.size() != 99) {
    throw std::invalid_argument(
        "xyz_linear_maps requires exactly 99 CES samples in tables.ces, "
        "got " +
        std::to_string(tables.ces.samples.size()));
  }

  const std::size_t n_in = input_wavelengths.size();
  const std::size_t n_conf = tables.wavelengths.size();

  XyzLinearMaps maps;
  maps.n_in = n_in;
  // Columns for a zero-filled conformed point, or an input wavelength
  // outside 380-780 nm, carry no integrand contribution and stay zero.
  // TM-30-20 S3.2 Eq. (1)-(4): source integrand.
  maps.source.assign(3 * n_in, 0.0);
  // TM-30-20 S3.6 Eq. (21)-(23): CES integrand.
  maps.ces.assign(99 * 3 * n_in, 0.0);

  const std::vector<double> &w = tables.trapezoidal_w;
  const std::vector<double> &xbar = tables.cmf_10deg.x_bar;
  const std::vector<double> &ybar = tables.cmf_10deg.y_bar;
  const std::vector<double> &zbar = tables.cmf_10deg.z_bar;

  // Merge walk: both input_wavelengths and tables.wavelengths are
  // strictly increasing (Spd validation), so a single left-to-right scan
  // finds every exact match in O(n_in + n_conf) instead of an O(n_in *
  // n_conf) search. TM-30-20 S3.5: an in-range input sample's wavelength
  // is unchanged in the conformed grid (Spd::normalize), so a match here
  // is always an exact double comparison, never a nearest-point pick.
  std::size_t p = 0; // next unconsidered index into input_wavelengths
  for (std::size_t j = 0; j < n_conf; ++j) {
    const double lam = tables.wavelengths[j];
    while (p < n_in && input_wavelengths[p] < lam) {
      ++p;
    }
    if (p >= n_in || input_wavelengths[p] != lam) {
      continue; // synthetic zero-fill point: no matching input column
    }
    const std::size_t col = p;

    // TM-30-20 S3.2 Eq. (1)-(4): source[c][col] is the per-column
    // integrand for X/Y/Z_c = k * sum_j w[j]*S(lambda_j)*cmf_c(lambda_j).
    const double wj = w[j];
    maps.source[0 * n_in + col] = wj * xbar[j];
    maps.source[1 * n_in + col] = wj * ybar[j];
    maps.source[2 * n_in + col] = wj * zbar[j];

    // TM-30-20 S3.6 Eq. (21)-(23): ces[(i,c)][col] is the per-column
    // integrand for CES i's X/Y/Z_c = k * sum_j w[j]*S(lambda_j)*
    // R_i(lambda_j)*cmf_c(lambda_j).
    for (std::size_t i = 0; i < 99; ++i) {
      const double wr = wj * tables.ces.samples[i][j];
      maps.ces[(i * 3 + 0) * n_in + col] = wr * xbar[j];
      maps.ces[(i * 3 + 1) * n_in + col] = wr * ybar[j];
      maps.ces[(i * 3 + 2) * n_in + col] = wr * zbar[j];
    }
  }

  return maps;
}

} // namespace tm30
