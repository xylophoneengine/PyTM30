// CIE 1964 10-deg tristimulus value computation.
// TM-30-20 S3.1: Colorimetric Observer
// TM-30-20 S3.2: Test Source - tristimulus values
// TM-30-20 S3.6: Calculation of Tristimulus Values
#include "tm30/xyz.hpp"
#include "tm30/integrate.hpp"
#include "tm30/spd.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility> // std::move
#include <vector>

namespace tm30 {

SourceXyz compute_source_xyz(const std::vector<double> &spd_wavelengths,
                             const std::vector<double> &spd_values,
                             const std::vector<double> &cmf_x_bar,
                             const std::vector<double> &cmf_y_bar,
                             const std::vector<double> &cmf_z_bar) {
  const std::size_t n = spd_wavelengths.size();

  if (n < 2) {
    throw std::invalid_argument(
        "compute_source_xyz requires at least 2 wavelength points");
  }

  // Build integrand for numerator of k: St(lambda) * ybar10(lambda)
  // TM-30-20 S3.2 Eq. (4): k = 100 / integral St(lambda) * ybar10(lambda)
  // dlambda
  std::vector<double> st_times_ybar(n);
  for (std::size_t i = 0; i < n; ++i) {
    st_times_ybar[i] = spd_values[i] * cmf_y_bar[i];
  }

  const double integral_st_ybar =
      trapezoidal_integrate(spd_wavelengths, st_times_ybar);

  // TM-30-20 S3.2 Eq. (4)
  const double k = 100.0 / integral_st_ybar;

  // Build integrands and integrate
  // TM-30-20 S3.2 Eq. (1): X = k * integral St(lambda) * xbar10(lambda) dlambda
  // TM-30-20 S3.2 Eq. (2): Y = k * integral St(lambda) * ybar10(lambda) dlambda
  // TM-30-20 S3.2 Eq. (3): Z = k * integral St(lambda) * zbar10(lambda) dlambda

  std::vector<double> st_times_xbar(n);
  std::vector<double> st_times_zbar(n);
  for (std::size_t i = 0; i < n; ++i) {
    st_times_xbar[i] = spd_values[i] * cmf_x_bar[i];
    st_times_zbar[i] = spd_values[i] * cmf_z_bar[i];
  }

  const double integral_st_xbar =
      trapezoidal_integrate(spd_wavelengths, st_times_xbar);
  // Reuse st_times_ybar already computed above
  const double integral_st_zbar =
      trapezoidal_integrate(spd_wavelengths, st_times_zbar);

  SourceXyz result;
  result.X = k * integral_st_xbar; // TM-30-20 S3.2 Eq. (1)
  result.Y = k * integral_st_ybar; // TM-30-20 S3.2 Eq. (2)
  result.Z = k * integral_st_zbar; // TM-30-20 S3.2 Eq. (3)
  result.k = k;                    // TM-30-20 S3.2 Eq. (4)

  return result;
}

std::array<XyzTriple, 99>
compute_ces_xyz(const std::vector<double> &spd_wavelengths,
                const std::vector<double> &spd_values, const CesData &ces_data,
                const std::vector<double> &cmf_x_bar,
                const std::vector<double> &cmf_y_bar,
                const std::vector<double> &cmf_z_bar, double k) {

  const std::size_t n = spd_wavelengths.size();

  if (ces_data.samples.size() != 99) {
    throw std::invalid_argument(
        "compute_ces_xyz requires exactly 99 CES samples, got " +
        std::to_string(ces_data.samples.size()));
  }

  std::array<XyzTriple, 99> result;

  // Trapezoidal weights for this wavelength grid depend only on the grid
  // itself, not on any CES sample - compute once per call (not once per
  // CES) and reuse for all 99 samples.
  // TM-30-20 S3.6: sum_i w[i]*f[i] == trapezoidal integration of f over
  // spd_wavelengths.
  const std::vector<double> w = trapezoidal_weights(spd_wavelengths);

  // St(lambda)*xbar10(lambda)/ybar10(lambda)/zbar10(lambda), pre-multiplied by
  // the trapezoidal weight, is also CES-independent - hoist it out of the
  // 99-CES loop so it is computed once per call instead of once per CES per
  // channel.
  std::vector<double> swx(n), swy(n), swz(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double sw = w[i] * spd_values[i];
    swx[i] = sw * cmf_x_bar[i];
    swy[i] = sw * cmf_y_bar[i];
    swz[i] = sw * cmf_z_bar[i];
  }

  for (std::size_t ces_idx = 0; ces_idx < 99; ++ces_idx) {
    const auto &reflectance = ces_data.samples[ces_idx];

    // TM-30-20 S3.6 Eq. (21)-(23): fused dot-product accumulators.
    double X = 0.0, Y = 0.0, Z = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      X += reflectance[i] * swx[i];
      Y += reflectance[i] * swy[i];
      Z += reflectance[i] * swz[i];
    }

    // TM-30-20 S3.6 Eq. (21)-(23):
    //   X_i = k * integral St(lambda) * R_i(lambda) * xbar10(lambda) dlambda
    //   Y_i = k * integral St(lambda) * R_i(lambda) * ybar10(lambda) dlambda
    //   Z_i = k * integral St(lambda) * R_i(lambda) * zbar10(lambda) dlambda
    result[ces_idx] = XyzTriple{k * X, k * Y, k * Z};
  }

  return result;
}

// -- Convenience functions ----------------------------------------------

namespace {

/// [lo, hi] inclusive indices into an existing wavelength array that fall
/// within [lambda_min, lambda_max]. Snaps inward to existing grid points -
/// never synthesizes a new point at the exact boundary requested.
struct ClipIndices {
  std::size_t lo;
  std::size_t hi;
};

// CIE/SI maximum luminous efficacy Km = 683 lm/W - a general photometric
// constant, not TM-30-20-specific (the spec works entirely in relative,
// Y=100-normalized colorimetry and never defines this). Matches this
// codebase's existing K=683.0 "photometric absolute" convention on
// spd_to_xyz. Whitelisted in tools/check_constants_whitelist.txt rather
// than cited to a TM-30-20 section, since no such section exists.
constexpr double kMaxLuminousEfficacy = 683.0;

/// Compute the [lo, hi] index range into `spd_wavelengths` that falls
/// within [lambda_min, lambda_max]. Both bounds optional; unset means the
/// full range. Throws if the resulting range has fewer than 2 points.
ClipIndices compute_clip_indices(const std::vector<double> &spd_wavelengths,
                                 std::optional<double> lambda_min,
                                 std::optional<double> lambda_max) {
  const std::size_t n = spd_wavelengths.size();
  std::size_t lo_idx = 0;
  std::size_t hi_idx = n - 1;

  if (lambda_min.has_value()) {
    auto it = std::lower_bound(spd_wavelengths.begin(), spd_wavelengths.end(),
                               lambda_min.value());
    lo_idx = static_cast<std::size_t>(it - spd_wavelengths.begin());
    if (lo_idx >= n)
      lo_idx = n - 1;
  }
  if (lambda_max.has_value()) {
    auto it = std::upper_bound(spd_wavelengths.begin(), spd_wavelengths.end(),
                               lambda_max.value());
    if (it != spd_wavelengths.begin()) {
      hi_idx = static_cast<std::size_t>(it - spd_wavelengths.begin()) - 1;
    } else {
      hi_idx = 0;
    }
  }

  if (hi_idx < lo_idx) {
    throw std::invalid_argument(
        "Integration range [" +
        (lambda_min.has_value() ? std::to_string(lambda_min.value()) : "min") +
        ", " +
        (lambda_max.has_value() ? std::to_string(lambda_max.value()) : "max") +
        "] produces no data points (lo_idx=" + std::to_string(lo_idx) +
        " > hi_idx=" + std::to_string(hi_idx) + ")");
  }

  const std::size_t clipped_n = hi_idx - lo_idx + 1;
  if (clipped_n < 2) {
    throw std::invalid_argument(
        "Integration range produces fewer than 2 wavelength points "
        "(need >=2 for trapezoidal integration)");
  }

  return {lo_idx, hi_idx};
}

/// Clip an SPD wavelength grid and values to [lambda_min, lambda_max].
/// Both bounds are optional; if unset, the full range is used.
/// Returns clipped copies. Assumes wavelengths are monotonically increasing.
/// @throws std::invalid_argument if the clip range produces <2 points.
void clip_spd(const std::vector<double> &spd_wavelengths,
              const std::vector<double> &spd_values,
              std::optional<double> lambda_min,
              std::optional<double> lambda_max,
              std::vector<double> &out_wavelengths,
              std::vector<double> &out_values) {
  if (!lambda_min.has_value() && !lambda_max.has_value()) {
    out_wavelengths = spd_wavelengths;
    out_values = spd_values;
    return;
  }

  const ClipIndices idx =
      compute_clip_indices(spd_wavelengths, lambda_min, lambda_max);

  out_wavelengths.assign(spd_wavelengths.begin() + idx.lo,
                         spd_wavelengths.begin() + idx.hi + 1);
  out_values.assign(spd_values.begin() + idx.lo,
                    spd_values.begin() + idx.hi + 1);
}

} // namespace

XyzTriple spd_to_xyz(const std::vector<double> &spd_wavelengths,
                     const std::vector<double> &spd_values,
                     const CmfData &cmf_data, std::optional<double> K,
                     std::optional<double> lambda_min,
                     std::optional<double> lambda_max) {
  // Clip to integration range
  std::vector<double> clip_wl, clip_vals;
  clip_spd(spd_wavelengths, spd_values, lambda_min, lambda_max, clip_wl,
           clip_vals);

  CmfData cmf_resampled = resample_cmf(clip_wl, cmf_data);
  SourceXyz src = compute_source_xyz(clip_wl, clip_vals, cmf_resampled.x_bar,
                                     cmf_resampled.y_bar, cmf_resampled.z_bar);
  // src.X = k*integral St*xbar, src.Y = k*integral St*ybar,
  // src.Z = k*integral St*zbar, where k = 100/integral St*ybar
  // TM-30-20 S3.2 Eq. (1)-(4)
  if (K.has_value()) {
    // Use caller-supplied multiplier: undo auto-k, apply K
    // TM-30-20 S3.2 Eq. (4): k auto-computed; user K replaces it
    double scale = K.value() / src.k; // TM-30-20 S3.2 Eq. (4)
    return XyzTriple{src.X * scale, src.Y * scale, src.Z * scale};
  } else {
    // Auto: Y = 100 (TM-30-20 S3.2 Eq. (2))
    return XyzTriple{src.X, src.Y, src.Z};
  }
}

std::vector<XyzTriple>
spd_to_xyz_batch_prepared(const std::vector<double> &spd_wavelengths,
                          const std::vector<std::vector<double>> &spd_matrix,
                          const CmfData &cmf_resampled,
                          std::optional<double> K) {
  if (cmf_resampled.x_bar.size() != spd_wavelengths.size()) {
    throw std::invalid_argument(
        "cmf_resampled does not match the wavelength grid: expected " +
        std::to_string(spd_wavelengths.size()) + " samples, got " +
        std::to_string(cmf_resampled.x_bar.size()));
  }
  std::vector<XyzTriple> results;
  results.reserve(spd_matrix.size());
  for (const auto &vals : spd_matrix) {
    SourceXyz src =
        compute_source_xyz(spd_wavelengths, vals, cmf_resampled.x_bar,
                           cmf_resampled.y_bar, cmf_resampled.z_bar);
    if (K.has_value()) {
      double scale = K.value() / src.k; // TM-30-20 S3.2 Eq. (4)
      results.push_back(XyzTriple{src.X * scale, src.Y * scale, src.Z * scale});
    } else {
      results.push_back(XyzTriple{src.X, src.Y, src.Z});
    }
  }
  return results;
}

std::vector<XyzTriple>
spd_to_xyz_batch(const std::vector<double> &spd_wavelengths,
                 const std::vector<std::vector<double>> &spd_matrix,
                 const CmfData &cmf_data, std::optional<double> K,
                 std::optional<double> lambda_min,
                 std::optional<double> lambda_max) {
  if (!lambda_min.has_value() && !lambda_max.has_value()) {
    // No clipping: the grid is the integration grid; delegate so the
    // prepared path and this one cannot diverge.
    return spd_to_xyz_batch_prepared(spd_wavelengths, spd_matrix,
                                     resample_cmf(spd_wavelengths, cmf_data),
                                     K);
  }
  const ClipIndices idx =
      compute_clip_indices(spd_wavelengths, lambda_min, lambda_max);

  std::vector<double> clip_wl(spd_wavelengths.begin() + idx.lo,
                              spd_wavelengths.begin() + idx.hi + 1);

  CmfData cmf_resampled = resample_cmf(clip_wl, cmf_data);
  std::vector<XyzTriple> results;
  results.reserve(spd_matrix.size());
  for (const auto &vals : spd_matrix) {
    std::vector<double> clip_vals(vals.begin() + idx.lo,
                                  vals.begin() + idx.hi + 1);
    SourceXyz src =
        compute_source_xyz(clip_wl, clip_vals, cmf_resampled.x_bar,
                           cmf_resampled.y_bar, cmf_resampled.z_bar);
    if (K.has_value()) {
      double scale = K.value() / src.k; // TM-30-20 S3.2 Eq. (4)
      results.push_back(XyzTriple{src.X * scale, src.Y * scale, src.Z * scale});
    } else {
      results.push_back(XyzTriple{src.X, src.Y, src.Z});
    }
  }
  return results;
}

YuvTriple spd_to_Yuv(const std::vector<double> &spd_wavelengths,
                     const std::vector<double> &spd_values,
                     const CmfData &cmf_data, std::optional<double> K,
                     std::optional<double> lambda_min,
                     std::optional<double> lambda_max) {
  XyzTriple xyz = spd_to_xyz(spd_wavelengths, spd_values, cmf_data, K,
                             lambda_min, lambda_max);
  return xyz_to_Yuv(xyz.X, xyz.Y, xyz.Z);
}

std::vector<YuvTriple>
spd_to_Yuv_batch(const std::vector<double> &spd_wavelengths,
                 const std::vector<std::vector<double>> &spd_matrix,
                 const CmfData &cmf_data, std::optional<double> K,
                 std::optional<double> lambda_min,
                 std::optional<double> lambda_max) {
  auto xyzs = spd_to_xyz_batch(spd_wavelengths, spd_matrix, cmf_data, K,
                               lambda_min, lambda_max);
  return xyz_to_Yuv_batch(xyzs);
}

std::vector<YuvTriple>
spd_to_Yuv_batch_prepared(const std::vector<double> &spd_wavelengths,
                          const std::vector<std::vector<double>> &spd_matrix,
                          const CmfData &cmf_resampled,
                          std::optional<double> K) {
  auto xyzs =
      spd_to_xyz_batch_prepared(spd_wavelengths, spd_matrix, cmf_resampled, K);
  return xyz_to_Yuv_batch(xyzs);
}

std::vector<YuvTriple> xyz_to_Yuv_batch(const std::vector<XyzTriple> &xyzs) {
  std::vector<YuvTriple> results;
  results.reserve(xyzs.size());
  for (const auto &xyz : xyzs) {
    results.push_back(xyz_to_Yuv(xyz.X, xyz.Y, xyz.Z));
  }
  return results;
}

XyzTriple cct_to_xyz(double cct, const std::vector<double> &wavelengths,
                     const DaylightBasis &basis, const CmfData &cmf_data,
                     std::optional<double> K) {
  // Resample exactly once - reused for both the reference-SPD
  // Y-normalization blend step and the final XYZ integration below.
  CmfData cmf_resampled = resample_cmf(wavelengths, cmf_data);
  std::vector<double> ref_spd =
      generate_reference_spd(cct, wavelengths, basis, cmf_resampled.y_bar);

  SourceXyz src = compute_source_xyz(wavelengths, ref_spd, cmf_resampled.x_bar,
                                     cmf_resampled.y_bar, cmf_resampled.z_bar);
  if (K.has_value()) {
    double scale = K.value() / src.k; // TM-30-20 S3.2 Eq. (4)
    return XyzTriple{src.X * scale, src.Y * scale, src.Z * scale};
  }
  return XyzTriple{src.X, src.Y, src.Z};
}

std::vector<XyzTriple> cct_to_xyz_batch_prepared(
    const std::vector<double> &ccts, const std::vector<double> &wavelengths,
    const DaylightBasis &basis, const CmfData &cmf_resampled,
    std::optional<double> K) {
  if (cmf_resampled.y_bar.size() != wavelengths.size()) {
    throw std::invalid_argument(
        "cmf_resampled does not match the wavelength grid: expected " +
        std::to_string(wavelengths.size()) + " samples, got " +
        std::to_string(cmf_resampled.y_bar.size()));
  }
  std::vector<XyzTriple> results;
  results.reserve(ccts.size());
  for (double cct : ccts) {
    std::vector<double> ref_spd =
        generate_reference_spd(cct, wavelengths, basis, cmf_resampled.y_bar);
    SourceXyz src =
        compute_source_xyz(wavelengths, ref_spd, cmf_resampled.x_bar,
                           cmf_resampled.y_bar, cmf_resampled.z_bar);
    if (K.has_value()) {
      double scale = K.value() / src.k; // TM-30-20 S3.2 Eq. (4)
      results.push_back(XyzTriple{src.X * scale, src.Y * scale, src.Z * scale});
    } else {
      results.push_back(XyzTriple{src.X, src.Y, src.Z});
    }
  }
  return results;
}

std::vector<XyzTriple> cct_to_xyz_batch(const std::vector<double> &ccts,
                                        const std::vector<double> &wavelengths,
                                        const DaylightBasis &basis,
                                        const CmfData &cmf_data,
                                        std::optional<double> K) {
  return cct_to_xyz_batch_prepared(ccts, wavelengths, basis,
                                   resample_cmf(wavelengths, cmf_data), K);
}

CctDuvResult spd_to_cct(const std::vector<double> &spd_wavelengths,
                        const std::vector<double> &spd_values,
                        const CmfData &cmf_data,
                        const PlanckianLut &planckian_lut) {
  // TM-30-20 S3.5: conform the SPD (drop samples outside 380-780 nm,
  // zero-fill to cover the range, reject steps > 5 nm) before any
  // integration -- the same conformance the full pipeline applies, so
  // this entry point cannot yield a different CCT for the same source.
  const Spd spd(spd_wavelengths, spd_values);
  CmfData cmf_resampled = resample_cmf(spd.wavelengths(), cmf_data);
  SourceXyz src =
      compute_source_xyz(spd.wavelengths(), spd.values(), cmf_resampled.x_bar,
                         cmf_resampled.y_bar, cmf_resampled.z_bar);
  return compute_cct_duv_from_xyz(src.X, src.Y, src.Z, planckian_lut);
}

std::vector<CctDuvResult>
spd_to_cct_batch_prepared(const std::vector<double> &raw_wavelengths,
                          const std::vector<std::vector<double>> &spd_matrix,
                          const CmfData &cmf_resampled,
                          const PlanckianLut &planckian_lut) {
  // Per-row S3.5 conformance. Spd is the single owner of the S3.5
  // recipe -- do not inline a copy of it here for speed (that is the
  // drift the remediation removed); the planned ValidatedGrid refactor
  // removes the per-row grid cost instead
  // (docs/PLAN_validate_once_per_grid.md).
  std::vector<CctDuvResult> results;
  results.reserve(spd_matrix.size());
  for (const auto &vals : spd_matrix) {
    const Spd spd(raw_wavelengths, vals);
    if (spd.wavelengths().size() != cmf_resampled.x_bar.size()) {
      throw std::invalid_argument(
          "cmf_resampled does not match the conformed grid: expected " +
          std::to_string(spd.wavelengths().size()) + " samples, got " +
          std::to_string(cmf_resampled.x_bar.size()));
    }
    SourceXyz src =
        compute_source_xyz(spd.wavelengths(), spd.values(), cmf_resampled.x_bar,
                           cmf_resampled.y_bar, cmf_resampled.z_bar);
    results.push_back(
        compute_cct_duv_from_xyz(src.X, src.Y, src.Z, planckian_lut));
  }
  return results;
}

std::vector<CctDuvResult>
spd_to_cct_batch(const std::vector<double> &spd_wavelengths,
                 const std::vector<std::vector<double>> &spd_matrix,
                 const CmfData &cmf_data, const PlanckianLut &planckian_lut) {
  // TM-30-20 S3.5 conformance, same as spd_to_cct. The shared grid is
  // conformed once via a unit-value probe so the CMF is resampled once;
  // rows are then handled by the prepared variant above.
  const Spd grid_probe(spd_wavelengths,
                       std::vector<double>(spd_wavelengths.size(), 1.0));
  CmfData cmf_resampled = resample_cmf(grid_probe.wavelengths(), cmf_data);
  return spd_to_cct_batch_prepared(spd_wavelengths, spd_matrix, cmf_resampled,
                                   planckian_lut);
}

double spd_to_power(const std::vector<double> &wavelengths,
                    const std::vector<double> &values, const CmfData &cmf_data,
                    bool photometric, std::optional<double> lambda_min,
                    std::optional<double> lambda_max) {
  std::vector<double> clip_wl, clip_vals;
  clip_spd(wavelengths, values, lambda_min, lambda_max, clip_wl, clip_vals);

  if (!photometric) {
    return trapezoidal_integrate(clip_wl, clip_vals);
  }

  CmfData cmf_resampled = resample_cmf(clip_wl, cmf_data);
  std::vector<double> weighted(clip_vals.size());
  for (std::size_t i = 0; i < clip_vals.size(); ++i) {
    weighted[i] = clip_vals[i] * cmf_resampled.y_bar[i];
  }
  return kMaxLuminousEfficacy * trapezoidal_integrate(clip_wl, weighted);
}

std::vector<double>
spd_to_power_batch_prepared(const std::vector<double> &wavelengths,
                            const std::vector<std::vector<double>> &spd_matrix,
                            const CmfData &cmf_resampled, bool photometric) {
  if (photometric && cmf_resampled.y_bar.size() != wavelengths.size()) {
    throw std::invalid_argument(
        "cmf_resampled does not match the wavelength grid: expected " +
        std::to_string(wavelengths.size()) + " samples, got " +
        std::to_string(cmf_resampled.y_bar.size()));
  }
  std::vector<double> results;
  results.reserve(spd_matrix.size());
  std::vector<double> weighted;
  for (const auto &vals : spd_matrix) {
    if (!photometric) {
      results.push_back(trapezoidal_integrate(wavelengths, vals));
      continue;
    }
    weighted.resize(vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i) {
      weighted[i] = vals[i] * cmf_resampled.y_bar[i];
    }
    results.push_back(kMaxLuminousEfficacy *
                      trapezoidal_integrate(wavelengths, weighted));
  }
  return results;
}

std::vector<double>
spd_to_power_batch(const std::vector<double> &wavelengths,
                   const std::vector<std::vector<double>> &spd_matrix,
                   const CmfData &cmf_data, bool photometric,
                   std::optional<double> lambda_min,
                   std::optional<double> lambda_max) {
  if (!lambda_min.has_value() && !lambda_max.has_value()) {
    // No clipping: delegate so the prepared path and this one cannot
    // diverge. The radiometric branch never touches the CMF; resample
    // only when it is used.
    return spd_to_power_batch_prepared(
        wavelengths, spd_matrix,
        photometric ? resample_cmf(wavelengths, cmf_data) : CmfData{},
        photometric);
  }
  const ClipIndices idx =
      compute_clip_indices(wavelengths, lambda_min, lambda_max);
  std::vector<double> clip_wl(wavelengths.begin() + idx.lo,
                              wavelengths.begin() + idx.hi + 1);

  std::vector<double> ybar_resampled;
  if (photometric) {
    CmfData cmf_resampled = resample_cmf(clip_wl, cmf_data);
    ybar_resampled = std::move(cmf_resampled.y_bar);
  }

  std::vector<double> results;
  results.reserve(spd_matrix.size());
  for (const auto &vals : spd_matrix) {
    std::vector<double> clip_vals(vals.begin() + idx.lo,
                                  vals.begin() + idx.hi + 1);
    if (!photometric) {
      results.push_back(trapezoidal_integrate(clip_wl, clip_vals));
      continue;
    }
    std::vector<double> weighted(clip_vals.size());
    for (std::size_t i = 0; i < clip_vals.size(); ++i) {
      weighted[i] = clip_vals[i] * ybar_resampled[i];
    }
    results.push_back(kMaxLuminousEfficacy *
                      trapezoidal_integrate(clip_wl, weighted));
  }
  return results;
}

} // namespace tm30
