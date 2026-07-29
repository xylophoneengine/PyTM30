// CIE 1964 10° tristimulus value computation.
// TM-30-20 §3.1: Colorimetric Observer
// TM-30-20 §3.2: Test Source - tristimulus values
// TM-30-20 §3.6: Calculation of Tristimulus Values
#include "tm30/xyz.hpp"
#include "tm30/integrate.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

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

  // Build integrand for numerator of k: St(λ) · ȳ₁₀(λ)
  // TM-30-20 §3.2 Eq. (4): k = 100 / ∫ St(λ) · ȳ₁₀(λ) dλ
  std::vector<double> st_times_ybar(n);
  for (std::size_t i = 0; i < n; ++i) {
    st_times_ybar[i] = spd_values[i] * cmf_y_bar[i];
  }

  const double integral_st_ybar =
      trapezoidal_integrate(spd_wavelengths, st_times_ybar);

  // TM-30-20 §3.2 Eq. (4)
  const double k = 100.0 / integral_st_ybar;

  // Build integrands and integrate
  // TM-30-20 §3.2 Eq. (1): X = k · ∫ St(λ) · x̄₁₀(λ) dλ
  // TM-30-20 §3.2 Eq. (2): Y = k · ∫ St(λ) · ȳ₁₀(λ) dλ
  // TM-30-20 §3.2 Eq. (3): Z = k · ∫ St(λ) · z̄₁₀(λ) dλ

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
  result.X = k * integral_st_xbar; // TM-30-20 §3.2 Eq. (1)
  result.Y = k * integral_st_ybar; // TM-30-20 §3.2 Eq. (2)
  result.Z = k * integral_st_zbar; // TM-30-20 §3.2 Eq. (3)
  result.k = k;                    // TM-30-20 §3.2 Eq. (4)

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

  // Pre-allocate integrand buffers (reused for each CES)
  std::vector<double> st_r_times_cmf(n);

  for (std::size_t ces_idx = 0; ces_idx < 99; ++ces_idx) {
    const auto &reflectance = ces_data.samples[ces_idx];

    // TM-30-20 §3.6 Eq. (21): X_i = k · ∫ St(λ) · R_i(λ) · x̄₁₀(λ) dλ
    for (std::size_t i = 0; i < n; ++i) {
      st_r_times_cmf[i] = spd_values[i] * reflectance[i] * cmf_x_bar[i];
    }
    const double X = k * trapezoidal_integrate(spd_wavelengths, st_r_times_cmf);

    // TM-30-20 §3.6 Eq. (22): Y_i = k · ∫ St(λ) · R_i(λ) · ȳ₁₀(λ) dλ
    for (std::size_t i = 0; i < n; ++i) {
      st_r_times_cmf[i] = spd_values[i] * reflectance[i] * cmf_y_bar[i];
    }
    const double Y = k * trapezoidal_integrate(spd_wavelengths, st_r_times_cmf);

    // TM-30-20 §3.6 Eq. (23): Z_i = k · ∫ St(λ) · R_i(λ) · z̄₁₀(λ) dλ
    for (std::size_t i = 0; i < n; ++i) {
      st_r_times_cmf[i] = spd_values[i] * reflectance[i] * cmf_z_bar[i];
    }
    const double Z = k * trapezoidal_integrate(spd_wavelengths, st_r_times_cmf);

    result[ces_idx] = XyzTriple{X, Y, Z};
  }

  return result;
}

// ── Convenience functions ──────────────────────────────────────────────

namespace {

/// Clip an SPD wavelength grid and values to [lambda_min, lambda_max].
/// Both bounds are optional; if unset, the full range is used.
/// Returns clipped copies.  Assumes wavelengths are monotonically increasing.
/// @throws std::invalid_argument if the clip range produces <2 points.
void clip_spd(const std::vector<double> &spd_wavelengths,
              const std::vector<double> &spd_values,
              std::optional<double> lambda_min,
              std::optional<double> lambda_max,
              std::vector<double> &out_wavelengths,
              std::vector<double> &out_values) {
  const size_t n = spd_wavelengths.size();

  if (!lambda_min.has_value() && !lambda_max.has_value()) {
    out_wavelengths = spd_wavelengths;
    out_values = spd_values;
    return;
  }

  size_t lo_idx = 0;
  size_t hi_idx = n - 1;

  if (lambda_min.has_value()) {
    auto it = std::lower_bound(spd_wavelengths.begin(), spd_wavelengths.end(),
                               lambda_min.value());
    lo_idx = static_cast<size_t>(it - spd_wavelengths.begin());
    if (lo_idx >= n)
      lo_idx = n - 1;
  }
  if (lambda_max.has_value()) {
    auto it = std::upper_bound(spd_wavelengths.begin(), spd_wavelengths.end(),
                               lambda_max.value());
    if (it != spd_wavelengths.begin()) {
      hi_idx = static_cast<size_t>(it - spd_wavelengths.begin()) - 1;
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

  size_t clipped_n = hi_idx - lo_idx + 1;
  if (clipped_n < 2) {
    throw std::invalid_argument(
        "Integration range produces fewer than 2 wavelength points "
        "(need ≥2 for trapezoidal integration)");
  }

  out_wavelengths.assign(spd_wavelengths.begin() + lo_idx,
                         spd_wavelengths.begin() + hi_idx + 1);
  out_values.assign(spd_values.begin() + lo_idx,
                    spd_values.begin() + hi_idx + 1);
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
  // src.X = k·∫St·x̄, src.Y = k·∫St·ȳ, src.Z = k·∫St·z̄ where k = 100/∫St·ȳ
  // TM-30-20 §3.2 Eq. (1)-(4)
  if (K.has_value()) {
    // Use caller-supplied multiplier: undo auto-k, apply K
    // TM-30-20 §3.2 Eq. (4): k auto-computed; user K replaces it
    double scale = K.value() / src.k; // TM-30-20 §3.2 Eq. (4)
    return XyzTriple{src.X * scale, src.Y * scale, src.Z * scale};
  } else {
    // Auto: Y = 100 (TM-30-20 §3.2 Eq. (2))
    return XyzTriple{src.X, src.Y, src.Z};
  }
}

std::vector<XyzTriple>
spd_to_xyz_batch(const std::vector<double> &spd_wavelengths,
                 const std::vector<std::vector<double>> &spd_matrix,
                 const CmfData &cmf_data, std::optional<double> K,
                 std::optional<double> lambda_min,
                 std::optional<double> lambda_max) {
  // Clip wavelengths once (all SPDs share the same wavelength grid);
  // build lo/hi indices for per-SPD value slicing.
  const size_t n = spd_wavelengths.size();
  size_t lo_idx = 0;
  size_t hi_idx = n - 1;

  if (lambda_min.has_value()) {
    auto it = std::lower_bound(spd_wavelengths.begin(), spd_wavelengths.end(),
                               lambda_min.value());
    lo_idx = static_cast<size_t>(it - spd_wavelengths.begin());
    if (lo_idx >= n)
      lo_idx = n - 1;
  }
  if (lambda_max.has_value()) {
    auto it = std::upper_bound(spd_wavelengths.begin(), spd_wavelengths.end(),
                               lambda_max.value());
    if (it != spd_wavelengths.begin()) {
      hi_idx = static_cast<size_t>(it - spd_wavelengths.begin()) - 1;
    } else {
      hi_idx = 0;
    }
  }

  if (hi_idx < lo_idx) {
    throw std::invalid_argument("Integration range produces empty data");
  }

  size_t clipped_n = hi_idx - lo_idx + 1;
  if (clipped_n < 2) {
    throw std::invalid_argument(
        "Integration range produces fewer than 2 wavelength points");
  }

  std::vector<double> clip_wl(spd_wavelengths.begin() + lo_idx,
                              spd_wavelengths.begin() + hi_idx + 1);

  CmfData cmf_resampled = resample_cmf(clip_wl, cmf_data);
  std::vector<XyzTriple> results;
  results.reserve(spd_matrix.size());
  for (const auto &vals : spd_matrix) {
    std::vector<double> clip_vals(vals.begin() + lo_idx,
                                  vals.begin() + hi_idx + 1);
    SourceXyz src =
        compute_source_xyz(clip_wl, clip_vals, cmf_resampled.x_bar,
                           cmf_resampled.y_bar, cmf_resampled.z_bar);
    if (K.has_value()) {
      double scale = K.value() / src.k; // TM-30-20 §3.2 Eq. (4)
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
  std::vector<YuvTriple> results;
  results.reserve(xyzs.size());
  for (const auto &xyz : xyzs) {
    results.push_back(xyz_to_Yuv(xyz.X, xyz.Y, xyz.Z));
  }
  return results;
}

} // namespace tm30
