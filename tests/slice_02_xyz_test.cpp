// Slice 2 - CMF integration -> XYZ tristimulus values (CIE 1964 10-deg
// observer).
//
// TM-30-20 §3.1: Colorimetric Observer
// TM-30-20 §3.2: Test Source Tristimulus Values
// TM-30-20 §3.6: Calculation of Tristimulus Values
// TM-30-20 §3.5: Range and Interpolation of Data

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "matchers.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/integrate.hpp"
#include "tm30/resample.hpp"
#include "tm30/spd.hpp"
#include "tm30/xyz.hpp"
#include "tolerances.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tm30::test {
namespace {

// -------------------------------------------------------------------------
// Test helpers
// -------------------------------------------------------------------------

std::string data_path(const std::string &filename) {
  return std::string(TM30_DATA_DIR) + "/" + filename;
}

/// Load CES reflectance data from a CSV file.
CesData load_ces(const std::string &path) {
  CsvTable table = load_csv(path);
  CesData data;

  for (const auto &row : table.rows) {
    data.wavelengths.push_back(row[0]);
  }

  const std::size_t n_ces = 99; // TM-30-20: 99 CES samples
  data.samples.resize(n_ces);
  for (std::size_t c = 0; c < n_ces; ++c) {
    data.samples[c].reserve(table.rows.size());
    for (const auto &row : table.rows) {
      data.samples[c].push_back(row[1 + c]);
    }
  }

  return data;
}

/// Load CIE 1964 10-deg CMF data from a CSV file.
CmfData load_cmf(const std::string &path) {
  CsvTable table = load_csv(path);
  CmfData data;

  for (const auto &row : table.rows) {
    data.wavelengths.push_back(row[0]);
    data.x_bar.push_back(row[1]);
    data.y_bar.push_back(row[2]);
    data.z_bar.push_back(row[3]);
  }

  return data;
}

/// Load CMF and resample to match the given SPD wavelength grid.
/// The CMF file may cover a wider range (e.g. 360-830 nm); this helper
/// resamples it to the SPD's wavelengths so compute_source_xyz() receives
/// matching arrays.
CmfData load_cmf_for_spd(const std::string &cmf_path,
                         const std::vector<double> &spd_wl) {
  CmfData cmf_source = load_cmf(cmf_path);
  return resample_cmf(spd_wl, cmf_source);
}

/// Load a simple two-column SPD CSV (wavelength, value).
/// Used for D65 and Illuminant A data files.
std::pair<std::vector<double>, std::vector<double>>
load_spd_csv(const std::string &path) {
  CsvTable table = load_csv(path);
  std::vector<double> wl;
  std::vector<double> vals;
  for (const auto &row : table.rows) {
    wl.push_back(row[0]);
    vals.push_back(row[1]);
  }
  return {wl, vals};
}

/// Build a 401-point wavelength grid from 380 to 780 nm at 1 nm step.
/// TM-30-20 §3.5: 380-780 nm range.
std::vector<double> full_1nm_grid() {
  std::vector<double> wl(401);
  for (int i = 0; i < 401; ++i) {
    wl[i] = 380.0 + static_cast<double>(i);
  }
  return wl;
}

/// Linear interpolation of spectral data to a new wavelength grid.
/// Uses flat extrapolation outside the source range.
std::vector<double> resample_spd(const std::vector<double> &target_wl,
                                 const std::vector<double> &source_wl,
                                 const std::vector<double> &source_vals) {
  std::vector<double> result;
  result.reserve(target_wl.size());

  std::size_t j = 0;

  for (double tw : target_wl) {
    // Flat extrapolation - low side
    if (tw <= source_wl.front()) {
      result.push_back(source_vals.front());
      continue;
    }

    // Flat extrapolation - high side
    if (tw >= source_wl.back()) {
      result.push_back(source_vals.back());
      continue;
    }

    // Advance j until source_wl[j] <= tw < source_wl[j+1]
    while (j + 1 < source_wl.size() && source_wl[j + 1] <= tw) {
      ++j;
    }

    // Linear interpolation between indices j and j+1
    const double w0 = source_wl[j];
    const double w1 = source_wl[j + 1];
    const double v0 = source_vals[j];
    const double v1 = source_vals[j + 1];

    const double t = (tw - w0) / (w1 - w0);
    result.push_back(v0 + t * (v1 - v0));
  }

  return result;
}

// -------------------------------------------------------------------------
// Integration accuracy
// -------------------------------------------------------------------------

TEST_CASE("Integrate - constant function over uniform grid",
          "[integrate][slice02]") {
  // f(lambda) = 1 over 380-780 nm at 1 nm step: 401 points.
  // Expected: 400.0 (401 points, 400 segments of width 1, avg = 1)
  auto wl = full_1nm_grid();
  std::vector<double> unity(wl.size(), 1.0);

  double result = trapezoidal_integrate(wl, unity);
  REQUIRE_THAT(result, Catch::Matchers::WithinAbs(400.0, 1e-12));
}

TEST_CASE("Integrate - linear function over uniform grid",
          "[integrate][slice02]") {
  // f(lambda) = lambda over [0, 10] with step 1.
  // Expected: integral 0^1^0 lambda dlambda = 50.0
  std::vector<double> wl = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                            6.0, 7.0, 8.0, 9.0, 10.0};
  double result = trapezoidal_integrate(wl, wl); // f(lambda) = lambda
  REQUIRE_THAT(result, Catch::Matchers::WithinAbs(50.0, 1e-12));
}

TEST_CASE("Integrate - non-uniform grid", "[integrate][slice02]") {
  // f(lambda) = 1 over [0, 10] with non-uniform steps: [0, 2, 5, 7, 10]
  // Expected: 10.0 (total interval length)
  std::vector<double> wl = {0.0, 2.0, 5.0, 7.0, 10.0};
  std::vector<double> unity(wl.size(), 1.0);
  double result = trapezoidal_integrate(wl, unity);
  REQUIRE_THAT(result, Catch::Matchers::WithinAbs(10.0, 1e-12));
}

TEST_CASE("Integrate - non-uniform grid with varying function",
          "[integrate][slice02]") {
  // f(lambda) = lambda^2 over [0, 1, 3, 6]
  // Expected: integral 0^6 lambda^2 dlambda = 72.0
  // Trapezoidal: 1/2*(0^2+1^2)*1 + 1/2*(1^2+3^2)*2 + 1/2*(3^2+6^2)*3
  // = 0.5 + 10 + 67.5 = 78.0 (approximation)
  std::vector<double> wl = {0.0, 1.0, 3.0, 6.0};
  std::vector<double> vals;
  for (double w : wl)
    vals.push_back(w * w);
  double result = trapezoidal_integrate(wl, vals);
  double expected = 0.5 + 10.0 + 67.5;
  REQUIRE_THAT(result, Catch::Matchers::WithinAbs(expected, 1e-12));
}

TEST_CASE("Integrate - single segment", "[integrate][slice02]") {
  // f(lambda) = 3 over [5, 10]: expected = 3 * 5 = 15
  std::vector<double> wl = {5.0, 10.0};
  std::vector<double> vals = {3.0, 3.0};
  REQUIRE_THAT(trapezoidal_integrate(wl, vals),
               Catch::Matchers::WithinAbs(15.0, 1e-12));
}

TEST_CASE("Integrate - throws on size mismatch", "[integrate][slice02]") {
  std::vector<double> wl = {1.0, 2.0};
  std::vector<double> vals = {1.0};
  REQUIRE_THROWS_AS(trapezoidal_integrate(wl, vals), std::invalid_argument);
}

TEST_CASE("Integrate - throws on fewer than 2 points", "[integrate][slice02]") {
  std::vector<double> wl = {1.0};
  std::vector<double> vals = {1.0};
  REQUIRE_THROWS_AS(trapezoidal_integrate(wl, vals), std::invalid_argument);
}

// -------------------------------------------------------------------------
// Trapezoidal weights - per-point weight equivalence
// -------------------------------------------------------------------------

TEST_CASE("Trapezoidal weights - uniform 1 nm grid with constant function",
          "[integrate][slice02]") {
  // f(lambda) = 1 over 380-780 nm at 1 nm step: 401 points.
  // sum w[i]*f[i] should equal trapezoidal_integrate result.
  auto wl = full_1nm_grid();
  std::vector<double> unity(wl.size(), 1.0);

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * unity[i];
  }

  double integral = trapezoidal_integrate(wl, unity);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - uniform 1 nm grid with linear function",
          "[integrate][slice02]") {
  // f(lambda) = lambda over 380-780 nm at 1 nm step: 401 points.
  auto wl = full_1nm_grid();

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * wl[i];
  }

  double integral = trapezoidal_integrate(wl, wl);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - uniform 1 nm grid with D65 spectrum",
          "[integrate][slice02]") {
  // f(lambda) = D65 SPD values over 380-780 nm.
  auto wl = full_1nm_grid();
  auto [d65_wl, d65_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  REQUIRE(d65_wl.size() == wl.size());

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * d65_vals[i];
  }

  double integral = trapezoidal_integrate(wl, d65_vals);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - uniform 5 nm grid with constant function",
          "[integrate][slice02]") {
  // Construct 5 nm grid: 380, 385, 390, ..., 780 (81 points)
  std::vector<double> wl;
  for (double w = 380.0; w <= 780.0; w += 5.0) {
    wl.push_back(w);
  }
  REQUIRE(wl.size() == 81);

  std::vector<double> unity(wl.size(), 1.0);
  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * unity[i];
  }

  double integral = trapezoidal_integrate(wl, unity);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - uniform 5 nm grid with linear function",
          "[integrate][slice02]") {
  std::vector<double> wl;
  for (double w = 380.0; w <= 780.0; w += 5.0) {
    wl.push_back(w);
  }

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * wl[i];
  }

  double integral = trapezoidal_integrate(wl, wl);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - uniform 5 nm grid with D65 spectrum",
          "[integrate][slice02]") {
  // Resample D65 to 5 nm grid
  std::vector<double> wl;
  for (double w = 380.0; w <= 780.0; w += 5.0) {
    wl.push_back(w);
  }

  auto [d65_wl, d65_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  // Resample d65 to match 5 nm grid
  std::vector<double> d65_5nm = resample_spd(wl, d65_wl, d65_vals);

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * d65_5nm[i];
  }

  double integral = trapezoidal_integrate(wl, d65_5nm);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - non-uniform grid with constant function",
          "[integrate][slice02]") {
  // Non-uniform: 1 nm from 380-500, then 2 nm from 500-780
  std::vector<double> wl;
  for (double w = 380.0; w <= 500.0; w += 1.0) {
    wl.push_back(w);
  }
  for (double w = 502.0; w <= 780.0; w += 2.0) {
    wl.push_back(w);
  }

  std::vector<double> unity(wl.size(), 1.0);
  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * unity[i];
  }

  double integral = trapezoidal_integrate(wl, unity);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - non-uniform grid with linear function",
          "[integrate][slice02]") {
  std::vector<double> wl;
  for (double w = 380.0; w <= 500.0; w += 1.0) {
    wl.push_back(w);
  }
  for (double w = 502.0; w <= 780.0; w += 2.0) {
    wl.push_back(w);
  }

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * wl[i];
  }

  double integral = trapezoidal_integrate(wl, wl);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - non-uniform grid with real spectrum",
          "[integrate][slice02]") {
  std::vector<double> wl;
  for (double w = 380.0; w <= 500.0; w += 1.0) {
    wl.push_back(w);
  }
  for (double w = 502.0; w <= 780.0; w += 2.0) {
    wl.push_back(w);
  }

  auto [d65_wl, d65_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  std::vector<double> d65_resampled = resample_spd(wl, d65_wl, d65_vals);

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * d65_resampled[i];
  }

  double integral = trapezoidal_integrate(wl, d65_resampled);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - minimal 2-point grid with constant function",
          "[integrate][slice02]") {
  std::vector<double> wl = {400.0, 500.0};
  std::vector<double> unity(wl.size(), 1.0);

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == 2);

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * unity[i];
  }

  double integral = trapezoidal_integrate(wl, unity);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - minimal 2-point grid with linear function",
          "[integrate][slice02]") {
  std::vector<double> wl = {400.0, 500.0};

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == 2);

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * wl[i];
  }

  double integral = trapezoidal_integrate(wl, wl);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - minimal 2-point grid with real spectrum",
          "[integrate][slice02]") {
  std::vector<double> wl = {400.0, 500.0};

  auto [d65_wl, d65_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  std::vector<double> d65_2pt = resample_spd(wl, d65_wl, d65_vals);

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == 2);

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * d65_2pt[i];
  }

  double integral = trapezoidal_integrate(wl, d65_2pt);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - narrow range grid with constant function",
          "[integrate][slice02]") {
  // Narrow range: 450-600 nm at 1 nm
  std::vector<double> wl;
  for (double w = 450.0; w <= 600.0; w += 1.0) {
    wl.push_back(w);
  }
  REQUIRE(wl.size() == 151);

  std::vector<double> unity(wl.size(), 1.0);
  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * unity[i];
  }

  double integral = trapezoidal_integrate(wl, unity);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - narrow range grid with linear function",
          "[integrate][slice02]") {
  std::vector<double> wl;
  for (double w = 450.0; w <= 600.0; w += 1.0) {
    wl.push_back(w);
  }

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * wl[i];
  }

  double integral = trapezoidal_integrate(wl, wl);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - narrow range grid with real spectrum",
          "[integrate][slice02]") {
  std::vector<double> wl;
  for (double w = 450.0; w <= 600.0; w += 1.0) {
    wl.push_back(w);
  }

  auto [d65_wl, d65_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  std::vector<double> d65_narrow = resample_spd(wl, d65_wl, d65_vals);

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * d65_narrow[i];
  }

  double integral = trapezoidal_integrate(wl, d65_narrow);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - fine grid with constant function",
          "[integrate][slice02]") {
  // Very fine: 0.5 nm steps over 500-520 nm (41 points)
  std::vector<double> wl;
  for (double w = 500.0; w <= 520.0; w += 0.5) {
    wl.push_back(w);
  }
  REQUIRE(wl.size() == 41);

  std::vector<double> unity(wl.size(), 1.0);
  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * unity[i];
  }

  double integral = trapezoidal_integrate(wl, unity);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - fine grid with linear function",
          "[integrate][slice02]") {
  std::vector<double> wl;
  for (double w = 500.0; w <= 520.0; w += 0.5) {
    wl.push_back(w);
  }

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * wl[i];
  }

  double integral = trapezoidal_integrate(wl, wl);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - fine grid with real spectrum",
          "[integrate][slice02]") {
  std::vector<double> wl;
  for (double w = 500.0; w <= 520.0; w += 0.5) {
    wl.push_back(w);
  }

  auto [d65_wl, d65_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  std::vector<double> d65_fine = resample_spd(wl, d65_wl, d65_vals);

  auto weights = trapezoidal_weights(wl);
  REQUIRE(weights.size() == wl.size());

  double weighted_sum = 0.0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weighted_sum += weights[i] * d65_fine[i];
  }

  double integral = trapezoidal_integrate(wl, d65_fine);
  REQUIRE_THAT(weighted_sum, Catch::Matchers::WithinAbs(integral, 1e-9));
}

TEST_CASE("Trapezoidal weights - throws on fewer than 2 points",
          "[integrate][slice02]") {
  std::vector<double> wl = {1.0};
  REQUIRE_THROWS_AS(trapezoidal_weights(wl), std::invalid_argument);
}

// -------------------------------------------------------------------------
// Energy conservation - CMF integrals
// -------------------------------------------------------------------------

TEST_CASE("CMF - xbar10 integral over 380-780 nm", "[cmf][slice02]") {
  CmfData cmf_full = load_cmf(data_path("cmf_1964_10.csv"));
  REQUIRE(cmf_full.wavelengths.size() == 471); // 360-830 nm at 1 nm

  // Clip to 380-780 nm for the TM-30 range integral.
  auto wl_380_780 = full_1nm_grid();
  CmfData cmf = resample_cmf(wl_380_780, cmf_full);

  double int_xbar = trapezoidal_integrate(cmf.wavelengths, cmf.x_bar);
  // Golden value computed from luxpy CMF data via trapezoidal integration:
  // integral xbar10 dlambda = 116.6475035076
  REQUIRE_THAT(int_xbar, WithinTolerance(Tol_Xyz, 116.6475035076));
}

TEST_CASE("CMF - ybar10 integral over 380-780 nm", "[cmf][slice02]") {
  CmfData cmf_full = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl_380_780 = full_1nm_grid();
  CmfData cmf = resample_cmf(wl_380_780, cmf_full);

  double int_ybar = trapezoidal_integrate(cmf.wavelengths, cmf.y_bar);
  // integral ybar10 dlambda = 116.6616192707
  REQUIRE_THAT(int_ybar, WithinTolerance(Tol_Xyz, 116.6616192707));
}

TEST_CASE("CMF - zbar10 integral over 380-780 nm", "[cmf][slice02]") {
  CmfData cmf_full = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl_380_780 = full_1nm_grid();
  CmfData cmf = resample_cmf(wl_380_780, cmf_full);

  double int_zbar = trapezoidal_integrate(cmf.wavelengths, cmf.z_bar);
  // integral zbar10 dlambda = 116.6717442180
  REQUIRE_THAT(int_zbar, WithinTolerance(Tol_Xyz, 116.6717442180));
}

// -------------------------------------------------------------------------
// D65 source XYZ
// -------------------------------------------------------------------------

TEST_CASE("XYZ - D65 source tristimulus (1 nm)", "[xyz][slice02]") {
  // Load D65 SPD at 1 nm, 380-780. CMF now covers 360-830 nm - resample.
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  REQUIRE(spd_wl.size() == 401);

  CmfData cmf = load_cmf_for_spd(data_path("cmf_1964_10.csv"), spd_wl);

  // TM-30-20 §3.2: compute source XYZ with CIE 1964 10-deg observer
  SourceXyz src =
      compute_source_xyz(spd_wl, spd_vals, cmf.x_bar, cmf.y_bar, cmf.z_bar);

  // TM-30-20 §3.2 Eq. (2): Y must equal 100.0 after normalisation
  REQUIRE_THAT(src.Y, WithinTolerance(Tol_Xyz, 100.0));

  // Golden values computed from the current (colour-science-sourced)
  // d65_1nm.csv data over 380-780 nm:
  // X = 94.8106411232, Z = 107.3036929917
  // Generated by trapezoidal integration of the same CSV data
  // that is loaded in this test.
  REQUIRE_THAT(src.X, WithinTolerance(Tol_Xyz, 94.8106411232));
  REQUIRE_THAT(src.Z, WithinTolerance(Tol_Xyz, 107.3036929917));

  // k must be positive and finite
  // TM-30-20 §3.2 Eq. (4): k = 100 / integral St(lambda) * ybar10(lambda)
  // dlambda
  REQUIRE(src.k > 0.0);
  REQUIRE(std::isfinite(src.k));
}

// -------------------------------------------------------------------------
// Illuminant A source XYZ
// -------------------------------------------------------------------------

TEST_CASE("XYZ - Illuminant A source tristimulus (1 nm)", "[xyz][slice02]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("illuminant_a_1nm.csv"));
  REQUIRE(spd_wl.size() == 401);

  CmfData cmf = load_cmf_for_spd(data_path("cmf_1964_10.csv"), spd_wl);

  SourceXyz src =
      compute_source_xyz(spd_wl, spd_vals, cmf.x_bar, cmf.y_bar, cmf.z_bar);

  // Y = 100.0 - TM-30-20 §3.2 Eq. (2)
  REQUIRE_THAT(src.Y, WithinTolerance(Tol_Xyz, 100.0));

  // Golden values: X = 111.1432899325, Z = 35.1999196709
  REQUIRE_THAT(src.X, WithinTolerance(Tol_Xyz, 111.1432899325));
  REQUIRE_THAT(src.Z, WithinTolerance(Tol_Xyz, 35.1999196709));

  REQUIRE(src.k > 0.0);
  REQUIRE(std::isfinite(src.k));
}

// -------------------------------------------------------------------------
// Normalisation verification
// -------------------------------------------------------------------------

TEST_CASE("XYZ - normalisation produces Y=100 for source", "[xyz][slice02]") {
  // TM-30-20 §3.2: Y should be exactly 100 for any valid source SPD.
  // Test with D65 and Illuminant A.

  auto [d65_wl, d65_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  auto [a_wl, a_vals] = load_spd_csv(data_path("illuminant_a_1nm.csv"));

  CmfData cmf_d65 = load_cmf_for_spd(data_path("cmf_1964_10.csv"), d65_wl);
  CmfData cmf_a = load_cmf_for_spd(data_path("cmf_1964_10.csv"), a_wl);

  SourceXyz d65_xyz = compute_source_xyz(d65_wl, d65_vals, cmf_d65.x_bar,
                                         cmf_d65.y_bar, cmf_d65.z_bar);
  SourceXyz a_xyz =
      compute_source_xyz(a_wl, a_vals, cmf_a.x_bar, cmf_a.y_bar, cmf_a.z_bar);

  REQUIRE_THAT(d65_xyz.Y, WithinTolerance(Tol_Xyz, 100.0));
  REQUIRE_THAT(a_xyz.Y, WithinTolerance(Tol_Xyz, 100.0));
}

// -------------------------------------------------------------------------
// CES tristimulus values
// -------------------------------------------------------------------------

/// Characterization-test oracle: verbatim copy of compute_ces_xyz's
/// pre-rewrite body (nested loop building a fresh integrand per CES per
/// channel and calling trapezoidal_integrate), preserved here so the
/// weights-based rewrite in src/tm30/xyz.cpp can be checked against it
/// directly. Do not "clean up" to match the new implementation - the whole
/// point is that this stays a faithful copy of the old algorithm.
std::array<XyzTriple, 99> compute_ces_xyz_reference(
    const std::vector<double> &spd_wavelengths,
    const std::vector<double> &spd_values, const CesData &ces_data,
    const std::vector<double> &cmf_x_bar, const std::vector<double> &cmf_y_bar,
    const std::vector<double> &cmf_z_bar, double k) {
  const std::size_t n = spd_wavelengths.size();

  if (ces_data.samples.size() != 99) {
    throw std::invalid_argument(
        "compute_ces_xyz requires exactly 99 CES samples, got " +
        std::to_string(ces_data.samples.size()));
  }

  std::array<XyzTriple, 99> result;
  std::vector<double> st_r_times_cmf(n);

  for (std::size_t ces_idx = 0; ces_idx < 99; ++ces_idx) {
    const auto &reflectance = ces_data.samples[ces_idx];

    for (std::size_t i = 0; i < n; ++i) {
      st_r_times_cmf[i] = spd_values[i] * reflectance[i] * cmf_x_bar[i];
    }
    const double X = k * trapezoidal_integrate(spd_wavelengths, st_r_times_cmf);

    for (std::size_t i = 0; i < n; ++i) {
      st_r_times_cmf[i] = spd_values[i] * reflectance[i] * cmf_y_bar[i];
    }
    const double Y = k * trapezoidal_integrate(spd_wavelengths, st_r_times_cmf);

    for (std::size_t i = 0; i < n; ++i) {
      st_r_times_cmf[i] = spd_values[i] * reflectance[i] * cmf_z_bar[i];
    }
    const double Z = k * trapezoidal_integrate(spd_wavelengths, st_r_times_cmf);

    result[ces_idx] = XyzTriple{X, Y, Z};
  }

  return result;
}

TEST_CASE("XYZ - CES output shape is 99", "[xyz][slice02]") {
  // TM-30-20 §3.6: compute tristimulus values for all 99 CES under D65.

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf_for_spd(data_path("cmf_1964_10.csv"), spd_wl);

  // Compute source normalisation first
  SourceXyz src =
      compute_source_xyz(spd_wl, spd_vals, cmf.x_bar, cmf.y_bar, cmf.z_bar);

  // Load and resample CES data
  CesData ces_1nm = load_ces(data_path("ces.csv"));
  REQUIRE(ces_1nm.samples.size() == 99);

  auto ces_xyz = compute_ces_xyz(spd_wl, spd_vals, ces_1nm, cmf.x_bar,
                                 cmf.y_bar, cmf.z_bar, src.k);

  // Must return exactly 99 results - TM-30-20 uses 99 CES samples
  REQUIRE(ces_xyz.size() == 99);
}

TEST_CASE("XYZ - CES tristimulus uses source normalisation constant",
          "[xyz][slice02]") {
  // Verify that compute_ces_xyz uses the provided k.
  // If we pass k=0, all XYZ should be 0.
  // If we pass valid k, values should be non-zero and reasonable.

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf_for_spd(data_path("cmf_1964_10.csv"), spd_wl);
  CesData ces_1nm = load_ces(data_path("ces.csv"));

  // With k=0, all should be zero
  auto ces_zero = compute_ces_xyz(spd_wl, spd_vals, ces_1nm, cmf.x_bar,
                                  cmf.y_bar, cmf.z_bar, 0.0);
  for (const auto &xyz : ces_zero) {
    REQUIRE(std::abs(xyz.X) < 1e-15);
    REQUIRE(std::abs(xyz.Y) < 1e-15);
    REQUIRE(std::abs(xyz.Z) < 1e-15);
  }

  // With valid k, values should be non-zero
  SourceXyz src =
      compute_source_xyz(spd_wl, spd_vals, cmf.x_bar, cmf.y_bar, cmf.z_bar);
  auto ces_xyz = compute_ces_xyz(spd_wl, spd_vals, ces_1nm, cmf.x_bar,
                                 cmf.y_bar, cmf.z_bar, src.k);

  // All CES Y values should be between 0 and 100 (reflectance <= 1)
  for (const auto &xyz : ces_xyz) {
    REQUIRE(xyz.Y >= 0.0);
    REQUIRE(xyz.Y <= 100.0);
  }

  // At least some CES should have distinct values (not all identical)
  bool has_variation = false;
  for (std::size_t i = 1; i < 99; ++i) {
    if (std::abs(ces_xyz[i].Y - ces_xyz[0].Y) > 1e-6) {
      has_variation = true;
      break;
    }
  }
  REQUIRE(has_variation);
}

TEST_CASE("XYZ - CES compute rejects wrong CES count", "[xyz][slice02]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf_for_spd(data_path("cmf_1964_10.csv"), spd_wl);

  CesData bad_ces;
  bad_ces.wavelengths = spd_wl;
  bad_ces.samples.resize(50); // should be 99

  REQUIRE_THROWS_AS(compute_ces_xyz(spd_wl, spd_vals, bad_ces, cmf.x_bar,
                                    cmf.y_bar, cmf.z_bar, 1.0),
                    std::invalid_argument);
}

TEST_CASE("XYZ - CES weights-based rewrite agrees with reference algorithm",
          "[xyz][slice02]") {
  // Old-vs-new regression test: compute_ces_xyz was rewritten to compute
  // trapezoidal_weights() once per call and reduce the per-CES work to
  // fused dot products, instead of building a fresh integrand array and
  // calling trapezoidal_integrate per channel per CES. This must remain
  // bit-level-equivalent (within floating-point reassociation noise) to
  // the pre-rewrite algorithm, captured verbatim above as
  // compute_ces_xyz_reference.

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf_for_spd(data_path("cmf_1964_10.csv"), spd_wl);
  CesData ces_1nm = load_ces(data_path("ces.csv"));

  SourceXyz src =
      compute_source_xyz(spd_wl, spd_vals, cmf.x_bar, cmf.y_bar, cmf.z_bar);

  auto ces_old = compute_ces_xyz_reference(spd_wl, spd_vals, ces_1nm, cmf.x_bar,
                                           cmf.y_bar, cmf.z_bar, src.k);
  auto ces_new = compute_ces_xyz(spd_wl, spd_vals, ces_1nm, cmf.x_bar,
                                 cmf.y_bar, cmf.z_bar, src.k);

  // Reassociating the summation order (per-point weighted dot product vs.
  // per-segment trapezoidal accumulation) introduces floating-point noise
  // at the ~1e-13 level when prototyped; 1e-9 leaves generous headroom
  // while still catching a real bug such as an off-by-one in the weights.
  for (std::size_t i = 0; i < 99; ++i) {
    REQUIRE_THAT(ces_new[i].X, Catch::Matchers::WithinAbs(ces_old[i].X, 1e-9));
    REQUIRE_THAT(ces_new[i].Y, Catch::Matchers::WithinAbs(ces_old[i].Y, 1e-9));
    REQUIRE_THAT(ces_new[i].Z, Catch::Matchers::WithinAbs(ces_old[i].Z, 1e-9));
  }
}

// -------------------------------------------------------------------------
// Scale invariance - doubling SPD should double XYZ (pre-normalisation)
// -------------------------------------------------------------------------

TEST_CASE("XYZ - scale invariance: doubled SPD gives same XYZ",
          "[xyz][slice02]") {
  // TM-30-20 normalisation means scaling the SPD by any factor
  // produces identical XYZ tristimulus values.
  // k changes inversely, but k * integral St*xbar stays constant.

  auto wl = full_1nm_grid();
  CmfData cmf = load_cmf_for_spd(data_path("cmf_1964_10.csv"), wl);

  // Build a synthetic SPD
  std::vector<double> spd_vals(wl.size(), 1.0);

  SourceXyz src1 =
      compute_source_xyz(wl, spd_vals, cmf.x_bar, cmf.y_bar, cmf.z_bar);

  // Double the SPD
  std::vector<double> spd2_vals(wl.size(), 2.0);
  SourceXyz src2 =
      compute_source_xyz(wl, spd2_vals, cmf.x_bar, cmf.y_bar, cmf.z_bar);

  // XYZ should be identical (normalisation absorbs the scale factor)
  // TM-30-20 §3.2: normalisation constant includes the integral
  REQUIRE_THAT(src1.X, WithinTolerance(Tol_Xyz, src2.X));
  REQUIRE_THAT(src1.Y, WithinTolerance(Tol_Xyz, src2.Y));
  REQUIRE_THAT(src1.Z, WithinTolerance(Tol_Xyz, src2.Z));

  // k should halve
  REQUIRE_THAT(src2.k, Catch::Matchers::WithinAbs(src1.k * 0.5, Tol_Xyz));
}

// -------------------------------------------------------------------------
// Resampled input (5 nm grid)
// -------------------------------------------------------------------------

TEST_CASE("XYZ - source XYZ from 5nm resampled data agrees with 1nm",
          "[xyz][slice02]") {
  // TM-30-20 §3.5: results from 5 nm data should agree with 1 nm within
  // tolerance.

  auto [spd_wl_1nm, spd_vals_1nm] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf_1nm = load_cmf_for_spd(data_path("cmf_1964_10.csv"), spd_wl_1nm);

  // Compute at 1 nm
  SourceXyz src_1nm = compute_source_xyz(
      spd_wl_1nm, spd_vals_1nm, cmf_1nm.x_bar, cmf_1nm.y_bar, cmf_1nm.z_bar);

  // Build 5 nm grid and resample
  std::vector<double> wl_5nm;
  std::vector<double> spd_5nm;
  for (int i = 0; i < 401; i += 5) {
    wl_5nm.push_back(spd_wl_1nm[i]);
    spd_5nm.push_back(spd_vals_1nm[i]);
  }

  // Resample CMF from the full file to 5 nm
  CmfData cmf_source = load_cmf(data_path("cmf_1964_10.csv")); // 360-830 nm
  CmfData cmf_5nm = resample_cmf(wl_5nm, cmf_source);

  SourceXyz src_5nm = compute_source_xyz(wl_5nm, spd_5nm, cmf_5nm.x_bar,
                                         cmf_5nm.y_bar, cmf_5nm.z_bar);

  // 1 nm and 5 nm results should agree within a reasonable tolerance.
  // TM-30-20 §3.5: increments not greater than 5 nm.
  // Grid invariance has higher expected error than oracle comparison;
  // subsampling from 1nm to 5nm loses spectral detail.
  constexpr double kGridInvarianceRelTol = 1e-3; // 0.1% relative
  REQUIRE_THAT(src_5nm.Y, WithinTolerance(Tol_Xyz, 100.0));
  REQUIRE_THAT(src_5nm.X, WithinRelTolerance(kGridInvarianceRelTol, src_1nm.X));
  REQUIRE_THAT(src_5nm.Z, WithinRelTolerance(kGridInvarianceRelTol, src_1nm.Z));
}

} // namespace
} // namespace tm30::test
