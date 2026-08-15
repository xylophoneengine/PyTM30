// Slice 1 - SPD container + CIE 15:2018 resampling tests.
//
// TM-30-20 S3.5: Range and Interpolation of Data
// TM-30-20 S3.2: Test Source Integration

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "matchers.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/resample.hpp"
#include "tm30/spd.hpp"
#include "tolerances.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tm30::test {
namespace {

// -------------------------------------------------------------------------
// Test helpers
// -------------------------------------------------------------------------

/// Construct a data directory path from the compile-time TM30_DATA_DIR.
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

  // 99 CES samples: columns 1-99 (skip wavelength column)
  const std::size_t n_ces = 99;
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

/// Create a synthetic test SPD: constant value 1.0 across the given wavelength
/// grid. Used for grid-invariance testing - the SPD itself is trivial, we test
/// resampling.
Spd make_test_spd(const std::vector<double> &wavelengths) {
  std::vector<double> values(wavelengths.size(),
                             1.0); // TM-30-20 S3.2: St(lambda)
  return Spd(wavelengths, values);
}

/// Extract every nth element from a vector (0-indexed, step-sized).
std::vector<double> decimate(const std::vector<double> &v, int step) {
  std::vector<double> result;
  for (std::size_t i = 0; i < v.size(); i += static_cast<std::size_t>(step)) {
    result.push_back(v[i]);
  }
  return result;
}

// -------------------------------------------------------------------------
// SPD construction & validation
// -------------------------------------------------------------------------

TEST_CASE("SPD construction - valid 1nm grid", "[spd][slice01]") {
  std::vector<double> wl(401);
  std::vector<double> vals(401, 1.0);
  for (int i = 0; i < 401; ++i) {
    wl[i] = 380.0 + static_cast<double>(i); // TM-30-20 S3.5: 380-780 nm
  }

  Spd spd(wl, vals);

  REQUIRE(spd.size() == 401);
  REQUIRE(spd.min_wavelength() == 380.0);
  REQUIRE(spd.max_wavelength() == 780.0);
  REQUIRE(spd.step() == 1.0);
}

TEST_CASE("SPD construction - valid 5nm grid", "[spd][slice01]") {
  std::vector<double> wl(81);
  std::vector<double> vals(81, 1.0);
  for (int i = 0; i < 81; ++i) {
    wl[i] = 380.0 + static_cast<double>(i) * 5.0; // TM-30-20 S3.5
  }

  Spd spd(wl, vals);

  REQUIRE(spd.size() == 81);
  REQUIRE(spd.min_wavelength() == 380.0);
  REQUIRE(spd.max_wavelength() == 780.0);
  REQUIRE(spd.step() == 5.0);
}

TEST_CASE("SPD construction - non-uniform grid returns step 0",
          "[spd][slice01]") {
  // TM-30-20 S3.5: Minimum required range is 400-700 nm.
  // Grid covers the range but with non-uniform spacing (5 nm then 2 nm,
  // both within the 5 nm maximum increment).
  std::vector<double> wl;
  for (double w = 380.0; w <= 600.0; w += 5.0)
    wl.push_back(w);
  for (double w = 602.0; w <= 780.0; w += 2.0)
    wl.push_back(w);
  std::vector<double> vals(wl.size(), 1.0);

  Spd spd(wl, vals);

  REQUIRE(spd.size() == wl.size());
  REQUIRE(spd.min_wavelength() == 380.0);
  REQUIRE(spd.max_wavelength() == 780.0);
  REQUIRE(spd.step() == 0.0);
}

// -- Validation: empty SPD -----------------------------------------------

TEST_CASE("SPD validation - empty throws", "[spd][slice01]") {
  REQUIRE_THROWS_AS(Spd({}, {}), InvalidSpd);
}

TEST_CASE("SPD validation - mismatched sizes throw", "[spd][slice01]") {
  REQUIRE_THROWS_AS(Spd({380.0, 780.0}, {1.0}), InvalidSpd);
}

// -- Validation: non-monotonic wavelengths -------------------------------

TEST_CASE("SPD validation - non-monotonic wavelengths throw",
          "[spd][slice01]") {
  // Wavelengths must be strictly increasing - TM-30-20 S3.5
  std::vector<double> wl = {380.0, 390.0, 385.0, 780.0};
  std::vector<double> vals(wl.size(), 1.0);
  REQUIRE_THROWS_AS(Spd(wl, vals), InvalidSpd);
}

TEST_CASE("SPD validation - duplicate wavelengths throw", "[spd][slice01]") {
  std::vector<double> wl = {380.0, 390.0, 390.0, 780.0};
  std::vector<double> vals(wl.size(), 1.0);
  REQUIRE_THROWS_AS(Spd(wl, vals), InvalidSpd);
}

// -- Validation: negative values -----------------------------------------

TEST_CASE("SPD validation - negative values throw", "[spd][slice01]") {
  // TM-30-20 S3.2: spectral power is non-negative
  std::vector<double> wl = {380.0, 500.0, 780.0};
  std::vector<double> vals = {1.0, -0.1, 1.0};
  REQUIRE_THROWS_AS(Spd(wl, vals), InvalidSpd);
}

// -- Validation: insufficient wavelength range ---------------------------

TEST_CASE("SPD validation - range < 400-700 throws", "[spd][slice01]") {
  // TM-30-20 S3.5: the minimum range an SPD must cover is 400-700 nm.
  std::vector<double> wl = {420.0, 600.0}; // starts above 400
  std::vector<double> vals(wl.size(), 1.0);
  REQUIRE_THROWS_AS(Spd(wl, vals), InvalidSpd);
}

TEST_CASE("SPD validation - range starts at 400 but ends before 700 throws",
          "[spd][slice01]") {
  std::vector<double> wl = {400.0, 600.0};
  std::vector<double> vals(wl.size(), 1.0);
  REQUIRE_THROWS_AS(Spd(wl, vals), InvalidSpd);
}

TEST_CASE("SPD validation - range exactly 400-700 passes", "[spd][slice01]") {
  // 400-700 nm at 5 nm (the largest permitted increment, TM-30-20 S3.5)
  std::vector<double> wl;
  for (double w = 400.0; w <= 700.0; w += 5.0)
    wl.push_back(w);
  std::vector<double> vals(wl.size(), 1.0);
  REQUIRE_NOTHROW(Spd(wl, vals));
}

TEST_CASE("SPD validation - range wider than 400-700 passes",
          "[spd][slice01]") {
  std::vector<double> wl;
  for (double w = 380.0; w <= 780.0; w += 5.0)
    wl.push_back(w);
  std::vector<double> vals(wl.size(), 1.0);
  REQUIRE_NOTHROW(Spd(wl, vals));
}

// -------------------------------------------------------------------------
// No-interpolation-of-test-SPD property
// -------------------------------------------------------------------------

TEST_CASE("SPD - test SPD values pass through unchanged", "[spd][slice01]") {
  // TM-30-20 S3.5 forbids interpolating or extrapolating the test SPD.
  // Full-range input at 5 nm: no drop, no zero-fill, stored exactly.
  std::vector<double> wl;
  std::vector<double> vals;
  for (double w = 380.0; w <= 780.0; w += 5.0) {
    wl.push_back(w);
    vals.push_back(0.5 + 0.001 * (w - 380.0));
  }

  Spd spd(wl, vals);

  // Verify the values are stored exactly as provided
  REQUIRE(spd.wavelengths().size() == wl.size());
  REQUIRE(spd.values().size() == vals.size());
  for (std::size_t i = 0; i < wl.size(); ++i) {
    REQUIRE(spd.wavelengths()[i] == wl[i]);
    REQUIRE(spd.values()[i] == vals[i]);
  }
  REQUIRE_FALSE(spd.zero_filled());
}

// -------------------------------------------------------------------------
// TM-30-20 S3.5 range handling: step limit, drop, zero-fill
// -------------------------------------------------------------------------

TEST_CASE("SPD validation - wavelength step above 5 nm throws",
          "[spd][slice01]") {
  // TM-30-20 S3.5: increments above 5 nm are not permitted.
  std::vector<double> wl = {400.0, 500.0, 700.0};
  std::vector<double> vals(wl.size(), 1.0);
  REQUIRE_THROWS_AS(Spd(wl, vals), InvalidSpd);
  // The message names the offending gap.
  try {
    Spd spd(wl, vals);
  } catch (const InvalidSpd &e) {
    const std::string msg = e.what();
    REQUIRE(msg.find("100") != std::string::npos);
    REQUIRE(msg.find("400") != std::string::npos);
    REQUIRE(msg.find("500") != std::string::npos);
  }
}

TEST_CASE("SPD - samples outside 380-780 nm are dropped", "[spd][slice01]") {
  // TM-30-20 S3.5: values outside the calculation range are dropped.
  std::vector<double> wl;
  std::vector<double> vals;
  for (double w = 360.0; w <= 800.0; w += 5.0) {
    wl.push_back(w);
    vals.push_back(1.0);
  }
  Spd spd(wl, vals);
  REQUIRE(spd.min_wavelength() == 380.0);
  REQUIRE(spd.max_wavelength() == 780.0);
  REQUIRE(spd.size() == 81); // 380..780 at 5 nm
  REQUIRE(spd.input_min_wavelength() == 360.0);
  REQUIRE(spd.input_max_wavelength() == 800.0);
  REQUIRE_FALSE(spd.zero_filled());
}

TEST_CASE("SPD - missing edge values are zero-filled to 380-780 nm",
          "[spd][slice01]") {
  // TM-30-20 S3.5: missing values within 380-780 nm are replaced by
  // zeros (input must still cover at least 400-700 nm).
  std::vector<double> wl;
  std::vector<double> vals;
  for (double w = 400.0; w <= 700.0; w += 5.0) {
    wl.push_back(w);
    vals.push_back(1.0);
  }
  Spd spd(wl, vals);
  REQUIRE(spd.zero_filled());
  REQUIRE(spd.min_wavelength() == 380.0);
  REQUIRE(spd.max_wavelength() == 780.0);
  REQUIRE(spd.input_min_wavelength() == 400.0);
  REQUIRE(spd.input_max_wavelength() == 700.0);
  // Filled edges carry zeros; original samples are untouched.
  for (std::size_t i = 0; i < spd.size(); ++i) {
    const double w = spd.wavelengths()[i];
    if (w < 400.0 || w > 700.0) {
      REQUIRE(spd.values()[i] == 0.0);
    } else {
      REQUIRE(spd.values()[i] == 1.0);
    }
  }
  // Native 5 nm step extends outward and lands exactly on 380/780 here.
  REQUIRE(spd.wavelengths().front() == 380.0);
  REQUIRE(spd.wavelengths()[1] == 385.0);
}

// -------------------------------------------------------------------------
// Linear interpolation accuracy
// -------------------------------------------------------------------------

TEST_CASE("Resample - linear interpolation accuracy", "[resample][slice01]") {
  // TM-30-20 S3.5 mandates linear interpolation.
  //
  // Construct CesData with two wavelengths and one CES sample.
  // Interpolate at midpoint - should give exact average.

  CesData source;
  source.wavelengths = {380.0, 390.0};
  source.samples.resize(99);

  // Set CES01 to known values, rest to 0
  for (auto &s : source.samples) {
    s = {0.0, 0.0};
  }
  source.samples[0] = {0.1, 0.2};

  std::vector<double> target = {385.0};

  CesData result = resample_ces(target, source);

  REQUIRE(result.wavelengths.size() == 1);
  REQUIRE(result.samples.size() == 99);
  REQUIRE(result.samples[0].size() == 1);
  // Exact midpoint: (0.1 + 0.2) / 2 = 0.15
  REQUIRE_THAT(result.samples[0][0], Catch::Matchers::WithinAbs(0.15, 1e-12));
}

TEST_CASE("Resample - linear interpolation at grid points returns exact values",
          "[resample][slice01]") {
  CesData source;
  source.wavelengths = {380.0, 390.0, 400.0};
  // Must have exactly 99 CES samples (per TM-30-20 spec)
  source.samples.resize(99);
  for (auto &s : source.samples) {
    s = {0.0, 0.0, 0.0};
  }
  source.samples[0] = {1.0, 2.0, 3.0};

  // Target exactly matches source wavelengths
  std::vector<double> target = {380.0, 390.0, 400.0};

  CesData result = resample_ces(target, source);

  REQUIRE(result.samples[0].size() == 3);
  REQUIRE(result.samples[0][0] == 1.0);
  REQUIRE(result.samples[0][1] == 2.0);
  REQUIRE(result.samples[0][2] == 3.0);
}

// -------------------------------------------------------------------------
// Flat extrapolation
// -------------------------------------------------------------------------

TEST_CASE("Resample - flat extrapolation below source range",
          "[resample][slice01]") {
  // TM-30-20 S1.3 (Errata): flat extrapolation for CES outside 400-700 nm.
  //
  // CES data starts at 400 nm. Request at 380 nm -> return 400 nm value.

  CesData source;
  source.wavelengths = {400.0, 450.0, 700.0};
  source.samples.resize(99);
  for (auto &s : source.samples) {
    s = {0.5, 0.6, 0.7};
  }

  std::vector<double> target = {380.0, 400.0, 780.0};

  CesData result = resample_ces(target, source);

  // At 380 nm (below 400): flat extrapolation -> use 400 nm value (0.5)
  REQUIRE(result.samples[0][0] == 0.5);
  // At 400 nm: exact match
  REQUIRE(result.samples[0][1] == 0.5);
  // At 780 nm (above 700): flat extrapolation -> use 700 nm value (0.7)
  REQUIRE(result.samples[0][2] == 0.7);
}

TEST_CASE("Resample - flat extrapolation above source range",
          "[resample][slice01]") {
  CesData source;
  source.wavelengths = {400.0, 500.0};
  // Must have exactly 99 CES samples
  source.samples.resize(99);
  for (auto &s : source.samples) {
    s = {0.3, 0.8};
  }

  std::vector<double> target = {700.0, 780.0};

  CesData result = resample_ces(target, source);

  // Above 500: both 700 and 780 -> use 500 nm value (0.8)
  REQUIRE(result.samples[0][0] == 0.8);
  REQUIRE(result.samples[0][1] == 0.8);
}

// -------------------------------------------------------------------------
// CMF resampling - basic accuracy
// -------------------------------------------------------------------------

TEST_CASE("Resample - CMF linear interpolation", "[resample][slice01]") {
  CmfData source;
  source.wavelengths = {380.0, 390.0};
  source.x_bar = {0.0, 1.0};
  source.y_bar = {0.0, 0.5};
  source.z_bar = {0.0, 0.2};

  std::vector<double> target = {385.0};

  CmfData result = resample_cmf(target, source);

  REQUIRE(result.x_bar[0] == 0.5);  // midpoint of 0 and 1
  REQUIRE(result.y_bar[0] == 0.25); // midpoint of 0 and 0.5
  REQUIRE(result.z_bar[0] == 0.1);  // midpoint of 0 and 0.2
}

// -------------------------------------------------------------------------
// Grid invariance: 1nm vs 5nm resampling
// -------------------------------------------------------------------------

TEST_CASE("Grid invariance - CES resampled at 1nm and 5nm agree at shared "
          "wavelengths",
          "[resample][slice01][grid-invariance]") {
  // TM-30-20 S3.5: interpolation should preserve accuracy across grid sizes.
  //
  // Load 1nm CES data, resample to 1nm (identity) and to 5nm target,
  // then compare at wavelengths shared by both grids.

  CesData ces_1nm = load_ces(data_path("ces.csv"));

  REQUIRE(ces_1nm.wavelengths.size() == 401); // 380-780 nm at 1 nm steps
  REQUIRE(ces_1nm.samples.size() == 99);

  // 1nm target: full 380-780 grid
  std::vector<double> wl_1nm = ces_1nm.wavelengths;
  CesData resampled_1nm = resample_ces(wl_1nm, ces_1nm);

  // 5nm target: every 5th wavelength from 380 to 780
  std::vector<double> wl_5nm = decimate(wl_1nm, 5);
  CesData resampled_5nm = resample_ces(wl_5nm, ces_1nm);

  // Compare at matching wavelengths (every 5nm).
  // The 1nm resampled values at those wavelengths should closely match
  // the 5nm resampled values (both derived from the same source via linear
  // interp).
  const std::size_t n_shared = wl_5nm.size();
  REQUIRE(resampled_5nm.wavelengths.size() == n_shared);

  // Use a per-element tolerance: linear interpolation from 1nm source
  // is exact at grid points, so the difference should be at most
  // floating-point noise when both paths interpolate the same source.
  const double tol = 1e-12;

  int mismatches = 0;
  for (std::size_t c = 0; c < 99; ++c) {
    for (std::size_t i = 0; i < n_shared; ++i) {
      // 1nm resample at the shared wavelength (index 5*i in 1nm output)
      double v1 = resampled_1nm.samples[c][i * 5];
      double v5 = resampled_5nm.samples[c][i];
      if (std::abs(v1 - v5) > tol) {
        ++mismatches;
      }
    }
  }

  // All 99 CES x 81 shared wavelengths should match exactly.
  REQUIRE(mismatches == 0);
}

TEST_CASE("Grid invariance - CMF resampled at 1nm and 5nm agree at shared "
          "wavelengths",
          "[resample][slice01][grid-invariance]") {
  CmfData cmf_1nm = load_cmf(data_path("cmf_1964_10.csv"));

  REQUIRE(cmf_1nm.wavelengths.size() == 471); // 360-830 nm at 1 nm

  // Use the 380-780 nm subset for grid-invariance test
  std::vector<double> wl_1nm = cmf_1nm.wavelengths;
  CmfData resampled_1nm = resample_cmf(wl_1nm, cmf_1nm);

  // Pick every 5th point from the 380-780 nm range for the 5 nm grid
  std::vector<double> wl_5nm;
  for (size_t i = 0; i < wl_1nm.size(); i += 5) {
    wl_5nm.push_back(wl_1nm[i]);
  }
  CmfData resampled_5nm = resample_cmf(wl_5nm, cmf_1nm);

  const double tol = 1e-12;
  const std::size_t n_shared = wl_5nm.size();

  int mismatches = 0;
  for (std::size_t i = 0; i < n_shared; ++i) {
    if (std::abs(resampled_1nm.x_bar[i * 5] - resampled_5nm.x_bar[i]) > tol)
      ++mismatches;
    if (std::abs(resampled_1nm.y_bar[i * 5] - resampled_5nm.y_bar[i]) > tol)
      ++mismatches;
    if (std::abs(resampled_1nm.z_bar[i * 5] - resampled_5nm.z_bar[i]) > tol)
      ++mismatches;
  }
  REQUIRE(mismatches == 0);
}

// -------------------------------------------------------------------------
// 5nm source data - resampling to 1nm
// -------------------------------------------------------------------------

TEST_CASE("Resample - 5nm CES source to 1nm target via linear interpolation",
          "[resample][slice01]") {
  // Load 5nm CES data, resample to 1nm grid.
  CesData ces_5nm = load_ces(data_path("ces_5nm.csv"));
  REQUIRE(ces_5nm.wavelengths.size() == 81); // 380-780 at 5nm steps

  std::vector<double> wl_1nm(401);
  for (int i = 0; i < 401; ++i) {
    wl_1nm[i] = 380.0 + static_cast<double>(i);
  }

  CesData resampled = resample_ces(wl_1nm, ces_5nm);

  REQUIRE(resampled.wavelengths.size() == 401);
  REQUIRE(resampled.samples.size() == 99);

  // At source grid points (every 5th), values should be exact.
  const double tol = 1e-12;
  int mismatches = 0;
  for (std::size_t c = 0; c < 99; ++c) {
    for (std::size_t i = 0; i < 81; ++i) {
      double v_resampled = resampled.samples[c][i * 5];
      double v_source = ces_5nm.samples[c][i];
      if (std::abs(v_resampled - v_source) > tol) {
        ++mismatches;
      }
    }
  }
  REQUIRE(mismatches == 0);
}

// -------------------------------------------------------------------------
// CES data count validation
// -------------------------------------------------------------------------

TEST_CASE("Resample - rejects wrong CES count", "[resample][slice01]") {
  CesData bad;
  bad.wavelengths = {380.0, 780.0};
  bad.samples.resize(50); // should be 99

  std::vector<double> target = {380.0};
  REQUIRE_THROWS_AS(resample_ces(target, bad), std::invalid_argument);
}

} // namespace
} // namespace tm30::test
