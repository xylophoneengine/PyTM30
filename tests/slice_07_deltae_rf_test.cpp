// Slice 7 - dE' color difference and Rf fidelity index tests.
// Validates compute_delta_e and compute_rf against golden fixtures,
// self-consistency, and range constraints.
//
// TM-30-20 S3.8: Color Difference Formula
// TM-30-20 S4.1: Fidelity Index (Rf)

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "matchers.hpp"
#include "tm30/cct.hpp"
#include "tm30/ciecam02.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/metrics.hpp"
#include "tm30/pipeline.hpp"
#include "tm30/reference.hpp"
#include "tm30/resample.hpp"
#include "tm30/spd.hpp"
#include "tm30/xyz.hpp"
#include "tolerances.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
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

std::string fixture_path(const std::string &spd_name,
                         const std::string &stage) {
  return std::string(TM30_DATA_DIR) + "/../tests/fixtures/" + spd_name + "/" +
         stage + ".json";
}

/// Simple JSON parser: extract a single numeric value for a given key.
double load_double_fixture(const std::string &filepath,
                           const std::string &key) {
  std::ifstream in(filepath);
  if (!in) {
    throw std::runtime_error("Cannot open fixture: " + filepath);
  }

  std::string content;
  {
    std::string line;
    while (std::getline(in, line)) {
      content += line + "\n";
    }
  }

  // Find the key
  auto search = "\"" + key + "\"";
  auto pos = content.find(search);
  if (pos == std::string::npos) {
    throw std::runtime_error("No '" + key + "' key in: " + filepath);
  }

  // Find the colon after the key
  pos = content.find(':', pos);
  if (pos == std::string::npos) {
    throw std::runtime_error("Malformed JSON in: " + filepath);
  }

  // Skip whitespace after colon
  ++pos;
  while (pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }

  // Parse number
  char *end = nullptr;
  double val = std::strtod(content.c_str() + pos, &end);

  if (end == content.c_str() + pos) {
    throw std::runtime_error("Failed to parse double from: " + filepath);
  }

  return val;
}

/// Simple JSON parser: extract array of doubles for "values" key.
std::vector<double> load_double_array_fixture(const std::string &filepath) {
  std::ifstream in(filepath);
  if (!in) {
    throw std::runtime_error("Cannot open fixture: " + filepath);
  }

  std::string content;
  {
    std::string line;
    while (std::getline(in, line)) {
      content += line + "\n";
    }
  }

  auto pos = content.find("\"values\"");
  if (pos == std::string::npos) {
    throw std::runtime_error("No 'values' key in: " + filepath);
  }

  pos = content.find('[', pos);
  if (pos == std::string::npos) {
    throw std::runtime_error("Malformed JSON in: " + filepath);
  }

  std::vector<double> result;
  std::size_t i = pos + 1;

  while (i < content.size()) {
    // Skip whitespace
    while (i < content.size() &&
           std::isspace(static_cast<unsigned char>(content[i]))) {
      ++i;
    }

    if (i >= content.size())
      break;
    if (content[i] == ']')
      break;

    // Parse number
    char *end = nullptr;
    double val = std::strtod(content.c_str() + i, &end);

    if (end == content.c_str() + i) {
      // Might be comma or bracket
      ++i;
      continue;
    }

    result.push_back(val);
    i = static_cast<std::size_t>(end - content.c_str());
  }

  return result;
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

/// Load a two-column SPD CSV (wavelength, value).
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

/// Generate a 401-point 1nm wavelength grid 380-780 nm.
std::vector<double> wl_1nm() {
  std::vector<double> wl(401);
  for (int i = 0; i < 401; ++i)
    wl[i] = 380.0 + static_cast<double>(i);
  return wl;
}

// -------------------------------------------------------------------------
// Global fixture data (loaded once)
// -------------------------------------------------------------------------

struct GlobalFixtures {
  CmfData cmf_2deg;
  CmfData cmf_10deg;
  CesData ces;
  DaylightBasis daylight_basis;
  PlanckianLut planckian_lut;

  static GlobalFixtures &instance() {
    static GlobalFixtures g;
    return g;
  }

private:
  GlobalFixtures() {
    cmf_10deg = load_cmf(data_path("cmf_1964_10.csv"));
    cmf_2deg = load_cmf(data_path("cie_1931_2.csv"));
    ces = load_ces(data_path("ces.csv"));
    daylight_basis = load_daylight_basis(data_path("daylight_basis.csv"));
    planckian_lut = load_planckian_lut(data_path("planckian_uv.csv"));
  }
};

// -------------------------------------------------------------------------
// Self-consistency: Planckian at 3000K -> Rf ~= 100, all dE' ~= 0
//
// TM-30-20 S3.8: When test = reference illuminant, all dE' = 0
// TM-30-20 S4.1: Rf = 100 for perfect fidelity
// -------------------------------------------------------------------------

TEST_CASE("dE' & Rf - planckian 3000K self-consistency: dE'~=0, Rf~=100",
          "[delta_e][rf][slice07][self-consistency]") {
  auto &G = GlobalFixtures::instance();

  auto wl = wl_1nm();
  auto spd_vals = generate_planckian(3000.0, wl);

  CesColorimetryResult result =
      compute_ces_colorimetry(wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // Check all 99 dE' values are near zero
  const auto delta_e = compute_delta_e(result.jab_test_ces, result.jab_ref_ces);

  double max_de = 0.0;
  for (std::size_t i = 0; i < 99; ++i) {
    if (delta_e[i] > max_de)
      max_de = delta_e[i];
  }

  INFO("Planckian 3000K self-consistency: max dE' = "
       << max_de << " (tolerance = " << Tol_DeltaE << ")");
  CHECK(max_de <= Tol_DeltaE);

  // Check Rf ~= 100
  INFO("Planckian 3000K: Rf = " << result.Rf << " (tolerance = " << Tol_Rf
                                << ")");
  CHECK_THAT(result.Rf, WithinTolerance(Tol_Rf, 100.0));

  // Check pipeline-computed values match direct computation
  const auto rf_direct = compute_rf(delta_e);
  INFO("Planckian 3000K: pipeline Rf = " << result.Rf
                                         << " direct Rf = " << rf_direct.Rf);
  CHECK_THAT(result.Rf, WithinTolerance(1e-12, rf_direct.Rf));
}

// -------------------------------------------------------------------------
// Self-consistency at multiple CCTs
// TM-30-20 S3.8: Valid across the full CCT range
// -------------------------------------------------------------------------

TEST_CASE("dE' & Rf - planckian 2700K self-consistency: Rf ~= 100",
          "[delta_e][rf][slice07][self-consistency]") {
  auto &G = GlobalFixtures::instance();

  auto wl = wl_1nm();
  auto spd_vals = generate_planckian(2700.0, wl);

  CesColorimetryResult result =
      compute_ces_colorimetry(wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // TM-30-20 S4.1: Self-consistency -> Rf ~= 100
  INFO("Planckian 2700K: Rf = " << result.Rf);
  CHECK_THAT(result.Rf, WithinTolerance(Tol_Rf, 100.0));

  // dE' values should be near zero
  const auto delta_e = compute_delta_e(result.jab_test_ces, result.jab_ref_ces);
  double max_de = 0.0;
  for (std::size_t i = 0; i < 99; ++i) {
    if (delta_e[i] > max_de)
      max_de = delta_e[i];
  }
  INFO("Planckian 2700K: max dE' = " << max_de);
  CHECK(max_de <= Tol_DeltaE);
}

TEST_CASE("dE' & Rf - planckian 3500K self-consistency: Rf ~= 100",
          "[delta_e][rf][slice07][self-consistency]") {
  auto &G = GlobalFixtures::instance();

  auto wl = wl_1nm();
  auto spd_vals = generate_planckian(3500.0, wl);

  CesColorimetryResult result =
      compute_ces_colorimetry(wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // TM-30-20 S4.1: Self-consistency -> Rf ~= 100
  INFO("Planckian 3500K: Rf = " << result.Rf);
  CHECK_THAT(result.Rf, WithinTolerance(Tol_Rf, 100.0));

  // dE' values should be near zero
  const auto delta_e = compute_delta_e(result.jab_test_ces, result.jab_ref_ces);
  double max_de = 0.0;
  for (std::size_t i = 0; i < 99; ++i) {
    if (delta_e[i] > max_de)
      max_de = delta_e[i];
  }
  INFO("Planckian 3500K: max dE' = " << max_de);
  CHECK(max_de <= Tol_DeltaE);
}

TEST_CASE("dE' & Rf - D65 self-consistency: Rf ~= 100",
          "[delta_e][rf][slice07][self-consistency]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // TM-30-20 S4.1: D65 is a daylight -> reference is also D65-like
  INFO("D65: Rf = " << result.Rf);
  CHECK_THAT(result.Rf, WithinTolerance(Tol_Rf, 100.0));

  // dE' values should be very small
  const auto delta_e = compute_delta_e(result.jab_test_ces, result.jab_ref_ces);
  double max_de = 0.0;
  for (std::size_t i = 0; i < 99; ++i) {
    if (delta_e[i] > max_de)
      max_de = delta_e[i];
  }
  INFO("D65: max dE' = " << max_de);
  CHECK(max_de <= Tol_DeltaE);
}

// -------------------------------------------------------------------------
// Golden fixture: D65 Rf
// -------------------------------------------------------------------------

TEST_CASE("dE' & Rf - D65 Rf matches golden fixture",
          "[delta_e][rf][slice07][fixture]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  double golden_rf =
      load_double_fixture(fixture_path("D65_1nm", "12_rf"), "Rf");

  INFO("D65: computed Rf = " << result.Rf << " golden Rf = " << golden_rf);
  CHECK_THAT(result.Rf, WithinTolerance(Tol_Rf, golden_rf));
}

// -------------------------------------------------------------------------
// Golden fixture: D65 dE' values
// -------------------------------------------------------------------------

TEST_CASE("dE' & Rf - D65 dE' values match golden fixture",
          "[delta_e][rf][slice07][fixture]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // Compute dE' directly
  const auto delta_e = compute_delta_e(result.jab_test_ces, result.jab_ref_ces);

  // Load golden values
  auto golden =
      load_double_array_fixture(fixture_path("D65_1nm", "10_delta_e_ces"));
  REQUIRE(golden.size() == 99);

  double max_de = 0.0;
  for (std::size_t i = 0; i < 99; ++i) {
    double d = std::abs(delta_e[i] - golden[i]);
    if (d > max_de)
      max_de = d;
  }

  INFO("D65_1nm dE': max delta = " << max_de << " (tolerance = " << Tol_DeltaE
                                   << ")");
  CHECK(max_de <= Tol_DeltaE);
}

// -------------------------------------------------------------------------
// Golden fixture: F1 Rf
// -------------------------------------------------------------------------

TEST_CASE("dE' & Rf - F1 Rf matches golden fixture",
          "[delta_e][rf][slice07][fixture]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl1_1nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  double golden_rf = load_double_fixture(fixture_path("F1", "12_rf"), "Rf");

  INFO("F1: computed Rf = " << result.Rf << " golden Rf = " << golden_rf);
  CHECK_THAT(result.Rf, WithinTolerance(Tol_Rf, golden_rf));
}

// -------------------------------------------------------------------------
// Golden fixture: HP1 Rf
// -------------------------------------------------------------------------

TEST_CASE("dE' & Rf - HP1 Rf matches golden fixture",
          "[delta_e][rf][slice07][fixture]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("hp1_5nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  double golden_rf = load_double_fixture(fixture_path("HP1", "12_rf"), "Rf");

  INFO("HP1: computed Rf = " << result.Rf << " golden Rf = " << golden_rf);
  CHECK_THAT(result.Rf, WithinTolerance(Tol_Rf, golden_rf));
}

// -------------------------------------------------------------------------
// Rf range: verify 0 <= Rf <= 100 for various SPDs
// TM-30-20 S4.1: Eq. (54) ensures Rf >= 0
// -------------------------------------------------------------------------

TEST_CASE("dE' & Rf - Rf in [0, 100] for various SPDs",
          "[delta_e][rf][slice07][range]") {
  auto &G = GlobalFixtures::instance();

  struct SpdCase {
    std::string name;
    std::string csv_file;
  };

  std::vector<SpdCase> cases = {
      {"D65", "d65_1nm.csv"},  {"F1", "fl1_1nm.csv"},  {"HP1", "hp1_5nm.csv"},
      {"F12", "fl12_1nm.csv"}, {"HP5", "hp5_5nm.csv"},
  };

  for (const auto &c : cases) {
    auto [spd_wl, spd_vals] = load_spd_csv(data_path(c.csv_file));

    CesColorimetryResult result =
        compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg,
                                G.ces, G.daylight_basis, G.planckian_lut);

    INFO(c.name << ": Rf = " << result.Rf);
    CHECK(result.Rf >= 0.0);
    CHECK(result.Rf <= 100.0 + 1e-10); // Allow epsilon for floating-point
  }
}

// -------------------------------------------------------------------------
// Unit tests: compute_delta_e and compute_rf functions directly
// -------------------------------------------------------------------------

TEST_CASE("dE' & Rf - compute_delta_e: zero when J'a'b' identical",
          "[delta_e][rf][slice07][unit]") {
  std::array<Cam02Ucs, 99> jab_test{};
  std::array<Cam02Ucs, 99> jab_ref{};

  // All zeros -> all dE' = 0
  for (std::size_t i = 0; i < 99; ++i) {
    jab_test[i] = Cam02Ucs{50.0, 0.0, 0.0};
    jab_ref[i] = Cam02Ucs{50.0, 0.0, 0.0};
  }

  auto delta_e = compute_delta_e(jab_test, jab_ref);

  for (std::size_t i = 0; i < 99; ++i) {
    CHECK(delta_e[i] <= 1e-15);
  }
}

TEST_CASE("dE' & Rf - compute_rf: Rf = 100 when all dE' = 0",
          "[delta_e][rf][slice07][unit]") {
  std::array<double, 99> delta_e{};
  // All zeros -> Rf = 100

  auto result = compute_rf(delta_e);

  // TM-30-20 S4.1 Eq. (53): Rf' = 100 - 6.73 * 0 = 100
  CHECK(result.Rf_prime == 100.0);

  // TM-30-20 S4.1 Eq. (54): Rf = 10 * ln(exp(10) + 1) ~= 100.00045
  // (not exactly 100 due to the exp/ln rounding; within Tol_Rf)
  CHECK_THAT(result.Rf, WithinTolerance(Tol_Rf, 100.0));

  CHECK(result.delta_e_avg == 0.0);
}

TEST_CASE("dE' & Rf - compute_rf: large dE' gives low Rf",
          "[delta_e][rf][slice07][unit]") {
  std::array<double, 99> delta_e{};
  // Large constant dE' -> low Rf
  delta_e.fill(15.0);

  auto result = compute_rf(delta_e);

  // dE_avg ~= 15 -> Rf' = 100 - 6.73 * 15 = 100 - 100.95 = -0.95
  // Rf = 10 * ln(exp(-0.095) + 1) ~= 10 * ln(1.909) ~= 6.47
  INFO("Rf_prime = " << result.Rf_prime << " Rf = " << result.Rf);
  CHECK(result.Rf_prime < 0.0); // Should be negative
  CHECK(result.Rf > 0.0);       // Eq. (54) ensures non-negative
  CHECK(result.Rf < 100.0);
}

TEST_CASE("dE' & Rf - compute_rf: Eq. (54) ensures Rf >= 0 even for huge dE'",
          "[delta_e][rf][slice07][unit]") {
  std::array<double, 99> delta_e{};
  // Very large dE' -> Rf' very negative, but Rf >= 0
  delta_e.fill(100.0);

  auto result = compute_rf(delta_e);

  INFO("Rf_prime = " << result.Rf_prime << " Rf = " << result.Rf);
  CHECK(result.Rf_prime < -500.0); // Very negative
  CHECK(result.Rf >= 0.0);         // Eq. (54): always non-negative
}

// -------------------------------------------------------------------------
// Pipeline: result contains Rf and delta_e_avg
// -------------------------------------------------------------------------

TEST_CASE("dE' & Rf - pipeline populates Rf and delta_e_avg",
          "[delta_e][rf][slice07][pipeline]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // Rf should be populated and reasonable
  CHECK(result.Rf >= 0.0);
  CHECK(result.Rf <= 100.0 + 1e-10);

  // delta_e_avg should be populated and non-negative
  CHECK(result.delta_e_avg >= 0.0);
}

// -------------------------------------------------------------------------
// dE' array length: must be exactly 99
// -------------------------------------------------------------------------

TEST_CASE("dE' & Rf - compute_delta_e returns exactly 99 values",
          "[delta_e][rf][slice07][unit]") {
  std::array<Cam02Ucs, 99> jab_test{};
  std::array<Cam02Ucs, 99> jab_ref{};

  auto delta_e = compute_delta_e(jab_test, jab_ref);

  CHECK(delta_e.size() == 99);
}

// -------------------------------------------------------------------------
// NaN handling: zero/zero doesn't produce NaN
// -------------------------------------------------------------------------

TEST_CASE("dE' & Rf - zero J'a'b' does not produce NaN",
          "[delta_e][rf][slice07][unit]") {
  std::array<Cam02Ucs, 99> jab_test{};
  std::array<Cam02Ucs, 99> jab_ref{};
  // All zeros

  auto delta_e = compute_delta_e(jab_test, jab_ref);
  for (std::size_t i = 0; i < 99; ++i) {
    CHECK_FALSE(std::isnan(delta_e[i]));
  }

  auto rf = compute_rf(delta_e);
  CHECK_FALSE(std::isnan(rf.Rf));
  CHECK_FALSE(std::isnan(rf.Rf_prime));
  CHECK_FALSE(std::isnan(rf.delta_e_avg));
}

} // anonymous namespace
} // namespace tm30::test
