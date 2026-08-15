// Slice 8 - Hue-angle binning tests.
// Validates bin_by_hue against golden fixtures, self-consistency,
// complete coverage (all 99 CES assigned), and boundary handling.
//
// TM-30-20 S4.3 + Figure 3: Hue-Angle Bins

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tm30/cct.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/hue_bins.hpp"
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
#include <numbers>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tm30::test {
namespace {

// -------------------------------------------------------------------------
// Test helpers (same pattern as other slice tests)
// -------------------------------------------------------------------------

std::string data_path(const std::string &filename) {
  return std::string(TM30_DATA_DIR) + "/" + filename;
}

std::string fixture_path(const std::string &spd_name,
                         const std::string &stage) {
  return std::string(TM30_DATA_DIR) + "/../tests/fixtures/" + spd_name + "/" +
         stage + ".json";
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

/// Simple JSON parser: extract the "hue_bin_index" array from a fixture file.
std::vector<int> load_hue_bin_index_fixture(const std::string &filepath) {
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

  auto pos = content.find("\"hue_bin_index\"");
  if (pos == std::string::npos) {
    throw std::runtime_error("No 'hue_bin_index' key in: " + filepath);
  }

  pos = content.find('[', pos);
  if (pos == std::string::npos) {
    throw std::runtime_error("Malformed JSON in: " + filepath);
  }

  std::vector<int> result;
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
      ++i;
      continue;
    }

    result.push_back(static_cast<int>(val));
    i = static_cast<std::size_t>(end - content.c_str());
  }

  return result;
}

/// Convert HueBins (array of vectors) to a per-CES bin index array (0-15).
std::array<int, 99> flatten_bins(const HueBins &bins) {
  std::array<int, 99> result{};
  for (int bin = 0; bin < 16; ++bin) {
    for (int idx : bins[bin]) {
      result[idx] = bin;
    }
  }
  return result;
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
// Fixture-based tests: bin assignments match golden data for various SPDs
// -------------------------------------------------------------------------

void verify_hue_bins_for_spd(const std::string &spd_name,
                             const std::string &csv_file,
                             const std::string &fixture_subdir) {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path(csv_file));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // Load golden hue_bin_index (0-indexed bins 0-15)
  auto golden_hue_bin_index =
      load_hue_bin_index_fixture(fixture_path(fixture_subdir, "11_hue_bins"));
  REQUIRE(golden_hue_bin_index.size() == 99);

  // Convert our HueBins to per-CES array
  auto computed = flatten_bins(result.hue_bins);

  int mismatches = 0;
  for (int i = 0; i < 99; ++i) {
    if (computed[i] != golden_hue_bin_index[i]) {
      INFO("CES[" << i << "]: computed bin = " << computed[i]
                  << ", golden bin = " << golden_hue_bin_index[i]);
      ++mismatches;
      if (mismatches >= 5) {
        INFO("... (too many mismatches, stopping)");
        break;
      }
    }
  }

  CHECK(mismatches == 0);
}

TEST_CASE("Hue binning - D65 bin assignments match fixture",
          "[hue_bins][slice08][fixture]") {
  verify_hue_bins_for_spd("D65", "d65_1nm.csv", "D65_1nm");
}

TEST_CASE("Hue binning - F1 bin assignments match fixture",
          "[hue_bins][slice08][fixture]") {
  verify_hue_bins_for_spd("F1", "fl1_1nm.csv", "F1");
}

TEST_CASE("Hue binning - HP1 bin assignments match fixture",
          "[hue_bins][slice08][fixture]") {
  verify_hue_bins_for_spd("HP1", "hp1_5nm.csv", "HP1");
}

TEST_CASE("Hue binning - F12 bin assignments match fixture",
          "[hue_bins][slice08][fixture]") {
  verify_hue_bins_for_spd("F12", "fl12_1nm.csv", "F12");
}

TEST_CASE("Hue binning - HP5 bin assignments match fixture",
          "[hue_bins][slice08][fixture]") {
  verify_hue_bins_for_spd("HP5", "hp5_5nm.csv", "HP5");
}

TEST_CASE("Hue binning - planckian 3000K bin assignments match fixture",
          "[hue_bins][slice08][fixture]") {
  auto &G = GlobalFixtures::instance();

  auto wl = wl_1nm();
  auto spd_vals = generate_planckian(3000.0, wl);

  CesColorimetryResult result =
      compute_ces_colorimetry(wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  auto golden_hue_bin_index = load_hue_bin_index_fixture(
      fixture_path("planckian_3000K", "11_hue_bins"));
  REQUIRE(golden_hue_bin_index.size() == 99);

  auto computed = flatten_bins(result.hue_bins);

  int mismatches = 0;
  for (int i = 0; i < 99; ++i) {
    if (computed[i] != golden_hue_bin_index[i]) {
      INFO("CES[" << i << "]: computed bin = " << computed[i]
                  << ", golden bin = " << golden_hue_bin_index[i]);
      ++mismatches;
      if (mismatches >= 5)
        break;
    }
  }
  CHECK(mismatches == 0);
}

// -------------------------------------------------------------------------
// Coverage: all 99 CES assigned exactly once
// -------------------------------------------------------------------------

TEST_CASE("Hue binning - all 99 CES assigned exactly once (D65)",
          "[hue_bins][slice08][coverage]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // Count total assigned
  int total = 0;
  for (int bin = 0; bin < 16; ++bin) {
    total += static_cast<int>(result.hue_bins[bin].size());
  }
  CHECK(total == 99);
}

TEST_CASE("Hue binning - no duplicate assignments (D65)",
          "[hue_bins][slice08][coverage]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // Check no CES appears in more than one bin
  std::set<int> seen;
  for (int bin = 0; bin < 16; ++bin) {
    for (int idx : result.hue_bins[bin]) {
      REQUIRE(
          seen.insert(idx).second); // Must be true (first time seeing this idx)
    }
  }
  CHECK(seen.size() == 99);
}

// -------------------------------------------------------------------------
// No empty bins for well-populated SPDs
// -------------------------------------------------------------------------

TEST_CASE("Hue binning - no empty bins for D65",
          "[hue_bins][slice08][coverage]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  for (int bin = 0; bin < 16; ++bin) {
    INFO("Bin " << bin << " is empty");
    CHECK_FALSE(result.hue_bins[bin].empty());
  }
}

TEST_CASE("Hue binning - no empty bins for planckian 3000K",
          "[hue_bins][slice08][coverage]") {
  auto &G = GlobalFixtures::instance();

  auto wl = wl_1nm();
  auto spd_vals = generate_planckian(3000.0, wl);

  CesColorimetryResult result =
      compute_ces_colorimetry(wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  for (int bin = 0; bin < 16; ++bin) {
    INFO("Bin " << bin << " is empty");
    CHECK_FALSE(result.hue_bins[bin].empty());
  }
}

TEST_CASE("Hue binning - no empty bins for HP1",
          "[hue_bins][slice08][coverage]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("hp1_5nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  for (int bin = 0; bin < 16; ++bin) {
    INFO("Bin " << bin << " is empty");
    CHECK_FALSE(result.hue_bins[bin].empty());
  }
}

// -------------------------------------------------------------------------
// Boundary handling tests
// -------------------------------------------------------------------------

TEST_CASE("Hue binning - boundary: hr = 0 maps to bin 0",
          "[hue_bins][slice08][boundary]") {
  // Create a single CES at hr = 0 (positive a' axis).
  // Set all others to a coordinate in bin 8 (180-deg) to avoid polluting bin 0.
  std::array<Cam02Ucs, 99> jab_ref{};
  for (int i = 0; i < 99; ++i) {
    jab_ref[i] = Cam02Ucs{50.0, -1.0, 0.0}; // hr = pi -> bin 8
  }
  jab_ref[0] = Cam02Ucs{50.0, 1.0, 0.0}; // hr = atan2(0, 1) = 0 -> bin 0

  HueBins bins = bin_by_hue(jab_ref);

  // CES 0 should be in bin 0
  CHECK(bins[0].size() == 1);
  CHECK(bins[0][0] == 0);
}

TEST_CASE("Hue binning - boundary: hr exactly at 22.5-deg maps to bin 1 "
          "(higher index)",
          "[hue_bins][slice08][boundary]") {
  // Bin 0 spans [0-deg, 22.5-deg), bin 1 spans [22.5-deg, 45.0-deg)
  // CES exactly at 22.5-deg should go to bin 1 (boundary tie-break: [start,
  // end))
  double angle_rad = 22.5 * std::numbers::pi / 180.0; // 0.392699... rad

  // Set all others to a coordinate in bin 8 (180-deg) to avoid polluting.
  std::array<Cam02Ucs, 99> jab_ref{};
  for (int i = 0; i < 99; ++i) {
    jab_ref[i] = Cam02Ucs{50.0, -1.0, 0.0}; // hr = pi -> bin 8
  }
  jab_ref[0] = Cam02Ucs{50.0, std::cos(angle_rad), std::sin(angle_rad)};

  HueBins bins = bin_by_hue(jab_ref);

  // Should be in bin 1 (0-indexed), not bin 0
  CHECK(bins[1].size() == 1);
  CHECK(bins[1][0] == 0);
  CHECK(bins[0].empty());
}

TEST_CASE("Hue binning - boundary: hr just below 22.5-deg maps to bin 0",
          "[hue_bins][slice08][boundary]") {
  // hr = 22.49-deg should be in bin 0
  double angle_rad = 22.49 * std::numbers::pi / 180.0;

  // Set all others to a coordinate in bin 8 (180-deg) to avoid polluting.
  std::array<Cam02Ucs, 99> jab_ref{};
  for (int i = 0; i < 99; ++i) {
    jab_ref[i] = Cam02Ucs{50.0, -1.0, 0.0}; // hr = pi -> bin 8
  }
  jab_ref[0] = Cam02Ucs{50.0, std::cos(angle_rad), std::sin(angle_rad)};

  HueBins bins = bin_by_hue(jab_ref);

  CHECK(bins[0].size() == 1);
  CHECK(bins[0][0] == 0);
}

TEST_CASE("Hue binning - boundary: hr = 360-deg (= 2pi) maps to bin 15",
          "[hue_bins][slice08][boundary]") {
  // hr just above 337.5-deg goes to bin 15
  double angle_rad = 350.0 * std::numbers::pi / 180.0;

  // Set all others to a coordinate in bin 0.
  std::array<Cam02Ucs, 99> jab_ref{};
  for (int i = 0; i < 99; ++i) {
    jab_ref[i] = Cam02Ucs{50.0, 1.0, 0.0}; // hr = 0 -> bin 0
  }
  jab_ref[0] = Cam02Ucs{50.0, std::cos(angle_rad), std::sin(angle_rad)};

  HueBins bins = bin_by_hue(jab_ref);

  // 350-deg / 22.5-deg = 15.55 -> bin 15
  CHECK(bins[15].size() == 1);
  CHECK(bins[15][0] == 0);
}

TEST_CASE("Hue binning - boundary: negative hr normalized correctly",
          "[hue_bins][slice08][boundary]") {
  // a' = 1, b' = -0.1 -> hr ~= -0.0997 rad -> normalized to ~6.183 rad (~=
  // 354.3-deg) Should be in bin 15 (337.5-deg to 360-deg)
  double angle_rad = -0.099668652; // ~= -5.71-deg -> ~354.29-deg

  // Set all others to a coordinate in bin 0.
  std::array<Cam02Ucs, 99> jab_ref{};
  for (int i = 0; i < 99; ++i) {
    jab_ref[i] = Cam02Ucs{50.0, 1.0, 0.0}; // hr = 0 -> bin 0
  }
  jab_ref[0] = Cam02Ucs{50.0, std::cos(angle_rad), std::sin(angle_rad)};

  HueBins bins = bin_by_hue(jab_ref);

  // After normalization: angle ~= 2pi - 0.0997 ~= 6.183 rad
  // 6.183 / 0.3927 ~= 15.75 -> bin 15
  CHECK(bins[15].size() == 1);
  CHECK(bins[15][0] == 0);
}

// -------------------------------------------------------------------------
// bin_by_hue returns exactly 16 bins
// -------------------------------------------------------------------------

TEST_CASE("Hue binning - bin_by_hue returns 16 bins",
          "[hue_bins][slice08][unit]") {
  std::array<Cam02Ucs, 99> jab_ref{};
  for (int i = 0; i < 99; ++i) {
    jab_ref[i] = Cam02Ucs{50.0, 1.0, 0.0}; // all at 0-deg, all in bin 0
  }

  HueBins bins = bin_by_hue(jab_ref);
  CHECK(bins.size() == 16);
  // All 99 should be in bin 0
  CHECK(bins[0].size() == 99);
}

// -------------------------------------------------------------------------
// Self-consistency: reference = test -> bins should be same
// (Binning is based on reference only, but we verify consistency)
// -------------------------------------------------------------------------

TEST_CASE("Hue binning - planckian 3000K self-consistency bins non-empty",
          "[hue_bins][slice08][self-consistency]") {
  auto &G = GlobalFixtures::instance();

  auto wl = wl_1nm();
  auto spd_vals = generate_planckian(3000.0, wl);

  CesColorimetryResult result =
      compute_ces_colorimetry(wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // For self-consistency: all bins should be non-empty
  int total = 0;
  for (int bin = 0; bin < 16; ++bin) {
    total += static_cast<int>(result.hue_bins[bin].size());
    CHECK_FALSE(result.hue_bins[bin].empty());
  }
  CHECK(total == 99);
}

// -------------------------------------------------------------------------
// Per-bin sample count within expected range (2 to 11 per spec)
// -------------------------------------------------------------------------

TEST_CASE("Hue binning - per-bin sample count in [2, 11] for D65",
          "[hue_bins][slice08][range]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  for (int bin = 0; bin < 16; ++bin) {
    int count = static_cast<int>(result.hue_bins[bin].size());
    INFO("Bin " << bin << " has " << count << " CES");
    CHECK(count >= 2);  // TM-30-20 S4.3: range 2-11
    CHECK(count <= 11); // TM-30-20 S4.3: range 2-11
  }
}

// -------------------------------------------------------------------------
// CES indices are within valid range
// -------------------------------------------------------------------------

TEST_CASE("Hue binning - all CES indices in valid range [0, 98]",
          "[hue_bins][slice08][range]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  for (int bin = 0; bin < 16; ++bin) {
    for (int idx : result.hue_bins[bin]) {
      CHECK(idx >= 0);
      CHECK(idx < 99);
    }
  }
}

} // anonymous namespace
} // namespace tm30::test
