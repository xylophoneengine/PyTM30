// Slice 5 - CES colorimetry pipeline integration tests.
// Validates the full end-to-end pipeline against golden fixture data.
//
// TM-30-20 §3.4: Color Evaluation Samples
// TM-30-20 §3.6: Calculation of Tristimulus Values

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "matchers.hpp"
#include "tm30/cct.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/pipeline.hpp"
#include "tm30/reference.hpp"
#include "tm30/resample.hpp"
#include "tm30/spd.hpp"
#include "tm30/xyz.hpp"
#include "tolerances.hpp"

#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tm30::test {
namespace {

// ─────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────

std::string data_path(const std::string &filename) {
  return std::string(TM30_DATA_DIR) + "/" + filename;
}

std::string fixture_path(const std::string &spd_name,
                         const std::string &stage) {
  return std::string(TM30_DATA_DIR) + "/../tests/fixtures/" + spd_name + "/" +
         stage + ".json";
}

/// Minimal JSON array-of-arrays parser.
/// Reads a .json file, finds the "values" key, and extracts
/// [[x,y,z], [x,y,z], ...] into a vector of XyzTriple.
std::vector<XyzTriple> load_xyz_fixture(const std::string &filepath) {
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

  // Find "values" key
  auto pos = content.find("\"values\"");
  if (pos == std::string::npos) {
    throw std::runtime_error("No 'values' key in: " + filepath);
  }

  // Advance past "values": [
  pos = content.find('[', pos);
  if (pos == std::string::npos) {
    throw std::runtime_error("Malformed JSON in: " + filepath);
  }

  // State-machine parser: walk through characters extracting numbers
  std::vector<XyzTriple> result;
  std::size_t i = pos + 1; // skip outer '['

  enum State { kSeekTripleStart, kParseX, kParseY, kParseZ, kDone };
  State state = kSeekTripleStart;
  int triple_idx = 0;
  double vals[3] = {};

  while (i < content.size() && state != kDone) {
    char c = content[i];

    // Skip whitespace
    if (std::isspace(static_cast<unsigned char>(c))) {
      ++i;
      continue;
    }

    switch (state) {
    case kSeekTripleStart:
      if (c == ']' || c == '}') {
        state = kDone;
        break;
      }
      if (c == '[') {
        ++i;
        triple_idx = 0;
        state = kParseX;
      } else if (c == ',') {
        ++i;
        // stay in kSeekTripleStart, next non-ws should be '['
      } else {
        ++i; // skip unexpected
      }
      break;

    case kParseX:
    case kParseY:
    case kParseZ: {
      // Parse a number
      std::size_t start = i;
      // Scan until delimiter: comma, ']', whitespace, or '}'
      while (i < content.size() && content[i] != ',' && content[i] != ']' &&
             content[i] != '}' &&
             !std::isspace(static_cast<unsigned char>(content[i]))) {
        ++i;
      }
      std::string num_str = content.substr(start, i - start);

      // std::strtod for locale-independent parsing
      char *end = nullptr;
      double val = std::strtod(num_str.c_str(), &end);
      if (end == num_str.c_str() || *end != '\0') {
        // If strtod didn't consume the whole token, try
        // checking if trailing chars are just formatting
        if (end != num_str.c_str()) {
          val = std::strtod(num_str.c_str(), nullptr);
        } else {
          throw std::runtime_error("Failed to parse float from: '" + num_str +
                                   "' in " + filepath);
        }
      }

      vals[triple_idx] = val;
      ++triple_idx;

      if (triple_idx >= 3) {
        result.push_back(XyzTriple{vals[0], vals[1], vals[2]});
        triple_idx = 0;
        state = kSeekTripleStart;
        // Skip trailing ']', ',', whitespace
        while (i < content.size() &&
               (content[i] == ']' || content[i] == ',' ||
                std::isspace(static_cast<unsigned char>(content[i])))) {
          ++i;
        }
      } else {
        // Advance to next state
        state = static_cast<State>(static_cast<int>(state) + 1);
        // Skip comma and whitespace after number
        while (i < content.size() &&
               (content[i] == ',' ||
                std::isspace(static_cast<unsigned char>(content[i])))) {
          ++i;
        }
      }
      break;
    }

    case kDone:
      break;
    }
  }

  return result;
}

/// Load CIE 1964 10° CMF data from a CSV file.
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

/// Generate a 401-point 1nm wavelength grid 380–780 nm.
std::vector<double> wl_1nm() {
  std::vector<double> wl(401);
  for (int i = 0; i < 401; ++i)
    wl[i] = 380.0 + static_cast<double>(i);
  return wl;
}

/// Check that two arrays of 99 XyzTriple are within tolerance.
/// Uses CHECK (not REQUIRE) to report all failures.
void check_xyz_match(const std::array<XyzTriple, 99> &computed,
                     const std::vector<XyzTriple> &expected, double tolerance,
                     const std::string &label) {
  REQUIRE(computed.size() == 99);
  REQUIRE(expected.size() == 99);

  double max_delta = 0.0;

  for (std::size_t i = 0; i < 99; ++i) {
    double dx = std::abs(computed[i].X - expected[i].X);
    double dy = std::abs(computed[i].Y - expected[i].Y);
    double dz = std::abs(computed[i].Z - expected[i].Z);
    double local_max = std::max(dx, std::max(dy, dz));
    if (local_max > max_delta)
      max_delta = local_max;
  }

  INFO(label << ": max delta = " << max_delta << " (tolerance = " << tolerance
             << ")");
  REQUIRE(max_delta <= tolerance);
}

// Fixture comparison tolerance - documented in PARITY.md Slice 5 entry.
// CES XYZ values have larger accumulated numerical deltas than source
// white-point XYZ due to multiplication chain (SPD × reflectance × CMF)
// and reference-SPD propagation. Tol_Xyz (5e-5) is for source XYZ.
constexpr double Tol_FixtureXyz = 1.0e-3;

// ─────────────────────────────────────────────────────────────────────────
// Global fixture data (loaded once)
// ─────────────────────────────────────────────────────────────────────────

// Lazy-init globals to avoid static init ordering issues
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

// ─────────────────────────────────────────────────────────────────────────
// Golden fixture tests
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("CES colorimetry - D65_1nm matches golden fixtures",
          "[pipeline][slice05]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  REQUIRE(spd_wl.size() == 401);

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // Verify CCT/Duv against golden
  // D65_1nm: cct ≈ 6501.90, duv ≈ 0.00321
  REQUIRE_THAT(result.cct, WithinTolerance(Tol_Cct, 6501.89789213571));
  REQUIRE_THAT(result.duv, WithinTolerance(Tol_Duv, 0.0032144638144609994));

  // Load golden CES XYZ
  auto golden_test =
      load_xyz_fixture(fixture_path("D65_1nm", "06_xyz_test_ces"));
  auto golden_ref = load_xyz_fixture(fixture_path("D65_1nm", "07_xyz_ref_ces"));

  check_xyz_match(result.xyz_test_ces, golden_test, Tol_FixtureXyz,
                  "D65_1nm test");
  check_xyz_match(result.xyz_ref_ces, golden_ref, Tol_FixtureXyz,
                  "D65_1nm ref");
}

TEST_CASE("CES colorimetry - F1 matches golden fixtures",
          "[pipeline][slice05]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl1_1nm.csv"));
  REQUIRE(spd_wl.size() == 401);

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // F1: cct ≈ 6425.40, duv ≈ 0.00719 (recomputed against the current,
  // colour-science-sourced fl1_1nm.csv; independent oracle, see
  // tools/oracle_recompute_12.py)
  REQUIRE_THAT(result.cct, WithinTolerance(Tol_Cct, 6425.401524207265));
  REQUIRE_THAT(result.duv, WithinTolerance(Tol_Duv, 0.007191972077681352));

  auto golden_test = load_xyz_fixture(fixture_path("F1", "06_xyz_test_ces"));
  auto golden_ref = load_xyz_fixture(fixture_path("F1", "07_xyz_ref_ces"));

  check_xyz_match(result.xyz_test_ces, golden_test, Tol_FixtureXyz, "F1 test");
  check_xyz_match(result.xyz_ref_ces, golden_ref, Tol_FixtureXyz, "F1 ref");
}

TEST_CASE("CES colorimetry - HP1 matches golden fixtures",
          "[pipeline][slice05]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("hp1_1nm.csv"));
  // HP1 is at 5nm grid, 81 points
  REQUIRE(spd_wl.size() == 81);

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // HP1: cct ≈ 1959.24, duv ≈ 0.00078
  // TM-30-20 §3.3 Eq. (14): Tt ≤ 4000 K → pure Planckian
  REQUIRE_THAT(result.cct, WithinTolerance(Tol_Cct, 1959.2357303373917));
  REQUIRE_THAT(result.duv, WithinTolerance(Tol_Duv, 0.0007823644283656693));

  auto golden_test = load_xyz_fixture(fixture_path("HP1", "06_xyz_test_ces"));
  auto golden_ref = load_xyz_fixture(fixture_path("HP1", "07_xyz_ref_ces"));

  check_xyz_match(result.xyz_test_ces, golden_test, Tol_FixtureXyz, "HP1 test");
  check_xyz_match(result.xyz_ref_ces, golden_ref, Tol_FixtureXyz, "HP1 ref");
}

TEST_CASE("CES colorimetry - planckian 3000K matches golden fixtures",
          "[pipeline][slice05]") {
  auto &G = GlobalFixtures::instance();

  // Generate planckian at 3000K on 1nm grid
  auto wl = wl_1nm();
  auto spd_vals = generate_planckian(3000.0, wl);

  CesColorimetryResult result =
      compute_ces_colorimetry(wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // planckian_3000K: cct ≈ 3000.03, duv ≈ 3.62e-6
  REQUIRE_THAT(result.cct, WithinTolerance(Tol_Cct, 3000.0304058052598));
  REQUIRE_THAT(result.duv, WithinTolerance(Tol_Duv, 3.622599766511914e-06));

  auto golden_test =
      load_xyz_fixture(fixture_path("planckian_3000K", "06_xyz_test_ces"));
  auto golden_ref =
      load_xyz_fixture(fixture_path("planckian_3000K", "07_xyz_ref_ces"));

  check_xyz_match(result.xyz_test_ces, golden_test, Tol_FixtureXyz,
                  "planckian_3000K test");
  check_xyz_match(result.xyz_ref_ces, golden_ref, Tol_FixtureXyz,
                  "planckian_3000K ref");
}

// ─────────────────────────────────────────────────────────────────────────
// Self-consistency: Planckian at 3000K → test ≈ reference
//
// NOTE: The CCT algorithm does not perfectly recover the input temperature
// for a Planckian source (e.g., Planckian at 3000.00 K → CCT ≈ 3000.03 K).
// This causes the reference illuminant (Planckian at 3000.03 K) to differ
// slightly from the test SPD (Planckian at 3000.00 K), producing CES XYZ
// deltas up to ~6e-4. The golden fixtures from luxpy exhibit the same
// behavior. A tolerance of 1e-3 is appropriate for this self-consistency
// check, reflecting the inherent CCT algorithm imprecision rather than
// an implementation error.
// TM-30-20 §3.3: CCT of a perfect Planckian is not identity.
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("CES colorimetry - planckian self-consistency: test ≈ reference",
          "[pipeline][slice05][self-consistency]") {
  auto &G = GlobalFixtures::instance();

  auto wl = wl_1nm();
  auto spd_vals = generate_planckian(3000.0, wl);

  CesColorimetryResult result =
      compute_ces_colorimetry(wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // For a perfect Planckian source at Tt ≤ 4000 K, the reference is also
  // Planckian at the computed CCT.  Due to Ohno 2014 CCT solver imprecision,
  // the recovered CCT differs from the input temperature by ~0.03 K, causing
  // small CES XYZ differences.
  // TM-30-20 §3.3 Eq. (14): Tt ≤ 4000 K → pure Planckian
  constexpr double self_consistency_tol = 1.0e-3;

  for (std::size_t i = 0; i < 99; ++i) {
    INFO("CES[" << (i + 1) << "] self-consistency");
    CHECK_THAT(result.xyz_test_ces[i].X,
               WithinTolerance(self_consistency_tol, result.xyz_ref_ces[i].X));
    CHECK_THAT(result.xyz_test_ces[i].Y,
               WithinTolerance(self_consistency_tol, result.xyz_ref_ces[i].Y));
    CHECK_THAT(result.xyz_test_ces[i].Z,
               WithinTolerance(self_consistency_tol, result.xyz_ref_ces[i].Z));
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Output shape: exactly 99 CES results
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("CES colorimetry - output shape is exactly 99 CES",
          "[pipeline][slice05]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  REQUIRE(result.xyz_test_ces.size() == 99);
  REQUIRE(result.xyz_ref_ces.size() == 99);
}

// ─────────────────────────────────────────────────────────────────────────
// Y normalisation: test source and reference both Y=100
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("CES colorimetry - Y normalisation: white Y = 100",
          "[pipeline][slice05]") {
  auto &G = GlobalFixtures::instance();

  // Test with D65
  {
    auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

    // Compute test source 10° XYZ directly
    CmfData cmf10 = resample_cmf(spd_wl, G.cmf_10deg);
    SourceXyz test_xyz = compute_source_xyz(spd_wl, spd_vals, cmf10.x_bar,
                                            cmf10.y_bar, cmf10.z_bar);
    REQUIRE_THAT(test_xyz.Y, WithinTolerance(Tol_Xyz, 100.0));

    // Run pipeline and check reference
    CesColorimetryResult result =
        compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg,
                                G.ces, G.daylight_basis, G.planckian_lut);

    // Reference source 10° XYZ should also have Y=100
    CmfData cmf10_r = resample_cmf(spd_wl, G.cmf_10deg);
    SourceXyz ref_xyz =
        compute_source_xyz(spd_wl, result.reference_spd_values, cmf10_r.x_bar,
                           cmf10_r.y_bar, cmf10_r.z_bar);
    REQUIRE_THAT(ref_xyz.Y, WithinTolerance(Tol_Xyz, 100.0));
  }

  // Test with Planckian 3000K
  {
    auto wl = wl_1nm();
    auto spd_vals = generate_planckian(3000.0, wl);

    CmfData cmf10 = resample_cmf(wl, G.cmf_10deg);
    SourceXyz test_xyz =
        compute_source_xyz(wl, spd_vals, cmf10.x_bar, cmf10.y_bar, cmf10.z_bar);
    REQUIRE_THAT(test_xyz.Y, WithinTolerance(Tol_Xyz, 100.0));

    CesColorimetryResult result =
        compute_ces_colorimetry(wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                                G.daylight_basis, G.planckian_lut);

    CmfData cmf10_r = resample_cmf(wl, G.cmf_10deg);
    SourceXyz ref_xyz =
        compute_source_xyz(wl, result.reference_spd_values, cmf10_r.x_bar,
                           cmf10_r.y_bar, cmf10_r.z_bar);
    REQUIRE_THAT(ref_xyz.Y, WithinTolerance(Tol_Xyz, 100.0));
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Reference SPD shape
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("CES colorimetry - reference SPD has correct shape",
          "[pipeline][slice05]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // Reference SPD should have same number of points as input
  REQUIRE(result.reference_spd_values.size() == spd_wl.size());

  // All values should be non-negative
  for (double v : result.reference_spd_values) {
    REQUIRE(v >= 0.0);
  }
}

} // namespace
} // namespace tm30::test
