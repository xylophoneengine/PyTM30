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
#include <utility>
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

/// Generate a 401-point 1nm wavelength grid 380-780 nm.
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

// ═══════════════════════════════════════════════════════════════════════════
// Grid-matrix pipeline equivalence tests.
//
// `compute_ces_xyz` was rewritten (see slice_02's "CES weights-based
// rewrite agrees with reference algorithm" test) to compute
// trapezoidal_weights() once per call instead of re-deriving per-CES
// integrands. This section proves that rewrite holds up through the *full*
// pipeline, across a matrix of wavelength grids, and across both call paths
// that reach compute_ces_xyz from pipeline.cpp:
//   (A) compute_ces_colorimetry()         - non-cached, resamples every call
//   (B) compute_ces_colorimetry_cached()  - cached, resamples once via
//                                           prepare_resampled_tables()
// Neither pipeline.cpp nor xyz.cpp is modified by this test file.
// ═══════════════════════════════════════════════════════════════════════════

namespace tm30::test {
namespace {

// ─────────────────────────────────────────────────────────────────────────
// Grid builders
// ─────────────────────────────────────────────────────────────────────────

/// Uniform grid from lo to hi inclusive (hi assumed reachable from lo by an
/// integer number of `step`s), used for grid rows 2, 4, 5.
std::vector<double> wl_uniform(double lo, double hi, double step) {
  std::vector<double> wl;
  const int n = static_cast<int>(std::llround((hi - lo) / step)) + 1;
  wl.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    wl.push_back(lo + step * static_cast<double>(i));
  }
  return wl;
}

/// Non-uniform grid: 1 nm steps below 500 nm, 2 nm steps at/above 500 nm,
/// full 380-780 nm span (grid row 3).
std::vector<double> wl_nonuniform_1_2() {
  std::vector<double> wl;
  for (double w = 380.0; w < 500.0; w += 1.0) {
    wl.push_back(w);
  }
  for (double w = 500.0; w <= 780.0 + 1e-9; w += 2.0) {
    wl.push_back(w);
  }
  return wl;
}

// ─────────────────────────────────────────────────────────────────────────
// Two "real" SPDs on a shared grid, for the (A)-vs-(B) and batch checks.
//
// Where a native measured CSV already sits on the exact grid under test
// (the 1 nm and 5 nm rows), we use it directly. For grids with no native
// CSV (non-uniform / narrow / fine), we generate a Planckian radiator and
// a CIE D-series daylight illuminant directly on the target grid via
// generate_planckian()/generate_cie_d() - both are standard, physically
// meaningful reference illuminants (the same generators the pipeline
// itself uses for the TM-30-20 reference SPD), not hand-rolled
// interpolation of arbitrary data. generate_cie_d() internally resamples
// the daylight basis via resample_daylight_basis() (existing library
// function) to the target grid.
// ─────────────────────────────────────────────────────────────────────────

std::pair<std::vector<double>, std::vector<double>>
two_synthetic_illuminants(const std::vector<double> &wl,
                          const DaylightBasis &basis) {
  std::vector<double> planckian_2700 = generate_planckian(2700.0, wl);
  std::vector<double> cie_d_6500 = generate_cie_d(6500.0, wl, basis);
  return {planckian_2700, cie_d_6500};
}

// ─────────────────────────────────────────────────────────────────────────
// Result comparison helpers
//
// Field set and tolerances match this file's/this suite's established
// conventions (see tolerances.hpp and existing usage in
// slice_09_rg_local_cvg_test.cpp, slice_10_sample_test.cpp): Rf-scale
// quantities (Rf itself, per-sample rf_cesi, rf_skin, per-bin Rf_hj) use
// Tol_Rf; ΔE'-scale quantities (delta_e_avg, per-bin DE_hj) use
// Tol_DeltaE; per-bin local shift fields use Tol_LocalShift; Rg uses
// Tol_Rg. No new tolerances are introduced.
// ─────────────────────────────────────────────────────────────────────────

/// NaN-aware absolute-tolerance comparison. Per gamut.hpp's documented
/// contract ("Skips empty bins (stores NaN for averages)"), per-bin gamut
/// fields are legitimately NaN whenever a hue bin has no CES assigned to
/// it (e.g. the row-6 minimal 2-point grid, where only a handful of the
/// 99 CES land in distinct bins). Both call paths run the identical
/// bin-assignment logic on identical inputs, so they land on NaN for the
/// exact same bins - that agreement (not "not NaN") is what's being
/// verified here, hence NaN==NaN is treated as a match.
bool nan_aware_close(double x, double y, double tol) {
  if (std::isnan(x) || std::isnan(y)) {
    return std::isnan(x) && std::isnan(y);
  }
  return std::abs(x - y) <= tol;
}

/// Compare two CesColorimetryResult values field-by-field at spec
/// tolerance. Uses CHECK (not REQUIRE) so a single mismatched field
/// doesn't hide others in the same run.
void check_results_match(const CesColorimetryResult &a,
                         const CesColorimetryResult &b,
                         const std::string &label) {
  INFO(label);

  CHECK(nan_aware_close(a.cct, b.cct, Tol_Cct));
  CHECK(nan_aware_close(a.duv, b.duv, Tol_Duv));
  CHECK(nan_aware_close(a.Rf, b.Rf, Tol_Rf));
  CHECK(nan_aware_close(a.delta_e_avg, b.delta_e_avg, Tol_DeltaE));
  CHECK(nan_aware_close(a.rf_skin, b.rf_skin, Tol_Rf));
  CHECK(nan_aware_close(a.gamut.Rg, b.gamut.Rg, Tol_Rg));

  for (std::size_t i = 0; i < 99; ++i) {
    INFO(label << " rf_cesi[" << i << "] a=" << a.rf_cesi[i]
               << " b=" << b.rf_cesi[i]);
    CHECK(nan_aware_close(a.rf_cesi[i], b.rf_cesi[i], Tol_Rf));
  }

  for (std::size_t j = 0; j < 16; ++j) {
    INFO(label << " hue bin " << j);
    CHECK(nan_aware_close(a.gamut.local.Rf_hj[j], b.gamut.local.Rf_hj[j],
                          Tol_Rf));
    CHECK(nan_aware_close(a.gamut.local.Rcs_hj[j], b.gamut.local.Rcs_hj[j],
                          Tol_LocalShift));
    CHECK(nan_aware_close(a.gamut.local.Rhs_hj[j], b.gamut.local.Rhs_hj[j],
                          Tol_LocalShift));
    CHECK(nan_aware_close(a.gamut.local.DE_hj[j], b.gamut.local.DE_hj[j],
                          Tol_DeltaE));
  }
}

/// Compare two CesColorimetryResult values as an implementation-
/// determinism check (same tables, same SPD, two separate calls) rather
/// than a spec-tolerance check: 1e-9 absolute, matching the convention
/// slice_02_xyz_test.cpp uses for its old-vs-new compute_ces_xyz
/// regression test. NaN-aware for the same reason as check_results_match.
void check_results_identical(const CesColorimetryResult &a,
                             const CesColorimetryResult &b,
                             const std::string &label) {
  constexpr double kDeterminismTol = 1e-9;
  INFO(label);

  CHECK(nan_aware_close(a.cct, b.cct, kDeterminismTol));
  CHECK(nan_aware_close(a.duv, b.duv, kDeterminismTol));
  CHECK(nan_aware_close(a.Rf, b.Rf, kDeterminismTol));
  CHECK(nan_aware_close(a.delta_e_avg, b.delta_e_avg, kDeterminismTol));
  CHECK(nan_aware_close(a.rf_skin, b.rf_skin, kDeterminismTol));
  CHECK(nan_aware_close(a.gamut.Rg, b.gamut.Rg, kDeterminismTol));

  for (std::size_t i = 0; i < 99; ++i) {
    INFO(label << " rf_cesi[" << i << "]");
    CHECK(nan_aware_close(a.rf_cesi[i], b.rf_cesi[i], kDeterminismTol));
  }
  for (std::size_t j = 0; j < 16; ++j) {
    INFO(label << " hue bin " << j);
    CHECK(nan_aware_close(a.gamut.local.Rcs_hj[j], b.gamut.local.Rcs_hj[j],
                          kDeterminismTol));
    CHECK(nan_aware_close(a.gamut.local.Rhs_hj[j], b.gamut.local.Rhs_hj[j],
                          kDeterminismTol));
  }
}

/// Run both call paths (A: non-cached, B: cached) for one SPD on one grid.
/// Building `tables` is left to the caller so that batch-coverage tests can
/// share a single ResampledTables instance across multiple SPDs.
CesColorimetryResult run_noncached(const std::vector<double> &wl,
                                   const std::vector<double> &spd,
                                   const GlobalFixtures &G) {
  return compute_ces_colorimetry(wl, spd, G.cmf_2deg, G.cmf_10deg, G.ces,
                                 G.daylight_basis, G.planckian_lut);
}

CesColorimetryResult run_cached(const std::vector<double> &spd,
                                const ResampledTables &tables,
                                const GlobalFixtures &G) {
  return compute_ces_colorimetry_cached(spd, tables, G.planckian_lut);
}

/// Row 1-5 driver: build `tables` once for `wl`, then for each of the two
/// given SPDs (sharing that grid) confirm the cached-path result matches
/// its own non-cached-path result. Because both SPDs are evaluated against
/// the *same* `tables` instance, this simultaneously satisfies the
/// per-brief (A)-vs-(B) comparison and the "batch" coverage requirement
/// (multiple SPDs sharing one ResampledTables, proving no state leaks
/// between calls).
void check_row_both_paths(const std::vector<double> &wl,
                          const std::vector<double> &spd_1,
                          const std::string &spd_1_name,
                          const std::vector<double> &spd_2,
                          const std::string &spd_2_name,
                          const std::string &row_name) {
  auto &G = GlobalFixtures::instance();

  ResampledTables tables = prepare_resampled_tables(wl, G.cmf_2deg, G.cmf_10deg,
                                                    G.ces, G.daylight_basis);

  {
    CesColorimetryResult non_cached = run_noncached(wl, spd_1, G);
    CesColorimetryResult cached = run_cached(spd_1, tables, G);
    check_results_match(non_cached, cached, row_name + " / " + spd_1_name);
  }
  {
    CesColorimetryResult non_cached = run_noncached(wl, spd_2, G);
    CesColorimetryResult cached = run_cached(spd_2, tables, G);
    check_results_match(non_cached, cached, row_name + " / " + spd_2_name);
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Row 1: Default 1 nm, 380-780 nm (401 pts) - standard case.
// Real measured SPDs already native to this exact grid: D65, F1.
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("Grid matrix row 1 - default 1nm 380-780nm: cached matches "
          "non-cached (D65, F1)",
          "[pipeline][slice05][gridmatrix]") {
  auto wl = wl_1nm();
  REQUIRE(wl.size() == 401);

  auto [d65_wl, d65_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  auto [f1_wl, f1_vals] = load_spd_csv(data_path("fl1_1nm.csv"));
  REQUIRE(d65_wl.size() == 401);
  REQUIRE(f1_wl.size() == 401);

  check_row_both_paths(wl, d65_vals, "D65", f1_vals, "F1", "row1-1nm");
}

// ─────────────────────────────────────────────────────────────────────────
// Row 2: Uniform 5 nm, 380-780 nm (81 pts) - coarser native grid (HP
// lamps). HP1 and HP2 are real measured SPDs already native to this grid.
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("Grid matrix row 2 - uniform 5nm 380-780nm: cached matches "
          "non-cached (HP1, HP2)",
          "[pipeline][slice05][gridmatrix]") {
  auto [hp1_wl, hp1_vals] = load_spd_csv(data_path("hp1_1nm.csv"));
  auto [hp2_wl, hp2_vals] = load_spd_csv(data_path("hp2_1nm.csv"));
  REQUIRE(hp1_wl.size() == 81);
  REQUIRE(hp2_wl.size() == 81);
  REQUIRE(hp1_wl == hp2_wl); // must share the exact same grid

  check_row_both_paths(hp1_wl, hp1_vals, "HP1", hp2_vals, "HP2", "row2-5nm");
}

// ─────────────────────────────────────────────────────────────────────────
// Row 3: Non-uniform grid - 1 nm below 500 nm, 2 nm at/above 500 nm,
// full 380-780 nm span. Exercises non-uniform Δλ end-to-end (resample_ces/
// resample_cmf/resample_daylight_basis, called internally by
// prepare_resampled_tables() and compute_ces_colorimetry(), handle the
// non-uniform target grid transparently).
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("Grid matrix row 3 - non-uniform 1nm/2nm grid: cached matches "
          "non-cached (Planckian 2700K, CIE D 6500K)",
          "[pipeline][slice05][gridmatrix]") {
  auto &G = GlobalFixtures::instance();
  auto wl = wl_nonuniform_1_2();
  REQUIRE(wl.size() > 100); // sanity: genuinely non-trivial grid
  REQUIRE(wl.front() == 380.0);
  REQUIRE(wl.back() == 780.0);

  auto [spd_a, spd_b] = two_synthetic_illuminants(wl, G.daylight_basis);

  check_row_both_paths(wl, spd_a, "Planckian2700K", spd_b, "CIE_D6500K",
                       "row3-nonuniform");
}

// ─────────────────────────────────────────────────────────────────────────
// Row 4: Narrow custom range, 450-650 nm, 1 nm steps (201 pts) - boundary
// handling.
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("Grid matrix row 4 - narrow 450-650nm range: cached matches "
          "non-cached (Planckian 2700K, CIE D 6500K)",
          "[pipeline][slice05][gridmatrix]") {
  auto &G = GlobalFixtures::instance();
  auto wl = wl_uniform(450.0, 650.0, 1.0);
  REQUIRE(wl.size() == 201);

  auto [spd_a, spd_b] = two_synthetic_illuminants(wl, G.daylight_basis);

  check_row_both_paths(wl, spd_a, "Planckian2700K", spd_b, "CIE_D6500K",
                       "row4-narrow450-650");
}

// ─────────────────────────────────────────────────────────────────────────
// Row 5: Very fine grid, 0.5 nm over 450-650 nm (401 pts, 200 nm wide) -
// high point-count performance/precision.
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("Grid matrix row 5 - fine 0.5nm grid over 450-650nm: cached "
          "matches non-cached (Planckian 2700K, CIE D 6500K)",
          "[pipeline][slice05][gridmatrix]") {
  auto &G = GlobalFixtures::instance();
  auto wl = wl_uniform(450.0, 650.0, 0.5);
  REQUIRE(wl.size() == 401);

  auto [spd_a, spd_b] = two_synthetic_illuminants(wl, G.daylight_basis);

  check_row_both_paths(wl, spd_a, "Planckian2700K", spd_b, "CIE_D6500K",
                       "row5-fine0.5nm");
}

// ─────────────────────────────────────────────────────────────────────────
// Row 6: Minimal 2-point grid - edge case. Empirically (verified while
// developing this test), a 2-point grid does not throw in either call
// path: it produces finite (if physically nonsensical, given only 2
// spectral samples) cct/duv/Rf/Rg values. Both call paths must agree
// (either both throw the same way, or both succeed with equivalent
// results) - this guards against the two code paths diverging on this
// edge case even though it's outside normal operating range.
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("Grid matrix row 6 - minimal 2-point grid: cached/non-cached "
          "agree (no crash either way)",
          "[pipeline][slice05][gridmatrix]") {
  auto &G = GlobalFixtures::instance();
  std::vector<double> wl = {400.0, 700.0};
  std::vector<double> spd = {50.0, 55.0};

  bool noncached_threw = false, cached_threw = false;
  std::string noncached_what, cached_what;
  CesColorimetryResult non_cached{};
  CesColorimetryResult cached{};

  try {
    non_cached = run_noncached(wl, spd, G);
  } catch (const std::exception &e) {
    noncached_threw = true;
    noncached_what = e.what();
  }

  ResampledTables tables = prepare_resampled_tables(wl, G.cmf_2deg, G.cmf_10deg,
                                                    G.ces, G.daylight_basis);
  try {
    cached = run_cached(spd, tables, G);
  } catch (const std::exception &e) {
    cached_threw = true;
    cached_what = e.what();
  }

  INFO("non-cached threw=" << noncached_threw << " (" << noncached_what
                           << "); cached threw=" << cached_threw << " ("
                           << cached_what << ")");
  REQUIRE(noncached_threw == cached_threw);

  if (noncached_threw) {
    // Both paths must fail the same way, not just "both fail".
    CHECK(noncached_what == cached_what);
  } else {
    // Neither threw: values must be finite (no UB-adjacent NaN escaping
    // to the caller for a well-formed, if minimal, 2-point SPD) and the
    // two paths must agree.
    CHECK_FALSE(std::isnan(non_cached.cct));
    CHECK_FALSE(std::isnan(non_cached.duv));
    CHECK_FALSE(std::isnan(non_cached.Rf));
    CHECK_FALSE(std::isnan(non_cached.gamut.Rg));
    check_results_match(non_cached, cached, "row6-2point");
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Row 7: All-zero SPD on the default 1 nm grid - degenerate input.
//
// Empirically (verified while developing this test, including an
// UndefinedBehaviorSanitizer run), an all-zero SPD does NOT throw in
// either call path. compute_source_xyz's normalisation k = 100/∫St·ȳ dλ
// divides by zero (∫ = 0 for an all-zero SPD), producing k = +inf, and
// the subsequent inf*0 multiplications propagate IEEE-754 NaN through
// cct/duv/Rf/Rg/rf_cesi/gamut - deterministically, not a crash, in the
// plain (non-sanitized) build this suite runs.
//
// NOTE (pre-existing, out of scope for this test-only task): under
// UBSan, this input also flags `static_cast<int>(h / kBinWidth)` in
// src/tm30/hue_bins.cpp:34 as undefined behavior when `h` is NaN
// (atan2(NaN, NaN) = NaN propagates through bin_by_hue's hue-angle
// computation). On this platform the NaN-to-int cast happens to yield 0,
// which stays in-bounds for the 16-slot HueBins array, so no crash is
// observed here - but per the C++ standard this is undefined behavior
// that a different compiler/architecture/optimization level could turn
// into an out-of-bounds array access. This predates and is unrelated to
// this plan's compute_ces_xyz/trapezoidal_weights rewrite (Tasks 1-2);
// hue_bins.cpp is not touched by this task's scope. Flagged here as a
// discovered issue for a future fix, not fixed in this test-only change.
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("Grid matrix row 7 - all-zero SPD: cached/non-cached agree (no "
          "crash either way)",
          "[pipeline][slice05][gridmatrix]") {
  auto &G = GlobalFixtures::instance();
  auto wl = wl_1nm();
  std::vector<double> spd(wl.size(), 0.0);

  bool noncached_threw = false, cached_threw = false;
  std::string noncached_what, cached_what;
  CesColorimetryResult non_cached{};
  CesColorimetryResult cached{};

  try {
    non_cached = run_noncached(wl, spd, G);
  } catch (const std::exception &e) {
    noncached_threw = true;
    noncached_what = e.what();
  }

  ResampledTables tables = prepare_resampled_tables(wl, G.cmf_2deg, G.cmf_10deg,
                                                    G.ces, G.daylight_basis);
  try {
    cached = run_cached(spd, tables, G);
  } catch (const std::exception &e) {
    cached_threw = true;
    cached_what = e.what();
  }

  INFO("non-cached threw=" << noncached_threw << " (" << noncached_what
                           << "); cached threw=" << cached_threw << " ("
                           << cached_what << ")");
  REQUIRE(noncached_threw == cached_threw);

  if (noncached_threw) {
    CHECK(noncached_what == cached_what);
  } else {
    // Both paths must land on the exact same degenerate (NaN-or-finite)
    // outcome for every field - isnan-aware equality, since NaN != NaN.
    auto agree = [](double x, double y) {
      return (std::isnan(x) && std::isnan(y)) || x == y;
    };
    CHECK(agree(non_cached.cct, cached.cct));
    CHECK(agree(non_cached.duv, cached.duv));
    CHECK(agree(non_cached.Rf, cached.Rf));
    CHECK(agree(non_cached.gamut.Rg, cached.gamut.Rg));
    for (std::size_t i = 0; i < 99; ++i) {
      INFO("rf_cesi[" << i << "]");
      CHECK(agree(non_cached.rf_cesi[i], cached.rf_cesi[i]));
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Tables immutability regression guard (grid row 1's grid): building
// `tables` once and calling compute_ces_colorimetry_cached against it
// twice - with an unrelated compute_ces_colorimetry() call (its own fresh
// grid/SPD/tables) sandwiched in between - must give identical results
// both times. compute_ces_colorimetry() takes ResampledTables's four
// source tables as their own fresh const-ref arguments, not `tables`
// itself, so it cannot mutate `tables`; this test asserts that invariant
// explicitly rather than relying on reading the (correct, const-ref)
// function signatures.
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("Tables immutability - unrelated compute_ces_colorimetry call "
          "does not perturb a cached ResampledTables",
          "[pipeline][slice05][gridmatrix]") {
  auto &G = GlobalFixtures::instance();

  auto wl = wl_1nm();
  ResampledTables tables = prepare_resampled_tables(wl, G.cmf_2deg, G.cmf_10deg,
                                                    G.ces, G.daylight_basis);

  auto [d65_wl, spd_a] = load_spd_csv(data_path("d65_1nm.csv"));
  REQUIRE(d65_wl == wl);

  CesColorimetryResult result_1 = run_cached(spd_a, tables, G);

  // Unrelated call: different explicit grid (5 nm, HP1's grid), different
  // SPD, its own fresh cmf/ces/daylight-basis tables.
  auto [hp1_wl, hp1_vals] = load_spd_csv(data_path("hp1_1nm.csv"));
  CesColorimetryResult interloper =
      compute_ces_colorimetry(hp1_wl, hp1_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);
  // Touch the interloper result so it isn't optimized away and so this
  // call is genuinely exercised, not just constructed.
  REQUIRE_FALSE(std::isnan(interloper.cct));

  CesColorimetryResult result_2 = run_cached(spd_a, tables, G);

  check_results_identical(result_1, result_2, "tables-immutability");
}

} // namespace
} // namespace tm30::test
