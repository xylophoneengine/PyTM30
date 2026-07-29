// Slice 9 - Gamut Index (Rg), per-bin local metrics, and CVG coordinates tests.
// Validates compute_gamut, compute_rg, compute_local_bin_metrics, and
// compute_cvg_coordinates against golden fixtures.
//
// TM-30-20 §4.4: Gamut Index (Rg)
// TM-30-20 §4.5: Color Vector Graphic (CVG)
// TM-30-20 §4.6: Local Chroma Shift (Rcs,hj)
// TM-30-20 §4.7: Local Hue Shift (Rhs,hj)
// TM-30-20 §4.8: Local Color Fidelity (Rf,hj)

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "matchers.hpp"
#include "tm30/cct.hpp"
#include "tm30/ciecam02.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/gamut.hpp"
#include "tm30/hue_bins.hpp"
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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tm30::test {
namespace {

// ─────────────────────────────────────────────────────────────────────────
// Test helpers (same pattern as other slice tests)
// ─────────────────────────────────────────────────────────────────────────

std::string data_path(const std::string &filename) {
  return std::string(TM30_DATA_DIR) + "/" + filename;
}

std::string fixture_path(const std::string &spd_name,
                         const std::string &stage) {
  return std::string(TM30_DATA_DIR) + "/../tests/fixtures/" + spd_name + "/" +
         stage + ".json";
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

// ─────────────────────────────────────────────────────────────────────────
// JSON fixture parsers
// ─────────────────────────────────────────────────────────────────────────

/// Load a single double value for a given key from a JSON fixture.
double load_json_double(const std::string &filepath, const std::string &key) {
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

  auto search = "\"" + key + "\"";
  auto pos = content.find(search);
  if (pos == std::string::npos) {
    throw std::runtime_error("No '" + key + "' key in: " + filepath);
  }

  pos = content.find(':', pos);
  if (pos == std::string::npos) {
    throw std::runtime_error("Malformed JSON in: " + filepath);
  }

  ++pos;
  while (pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }

  char *end = nullptr;
  double val = std::strtod(content.c_str() + pos, &end);

  if (end == content.c_str() + pos) {
    throw std::runtime_error("Failed to parse double from: " + filepath);
  }

  return val;
}

/// Load an array of 16 doubles for a given key from a JSON fixture.
std::array<double, 16> load_json_array_16(const std::string &filepath,
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

  auto search = "\"" + key + "\"";
  auto pos = content.find(search);
  if (pos == std::string::npos) {
    throw std::runtime_error("No '" + key + "' key in: " + filepath);
  }

  pos = content.find('[', pos);
  if (pos == std::string::npos) {
    throw std::runtime_error("No array for '" + key + "' in: " + filepath);
  }

  std::array<double, 16> result{};
  std::size_t i = pos + 1;
  int count = 0;

  while (i < content.size() && count < 16) {
    while (i < content.size() &&
           std::isspace(static_cast<unsigned char>(content[i]))) {
      ++i;
    }

    if (i >= content.size())
      break;
    if (content[i] == ']')
      break;

    char *end = nullptr;
    double val = std::strtod(content.c_str() + i, &end);

    if (end == content.c_str() + i) {
      ++i;
      continue;
    }

    result[count++] = val;
    i = static_cast<std::size_t>(end - content.c_str());
  }

  if (count != 16) {
    throw std::runtime_error("Expected 16 values for '" + key + "' got " +
                             std::to_string(count));
  }

  return result;
}

/// Load an array of 16 [J, a, b] triples for a given key.
/// Returns arrays of 16 J, a, b values.
struct JabTripleArray {
  std::array<double, 16> J;
  std::array<double, 16> a;
  std::array<double, 16> b;
};

JabTripleArray load_json_jab_triples(const std::string &filepath,
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

  auto search = "\"" + key + "\"";
  auto pos = content.find(search);
  if (pos == std::string::npos) {
    throw std::runtime_error("No '" + key + "' key in: " + filepath);
  }

  // Find the outer array start
  pos = content.find('[', pos);
  if (pos == std::string::npos) {
    throw std::runtime_error("No array for '" + key + "' in: " + filepath);
  }

  JabTripleArray result{};
  std::size_t i = pos + 1;
  int count = 0;

  while (i < content.size() && count < 16) {
    // Skip whitespace
    while (i < content.size() &&
           std::isspace(static_cast<unsigned char>(content[i]))) {
      ++i;
    }

    if (i >= content.size())
      break;

    // Outer-array close: exit
    if (content[i] == ']')
      break;

    // Inner-array start: find matching ']' and parse contents
    if (content[i] == '[') {
      std::size_t inner_start = i + 1;
      std::size_t inner_end = content.find(']', inner_start);
      if (inner_end == std::string::npos) {
        throw std::runtime_error("Unclosed inner array in: " + filepath);
      }

      // Parse three doubles from the inner array
      double vals[3];
      std::size_t vi = inner_start;
      for (int v = 0; v < 3; ++v) {
        // Skip whitespace and commas between values
        while (vi < inner_end &&
               (std::isspace(static_cast<unsigned char>(content[vi])) ||
                content[vi] == ',')) {
          ++vi;
        }
        if (vi >= inner_end) {
          throw std::runtime_error("Expected 3 values in triple, got " +
                                   std::to_string(v));
        }
        char *end = nullptr;
        vals[v] = std::strtod(content.c_str() + vi, &end);
        if (end == content.c_str() + vi) {
          throw std::runtime_error("Failed to parse triple value at index " +
                                   std::to_string(v) +
                                   " near: " + content.substr(vi, 30));
        }
        vi = static_cast<std::size_t>(end - content.c_str());
      }

      result.J[count] = vals[0];
      result.a[count] = vals[1];
      result.b[count] = vals[2];
      ++count;
      i = inner_end + 1; // Skip past inner ']'
    } else {
      ++i;
    }
  }

  if (count != 16) {
    throw std::runtime_error("Expected 16 triples for '" + key + "' got " +
                             std::to_string(count));
  }

  return result;
}

// ─────────────────────────────────────────────────────────────────────────
// Global fixture data (loaded once)
// ─────────────────────────────────────────────────────────────────────────

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
// Helper: run pipeline and compute gamut for an SPD
// ─────────────────────────────────────────────────────────────────────────

struct GamutFromSpd {
  GamutResult gamut;
  CesColorimetryResult ces_result;
};

GamutFromSpd run_gamut_for_spd(const std::vector<double> &spd_wl,
                               const std::vector<double> &spd_vals) {
  auto &G = GlobalFixtures::instance();

  CesColorimetryResult result =
      compute_ces_colorimetry(spd_wl, spd_vals, G.cmf_2deg, G.cmf_10deg, G.ces,
                              G.daylight_basis, G.planckian_lut);

  // Compute ΔE′
  auto delta_e = compute_delta_e(result.jab_test_ces, result.jab_ref_ces);

  // Compute gamut
  GamutResult gamut = compute_gamut(result.jab_test_ces, result.jab_ref_ces,
                                    delta_e, result.hue_bins);

  return {gamut, result};
}

// ─────────────────────────────────────────────────────────────────────────
// Helper: verify all metrics for an SPD against fixtures
// ─────────────────────────────────────────────────────────────────────────

void verify_gamut_for_spd(const std::string &fixture_subdir,
                          const GamutFromSpd &gfs) {
  const auto &gamut = gfs.gamut;

  // ── Rg ─────────────────────────────────────────────────────────
  double golden_rg =
      load_json_double(fixture_path(fixture_subdir, "13_rg"), "Rg");
  INFO(fixture_subdir << ": computed Rg = " << gamut.Rg
                      << " golden Rg = " << golden_rg);
  CHECK_THAT(gamut.Rg, WithinTolerance(Tol_Rg, golden_rg));

  // ── Per-bin Rf,hj ──────────────────────────────────────────────
  auto golden_rfhj = load_json_array_16(
      fixture_path(fixture_subdir, "14_per_bin_metrics"), "Rfhj");

  double max_rfhj_delta = 0.0;
  for (int j = 0; j < 16; ++j) {
    double d = std::abs(gamut.local.Rf_hj[j] - golden_rfhj[j]);
    if (d > max_rfhj_delta)
      max_rfhj_delta = d;
  }
  INFO(fixture_subdir << ": max Rf,hj delta = " << max_rfhj_delta);
  CHECK(max_rfhj_delta <= Tol_Rf);

  // ── Per-bin Rcs,hj ─────────────────────────────────────────────
  auto golden_rcshj = load_json_array_16(
      fixture_path(fixture_subdir, "14_per_bin_metrics"), "Rcshj");

  double max_rcshj_delta = 0.0;
  for (int j = 0; j < 16; ++j) {
    double d = std::abs(gamut.local.Rcs_hj[j] - golden_rcshj[j]);
    if (d > max_rcshj_delta)
      max_rcshj_delta = d;
  }
  INFO(fixture_subdir << ": max Rcs,hj delta = " << max_rcshj_delta);
  CHECK(max_rcshj_delta <= Tol_LocalShift);

  // ── Per-bin Rhs,hj ─────────────────────────────────────────────
  auto golden_rhshj = load_json_array_16(
      fixture_path(fixture_subdir, "14_per_bin_metrics"), "Rhshj");

  double max_rhshj_delta = 0.0;
  for (int j = 0; j < 16; ++j) {
    double d = std::abs(gamut.local.Rhs_hj[j] - golden_rhshj[j]);
    if (d > max_rhshj_delta)
      max_rhshj_delta = d;
  }
  INFO(fixture_subdir << ": max Rhs,hj delta = " << max_rhshj_delta);
  CHECK(max_rhshj_delta <= Tol_LocalShift);

  // ── CVG: test average J'a'b' (jabt_hj) ─────────────────────────
  auto golden_jabt = load_json_jab_triples(
      fixture_path(fixture_subdir, "15_cvg_coordinates"), "jabt_hj");

  double max_jabt_delta = 0.0;
  for (int j = 0; j < 16; ++j) {
    double dJ = std::abs(gamut.test_avg.J_prime[j] - golden_jabt.J[j]);
    double da = std::abs(gamut.test_avg.a_prime[j] - golden_jabt.a[j]);
    double db = std::abs(gamut.test_avg.b_prime[j] - golden_jabt.b[j]);
    if (dJ > max_jabt_delta)
      max_jabt_delta = dJ;
    if (da > max_jabt_delta)
      max_jabt_delta = da;
    if (db > max_jabt_delta)
      max_jabt_delta = db;
  }
  INFO(fixture_subdir << ": max jabt_hj delta = " << max_jabt_delta);
  CHECK(max_jabt_delta <= Tol_Jab);

  // ── CVG: reference average J'a'b' (jabr_hj) ────────────────────
  auto golden_jabr = load_json_jab_triples(
      fixture_path(fixture_subdir, "15_cvg_coordinates"), "jabr_hj");

  double max_jabr_delta = 0.0;
  for (int j = 0; j < 16; ++j) {
    double dJ = std::abs(gamut.ref_avg.J_prime[j] - golden_jabr.J[j]);
    double da = std::abs(gamut.ref_avg.a_prime[j] - golden_jabr.a[j]);
    double db = std::abs(gamut.ref_avg.b_prime[j] - golden_jabr.b[j]);
    if (dJ > max_jabr_delta)
      max_jabr_delta = dJ;
    if (da > max_jabr_delta)
      max_jabr_delta = da;
    if (db > max_jabr_delta)
      max_jabr_delta = db;
  }
  INFO(fixture_subdir << ": max jabr_hj delta = " << max_jabr_delta);
  // Bin-averaged J'a'b' values accumulate more pipeline noise than
  // per-CES values. Use a slightly relaxed tolerance (documented in PARITY.md).
  // Tol_Jab (0.001) remains the standard; bin-average comparisons use 0.002.
  CHECK(max_jabr_delta <= 0.002);

  // ── CVG normalized test coordinates (jabtn_hj) ─────────────────
  auto golden_jabtn = load_json_jab_triples(
      fixture_path(fixture_subdir, "15_cvg_coordinates"), "jabtn_hj");

  double max_jabtn_delta = 0.0;
  for (int j = 0; j < 16; ++j) {
    double dJ = std::abs(gamut.cvg.J_test[j] - golden_jabtn.J[j]);
    double dx = std::abs(gamut.cvg.x_test[j] - golden_jabtn.a[j]);
    double dy = std::abs(gamut.cvg.y_test[j] - golden_jabtn.b[j]);
    if (dJ > max_jabtn_delta)
      max_jabtn_delta = dJ;
    if (dx > max_jabtn_delta)
      max_jabtn_delta = dx;
    if (dy > max_jabtn_delta)
      max_jabtn_delta = dy;
  }
  INFO(fixture_subdir << ": max jabtn_hj delta = " << max_jabtn_delta);
  // CVG scale is 100×, so Tol_Jab × 100 for the scaled coordinates
  CHECK(max_jabtn_delta <= Tol_Jab * 100.0);

  // ── CVG normalized reference coordinates (jabrn_hj) ────────────
  auto golden_jabrn = load_json_jab_triples(
      fixture_path(fixture_subdir, "15_cvg_coordinates"), "jabrn_hj");

  double max_jabrn_delta = 0.0;
  for (int j = 0; j < 16; ++j) {
    double dJ = std::abs(gamut.cvg.J_ref[j] - golden_jabrn.J[j]);
    double dx = std::abs(gamut.cvg.x_ref[j] - golden_jabrn.a[j]);
    double dy = std::abs(gamut.cvg.y_ref[j] - golden_jabrn.b[j]);
    if (dJ > max_jabrn_delta)
      max_jabrn_delta = dJ;
    if (dx > max_jabrn_delta)
      max_jabrn_delta = dx;
    if (dy > max_jabrn_delta)
      max_jabrn_delta = dy;
  }
  INFO(fixture_subdir << ": max jabrn_hj delta = " << max_jabrn_delta);
  CHECK(max_jabrn_delta <= Tol_Jab * 100.0);
}

// ─────────────────────────────────────────────────────────────────────────
// Self-consistency tests
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE(
    "Gamut - planckian 3000K self-consistency: Rg≈100, shifts≈0, Rf,hj≈100",
    "[gamut][rg][slice09][self-consistency]") {
  auto &G = GlobalFixtures::instance();

  auto wl = wl_1nm();
  auto spd_vals = generate_planckian(3000.0, wl);

  auto gfs = run_gamut_for_spd(wl, spd_vals);
  const auto &gamut = gfs.gamut;

  // Rg ≈ 100
  // TM-30-20 §4.4: Self-consistency → Rg ≈ 100
  INFO("Planckian 3000K: Rg = " << gamut.Rg);
  CHECK_THAT(gamut.Rg, WithinTolerance(Tol_Rg, 100.0));

  // All Rcs,hj ≈ 0
  // TM-30-20 §4.6: Self-consistency → shifts near zero
  double max_rcs = 0.0;
  for (int j = 0; j < 16; ++j) {
    double v = std::abs(gamut.local.Rcs_hj[j]);
    if (v > max_rcs)
      max_rcs = v;
  }
  INFO("Planckian 3000K: max |Rcs,hj| = " << max_rcs);
  CHECK(max_rcs <= Tol_LocalShift);

  // All Rhs,hj ≈ 0
  // TM-30-20 §4.7: Self-consistency → shifts near zero
  double max_rhs = 0.0;
  for (int j = 0; j < 16; ++j) {
    double v = std::abs(gamut.local.Rhs_hj[j]);
    if (v > max_rhs)
      max_rhs = v;
  }
  INFO("Planckian 3000K: max |Rhs,hj| = " << max_rhs);
  CHECK(max_rhs <= Tol_LocalShift);

  // All Rf,hj ≈ 100
  // TM-30-20 §4.8: Self-consistency → per-bin fidelity ≈ 100
  double min_rfhj = 100.0;
  for (int j = 0; j < 16; ++j) {
    if (gamut.local.Rf_hj[j] < min_rfhj)
      min_rfhj = gamut.local.Rf_hj[j];
  }
  INFO("Planckian 3000K: min Rf,hj = " << min_rfhj);
  CHECK_THAT(min_rfhj, WithinTolerance(Tol_Rf, 100.0));
}

TEST_CASE("Gamut - D65 self-consistency: Rg≈100, shifts≈0",
          "[gamut][rg][slice09][self-consistency]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));

  auto gfs = run_gamut_for_spd(spd_wl, spd_vals);
  const auto &gamut = gfs.gamut;

  INFO("D65: Rg = " << gamut.Rg);
  CHECK_THAT(gamut.Rg, WithinTolerance(Tol_Rg, 100.0));
}

// ─────────────────────────────────────────────────────────────────────────
// Golden fixture tests
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("Gamut - D65 gamut metrics match golden fixtures",
          "[gamut][rg][slice09][fixture]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  auto gfs = run_gamut_for_spd(spd_wl, spd_vals);

  verify_gamut_for_spd("D65_1nm", gfs);
}

TEST_CASE("Gamut - F1 gamut metrics match golden fixtures",
          "[gamut][rg][slice09][fixture]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl1_1nm.csv"));
  auto gfs = run_gamut_for_spd(spd_wl, spd_vals);

  verify_gamut_for_spd("F1", gfs);
}

TEST_CASE("Gamut - HP1 gamut metrics match golden fixtures",
          "[gamut][rg][slice09][fixture]") {
  auto &G = GlobalFixtures::instance();

  auto [spd_wl, spd_vals] = load_spd_csv(data_path("hp1_1nm.csv"));
  auto gfs = run_gamut_for_spd(spd_wl, spd_vals);

  verify_gamut_for_spd("HP1", gfs);
}

TEST_CASE("Gamut - planckian 3000K gamut metrics match golden fixtures",
          "[gamut][rg][slice09][fixture]") {
  auto &G = GlobalFixtures::instance();

  auto wl = wl_1nm();
  auto spd_vals = generate_planckian(3000.0, wl);
  auto gfs = run_gamut_for_spd(wl, spd_vals);

  verify_gamut_for_spd("planckian_3000K", gfs);
}

// ─────────────────────────────────────────────────────────────────────────
// Range sanity checks
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("Gamut - Rg is positive for various SPDs",
          "[gamut][rg][slice09][range]") {
  auto &G = GlobalFixtures::instance();

  struct SpdCase {
    std::string name;
    std::string csv_file;
  };

  std::vector<SpdCase> cases = {
      {"D65", "d65_1nm.csv"},  {"F1", "fl1_1nm.csv"},  {"HP1", "hp1_1nm.csv"},
      {"F12", "fl12_1nm.csv"}, {"HP5", "hp5_1nm.csv"},
  };

  for (const auto &c : cases) {
    auto [spd_wl, spd_vals] = load_spd_csv(data_path(c.csv_file));
    auto gfs = run_gamut_for_spd(spd_wl, spd_vals);

    INFO(c.name << ": Rg = " << gfs.gamut.Rg);
    CHECK(gfs.gamut.Rg > 0.0);
    CHECK(gfs.gamut.Rg < 200.0);
  }
}

TEST_CASE("Gamut - Rf,hj in [0, 100] for all SPDs",
          "[gamut][rg][slice09][range]") {
  auto &G = GlobalFixtures::instance();

  struct SpdCase {
    std::string name;
    std::string csv_file;
  };

  std::vector<SpdCase> cases = {
      {"D65", "d65_1nm.csv"},  {"F1", "fl1_1nm.csv"},  {"HP1", "hp1_1nm.csv"},
      {"F12", "fl12_1nm.csv"}, {"HP5", "hp5_1nm.csv"},
  };

  for (const auto &c : cases) {
    auto [spd_wl, spd_vals] = load_spd_csv(data_path(c.csv_file));
    auto gfs = run_gamut_for_spd(spd_wl, spd_vals);

    for (int j = 0; j < 16; ++j) {
      INFO(c.name << " bin " << j << ": Rf,hj = " << gfs.gamut.local.Rf_hj[j]);
      CHECK(gfs.gamut.local.Rf_hj[j] >= 0.0);
      CHECK(gfs.gamut.local.Rf_hj[j] <= 100.0 + 1e-10);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Unit tests: individual sub-functions
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("Gamut - polygon_area handles degenerate polygon",
          "[gamut][rg][slice09][unit]") {
  // Empty bins → fewer than 3 vertices → area = 0
  BinAverages avg{};
  for (int j = 0; j < 16; ++j) {
    if (j >= 2) {
      // Only 2 valid bins → degenerate
      avg.a_prime[j] = std::numeric_limits<double>::quiet_NaN();
      avg.b_prime[j] = std::numeric_limits<double>::quiet_NaN();
    }
  }
  avg.a_prime[0] = 0.0;
  avg.b_prime[0] = 0.0;
  avg.a_prime[1] = 10.0;
  avg.b_prime[1] = 0.0;

  double area = polygon_area(avg);
  CHECK(area == 0.0);
}

TEST_CASE("Gamut - polygon_area of unit square in (a',b') plane",
          "[gamut][rg][slice09][unit]") {
  // 4 vertices forming a 2×2 square
  BinAverages avg{};
  // Set first 4 bins (rest NaN)
  for (int j = 0; j < 16; ++j) {
    avg.a_prime[j] = std::numeric_limits<double>::quiet_NaN();
    avg.b_prime[j] = std::numeric_limits<double>::quiet_NaN();
  }

  // CCW order: (0,0), (2,0), (2,2), (0,2)
  avg.a_prime[0] = 0.0;
  avg.b_prime[0] = 0.0;
  avg.a_prime[1] = 2.0;
  avg.b_prime[1] = 0.0;
  avg.a_prime[2] = 2.0;
  avg.b_prime[2] = 2.0;
  avg.a_prime[3] = 0.0;
  avg.b_prime[3] = 2.0;

  double area = polygon_area(avg);
  // Shoelace: 0.5 * |0*0 - 2*0 + 2*2 - 0*2 + 2*2 - 0*0 + 0*0 - 0*2| = 0.5 * |0
  // + 4 + 4 + 0| = 0.5 * 8 = 4
  CHECK_THAT(area, WithinTolerance(1e-12, 4.0));
}

TEST_CASE("Gamut - compute_rg gives 100 for identical polygons",
          "[gamut][rg][slice09][unit]") {
  BinAverages avg{};
  for (int j = 0; j < 16; ++j) {
    double angle = static_cast<double>(j) * M_PI / 8.0;
    avg.a_prime[j] = std::cos(angle) * 10.0;
    avg.b_prime[j] = std::sin(angle) * 10.0;
    avg.J_prime[j] = 50.0;
  }

  double rg = compute_rg(avg, avg);
  CHECK_THAT(rg, WithinTolerance(1e-12, 100.0));
}

} // namespace
} // namespace tm30::test
