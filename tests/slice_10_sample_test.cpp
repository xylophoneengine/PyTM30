// Slice 10 - Per-sample fidelity Rf,CESi, Rf,skin, Annex E.
// TM-30-20 §4.2, Annex E.

#include <catch2/catch_test_macros.hpp>

#include "matchers.hpp"
#include "tm30/cct.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/metrics.hpp"
#include "tm30/pipeline.hpp"
#include "tm30/reference.hpp"
#include "tm30/resample.hpp"
#include "tolerances.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace tm30::test {
namespace {

std::string data_path(const std::string &filename) {
  return std::string(TM30_DATA_DIR) + "/" + filename;
}

std::string fixture_path(const std::string &spd, const std::string &stage) {
  return std::string(TM30_FIXTURE_DIR) + "/" + spd + "/" + stage + ".json";
}

// Simple JSON value reader
double read_json_value(const std::string &path, const std::string &key) {
  std::ifstream f(path);
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  auto pos = content.find("\"" + key + "\"");
  if (pos == std::string::npos)
    throw std::runtime_error("key not found: " + key);
  pos = content.find(":", pos);
  pos = content.find_first_not_of(" \t\n\r", pos + 1);
  return std::stod(content.substr(pos));
}

std::pair<std::vector<double>, std::vector<double>>
load_spd_csv(const std::string &path) {
  CsvTable table = load_csv(path);
  std::vector<double> wl, vals;
  for (const auto &row : table.rows) {
    wl.push_back(row[0]);
    vals.push_back(row[1]);
  }
  return {wl, vals};
}

CmfData load_cmf(const std::string &path) {
  CmfData data;
  CsvTable table = load_csv(path);
  for (const auto &row : table.rows) {
    data.wavelengths.push_back(row[0]);
    data.x_bar.push_back(row[1]);
    data.y_bar.push_back(row[2]);
    data.z_bar.push_back(row[3]);
  }
  return data;
}

CesData load_ces(const std::string &path) {
  CsvTable table = load_csv(path);
  // ces.csv format: wavelength, CES01, CES02, ..., CES99
  // 401 rows × 100 columns
  CesData data;
  std::size_t n_ces = table.headers.size() - 1; // 99 CES columns

  // Initialize 99 empty sample vectors
  data.samples.resize(n_ces);

  for (const auto &row : table.rows) {
    data.wavelengths.push_back(row[0]);
    for (std::size_t c = 1; c < row.size(); ++c) {
      data.samples[c - 1].push_back(row[c]);
    }
  }
  return data;
}

// ── Rf,CESi ────────────────────────────────────────────────────────

TEST_CASE("Sample fidelity - Rf,CESi for known deltaE values",
          "[sample][slice10]") {
  // If all ΔE' = 0, Rf,CESi should be ≈100 (log rescaling gives ~100.00045)
  std::array<double, 99> zero_de;
  zero_de.fill(0.0);
  auto rf_cesi = compute_rf_cesi(zero_de);
  for (int i = 0; i < 99; ++i) {
    // Rf' = 100, Rf = 10·ln(exp(10)+1) ≈ 100.00045
    REQUIRE_THAT(rf_cesi[i], WithinTolerance(0.001, 100.0));
  }

  // If ΔE' = 10, Rf,CESi' = 100 - 6.73*10 = 32.7
  // Rf,CESi = 10 * ln(exp(3.27) + 1) ≈ 10 * ln(26.31 + 1) = 10 * 3.309 = 33.09
  std::array<double, 99> de10;
  de10.fill(10.0);
  auto rf_cesi10 = compute_rf_cesi(de10);
  double expected =
      10.0 * std::log(std::exp((100.0 - 6.73 * 10.0) / 10.0) + 1.0);
  REQUIRE_THAT(rf_cesi10[0], WithinTolerance(1e-4, expected));
}

// ── Rf,skin ─────────────────────────────────────────────────────────

TEST_CASE("Skin fidelity - Rf,skin from CES15 and CES18", "[sample][slice10]") {
  std::array<double, 99> rf_cesi;
  rf_cesi.fill(80.0);
  rf_cesi[14] = 90.0; // CES15 (0-indexed: 14)
  rf_cesi[17] = 70.0; // CES18 (0-indexed: 17)

  double rf_skin = compute_rf_skin(rf_cesi);
  REQUIRE_THAT(rf_skin, WithinTolerance(1e-10, 80.0)); // (90+70)/2 = 80
}

// ── Pipeline integration ────────────────────────────────────────────

TEST_CASE("Sample fidelity - D65 pipeline produces valid Rf,CESi",
          "[sample][slice10]") {
  auto cmf_2deg = load_cmf(data_path("cie_1931_2.csv"));
  auto cmf_10deg = load_cmf(data_path("cmf_1964_10.csv"));
  auto ces = load_ces(data_path("ces.csv"));
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  auto lut = load_planckian_lut(data_path("planckian_uv.csv"));

  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));
  auto result =
      compute_ces_colorimetry(wl, vals, cmf_2deg, cmf_10deg, ces, basis, lut);

  // All Rf,CESi should be in [0, 100] approximately (log rescaling may give
  // ~100.0001)
  for (int i = 0; i < 99; ++i) {
    REQUIRE(result.rf_cesi[i] >= 0.0);
    REQUIRE(result.rf_cesi[i] <= 100.001); // small overshoot from log rescaling
  }

  // Rf,skin should be in [0, 100]
  REQUIRE(result.rf_skin >= 0.0);
  REQUIRE(result.rf_skin <= 100.0);

  // D65 is a reference illuminant, so all values should be near 100
  REQUIRE_THAT(result.Rf, WithinTolerance(Tol_Rf, 100.0));
  REQUIRE_THAT(result.rf_skin, WithinTolerance(Tol_Rf, 100.0));
}

// ── Annex E ─────────────────────────────────────────────────────────

TEST_CASE("Annex E - priority level constants", "[sample][slice10]") {
  // TM-30-20 Annex E: three priority levels
  REQUIRE(AnnexE::P1 == 1); // TM-30-20 Annex E
  REQUIRE(AnnexE::P2 == 2); // TM-30-20 Annex E
  REQUIRE(AnnexE::P3 == 3); // TM-30-20 Annex E
}

} // namespace
} // namespace tm30::test
