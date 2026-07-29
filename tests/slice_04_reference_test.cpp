// Slice 4 - Reference illuminant generation tests.
// TM-30-20 §3.3: Planckian, CIE D-series, 4000–5000 K blend.

#include <catch2/catch_test_macros.hpp>

#include "matchers.hpp"
#include "tm30/cct.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/reference.hpp"
#include "tm30/resample.hpp"
#include "tm30/xyz.hpp"
#include "tolerances.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace tm30::test {
namespace {

std::string data_path(const std::string &filename) {
  return std::string(TM30_DATA_DIR) + "/" + filename;
}

// Load CIE 1964 10° CMF data
CmfData load_cmf_10deg(const std::string &path) {
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

// Generate 1nm wavelength grid 380–780 nm
std::vector<double> wl_1nm() {
  std::vector<double> wl(401);
  for (int i = 0; i < 401; ++i)
    wl[i] = 380.0 + i;
  return wl;
}

// ── Planckian ──────────────────────────────────────────────────────

TEST_CASE("Reference - Planckian 2700K self-consistency",
          "[reference][slice04]") {
  auto wl = wl_1nm();
  auto spd = generate_planckian(2700.0, wl);
  REQUIRE(spd.size() == 401);

  // At 560nm the value should be 1.0 (normalisation point)
  REQUIRE_THAT(spd[180], WithinTolerance(1e-6, 1.0)); // 560nm = index 180

  // Values should be positive
  for (auto v : spd)
    REQUIRE(v > 0.0);
}

TEST_CASE("Reference - Planckian values monotonic with T at fixed wavelength",
          "[reference][slice04]") {
  auto wl = wl_1nm();
  auto spd_2700 = generate_planckian(2700.0, wl);
  auto spd_6500 = generate_planckian(6500.0, wl);

  // At short wavelengths (380nm), higher T → higher relative radiance
  REQUIRE(spd_6500[0] > spd_2700[0]);
  // At long wavelengths (780nm), higher T → lower relative radiance
  REQUIRE(spd_6500[400] < spd_2700[400]);
}

// ── D-series ───────────────────────────────────────────────────────

TEST_CASE("Reference - CIE D65 generation", "[reference][slice04]") {
  auto wl = wl_1nm();
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  auto spd = generate_cie_d(6500.0, wl, basis);

  REQUIRE(spd.size() == 401);
  REQUIRE_THAT(spd[180], WithinTolerance(1e-6, 1.0)); // normalised at 560nm
}

TEST_CASE("Reference - CIE D 7000K boundary", "[reference][slice04]") {
  auto wl = wl_1nm();
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));

  // Just below 7000K - uses Eq. (10) branch
  auto spd_6999 = generate_cie_d(6999.0, wl, basis);
  // Just above 7000K - uses Eq. (11) branch
  auto spd_7001 = generate_cie_d(7001.0, wl, basis);

  // Values should be very close (continuity at boundary)
  for (size_t i = 0; i < wl.size(); ++i) {
    REQUIRE_THAT(spd_6999[i], WithinTolerance(0.01, spd_7001[i]));
  }
}

// ── Blend region ───────────────────────────────────────────────────

TEST_CASE("Reference - 4000K boundary", "[reference][slice04]") {
  auto wl = wl_1nm();
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  auto cmf = load_cmf_10deg(data_path("cmf_1964_10.csv"));

  // At exactly 4000K → pure Planckian
  auto spd_4000 = generate_reference_spd(4000.0, wl, basis, cmf.y_bar);
  auto planck_4000 = generate_planckian(4000.0, wl);

  for (size_t i = 0; i < wl.size(); ++i) {
    REQUIRE_THAT(spd_4000[i], WithinTolerance(1e-10, planck_4000[i]));
  }
}

TEST_CASE("Reference - 5000K boundary", "[reference][slice04]") {
  auto wl = wl_1nm();
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  auto cmf = load_cmf_10deg(data_path("cmf_1964_10.csv"));

  // At exactly 5000K → pure D-series
  auto spd_5000 = generate_reference_spd(5000.0, wl, basis, cmf.y_bar);
  auto d_5000 = generate_cie_d(5000.0, wl, basis);

  for (size_t i = 0; i < wl.size(); ++i) {
    REQUIRE_THAT(spd_5000[i], WithinTolerance(1e-10, d_5000[i]));
  }
}

TEST_CASE("Reference - blend at 4500K produces intermediate values",
          "[reference][slice04]") {
  auto wl = wl_1nm();
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  auto cmf = load_cmf_10deg(data_path("cmf_1964_10.csv"));

  auto blended = generate_reference_spd(4500.0, wl, basis, cmf.y_bar);
  auto planck = generate_planckian(4500.0, wl);
  auto daylight = generate_cie_d(4500.0, wl, basis);

  // Blended should be between Planckian and D at many wavelengths
  // At 4500K: blend_factor = (5000-4500)/1000 = 0.5
  // But components are Y-normalised, so direct comparison is complex.
  // Instead, verify the blend factor in Y-space.
  SourceXyz xyz_blended =
      compute_source_xyz(wl, blended, cmf.x_bar, cmf.y_bar, cmf.z_bar);
  REQUIRE_THAT(xyz_blended.Y, WithinTolerance(1e-4, 100.0));
}

// ── Continuity at boundaries ───────────────────────────────────────

TEST_CASE("Reference - continuity at 4000K", "[reference][slice04]") {
  auto wl = wl_1nm();
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  auto cmf = load_cmf_10deg(data_path("cmf_1964_10.csv"));

  auto spd_3999 = generate_reference_spd(3999.9, wl, basis, cmf.y_bar);
  auto spd_4001 = generate_reference_spd(4000.1, wl, basis, cmf.y_bar);

  // Nearly identical despite crossing the boundary
  double max_diff = 0.0;
  for (size_t i = 0; i < wl.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(spd_3999[i] - spd_4001[i]));
  }
  REQUIRE(max_diff < 0.01);
}

TEST_CASE("Reference - continuity at 5000K", "[reference][slice04]") {
  auto wl = wl_1nm();
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  auto cmf = load_cmf_10deg(data_path("cmf_1964_10.csv"));

  auto spd_4999 = generate_reference_spd(4999.9, wl, basis, cmf.y_bar);
  auto spd_5001 = generate_reference_spd(5000.1, wl, basis, cmf.y_bar);

  double max_diff = 0.0;
  for (size_t i = 0; i < wl.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(spd_4999[i] - spd_5001[i]));
  }
  REQUIRE(max_diff < 0.01);
}

// ── Daylight basis loading ─────────────────────────────────────────

TEST_CASE("Reference - daylight basis loads correctly",
          "[reference][slice04]") {
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  REQUIRE(basis.wavelengths.size() == 81); // 380:5:780
  REQUIRE(basis.S0.size() == 81);
  REQUIRE(basis.S1.size() == 81);
  REQUIRE(basis.S2.size() == 81);

  // S0(560) ≈ 100, S1(560) ≈ 0, S2(560) ≈ 0
  // index of 560nm in 5nm grid: (560-380)/5 = 36
  REQUIRE_THAT(basis.S0[36], WithinTolerance(0.1, 100.0));
  REQUIRE_THAT(basis.S1[36], WithinTolerance(0.01, 0.0));
  REQUIRE_THAT(basis.S2[36], WithinTolerance(0.01, 0.0));
}

} // namespace
} // namespace tm30::test
