// Tests for spd_to_xyz, spd_to_Yuv, and xyz_to_Yuv convenience functions.
//
// CIE 15:2004 §8.2.1: CIE 1976 Y,u′,v′
// TM-30-20 §3.2: Test Source Tristimulus Values

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "matchers.hpp"
#include "tm30/chromaticity.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/resample.hpp"
#include "tm30/spd.hpp"
#include "tm30/xyz.hpp"
#include "tolerances.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace tm30::test {
namespace {

// ── Test helpers ─────────────────────────────────────────────────────────

std::string data_path(const std::string &filename) {
  return std::string(TM30_DATA_DIR) + "/" + filename;
}

/// Load a simple two-column SPD CSV (wavelength, value).
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

/// Build a 401-point wavelength grid from 380 to 780 nm at 1 nm step.
std::vector<double> full_1nm_grid() {
  std::vector<double> wl(401);
  for (int i = 0; i < 401; ++i) {
    wl[i] = 380.0 + static_cast<double>(i);
  }
  return wl;
}

// ─────────────────────────────────────────────────────────────────────────
// xyz_to_Yuv - CIE 1976 Y,u′,v′
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("xyz_to_Yuv - known XYZ values", "[xyz][Yuv]") {
  // CIE 15:2004 §8.2.1: u′ = 4X/(X+15Y+3Z), v′ = 9Y/(X+15Y+3Z)

  // Equal-energy white: X=Y=Z=100
  YuvTriple eew = xyz_to_Yuv(100.0, 100.0, 100.0);
  // denom = 100 + 1500 + 300 = 1900
  // u′ = 400/1900 = 0.21052631578947367
  // v′ = 900/1900 = 0.47368421052631576
  REQUIRE_THAT(eew.Y, Catch::Matchers::WithinAbs(100.0, 1e-12));
  REQUIRE_THAT(eew.u_prime, Catch::Matchers::WithinAbs(400.0 / 1900.0, 1e-15));
  REQUIRE_THAT(eew.v_prime, Catch::Matchers::WithinAbs(900.0 / 1900.0, 1e-15));

  // Pure red (monochromatic): X big, Y=Z=0
  // denom = X + 0 + 0 = X
  YuvTriple red = xyz_to_Yuv(100.0, 0.0, 0.0);
  REQUIRE_THAT(red.u_prime, Catch::Matchers::WithinAbs(4.0, 1e-12)); // 4X/X = 4
  REQUIRE_THAT(red.v_prime, Catch::Matchers::WithinAbs(0.0, 1e-12)); // 0/X = 0

  // Pure green: Y big, X=Z=0
  // denom = 0 + 15Y + 0 = 15Y
  YuvTriple green = xyz_to_Yuv(0.0, 100.0, 0.0);
  REQUIRE_THAT(green.u_prime,
               Catch::Matchers::WithinAbs(0.0, 1e-12)); // 0/15Y = 0
  REQUIRE_THAT(green.v_prime,
               Catch::Matchers::WithinAbs(9.0 / 15.0, 1e-12)); // 9Y/15Y = 0.6
}

TEST_CASE("xyz_to_Yuv - relationship to CIE 1960 uv", "[xyz][Yuv]") {
  // CIE 1976 u′ equals CIE 1960 u (same numerator/denominator for u).
  // CIE 1976 v′ = 1.5 × CIE 1960 v.

  UvCoord uv = xyz_to_uv(95.0, 100.0, 108.9);
  YuvTriple yuv = xyz_to_Yuv(95.0, 100.0, 108.9);

  // u′ == u
  REQUIRE_THAT(yuv.u_prime, Catch::Matchers::WithinAbs(uv.u, 1e-15));

  // v′ == 1.5 × v
  REQUIRE_THAT(yuv.v_prime, Catch::Matchers::WithinAbs(uv.v * 1.5, 1e-15));
}

// ─────────────────────────────────────────────────────────────────────────
// spd_to_xyz - convenience wrapper
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("spd_to_xyz - D65 agrees with compute_source_xyz",
          "[xyz][spd_to_xyz]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  // Via convenience function (auto K)
  XyzTriple xyz = spd_to_xyz(spd_wl, spd_vals, cmf);

  // Via low-level (must match)
  CmfData cmf_resampled = resample_cmf(spd_wl, cmf);
  SourceXyz src = compute_source_xyz(spd_wl, spd_vals, cmf_resampled.x_bar,
                                     cmf_resampled.y_bar, cmf_resampled.z_bar);

  REQUIRE_THAT(xyz.X, Catch::Matchers::WithinAbs(src.X, 1e-14));
  REQUIRE_THAT(xyz.Y, Catch::Matchers::WithinAbs(src.Y, 1e-14));
  REQUIRE_THAT(xyz.Z, Catch::Matchers::WithinAbs(src.Z, 1e-14));

  // Y must be 100 - TM-30-20 §3.2 Eq. (2)
  REQUIRE_THAT(xyz.Y, WithinTolerance(Tol_Xyz, 100.0));
}

TEST_CASE("spd_to_xyz - Illuminant A", "[xyz][spd_to_xyz]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("illuminant_a_1nm.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  XyzTriple xyz = spd_to_xyz(spd_wl, spd_vals, cmf);

  // Y = 100 - TM-30-20 §3.2 Eq. (2)
  REQUIRE_THAT(xyz.Y, WithinTolerance(Tol_Xyz, 100.0));

  // Golden values from compute_source_xyz (verified in slice_02_xyz_test)
  REQUIRE_THAT(xyz.X, WithinTolerance(Tol_Xyz, 111.1432899325));
  REQUIRE_THAT(xyz.Z, WithinTolerance(Tol_Xyz, 35.1999196709));
}

TEST_CASE("spd_to_xyz - scale invariance", "[xyz][spd_to_xyz]") {
  // Doubling the SPD should give identical XYZ (normalisation absorbs scale).
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals1(wl.size(), 1.0);
  std::vector<double> vals2(wl.size(), 2.0);

  XyzTriple xyz1 = spd_to_xyz(wl, vals1, cmf);
  XyzTriple xyz2 = spd_to_xyz(wl, vals2, cmf);

  REQUIRE_THAT(xyz1.X, WithinTolerance(Tol_Xyz, xyz2.X));
  REQUIRE_THAT(xyz1.Y, WithinTolerance(Tol_Xyz, xyz2.Y));
  REQUIRE_THAT(xyz1.Z, WithinTolerance(Tol_Xyz, xyz2.Z));
}

// ─────────────────────────────────────────────────────────────────────────
// spd_to_xyz_batch - multiple SPDs
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("spd_to_xyz_batch - single SPD matches scalar", "[xyz][batch]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  XyzTriple single = spd_to_xyz(spd_wl, spd_vals, cmf);

  std::vector<std::vector<double>> batch{spd_vals};
  auto results = spd_to_xyz_batch(spd_wl, batch, cmf);

  REQUIRE(results.size() == 1);
  REQUIRE_THAT(results[0].X, Catch::Matchers::WithinAbs(single.X, 1e-14));
  REQUIRE_THAT(results[0].Y, Catch::Matchers::WithinAbs(single.Y, 1e-14));
  REQUIRE_THAT(results[0].Z, Catch::Matchers::WithinAbs(single.Z, 1e-14));
}

TEST_CASE("spd_to_xyz_batch - multiple SPDs", "[xyz][batch]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals1(wl.size(), 1.0);
  std::vector<double> vals2(wl.size(), 2.0);

  std::vector<std::vector<double>> spds{vals1, vals2};
  auto results = spd_to_xyz_batch(wl, spds, cmf);

  REQUIRE(results.size() == 2);

  // Both should have Y=100 due to normalisation
  REQUIRE_THAT(results[0].Y, WithinTolerance(Tol_Xyz, 100.0));
  REQUIRE_THAT(results[1].Y, WithinTolerance(Tol_Xyz, 100.0));

  // Scale invariance: vals1 and vals2 should give identical XYZ
  REQUIRE_THAT(results[0].X, WithinTolerance(Tol_Xyz, results[1].X));
  REQUIRE_THAT(results[0].Z, WithinTolerance(Tol_Xyz, results[1].Z));
}

// ─────────────────────────────────────────────────────────────────────────
// spd_to_Yuv - convenience wrapper (chains spd_to_xyz → xyz_to_Yuv)
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("spd_to_Yuv - D65", "[xyz][Yuv]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  YuvTriple yuv = spd_to_Yuv(spd_wl, spd_vals, cmf);

  // Y = 100 (from XYZ normalisation)
  REQUIRE_THAT(yuv.Y, WithinTolerance(Tol_Xyz, 100.0));

  // Verify by manual chain
  XyzTriple xyz = spd_to_xyz(spd_wl, spd_vals, cmf);
  YuvTriple expected = xyz_to_Yuv(xyz.X, xyz.Y, xyz.Z);
  REQUIRE_THAT(yuv.u_prime,
               Catch::Matchers::WithinAbs(expected.u_prime, 1e-14));
  REQUIRE_THAT(yuv.v_prime,
               Catch::Matchers::WithinAbs(expected.v_prime, 1e-14));
}

TEST_CASE("spd_to_Yuv - Illuminant A", "[xyz][Yuv]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("illuminant_a_1nm.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  YuvTriple yuv = spd_to_Yuv(spd_wl, spd_vals, cmf);

  // Y = 100
  REQUIRE_THAT(yuv.Y, WithinTolerance(Tol_Xyz, 100.0));

  // Self-consistency: chain spd_to_xyz → xyz_to_Yuv
  XyzTriple xyz = spd_to_xyz(spd_wl, spd_vals, cmf);
  YuvTriple expected = xyz_to_Yuv(xyz.X, xyz.Y, xyz.Z);
  REQUIRE_THAT(yuv.u_prime,
               Catch::Matchers::WithinAbs(expected.u_prime, 1e-14));
  REQUIRE_THAT(yuv.v_prime,
               Catch::Matchers::WithinAbs(expected.v_prime, 1e-14));
}

// ─────────────────────────────────────────────────────────────────────────
// spd_to_Yuv_batch
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("spd_to_Yuv_batch - matches scalar chain", "[xyz][Yuv][batch]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  YuvTriple single = spd_to_Yuv(spd_wl, spd_vals, cmf);

  std::vector<std::vector<double>> batch{spd_vals};
  auto results = spd_to_Yuv_batch(spd_wl, batch, cmf);

  REQUIRE(results.size() == 1);
  REQUIRE_THAT(results[0].Y, Catch::Matchers::WithinAbs(single.Y, 1e-14));
  REQUIRE_THAT(results[0].u_prime,
               Catch::Matchers::WithinAbs(single.u_prime, 1e-14));
  REQUIRE_THAT(results[0].v_prime,
               Catch::Matchers::WithinAbs(single.v_prime, 1e-14));
}

TEST_CASE("spd_to_Yuv_batch - two different SPDs", "[xyz][Yuv][batch]") {
  auto [d65_wl, d65_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  auto [a_wl, a_vals] = load_spd_csv(data_path("illuminant_a_1nm.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  std::vector<std::vector<double>> spds{d65_vals, a_vals};
  auto results = spd_to_Yuv_batch(d65_wl, spds, cmf);

  REQUIRE(results.size() == 2);

  // Both should have Y=100
  for (const auto &r : results) {
    REQUIRE_THAT(r.Y, WithinTolerance(Tol_Xyz, 100.0));
  }

  // D65 and Illuminant A should differ in chromaticity
  REQUIRE(std::abs(results[0].u_prime - results[1].u_prime) > 1e-4);
  REQUIRE(std::abs(results[0].v_prime - results[1].v_prime) > 1e-4);
}

// ─────────────────────────────────────────────────────────────────────────
// K parameter - user-specified normalisation constant
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("spd_to_xyz - K=nullopt returns Y=100", "[xyz][K]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals(wl.size(), 1.0); // flat SPD

  XyzTriple xyz = spd_to_xyz(wl, vals, cmf); // K = std::nullopt
  // TM-30-20 §3.2 Eq. (2): Y = 100
  REQUIRE_THAT(xyz.Y, WithinTolerance(Tol_Xyz, 100.0));
}

TEST_CASE("spd_to_xyz - K=1.0 returns raw integrals", "[xyz][K]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals(wl.size(), 1.0); // flat SPD

  XyzTriple rel = spd_to_xyz(wl, vals, cmf);      // K = nullopt → Y=100
  XyzTriple raw = spd_to_xyz(wl, vals, cmf, 1.0); // K = 1.0 → raw

  // raw * k = rel (where k = 100 / raw.Y)
  double k = 100.0 / raw.Y;
  REQUIRE_THAT(raw.X * k, Catch::Matchers::WithinAbs(rel.X, 1e-12));
  REQUIRE_THAT(raw.Y * k, Catch::Matchers::WithinAbs(rel.Y, 1e-12));
  REQUIRE_THAT(raw.Z * k, Catch::Matchers::WithinAbs(rel.Z, 1e-12));
}

TEST_CASE("spd_to_xyz - scale with SPD magnitude when K=1.0", "[xyz][K]") {
  // K=1.0: doubling the SPD doubles XYZ (raw integrals).
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals1(wl.size(), 1.0);
  std::vector<double> vals2(wl.size(), 2.0);

  XyzTriple raw1 = spd_to_xyz(wl, vals1, cmf, 1.0);
  XyzTriple raw2 = spd_to_xyz(wl, vals2, cmf, 1.0);

  // Raw XYZ should double
  REQUIRE_THAT(raw2.X, Catch::Matchers::WithinRel(2.0 * raw1.X, 1e-12));
  REQUIRE_THAT(raw2.Y, Catch::Matchers::WithinRel(2.0 * raw1.Y, 1e-12));
  REQUIRE_THAT(raw2.Z, Catch::Matchers::WithinRel(2.0 * raw1.Z, 1e-12));
}

TEST_CASE("spd_to_xyz - custom K multiplier", "[xyz][K]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals(wl.size(), 1.0);

  XyzTriple raw = spd_to_xyz(wl, vals, cmf, 1.0);    // K = 1.0
  XyzTriple k50 = spd_to_xyz(wl, vals, cmf, 50.0);   // K = 50.0
  XyzTriple k683 = spd_to_xyz(wl, vals, cmf, 683.0); // K = 683.0

  REQUIRE_THAT(k50.X, Catch::Matchers::WithinRel(50.0 * raw.X, 1e-12));
  REQUIRE_THAT(k50.Y, Catch::Matchers::WithinRel(50.0 * raw.Y, 1e-12));
  REQUIRE_THAT(k683.X, Catch::Matchers::WithinRel(683.0 * raw.X, 1e-12));
  REQUIRE_THAT(k683.Y, Catch::Matchers::WithinRel(683.0 * raw.Y, 1e-12));
}

TEST_CASE("spd_to_Yuv - K parameter passes through", "[xyz][Yuv][K]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  YuvTriple yuv_auto = spd_to_Yuv(spd_wl, spd_vals, cmf);     // K = nullopt
  YuvTriple yuv_raw = spd_to_Yuv(spd_wl, spd_vals, cmf, 1.0); // K = 1.0

  // Auto: Y = 100
  REQUIRE_THAT(yuv_auto.Y, WithinTolerance(Tol_Xyz, 100.0));

  // Raw: Y ≠ 100, but u',v' identical (Y cancels in ratio)
  REQUIRE(yuv_raw.Y != 100.0);
  REQUIRE_THAT(yuv_raw.u_prime,
               Catch::Matchers::WithinAbs(yuv_auto.u_prime, 1e-14));
  REQUIRE_THAT(yuv_raw.v_prime,
               Catch::Matchers::WithinAbs(yuv_auto.v_prime, 1e-14));
}

} // namespace
} // namespace tm30::test
