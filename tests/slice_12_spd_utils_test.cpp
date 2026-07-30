// Tests for spd_to_xyz, spd_to_Yuv, and xyz_to_Yuv convenience functions.
//
// CIE 15:2004 §8.2.1: CIE 1976 Y,u′,v′
// TM-30-20 §3.2: Test Source Tristimulus Values

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "matchers.hpp"
#include "tm30/chromaticity.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/reference.hpp"
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

TEST_CASE("xyz_to_Yuv_batch - matches scalar calls one-for-one",
          "[xyz][Yuv][batch]") {
  std::vector<XyzTriple> xyzs{
      {100.0, 100.0, 100.0}, // equal-energy white
      {100.0, 0.0, 0.0},     // pure red
      {0.0, 100.0, 0.0},     // pure green
      {95.0, 100.0, 108.9},  // D65-like
  };

  auto batch_results = xyz_to_Yuv_batch(xyzs);
  REQUIRE(batch_results.size() == xyzs.size());

  for (std::size_t i = 0; i < xyzs.size(); ++i) {
    YuvTriple expected = xyz_to_Yuv(xyzs[i].X, xyzs[i].Y, xyzs[i].Z);
    REQUIRE_THAT(batch_results[i].Y,
                 Catch::Matchers::WithinAbs(expected.Y, 1e-14));
    REQUIRE_THAT(batch_results[i].u_prime,
                 Catch::Matchers::WithinAbs(expected.u_prime, 1e-14));
    REQUIRE_THAT(batch_results[i].v_prime,
                 Catch::Matchers::WithinAbs(expected.v_prime, 1e-14));
  }
}

TEST_CASE("xyz_to_Yuv_batch - empty input returns empty output",
          "[xyz][Yuv][batch]") {
  std::vector<XyzTriple> empty;
  auto results = xyz_to_Yuv_batch(empty);
  REQUIRE(results.empty());
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

// ─────────────────────────────────────────────────────────────────────────
// lambda_min/lambda_max - restrict to existing grid points, never resample
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("spd_to_xyz - lambda_min/lambda_max restrict to existing grid "
          "points, never resample/interpolate at the boundary",
          "[xyz][lambda]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid(); // 380..780nm, integer steps, 401 points
  std::vector<double> vals(wl.size(), 1.0);

  // Bounds that do NOT land on existing grid points (grid is integer nm).
  XyzTriple clipped_offgrid = spd_to_xyz(wl, vals, cmf, 1.0, 400.5, 500.5);

  // If clipping only *restricts* to existing points (never interpolates a
  // new sample exactly at the boundary), this must be bit-for-bit identical
  // to explicitly requesting the snapped-to integer bounds [401, 500].
  XyzTriple clipped_ongrid = spd_to_xyz(wl, vals, cmf, 1.0, 401.0, 500.0);

  REQUIRE_THAT(clipped_offgrid.X,
               Catch::Matchers::WithinAbs(clipped_ongrid.X, 1e-14));
  REQUIRE_THAT(clipped_offgrid.Y,
               Catch::Matchers::WithinAbs(clipped_ongrid.Y, 1e-14));
  REQUIRE_THAT(clipped_offgrid.Z,
               Catch::Matchers::WithinAbs(clipped_ongrid.Z, 1e-14));

  // Sanity check that clipping actually did something (didn't silently
  // fall back to the full range).
  XyzTriple full = spd_to_xyz(wl, vals, cmf, 1.0);
  REQUIRE(std::abs(clipped_ongrid.Y - full.Y) > 1e-6);
}

TEST_CASE("spd_to_xyz - lambda_min only vs lambda_max only both narrow the "
          "integration range",
          "[xyz][lambda]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals(wl.size(), 1.0);

  XyzTriple full = spd_to_xyz(wl, vals, cmf, 1.0);
  XyzTriple min_only = spd_to_xyz(wl, vals, cmf, 1.0, 700.0, std::nullopt);
  XyzTriple max_only = spd_to_xyz(wl, vals, cmf, 1.0, std::nullopt, 400.0);

  // All values are non-negative (vals=1.0, cmf ybar >= 0), so restricting
  // to a sub-range must not increase the integral relative to full range.
  REQUIRE(min_only.Y < full.Y);
  REQUIRE(max_only.Y < full.Y);
}

TEST_CASE("spd_to_Yuv - lambda_min/lambda_max forwards through to spd_to_xyz",
          "[xyz][Yuv][lambda]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  YuvTriple yuv_clipped =
      spd_to_Yuv(spd_wl, spd_vals, cmf, std::nullopt, 400.0, 700.0);
  XyzTriple xyz_clipped =
      spd_to_xyz(spd_wl, spd_vals, cmf, std::nullopt, 400.0, 700.0);
  YuvTriple expected = xyz_to_Yuv(xyz_clipped.X, xyz_clipped.Y, xyz_clipped.Z);

  REQUIRE_THAT(yuv_clipped.u_prime,
               Catch::Matchers::WithinAbs(expected.u_prime, 1e-14));
  REQUIRE_THAT(yuv_clipped.v_prime,
               Catch::Matchers::WithinAbs(expected.v_prime, 1e-14));
}

TEST_CASE("spd_to_xyz_batch - lambda_min/lambda_max clips every row "
          "identically to the scalar version",
          "[xyz][batch][lambda]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals1(wl.size(), 1.0);
  std::vector<double> vals2(wl.size(), 2.0);

  std::vector<std::vector<double>> spds{vals1, vals2};
  auto batch_results = spd_to_xyz_batch(wl, spds, cmf, 1.0, 400.0, 500.0);

  XyzTriple single1 = spd_to_xyz(wl, vals1, cmf, 1.0, 400.0, 500.0);
  XyzTriple single2 = spd_to_xyz(wl, vals2, cmf, 1.0, 400.0, 500.0);

  REQUIRE(batch_results.size() == 2);
  REQUIRE_THAT(batch_results[0].Y,
               Catch::Matchers::WithinAbs(single1.Y, 1e-12));
  REQUIRE_THAT(batch_results[1].Y,
               Catch::Matchers::WithinAbs(single2.Y, 1e-12));
}

TEST_CASE("spd_to_xyz - lambda range producing fewer than 2 points throws",
          "[xyz][lambda]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals(wl.size(), 1.0);

  // A window narrower than the 1nm grid spacing collapses to <2 points.
  REQUIRE_THROWS_AS(spd_to_xyz(wl, vals, cmf, 1.0, 500.0, 500.4),
                    std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────
// cct_to_xyz - reference illuminant XYZ at a given CCT
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("cct_to_xyz - matches manual generate_reference_spd + spd_to_xyz "
          "chain across Planckian, blend, and daylight branches",
          "[xyz][cct]") {
  auto wl = full_1nm_grid();
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  // Covers all three TM-30-20 §3.3 branches: pure Planckian (<=4000K),
  // blend (4000-5000K), pure CIE D-series (>=5000K).
  for (double cct : {2700.0, 3500.0, 4000.0, 4500.0, 5000.0, 6500.0, 9000.0}) {
    XyzTriple actual = cct_to_xyz(cct, wl, basis, cmf);

    CmfData cmf_resampled = resample_cmf(wl, cmf);
    auto ref_spd = generate_reference_spd(cct, wl, basis, cmf_resampled.y_bar);
    XyzTriple expected = spd_to_xyz(wl, ref_spd, cmf);

    REQUIRE_THAT(actual.X, Catch::Matchers::WithinAbs(expected.X, 1e-12));
    REQUIRE_THAT(actual.Y, Catch::Matchers::WithinAbs(expected.Y, 1e-12));
    REQUIRE_THAT(actual.Z, Catch::Matchers::WithinAbs(expected.Z, 1e-12));
  }
}

TEST_CASE("cct_to_xyz - Y=100 by default (auto-normalisation)", "[xyz][cct]") {
  auto wl = full_1nm_grid();
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  for (double cct : {2700.0, 5000.0, 6500.0}) {
    XyzTriple xyz = cct_to_xyz(cct, wl, basis, cmf);
    REQUIRE_THAT(xyz.Y, WithinTolerance(Tol_Xyz, 100.0));
  }
}

TEST_CASE("cct_to_xyz - K parameter behaves like spd_to_xyz's K", "[xyz][cct][K]") {
  auto wl = full_1nm_grid();
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  XyzTriple auto_norm = cct_to_xyz(6500.0, wl, basis, cmf); // K=nullopt
  XyzTriple raw = cct_to_xyz(6500.0, wl, basis, cmf, 1.0);  // K=1.0

  double k = 100.0 / raw.Y;
  REQUIRE_THAT(raw.X * k, Catch::Matchers::WithinAbs(auto_norm.X, 1e-10));
  REQUIRE_THAT(raw.Y * k, Catch::Matchers::WithinAbs(auto_norm.Y, 1e-10));
  REQUIRE_THAT(raw.Z * k, Catch::Matchers::WithinAbs(auto_norm.Z, 1e-10));
}

TEST_CASE("cct_to_xyz_batch - matches scalar calls one-for-one",
          "[xyz][cct][batch]") {
  auto wl = full_1nm_grid();
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  std::vector<double> ccts{2700.0, 4500.0, 6500.0, 9000.0};
  auto batch_results = cct_to_xyz_batch(ccts, wl, basis, cmf);
  REQUIRE(batch_results.size() == ccts.size());

  for (std::size_t i = 0; i < ccts.size(); ++i) {
    XyzTriple single = cct_to_xyz(ccts[i], wl, basis, cmf);
    REQUIRE_THAT(batch_results[i].X, Catch::Matchers::WithinAbs(single.X, 1e-12));
    REQUIRE_THAT(batch_results[i].Y, Catch::Matchers::WithinAbs(single.Y, 1e-12));
    REQUIRE_THAT(batch_results[i].Z, Catch::Matchers::WithinAbs(single.Z, 1e-12));
  }
}

TEST_CASE("cct_to_xyz - chromaticity moves bluer as CCT increases, matching "
          "the Planckian locus direction",
          "[xyz][cct]") {
  auto wl = full_1nm_grid();
  auto basis = load_daylight_basis(data_path("daylight_basis.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  XyzTriple warm = cct_to_xyz(2700.0, wl, basis, cmf);
  XyzTriple cool = cct_to_xyz(9000.0, wl, basis, cmf);

  YuvTriple warm_uv = xyz_to_Yuv(warm.X, warm.Y, warm.Z);
  YuvTriple cool_uv = xyz_to_Yuv(cool.X, cool.Y, cool.Z);

  REQUIRE(cool_uv.u_prime < warm_uv.u_prime);
}

// ─────────────────────────────────────────────────────────────────────────
// spd_to_power - radiometric (W) or photometric (lm) total power
// ─────────────────────────────────────────────────────────────────────────

TEST_CASE("spd_to_power - radiometric: flat SPD integrates to exact width",
          "[power][radiometric]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv")); // unused when photometric=false
  auto wl = full_1nm_grid(); // 380..780nm, 400nm wide
  std::vector<double> vals(wl.size(), 1.0);

  double power = spd_to_power(wl, vals, cmf, /*photometric=*/false);
  REQUIRE_THAT(power, Catch::Matchers::WithinAbs(400.0, 1e-10));
}

TEST_CASE("spd_to_power - radiometric scales linearly with SPD magnitude",
          "[power][radiometric]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals(wl.size(), 2.5);

  double power = spd_to_power(wl, vals, cmf, false);
  REQUIRE_THAT(power, Catch::Matchers::WithinAbs(1000.0, 1e-9)); // 2.5 * 400
}

TEST_CASE("spd_to_power - radiometric respects lambda_min/lambda_max",
          "[power][radiometric][lambda]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals(wl.size(), 1.0);

  double power = spd_to_power(wl, vals, cmf, false, 400.0, 500.0);
  REQUIRE_THAT(power, Catch::Matchers::WithinAbs(100.0, 1e-10));
}

TEST_CASE("spd_to_power - photometric equals spd_to_xyz(K=683.0).Y exactly",
          "[power][photometric]") {
  // spd_to_xyz's K-path computes Y_out = K * (raw integral of S*ybar)
  // exactly (src.Y auto-normalizes to 100, then Y_out = src.Y*K/src.k =
  // K * integral_st_ybar). With K=683.0 this is algebraically identical to
  // spd_to_power's photometric definition (Km * integral_st_ybar) - an
  // exact cross-check with no external oracle or golden literal needed.
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  double power = spd_to_power(spd_wl, spd_vals, cmf, /*photometric=*/true);
  XyzTriple xyz_k683 = spd_to_xyz(spd_wl, spd_vals, cmf, 683.0);

  REQUIRE_THAT(power, Catch::Matchers::WithinRel(xyz_k683.Y, 1e-9));
}

TEST_CASE("spd_to_power - photometric with lambda_min/lambda_max still "
          "matches spd_to_xyz(K=683.0, same bounds).Y",
          "[power][photometric][lambda]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  double power = spd_to_power(spd_wl, spd_vals, cmf, true, 400.0, 700.0);
  XyzTriple xyz_k683 = spd_to_xyz(spd_wl, spd_vals, cmf, 683.0, 400.0, 700.0);

  REQUIRE_THAT(power, Catch::Matchers::WithinRel(xyz_k683.Y, 1e-9));
}

TEST_CASE("spd_to_power - photometric and radiometric give different "
          "values for the same SPD",
          "[power]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));

  double radiometric = spd_to_power(spd_wl, spd_vals, cmf, false);
  double photometric = spd_to_power(spd_wl, spd_vals, cmf, true);

  REQUIRE(std::abs(radiometric - photometric) > 1.0);
}

TEST_CASE("spd_to_power_batch - matches scalar calls one-for-one, both modes",
          "[power][batch]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals1(wl.size(), 1.0);
  std::vector<double> vals2(wl.size(), 3.0);
  std::vector<std::vector<double>> spds{vals1, vals2};

  auto radiometric_batch = spd_to_power_batch(wl, spds, cmf, false);
  auto photometric_batch = spd_to_power_batch(wl, spds, cmf, true);

  REQUIRE_THAT(radiometric_batch[0],
               Catch::Matchers::WithinAbs(spd_to_power(wl, vals1, cmf, false),
                                          1e-10));
  REQUIRE_THAT(radiometric_batch[1],
               Catch::Matchers::WithinAbs(spd_to_power(wl, vals2, cmf, false),
                                          1e-10));
  REQUIRE_THAT(photometric_batch[0],
               Catch::Matchers::WithinAbs(spd_to_power(wl, vals1, cmf, true),
                                          1e-9));
  REQUIRE_THAT(photometric_batch[1],
               Catch::Matchers::WithinAbs(spd_to_power(wl, vals2, cmf, true),
                                          1e-9));
}

TEST_CASE("spd_to_power_batch - returns one scalar per SPD, not a triple",
          "[power][batch]") {
  CmfData cmf = load_cmf(data_path("cmf_1964_10.csv"));
  auto wl = full_1nm_grid();
  std::vector<double> vals(wl.size(), 1.0);
  std::vector<std::vector<double>> spds{vals, vals, vals};

  auto results = spd_to_power_batch(wl, spds, cmf, false);
  REQUIRE(results.size() == 3);
}

} // namespace
} // namespace tm30::test
