// Slice 3 - CCT and Duv via Ohno 2014 method (CIE 1931 2-deg observer).
//
// TM-30-20 §3.1: CCT uses CIE 1931 2-deg observer (exception).
// TM-30-20 §3.3: CCT determination - Ohno 2013 method.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "matchers.hpp"
#include "tm30/cct.hpp"
#include "tm30/chromaticity.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/resample.hpp"
#include "tm30/spd.hpp"
#include "tm30/xyz.hpp"
#include "tolerances.hpp"

#include <cmath>
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

/// Load CIE 1931 2-deg CMF data from CSV (wavelength, x_bar, y_bar, z_bar).
CmfData load_cmf_2deg(const std::string &path) {
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

/// Compute 2-deg XYZ for a source SPD using CIE 1931 2-deg CMFs.
/// Returns normalized XYZ (Y=100) since chromaticity is scale-invariant.
XyzTriple compute_xyz_2deg(const std::vector<double> &spd_wl,
                           const std::vector<double> &spd_vals,
                           const CmfData &cmf_2deg) {
  // Resample CMFs to the SPD wavelength grid.
  // The SPD may be at 1nm or 5nm; CMFs are at 1nm.
  // TM-30-20 §3.5: CMFs should be interpolated to the test source increment.
  CmfData cmf_resampled = resample_cmf(spd_wl, cmf_2deg);
  SourceXyz src = compute_source_xyz(spd_wl, spd_vals, cmf_resampled.x_bar,
                                     cmf_resampled.y_bar, cmf_resampled.z_bar);
  return XyzTriple{src.X, src.Y, src.Z};
}

// -------------------------------------------------------------------------
// (u,v) chromaticity from XYZ
// -------------------------------------------------------------------------

TEST_CASE("Chromaticity - xyz_to_uv for known illuminants",
          "[chromaticity][slice03]") {
  // CIE 1960 UCS transformation: u = 4X/(X+15Y+3Z), v = 6Y/(X+15Y+3Z)

  // Illuminant E (equal energy): X=Y=Z -> u=4/19~=0.2105, v=6/19~=0.3158
  {
    UvCoord uv = xyz_to_uv(100.0, 100.0, 100.0);
    REQUIRE_THAT(uv.u, Catch::Matchers::WithinAbs(0.21052631578947367, 1e-12));
    REQUIRE_THAT(uv.v, Catch::Matchers::WithinAbs(0.3157894736842105, 1e-12));
  }

  // D65 10-deg: X~=94.81, Y=100, Z~=107.30
  // u ~= 4*94.81 / (94.81 + 1500 + 321.9) = 379.24 / 1916.71 ~= 0.19786
  {
    UvCoord uv = xyz_to_uv(94.81, 100.0, 107.30);
    // u = 4*94.81 / (94.81 + 1500 + 321.9) = 379.24 / 1916.71
    // v = 6*100 / 1916.71 = 600 / 1916.71
    REQUIRE_THAT(uv.u, Catch::Matchers::WithinAbs(379.24 / 1916.71, 1e-10));
    REQUIRE_THAT(uv.v, Catch::Matchers::WithinAbs(600.0 / 1916.71, 1e-10));
  }

  // Scale invariance: doubling XYZ should give same (u,v)
  {
    UvCoord uv1 = xyz_to_uv(50.0, 50.0, 50.0);
    UvCoord uv2 = xyz_to_uv(100.0, 100.0, 100.0);
    REQUIRE_THAT(uv1.u, Catch::Matchers::WithinAbs(uv2.u, 1e-15));
    REQUIRE_THAT(uv1.v, Catch::Matchers::WithinAbs(uv2.v, 1e-15));
  }
}

// -------------------------------------------------------------------------
// CCT - LUT loading
// -------------------------------------------------------------------------

TEST_CASE("CCT - Planckian LUT loads correctly", "[cct][slice03]") {
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  REQUIRE(lut.T.size() >= 1000); // should have many points
  REQUIRE(lut.T.size() == lut.u.size());
  REQUIRE(lut.T.size() == lut.v.size());

  // Temperatures should be monotonically increasing
  for (std::size_t i = 1; i < lut.T.size(); ++i) {
    REQUIRE(lut.T[i] > lut.T[i - 1]);
  }

  // LUT should cover the TM-30 range (1000 K to 41000 K)
  REQUIRE(lut.T.front() <= 1001.0);
  REQUIRE(lut.T.back() >= 40000.0);

  // Check a known point: near 2856 K
  // u should be around 0.256, v around 0.349
  std::size_t idx_2856 = 0;
  double min_diff = 1e9;
  for (std::size_t i = 0; i < lut.T.size(); ++i) {
    double d = std::abs(lut.T[i] - 2856.0);
    if (d < min_diff) {
      min_diff = d;
      idx_2856 = i;
    }
  }
  REQUIRE_THAT(lut.u[idx_2856], Catch::Matchers::WithinAbs(0.256, 0.02));
  REQUIRE_THAT(lut.v[idx_2856], Catch::Matchers::WithinAbs(0.349, 0.02));
}

TEST_CASE("CCT - LUT loading fails on missing file", "[cct][slice03]") {
  REQUIRE_THROWS_AS(load_planckian_lut(data_path("nonexistent_file.csv")),
                    std::runtime_error);
}

// -------------------------------------------------------------------------
// CCT - D65 and Illuminant A (2-deg observer)
// -------------------------------------------------------------------------

TEST_CASE("CCT - D65 (2-deg observer)", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult result = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);

  // TM-30-20 §3.3: D65 should be near 6500 K
  // tools/oracle_recompute_12.py: CCT=6501.8485, Duv=0.00321508
  REQUIRE_THAT(result.cct, WithinTolerance(Tol_Cct, 6501.8485));
  REQUIRE_THAT(result.duv, WithinTolerance(Tol_Duv, 0.00321508));
}

TEST_CASE("CCT - Illuminant A (2-deg observer)", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("illuminant_a_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult result = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);

  // TM-30-20 §3.3: Illuminant A should be near 2856 K
  // tools/oracle_recompute_12.py: CCT=2855.5796, Duv=0.00000332
  REQUIRE_THAT(result.cct, WithinTolerance(Tol_Cct, 2855.5796));
  REQUIRE_THAT(result.duv, WithinTolerance(Tol_Duv, 0.00000332));
}

// -------------------------------------------------------------------------
// CCT - FL1-FL12 (full CIE F-series illuminants)
// -------------------------------------------------------------------------

TEST_CASE("CCT - FL1", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl1_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=6425.4015, Duv=0.00719197
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 6425.4015));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00719197));
}

TEST_CASE("CCT - FL2", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl2_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=4225.1675, Duv=0.00186286
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 4225.1675));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00186286));
}

TEST_CASE("CCT - FL3", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl3_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=3447.3487, Duv=0.00074405
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 3447.3487));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00074405));
}

TEST_CASE("CCT - FL4", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl4_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=2939.5890, Duv=-0.00074010
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 2939.5890));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, -0.00074010));
}

TEST_CASE("CCT - FL5", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl5_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=6342.7591, Duv=0.01080438
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 6342.7591));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.01080438));
}

TEST_CASE("CCT - FL6", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl6_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=4148.9892, Duv=0.00610277
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 4148.9892));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00610277));
}

TEST_CASE("CCT - FL7", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl7_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=6490.0166, Duv=0.00326512
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 6490.0166));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00326512));
}

TEST_CASE("CCT - FL8", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl8_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=4994.7943, Duv=0.00324352
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 4994.7943));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00324352));
}

TEST_CASE("CCT - FL9", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl9_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=4148.0210, Duv=0.00003905
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 4148.0210));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00003905));
}

TEST_CASE("CCT - FL10", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl10_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=4998.7696, Duv=0.00338355
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 4998.7696));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00338355));
}

TEST_CASE("CCT - FL11", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl11_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=4000.7527, Duv=0.00015497
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 4000.7527));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00015497));
}

TEST_CASE("CCT - FL12", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl12_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=3002.5891, Duv=0.00013385
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 3002.5891));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00013385));
}

// -------------------------------------------------------------------------
// CCT - HP1-HP5 (narrowband sources that stress the solver)
// -------------------------------------------------------------------------

TEST_CASE("CCT - HP1 (high-pressure sodium, narrowband)", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("hp1_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=1959.2574, Duv=0.00078265
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 1959.2574));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00078265));
}

TEST_CASE("CCT - HP2", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("hp2_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=2506.3494, Duv=0.00071410
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 2506.3494));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00071410));
}

TEST_CASE("CCT - HP3", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("hp3_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=3144.0166, Duv=0.00237235
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 3144.0166));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00237235));
}

TEST_CASE("CCT - HP4", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("hp4_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=4002.1404, Duv=0.00117838
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 4002.1404));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00117838));
}

TEST_CASE("CCT - HP5", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("hp5_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  XyzTriple xyz = compute_xyz_2deg(spd_wl, spd_vals, cmf);
  CctDuvResult r = compute_cct_duv_from_xyz(xyz.X, xyz.Y, xyz.Z, lut);
  // tools/oracle_recompute_12.py: CCT=4039.4218, Duv=-0.00170817
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 4039.4218));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, -0.00170817));
}

// -------------------------------------------------------------------------
// CCT - Self-consistency (Planckian sources should recover their own CCT)
// -------------------------------------------------------------------------

TEST_CASE("CCT - self-consistency: Planckian at 3000 K", "[cct][slice03]") {
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  // For a pure Planckian radiator, the (u,v) is exactly on the locus.
  // Interpolate LUT (u,v) at 3000 K.
  // The CCT should recover close to 3000 K with Duv ~= 0.
  // Find the exact (u,v) from the LUT for T near 3000.
  // Since the test point is exactly on the LUT curve, the solver
  // should recover the temperature precisely.

  // Use the LUT point closest to 3000K as the test point.
  std::size_t idx3000 = 0;
  double best = 1e9;
  for (std::size_t i = 0; i < lut.T.size(); ++i) {
    double d = std::abs(lut.T[i] - 3000.0);
    if (d < best) {
      best = d;
      idx3000 = i;
    }
  }

  double u_bb = lut.u[idx3000];
  double v_bb = lut.v[idx3000];
  double T_expected = lut.T[idx3000];

  // Compute CCT from the exact Planckian locus point
  CctDuvResult r = compute_cct_duv(u_bb, v_bb, lut);

  // CCT should match the LUT temperature closely
  // (within one LUT step of about 7.5K at 3000K with 0.25% spacing)
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, T_expected));

  // Duv should be near zero since we're exactly on the locus
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.0));
}

// -------------------------------------------------------------------------
// CCT - Boundary temperature regions
// -------------------------------------------------------------------------

/// Helper: test CCT at a specific Planckian (u,v) from the LUT.
void test_planckian_self(double T_target, const PlanckianLut &lut) {
  // Find closest LUT point
  std::size_t idx = 0;
  double best = 1e9;
  for (std::size_t i = 0; i < lut.T.size(); ++i) {
    double d = std::abs(lut.T[i] - T_target);
    if (d < best) {
      best = d;
      idx = i;
    }
  }
  double u_bb = lut.u[idx];
  double v_bb = lut.v[idx];
  double T_expected = lut.T[idx];

  CctDuvResult r = compute_cct_duv(u_bb, v_bb, lut);
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, T_expected));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.0));
}

TEST_CASE("CCT - boundary: 3999.9 K region", "[cct][slice03]") {
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  test_planckian_self(3999.9, lut);
}

TEST_CASE("CCT - boundary: 4000.1 K region", "[cct][slice03]") {
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  test_planckian_self(4000.1, lut);
}

TEST_CASE("CCT - boundary: 4999.9 K region", "[cct][slice03]") {
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  test_planckian_self(4999.9, lut);
}

TEST_CASE("CCT - boundary: 5000.1 K region", "[cct][slice03]") {
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));
  test_planckian_self(5000.1, lut);
}

// -------------------------------------------------------------------------
// CCT - Duv sign convention
// -------------------------------------------------------------------------

TEST_CASE("CCT - Duv is signed distance", "[cct][slice03]") {
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  // Get a point on the locus near 4000K
  std::size_t idx = 0;
  double best = 1e9;
  for (std::size_t i = 0; i < lut.T.size(); ++i) {
    double d = std::abs(lut.T[i] - 4000.0);
    if (d < best) {
      best = d;
      idx = i;
    }
  }

  double u0 = lut.u[idx];
  double v0 = lut.v[idx];

  // Shift above the locus (increase v)
  {
    CctDuvResult r = compute_cct_duv(u0, v0 + 0.001, lut);
    REQUIRE(r.duv > 0.0); // positive Duv
  }

  // Shift below the locus (decrease v)
  {
    CctDuvResult r = compute_cct_duv(u0, v0 - 0.001, lut);
    REQUIRE(r.duv < 0.0); // negative Duv
  }
}

// -------------------------------------------------------------------------
// spd_to_cct / spd_to_cct_batch - SPD -> CCT/Duv convenience wrappers
// -------------------------------------------------------------------------

TEST_CASE("CCT - spd_to_cct matches the manual resample+compute chain (D65)",
          "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  CctDuvResult r = spd_to_cct(spd_wl, spd_vals, cmf, lut);

  // Same value as the "CCT - D65 (2-deg observer)" test above
  // (tools/oracle_recompute_12.py): CCT=6501.8485, Duv=0.00321508
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 6501.8485));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00321508));
}

TEST_CASE("CCT - spd_to_cct matches the manual chain (FL1)", "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("fl1_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  CctDuvResult r = spd_to_cct(spd_wl, spd_vals, cmf, lut);

  // tools/oracle_recompute_12.py: CCT=6425.4015, Duv=0.00719197
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 6425.4015));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00719197));
}

TEST_CASE("CCT - spd_to_cct matches the manual chain (HP1, narrowband)",
          "[cct][slice03]") {
  auto [spd_wl, spd_vals] = load_spd_csv(data_path("hp1_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  CctDuvResult r = spd_to_cct(spd_wl, spd_vals, cmf, lut);

  // tools/oracle_recompute_12.py: CCT=1959.2574, Duv=0.00078265
  REQUIRE_THAT(r.cct, WithinTolerance(Tol_Cct, 1959.2574));
  REQUIRE_THAT(r.duv, WithinTolerance(Tol_Duv, 0.00078265));
}

TEST_CASE("CCT - spd_to_cct_batch matches per-SPD spd_to_cct bit-for-bit",
          "[cct][slice03]") {
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  auto [wl_d65, vals_d65] = load_spd_csv(data_path("d65_1nm.csv"));
  auto [wl_a, vals_a] = load_spd_csv(data_path("illuminant_a_1nm.csv"));
  auto [wl_fl1, vals_fl1] = load_spd_csv(data_path("fl1_1nm.csv"));

  // All three SPDs share the same 1 nm, 380-780 nm grid.
  REQUIRE(wl_d65 == wl_a);
  REQUIRE(wl_d65 == wl_fl1);

  std::vector<std::vector<double>> batch = {vals_d65, vals_a, vals_fl1};
  std::vector<CctDuvResult> batch_results =
      spd_to_cct_batch(wl_d65, batch, cmf, lut);

  REQUIRE(batch_results.size() == 3);

  CctDuvResult single_d65 = spd_to_cct(wl_d65, vals_d65, cmf, lut);
  CctDuvResult single_a = spd_to_cct(wl_a, vals_a, cmf, lut);
  CctDuvResult single_fl1 = spd_to_cct(wl_fl1, vals_fl1, cmf, lut);

  REQUIRE(batch_results[0].cct == single_d65.cct);
  REQUIRE(batch_results[0].duv == single_d65.duv);
  REQUIRE(batch_results[1].cct == single_a.cct);
  REQUIRE(batch_results[1].duv == single_a.duv);
  REQUIRE(batch_results[2].cct == single_fl1.cct);
  REQUIRE(batch_results[2].duv == single_fl1.duv);
}

// -------------------------------------------------------------------------
// spd_to_cct §3.5 conformance (grid handling identical to the pipeline)
// -------------------------------------------------------------------------

TEST_CASE("CCT - spd_to_cct zero-fills a 400-700 nm SPD per §3.5",
          "[cct][slice03]") {
  auto [wl_full, vals_full] = load_spd_csv(data_path("d65_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  // Slice the bundled D65 down to the §3.5 minimum range 400-700 nm.
  std::vector<double> wl_narrow, vals_narrow;
  for (std::size_t i = 0; i < wl_full.size(); ++i) {
    if (wl_full[i] >= 400.0 && wl_full[i] <= 700.0) {
      wl_narrow.push_back(wl_full[i]);
      vals_narrow.push_back(vals_full[i]);
    }
  }
  REQUIRE(wl_narrow.size() == 301);

  // Hand-build the §3.5-conformed equivalent: full 380-780 grid, zeros
  // outside 400-700.
  std::vector<double> vals_padded(wl_full.size(), 0.0);
  for (std::size_t i = 0; i < wl_full.size(); ++i) {
    if (wl_full[i] >= 400.0 && wl_full[i] <= 700.0) {
      vals_padded[i] = vals_full[i];
    }
  }

  CctDuvResult narrow = spd_to_cct(wl_narrow, vals_narrow, cmf, lut);
  CctDuvResult padded = spd_to_cct(wl_full, vals_padded, cmf, lut);

  // Bit-identical: both routes reduce to the same conformed arrays.
  REQUIRE(narrow.cct == padded.cct);
  REQUIRE(narrow.duv == padded.duv);
}

TEST_CASE("CCT - spd_to_cct rejects a wavelength step above 5 nm (§3.5)",
          "[cct][slice03]") {
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  std::vector<double> wl, vals;
  for (double w = 380.0; w <= 780.0; w += 10.0) {
    wl.push_back(w);
    vals.push_back(1.0);
  }
  REQUIRE_THROWS_AS(spd_to_cct(wl, vals, cmf, lut), InvalidSpd);
  std::vector<std::vector<double>> batch = {vals};
  REQUIRE_THROWS_AS(spd_to_cct_batch(wl, batch, cmf, lut), InvalidSpd);
}

TEST_CASE("CCT - spd_to_cct_batch matches spd_to_cct on a 400-700 nm grid",
          "[cct][slice03]") {
  auto [wl_full, vals_d65] = load_spd_csv(data_path("d65_1nm.csv"));
  auto [wl_a, vals_a] = load_spd_csv(data_path("illuminant_a_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  std::vector<double> wl_narrow;
  std::vector<double> d65_narrow, a_narrow;
  for (std::size_t i = 0; i < wl_full.size(); ++i) {
    if (wl_full[i] >= 400.0 && wl_full[i] <= 700.0) {
      wl_narrow.push_back(wl_full[i]);
      d65_narrow.push_back(vals_d65[i]);
      a_narrow.push_back(vals_a[i]);
    }
  }

  std::vector<std::vector<double>> batch = {d65_narrow, a_narrow};
  auto batch_results = spd_to_cct_batch(wl_narrow, batch, cmf, lut);
  CctDuvResult single_d65 = spd_to_cct(wl_narrow, d65_narrow, cmf, lut);
  CctDuvResult single_a = spd_to_cct(wl_narrow, a_narrow, cmf, lut);

  REQUIRE(batch_results.size() == 2);
  REQUIRE(batch_results[0].cct == single_d65.cct);
  REQUIRE(batch_results[0].duv == single_d65.duv);
  REQUIRE(batch_results[1].cct == single_a.cct);
  REQUIRE(batch_results[1].duv == single_a.duv);
}

TEST_CASE("CCT - spd_to_cct_batch_prepared matches spd_to_cct_batch "
          "bit-for-bit",
          "[cct][slice03]") {
  auto [wl_full, vals_d65] = load_spd_csv(data_path("d65_1nm.csv"));
  auto [wl_a, vals_a] = load_spd_csv(data_path("illuminant_a_1nm.csv"));
  CmfData cmf = load_cmf_2deg(data_path("cie_1931_2.csv"));
  PlanckianLut lut = load_planckian_lut(data_path("planckian_uv.csv"));

  // Narrow grid so the §3.5 conform is not the identity.
  std::vector<double> wl_narrow, d65_narrow, a_narrow;
  for (std::size_t i = 0; i < wl_full.size(); ++i) {
    if (wl_full[i] >= 400.0 && wl_full[i] <= 700.0) {
      wl_narrow.push_back(wl_full[i]);
      d65_narrow.push_back(vals_d65[i]);
      a_narrow.push_back(vals_a[i]);
    }
  }
  std::vector<std::vector<double>> batch = {d65_narrow, a_narrow};

  // Prepared inputs, exactly as the bindings cache builds them.
  Spd probe(wl_narrow, std::vector<double>(wl_narrow.size(), 1.0));
  CmfData cmf_resampled = resample_cmf(probe.wavelengths(), cmf);

  auto plain = spd_to_cct_batch(wl_narrow, batch, cmf, lut);
  auto prepared =
      spd_to_cct_batch_prepared(wl_narrow, batch, cmf_resampled, lut);

  REQUIRE(prepared.size() == plain.size());
  for (std::size_t i = 0; i < plain.size(); ++i) {
    REQUIRE(prepared[i].cct == plain[i].cct);
    REQUIRE(prepared[i].duv == plain[i].duv);
  }

  // Mismatched CMF (resampled to a different-length grid) is rejected.
  std::vector<double> wl_5nm;
  for (double w = 380.0; w <= 780.0; w += 5.0) {
    wl_5nm.push_back(w);
  }
  CmfData cmf_81 = resample_cmf(wl_5nm, cmf);
  REQUIRE_THROWS_AS(spd_to_cct_batch_prepared(wl_narrow, batch, cmf_81, lut),
                    std::invalid_argument);
}

} // namespace
} // namespace tm30::test
