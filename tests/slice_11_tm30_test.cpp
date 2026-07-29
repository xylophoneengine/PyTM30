// Slice 11 - Lazy memoized Tm30 handle + batch evaluate() + Validity flags.
//
// Tests for the ergonomics layer:
//   1. Lazy evaluation (construction doesn't compute)
//   2. Memoization (second accessor call returns cached)
//   3. Cache invalidation
//   4. All accessors return correct values
//   5. Validity flags (CCT bounds, Duv bounds, extrapolation)
//   6. Batch evaluate() with valid and invalid SPDs
//   7. Batch request flags
//   8. SpdView construction and usage

#include <catch2/catch_test_macros.hpp>

#include "matchers.hpp"
#include "tm30/cct.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/errors.hpp"
#include "tm30/pipeline.hpp"
#include "tm30/reference.hpp"
#include "tm30/resample.hpp"
#include "tm30/spd.hpp"
#include "tm30/tm30.hpp"
#include "tolerances.hpp"

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace tm30::test {
namespace {

// ── Test helpers ─────────────────────────────────────────────────────

std::string data_path(const std::string &filename) {
  return std::string(TM30_DATA_DIR) + "/" + filename;
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
  CesData data;
  std::size_t n_ces = table.headers.size() - 1;
  data.samples.resize(n_ces);
  for (const auto &row : table.rows) {
    data.wavelengths.push_back(row[0]);
    for (std::size_t c = 1; c < row.size(); ++c) {
      data.samples[c - 1].push_back(row[c]);
    }
  }
  return data;
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

// Pre-load all data tables once (shared across tests).
struct Tables {
  CmfData cmf_2deg;
  CmfData cmf_10deg;
  CesData ces;
  DaylightBasis basis;
  PlanckianLut lut;

  Tables() {
    cmf_2deg = load_cmf(data_path("cie_1931_2.csv"));
    cmf_10deg = load_cmf(data_path("cmf_1964_10.csv"));
    ces = load_ces(data_path("ces.csv"));
    basis = load_daylight_basis(data_path("daylight_basis.csv"));
    lut = load_planckian_lut(data_path("planckian_uv.csv"));
  }
};

Tables &tables() {
  static Tables t;
  return t;
}

// ── SpdView helpers ──────────────────────────────────────────────────

SpdView make_spd_view(const std::vector<double> &wl,
                      const std::vector<double> &vals) {
  return SpdView{std::span<const double>(wl.data(), wl.size()),
                 std::span<const double>(vals.data(), vals.size())};
}

// ── Tm30 construction ────────────────────────────────────────────────

TEST_CASE("Tm30 - construction does not compute", "[tm30][slice11][lazy]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));
  Spd spd(std::move(wl), std::move(vals));

  Tm30 m(std::move(spd), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  // After construction, nothing should be computed yet.
  REQUIRE_FALSE(m.is_computed());
}

TEST_CASE("Tm30 - SPD validation at construction",
          "[tm30][slice11][validation]") {
  auto &t = tables();

  // SPD with insufficient wavelength range (< 400–700 nm) should throw.
  REQUIRE_THROWS_AS(Tm30(Spd({400.0, 500.0}, {1.0, 1.0}), t.cmf_2deg,
                         t.cmf_10deg, t.ces, t.basis, t.lut),
                    InvalidSpd);
}

// ── Memoization ──────────────────────────────────────────────────────

TEST_CASE("Tm30 - first access triggers computation", "[tm30][slice11][memo]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));
  Spd spd(std::move(wl), std::move(vals));

  Tm30 m(std::move(spd), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  REQUIRE_FALSE(m.is_computed());

  // First access triggers computation.
  double rf = m.rf();
  REQUIRE(m.is_computed());
  // Just check it returns a plausible value.
  REQUIRE(rf >= 0.0);
  REQUIRE(rf <= 100.001);
}

TEST_CASE("Tm30 - subsequent accesses reuse cache", "[tm30][slice11][memo]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));
  Spd spd(std::move(wl), std::move(vals));

  Tm30 m(std::move(spd), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  // First call: rf() triggers compute.
  double rf1 = m.rf();
  REQUIRE(m.is_computed());

  // Second call: rg() should reuse cache (no recompute).
  double rg1 = m.rg();
  REQUIRE(m.is_computed());

  // Third call: rf() again should return same value.
  double rf2 = m.rf();
  REQUIRE(rf1 == rf2);
}

TEST_CASE("Tm30 - multiple accessors all work from cache",
          "[tm30][slice11][memo]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));
  Spd spd(std::move(wl), std::move(vals));

  Tm30 m(std::move(spd), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  // All of these should work without issues.
  double cct = m.cct();
  double duv = m.duv();
  double rf = m.rf();
  double rg = m.rg();
  double de = m.delta_e_avg();
  double skin = m.rf_skin();

  REQUIRE(cct > 0.0);
  REQUIRE(rf >= 0.0);
  REQUIRE(rg >= 0.0);

  // Access array and struct references.
  const auto &cesi = m.rf_cesi();
  REQUIRE(cesi.size() == 99);
  for (int i = 0; i < 99; ++i) {
    REQUIRE(cesi[i] >= 0.0);
  }

  const auto &gam = m.gamut();
  REQUIRE(gam.Rg >= 0.0);

  const auto &local = m.local_chroma_shift();
  REQUIRE(local.Rf_hj[0] >= 0.0);

  const auto &cvg = m.cvg();
  REQUIRE(std::isfinite(cvg.x_test[0]));

  const auto &val = m.validity();
  REQUIRE_FALSE(val.cct_out_of_range); // D65 at ~6500 K is in range

  const auto &full = m.result();
  REQUIRE(full.colorimetry.Rf == rf);
  REQUIRE(full.validity.duv_out_of_range == val.duv_out_of_range);
}

// ── Cache invalidation ───────────────────────────────────────────────

TEST_CASE("Tm30 - invalidate clears cache", "[tm30][slice11][memo]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));
  Spd spd(std::move(wl), std::move(vals));

  Tm30 m(std::move(spd), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  REQUIRE_FALSE(m.is_computed());
  double rf1 = m.rf();
  REQUIRE(m.is_computed());

  // Invalidate should reset the cache.
  m.invalidate();
  REQUIRE_FALSE(m.is_computed());

  // Next access recomputes.
  double rf2 = m.rf();
  REQUIRE(m.is_computed());
  REQUIRE(rf1 == rf2); // Same SPD → same result.
}

// ── Accessor consistency with pipeline ───────────────────────────────

TEST_CASE("Tm30 - results match direct pipeline call",
          "[tm30][slice11][consistency]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));

  // Direct pipeline call.
  auto direct = compute_ces_colorimetry(wl, vals, t.cmf_2deg, t.cmf_10deg,
                                        t.ces, t.basis, t.lut);

  // Via Tm30 handle.
  Spd spd(std::move(wl), std::move(vals));
  Tm30 m(std::move(spd), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  REQUIRE_THAT(m.rf(), WithinTolerance(1e-12, direct.Rf));
  REQUIRE_THAT(m.rg(), WithinTolerance(1e-12, direct.gamut.Rg));
  REQUIRE_THAT(m.cct(), WithinTolerance(1e-12, direct.cct));
  REQUIRE_THAT(m.duv(), WithinTolerance(1e-12, direct.duv));
  REQUIRE_THAT(m.delta_e_avg(), WithinTolerance(1e-12, direct.delta_e_avg));
  REQUIRE_THAT(m.rf_skin(), WithinTolerance(1e-12, direct.rf_skin));
}

// ── Validity flags ───────────────────────────────────────────────────

TEST_CASE("Validity - D65 (reference illuminant) is fully valid",
          "[tm30][slice11][validity]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));
  Spd spd(std::move(wl), std::move(vals));

  Tm30 m(std::move(spd), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  const auto &v = m.validity();

  // D65 has CCT ≈ 6500 K - well within [1000, 25000].
  REQUIRE_FALSE(v.cct_out_of_range);

  // D65 is on the daylight locus, near Planckian - Duv should be small.
  REQUIRE_FALSE(v.duv_out_of_range);

  // D65 1nm covers 300-830 nm, which includes 380-780 - no extrapolation.
  REQUIRE_FALSE(v.extrapolated);
}

TEST_CASE("Validity - D65 self-consistency values",
          "[tm30][slice11][validity]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));
  Spd spd(std::move(wl), std::move(vals));

  Tm30 m(std::move(spd), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  // D65 is a reference illuminant → Rf/Rg should be near 100.
  REQUIRE_THAT(m.rf(), WithinTolerance(Tol_Rf, 100.0));
  REQUIRE_THAT(m.rg(), WithinTolerance(Tol_Rg, 100.0));

  REQUIRE(m.cct() > 4000.0); // D65 is well above 4000 K
  REQUIRE(std::abs(m.duv()) < 0.01);
}

TEST_CASE("Validity - extrapolated flag for truncated SPD",
          "[tm30][slice11][validity]") {
  auto &t = tables();

  // Construct an SPD covering only 400–700 nm (valid per Spd rules,
  // but requires extrapolation for the full 380–780 nm CES range).
  // TM-30-20 §3.5: Missing edges are zero-filled.
  std::vector<double> wl, vals;
  for (int lam = 400; lam <= 700; lam += 5) {
    wl.push_back(static_cast<double>(lam));
    vals.push_back(1.0); // flat SPD (equal-energy-like)
  }
  Spd spd(std::move(wl), std::move(vals));

  Tm30 m(std::move(spd), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  const auto &v = m.validity();

  // 400–700 nm is within the minimum range but doesn't cover
  // the full 380–780 nm CES range. Extrapolation is needed.
  REQUIRE(v.extrapolated);
}

TEST_CASE("Validity - struct default values", "[tm30][slice11][validity]") {
  Validity v;
  REQUIRE_FALSE(v.duv_out_of_range);
  REQUIRE_FALSE(v.cct_out_of_range);
  REQUIRE_FALSE(v.extrapolated);
}

TEST_CASE("Validity - struct can be set", "[tm30][slice11][validity]") {
  Validity v;
  v.duv_out_of_range = true;
  v.cct_out_of_range = true;
  v.extrapolated = true;

  REQUIRE(v.duv_out_of_range);
  REQUIRE(v.cct_out_of_range);
  REQUIRE(v.extrapolated);
}

// ── Batch API ────────────────────────────────────────────────────────

TEST_CASE("Batch - evaluate single D65 SPD", "[tm30][slice11][batch]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));

  std::vector<SpdView> spds;
  spds.push_back(make_spd_view(wl, vals));

  auto results =
      try_evaluate(spds, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  REQUIRE(results.size() == 1);
  REQUIRE(results[0].has_value());

  const auto &r = *results[0];
  REQUIRE(r.colorimetry.Rf >= 0.0);
  REQUIRE(r.colorimetry.gamut.Rg >= 0.0);

  // D65 should be valid.
  REQUIRE_FALSE(r.validity.cct_out_of_range);
  REQUIRE_FALSE(r.validity.duv_out_of_range);
}

TEST_CASE("Batch - invalid SPD returns nullopt", "[tm30][slice11][batch]") {
  auto &t = tables();

  // SPD with insufficient wavelength range.
  std::vector<double> bad_wl = {400.0, 500.0};
  std::vector<double> bad_vals = {1.0, 1.0};

  std::vector<SpdView> spds;
  spds.push_back(make_spd_view(bad_wl, bad_vals));

  auto results =
      try_evaluate(spds, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  REQUIRE(results.size() == 1);
  REQUIRE_FALSE(results[0].has_value());
}

TEST_CASE("Batch - mixed valid and invalid SPDs", "[tm30][slice11][batch]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));

  // Create a bad SPD
  std::vector<double> bad_wl = {400.0, 500.0};
  std::vector<double> bad_vals = {1.0, 1.0};

  std::vector<SpdView> spds;
  spds.push_back(make_spd_view(wl, vals));         // valid
  spds.push_back(make_spd_view(bad_wl, bad_vals)); // invalid
  spds.push_back(make_spd_view(wl, vals));         // valid

  auto results =
      try_evaluate(spds, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  REQUIRE(results.size() == 3);
  REQUIRE(results[0].has_value());
  REQUIRE_FALSE(results[1].has_value());
  REQUIRE(results[2].has_value());

  // Valid results should be consistent with each other (same SPD).
  REQUIRE_THAT(results[0]->colorimetry.Rf,
               WithinTolerance(1e-12, results[2]->colorimetry.Rf));
}

TEST_CASE("Batch - empty input returns empty output",
          "[tm30][slice11][batch]") {
  auto &t = tables();
  std::vector<SpdView> empty_spds;
  auto results =
      try_evaluate(empty_spds, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);
  REQUIRE(results.empty());
}

TEST_CASE("Batch - batch output matches single Tm30 handle",
          "[tm30][slice11][batch]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));

  // Via Tm30 handle.
  Spd spd1{std::vector<double>(wl), std::vector<double>(vals)};
  Tm30 m(std::move(spd1), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  // Via batch.
  std::vector<SpdView> spds;
  spds.push_back(make_spd_view(wl, vals));
  auto results =
      try_evaluate(spds, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  REQUIRE(results.size() == 1);
  REQUIRE(results[0].has_value());

  REQUIRE_THAT(results[0]->colorimetry.Rf, WithinTolerance(1e-12, m.rf()));
  REQUIRE_THAT(results[0]->colorimetry.gamut.Rg,
               WithinTolerance(1e-12, m.rg()));
  REQUIRE_THAT(results[0]->colorimetry.cct, WithinTolerance(1e-12, m.cct()));
  REQUIRE(results[0]->validity.cct_out_of_range ==
          m.validity().cct_out_of_range);
  REQUIRE(results[0]->validity.duv_out_of_range ==
          m.validity().duv_out_of_range);
  REQUIRE(results[0]->validity.extrapolated == m.validity().extrapolated);
}

// ── Request flags ────────────────────────────────────────────────────

TEST_CASE("Tm30Request - default values", "[tm30][slice11][request]") {
  Tm30Request req;
  REQUIRE(req.bins == true);
  REQUIRE(req.samples == false);
}

TEST_CASE("Tm30Request - can be customized", "[tm30][slice11][request]") {
  Tm30Request req{.bins = false, .samples = true};
  REQUIRE_FALSE(req.bins);
  REQUIRE(req.samples);
}

TEST_CASE("Batch - accepts request flags", "[tm30][slice11][request]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));

  std::vector<SpdView> spds;
  spds.push_back(make_spd_view(wl, vals));

  // Request without bins and with samples.
  Tm30Request req{.bins = false, .samples = true};
  auto results =
      try_evaluate(spds, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut, req);

  REQUIRE(results.size() == 1);
  REQUIRE(results[0].has_value());

  // The full pipeline still runs (flags bound output, not compute).
  // Verify the result is complete.
  const auto &r = *results[0];
  REQUIRE(r.colorimetry.Rf >= 0.0);
  REQUIRE(r.colorimetry.gamut.Rg >= 0.0);
  REQUIRE(r.colorimetry.rf_cesi[0] >= 0.0);
}

// ── SpdView ──────────────────────────────────────────────────────────

TEST_CASE("SpdView - from vector data", "[tm30][slice11][spdview]") {
  std::vector<double> wl = {380.0, 385.0, 390.0};
  std::vector<double> vals = {1.0, 2.0, 3.0};

  SpdView sv{std::span<const double>(wl), std::span<const double>(vals)};

  REQUIRE(sv.wavelengths.size() == 3);
  REQUIRE(sv.values.size() == 3);
  REQUIRE(sv.wavelengths[0] == 380.0);
  REQUIRE(sv.values[2] == 3.0);
}

TEST_CASE("SpdView - empty spans", "[tm30][slice11][spdview]") {
  SpdView sv;
  REQUIRE(sv.wavelengths.empty());
  REQUIRE(sv.values.empty());
}

// ── Tm30Result ───────────────────────────────────────────────────────

TEST_CASE("Tm30Result - contains colorimetry and validity",
          "[tm30][slice11][result]") {
  // Tm30Result fields are set by the pipeline. Test that after computation
  // the result is self-consistent.
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));
  Spd spd{std::move(wl), std::move(vals)};

  Tm30 m(std::move(spd), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  const Tm30Result &r = m.result();

  // colorimetry fields are populated.
  REQUIRE(r.colorimetry.cct > 0.0);
  REQUIRE(r.colorimetry.Rf >= 0.0);
  REQUIRE(r.colorimetry.gamut.Rg >= 0.0);

  // validity fields are populated.
  // D65 is fully in range → no warnings.
  REQUIRE_FALSE(r.validity.duv_out_of_range);
  REQUIRE_FALSE(r.validity.cct_out_of_range);
}

// ── Error model ──────────────────────────────────────────────────────

TEST_CASE("Error model - batch never throws", "[tm30][slice11][errors]") {
  auto &t = tables();

  // Multiple invalid SPDs - batch should return nullopt, never throw.
  std::vector<double> bad_wl = {400.0, 500.0};
  std::vector<double> bad_vals = {1.0, 1.0};

  std::vector<SpdView> spds;
  for (int i = 0; i < 10; ++i) {
    spds.push_back(make_spd_view(bad_wl, bad_vals));
  }

  REQUIRE_NOTHROW([&]() {
    auto results =
        try_evaluate(spds, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);
    REQUIRE(results.size() == 10);
    for (const auto &r : results) {
      REQUIRE_FALSE(r.has_value());
    }
  }());
}

// ── Memoization correctness ──────────────────────────────────────────

TEST_CASE("Tm30 - colorimetry_result() accessor", "[tm30][slice11][memo]") {
  auto &t = tables();
  auto [wl, vals] = load_spd_csv(data_path("d65_1nm.csv"));
  Spd spd(std::move(wl), std::move(vals));

  Tm30 m(std::move(spd), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut);

  const auto &cr = m.colorimetry_result();
  REQUIRE(cr.cct > 0.0);
  REQUIRE(m.is_computed());
}

} // namespace
} // namespace tm30::test
