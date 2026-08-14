// Slice 13 - opt-in parallel batch evaluation (n_workers).
//
// Tests for the n_workers parameter on try_evaluate() and
// try_evaluate_cached():
//   1. Determinism: bit-identical results across n_workers in {1,2,4,8}
//      (task parallelism - exact equality, not tolerance).
//   2. Failure-position correctness under chunking: InvalidSpd -> nullopt
//      lands at exactly the right indices for every n_workers, including
//      at chunk boundaries.
//   3. actual_workers capping: n_workers > batch size spawns no
//      empty-work threads, doesn't crash, gives correct results.
//   4. Mandatory timing-regression test: the n_workers<=1 path must NOT
//      regress vs the pre-change baseline recorded on this machine -
//      spawning even one std::thread costs ~40 us here against a ~140 us
//      per-SPD workload (a double-digit-percent regression on the
//      default path). This is the automated enforcement of the
//      non-negotiable "n_workers<=1 contains zero std::thread" rule.
//   5. Grid-matrix sanity: parallel == sequential bit-for-bit on the
//      default 1 nm grid and a custom grid, both cached and per-call.

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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tm30::test {
namespace {

// -- Test helpers (same pattern as slice_11) --------------------------

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
    for (std::size_t c = 1; c < row.size(); ++c)
      data.samples[c - 1].push_back(row[c]);
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

// The bundled 19-illuminant corpus (same set benchmarks use), all
// resampled onto the common 1 nm grid (380-780, 401 pts).
const std::vector<std::string> kCorpusNames = {
    "d65_1nm",  "fl1_1nm",  "fl2_1nm",  "fl3_1nm",         "fl4_1nm",
    "fl5_1nm",  "fl6_1nm",  "fl7_1nm",  "fl8_1nm",         "fl9_1nm",
    "fl10_1nm", "fl11_1nm", "fl12_1nm", "hp1_1nm",         "hp2_1nm",
    "hp3_1nm",  "hp4_1nm",  "hp5_1nm",  "illuminant_a_1nm"};

std::vector<double> resample_to_1nm(const std::vector<double> &wl,
                                    const std::vector<double> &vals) {
  // Linear interpolation onto 380..780 @ 1 nm (same as
  // benchmarks/benchmark_tm30.py's np.interp).
  std::vector<double> out;
  out.reserve(401);
  for (std::size_t j = 0; j < 401; ++j) {
    const double x = 380.0 + static_cast<double>(j);
    std::size_t k = 0;
    while (k + 1 < wl.size() && wl[k + 1] < x)
      ++k;
    const double x0 = wl[k], x1 = wl[k + 1];
    const double y0 = vals[k], y1 = vals[k + 1];
    out.push_back((x1 == x0) ? y0 : y0 + (y1 - y0) * (x - x0) / (x1 - x0));
  }
  return out;
}

struct Corpus {
  std::vector<double> wl; // 380..780 @ 1 nm (401 pts)
  std::vector<Spd> owned;
  std::vector<SpdView> views;

  Corpus() {
    wl.reserve(401);
    for (std::size_t j = 0; j < 401; ++j)
      wl.push_back(380.0 + static_cast<double>(j));
    for (const auto &name : kCorpusNames) {
      auto [w, v] = load_spd_csv(data_path(name + ".csv"));
      std::vector<double> vals =
          (w.size() == wl.size()) ? std::move(v) : resample_to_1nm(w, v);
      owned.emplace_back(wl, std::move(vals));
    }
    for (auto &s : owned)
      views.push_back({s.wavelengths(), s.values()});
  }
};

Corpus &corpus() {
  static Corpus c;
  return c;
}

SpdView make_spd_view(const std::vector<double> &wl,
                      const std::vector<double> &vals) {
  return SpdView{std::span<const double>(wl.data(), wl.size()),
                 std::span<const double>(vals.data(), vals.size())};
}

// -- Bit-identical comparison of two full result vectors --------------

bool cam02_equal(const Cam02Ucs &a, const Cam02Ucs &b) {
  return a.J_prime == b.J_prime && a.a_prime == b.a_prime &&
         a.b_prime == b.b_prime;
}

bool xyz_equal(const XyzTriple &a, const XyzTriple &b) {
  return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
}

bool results_equal(const std::optional<Tm30Result> &a,
                   const std::optional<Tm30Result> &b) {
  if (a.has_value() != b.has_value())
    return false;
  if (!a.has_value())
    return true;
  const auto &ca = a->colorimetry;
  const auto &cb = b->colorimetry;
  if (!(ca.cct == cb.cct && ca.duv == cb.duv &&
        ca.delta_e_avg == cb.delta_e_avg && ca.Rf == cb.Rf &&
        ca.rf_skin == cb.rf_skin))
    return false;
  if (ca.reference_spd_values != cb.reference_spd_values)
    return false;
  for (std::size_t i = 0; i < 99; ++i) {
    if (!xyz_equal(ca.xyz_test_ces[i], cb.xyz_test_ces[i]))
      return false;
    if (!xyz_equal(ca.xyz_ref_ces[i], cb.xyz_ref_ces[i]))
      return false;
    if (!cam02_equal(ca.jab_test_ces[i], cb.jab_test_ces[i]))
      return false;
    if (!cam02_equal(ca.jab_ref_ces[i], cb.jab_ref_ces[i]))
      return false;
  }
  for (int j = 0; j < 16; ++j) {
    if (ca.hue_bins[j] != cb.hue_bins[j])
      return false;
    const auto &ga = ca.gamut;
    const auto &gb = cb.gamut;
    if (ga.Rg != gb.Rg)
      return false;
    if (ga.test_avg.J_prime[j] != gb.test_avg.J_prime[j] ||
        ga.test_avg.a_prime[j] != gb.test_avg.a_prime[j] ||
        ga.test_avg.b_prime[j] != gb.test_avg.b_prime[j])
      return false;
    if (ga.ref_avg.J_prime[j] != gb.ref_avg.J_prime[j] ||
        ga.ref_avg.a_prime[j] != gb.ref_avg.a_prime[j] ||
        ga.ref_avg.b_prime[j] != gb.ref_avg.b_prime[j])
      return false;
    if (ga.local.Rf_hj[j] != gb.local.Rf_hj[j] ||
        ga.local.Rcs_hj_percent[j] != gb.local.Rcs_hj_percent[j] ||
        ga.local.Rhs_hj[j] != gb.local.Rhs_hj[j] ||
        ga.local.DE_hj[j] != gb.local.DE_hj[j])
      return false;
    const auto &cvg_a = ga.cvg;
    const auto &cvg_b = gb.cvg;
    if (cvg_a.J_test[j] != cvg_b.J_test[j] ||
        cvg_a.x_test[j] != cvg_b.x_test[j] ||
        cvg_a.y_test[j] != cvg_b.y_test[j] ||
        cvg_a.J_ref[j] != cvg_b.J_ref[j] || cvg_a.x_ref[j] != cvg_b.x_ref[j] ||
        cvg_a.y_ref[j] != cvg_b.y_ref[j])
      return false;
  }
  const auto &va = a->validity;
  const auto &vb = b->validity;
  return va.cct_out_of_range == vb.cct_out_of_range &&
         va.duv_out_of_range == vb.duv_out_of_range &&
         va.extrapolated == vb.extrapolated;
}

bool vectors_equal(const std::vector<std::optional<Tm30Result>> &a,
                   const std::vector<std::optional<Tm30Result>> &b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!results_equal(a[i], b[i]))
      return false;
  return true;
}

// -- Timing helpers ----------------------------------------------------

using Clock = std::chrono::steady_clock;

double median_ms(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2] * 1e3;
}

// Min-of-3-medians protocol: robust to transient machine load and the
// frequency-management jitter of the calibration machine (Ryzen APU with
// the schedutil governor).
double min_median_ms(std::size_t reps, const std::vector<SpdView> &batch,
                     const ResampledTables &tables, const PlanckianLut &lut,
                     std::size_t n_workers) {
  double best = 1e30;
  for (int sub = 0; sub < 3; ++sub) {
    std::vector<double> t;
    t.reserve(reps);
    for (std::size_t i = 0; i < reps; ++i) {
      auto t0 = Clock::now();
      auto r = try_evaluate_cached(batch, tables, lut, Tm30Request{true, true},
                                   n_workers);
      (void)r;
      auto t1 = Clock::now();
      t.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    best = std::min(best, median_ms(std::move(t)));
  }
  return best;
}

// ======================================================================
//  1. Determinism - bit-identical across n_workers
// ======================================================================

TEST_CASE("Parallel - results bit-identical across n_workers (cached path)",
          "[parallel][slice13][determinism]") {
  auto &t = tables();
  auto &c = corpus();
  ResampledTables rtab =
      prepare_resampled_tables(c.wl, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis);
  Tm30Request req{/*bins=*/true, /*samples=*/true};

  auto seq = try_evaluate_cached(c.views, rtab, t.lut, req, /*n_workers=*/1);
  REQUIRE(seq.size() == 19);
  for (std::size_t nw : {std::size_t(2), std::size_t(4), std::size_t(8)}) {
    auto par = try_evaluate_cached(c.views, rtab, t.lut, req, nw);
    REQUIRE(vectors_equal(seq, par)); // exact ==, not tolerance
  }
}

TEST_CASE("Parallel - results bit-identical across n_workers (plain path)",
          "[parallel][slice13][determinism]") {
  auto &t = tables();
  auto &c = corpus();

  auto seq = try_evaluate(c.views, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis,
                          t.lut, Tm30Request{true, true}, /*n_workers=*/1);
  for (std::size_t nw : {std::size_t(2), std::size_t(4), std::size_t(8)}) {
    auto par = try_evaluate(c.views, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis,
                            t.lut, Tm30Request{true, true}, nw);
    REQUIRE(vectors_equal(seq, par));
  }
}

TEST_CASE("Parallel - samples/bins request flags still gate output",
          "[parallel][slice13][determinism]") {
  auto &t = tables();
  auto &c = corpus();
  ResampledTables rtab =
      prepare_resampled_tables(c.wl, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis);

  for (std::size_t nw : {std::size_t(1), std::size_t(4)}) {
    auto a =
        try_evaluate_cached(c.views, rtab, t.lut, Tm30Request{true, false}, nw);
    auto b =
        try_evaluate_cached(c.views, rtab, t.lut, Tm30Request{false, true}, nw);
    REQUIRE(vectors_equal(a, b));
    REQUIRE(a.size() == 19);
  }
}

// ======================================================================
//  2. Failure-position correctness under chunking
// ======================================================================

TEST_CASE("Parallel - InvalidSpd -> nullopt at exact positions under "
          "chunking",
          "[parallel][slice13][failure-position]") {
  auto &t = tables();
  auto &c = corpus();
  ResampledTables rtab =
      prepare_resampled_tables(c.wl, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis);

  // Invalid SPD: 2-point grid {400, 500} nm does not cover 400-700 nm,
  // so Spd construction inside the per-iteration loop throws InvalidSpd.
  // Positions cover first/last and every chunk boundary for n_workers in
  // {1,2,4,8} on a 19-SPD batch (chunk sizes: 1:19, 2:10+9, 4:5+5+5+4,
  // 8:3+3+3+2+2+2+2+2).
  const std::vector<std::size_t> bad_positions = {0,  2,  3,  5,  9, 10,
                                                  11, 14, 15, 17, 18};

  for (const std::size_t pos : bad_positions) {
    // Build a fresh view batch with exactly one invalid SPD at `pos`.
    std::vector<Spd> owned;
    std::vector<SpdView> views;
    for (std::size_t i = 0; i < 19; ++i) {
      if (i == pos) {
        // Non-owning view over static invalid arrays - never constructed
        // as an Spd, so try_evaluate's per-iteration catch sees it.
        static const std::vector<double> bad_wl = {400.0, 500.0};
        static const std::vector<double> bad_vals = {1.0, 1.0};
        views.push_back(make_spd_view(bad_wl, bad_vals));
      } else {
        owned.push_back(c.owned[i]); // copy of a valid Spd
        views.push_back({owned.back().wavelengths(), owned.back().values()});
      }
    }

    auto seq = try_evaluate_cached(views, rtab, t.lut, Tm30Request{true, true},
                                   /*n_workers=*/1);
    REQUIRE(seq.size() == 19);
    REQUIRE_FALSE(seq[pos].has_value());
    for (std::size_t i = 0; i < 19; ++i)
      if (i != pos)
        REQUIRE(seq[i].has_value());

    for (std::size_t nw : {std::size_t(2), std::size_t(4), std::size_t(8)}) {
      auto par =
          try_evaluate_cached(views, rtab, t.lut, Tm30Request{true, true}, nw);
      REQUIRE(vectors_equal(seq, par)); // nullopt at exactly the same idx
    }
  }
}

// ======================================================================
//  3. actual_workers capping (n_workers > batch size)
// ======================================================================

TEST_CASE("Parallel - n_workers larger than batch size", "[parallel][slice13]"
                                                         "[capping]") {
  auto &t = tables();
  auto &c = corpus();
  ResampledTables rtab =
      prepare_resampled_tables(c.wl, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis);

  // 3-SPD batch with n_workers=8: must not spawn empty-work threads,
  // must not crash, must give correct results.
  std::vector<SpdView> small(c.views.begin(), c.views.begin() + 3);
  auto seq = try_evaluate_cached(small, rtab, t.lut, Tm30Request{true, true},
                                 /*n_workers=*/1);
  auto par = try_evaluate_cached(small, rtab, t.lut, Tm30Request{true, true},
                                 /*n_workers=*/8);
  REQUIRE(vectors_equal(seq, par));

  // 1-SPD batch with n_workers=4 (parallel path with a single chunk).
  std::vector<SpdView> single1{c.views[0]};
  auto seq1 = try_evaluate_cached(single1, rtab, t.lut, Tm30Request{true, true},
                                  /*n_workers=*/1);
  auto par1 = try_evaluate_cached(single1, rtab, t.lut, Tm30Request{true, true},
                                  /*n_workers=*/4);
  REQUIRE(vectors_equal(seq1, par1));

  // 19-SPD batch with n_workers=100 (extreme oversubscription).
  auto seq19 =
      try_evaluate_cached(c.views, rtab, t.lut, Tm30Request{true, true},
                          /*n_workers=*/1);
  auto par19 =
      try_evaluate_cached(c.views, rtab, t.lut, Tm30Request{true, true},
                          /*n_workers=*/100);
  REQUIRE(vectors_equal(seq19, par19));

  // Empty batch must not divide by zero or crash.
  std::vector<SpdView> empty;
  auto e1 = try_evaluate_cached(empty, rtab, t.lut, Tm30Request{true, true},
                                /*n_workers=*/1);
  auto e8 = try_evaluate_cached(empty, rtab, t.lut, Tm30Request{true, true},
                                /*n_workers=*/8);
  REQUIRE(e1.empty());
  REQUIRE(vectors_equal(e1, e8));
}

// ======================================================================
//  4. Mandatory timing-regression test (n_workers=1 default path)
// ======================================================================
//
// Baselines measured on THIS machine (2026-08-04, AMD Ryzen 3 PRO 3300U,
// 4 cores, GCC 13.3 -O3) BEFORE the n_workers feature landed, via
// tools/bench_cpp_baseline.cpp:
//   batch1  (1 SPD/call):  median 0.1437-0.1444 ms  -> constant 0.145
//   batch19 (19 SPD/call): median 2.8486-2.8569 ms  -> constant 2.86
// A std::thread spawn+join costs ~40 us on this machine, so a buggy
// n_workers=1 that spawns one thread would land at ~0.185 ms on batch1 -
// +28%, far above the 1.10x gate. Median noise across runs is <0.5%,
// so 1.10x is a ~20x margin over noise. If this machine changes, re-run
// tools/bench_cpp_baseline.cpp and update the constants.

TEST_CASE("Parallel - n_workers=1 timing regression vs pre-change baseline",
          "[parallel][slice13][timing]") {
  auto &t = tables();
  auto &c = corpus();
  ResampledTables rtab =
      prepare_resampled_tables(c.wl, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis);

  // Warm-up (tables + first-touch allocation effects).
  try_evaluate_cached(c.views, rtab, t.lut, Tm30Request{true, true}, 1);

  // batch1: the per-call overhead is the largest fraction here - a
  // single spawned thread (+40 us) would be a +28% regression, easily
  // above the 10% gate.
  const double t1 =
      min_median_ms(300, std::vector<SpdView>{c.views[0]}, rtab, t.lut, 1);
  INFO("n_workers=1 batch1 median: " << t1 << " ms (baseline 0.145, gate "
                                     << 0.145 * 1.10 << ")");
  REQUIRE(t1 < 0.145 * 1.10);

  // batch19: gross-regression gate (spawn-1-thread is only +1.4% here,
  // but the batch1 leg above catches that specific bug).
  const double t19 = min_median_ms(100, c.views, rtab, t.lut, 1);
  INFO("n_workers=1 batch19 median: " << t19 << " ms (baseline 2.86, gate "
                                      << 2.86 * 1.10 << ")");
  REQUIRE(t19 < 2.86 * 1.10);
}

// ======================================================================
//  5. Grid-matrix sanity (parallel == sequential, default + custom grid)
// ======================================================================

TEST_CASE("Parallel - grid matrix: 1 nm cached grid, bit-identical",
          "[parallel][slice13][gridmatrix]") {
  auto &t = tables();
  auto &c = corpus();
  ResampledTables rtab =
      prepare_resampled_tables(c.wl, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis);

  auto seq = try_evaluate_cached(c.views, rtab, t.lut, Tm30Request{true, true},
                                 /*n_workers=*/1);
  auto par = try_evaluate_cached(c.views, rtab, t.lut, Tm30Request{true, true},
                                 /*n_workers=*/4);
  REQUIRE(vectors_equal(seq, par));
}

TEST_CASE("Parallel - grid matrix: custom 5 nm grid via per-call path",
          "[parallel][slice13][gridmatrix]") {
  auto &t = tables();
  auto &c = corpus();

  // Custom grid: 380..780 @ 5 nm (81 pts) - different from the default
  // 1 nm grid, exercising the per-call (non-cached) try_evaluate path.
  std::vector<double> wl5;
  for (std::size_t j = 0; j < 81; ++j)
    wl5.push_back(380.0 + 5.0 * static_cast<double>(j));

  std::vector<Spd> owned;
  std::vector<SpdView> views;
  for (std::size_t i = 0; i < 19; ++i) {
    auto [w, v] = load_spd_csv(data_path(kCorpusNames[i] + ".csv"));
    std::vector<double> vals(w.size() == wl5.size() ? std::move(v)
                                                    : std::vector<double>());
    if (vals.empty()) {
      // linear interp onto wl5
      vals.reserve(wl5.size());
      for (double x : wl5) {
        std::size_t k = 0;
        while (k + 1 < w.size() && w[k + 1] < x)
          ++k;
        const double x0 = w[k], x1 = w[k + 1];
        const double y0 = v[k], y1 = v[k + 1];
        vals.push_back((x1 == x0) ? y0 : y0 + (y1 - y0) * (x - x0) / (x1 - x0));
      }
    }
    owned.emplace_back(wl5, std::move(vals));
    views.push_back({owned.back().wavelengths(), owned.back().values()});
  }

  auto seq = try_evaluate(views, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut,
                          Tm30Request{true, true}, /*n_workers=*/1);
  auto par = try_evaluate(views, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut,
                          Tm30Request{true, true}, /*n_workers=*/4);
  REQUIRE(vectors_equal(seq, par));
}

// ======================================================================
//  Phase 2 - persistent workers (TaskPool)
// ======================================================================

TEST_CASE("Parallel - TaskPool partition: every index exactly once",
          "[parallel][slice13][persistent]") {
  for (std::size_t nw : {std::size_t(1), std::size_t(2), std::size_t(3),
                         std::size_t(4), std::size_t(8)}) {
    TaskPool pool(nw);
    for (std::size_t n :
         {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(3),
          std::size_t(5), std::size_t(19), std::size_t(100)}) {
      std::mutex m;
      std::vector<int> seen(n, 0);
      pool.run_chunked(n, [&](std::size_t i) {
        std::lock_guard<std::mutex> lock(m);
        ++seen[i];
      });
      for (std::size_t i = 0; i < n; ++i)
        REQUIRE(seen[i] == 1); // each index exactly once, none twice
    }
  }
}

TEST_CASE("Parallel - TaskPool fault isolation: failed job leaves all "
          "workers alive",
          "[parallel][slice13][persistent]") {
  TaskPool pool(4);
  std::atomic<int> calls{0};
  // A job whose LAST iteration throws: every chunk runs to completion,
  // the first exception is captured and rethrown on the caller (never
  // kills a worker)...
  REQUIRE_THROWS_AS(pool.run_chunked(8,
                                     [&](std::size_t i) {
                                       calls.fetch_add(1);
                                       if (i == 7)
                                         throw std::runtime_error("boom");
                                     }),
                    std::runtime_error);
  REQUIRE(calls.load() == 8); // all 8 iterations ran before the throw

  // ...and the pool must run the next job correctly at full strength.
  std::atomic<long> sum{0};
  pool.run_chunked(8,
                   [&](std::size_t i) { sum.fetch_add(static_cast<long>(i)); });
  REQUIRE(sum.load() == 28); // 0+...+7
}

TEST_CASE("Parallel - TaskPool construct/destroy without use, and after "
          "heavy use",
          "[parallel][slice13][persistent]") {
  {
    TaskPool pool(4);
  } // constructed, never used, destroyed: no hang

  {
    TaskPool pool(4); // 100 jobs then destroy: clean shutdown
    for (int k = 0; k < 100; ++k)
      pool.run_chunked(19, [](std::size_t) {});
  }
}

TEST_CASE("Parallel - persistent workers: 50 repeated calls bit-identical "
          "to sequential (cached path)",
          "[parallel][slice13][persistent][determinism]") {
  auto &t = tables();
  auto &c = corpus();
  ResampledTables rtab =
      prepare_resampled_tables(c.wl, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis);

  TaskPool pool(4);
  auto seq = try_evaluate_cached(c.views, rtab, t.lut, Tm30Request{true, true},
                                 /*n_workers=*/1);
  for (int k = 0; k < 50; ++k) {
    auto r = try_evaluate_cached(c.views, rtab, t.lut, Tm30Request{true, true},
                                 /*n_workers=*/4, &pool);
    REQUIRE(vectors_equal(seq, r)); // every call bit-identical to #1
  }
}

TEST_CASE("Parallel - persistent workers: plain path bit-identical, and "
          "failure positions under pool chunking",
          "[parallel][slice13][persistent][determinism]") {
  auto &t = tables();
  auto &c = corpus();

  // Plain (non-cached) path through the pool.
  TaskPool pool(4);
  auto seq = try_evaluate(c.views, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis,
                          t.lut, Tm30Request{true, true}, /*n_workers=*/1);
  auto par =
      try_evaluate(c.views, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis, t.lut,
                   Tm30Request{true, true}, /*n_workers=*/4, &pool);
  REQUIRE(vectors_equal(seq, par));

  // Failure positions through the pool: nullopt must land exactly where
  // the invalid SPDs are, for a batch smaller than the pool too.
  ResampledTables rtab =
      prepare_resampled_tables(c.wl, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis);
  static const std::vector<double> bad_wl = {400.0, 500.0};
  static const std::vector<double> bad_vals = {1.0, 1.0};

  for (const std::size_t pos :
       {std::size_t(0), std::size_t(3), std::size_t(7), std::size_t(18)}) {
    std::vector<Spd> owned;
    std::vector<SpdView> views;
    for (std::size_t i = 0; i < 19; ++i) {
      if (i == pos) {
        views.push_back(make_spd_view(bad_wl, bad_vals));
      } else {
        owned.push_back(c.owned[i]);
        views.push_back({owned.back().wavelengths(), owned.back().values()});
      }
    }
    auto seq_r =
        try_evaluate_cached(views, rtab, t.lut, Tm30Request{true, true},
                            /*n_workers=*/1);
    auto pool_r =
        try_evaluate_cached(views, rtab, t.lut, Tm30Request{true, true},
                            /*n_workers=*/4, &pool);
    REQUIRE(vectors_equal(seq_r, pool_r));
    REQUIRE_FALSE(pool_r[pos].has_value());
  }
}

// ======================================================================
//  Cached path vs TM-30-20 §3.5 grid conformance
// ======================================================================

TEST_CASE("Cached path - narrow bound grid conforms like Spd",
          "[parallel][slice13][cached][spd]") {
  // A 400-700 nm @ 5 nm input grid is zero-filled to 380-780 nm inside
  // Spd (TM-30-20 §3.5). The cached tables must be resampled to that
  // SAME conformed grid, or per-row values and tables misalign. Binding
  // through an Spd probe (as the Python layer does) and comparing the
  // cached result against the plain per-row path must be bit-identical.
  auto &t = tables();

  std::vector<double> wl;
  for (double w = 400.0; w <= 700.0; w += 5.0)
    wl.push_back(w);
  std::vector<double> vals(wl.size());
  for (std::size_t i = 0; i < wl.size(); ++i)
    vals[i] = 1.0 + 0.001 * (wl[i] - 400.0);

  const Spd probe(wl, std::vector<double>(wl.size(), 1.0));
  ResampledTables rtab = prepare_resampled_tables(
      probe.wavelengths(), t.cmf_2deg, t.cmf_10deg, t.ces, t.basis);

  std::vector<SpdView> batch{SpdView{wl, vals}};
  Tm30Request req{/*bins=*/true, /*samples=*/true};

  auto cached = try_evaluate_cached(batch, rtab, t.lut, req, /*n_workers=*/1);
  auto plain = try_evaluate(batch, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis,
                            t.lut, req, /*n_workers=*/1);

  REQUIRE(cached.size() == 1);
  REQUIRE(plain.size() == 1);
  REQUIRE(cached[0].has_value());
  REQUIRE(plain[0].has_value());
  REQUIRE(vectors_equal(cached, plain));   // exact ==, not tolerance
  CHECK(cached[0]->validity.extrapolated); // zero-fill happened
}

TEST_CASE("Cached path - grid/tables misalignment yields nullopt, not "
          "garbage",
          "[parallel][slice13][cached][spd]") {
  // Tables resampled to the RAW (unconformed) narrow grid cannot serve
  // rows whose Spd conforms to 380-780 nm; the guard must turn the row
  // into nullopt instead of integrating misaligned arrays.
  auto &t = tables();

  std::vector<double> wl;
  for (double w = 400.0; w <= 700.0; w += 5.0)
    wl.push_back(w);
  std::vector<double> vals(wl.size(), 1.0);

  ResampledTables raw_tab =
      prepare_resampled_tables(wl, t.cmf_2deg, t.cmf_10deg, t.ces, t.basis);

  std::vector<SpdView> batch{SpdView{wl, vals}};
  Tm30Request req{/*bins=*/true, /*samples=*/true};

  auto res = try_evaluate_cached(batch, raw_tab, t.lut, req, /*n_workers=*/1);
  REQUIRE(res.size() == 1);
  REQUIRE_FALSE(res[0].has_value());
}

} // namespace
} // namespace tm30::test
