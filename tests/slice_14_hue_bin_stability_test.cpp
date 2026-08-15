// Slice 14 - hue-bin assignment is discontinuous: the runnable evidence.
//
// This file backs the README section "Known issue: hue-bin assignment is
// discontinuous". Run it with
//
//     ctest -R hue-bin -V
//
// and it prints every number that section quotes.
//
// TM-30-20 S4.3 sorts the 99 CES into 16 hue bins of exactly 22.5 deg
// using the reference hue angle hr = atan2(b'r, a'r) and half-open
// intervals [j x 22.5, (j+1) x 22.5). Bin assignment is therefore a step
// function of a continuous quantity. When a sample crosses, it leaves one
// bin and joins another; both bins' means move, so Rf,hj, Rcs,hj, Rhs,hj
// and Rg jump discontinuously, while Rf, CCT and Duv -- which never look
// at a bin -- do not. This is a property of the standard, not of this
// implementation.
//
// ---------------------------------------------------------------------
// WHY THIS FILE CONSTRUCTS ITS OWN EDGE CASES
// ---------------------------------------------------------------------
// The obvious experiment - perturb a corpus of real spectra and count how
// many of the 99 x N samples changed bin - produces a diluted statistic.
// The median sample sits ~5 deg from a boundary against a half-width of
// 11.25 deg, so most samples are nowhere near an edge and could not flip
// under any realistic perturbation. "0 of 2970 flipped" and "0 of the 0
// samples that were actually at risk flipped" are the same number and
// mean entirely different things.
//
// So the corpus sweep here always reports the AT-RISK count (samples
// whose boundary distance is smaller than the perturbation's hue shift)
// next to the flip count, and it is presented as context only.
//
// The evidence proper is constructed. A mixing fraction between two
// bundled SPDs is a continuous one-parameter family: for a fixed CES
// index the reference hue angle h_i(t) varies continuously with t, so
// bisecting on t drives a chosen sample arbitrarily close to a 22.5 deg
// multiple -- reproducibly, with no reliance on the corpus happening to
// contain a near-boundary sample. On that constructed SPD the claim can
// be demonstrated directly, and at several boundary distances, so the
// scaling is visible too.
//
// The family is D65 mixed with HP1 (high-pressure sodium). Blackbody CCT
// is the other obvious parameterisation and it was tried first, but it is
// degenerate for this purpose: a Planckian test source at T < 4000 K gets
// a Planckian reference at the same CCT (TM-30-20 S3.3 Eq. (14)), so test
// and reference colorimetry very nearly coincide. A bin flip then moves
// the test polygon and the reference polygon by the same amount, the Rg
// area ratio hardly notices, and every local metric is pinned near its
// perfect-rendering value. Measured that way the jump comes out around
// 8e-6 in Rg -- real, but it understates the phenomenon by five orders of
// magnitude purely because of the choice of family. A source far from its
// own reference is what makes the discontinuity visible.
//
// ---------------------------------------------------------------------
// WHAT IS ASSERTED, AND WHAT IS ONLY REPORTED
// ---------------------------------------------------------------------
// ASSERTED (true on any libm, because each is a statement about
// quantities this test measures on the machine it runs on):
//   * bisection reaches every requested boundary distance, down to
//     1e-7 deg -- near-boundary samples are constructible, not a
//     hypothetical;
//   * a perturbation whose measured hue shift is smaller than the
//     sample's boundary distance does NOT change its bin;
//   * a perturbation whose measured hue shift exceeds it DOES;
//   * a perturbation landing the sample exactly ON the boundary follows
//     the library's half-open intervals -- it stays in the bin that starts
//     at that boundary, so it is a flip for an upward crossing and not one
//     for a downward crossing (docs/divergences.md, "Hue angles exactly on
//     a bin boundary"). This case is reached, not hypothetical: the eps
//     bisection runs to adjacent doubles, and which side of the boundary
//     the last non-flipping perturbation lands on is a last-bit matter
//     that differs between the x86-64 and arm64 CI legs;
//   * the perturbation magnitude at that threshold scales down with the
//     boundary distance;
//   * across the threshold, Rf, CCT and Duv are unchanged to rounding
//     while Rg and the local per-bin metrics jump by an amount that does
//     not shrink with the boundary distance;
//   * corpus context: no corpus sample sits exactly on a boundary, a
//     float64 rounding-scale hue shift flips none of them, and max |dRg|
//     grows monotonically with the noise level.
//
// REPORTED, NEVER ASSERTED: the exact corpus flip counts and boundary
// percentiles. Those depend on the last few bits of atan2/exp/pow, and
// glibc, Apple libm and MSVC UCRT disagree there; CI runs all three
// (.github/workflows/ci.yml). Asserting "exactly 5 samples flip at 1%
// noise" would go red on one leg and green on another -- which is the
// very phenomenon this file documents, so asserting it would be
// self-defeating.
//
// ---------------------------------------------------------------------
// DETERMINISM
// ---------------------------------------------------------------------
// Both the bisections and the perturbation are deterministic. The
// perturbation is a +/-1 sign pattern from a self-contained LCG: the
// <random> distributions (std::normal_distribution,
// std::uniform_real_distribution, ...) are NOT specified to produce
// identical sequences across libstdc++, libc++ and MSVC, so they cannot
// be used where numbers must reproduce bit-for-bit. std::mt19937_64
// itself is specified but its consumers are not, so the whole family is
// avoided. The same LCG drives the Python study these numbers came from
// (downloads/perf-review/harness/binflip_study.py).
//
// Runtime: a few hundred single-SPD pipeline runs, well under 0.5 s.

#include <catch2/catch_test_macros.hpp>

#include "tm30/cct.hpp"
#include "tm30/ciecam02.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/gamut.hpp"
#include "tm30/hue_bins.hpp"
#include "tm30/pipeline.hpp"
#include "tm30/reference.hpp"
#include "tm30/resample.hpp"
#include "tm30/spd.hpp"
#include "tm30/tm30.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tm30::test {
namespace {

// =====================================================================
//  Loading helpers (same pattern as slice_08 / slice_13)
// =====================================================================

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
  const std::size_t n_ces = table.headers.size() - 1;
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
  std::vector<double> wl;
  std::vector<double> vals;
  for (const auto &row : table.rows) {
    wl.push_back(row[0]);
    vals.push_back(row[1]);
  }
  return {wl, vals};
}

/// Bundled corpus basenames, from data/illuminant_corpus.txt -- the same
/// single source of truth slice_13 and the benchmark scripts read.
std::vector<std::string> load_corpus_names() {
  std::ifstream in(data_path("illuminant_corpus.txt"));
  if (!in)
    throw std::runtime_error("cannot open data/illuminant_corpus.txt");
  std::vector<std::string> names;
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos)
      line.erase(hash);
    const std::size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
      continue;
    const std::size_t last = line.find_last_not_of(" \t\r\n");
    names.push_back(line.substr(first, last - first + 1));
  }
  if (names.empty())
    throw std::runtime_error("data/illuminant_corpus.txt lists no SPDs");
  return names;
}

/// Linear interpolation onto `grid`, matching numpy.interp exactly: out-
/// of-range queries clamp to the end values, and a query landing on a
/// source node reproduces that node bit-for-bit (numpy selects the
/// segment with wl[k] <= x, so the offset term is exactly zero). Matching
/// numpy matters because the Python study this file mirrors resamples the
/// same corpus with np.interp.
std::vector<double> interp_to(const std::vector<double> &grid,
                              const std::vector<double> &wl,
                              const std::vector<double> &vals) {
  std::vector<double> out;
  out.reserve(grid.size());
  for (const double x : grid) {
    if (x <= wl.front()) {
      out.push_back(vals.front());
      continue;
    }
    if (x >= wl.back()) {
      out.push_back(vals.back());
      continue;
    }
    std::size_t k = 0;
    while (k + 2 < wl.size() && wl[k + 1] <= x)
      ++k;
    const double slope = (vals[k + 1] - vals[k]) / (wl[k + 1] - wl[k]);
    out.push_back(slope * (x - wl[k]) + vals[k]);
  }
  return out;
}

// =====================================================================
//  Deterministic perturbation
// =====================================================================

/// Deterministic +/-1 sign pattern. Bit-identical on every platform,
/// unlike any <random> distribution (see the file banner). Identical to
/// lcg_signs() in downloads/perf-review/harness/binflip_study.py.
std::vector<double> lcg_signs(std::size_t n) {
  std::vector<double> out(n);
  std::uint64_t x = 0x2545F4914F6CDD1DULL; // study seed, not a spec value
  for (std::size_t i = 0; i < n; ++i) {
    // Knuth/MMIX LCG constants - a fully specified, self-contained PRNG.
    x = 6364136223846793005ULL * x + 1442695040888963407ULL;
    out[i] = (x >> 63) ? 1.0 : -1.0;
  }
  return out;
}

// =====================================================================
//  Hue-angle geometry. Diagnostics only: every bin assignment under test
//  comes from the library's own bin_by_hue() / CesColorimetryResult.
// =====================================================================

constexpr double kBinWidthDeg = 22.5; // TM-30-20 S4.3: 16 bins x 22.5 deg
constexpr int kNumBins = 16;          // TM-30-20 S4.3
constexpr std::size_t kNumCes = 99;   // TM-30-20 S4.0: 99 CES
constexpr double kRadToDeg = 180.0 / std::numbers::pi;

/// Reference hue angle in [0, 360), matching hue_bins.cpp's convention.
double hue_deg(const Cam02Ucs &j) {
  double h = std::atan2(j.b_prime, j.a_prime) * kRadToDeg; // TM-30-20 S4.3
  if (h < 0.0)
    h += 360.0; // TM-30-20 S4.3: map [-pi, pi] onto [0, 2pi)
  return h;
}

/// Distance from a hue angle to the nearest 22.5 deg boundary.
double dist_to_boundary_deg(double h) {
  const double pos = h / kBinWidthDeg;
  const double frac = pos - std::floor(pos);
  return std::min(frac, 1.0 - frac) * kBinWidthDeg;
}

/// Signed hue difference a - b, wrapped into (-180, 180].
double signed_hue_diff(double a, double b) {
  double d = a - b;
  while (d > 180.0)
    d -= 360.0;
  while (d <= -180.0)
    d += 360.0;
  return d;
}

double abs_hue_diff(double a, double b) {
  return std::abs(signed_hue_diff(a, b));
}

/// Flatten the library's HueBins into a per-CES bin index array.
std::array<int, kNumCes> flatten_bins(const HueBins &bins) {
  std::array<int, kNumCes> out{};
  for (int bin = 0; bin < kNumBins; ++bin)
    for (int idx : bins[bin])
      out[static_cast<std::size_t>(idx)] = bin;
  return out;
}

/// numpy.percentile with the default 'linear' interpolation, so printed
/// percentiles are directly comparable to the Python study's.
double percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  const double idx = q / 100.0 * static_cast<double>(v.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
  const double frac = idx - static_cast<double>(lo);
  return v[lo] + frac * (v[hi] - v[lo]);
}

double median_of(std::vector<double> v) { return percentile(std::move(v), 50); }

// =====================================================================
//  Pipeline harness: one set of tables, one grid, single-SPD evaluation.
// =====================================================================

class Harness {
public:
  static const Harness &instance() {
    static Harness h;
    return h;
  }

  const std::vector<double> &wl() const { return wl_; }

  std::vector<std::optional<Tm30Result>>
  eval(const std::vector<std::vector<double>> &matrix) const {
    std::vector<SpdView> views;
    views.reserve(matrix.size());
    for (const auto &row : matrix)
      views.push_back(
          SpdView{std::span<const double>(wl_), std::span<const double>(row)});
    return try_evaluate_cached(views, rtab_, lut_,
                               Tm30Request{/*bins=*/true, /*samples=*/true},
                               /*n_workers=*/1);
  }

  Tm30Result eval_one(const std::vector<double> &values) const {
    auto r = eval(std::vector<std::vector<double>>{values});
    if (r.size() != 1 || !r[0].has_value())
      throw std::runtime_error("slice 14: SPD failed to evaluate");
    return *r[0];
  }

  /// Planck's law with the CODATA radiation constants, peak-normalised to
  /// 100. Deliberately NOT tm30::generate_planckian(), which uses the
  /// rounded c2 = 1.4388e-2 of TM-30-20 S3.3 Eq. (6) because it builds
  /// the *reference* illuminant; these are *test* SPDs standing in for
  /// real sources, and they mirror the Python study's spectra exactly.
  std::vector<double> blackbody(double T) const {
    const double c1 = 3.741771852e-16; // CODATA first radiation constant
    const double c2 = 1.438776877e-2;  // CODATA second radiation constant
    std::vector<double> bb(wl_.size());
    double peak = 0.0;
    for (std::size_t i = 0; i < wl_.size(); ++i) {
      const double lam = wl_[i] * 1e-9; // nm -> m
      bb[i] = c1 / (std::pow(lam, 5.0) * (std::exp(c2 / (lam * T)) - 1.0));
      peak = std::max(peak, bb[i]);
    }
    for (auto &x : bb)
      x = x / peak * 100.0;
    return bb;
  }

  const std::vector<double> &signs() const { return signs_; }

  /// spd[i] * (1 + eps * dir * sign[i]) - the study's perturbation.
  std::vector<double> perturb(const std::vector<double> &spd, double eps,
                              double dir) const {
    std::vector<double> out(spd.size());
    for (std::size_t i = 0; i < spd.size(); ++i)
      out[i] = spd[i] * (1.0 + eps * dir * signs_[i]);
    return out;
  }

private:
  Harness() {
    wl_.reserve(401);
    for (int i = 0; i < 401; ++i)
      wl_.push_back(380.0 + static_cast<double>(i)); // 380-780 nm @ 1 nm

    cmf_2deg_ = load_cmf(data_path("cie_1931_2.csv"));
    cmf_10deg_ = load_cmf(data_path("cmf_1964_10.csv"));
    ces_ = load_ces(data_path("ces.csv"));
    basis_ = load_daylight_basis(data_path("daylight_basis.csv"));
    lut_ = load_planckian_lut(data_path("planckian_uv.csv"));
    rtab_ = prepare_resampled_tables(wl_, cmf_2deg_, cmf_10deg_, ces_, basis_);
    signs_ = lcg_signs(wl_.size());
  }

  std::vector<double> wl_;
  CmfData cmf_2deg_;
  CmfData cmf_10deg_;
  CesData ces_;
  DaylightBasis basis_;
  PlanckianLut lut_;
  ResampledTables rtab_;
  std::vector<double> signs_;
};

// =====================================================================
//  Part 1 (context): the bundled corpus + synthetic spectra
// =====================================================================

struct NoiseRow {
  double eps = 0.0;                  ///< relative SPD perturbation
  double median_hue_shift_deg = 0.0; ///< over all samples
  double max_hue_shift_deg = 0.0;
  int at_risk_global = 0; ///< boundary distance < corpus-wide max hue shift
  int at_risk_own = 0;    ///< boundary distance < this sample's own hue shift
  int flips = 0;          ///< bin assignments that actually changed
  double max_abs_drg = 0.0;
  std::size_t n_compared = 0;
};

class CorpusStudy {
public:
  static const CorpusStudy &instance() {
    static CorpusStudy s;
    return s;
  }

  const std::vector<double> &dists() const { return dists_; }
  const std::vector<NoiseRow> &rows() const { return rows_; }
  const std::vector<std::array<Cam02Ucs, kNumCes>> &jab() const { return jab_; }
  const std::vector<std::array<int, kNumCes>> &bins() const { return bins_; }
  std::size_t n_rows() const { return n_rows_; }
  std::size_t n_valid() const { return jab_.size(); }
  std::size_t n_samples() const { return jab_.size() * kNumCes; }

  /// Largest hue displacement an allowance of 78 ULP on a'/b' could
  /// produce anywhere in the corpus, in degrees. 78 is a deliberately
  /// generous stand-in for float64 rounding accumulated through the
  /// pipeline (the realistic figure is a couple of ULP); only the
  /// exponent matters here, and 78 makes the comparison harsher.
  double ulp_hue_shift_deg() const { return ulp_hue_shift_deg_; }

private:
  CorpusStudy() {
    const Harness &H = Harness::instance();
    build_corpus();

    auto base = H.eval(spds_);
    n_rows_ = base.size();
    for (const auto &r : base) {
      if (!r.has_value())
        continue;
      const auto &c = r->colorimetry;
      jab_.push_back(c.jab_ref_ces);
      bins_.push_back(flatten_bins(c.hue_bins));
      rg_.push_back(c.gamut.Rg);
      for (std::size_t i = 0; i < kNumCes; ++i)
        hue0_.push_back(hue_deg(c.jab_ref_ces[i]));
    }

    dists_.reserve(hue0_.size());
    for (const double h : hue0_)
      dists_.push_back(dist_to_boundary_deg(h));

    // float64 rounding scale: spacing of the largest |a'| in the corpus,
    // times the 78 ULP allowance, over each sample's chroma radius.
    double a_max = 0.0;
    for (const auto &row : jab_)
      for (const auto &j : row)
        a_max = std::max(a_max, std::abs(j.a_prime));
    const double spacing =
        std::nextafter(a_max, std::numeric_limits<double>::infinity()) - a_max;
    for (const auto &row : jab_)
      for (const auto &j : row) {
        const double radius = std::hypot(j.a_prime, j.b_prime);
        ulp_hue_shift_deg_ =
            std::max(ulp_hue_shift_deg_, (78.0 * spacing / radius) * kRadToDeg);
      }

    for (const double eps : {1e-6, 1e-5, 1e-4, 1e-3, 1e-2})
      rows_.push_back(sweep(eps));
  }

  NoiseRow sweep(double eps) {
    const Harness &H = Harness::instance();
    std::vector<std::vector<double>> pert;
    pert.reserve(spds_.size());
    for (const auto &row : spds_)
      pert.push_back(H.perturb(row, eps, +1.0));

    auto res = H.eval(pert);
    NoiseRow nr;
    nr.eps = eps;
    std::vector<double> shifts;
    std::vector<double> per_sample_dist;
    shifts.reserve(n_samples());
    std::size_t v = 0;
    for (const auto &r : res) {
      if (!r.has_value() || v >= jab_.size())
        continue;
      const auto &c = r->colorimetry;
      const auto b1 = flatten_bins(c.hue_bins);
      for (std::size_t i = 0; i < kNumCes; ++i) {
        const double h1 = hue_deg(c.jab_ref_ces[i]);
        const std::size_t k = v * kNumCes + i;
        shifts.push_back(abs_hue_diff(h1, hue0_[k]));
        per_sample_dist.push_back(dists_[k]);
        if (b1[i] != bins_[v][i])
          ++nr.flips;
      }
      nr.max_abs_drg = std::max(nr.max_abs_drg, std::abs(c.gamut.Rg - rg_[v]));
      ++v;
    }
    nr.n_compared = v;
    nr.median_hue_shift_deg = median_of(shifts);
    nr.max_hue_shift_deg = *std::max_element(shifts.begin(), shifts.end());
    for (std::size_t k = 0; k < shifts.size(); ++k) {
      if (per_sample_dist[k] < nr.max_hue_shift_deg)
        ++nr.at_risk_global;
      if (per_sample_dist[k] < shifts[k])
        ++nr.at_risk_own;
    }
    return nr;
  }

  void build_corpus() {
    const Harness &H = Harness::instance();
    const auto &wl = H.wl();

    // 19 bundled standard illuminants, resampled onto the 1 nm grid.
    for (const auto &name : load_corpus_names()) {
      auto [w, v] = load_spd_csv(data_path(name + ".csv"));
      spds_.push_back(interp_to(wl, w, v));
    }
    // 7 blackbodies.
    for (const double T :
         {2000.0, 2856.0, 3500.0, 4000.0, 5000.0, 6500.0, 10000.0})
      spds_.push_back(H.blackbody(T));
    // 3 narrowband gaussians, sigma 8 nm.
    for (const double center : {450.0, 550.0, 630.0}) {
      std::vector<double> g(wl.size());
      for (std::size_t i = 0; i < wl.size(); ++i) {
        const double z = (wl[i] - center) / 8.0; // sigma = 8 nm
        g[i] = std::exp(-0.5 * z * z) * 100.0;
      }
      spds_.push_back(std::move(g));
    }
    // 1 equal-energy.
    spds_.push_back(std::vector<double>(wl.size(), 100.0));
  }

  std::vector<std::vector<double>> spds_;
  std::size_t n_rows_ = 0;
  std::vector<std::array<Cam02Ucs, kNumCes>> jab_;
  std::vector<std::array<int, kNumCes>> bins_;
  std::vector<double> rg_;
  std::vector<double> hue0_;
  std::vector<double> dists_;
  double ulp_hue_shift_deg_ = 0.0;
  std::vector<NoiseRow> rows_;
};

// =====================================================================
//  Part 2 (the evidence): constructed near-boundary samples
// =====================================================================

/// One constructed case: a D65/HP1 mixture whose CES `ces` sits
/// `dist_deg` from the 22.5 deg boundary `boundary_deg`, plus the two
/// adjacent-double SPD perturbations that straddle its bin flip.
struct EdgeCase {
  double target_deg = 0.0; ///< requested boundary distance
  double mix_t = 0.0;      ///< mixing fraction: (1-t) x D65 + t x HP1
  std::size_t ces = 0;
  double boundary_deg = 0.0;
  double dist_deg = 0.0; ///< achieved boundary distance
  /// boundary_deg - hue, wrapped: negative when the sample sits ABOVE its
  /// boundary and the crossing therefore runs downward. Decides which side
  /// of the flip inequality is the non-strict one - see the flip-threshold
  /// test case.
  double signed_to_edge = 0.0;
  int bin_before = -1;
  int bin_after = -1;
  double dir = 1.0; ///< LCG sign multiplier that moves hue toward the edge

  double eps_no_flip = 0.0; ///< largest perturbation that keeps the bin
  double eps_flip = 0.0;    ///< next double up; changes the bin
  double shift_no_flip_deg = 0.0;
  double shift_flip_deg = 0.0;

  // Metric deltas across the two adjacent-double perturbations, i.e.
  // across the flip and nothing else.
  double d_rg = 0.0;
  double d_rf = 0.0;
  double d_cct = 0.0;
  double d_duv = 0.0;
  double d_rf_hj_before = 0.0, d_rf_hj_after = 0.0;
  double d_rcs_hj_before = 0.0, d_rcs_hj_after = 0.0;
  double d_rhs_hj_before = 0.0, d_rhs_hj_after = 0.0;

  // What the constructed source actually is, for context.
  double base_cct = 0.0, base_rf = 0.0, base_rg = 0.0;

  double max_abs_d_rf_hj() const {
    return std::max(std::abs(d_rf_hj_before), std::abs(d_rf_hj_after));
  }
  /// The largest of the quantities that must NOT jump.
  double max_abs_continuous() const {
    return std::max({std::abs(d_rf), std::abs(d_cct), std::abs(d_duv)});
  }
};

class EdgeCases {
public:
  static const EdgeCases &instance() {
    static EdgeCases e;
    return e;
  }

  const std::vector<EdgeCase> &cases() const { return cases_; }
  std::size_t scan_evals() const { return scan_evals_; }

private:
  struct Probe {
    double hue = 0.0;
    int bin = -1;
  };

  /// The one-parameter family: (1-t) x D65 + t x HP1, both peak-
  /// normalised to 100 first so t is a meaningful mixing fraction.
  std::vector<double> mix(double t) const {
    std::vector<double> out(a_.size());
    for (std::size_t i = 0; i < a_.size(); ++i)
      out[i] = (1.0 - t) * a_[i] + t * b_[i];
    return out;
  }

  Probe probe(double t, std::size_t ces) const {
    const Tm30Result r = Harness::instance().eval_one(mix(t));
    Probe p;
    p.hue = hue_deg(r.colorimetry.jab_ref_ces[ces]);
    p.bin = flatten_bins(r.colorimetry.hue_bins)[ces];
    return p;
  }

  static std::vector<double> load_normalised(const std::string &name) {
    auto [w, v] = load_spd_csv(data_path(name + ".csv"));
    std::vector<double> s = interp_to(Harness::instance().wl(), w, v);
    const double peak = *std::max_element(s.begin(), s.end());
    for (auto &x : s)
      x = x / peak * 100.0;
    return s;
  }

  EdgeCases() {
    const Harness &H = Harness::instance();
    a_ = load_normalised("d65_1nm");
    b_ = load_normalised("hp1_5nm");

    // -- 1. Find the strongest bin crossing in the family -------------
    // Scan t on a fixed 201-point grid, then take the consecutive pair
    // across which exactly ONE CES changes bin and Rg moves the most.
    // Requiring a single crossing isolates one flip; picking the largest
    // Rg step picks a sample whose bin membership actually matters, so
    // the measured jump is representative rather than incidental. Both
    // criteria are deterministic - no search over random starts.
    const std::size_t n_scan = 201;
    std::vector<double> ts;
    std::vector<std::array<int, kNumCes>> scan_bins;
    std::vector<double> scan_rg;
    for (std::size_t k = 0; k < n_scan; ++k) {
      const double t = static_cast<double>(k) / static_cast<double>(n_scan - 1);
      const Tm30Result r = H.eval_one(mix(t));
      ts.push_back(t);
      scan_bins.push_back(flatten_bins(r.colorimetry.hue_bins));
      scan_rg.push_back(r.colorimetry.gamut.Rg);
      ++scan_evals_;
    }

    double t_lo = 0.0, t_hi = 0.0, best_step = -1.0;
    std::size_t ces = 0;
    for (std::size_t k = 0; k + 1 < ts.size(); ++k) {
      std::size_t changed = 0;
      std::size_t which = 0;
      for (std::size_t i = 0; i < kNumCes; ++i)
        if (scan_bins[k][i] != scan_bins[k + 1][i]) {
          ++changed;
          which = i;
        }
      if (changed != 1)
        continue;
      const double step = std::abs(scan_rg[k + 1] - scan_rg[k]);
      if (step > best_step) {
        best_step = step;
        t_lo = ts[k];
        t_hi = ts[k + 1];
        ces = which;
      }
    }
    if (best_step < 0.0)
      throw std::runtime_error(
          "slice 14: no isolated hue-bin crossing found in the D65/HP1 family");

    // -- 2. Bisect t toward the crossing, snapshotting each target ----
    // Every halving of the bracket roughly halves the chosen sample's
    // distance to the boundary, so one descent serves all targets.
    const std::vector<double> targets = {1e-3, 1e-5, 1e-7};
    std::size_t next_target = 0;
    Probe p_lo = probe(t_lo, ces);
    Probe p_hi = probe(t_hi, ces);
    scan_evals_ += 2;
    const int bin0 = p_lo.bin;

    for (int it = 0; it < 200 && next_target < targets.size(); ++it) {
      while (next_target < targets.size() &&
             dist_to_boundary_deg(p_lo.hue) <= targets[next_target]) {
        cases_.push_back(
            make_case(targets[next_target], t_lo, ces, p_lo, p_hi, bin0));
        ++next_target;
      }
      if (next_target >= targets.size())
        break;
      const double t_mid = 0.5 * (t_lo + t_hi);
      if (t_mid <= t_lo || t_mid >= t_hi)
        break; // adjacent doubles: cannot refine further
      const Probe p_mid = probe(t_mid, ces);
      ++scan_evals_;
      if (p_mid.bin == bin0) {
        t_lo = t_mid;
        p_lo = p_mid;
      } else {
        t_hi = t_mid;
        p_hi = p_mid;
      }
    }
  }

  EdgeCase make_case(double target, double t, std::size_t ces,
                     const Probe &p_lo, const Probe &p_hi, int bin0) {
    const Harness &H = Harness::instance();
    EdgeCase c;
    c.target_deg = target;
    c.mix_t = t;
    c.ces = ces;
    c.bin_before = bin0;
    c.bin_after = p_hi.bin;

    // The crossed boundary is the 22.5 deg multiple nearest the sample.
    c.boundary_deg = kBinWidthDeg * std::round(p_lo.hue / kBinWidthDeg);
    const double signed_to_edge = signed_hue_diff(c.boundary_deg, p_lo.hue);
    c.signed_to_edge = signed_to_edge;
    c.dist_deg = std::abs(signed_to_edge);

    const std::vector<double> spd = mix(t);

    // -- Which LCG sign direction moves the hue toward the boundary? --
    const double eps_probe = 1e-4;
    double response = 0.0; // deg of hue shift per unit relative perturbation
    for (const double dir : {+1.0, -1.0}) {
      const Tm30Result r = H.eval_one(H.perturb(spd, eps_probe, dir));
      const double dh =
          signed_hue_diff(hue_deg(r.colorimetry.jab_ref_ces[ces]), p_lo.hue);
      // Strict: a zero shift is not "the right direction".
      if (dh != 0.0 && (dh > 0.0) == (signed_to_edge > 0.0)) {
        c.dir = dir;
        response = std::abs(dh) / eps_probe;
      }
    }
    if (response <= 0.0)
      throw std::runtime_error("slice 14: no perturbation direction moves "
                               "the sample toward its bin boundary");

    // -- Bracket the flip, then bisect to adjacent doubles ------------
    // Linear response gives the first guess; grow until the bin actually
    // changes (bounded, so a mis-estimate cannot loop forever).
    double hi = std::max(4.0 * c.dist_deg / response, 1e-300);
    int grow = 0;
    while (!flips(spd, hi, c.dir, ces, bin0) && hi < 5e-2 && grow < 40) {
      hi *= 4.0;
      ++grow;
    }
    if (!flips(spd, hi, c.dir, ces, bin0))
      throw std::runtime_error("slice 14: could not bracket the bin flip");

    double lo = hi * 1e-9;
    int shrink = 0;
    while (flips(spd, lo, c.dir, ces, bin0) && shrink < 40) {
      lo *= 1e-3;
      ++shrink;
    }

    for (int it = 0; it < 400; ++it) {
      const double mid = 0.5 * (lo + hi);
      if (mid <= lo || mid >= hi)
        break; // lo and hi are now adjacent doubles
      if (flips(spd, mid, c.dir, ces, bin0))
        hi = mid;
      else
        lo = mid;
    }
    c.eps_no_flip = lo;
    c.eps_flip = hi;

    // -- Everything below is measured across those two adjacent-double
    //    perturbations, so the ONLY material difference between the two
    //    evaluations is the bin the sample landed in.
    const Tm30Result r_lo = H.eval_one(H.perturb(spd, lo, c.dir));
    const Tm30Result r_hi = H.eval_one(H.perturb(spd, hi, c.dir));
    const auto &a = r_lo.colorimetry;
    const auto &b = r_hi.colorimetry;

    c.shift_no_flip_deg =
        std::abs(signed_hue_diff(hue_deg(a.jab_ref_ces[ces]), p_lo.hue));
    c.shift_flip_deg =
        std::abs(signed_hue_diff(hue_deg(b.jab_ref_ces[ces]), p_lo.hue));
    c.bin_before = flatten_bins(a.hue_bins)[ces];
    c.bin_after = flatten_bins(b.hue_bins)[ces];

    c.d_rg = b.gamut.Rg - a.gamut.Rg;
    c.d_rf = b.Rf - a.Rf;
    c.d_cct = b.cct - a.cct;
    c.d_duv = b.duv - a.duv;

    const std::size_t jb = static_cast<std::size_t>(c.bin_before);
    const std::size_t ja = static_cast<std::size_t>(c.bin_after);
    c.d_rf_hj_before = b.gamut.local.Rf_hj[jb] - a.gamut.local.Rf_hj[jb];
    c.d_rf_hj_after = b.gamut.local.Rf_hj[ja] - a.gamut.local.Rf_hj[ja];
    c.d_rcs_hj_before =
        b.gamut.local.Rcs_hj_percent[jb] - a.gamut.local.Rcs_hj_percent[jb];
    c.d_rcs_hj_after =
        b.gamut.local.Rcs_hj_percent[ja] - a.gamut.local.Rcs_hj_percent[ja];
    c.d_rhs_hj_before = b.gamut.local.Rhs_hj[jb] - a.gamut.local.Rhs_hj[jb];
    c.d_rhs_hj_after = b.gamut.local.Rhs_hj[ja] - a.gamut.local.Rhs_hj[ja];

    const Tm30Result r0 = H.eval_one(spd);
    c.base_cct = r0.colorimetry.cct;
    c.base_rf = r0.colorimetry.Rf;
    c.base_rg = r0.colorimetry.gamut.Rg;
    return c;
  }

  bool flips(const std::vector<double> &spd, double eps, double dir,
             std::size_t ces, int bin0) const {
    const Harness &H = Harness::instance();
    const Tm30Result r = H.eval_one(H.perturb(spd, eps, dir));
    return flatten_bins(r.colorimetry.hue_bins)[ces] != bin0;
  }

  std::vector<double> a_; ///< D65, peak-normalised
  std::vector<double> b_; ///< HP1, peak-normalised
  std::vector<EdgeCase> cases_;
  std::size_t scan_evals_ = 0;
};

// =====================================================================
//  Formatting helpers
// =====================================================================

std::string sci(double v, int prec = 3) {
  std::ostringstream o;
  o << std::scientific << std::setprecision(prec) << v;
  return o.str();
}

std::string fixed(double v, int prec = 4) {
  std::ostringstream o;
  o << std::fixed << std::setprecision(prec) << v;
  return o.str();
}

// =====================================================================
//  Tests - the constructed evidence
// =====================================================================

TEST_CASE("hue-bin stability - bisection constructs samples arbitrarily "
          "close to a bin boundary",
          "[hue_bins][slice14][constructed]") {
  const EdgeCases &E = EdgeCases::instance();
  REQUIRE(E.cases().size() == 3);

  std::ostringstream o;
  o << "\nconstructed near-boundary samples\n"
    << "(family: (1-t) x D65 + t x HP1, both peak-normalised; bisecting on\n"
    << " t drives one CES's reference hue angle onto a 22.5 deg boundary.\n"
    << " Bins are printed 1-indexed, as TM-30-20 numbers them.)\n";
  for (const auto &c : E.cases()) {
    o << "  target " << sci(c.target_deg, 0)
      << " deg  ->  t = " << fixed(c.mix_t, 17) << "\n"
      << "      CES " << (c.ces + 1) << " (index " << c.ces << "), boundary "
      << fixed(c.boundary_deg, 1) << " deg, distance " << sci(c.dist_deg)
      << " deg, bin h" << (c.bin_before + 1) << " -> h" << (c.bin_after + 1)
      << "\n"
      << "      source: CCT " << fixed(c.base_cct, 1) << " K, Rf "
      << fixed(c.base_rf, 2) << ", Rg " << fixed(c.base_rg, 2) << "\n";
  }
  o << "  (" << E.scan_evals()
    << " pipeline runs for the t-scan and its bisection)\n";
  std::cout << o.str() << std::flush;

  // ASSERTED: the construction actually reaches every requested distance,
  // and the samples are genuinely inside a bin, not on the boundary.
  for (const auto &c : E.cases()) {
    INFO("target " << c.target_deg << " deg");
    CHECK(c.dist_deg > 0.0);
    CHECK(c.dist_deg <= c.target_deg);
    CHECK(c.bin_before != c.bin_after);
  }
}

TEST_CASE("hue-bin stability - the perturbation needed to flip a bin "
          "scales with the boundary distance",
          "[hue_bins][slice14][constructed]") {
  const EdgeCases &E = EdgeCases::instance();
  REQUIRE(E.cases().size() == 3);

  std::ostringstream o;
  o << "\nflip threshold vs boundary distance\n"
    << "(relative SPD perturbation, deterministic LCG +/-1 sign pattern;\n"
    << " eps_no_flip and eps_flip are adjacent doubles straddling the flip)\n";
  for (const auto &c : E.cases()) {
    o << "  distance " << sci(c.dist_deg) << " deg  ->  eps* "
      << sci(c.eps_no_flip) << " (" << sci(c.eps_no_flip * 100.0, 2)
      << "% noise)\n"
      << "      below: hue shift "
      << fixed(c.shift_no_flip_deg / c.dist_deg, 12) << " x distance  -> bin h"
      << (c.bin_before + 1) << " (unchanged)\n"
      << "      above: hue shift " << fixed(c.shift_flip_deg / c.dist_deg, 12)
      << " x distance  -> bin h" << (c.bin_after + 1) << " (flipped)\n";
  }
  std::cout << o.str() << std::flush;

  // ASSERTED: the bin changes exactly when the measured hue shift crosses
  // the measured boundary distance. Both quantities are measured on the
  // machine running the test, so no libm agreement is required.
  //
  // Which side of that crossing owns the boundary itself is the library's
  // choice, not the standard's (docs/divergences.md, "Hue angles exactly on
  // a bin boundary"): bin assignment truncates, so the intervals are
  // [lo, hi) and a sample landing exactly ON a boundary stays in the bin
  // that starts there. For a downward crossing -- bin_before is the upper
  // bin, signed_to_edge < 0 -- a hue shift exactly equal to the boundary
  // distance is therefore not yet a flip, and the no-flip inequality is the
  // non-strict one; an upward crossing is the mirror image.
  //
  // That equality is reached in practice, not just in principle. The eps
  // bisection runs to adjacent doubles, so the last non-flipping
  // perturbation lands within one ULP of the boundary: on a 1e-7 deg
  // boundary distance the CI x86-64 legs land exactly on it (hue shift
  // 1.000000000000 x distance) while the arm64 legs land one ULP short.
  // Asserting the strict inequality on both sides fails on the former.
  for (const auto &c : E.cases()) {
    INFO("distance " << c.dist_deg << " deg");
    if (c.signed_to_edge < 0.0) {
      CHECK(c.shift_no_flip_deg <= c.dist_deg); // reached it at most
      CHECK(c.shift_flip_deg > c.dist_deg);     // passed it
    } else {
      CHECK(c.shift_no_flip_deg < c.dist_deg); // did not reach the boundary
      CHECK(c.shift_flip_deg >= c.dist_deg);   // reached or passed it
    }
    CHECK(c.eps_no_flip > 0.0);
    CHECK(c.eps_flip > c.eps_no_flip);
  }

  // ASSERTED: arbitrarily small perturbations suffice - the threshold
  // shrinks with the boundary distance. Targets are 100x apart, so a
  // loose factor-of-10 bound is enough and holds under any libm.
  for (std::size_t i = 1; i < E.cases().size(); ++i) {
    const auto &prev = E.cases()[i - 1];
    const auto &cur = E.cases()[i];
    INFO("eps* " << prev.eps_no_flip << " -> " << cur.eps_no_flip);
    CHECK(cur.dist_deg < prev.dist_deg);
    CHECK(cur.eps_no_flip < prev.eps_no_flip / 10.0);
  }
}

TEST_CASE("hue-bin stability - a flip moves Rg and the local per-bin "
          "metrics but leaves Rf, CCT and Duv alone",
          "[hue_bins][slice14][constructed]") {
  const EdgeCases &E = EdgeCases::instance();
  REQUIRE(E.cases().size() == 3);

  std::ostringstream o;
  o << "\nwhat changes when the bin flips\n"
    << "(deltas between two ADJACENT-DOUBLE perturbations, so the only\n"
    << " material difference between the two runs is the bin assignment)\n";
  for (const auto &c : E.cases()) {
    o << "  distance " << sci(c.dist_deg) << " deg, CES " << (c.ces + 1)
      << ": bin h" << (c.bin_before + 1) << " -> h" << (c.bin_after + 1) << "\n"
      << "      discontinuous:  dRg " << sci(c.d_rg) << "   dRf,h"
      << (c.bin_before + 1) << " " << sci(c.d_rf_hj_before) << "   dRf,h"
      << (c.bin_after + 1) << " " << sci(c.d_rf_hj_after) << "\n"
      << "                      dRcs,h" << (c.bin_before + 1) << " "
      << sci(c.d_rcs_hj_before) << "   dRcs,h" << (c.bin_after + 1) << " "
      << sci(c.d_rcs_hj_after) << "\n"
      << "                      dRhs,h" << (c.bin_before + 1) << " "
      << sci(c.d_rhs_hj_before) << "   dRhs,h" << (c.bin_after + 1) << " "
      << sci(c.d_rhs_hj_after) << "\n"
      << "      continuous:     dRf " << sci(c.d_rf) << "   dCCT "
      << sci(c.d_cct) << " K   dDuv " << sci(c.d_duv) << "\n";
  }
  std::cout << o.str() << std::flush;

  for (const auto &c : E.cases()) {
    INFO("distance " << c.dist_deg << " deg");

    // ASSERTED: Rf, CCT and Duv never consult a bin (TM-30-20 S3.3,
    // S4.1), so across a one-ULP change in the perturbation they move
    // only by rounding.
    CHECK(std::abs(c.d_rf) < 1e-6);
    CHECK(std::abs(c.d_cct) < 1e-6);
    CHECK(std::abs(c.d_duv) < 1e-9);

    // ASSERTED: Rg and the local per-bin metrics (TM-30-20 S4.4, S4.6-
    // S4.8) jump by a finite amount. The absolute floors are loose (the
    // measured jumps are ~70x and ~350x above them) so they survive a
    // libm that steers the construction to a neighbouring crossing.
    CHECK(std::abs(c.d_rg) > 1e-4);
    CHECK(c.max_abs_d_rf_hj() > 1e-2);
    CHECK(std::abs(c.d_rcs_hj_before) > 1e-3);
    CHECK(std::abs(c.d_rhs_hj_before) > 1e-4);

    // ASSERTED, and this is the crux: the jump is not a scaled-down
    // version of the perturbation. Two SPDs one ULP apart move Rg by
    // orders of magnitude more than they move Rf.
    CHECK(std::abs(c.d_rg) > 1.0e4 * c.max_abs_continuous());
  }

  // ASSERTED: the jump does not decay as the constructed sample is
  // pushed closer to the boundary. The last case sits ~3500x closer to
  // the boundary than the first, yet moves Rg by the same amount - which
  // is what "discontinuous" means, as opposed to merely steep.
  const double first = std::abs(E.cases().front().d_rg);
  const double last = std::abs(E.cases().back().d_rg);
  INFO("|dRg| " << first << " at " << E.cases().front().dist_deg << " deg vs "
                << last << " at " << E.cases().back().dist_deg << " deg");
  CHECK(last > 0.5 * first);
  CHECK(last < 2.0 * first);
}

// =====================================================================
//  Tests - corpus context
// =====================================================================

TEST_CASE("hue-bin stability - corpus context: how close real spectra sit "
          "to a bin boundary",
          "[hue_bins][slice14][corpus]") {
  const CorpusStudy &s = CorpusStudy::instance();
  REQUIRE(s.n_rows() == 30);  // 19 bundled + 7 blackbody + 3 narrowband + 1 EE
  REQUIRE(s.n_valid() == 30); // every corpus SPD evaluated successfully
  REQUIRE(s.n_samples() == 2970);

  const auto &d = s.dists();
  const double dmin = *std::min_element(d.begin(), d.end());

  std::ostringstream o;
  o << "\ncorpus context: " << s.n_valid() << " valid SPDs, " << s.n_samples()
    << " CES samples\n"
    << "distance to nearest " << kBinWidthDeg
    << " deg bin boundary (reported, not asserted):\n"
    << "  minimum    " << fixed(dmin) << " deg\n"
    << "  1st pct    " << fixed(percentile(d, 1)) << " deg\n"
    << "  25th pct   " << fixed(percentile(d, 25)) << " deg\n"
    << "  50th pct   " << fixed(percentile(d, 50)) << " deg\n"
    << "  (11.2500 deg is the half-width; 5.6250 deg the uniform median, so\n"
    << "   most samples sit far from any edge - which is exactly why a raw\n"
    << "   corpus flip count is a diluted statistic, and why the evidence\n"
    << "   proper lives in the 'hue-bin stability - ... constructed' cases)\n";
  std::cout << o.str() << std::flush;

  // ASSERTED: every sample's bin is well defined - none sits exactly on a
  // boundary, where the half-open rule of TM-30-20 S4.3 would be decided
  // by the sign of a rounding error.
  CHECK(dmin > 0.0);
}

TEST_CASE("hue-bin stability - corpus context: SPD noise, samples at risk "
          "and samples that flipped",
          "[hue_bins][slice14][corpus]") {
  const CorpusStudy &s = CorpusStudy::instance();
  const auto &rows = s.rows();
  REQUIRE(rows.size() == 5);

  std::ostringstream o;
  o << "\ncorpus context: relative SPD perturbation, full pipeline re-run\n"
    << "(at-risk = boundary distance below the hue shift actually applied;\n"
    << " flip counts are REPORTED, not asserted - they are libm-dependent)\n";
  for (const auto &r : rows) {
    o << "  " << fixed(r.eps * 100.0) << "% noise -> hue shift median "
      << sci(r.median_hue_shift_deg, 2) << " deg, max "
      << sci(r.max_hue_shift_deg, 2) << " deg\n"
      << "      at risk " << r.at_risk_own << "/" << s.n_samples()
      << " (own shift), " << r.at_risk_global << "/" << s.n_samples()
      << " (corpus max shift)  ->  " << r.flips << " flipped";
    if (r.at_risk_own == 0 && r.at_risk_global == 0)
      o << "\n      [no sample could reach a boundary at this scale: the flip"
           " count says\n       nothing about robustness]";
    else if (r.at_risk_own == 0)
      o << "\n      [no sample's own shift reached its boundary; the "
           "corpus-max column is\n       an upper bound over samples that "
           "moved less than the worst case]";
    o << "\n      max |dRg| " << sci(r.max_abs_drg) << "\n";
  }
  std::cout << o.str() << std::flush;

  for (const auto &r : rows) {
    INFO("noise level " << r.eps);
    CHECK(r.n_compared == s.n_valid()); // perturbation must not drop rows
    // ASSERTED: a sample can only flip if its own hue shift reached the
    // boundary. This is what makes the at-risk column the meaningful
    // denominator rather than 2970.
    CHECK(r.flips <= r.at_risk_own);
  }

  // ASSERTED: dRg is linear in the noise everywhere, including the row
  // where flips appear - a flipped sample was at a bin edge to begin
  // with, so it barely moves that bin's mean. Monotonicity is the robust
  // form of that claim.
  for (std::size_t i = 1; i < rows.size(); ++i) {
    INFO("|dRg| at eps=" << rows[i].eps << " vs eps=" << rows[i - 1].eps);
    CHECK(rows[i].max_abs_drg > rows[i - 1].max_abs_drg);
    CHECK(rows[i].max_hue_shift_deg > rows[i - 1].max_hue_shift_deg);
  }
}

TEST_CASE("hue-bin stability - corpus context: a float64 rounding-scale "
          "hue shift flips no bins",
          "[hue_bins][slice14][corpus]") {
  const CorpusStudy &s = CorpusStudy::instance();
  const double ulp_deg = s.ulp_hue_shift_deg();
  const auto &d = s.dists();
  const double dmin = *std::min_element(d.begin(), d.end());

  // Rotate every corpus sample by the LARGEST hue displacement 78 ULP on
  // a'/b' could produce anywhere in the corpus - a uniform rotation that
  // is conservative for every individual sample - and re-bin through the
  // library's own bin_by_hue(). Both directions: a boundary can lie on
  // either side.
  const double ulp_rad = ulp_deg / kRadToDeg;
  int flips = 0;
  for (const double dir : {+1.0, -1.0}) {
    const double ca = std::cos(dir * ulp_rad);
    const double sa = std::sin(dir * ulp_rad);
    for (std::size_t r = 0; r < s.jab().size(); ++r) {
      std::array<Cam02Ucs, kNumCes> rot = s.jab()[r];
      for (auto &j : rot) {
        const double a = j.a_prime;
        const double b = j.b_prime;
        j.a_prime = a * ca - b * sa;
        j.b_prime = a * sa + b * ca;
      }
      const auto rb = flatten_bins(bin_by_hue(rot));
      for (std::size_t i = 0; i < kNumCes; ++i)
        if (rb[i] != s.bins()[r][i])
          ++flips;
    }
  }

  std::ostringstream o;
  o << "\ncorpus context: float64 rounding scale\n"
    << "  78 ULP on a'/b'  ->  max hue shift " << sci(ulp_deg, 2)
    << " deg  ->  " << flips << " flips\n"
    << "  closest corpus sample is " << fixed(dmin) << " deg away, "
    << sci(dmin / ulp_deg, 1) << "x further\n";
  std::cout << o.str() << std::flush;

  // ASSERTED: rounding-scale hue error cannot move any corpus sample
  // across a boundary, and the margin is enormous. The bound is
  // deliberately loose (1e6 against the ~3e8 observed) so it holds under
  // any libm. Measurement uncertainty, not arithmetic, is what puts a
  // sample at risk.
  CHECK(ulp_deg > 0.0);
  CHECK(flips == 0);
  CHECK(dmin / ulp_deg > 1.0e6);
}

} // namespace
} // namespace tm30::test
