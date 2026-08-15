// TM-30-20 lazy memoized handle + batch API implementation.
#include "tm30/tm30.hpp"

#include "tm30/errors.hpp"
#include "tm30/pipeline.hpp"

#include <algorithm> // std::min
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility> // std::move
#include <vector>

namespace tm30 {

// ==========================================================================
//  Validity thresholds - pytm30 implementation choice (NOT in TM-30-20 text).
//
//  TM-30-20 S2.0 Scope: the method "is best suited to characterize nominally
//  white light sources (i.e., those that fall on or near the Planckian
//  locus)"; sources whose chromaticity "falls outside of the chromaticity
//  bins defined in ANSI C78.377-2017... should be interpreted with caution."
//  The standard states the domain qualitatively; it does NOT print numerical
//  CCT bounds or a Duv threshold anywhere. The constants below are pytm30's
//  own sanity bounds -- whitelisted in tools/check_constants_whitelist.txt
//  because no honest "Sx.y" citation exists.
// ==========================================================================

namespace {
// pytm30 impl choice (NOT in TM-30-20 text). See banner above.
constexpr double kCctMin = 1000.0;
constexpr double kCctMax = 25000.0;

// pytm30 impl choice (NOT in TM-30-20 text). See banner above.
constexpr double kDuvMaxAbs = 0.05;

// TM-30-20 S3.5: Full CES wavelength range is 380-780 nm.
// If the test SPD does not cover this range, extrapolation is needed.
} // namespace

// ==========================================================================
//  Validity computation
// ==========================================================================

static Validity compute_validity(const CesColorimetryResult &cr,
                                 const Spd &spd) {
  Validity v;

  // Advisory: CCT far from the range where TM-30 is typically applied.
  // pytm30 impl choice; TM-30-20 S2.0 gives no numerical bounds.
  v.cct_out_of_range = (cr.cct < kCctMin || cr.cct > kCctMax);

  // Advisory: chromaticity far from the Planckian locus. pytm30 impl choice;
  // TM-30-20 S2.0 states the domain qualitatively (near-locus, ANSI
  // C78.377-2017 white bins), no numerical Duv threshold.
  v.duv_out_of_range = (std::abs(cr.duv) > kDuvMaxAbs);

  // Test SPD did not cover the full 380-780 nm grid; the missing edge
  // values were zero-filled at Spd construction per TM-30-20 S3.5, which
  // requires zeros for missing values and forbids interpolating or
  // extrapolating the test SPD. Flat-extrapolation of CES/CMF reference
  // tables from 400-700 to 380-780 is a separate provision (TM-30-20
  // S1.3 / Annex A) and is unrelated to this flag.
  v.extrapolated = spd.zero_filled();

  return v;
}

// ==========================================================================
//  Tm30 constructor
// ==========================================================================

Tm30::Tm30(Spd spd, const CmfData &cmf_2deg, const CmfData &cmf_10deg,
           const CesData &ces_data, const DaylightBasis &daylight_basis,
           const PlanckianLut &planckian_lut)
    : spd_(std::move(spd)), cmf_2deg_(cmf_2deg), cmf_10deg_(cmf_10deg),
      ces_data_(ces_data), daylight_basis_(daylight_basis),
      planckian_lut_(planckian_lut) {
  // SPD already validated by Spd constructor.
  // No computation performed here - fully lazy.
}

// ==========================================================================
//  ensure_computed - run the full pipeline if not yet cached
// ==========================================================================

void Tm30::ensure_computed() const {
  if (computed_)
    return;

  // Run the full CES colorimetry pipeline.
  // TM-30-20 S3.4, S3.6: All steps including CIECAM02, dE', Rf,
  // hue binning, gamut, and per-sample fidelity.
  cached_.colorimetry = compute_ces_colorimetry(
      spd_.wavelengths(), spd_.values(), cmf_2deg_, cmf_10deg_, ces_data_,
      daylight_basis_, planckian_lut_);

  // Compute domain validity flags from the result.
  cached_.validity = compute_validity(cached_.colorimetry, spd_);

  computed_ = true;
}

// ==========================================================================
//  Accessors
// ==========================================================================

double Tm30::cct() const {
  ensure_computed();
  return cached_.colorimetry.cct;
}

double Tm30::duv() const {
  ensure_computed();
  return cached_.colorimetry.duv;
}

double Tm30::rf() const {
  ensure_computed();
  return cached_.colorimetry.Rf;
}

double Tm30::rg() const {
  ensure_computed();
  return cached_.colorimetry.gamut.Rg;
}

double Tm30::delta_e_avg() const {
  ensure_computed();
  return cached_.colorimetry.delta_e_avg;
}

const std::array<double, 99> &Tm30::rf_cesi() const {
  ensure_computed();
  return cached_.colorimetry.rf_cesi;
}

double Tm30::rf_skin() const {
  ensure_computed();
  return cached_.colorimetry.rf_skin;
}

const GamutResult &Tm30::gamut() const {
  ensure_computed();
  return cached_.colorimetry.gamut;
}

const LocalBinMetrics &Tm30::local_chroma_shift() const {
  ensure_computed();
  return cached_.colorimetry.gamut.local;
}

const CvgCoordinates &Tm30::cvg() const {
  ensure_computed();
  return cached_.colorimetry.gamut.cvg;
}

const Validity &Tm30::validity() const {
  ensure_computed();
  return cached_.validity;
}

const Tm30Result &Tm30::result() const {
  ensure_computed();
  return cached_;
}

const CesColorimetryResult &Tm30::colorimetry_result() const {
  ensure_computed();
  return cached_.colorimetry;
}

// ==========================================================================
//  Batch evaluate
// ==========================================================================

std::vector<std::optional<Tm30Result>>
try_evaluate(std::span<const SpdView> spds, const CmfData &cmf_2deg,
             const CmfData &cmf_10deg, const CesData &ces_data,
             const DaylightBasis &daylight_basis,
             const PlanckianLut &planckian_lut, Tm30Request /*request*/,
             std::size_t n_workers, TaskPool *pool) {

  std::vector<std::optional<Tm30Result>> results;
  results.resize(spds.size());

  if (n_workers <= 1) {
    // ==================================================================
    // Sequential path - the project's default. MUST stay free of any
    // std::thread construction: spawning+joining even one thread costs
    // ~40 us on real hardware against a ~140 us per-SPD workload here -
    // a double-digit-percent regression on the exact path every test
    // and benchmark has run against. Keep this loop body in sync with
    // the worker lambda in the parallel path below.
    // ==================================================================
    for (std::size_t i = 0; i < spds.size(); i++) {
      const auto &sv = spds[i];
      try {
        // Validate: construct an Spd from the SpdView data.
        // This copies the data into vectors, then validates.
        // Throws InvalidSpd if validation fails.
        Spd spd(
            std::vector<double>(sv.wavelengths.begin(), sv.wavelengths.end()),
            std::vector<double>(sv.values.begin(), sv.values.end()));

        // Run the full pipeline.
        Tm30Result result;
        result.colorimetry = compute_ces_colorimetry(
            spd.wavelengths(), spd.values(), cmf_2deg, cmf_10deg, ces_data,
            daylight_basis, planckian_lut);

        // Compute validity.
        result.validity = compute_validity(result.colorimetry, spd);

        results[i] = std::move(result);
      } catch (const InvalidSpd &) {
        // Per-SPD validation failure -> nullopt.
        // Batch never throws.
        results[i] = std::nullopt;
      }
    }
    return results;
  }

  // ==================================================================
  // Parallel path - only reachable when n_workers > 1. Task-parallel:
  // each SPD's whole computation runs start-to-finish on one thread, so
  // results are bit-identical to the sequential path. Static contiguous
  // chunking (per-SPD cost is uniform for a given grid), each thread
  // writes only its own results[i] indices; all tables are const& -
  // no locks, no shared mutable state. InvalidSpd is still caught
  // locally per iteration; nothing crosses a thread boundary.
  // ==================================================================

  // Persistent path (Phase 2): route around the spawn-per-call path
  // when a worker pool is active. run_chunked() uses the same partition
  // math and blocks until every chunk is done; exceptions never escape
  // the worker threads (InvalidSpd is caught locally per iteration, and
  // the pool catches anything else and rethrows it here).
  if (pool != nullptr) {
    pool->run_chunked(spds.size(), [&](std::size_t i) {
      const auto &sv = spds[i];
      try {
        // Keep in sync with the sequential loop body above.
        Spd spd(
            std::vector<double>(sv.wavelengths.begin(), sv.wavelengths.end()),
            std::vector<double>(sv.values.begin(), sv.values.end()));

        Tm30Result result;
        result.colorimetry = compute_ces_colorimetry(
            spd.wavelengths(), spd.values(), cmf_2deg, cmf_10deg, ces_data,
            daylight_basis, planckian_lut);

        result.validity = compute_validity(result.colorimetry, spd);

        results[i] = std::move(result);
      } catch (const InvalidSpd &) {
        results[i] = std::nullopt;
      }
    });
    return results;
  }

  const std::size_t actual_workers = std::min(n_workers, spds.size());
  if (actual_workers == 0)
    return results; // empty batch - nothing to do

  // Balanced contiguous chunks: sizes differ by at most 1, no thread
  // gets zero work (actual_workers <= spds.size() by construction).
  const std::size_t chunk_base = spds.size() / actual_workers;
  const std::size_t chunk_extra = spds.size() % actual_workers;

  std::vector<std::thread> threads;
  threads.reserve(actual_workers);
  for (std::size_t t = 0; t < actual_workers; ++t) {
    const std::size_t begin = t * chunk_base + std::min(t, chunk_extra);
    const std::size_t end = (t + 1) * chunk_base + std::min(t + 1, chunk_extra);
    threads.emplace_back([&, begin, end] {
      for (std::size_t i = begin; i < end; ++i) {
        const auto &sv = spds[i];
        try {
          // Keep in sync with the sequential loop body above.
          Spd spd(
              std::vector<double>(sv.wavelengths.begin(), sv.wavelengths.end()),
              std::vector<double>(sv.values.begin(), sv.values.end()));

          Tm30Result result;
          result.colorimetry = compute_ces_colorimetry(
              spd.wavelengths(), spd.values(), cmf_2deg, cmf_10deg, ces_data,
              daylight_basis, planckian_lut);

          result.validity = compute_validity(result.colorimetry, spd);

          results[i] = std::move(result);
        } catch (const InvalidSpd &) {
          results[i] = std::nullopt;
        }
      }
    });
  }
  for (auto &th : threads)
    th.join();

  return results;
}

// ==========================================================================
//  Batch evaluate - pre-resampled, grid-fixed tables
// ==========================================================================

std::vector<std::optional<Tm30Result>>
try_evaluate_cached(std::span<const SpdView> spds,
                    const ResampledTables &tables,
                    const PlanckianLut &planckian_lut, Tm30Request /*request*/,
                    std::size_t n_workers, TaskPool *pool) {

  std::vector<std::optional<Tm30Result>> results;
  results.resize(spds.size());

  if (n_workers <= 1) {
    // ==================================================================
    // Sequential path - mirrors try_evaluate() exactly; same zero-thread
    // rule applies (see the comment there).
    // ==================================================================
    for (std::size_t i = 0; i < spds.size(); i++) {
      const auto &sv = spds[i];
      try {
        // Validate: construct an Spd from the SpdView data.
        // This copies the data into vectors, then validates.
        // Throws InvalidSpd if validation fails.
        Spd spd(
            std::vector<double>(sv.wavelengths.begin(), sv.wavelengths.end()),
            std::vector<double>(sv.values.begin(), sv.values.end()));

        // Run the pipeline using the pre-resampled tables - no CES/CMF
        // resampling happens here.
        // Guard: the row's S3.5-conformed grid must match the grid the
        // cached tables were resampled to. Rows and the bound grid are
        // conformed by the same Spd recipe, so a mismatch means this
        // row's input grid differs from the bound grid.
        if (spd.wavelengths() != tables.wavelengths) {
          throw InvalidSpd("SPD wavelength grid does not match the fixed "
                           "grid the cached tables are bound to");
        }

        Tm30Result result;
        result.colorimetry =
            compute_ces_colorimetry_cached(spd.values(), tables, planckian_lut);

        // Compute validity.
        result.validity = compute_validity(result.colorimetry, spd);

        results[i] = std::move(result);
      } catch (const InvalidSpd &) {
        // Per-SPD validation failure -> nullopt.
        // Batch never throws.
        results[i] = std::nullopt;
      }
    }
    return results;
  }

  // Persistent path (Phase 2) - mirrors try_evaluate()'s persistent
  // path; same partition math, same per-chunk body.
  if (pool != nullptr) {
    pool->run_chunked(spds.size(), [&](std::size_t i) {
      const auto &sv = spds[i];
      try {
        // Keep in sync with the sequential loop body above.
        Spd spd(
            std::vector<double>(sv.wavelengths.begin(), sv.wavelengths.end()),
            std::vector<double>(sv.values.begin(), sv.values.end()));

        // Guard: the row's S3.5-conformed grid must match the grid the
        // cached tables were resampled to. Rows and the bound grid are
        // conformed by the same Spd recipe, so a mismatch means this
        // row's input grid differs from the bound grid.
        if (spd.wavelengths() != tables.wavelengths) {
          throw InvalidSpd("SPD wavelength grid does not match the fixed "
                           "grid the cached tables are bound to");
        }

        Tm30Result result;
        result.colorimetry =
            compute_ces_colorimetry_cached(spd.values(), tables, planckian_lut);

        result.validity = compute_validity(result.colorimetry, spd);

        results[i] = std::move(result);
      } catch (const InvalidSpd &) {
        results[i] = std::nullopt;
      }
    });
    return results;
  }

  // Parallel path - mirrors try_evaluate()'s parallel path. Same static
  // contiguous chunking, same per-thread disjoint index writes.
  const std::size_t actual_workers = std::min(n_workers, spds.size());
  if (actual_workers == 0)
    return results; // empty batch - nothing to do

  const std::size_t chunk_base = spds.size() / actual_workers;
  const std::size_t chunk_extra = spds.size() % actual_workers;

  std::vector<std::thread> threads;
  threads.reserve(actual_workers);
  for (std::size_t t = 0; t < actual_workers; ++t) {
    const std::size_t begin = t * chunk_base + std::min(t, chunk_extra);
    const std::size_t end = (t + 1) * chunk_base + std::min(t + 1, chunk_extra);
    threads.emplace_back([&, begin, end] {
      for (std::size_t i = begin; i < end; ++i) {
        const auto &sv = spds[i];
        try {
          // Keep in sync with the sequential loop body above.
          Spd spd(
              std::vector<double>(sv.wavelengths.begin(), sv.wavelengths.end()),
              std::vector<double>(sv.values.begin(), sv.values.end()));

          // Guard: see the sequential loop body above.
          if (spd.wavelengths() != tables.wavelengths) {
            throw InvalidSpd("SPD wavelength grid does not match the fixed "
                             "grid the cached tables are bound to");
          }

          Tm30Result result;
          result.colorimetry = compute_ces_colorimetry_cached(
              spd.values(), tables, planckian_lut);

          result.validity = compute_validity(result.colorimetry, spd);

          results[i] = std::move(result);
        } catch (const InvalidSpd &) {
          results[i] = std::nullopt;
        }
      }
    });
  }
  for (auto &th : threads)
    th.join();

  return results;
}

} // namespace tm30
