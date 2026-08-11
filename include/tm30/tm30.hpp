#pragma once

/// @file tm30.hpp
/// Lazy memoized TM-30 handle + batch API with Validity flags.
///
/// Provides the ergonomics layer on top of the fully verified color-science
/// core (Slices 0-10). Two entry points:
///
///   Single SPD -> Tm30 handle (lazy evaluation, memoized cache)
///   Batch SPDs -> try_evaluate() with request flags
///
/// Thread safety: The Tm30 handle's `const` accessors mutate an internal
/// cache and are NOT thread-safe. Callers parallelize across SPDs, not
/// within one SPD. No mutex, no atomics.

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "tm30/cct.hpp"       // PlanckianLut
#include "tm30/gamut.hpp"     // GamutResult, LocalBinMetrics, CvgCoordinates
#include "tm30/pipeline.hpp"  // CesColorimetryResult
#include "tm30/reference.hpp" // DaylightBasis
#include "tm30/resample.hpp"  // CesData, CmfData
#include "tm30/spd.hpp"       // Spd
#include "tm30/task_pool.hpp" // TaskPool (persistent workers, Phase 2)

namespace tm30 {

// ==========================================================================
//  SpdView - non-owning view of wavelength/value arrays for batch input.
// ==========================================================================

/// Non-owning view of a Spectral Power Distribution.
///
/// Used in the batch API for zero-copy referencing of many SPDs.
/// Callers are responsible for keeping the underlying data alive.
struct SpdView {
  std::span<const double> wavelengths; ///< Wavelength grid (nm), monotonic
  std::span<const double> values;      ///< Spectral power values St(lambda)
};

// ==========================================================================
//  Tm30Request - batch evaluation flags.
// ==========================================================================

/// Controls what gets included in batch output.
///
/// One decision per batch run. In the batch path, flags bound the output
/// buffer size (memory-bandwidth win, not FLOP win - ~85-90% of compute
/// is upstream of the binning fork).
struct Tm30Request {
  /// Include per-bin metrics (Rf,hj, Rcs,hj, Rhs,hj, DE_hj) and CVG
  /// coordinates in the output.
  bool bins = true;

  /// Include per-sample fidelity Rf,CESi and Rf,skin in the output.
  bool samples = false;
};

// ==========================================================================
//  Validity - domain-range warning flags (result data, not errors).
// ==========================================================================

/// Domain validity flags for TM-30 results.
///
/// These are advisory warnings - the pipeline computes all metrics
/// regardless. Callers decide how to act on them.
struct Validity {
  /// Duv is far from the Planckian locus. Advisory only; pytm30 impl
  /// choice (TM-30-20 §2.0 states the near-locus domain qualitatively and
  /// prints no numerical Duv bound).
  bool duv_out_of_range = false;

  /// CCT is far from the range where TM-30 is typically applied. Advisory
  /// only; pytm30 impl choice (TM-30-20 §2.0 prints no numerical CCT
  /// bounds).
  bool cct_out_of_range = false;

  /// The test SPD does not cover the full 380-780 nm grid. Missing values
  /// were zero-filled per TM-30-20 §3.5 (which forbids interpolating or
  /// extrapolating the test SPD). Not the same as CES/CMF flat
  /// extrapolation (TM-30-20 §1.3 / Annex A).
  bool extrapolated = false;
};

// ==========================================================================
//  Tm30Result - full TM-30 output for a single SPD.
// ==========================================================================

/// Complete TM-30 evaluation result: all colorimetric outputs plus
/// domain-validity flags.
struct Tm30Result {
  CesColorimetryResult colorimetry; ///< Full pipeline result
  Validity validity;                ///< Domain-range warning flags
};

// ==========================================================================
//  Tm30 - lazy memoized handle for a single SPD.
// ==========================================================================

/// Lazy, memoized TM-30 evaluator for a single SPD.
///
/// Construction validates the SPD and stores pre-loaded data table
/// references, but computes nothing. The full pipeline runs on the first
/// accessor call, and all subsequent calls reuse the cached result.
///
/// Thread safety: `const` accessors mutate an internal cache and are
/// explicitly NOT thread-safe. Parallelize across SPDs, not within one.
///
/// ```cpp
/// Tm30 m(spd, cmf_2deg, cmf_10deg, ces, basis, lut);
/// double rf = m.rf();    // runs pipeline, caches
/// double rg = m.rg();    // reuses cache
/// ```
class Tm30 {
public:
  /// Construct a lazy evaluator for the given test SPD.
  ///
  /// Validates the SPD at construction. The pre-loaded data tables must
  /// outlive the Tm30 instance.
  ///
  /// @param spd           Test source SPD (owned - moved into the handle).
  /// @param cmf_2deg      CIE 1931 2-deg CMF data (for CCT computation).
  /// @param cmf_10deg     CIE 1964 10-deg CMF data (for tristimulus
  /// integration).
  /// @param ces_data      CES reflectance data (99 samples, 1-nm native).
  /// @param daylight_basis Daylight basis vectors (S0, S1, S2).
  /// @param planckian_lut Planckian locus LUT (CIE 1931 2-deg observer).
  ///
  /// @throws InvalidSpd if the SPD fails validation.
  Tm30(Spd spd, const CmfData &cmf_2deg, const CmfData &cmf_10deg,
       const CesData &ces_data, const DaylightBasis &daylight_basis,
       const PlanckianLut &planckian_lut);

  // -- Accessors ------------------------------------------------------
  // All accessors are const but mutate the internal cache.
  // NOT thread-safe. First call triggers full pipeline computation.

  /// Correlated Color Temperature (K).                    TM-30-20 §3.3
  double cct() const;

  /// Distance from Planckian locus in CIE 1960 UCS.       TM-30-20 §3.3
  double duv() const;

  /// Fidelity index Rf [0, 100].                          TM-30-20 §4.1
  double rf() const;

  /// Gamut area index Rg.                                 TM-30-20 §4.4
  double rg() const;

  /// Average color difference dE' across 99 CES.          TM-30-20 §4.1
  double delta_e_avg() const;

  /// Per-sample fidelity Rf,CESi (99 values).             TM-30-20 §4.2
  const std::array<double, 99> &rf_cesi() const;

  /// Skin fidelity Rf,skin (average of CES15 + CES18).    TM-30-20 §4.2
  double rf_skin() const;

  /// Full gamut result: Rg, per-bin metrics, CVG.         TM-30-20 §4.4-§4.8
  const GamutResult &gamut() const;

  /// Per-bin local metrics (Rf,hj, Rcs,hj, Rhs,hj, DE_hj). TM-30-20 §4.6-§4.8
  const LocalBinMetrics &local_chroma_shift() const;

  /// CVG-normalized bin-average coordinates.              TM-30-20 §4.5
  const CvgCoordinates &cvg() const;

  /// Domain validity flags.
  const Validity &validity() const;

  /// Access the full result (triggers computation if not yet cached).
  const Tm30Result &result() const;

  /// The raw CesColorimetryResult (triggers computation if not yet cached).
  const CesColorimetryResult &colorimetry_result() const;

  // -- Cache control --------------------------------------------------

  /// True if the pipeline has already been computed and cached.
  bool is_computed() const noexcept { return computed_; }

  /// Invalidate the cache.  Next accessor call will recompute.
  void invalidate() noexcept { computed_ = false; }

private:
  /// Ensure the pipeline has been run. Called by every accessor.
  void ensure_computed() const;

  // -- Owned data -----------------------------------------------------
  Spd spd_;

  // -- Borrowed references (caller must keep these alive) -------------
  const CmfData &cmf_2deg_;
  const CmfData &cmf_10deg_;
  const CesData &ces_data_;
  const DaylightBasis &daylight_basis_;
  const PlanckianLut &planckian_lut_;

  // -- Mutable cache --------------------------------------------------
  mutable bool computed_ = false;
  mutable Tm30Result cached_;
};

// ==========================================================================
//  Batch API
// ==========================================================================

/// Evaluate TM-30 for a batch of SPDs.
///
/// One decision per run, packed contiguous output. Never throws -
/// per-SPD failures return `std::nullopt` at the corresponding index.
///
/// Embarrassingly parallel across SPDs (callers may parallelize the loop).
///
/// @param spds           Non-owning views of test source SPDs.
/// @param cmf_2deg       CIE 1931 2-deg CMF data.
/// @param cmf_10deg      CIE 1964 10-deg CMF data.
/// @param ces_data       CES reflectance data (99 samples).
/// @param daylight_basis Daylight basis vectors (S0, S1, S2).
/// @param planckian_lut  Planckian locus LUT.
/// @param request        Output control flags (bins, samples).
/// @param n_workers      Number of worker threads to use for this call.
///                       n_workers <= 1 (the default) runs the sequential
///                       loop with ZERO std::thread construction; values
///                       > 1 spawn min(n_workers, spds.size()) threads,
///                       one contiguous chunk each (static balancing,
///                       results bit-identical to sequential).
/// @param pool           Optional persistent worker pool (Phase 2). When
///                       non-null AND n_workers > 1, the batch runs on
///                       the pool's persistent threads instead of
///                       spawning per call - same partition math, same
///                       bit-identical results. Ignored when
///                       n_workers <= 1.
///
/// @return Vector of optional Tm30Result - one per input SPD.
///         `nullopt` at index i means SPD i failed validation.
std::vector<std::optional<Tm30Result>>
try_evaluate(std::span<const SpdView> spds, const CmfData &cmf_2deg,
             const CmfData &cmf_10deg, const CesData &ces_data,
             const DaylightBasis &daylight_basis,
             const PlanckianLut &planckian_lut, Tm30Request request = {},
             std::size_t n_workers = 1, TaskPool *pool = nullptr);

/// Evaluate TM-30 for a batch of SPDs using pre-resampled, grid-fixed
/// tables (see prepare_resampled_tables() in pipeline.hpp).
///
/// Mirrors try_evaluate() exactly (same try/catch-per-row structure, same
/// Tm30Request handling) but calls compute_ces_colorimetry_cached() instead
/// of compute_ces_colorimetry() - skipping CES/CMF resampling for every
/// SPD in the batch, since `tables` was already resampled once to the
/// shared grid all SPDs in `spds` are assumed to share.
///
/// @param spds           Non-owning views of test source SPDs, all sharing
///                        tables.wavelengths as their grid.
/// @param tables         Pre-resampled CES/CMF/daylight-basis tables.
/// @param planckian_lut  Planckian locus LUT.
/// @param request        Output control flags (bins, samples).
/// @param n_workers      Number of worker threads for this call; same
///                       semantics as try_evaluate()'s parameter.
/// @param pool           Optional persistent worker pool (Phase 2); same
///                       semantics as try_evaluate()'s parameter.
///
/// @return Vector of optional Tm30Result - one per input SPD.
///         `nullopt` at index i means SPD i failed validation.
std::vector<std::optional<Tm30Result>>
try_evaluate_cached(std::span<const SpdView> spds,
                    const ResampledTables &tables,
                    const PlanckianLut &planckian_lut, Tm30Request request = {},
                    std::size_t n_workers = 1, TaskPool *pool = nullptr);

} // namespace tm30
