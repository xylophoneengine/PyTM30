#include "tm30/cct.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/errors.hpp"
#include "tm30/gamut.hpp"
#include "tm30/pipeline.hpp"
#include "tm30/reference.hpp"
#include "tm30/resample.hpp"
#include "tm30/spd.hpp"
#include "tm30/tm30.hpp"
#include "tm30/xyz.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility> // std::move
#include <vector>

namespace nb = nanobind;

// ==========================================================================
//  Data loading helpers
// ==========================================================================

namespace {

std::string data_file(const std::string &dir, const std::string &name) {
  if (dir.empty())
    return name;
  if (dir.back() == '/')
    return dir + name;
  return dir + "/" + name;
}

tm30::CmfData load_cmf(const std::string &path) {
  tm30::CsvTable table = tm30::load_csv(path);
  tm30::CmfData data;
  for (const auto &row : table.rows) {
    data.wavelengths.push_back(row[0]);
    data.x_bar.push_back(row[1]);
    data.y_bar.push_back(row[2]);
    data.z_bar.push_back(row[3]);
  }
  return data;
}

tm30::CesData load_ces(const std::string &path) {
  tm30::CsvTable table = tm30::load_csv(path);
  tm30::CesData data;
  std::size_t n_ces = table.headers.size() - 1;
  data.samples.resize(n_ces);
  for (const auto &row : table.rows) {
    data.wavelengths.push_back(row[0]);
    for (std::size_t c = 1; c < row.size(); ++c)
      data.samples[c - 1].push_back(row[c]);
  }
  return data;
}

/// True if `arr` is C-contiguous (row-major, no gaps) - i.e. a flat
/// `data() .. data()+size()` walk visits every element in logical order.
/// A column slice of a 2-D array (e.g. `csv[:, 0]`), a transpose, or a
/// reversed view (`arr[::-1]`) are all NOT contiguous: nanobind's
/// `ndarray<>::data()` returns a raw pointer with no stride information,
/// so reading it as if contiguous silently walks the wrong memory.
/// DLPack (which nanobind's ndarray is built on) always reports strides
/// in element counts, not bytes, so this compares directly against shape.
bool is_c_contiguous(const nb::ndarray<> &arr) {
  int64_t expected = 1;
  for (int i = static_cast<int>(arr.ndim()) - 1; i >= 0; --i) {
    if (arr.shape(i) > 1 && arr.stride(i) != expected)
      return false;
    expected *= static_cast<int64_t>(arr.shape(i));
  }
  return true;
}

void require_c_contiguous(const nb::ndarray<> &arr, const char *name) {
  if (!is_c_contiguous(arr)) {
    throw std::invalid_argument(
        std::string(name) +
        " must be C-contiguous - a column slice of a 2-D array, a "
        "transpose, or a reversed view (arr[::-1]) is not. Call "
        "np.ascontiguousarray(" +
        name + ") or " + name + ".copy() first.");
  }
}

} // namespace

// ==========================================================================
//  PyTm30 - Python wrapper owning data + Tm30 handle
// ==========================================================================

struct PyTm30 {
  tm30::CmfData cmf_2deg_;
  tm30::CmfData cmf_10deg_;
  tm30::CesData ces_data_;
  tm30::DaylightBasis daylight_basis_;
  tm30::PlanckianLut planckian_lut_;
  std::unique_ptr<tm30::Tm30> tm30_;

  /// Construct from a numpy array of SPD values (and optional wavelengths).
  PyTm30(nb::object spd_values_arg, nb::object spd_wavelengths_arg,
         const std::string &data_dir_arg) {
    // -- Cast spd_values to ndarray ---------------------------------
    auto spd_arr = nb::cast<nb::ndarray<>>(spd_values_arg);
    if (spd_arr.ndim() != 1) {
      throw std::invalid_argument("spd_values must be a 1-D array");
    }
    if (spd_arr.dtype() != nb::dtype<double>()) {
      throw std::invalid_argument("spd_values must have dtype float64");
    }
    require_c_contiguous(spd_arr, "spd_values");
    size_t n_vals = spd_arr.shape(0);
    const double *spd_data = static_cast<const double *>(spd_arr.data());

    // -- Determine wavelengths --------------------------------------
    std::vector<double> wl;
    if (spd_wavelengths_arg.is_none()) {
      if (n_vals != 401) {
        throw std::invalid_argument(
            "spd_values has " + std::to_string(n_vals) +
            " elements but default wavelengths expect 401 (380-780 nm). "
            "Pass explicit spd_wavelengths.");
      }
      wl.resize(401);
      for (size_t i = 0; i < 401; ++i)
        wl[i] = 380.0 + static_cast<double>(i);
    } else {
      auto wl_arr = nb::cast<nb::ndarray<>>(spd_wavelengths_arg);
      if (wl_arr.ndim() != 1) {
        throw std::invalid_argument("spd_wavelengths must be a 1-D array");
      }
      require_c_contiguous(wl_arr, "spd_wavelengths");
      size_t nw = wl_arr.shape(0);
      const double *wl_data = static_cast<const double *>(wl_arr.data());
      wl.assign(wl_data, wl_data + nw);
    }

    // -- Copy SPD values --------------------------------------------
    std::vector<double> vals(spd_data, spd_data + n_vals);

    // -- Load data tables -------------------------------------------
    std::string dir = data_dir_arg.empty() ? TM30_DATA_DIR : data_dir_arg;
    cmf_2deg_ = load_cmf(data_file(dir, "cie_1931_2.csv"));
    cmf_10deg_ = load_cmf(data_file(dir, "cmf_1964_10.csv"));
    ces_data_ = load_ces(data_file(dir, "ces.csv"));
    daylight_basis_ =
        tm30::load_daylight_basis(data_file(dir, "daylight_basis.csv"));
    planckian_lut_ =
        tm30::load_planckian_lut(data_file(dir, "planckian_uv.csv"));

    // -- Create Tm30 handle (validates SPD, computes nothing) -------
    tm30_ = std::make_unique<tm30::Tm30>(
        tm30::Spd(std::move(wl), std::move(vals)), cmf_2deg_, cmf_10deg_,
        ces_data_, daylight_basis_, planckian_lut_);
  }

  // -- Properties ---------------------------------------------------

  double rf() const { return tm30_->rf(); }
  double rg() const { return tm30_->rg(); }
  double cct() const { return tm30_->cct(); }
  double duv() const { return tm30_->duv(); }
  double delta_e_avg() const { return tm30_->delta_e_avg(); }
  double rf_skin() const { return tm30_->rf_skin(); }
  const tm30::Validity &validity() const { return tm30_->validity(); }

  /// Per-sample fidelity Rf,CESi - returns a numpy array.
  /// Per-sample fidelity Rf,CESi - returns a numpy array.
  nb::object rf_cesi() const {
    const auto &arr = tm30_->rf_cesi();
    size_t n = arr.size();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(n, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    std::copy(arr.begin(), arr.end(), buf);
    return result;
  }

  /// Per-bin chroma shift Rcs,hj, in percent (16 values) - numpy array.
  nb::object rcs_hj() const {
    const auto &lcm = tm30_->local_chroma_shift();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(16, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (int j = 0; j < 16; ++j)
      buf[j] = lcm.Rcs_hj_percent[j];
    return result;
  }

  /// Per-bin hue shift Rhs,hj, dimensionless ratio (16 values) - numpy array.
  nb::object rhs_hj() const {
    const auto &lcm = tm30_->local_chroma_shift();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(16, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (int j = 0; j < 16; ++j)
      buf[j] = lcm.Rhs_hj[j];
    return result;
  }

  /// Per-bin local fidelity Rf,hj (16 values) - returns numpy array.
  nb::object rf_hj() const {
    const auto &lcm = tm30_->local_chroma_shift();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(16, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (int j = 0; j < 16; ++j)
      buf[j] = lcm.Rf_hj[j];
    return result;
  }

  /// Per-bin mean dE', DE_hj (16 values) - returns numpy array.
  nb::object de_hj() const {
    const auto &lcm = tm30_->local_chroma_shift();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(16, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (int j = 0; j < 16; ++j)
      buf[j] = lcm.DE_hj[j];
    return result;
  }

  /// CVG test-vector J' (16 values) - returns numpy array.
  nb::object cvg_j_test() const {
    const auto &cvg = tm30_->cvg();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(16, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (int j = 0; j < 16; ++j)
      buf[j] = cvg.J_test[j];
    return result;
  }

  /// CVG test-vector x (16 values) - returns numpy array.
  nb::object cvg_x_test() const {
    const auto &cvg = tm30_->cvg();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(16, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (int j = 0; j < 16; ++j)
      buf[j] = cvg.x_test[j];
    return result;
  }

  /// CVG test-vector y (16 values) - returns numpy array.
  nb::object cvg_y_test() const {
    const auto &cvg = tm30_->cvg();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(16, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (int j = 0; j < 16; ++j)
      buf[j] = cvg.y_test[j];
    return result;
  }

  /// CVG reference-circle J' (16 values) - returns numpy array.
  nb::object cvg_j_ref() const {
    const auto &cvg = tm30_->cvg();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(16, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (int j = 0; j < 16; ++j)
      buf[j] = cvg.J_ref[j];
    return result;
  }

  /// CVG reference-circle x (16 values) - returns numpy array.
  nb::object cvg_x_ref() const {
    const auto &cvg = tm30_->cvg();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(16, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (int j = 0; j < 16; ++j)
      buf[j] = cvg.x_ref[j];
    return result;
  }

  /// CVG reference-circle y (16 values) - returns numpy array.
  nb::object cvg_y_ref() const {
    const auto &cvg = tm30_->cvg();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(16, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (int j = 0; j < 16; ++j)
      buf[j] = cvg.y_ref[j];
    return result;
  }

  /// Reference-illuminant SPD, resampled to the input wavelength grid -
  /// returns numpy array (length matches the test SPD's wavelength grid).
  nb::object reference_spd() const {
    const auto &cr = tm30_->colorimetry_result();
    size_t n = cr.reference_spd_values.size();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(n, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    std::copy(cr.reference_spd_values.begin(), cr.reference_spd_values.end(),
              buf);
    return result;
  }

  /// Per-CES XYZ under the test source (99x3) - returns numpy array.
  nb::object xyz_test_ces() const {
    const auto &cr = tm30_->colorimetry_result();
    auto np = nb::module_::import_("numpy");
    auto result =
        np.attr("empty")(nb::make_tuple(99, 3), nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (size_t i = 0; i < 99; ++i) {
      buf[i * 3 + 0] = cr.xyz_test_ces[i].X;
      buf[i * 3 + 1] = cr.xyz_test_ces[i].Y;
      buf[i * 3 + 2] = cr.xyz_test_ces[i].Z;
    }
    return result;
  }

  /// Per-CES XYZ under the reference illuminant (99x3) - returns numpy array.
  nb::object xyz_ref_ces() const {
    const auto &cr = tm30_->colorimetry_result();
    auto np = nb::module_::import_("numpy");
    auto result =
        np.attr("empty")(nb::make_tuple(99, 3), nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (size_t i = 0; i < 99; ++i) {
      buf[i * 3 + 0] = cr.xyz_ref_ces[i].X;
      buf[i * 3 + 1] = cr.xyz_ref_ces[i].Y;
      buf[i * 3 + 2] = cr.xyz_ref_ces[i].Z;
    }
    return result;
  }

  /// Per-CES CAM02-UCS J'a'b' under the test source (99x3) - numpy array.
  /// TM-30-20 §3.7.1 Eq. (48)-(50).
  nb::object jab_test_ces() const {
    const auto &cr = tm30_->colorimetry_result();
    auto np = nb::module_::import_("numpy");
    auto result =
        np.attr("empty")(nb::make_tuple(99, 3), nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (size_t i = 0; i < 99; ++i) {
      buf[i * 3 + 0] = cr.jab_test_ces[i].J_prime;
      buf[i * 3 + 1] = cr.jab_test_ces[i].a_prime;
      buf[i * 3 + 2] = cr.jab_test_ces[i].b_prime;
    }
    return result;
  }

  /// Per-CES CAM02-UCS J'a'b' under the reference illuminant (99x3) - numpy
  /// array. TM-30-20 §3.7.1 Eq. (48)-(50).
  nb::object jab_ref_ces() const {
    const auto &cr = tm30_->colorimetry_result();
    auto np = nb::module_::import_("numpy");
    auto result =
        np.attr("empty")(nb::make_tuple(99, 3), nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (size_t i = 0; i < 99; ++i) {
      buf[i * 3 + 0] = cr.jab_ref_ces[i].J_prime;
      buf[i * 3 + 1] = cr.jab_ref_ces[i].a_prime;
      buf[i * 3 + 2] = cr.jab_ref_ces[i].b_prime;
    }
    return result;
  }

  /// Per-CES hue-angle bin index (0-15), assigned from the reference hue
  /// angle hr = atan2(b'r, a'r) - TM-30-20 §4.3. Same assignment is used
  /// for both test and reference bin averages (see hue_bins.cpp). Returns
  /// a numpy int array, inverted from the internal per-bin index lists.
  nb::object hue_bin_index() const {
    const auto &cr = tm30_->colorimetry_result();
    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(99, nb::arg("dtype") = "int64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    int64_t *buf = static_cast<int64_t *>(nd.data());
    for (int j = 0; j < 16; ++j) {
      for (int ces_idx : cr.hue_bins[j]) {
        buf[ces_idx] = j;
      }
    }
    return result;
  }
};

// ======================================================================
//  Batch evaluate() - Python wrapper
// ======================================================================

struct BatchContext {
  tm30::CmfData cmf_2deg;
  tm30::CmfData cmf_10deg;
  tm30::CesData ces_data;
  tm30::DaylightBasis daylight_basis;
  tm30::PlanckianLut planckian_lut;
  std::vector<tm30::Spd> owned_spds;
  std::vector<tm30::SpdView> views;

  // Grid-fixed cache: CES/CMF/daylight-basis tables pre-resampled to a
  // fixed wavelength grid, set once via set_fixed_grid() (called from
  // TM30Calc.__init__) and reused by evaluate_cached() for every SPD that
  // shares that grid. Unset (nullopt) until set_fixed_grid() is called.
  std::optional<tm30::ResampledTables> fixed_tables_;

  // Phase 2: persistent worker pool. Created eagerly when the context is
  // constructed with persistent_workers=true and n_workers>1; passed to
  // try_evaluate/try_evaluate_cached so repeated eval() calls reuse the
  // same threads instead of spawning per call. unique_ptr makes this
  // type move-only (a thread pool cannot be copied); nanobind does not
  // copy bound objects, so this is safe.
  std::unique_ptr<tm30::TaskPool> pool_;

  /// Shared constructor tail: validate n_workers and create the
  /// persistent pool when requested.
  //  persistent_workers=true with n_workers<=1 is SILENTLY INERT
  //  no pool is created and behaviour
  /// is exactly the n_workers<=1 sequential path. n_workers<1 always
  /// raises (Phase 1 decision (a): reject, no auto-detect).
  void init_pool(int n_workers, bool persistent_workers) {
    if (n_workers < 1) {
      throw std::invalid_argument(
          "n_workers must be >= 1 (got " + std::to_string(n_workers) +
          "); n_workers=0/-1 auto-detection is not implemented");
    }
    if (persistent_workers && n_workers > 1) {
      pool_ =
          std::make_unique<tm30::TaskPool>(static_cast<std::size_t>(n_workers));
    }
  }

  BatchContext(const std::string &data_dir, int n_workers = 1,
               bool persistent_workers = false) {
    init_pool(n_workers, persistent_workers);
    auto df = [&](const std::string &name) {
      return data_dir.empty() ? name : data_dir + "/" + name;
    };
    cmf_2deg = load_cmf(df("cie_1931_2.csv"));
    cmf_10deg = load_cmf(df("cmf_1964_10.csv"));
    ces_data = load_ces(df("ces.csv"));
    daylight_basis = tm30::load_daylight_basis(df("daylight_basis.csv"));
    planckian_lut = tm30::load_planckian_lut(df("planckian_uv.csv"));
  }

  /// Constructor with explicit CMF file paths.
  /// data_dir is used for the non-CMF tables only.
  BatchContext(const std::string &data_dir, const std::string &cmf_2deg_path,
               const std::string &cmf_10deg_path, int n_workers = 1,
               bool persistent_workers = false) {
    init_pool(n_workers, persistent_workers);
    auto df = [&](const std::string &name) {
      return data_dir.empty() ? name : data_dir + "/" + name;
    };
    cmf_2deg = load_cmf(cmf_2deg_path);
    cmf_10deg = load_cmf(cmf_10deg_path);
    ces_data = load_ces(df("ces.csv"));
    daylight_basis = tm30::load_daylight_basis(df("daylight_basis.csv"));
    planckian_lut = tm30::load_planckian_lut(df("planckian_uv.csv"));
  }

  void prepare_batch(nb::ndarray<> spd_matrix, nb::object wl_arg) {
    if (spd_matrix.ndim() != 2)
      throw std::invalid_argument("spd_matrix must be 2-D (N_spds x N_wl)");
    require_c_contiguous(spd_matrix, "spd_matrix");
    size_t N = spd_matrix.shape(0);
    size_t nwl = spd_matrix.shape(1);
    const double *data = static_cast<const double *>(spd_matrix.data());

    std::vector<double> wl;
    if (wl_arg.is_none()) {
      if (nwl != 401)
        throw std::invalid_argument(
            "Expected 401 wavelengths (380-780 nm), got " +
            std::to_string(nwl));
      wl.resize(401);
      for (size_t i = 0; i < 401; ++i)
        wl[i] = 380.0 + i;
    } else {
      auto wl_arr = nb::cast<nb::ndarray<>>(wl_arg);
      if (wl_arr.ndim() != 1 || wl_arr.shape(0) != nwl)
        throw std::invalid_argument(
            "wavelengths must match spd_matrix columns");
      require_c_contiguous(wl_arr, "wavelengths");
      const double *wl_data = static_cast<const double *>(wl_arr.data());
      wl.assign(wl_data, wl_data + nwl);
    }

    owned_spds.clear();
    views.clear();
    owned_spds.reserve(N);
    views.reserve(N);

    for (size_t i = 0; i < N; ++i) {
      std::vector<double> vals(data + i * nwl, data + (i + 1) * nwl);
      std::vector<double> wl_copy = wl; // each Spd owns its wavelengths
      owned_spds.emplace_back(std::move(wl_copy), std::move(vals));
      const auto &spd = owned_spds.back();
      views.push_back({spd.wavelengths(), spd.values()});
    }
  }

  /// Precompute and cache CES/CMF/daylight-basis tables resampled to `wl`
  /// - the fixed wavelength grid a TM30Calc instance is bound to at
  /// construction. Called once by TM30Calc.__init__; evaluate_cached()
  /// reuses this cache for every SPD sharing this grid (the common case),
  /// skipping resampling entirely.
  void set_fixed_grid(nb::ndarray<> wl) {
    if (wl.ndim() != 1) {
      throw std::invalid_argument("wavelengths must be a 1-D array");
    }
    require_c_contiguous(wl, "wavelengths");
    size_t nwl = wl.shape(0);
    const double *wl_data = static_cast<const double *>(wl.data());
    std::vector<double> wl_vec(wl_data, wl_data + nwl);

    fixed_tables_ = tm30::prepare_resampled_tables(wl_vec, cmf_2deg, cmf_10deg,
                                                   ces_data, daylight_basis);
  }

  // NOTE: `bins`/`samples` gate which arrays get allocated and copied into
  // the per-SPD Python dict at *this* layer - `samples` controls `rf_cesi`,
  // `bins` controls `rcs_hj`/`rhs_hj`. That's the actual memory-bandwidth
  // win the docs describe (not a FLOP win): the underlying C++ pipeline in
  // src/tm30/tm30.cpp still computes every field unconditionally regardless
  // of these flags - see Tm30Request's doc comment in include/tm30/tm30.hpp
  // ("~85-90% of compute is upstream of the binning fork"). That part is
  // unchanged and intentionally out of scope here. `extras` below is the
  // same kind of gate: it only controls whether its additional fields get
  // array-copied into the per-SPD dict (the C++ pipeline computes them
  // unconditionally either way, same as bins/samples).
  nb::list evaluate(bool bins, bool samples, bool extras, int n_workers) {
    if (n_workers < 1) {
      throw std::invalid_argument(
          "n_workers must be >= 1 (got " + std::to_string(n_workers) +
          "); n_workers=0/-1 auto-detection is not implemented");
    }
    tm30::Tm30Request req{bins, samples};
    // The C++ parallel region runs with the GIL released so a big batch
    // call does not block other Python threads for its duration. All
    // dict/array marshaling below happens AFTER this scope, with the GIL
    // re-acquired (numpy calls require it).
    auto results = [&]() {
      nb::gil_scoped_release release;
      return tm30::try_evaluate(
          views, cmf_2deg, cmf_10deg, ces_data, daylight_basis, planckian_lut,
          req, static_cast<std::size_t>(n_workers), pool_.get());
    }();
    nb::list out;
    auto np = nb::module_::import_("numpy");
    for (auto &opt : results) {
      if (!opt) {
        out.append(nb::none());
        continue;
      }
      nb::dict d;
      d["rf"] = opt->colorimetry.Rf;
      d["rg"] = opt->colorimetry.gamut.Rg;
      d["cct"] = opt->colorimetry.cct;
      d["duv"] = opt->colorimetry.duv;
      d["delta_e_avg"] = opt->colorimetry.delta_e_avg;
      d["rf_skin"] = opt->colorimetry.rf_skin;

      if (samples) {
        // rf_cesi as numpy array
        auto arr_cesi = np.attr("empty")(99, nb::arg("dtype") = "float64");
        auto nd_cesi = nb::cast<nb::ndarray<>>(arr_cesi);
        double *buf_cesi = static_cast<double *>(nd_cesi.data());
        std::copy(opt->colorimetry.rf_cesi.begin(),
                  opt->colorimetry.rf_cesi.end(), buf_cesi);
        d["rf_cesi"] = arr_cesi;
      }

      if (bins) {
        // Per-bin as numpy arrays
        auto arr_rcs = np.attr("empty")(16, nb::arg("dtype") = "float64");
        auto arr_rhs = np.attr("empty")(16, nb::arg("dtype") = "float64");
        auto nd_rcs = nb::cast<nb::ndarray<>>(arr_rcs);
        auto nd_rhs = nb::cast<nb::ndarray<>>(arr_rhs);
        double *buf_rcs = static_cast<double *>(nd_rcs.data());
        double *buf_rhs = static_cast<double *>(nd_rhs.data());
        for (int j = 0; j < 16; ++j) {
          buf_rcs[j] = opt->colorimetry.gamut.local.Rcs_hj_percent[j];
          buf_rhs[j] = opt->colorimetry.gamut.local.Rhs_hj[j];
        }
        d["rcs_hj"] = arr_rcs;
        d["rhs_hj"] = arr_rhs;
      }

      if (extras) {
        // Rf,hj + DE_hj (16 values each)
        auto arr_rfhj = np.attr("empty")(16, nb::arg("dtype") = "float64");
        auto arr_dehj = np.attr("empty")(16, nb::arg("dtype") = "float64");
        auto nd_rfhj = nb::cast<nb::ndarray<>>(arr_rfhj);
        auto nd_dehj = nb::cast<nb::ndarray<>>(arr_dehj);
        double *buf_rfhj = static_cast<double *>(nd_rfhj.data());
        double *buf_dehj = static_cast<double *>(nd_dehj.data());
        for (int j = 0; j < 16; ++j) {
          buf_rfhj[j] = opt->colorimetry.gamut.local.Rf_hj[j];
          buf_dehj[j] = opt->colorimetry.gamut.local.DE_hj[j];
        }
        d["rf_hj"] = arr_rfhj;
        d["de_hj"] = arr_dehj;

        // CVG coordinates (6 arrays of 16 values each)
        const auto &cvg = opt->colorimetry.gamut.cvg;
        auto copy16 = [&](const std::array<double, 16> &src) {
          auto arr = np.attr("empty")(16, nb::arg("dtype") = "float64");
          auto nd = nb::cast<nb::ndarray<>>(arr);
          double *buf = static_cast<double *>(nd.data());
          for (int j = 0; j < 16; ++j)
            buf[j] = src[j];
          return arr;
        };
        d["cvg_j_test"] = copy16(cvg.J_test);
        d["cvg_x_test"] = copy16(cvg.x_test);
        d["cvg_y_test"] = copy16(cvg.y_test);
        d["cvg_j_ref"] = copy16(cvg.J_ref);
        d["cvg_x_ref"] = copy16(cvg.x_ref);
        d["cvg_y_ref"] = copy16(cvg.y_ref);

        // Reference-illuminant SPD, resampled to the input grid (variable
        // length)
        const auto &ref_spd = opt->colorimetry.reference_spd_values;
        auto arr_refspd =
            np.attr("empty")(ref_spd.size(), nb::arg("dtype") = "float64");
        auto nd_refspd = nb::cast<nb::ndarray<>>(arr_refspd);
        double *buf_refspd = static_cast<double *>(nd_refspd.data());
        std::copy(ref_spd.begin(), ref_spd.end(), buf_refspd);
        d["reference_spd"] = arr_refspd;

        // Per-CES XYZ under test / reference illuminant (99x3 each)
        auto copy99x3 = [&](const std::array<tm30::XyzTriple, 99> &src) {
          auto arr = np.attr("empty")(nb::make_tuple(99, 3),
                                      nb::arg("dtype") = "float64");
          auto nd = nb::cast<nb::ndarray<>>(arr);
          double *buf = static_cast<double *>(nd.data());
          for (size_t i = 0; i < 99; ++i) {
            buf[i * 3 + 0] = src[i].X;
            buf[i * 3 + 1] = src[i].Y;
            buf[i * 3 + 2] = src[i].Z;
          }
          return arr;
        };
        d["xyz_test_ces"] = copy99x3(opt->colorimetry.xyz_test_ces);
        d["xyz_ref_ces"] = copy99x3(opt->colorimetry.xyz_ref_ces);

        // Per-CES CAM02-UCS [J', a', b'] under test / reference illuminant
        // (99x3 each)
        auto copy99x3_jab = [&](const std::array<tm30::Cam02Ucs, 99> &src) {
          auto arr = np.attr("empty")(nb::make_tuple(99, 3),
                                      nb::arg("dtype") = "float64");
          auto nd = nb::cast<nb::ndarray<>>(arr);
          double *buf = static_cast<double *>(nd.data());
          for (size_t i = 0; i < 99; ++i) {
            buf[i * 3 + 0] = src[i].J_prime;
            buf[i * 3 + 1] = src[i].a_prime;
            buf[i * 3 + 2] = src[i].b_prime;
          }
          return arr;
        };
        d["jab_test_ces"] = copy99x3_jab(opt->colorimetry.jab_test_ces);
        d["jab_ref_ces"] = copy99x3_jab(opt->colorimetry.jab_ref_ces);

        // Per-CES hue-angle bin index (0-15), reference-hue-assigned - TM-30-20
        // §4.3
        auto arr_binidx = np.attr("empty")(99, nb::arg("dtype") = "int64");
        auto nd_binidx = nb::cast<nb::ndarray<>>(arr_binidx);
        int64_t *buf_binidx = static_cast<int64_t *>(nd_binidx.data());
        for (int j = 0; j < 16; ++j) {
          for (int ces_idx : opt->colorimetry.hue_bins[j]) {
            buf_binidx[ces_idx] = j;
          }
        }
        d["hue_bin_index"] = arr_binidx;
      }

      out.append(d);
    }
    return out;
  }

  /// Mirrors evaluate() exactly (same dict-construction logic) but runs
  /// the grid-fixed cached path - compute_ces_colorimetry_cached() via
  /// try_evaluate_cached() - using the tables set up by set_fixed_grid().
  /// No CES/CMF/daylight-basis resampling happens here; that work was
  /// already done once, at set_fixed_grid() time.
  nb::list evaluate_cached(bool bins, bool samples, bool extras,
                           int n_workers) {
    if (n_workers < 1) {
      throw std::invalid_argument(
          "n_workers must be >= 1 (got " + std::to_string(n_workers) +
          "); n_workers=0/-1 auto-detection is not implemented");
    }
    if (!fixed_tables_.has_value()) {
      throw std::runtime_error(
          "evaluate_cached() called before set_fixed_grid() was ever "
          "called - the grid-fixed resampled tables are not initialized.");
    }
    tm30::Tm30Request req{bins, samples};
    // Same scoped-GIL-release pattern as evaluate() above.
    auto results = [&]() {
      nb::gil_scoped_release release;
      return tm30::try_evaluate_cached(views, *fixed_tables_, planckian_lut,
                                       req, static_cast<std::size_t>(n_workers),
                                       pool_.get());
    }();
    nb::list out;
    auto np = nb::module_::import_("numpy");
    for (auto &opt : results) {
      if (!opt) {
        out.append(nb::none());
        continue;
      }
      nb::dict d;
      d["rf"] = opt->colorimetry.Rf;
      d["rg"] = opt->colorimetry.gamut.Rg;
      d["cct"] = opt->colorimetry.cct;
      d["duv"] = opt->colorimetry.duv;
      d["delta_e_avg"] = opt->colorimetry.delta_e_avg;
      d["rf_skin"] = opt->colorimetry.rf_skin;

      if (samples) {
        // rf_cesi as numpy array
        auto arr_cesi = np.attr("empty")(99, nb::arg("dtype") = "float64");
        auto nd_cesi = nb::cast<nb::ndarray<>>(arr_cesi);
        double *buf_cesi = static_cast<double *>(nd_cesi.data());
        std::copy(opt->colorimetry.rf_cesi.begin(),
                  opt->colorimetry.rf_cesi.end(), buf_cesi);
        d["rf_cesi"] = arr_cesi;
      }

      if (bins) {
        // Per-bin as numpy arrays
        auto arr_rcs = np.attr("empty")(16, nb::arg("dtype") = "float64");
        auto arr_rhs = np.attr("empty")(16, nb::arg("dtype") = "float64");
        auto nd_rcs = nb::cast<nb::ndarray<>>(arr_rcs);
        auto nd_rhs = nb::cast<nb::ndarray<>>(arr_rhs);
        double *buf_rcs = static_cast<double *>(nd_rcs.data());
        double *buf_rhs = static_cast<double *>(nd_rhs.data());
        for (int j = 0; j < 16; ++j) {
          buf_rcs[j] = opt->colorimetry.gamut.local.Rcs_hj_percent[j];
          buf_rhs[j] = opt->colorimetry.gamut.local.Rhs_hj[j];
        }
        d["rcs_hj"] = arr_rcs;
        d["rhs_hj"] = arr_rhs;
      }

      if (extras) {
        // Rf,hj + DE_hj (16 values each)
        auto arr_rfhj = np.attr("empty")(16, nb::arg("dtype") = "float64");
        auto arr_dehj = np.attr("empty")(16, nb::arg("dtype") = "float64");
        auto nd_rfhj = nb::cast<nb::ndarray<>>(arr_rfhj);
        auto nd_dehj = nb::cast<nb::ndarray<>>(arr_dehj);
        double *buf_rfhj = static_cast<double *>(nd_rfhj.data());
        double *buf_dehj = static_cast<double *>(nd_dehj.data());
        for (int j = 0; j < 16; ++j) {
          buf_rfhj[j] = opt->colorimetry.gamut.local.Rf_hj[j];
          buf_dehj[j] = opt->colorimetry.gamut.local.DE_hj[j];
        }
        d["rf_hj"] = arr_rfhj;
        d["de_hj"] = arr_dehj;

        // CVG coordinates (6 arrays of 16 values each)
        const auto &cvg = opt->colorimetry.gamut.cvg;
        auto copy16 = [&](const std::array<double, 16> &src) {
          auto arr = np.attr("empty")(16, nb::arg("dtype") = "float64");
          auto nd = nb::cast<nb::ndarray<>>(arr);
          double *buf = static_cast<double *>(nd.data());
          for (int j = 0; j < 16; ++j)
            buf[j] = src[j];
          return arr;
        };
        d["cvg_j_test"] = copy16(cvg.J_test);
        d["cvg_x_test"] = copy16(cvg.x_test);
        d["cvg_y_test"] = copy16(cvg.y_test);
        d["cvg_j_ref"] = copy16(cvg.J_ref);
        d["cvg_x_ref"] = copy16(cvg.x_ref);
        d["cvg_y_ref"] = copy16(cvg.y_ref);

        // Reference-illuminant SPD, resampled to the input grid (variable
        // length)
        const auto &ref_spd = opt->colorimetry.reference_spd_values;
        auto arr_refspd =
            np.attr("empty")(ref_spd.size(), nb::arg("dtype") = "float64");
        auto nd_refspd = nb::cast<nb::ndarray<>>(arr_refspd);
        double *buf_refspd = static_cast<double *>(nd_refspd.data());
        std::copy(ref_spd.begin(), ref_spd.end(), buf_refspd);
        d["reference_spd"] = arr_refspd;

        // Per-CES XYZ under test / reference illuminant (99x3 each)
        auto copy99x3 = [&](const std::array<tm30::XyzTriple, 99> &src) {
          auto arr = np.attr("empty")(nb::make_tuple(99, 3),
                                      nb::arg("dtype") = "float64");
          auto nd = nb::cast<nb::ndarray<>>(arr);
          double *buf = static_cast<double *>(nd.data());
          for (size_t i = 0; i < 99; ++i) {
            buf[i * 3 + 0] = src[i].X;
            buf[i * 3 + 1] = src[i].Y;
            buf[i * 3 + 2] = src[i].Z;
          }
          return arr;
        };
        d["xyz_test_ces"] = copy99x3(opt->colorimetry.xyz_test_ces);
        d["xyz_ref_ces"] = copy99x3(opt->colorimetry.xyz_ref_ces);

        // Per-CES CAM02-UCS [J', a', b'] under test / reference illuminant
        // (99x3 each)
        auto copy99x3_jab = [&](const std::array<tm30::Cam02Ucs, 99> &src) {
          auto arr = np.attr("empty")(nb::make_tuple(99, 3),
                                      nb::arg("dtype") = "float64");
          auto nd = nb::cast<nb::ndarray<>>(arr);
          double *buf = static_cast<double *>(nd.data());
          for (size_t i = 0; i < 99; ++i) {
            buf[i * 3 + 0] = src[i].J_prime;
            buf[i * 3 + 1] = src[i].a_prime;
            buf[i * 3 + 2] = src[i].b_prime;
          }
          return arr;
        };
        d["jab_test_ces"] = copy99x3_jab(opt->colorimetry.jab_test_ces);
        d["jab_ref_ces"] = copy99x3_jab(opt->colorimetry.jab_ref_ces);

        // Per-CES hue-angle bin index (0-15), reference-hue-assigned - TM-30-20
        // §4.3
        auto arr_binidx = np.attr("empty")(99, nb::arg("dtype") = "int64");
        auto nd_binidx = nb::cast<nb::ndarray<>>(arr_binidx);
        int64_t *buf_binidx = static_cast<int64_t *>(nd_binidx.data());
        for (int j = 0; j < 16; ++j) {
          for (int ces_idx : opt->colorimetry.hue_bins[j]) {
            buf_binidx[ces_idx] = j;
          }
        }
        d["hue_bin_index"] = arr_binidx;
      }

      out.append(d);
    }
    return out;
  }

  /// Compute source XYZ for all SPDs in the batch (N_spds x N_wl).
  /// Returns a numpy array of shape (N_spds, 3) -- [X, Y, Z] per SPD.
  /// Uses the pre-loaded CIE 1964 10 degree CMFs.
  nb::object spd_to_xyz(nb::ndarray<> spd_matrix, nb::object wl_arg,
                        nb::object K_arg, nb::object lambda_min_arg,
                        nb::object lambda_max_arg, nb::object cmf_path_arg) {
    if (spd_matrix.ndim() != 2)
      throw std::invalid_argument("spd_matrix must be 2-D (N_spds x N_wl)");
    require_c_contiguous(spd_matrix, "spd_matrix");
    size_t N = spd_matrix.shape(0);
    size_t nwl = spd_matrix.shape(1);
    const double *data = static_cast<const double *>(spd_matrix.data());

    std::vector<double> wl;
    if (wl_arg.is_none()) {
      if (nwl != 401)
        throw std::invalid_argument(
            "Expected 401 wavelengths (380-780 nm), got " +
            std::to_string(nwl));
      wl.resize(401);
      for (size_t i = 0; i < 401; ++i)
        wl[i] = 380.0 + i;
    } else {
      auto wl_arr = nb::cast<nb::ndarray<>>(wl_arg);
      if (wl_arr.ndim() != 1 || wl_arr.shape(0) != nwl)
        throw std::invalid_argument(
            "wavelengths must match spd_matrix columns");
      require_c_contiguous(wl_arr, "wavelengths");
      const double *wl_data = static_cast<const double *>(wl_arr.data());
      wl.assign(wl_data, wl_data + nwl);
    }

    std::vector<std::vector<double>> spd_vecs(N);
    for (size_t i = 0; i < N; ++i) {
      spd_vecs[i].assign(data + i * nwl, data + (i + 1) * nwl);
    }

    std::optional<double> K_opt;
    if (!K_arg.is_none()) {
      K_opt = nb::cast<double>(K_arg);
    }
    std::optional<double> lambda_min_opt;
    if (!lambda_min_arg.is_none()) {
      lambda_min_opt = nb::cast<double>(lambda_min_arg);
    }
    std::optional<double> lambda_max_opt;
    if (!lambda_max_arg.is_none()) {
      lambda_max_opt = nb::cast<double>(lambda_max_arg);
    }

    tm30::CmfData fresh_cmf;
    const tm30::CmfData *cmf_to_use = &cmf_10deg;
    if (!cmf_path_arg.is_none()) {
      fresh_cmf = load_cmf(nb::cast<std::string>(cmf_path_arg));
      cmf_to_use = &fresh_cmf;
    }

    auto xyzs = tm30::spd_to_xyz_batch(wl, spd_vecs, *cmf_to_use, K_opt,
                                       lambda_min_opt, lambda_max_opt);

    auto np = nb::module_::import_("numpy");
    auto result =
        np.attr("empty")(nb::make_tuple(N, 3), nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (size_t i = 0; i < N; ++i) {
      buf[i * 3 + 0] = xyzs[i].X;
      buf[i * 3 + 1] = xyzs[i].Y;
      buf[i * 3 + 2] = xyzs[i].Z;
    }
    return result;
  }

  /// Compute CIE 1976 Y,u',v' for all SPDs in the batch (N_spds x N_wl).
  /// Returns a numpy array of shape (N_spds, 3) -- [Y, u', v'] per SPD.
  /// Chains spd_to_xyz then xyz_to_Yuv using pre-loaded CIE 1964 10 deg CMFs.
  nb::object spd_to_Yuv(nb::ndarray<> spd_matrix, nb::object wl_arg,
                        nb::object K_arg, nb::object lambda_min_arg,
                        nb::object lambda_max_arg, nb::object cmf_path_arg) {
    if (spd_matrix.ndim() != 2)
      throw std::invalid_argument("spd_matrix must be 2-D (N_spds x N_wl)");
    require_c_contiguous(spd_matrix, "spd_matrix");
    size_t N = spd_matrix.shape(0);
    size_t nwl = spd_matrix.shape(1);
    const double *data = static_cast<const double *>(spd_matrix.data());

    std::vector<double> wl;
    if (wl_arg.is_none()) {
      if (nwl != 401)
        throw std::invalid_argument(
            "Expected 401 wavelengths (380-780 nm), got " +
            std::to_string(nwl));
      wl.resize(401);
      for (size_t i = 0; i < 401; ++i)
        wl[i] = 380.0 + i;
    } else {
      auto wl_arr = nb::cast<nb::ndarray<>>(wl_arg);
      if (wl_arr.ndim() != 1 || wl_arr.shape(0) != nwl)
        throw std::invalid_argument(
            "wavelengths must match spd_matrix columns");
      require_c_contiguous(wl_arr, "wavelengths");
      const double *wl_data = static_cast<const double *>(wl_arr.data());
      wl.assign(wl_data, wl_data + nwl);
    }

    std::vector<std::vector<double>> spd_vecs(N);
    for (size_t i = 0; i < N; ++i) {
      spd_vecs[i].assign(data + i * nwl, data + (i + 1) * nwl);
    }

    std::optional<double> K_opt;
    if (!K_arg.is_none()) {
      K_opt = nb::cast<double>(K_arg);
    }
    std::optional<double> lambda_min_opt;
    if (!lambda_min_arg.is_none()) {
      lambda_min_opt = nb::cast<double>(lambda_min_arg);
    }
    std::optional<double> lambda_max_opt;
    if (!lambda_max_arg.is_none()) {
      lambda_max_opt = nb::cast<double>(lambda_max_arg);
    }

    tm30::CmfData fresh_cmf;
    const tm30::CmfData *cmf_to_use = &cmf_10deg;
    if (!cmf_path_arg.is_none()) {
      fresh_cmf = load_cmf(nb::cast<std::string>(cmf_path_arg));
      cmf_to_use = &fresh_cmf;
    }

    auto yuvs = tm30::spd_to_Yuv_batch(wl, spd_vecs, *cmf_to_use, K_opt,
                                       lambda_min_opt, lambda_max_opt);

    auto np = nb::module_::import_("numpy");
    auto result =
        np.attr("empty")(nb::make_tuple(N, 3), nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (size_t i = 0; i < N; ++i) {
      buf[i * 3 + 0] = yuvs[i].Y;
      buf[i * 3 + 1] = yuvs[i].u_prime;
      buf[i * 3 + 2] = yuvs[i].v_prime;
    }
    return result;
  }

  /// Convert XYZ tristimulus triples to CIE 1976 Y,u',v' (N x 3 in, N x 3
  /// out). Pure coordinate transform - no CMF or wavelength dependency.
  nb::object xyz_to_Yuv(nb::ndarray<> xyz_matrix) {
    if (xyz_matrix.ndim() != 2 || xyz_matrix.shape(1) != 3)
      throw std::invalid_argument("xyz_matrix must be 2-D (N x 3)");
    require_c_contiguous(xyz_matrix, "xyz_matrix");
    size_t N = xyz_matrix.shape(0);
    const double *data = static_cast<const double *>(xyz_matrix.data());

    std::vector<tm30::XyzTriple> xyzs(N);
    for (size_t i = 0; i < N; ++i) {
      xyzs[i] =
          tm30::XyzTriple{data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]};
    }
    auto yuvs = tm30::xyz_to_Yuv_batch(xyzs);

    auto np = nb::module_::import_("numpy");
    auto result =
        np.attr("empty")(nb::make_tuple(N, 3), nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (size_t i = 0; i < N; ++i) {
      buf[i * 3 + 0] = yuvs[i].Y;
      buf[i * 3 + 1] = yuvs[i].u_prime;
      buf[i * 3 + 2] = yuvs[i].v_prime;
    }
    return result;
  }

  /// Compute XYZ for the reference illuminant at each CCT (N,) in,
  /// returns (N, 3). Always uses this context's grid-fixed wavelengths (set
  /// via set_fixed_grid()) - no per-call wavelengths override. cmf_path=None
  /// (default): use this context's bound cmf_10deg. cmf_path=str: load+
  /// resample a different CMF for this call only.
  nb::object cct_to_xyz(nb::ndarray<> cct_array, nb::object cmf_path_arg,
                        nb::object K_arg) {
    if (!fixed_tables_.has_value())
      throw std::runtime_error(
          "cct_to_xyz() called before set_fixed_grid() was ever called.");
    if (cct_array.ndim() != 1)
      throw std::invalid_argument("cct_array must be a 1-D array");
    require_c_contiguous(cct_array, "cct_array");
    size_t N = cct_array.shape(0);
    const double *cct_data = static_cast<const double *>(cct_array.data());
    std::vector<double> ccts(cct_data, cct_data + N);

    std::optional<double> K_opt;
    if (!K_arg.is_none())
      K_opt = nb::cast<double>(K_arg);

    std::vector<tm30::XyzTriple> xyzs;
    if (cmf_path_arg.is_none()) {
      xyzs = tm30::cct_to_xyz_batch(ccts, fixed_tables_->wavelengths,
                                    fixed_tables_->daylight_basis, cmf_10deg,
                                    K_opt);
    } else {
      tm30::CmfData fresh_cmf = load_cmf(nb::cast<std::string>(cmf_path_arg));
      xyzs = tm30::cct_to_xyz_batch(ccts, fixed_tables_->wavelengths,
                                    fixed_tables_->daylight_basis, fresh_cmf,
                                    K_opt);
    }

    auto np = nb::module_::import_("numpy");
    auto result =
        np.attr("empty")(nb::make_tuple(N, 3), nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (size_t i = 0; i < N; ++i) {
      buf[i * 3 + 0] = xyzs[i].X;
      buf[i * 3 + 1] = xyzs[i].Y;
      buf[i * 3 + 2] = xyzs[i].Z;
    }
    return result;
  }

  /// Integrate each SPD in the batch to a single power value. Returns
  /// shape (N,) - NOT (N,3) - power is one scalar per SPD.
  /// photometric=false (default): radiometric (W). photometric=true:
  /// Km=683.0 x ybar-weighted (lm). cmf_path=None: use this context's
  /// bound cmf_10deg (ignored when photometric=false).
  nb::object spd_to_power(nb::ndarray<> spd_matrix, nb::object wl_arg,
                          nb::object cmf_path_arg, bool photometric,
                          nb::object lambda_min_arg,
                          nb::object lambda_max_arg) {
    if (spd_matrix.ndim() != 2)
      throw std::invalid_argument("spd_matrix must be 2-D (N_spds x N_wl)");
    require_c_contiguous(spd_matrix, "spd_matrix");
    size_t N = spd_matrix.shape(0);
    size_t nwl = spd_matrix.shape(1);
    const double *data = static_cast<const double *>(spd_matrix.data());

    std::vector<double> wl;
    if (wl_arg.is_none()) {
      if (nwl != 401)
        throw std::invalid_argument(
            "Expected 401 wavelengths (380-780 nm), got " +
            std::to_string(nwl));
      wl.resize(401);
      for (size_t i = 0; i < 401; ++i)
        wl[i] = 380.0 + i;
    } else {
      auto wl_arr = nb::cast<nb::ndarray<>>(wl_arg);
      if (wl_arr.ndim() != 1 || wl_arr.shape(0) != nwl)
        throw std::invalid_argument(
            "wavelengths must match spd_matrix columns");
      require_c_contiguous(wl_arr, "wavelengths");
      const double *wl_data = static_cast<const double *>(wl_arr.data());
      wl.assign(wl_data, wl_data + nwl);
    }

    std::vector<std::vector<double>> spd_vecs(N);
    for (size_t i = 0; i < N; ++i) {
      spd_vecs[i].assign(data + i * nwl, data + (i + 1) * nwl);
    }

    std::optional<double> lambda_min_opt;
    if (!lambda_min_arg.is_none())
      lambda_min_opt = nb::cast<double>(lambda_min_arg);
    std::optional<double> lambda_max_opt;
    if (!lambda_max_arg.is_none())
      lambda_max_opt = nb::cast<double>(lambda_max_arg);

    std::vector<double> results;
    if (cmf_path_arg.is_none()) {
      results = tm30::spd_to_power_batch(wl, spd_vecs, cmf_10deg, photometric,
                                         lambda_min_opt, lambda_max_opt);
    } else {
      tm30::CmfData fresh_cmf = load_cmf(nb::cast<std::string>(cmf_path_arg));
      results = tm30::spd_to_power_batch(wl, spd_vecs, fresh_cmf, photometric,
                                         lambda_min_opt, lambda_max_opt);
    }

    auto np = nb::module_::import_("numpy");
    auto result = np.attr("empty")(N, nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    std::copy(results.begin(), results.end(), buf);
    return result;
  }

  /// Compute CCT and Duv for all SPDs in the batch (N_spds x N_wl).
  /// Returns a numpy array of shape (2, N_spds): row 0 = cct (K), row 1 =
  /// duv. Uses the pre-loaded CIE 1931 2-deg CMFs by default (TM-30-20
  /// §3.1 exception - CCT determination is the one calculation that uses
  /// the 2-deg, not 10-deg, observer). cmf_path=None: use this context's
  /// bound cmf_2deg. cmf_path=str: load+resample a different 2-deg CMF for
  /// this call only.
  nb::object spd_to_cct(nb::ndarray<> spd_matrix, nb::object wl_arg,
                        nb::object cmf_path_arg) {
    if (spd_matrix.ndim() != 2)
      throw std::invalid_argument("spd_matrix must be 2-D (N_spds x N_wl)");
    require_c_contiguous(spd_matrix, "spd_matrix");
    size_t N = spd_matrix.shape(0);
    size_t nwl = spd_matrix.shape(1);
    const double *data = static_cast<const double *>(spd_matrix.data());

    std::vector<double> wl;
    if (wl_arg.is_none()) {
      if (nwl != 401)
        throw std::invalid_argument(
            "Expected 401 wavelengths (380-780 nm), got " +
            std::to_string(nwl));
      wl.resize(401);
      for (size_t i = 0; i < 401; ++i)
        wl[i] = 380.0 + i;
    } else {
      auto wl_arr = nb::cast<nb::ndarray<>>(wl_arg);
      if (wl_arr.ndim() != 1 || wl_arr.shape(0) != nwl)
        throw std::invalid_argument(
            "wavelengths must match spd_matrix columns");
      require_c_contiguous(wl_arr, "wavelengths");
      const double *wl_data = static_cast<const double *>(wl_arr.data());
      wl.assign(wl_data, wl_data + nwl);
    }

    std::vector<std::vector<double>> spd_vecs(N);
    for (size_t i = 0; i < N; ++i) {
      spd_vecs[i].assign(data + i * nwl, data + (i + 1) * nwl);
    }

    tm30::CmfData fresh_cmf;
    const tm30::CmfData *cmf_to_use = &cmf_2deg;
    if (!cmf_path_arg.is_none()) {
      fresh_cmf = load_cmf(nb::cast<std::string>(cmf_path_arg));
      cmf_to_use = &fresh_cmf;
    }

    auto results =
        tm30::spd_to_cct_batch(wl, spd_vecs, *cmf_to_use, planckian_lut);

    auto np = nb::module_::import_("numpy");
    auto result =
        np.attr("empty")(nb::make_tuple(2, N), nb::arg("dtype") = "float64");
    auto nd = nb::cast<nb::ndarray<>>(result);
    double *buf = static_cast<double *>(nd.data());
    for (size_t i = 0; i < N; ++i) {
      buf[i] = results[i].cct;     // row 0, contiguous
      buf[N + i] = results[i].duv; // row 1, contiguous
    }
    return result;
  }
};

// ==========================================================================
//  Module
// ==========================================================================

NB_MODULE(tm30_core, m) {
  m.doc() = R"pbdoc(
    TM-30-20 Colour Rendition - C++20 core with nanobind Python bindings.

    Usage:
        import numpy as np
        import tm30_core

        # From numpy array (default 380-780 nm at 1 nm):
        spd = np.loadtxt('my_spectrum.csv')
        m = tm30_core.Tm30(spd)

        # With explicit wavelengths:
        m = tm30_core.Tm30(spd_values, spd_wavelengths)

        print(m.rf, m.rg, m.cct, m.duv)
  )pbdoc";
  m.attr("__version__") = "0.1.0";

  // -- Validity ------------------------------------------------------

  nb::class_<tm30::Validity>(
      m, "Validity",
      "Domain validity flags for TM-30 results (advisory warnings).")
      .def(nb::init<>())
      .def_rw("duv_out_of_range", &tm30::Validity::duv_out_of_range,
              "Duv far from the Planckian locus (pytm30 advisory; TM-30-20 "
              "§2.0 prints no numerical Duv bound).")
      .def_rw("cct_out_of_range", &tm30::Validity::cct_out_of_range,
              "CCT far from the range where TM-30 is typically applied "
              "(pytm30 advisory; TM-30-20 §2.0 prints no numerical CCT "
              "bounds).")
      .def_rw("extrapolated", &tm30::Validity::extrapolated,
              "Test SPD does not cover 380-780 nm; zero-fill was applied "
              "per TM-30-20 §3.5. CES/CMF tables are flat-extrapolated "
              "per §1.3 / Annex A -- unrelated to this flag.");

  // -- Tm30 class --------------------------------------------------

  nb::class_<PyTm30>(m, "Tm30",
                     R"pbdoc(
      Lazy, memoized TM-30 evaluator for a single SPD.

      Construction validates the SPD but computes nothing.  The full
      pipeline runs on first property access (rf, rg, cct, duv, ...)
      and results are cached.

      Parameters
      ----------
      spd_values : numpy.ndarray (1-D, float64)
          Spectral power values St(lambda).
      spd_wavelengths : numpy.ndarray (1-D, float64) or None, optional
          Wavelength grid in nm.  If None, defaults to 380-780 nm at 1 nm step.
      data_dir : str, optional
          Directory containing the TM-30 data files (ces.csv,
          cmf_1964_10.csv, etc.).
      )pbdoc")
      .def(nb::init<nb::object, nb::object, const std::string &>(),
           nb::arg("spd_values"), nb::arg("spd_wavelengths") = nb::none(),
           nb::arg("data_dir") = std::string(TM30_DATA_DIR),
           "Create a Tm30 evaluator from a numpy array of SPD values.\n"
           "spd_wavelengths defaults to 380-780 nm (1 nm step) if None.")
      .def_prop_ro("rf", &PyTm30::rf,
                   "Fidelity index Rf [0, 100].  TM-30-20 §4.1.")
      .def_prop_ro("rg", &PyTm30::rg, "Gamut area index Rg.  TM-30-20 §4.4.")
      .def_prop_ro("cct", &PyTm30::cct,
                   "Correlated Color Temperature (K).  TM-30-20 §3.3.")
      .def_prop_ro(
          "duv", &PyTm30::duv,
          "Distance from Planckian locus in CIE 1960 UCS.  TM-30-20 §3.3.")
      .def_prop_ro("delta_e_avg", &PyTm30::delta_e_avg,
                   "Average dE' across 99 CES.  TM-30-20 §4.1.")
      .def_prop_ro(
          "rf_skin", &PyTm30::rf_skin,
          "Skin fidelity Rf,skin (average of CES15 + CES18).  PyTM30 "
          "research extension informed by TM-30-20 §4.2; not a "
          "standardised TM-30 measure (§1.2, §4.0).")
      .def_prop_ro("rf_cesi", &PyTm30::rf_cesi,
                   "Per-sample fidelity Rf,CESi - numpy array of 99 values.  "
                   "TM-30-20 §4.2.")
      .def_prop_ro("rcs_hj", &PyTm30::rcs_hj,
                   "Per-bin chroma shift Rcs,hj, in percent - numpy array of "
                   "16 values.  TM-30-20 §4.6 (percentage representation).")
      .def_prop_ro("rhs_hj", &PyTm30::rhs_hj,
                   "Per-bin hue shift Rhs,hj, dimensionless ratio - numpy "
                   "array of 16 values.  TM-30-20 §4.7.")
      .def_prop_ro("rf_hj", &PyTm30::rf_hj,
                   "Per-bin local fidelity Rf,hj - numpy array of 16 values.  "
                   "TM-30-20 §4.8.")
      .def_prop_ro(
          "de_hj", &PyTm30::de_hj,
          "Per-bin mean dE', DE_hj - numpy array of 16 values.  TM-30-20 §4.8.")
      .def_prop_ro(
          "cvg_j_test", &PyTm30::cvg_j_test,
          "CVG test-vector J' - numpy array of 16 values.  TM-30-20 §4.5.")
      .def_prop_ro("cvg_x_test", &PyTm30::cvg_x_test,
                   "CVG test-vector x - numpy array of 16 values.  TM-30-20 "
                   "§4.5 Eq. (60).")
      .def_prop_ro("cvg_y_test", &PyTm30::cvg_y_test,
                   "CVG test-vector y - numpy array of 16 values.  TM-30-20 "
                   "§4.5 Eq. (61).")
      .def_prop_ro(
          "cvg_j_ref", &PyTm30::cvg_j_ref,
          "CVG reference-circle J' - numpy array of 16 values.  TM-30-20 §4.5.")
      .def_prop_ro("cvg_x_ref", &PyTm30::cvg_x_ref,
                   "CVG reference-circle x - numpy array of 16 values.  "
                   "TM-30-20 §4.5 Eq. (58).")
      .def_prop_ro("cvg_y_ref", &PyTm30::cvg_y_ref,
                   "CVG reference-circle y - numpy array of 16 values.  "
                   "TM-30-20 §4.5 Eq. (59).")
      .def_prop_ro(
          "reference_spd", &PyTm30::reference_spd,
          "Reference-illuminant SPD, resampled to the input wavelength grid - "
          "numpy array.  TM-30-20 §3.3 Eq. (13)-(16).")
      .def_prop_ro(
          "xyz_test_ces", &PyTm30::xyz_test_ces,
          "Per-CES XYZ under the test source - numpy array of shape (99, 3).  "
          "TM-30-20 §3.6 Eq. (21)-(23).")
      .def_prop_ro(
          "xyz_ref_ces", &PyTm30::xyz_ref_ces,
          "Per-CES XYZ under the reference illuminant - numpy array of shape "
          "(99, 3).  TM-30-20 §3.6 Eq. (25)-(27).")
      .def_prop_ro(
          "jab_test_ces", &PyTm30::jab_test_ces,
          "Per-CES CAM02-UCS [J', a', b'] under the test source - numpy array "
          "of shape (99, 3).  TM-30-20 §3.7.1 Eq. (48)-(50).")
      .def_prop_ro(
          "jab_ref_ces", &PyTm30::jab_ref_ces,
          "Per-CES CAM02-UCS [J', a', b'] under the reference illuminant - "
          "numpy array of shape (99, 3).  TM-30-20 §3.7.1 Eq. (48)-(50).")
      .def_prop_ro(
          "hue_bin_index", &PyTm30::hue_bin_index,
          "Per-CES hue-angle bin index (0-15), assigned from the reference "
          "hue angle hr = atan2(b'r, a'r) - numpy int array of 99 values.  "
          "TM-30-20 §4.3.")
      .def_prop_ro("validity", &PyTm30::validity,
                   "Domain validity flags (Validity named tuple).");

  // -- Batch evaluation -------------------------------------------

  nb::class_<BatchContext>(
      m, "BatchContext",
      "Pre-loaded data tables for batch TM-30 evaluation.\\n"
      "Call prepare_batch() then evaluate() to process many SPDs.")
      .def(nb::init<const std::string &, int, bool>(),
           nb::arg("data_dir") = std::string(TM30_DATA_DIR),
           nb::arg("n_workers") = 1, nb::arg("persistent_workers") = false,
           "Create a batch context with pre-loaded data tables.\n"
           "CMF paths default to cmf_1964_10.csv / cie_1931_2.csv.\n"
           "n_workers>1 parallelizes across SPDs (bit-identical results);\n"
           "persistent_workers=true (with n_workers>1) keeps the worker\n"
           "threads alive across calls instead of spawning per call;\n"
           "persistent_workers with n_workers<=1 is silently inert.")
      .def(nb::init<const std::string &, const std::string &,
                    const std::string &, int, bool>(),
           nb::arg("data_dir"), nb::arg("cmf_2deg_path"),
           nb::arg("cmf_10deg_path"), nb::arg("n_workers") = 1,
           nb::arg("persistent_workers") = false,
           "Create a batch context with explicit CMF file paths.\n"
           "cmf_2deg_path: path to the 2-deg CMF CSV (for CCT).\n"
           "cmf_10deg_path: path to the 10-deg CMF CSV (for tristimulus).\n"
           "n_workers>1 parallelizes across SPDs (bit-identical results);\n"
           "persistent_workers=true (with n_workers>1) keeps the worker\n"
           "threads alive across calls instead of spawning per call;\n"
           "persistent_workers with n_workers<=1 is silently inert.")
      .def("prepare_batch", &BatchContext::prepare_batch, nb::arg("spd_matrix"),
           nb::arg("wavelengths") = nb::none(),
           "Load SPDs from a 2-D numpy array (N_spds x N_wl). "
           "wavelengths defaults to 380-780 nm (1 nm step) if None.")
      .def("evaluate", &BatchContext::evaluate, nb::arg("bins") = true,
           nb::arg("samples") = true, nb::arg("extras") = false,
           nb::arg("n_workers") = 1,
           "Run TM-30 on all prepared SPDs. Returns list of dicts "
           "(or None for failed SPDs). extras=True additionally includes "
           "rf_hj, de_hj, cvg_{j,x,y}_{test,ref}, reference_spd, "
           "xyz_test_ces, and xyz_ref_ces in each dict. n_workers>1 "
           "parallelizes across SPDs (bit-identical results); n_workers<1 "
           "raises ValueError.")
      .def("set_fixed_grid", &BatchContext::set_fixed_grid,
           nb::arg("wavelengths"),
           "Precompute and cache CES/CMF/daylight-basis tables resampled to "
           "`wavelengths`. Call once at construction; evaluate_cached() then "
           "reuses this cache for every SPD sharing this grid.")
      .def("evaluate_cached", &BatchContext::evaluate_cached,
           nb::arg("bins") = true, nb::arg("samples") = true,
           nb::arg("extras") = false, nb::arg("n_workers") = 1,
           "Like evaluate(), but uses the grid-fixed tables cached by "
           "set_fixed_grid() and skips CES/CMF/daylight-basis resampling "
           "entirely. Raises RuntimeError if set_fixed_grid() was never "
           "called. n_workers>1 parallelizes across SPDs (bit-identical "
           "results); n_workers<1 raises ValueError.")
      .def("spd_to_xyz", &BatchContext::spd_to_xyz, nb::arg("spd_matrix"),
           nb::arg("wavelengths") = nb::none(), nb::arg("K") = nb::none(),
           nb::arg("lambda_min") = nb::none(),
           nb::arg("lambda_max") = nb::none(), nb::arg("cmf") = nb::none(),
           "Compute source XYZ for all SPDs (N_spds x N_wl). "
           "Returns numpy array (N_spds, 3) with [X, Y, Z]. "
           "Uses CIE 1964 10 degree CMFs by default, or the CMF at "
           "cmf= (a CSV path) for this call only. "
           "K=None (default): auto-normalise Y=100 (TM-30-20 §3.2). "
           "K=float: use as multiplier for raw integrals (1.0 = raw, 683 = "
           "photometric). "
           "lambda_min, lambda_max=None: clip integration range (nm).")
      .def("spd_to_Yuv", &BatchContext::spd_to_Yuv, nb::arg("spd_matrix"),
           nb::arg("wavelengths") = nb::none(), nb::arg("K") = nb::none(),
           nb::arg("lambda_min") = nb::none(),
           nb::arg("lambda_max") = nb::none(), nb::arg("cmf") = nb::none(),
           "Compute CIE 1976 Y,u',v' for all SPDs (N_spds x N_wl). "
           "Returns numpy array (N_spds, 3) with [Y, u', v']. "
           "Chains spd_to_xyz then xyz_to_Yuv. Uses CIE 1964 10 degree CMFs "
           "by default, or the CMF at cmf= (a CSV path) for this call only. "
           "K=None (default): auto-normalise Y=100. "
           "K=float: use as multiplier for raw integrals. "
           "lambda_min, lambda_max=None: clip integration range (nm).")
      .def("xyz_to_Yuv", &BatchContext::xyz_to_Yuv, nb::arg("xyz_matrix"),
           "Convert XYZ tristimulus triples to CIE 1976 Y,u',v'. "
           "Input/output shape (N, 3). Pure coordinate transform - no CMF "
           "or wavelength dependency.")
      .def("cct_to_xyz", &BatchContext::cct_to_xyz, nb::arg("cct_array"),
           nb::arg("cmf_path") = nb::none(), nb::arg("K") = nb::none(),
           "Compute XYZ for the TM-30-20 reference illuminant at each CCT. "
           "Input shape (N,), output shape (N, 3). Always uses this "
           "context's grid-fixed wavelengths (set via set_fixed_grid()) - "
           "no wavelengths override. cmf_path=None (default): use this "
           "context's bound CMF. cmf_path=str: load+resample a different "
           "CMF for this call only.")
      .def("spd_to_power", &BatchContext::spd_to_power, nb::arg("spd_matrix"),
           nb::arg("wavelengths") = nb::none(),
           nb::arg("cmf_path") = nb::none(), nb::arg("photometric") = false,
           nb::arg("lambda_min") = nb::none(),
           nb::arg("lambda_max") = nb::none(),
           "Integrate each SPD to a single power value. Returns shape (N,), "
           "NOT (N,3) - power is one scalar per SPD. photometric=False "
           "(default): radiometric (W), unweighted. photometric=True: "
           "Km=683.0 x ybar-weighted (lm).")
      .def("spd_to_cct", &BatchContext::spd_to_cct, nb::arg("spd_matrix"),
           nb::arg("wavelengths") = nb::none(), nb::arg("cmf") = nb::none(),
           "Compute CCT and Duv for all SPDs (N_spds x N_wl). Returns "
           "numpy array shape (2, N_spds): row 0 = cct (K), row 1 = duv. "
           "Uses CIE 1931 2-deg CMFs by default (TM-30-20 §3.1 exception), "
           "or the CMF at cmf= (a CSV path) for this call only.");

  // -- Exception translation ----------------------------------------

  nb::register_exception_translator(
      [](const std::exception_ptr &p, void * /*payload*/) {
        try {
          if (p)
            std::rethrow_exception(p);
        } catch (const tm30::InvalidSpd &e) {
          PyErr_SetString(PyExc_ValueError, e.what());
        }
      });
}
