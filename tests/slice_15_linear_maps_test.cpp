// Slice 15 - Linear tristimulus maps (xyz_linear_maps).
//
// TM-30-20 S3.2: Test Source Tristimulus Values (Eq. 1-4)
// TM-30-20 S3.5: Range and Interpolation of Data
// TM-30-20 S3.6: Calculation of Tristimulus Values (Eq. 21-23)

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tm30/csv_loader.hpp"
#include "tm30/pipeline.hpp"
#include "tm30/reference.hpp"
#include "tm30/resample.hpp"
#include "tm30/spd.hpp"
#include "tm30/xyz.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace tm30::test {
namespace {

std::string data_path(const std::string &filename) {
  return std::string(TM30_DATA_DIR) + "/" + filename;
}

/// Load CES reflectance data from a CSV file.
CesData load_ces(const std::string &path) {
  CsvTable table = load_csv(path);
  CesData data;
  for (const auto &row : table.rows) {
    data.wavelengths.push_back(row[0]);
  }
  const std::size_t n_ces = 99; // TM-30-20: 99 CES samples
  data.samples.resize(n_ces);
  for (std::size_t c = 0; c < n_ces; ++c) {
    data.samples[c].reserve(table.rows.size());
    for (const auto &row : table.rows) {
      data.samples[c].push_back(row[1 + c]);
    }
  }
  return data;
}

/// Load CIE 1964 10-deg (or 1931 2-deg) CMF data from a CSV file.
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

/// A smooth, strictly positive synthetic test SPD: a gaussian bump on a
/// constant floor, evaluated directly on the caller's own input grid.
std::vector<double> smooth_test_spd(const std::vector<double> &wl) {
  std::vector<double> s(wl.size());
  for (std::size_t i = 0; i < wl.size(); ++i) {
    const double t = (wl[i] - 560.0) / 80.0;
    s[i] = 0.2 + std::exp(-(t * t));
  }
  return s;
}

/// Uniform wavelength grid from lo to hi inclusive (hi assumed reachable
/// from lo by an integer number of `step`s).
std::vector<double> wl_uniform(double lo, double hi, double step) {
  std::vector<double> wl;
  const int n = static_cast<int>(std::llround((hi - lo) / step)) + 1;
  wl.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    wl.push_back(lo + step * static_cast<double>(i));
  }
  return wl;
}

struct GlobalFixtures {
  CmfData cmf_2deg;
  CmfData cmf_10deg;
  CesData ces;
  DaylightBasis daylight_basis;

  static GlobalFixtures &instance() {
    static GlobalFixtures g;
    return g;
  }

private:
  GlobalFixtures() {
    cmf_10deg = load_cmf(data_path("cmf_1964_10.csv"));
    cmf_2deg = load_cmf(data_path("cie_1931_2.csv"));
    ces = load_ces(data_path("ces.csv"));
    daylight_basis = load_daylight_basis(data_path("daylight_basis.csv"));
  }
};

/// Run the full xyz_linear_maps check for one input wavelength grid:
/// builds a smooth test SPD on `input_wl`, resamples tables on the S3.5-
/// conformed grid, builds the linear maps, and checks them against
/// compute_source_xyz()/compute_ces_xyz() run on the conformed data.
void check_grid(const std::vector<double> &input_wl, const std::string &label) {
  INFO(label);
  auto &G = GlobalFixtures::instance();

  const std::vector<double> s = smooth_test_spd(input_wl);
  const Spd probe(input_wl, s);

  const ResampledTables tables = prepare_resampled_tables(
      probe.wavelengths(), G.cmf_2deg, G.cmf_10deg, G.ces, G.daylight_basis);

  const XyzLinearMaps maps = xyz_linear_maps(input_wl, tables);

  REQUIRE(maps.n_in == input_wl.size());
  REQUIRE(maps.source.size() == 3 * maps.n_in);
  REQUIRE(maps.ces.size() == 99 * 3 * maps.n_in);

  // Reference: source/CES XYZ computed the ordinary way on the conformed
  // grid, using the resampled (tables.cmf_10deg) CMF -- TM-30-20 S3.2
  // Eq. (1)-(4), S3.6 Eq. (21)-(23).
  const SourceXyz src = compute_source_xyz(
      probe.wavelengths(), probe.values(), tables.cmf_10deg.x_bar,
      tables.cmf_10deg.y_bar, tables.cmf_10deg.z_bar);
  const auto ces_xyz =
      compute_ces_xyz(probe.wavelengths(), probe.values(), tables.ces,
                      tables.cmf_10deg.x_bar, tables.cmf_10deg.y_bar,
                      tables.cmf_10deg.z_bar, src.k, &tables.trapezoidal_w);

  const std::size_t n_in = maps.n_in;

  // Ws = W @ s over the INPUT s (not the conformed values) -- columns for
  // dropped/zero-fill points are zero, so this equals the conformed-grid
  // integral. TM-30-20 S3.2 Eq. (1)-(3).
  double Ws[3] = {0.0, 0.0, 0.0};
  for (std::size_t c = 0; c < 3; ++c) {
    double acc = 0.0;
    for (std::size_t col = 0; col < n_in; ++col) {
      acc += maps.source[c * n_in + col] * s[col];
    }
    Ws[c] = acc;
  }

  // TM-30-20 S3.2 Eq. (4)
  const double k = 100.0 / Ws[1];
  REQUIRE_THAT(k, Catch::Matchers::WithinRel(src.k, 1e-12));

  REQUIRE_THAT(k * Ws[0], Catch::Matchers::WithinAbs(src.X, 1e-9));
  REQUIRE_THAT(k * Ws[1], Catch::Matchers::WithinAbs(src.Y, 1e-9));
  REQUIRE_THAT(k * Ws[2], Catch::Matchers::WithinAbs(src.Z, 1e-9));

  for (std::size_t i = 0; i < 99; ++i) {
    INFO("CES index: " << i);
    double Ms[3] = {0.0, 0.0, 0.0};
    for (std::size_t c = 0; c < 3; ++c) {
      double acc = 0.0;
      const double *row = &maps.ces[(i * 3 + c) * n_in];
      for (std::size_t col = 0; col < n_in; ++col) {
        acc += row[col] * s[col];
      }
      Ms[c] = acc;
    }
    REQUIRE_THAT(k * Ms[0], Catch::Matchers::WithinAbs(ces_xyz[i].X, 1e-9));
    REQUIRE_THAT(k * Ms[1], Catch::Matchers::WithinAbs(ces_xyz[i].Y, 1e-9));
    REQUIRE_THAT(k * Ms[2], Catch::Matchers::WithinAbs(ces_xyz[i].Z, 1e-9));
  }
}

// -------------------------------------------------------------------------
// Grid matrix
// -------------------------------------------------------------------------

TEST_CASE("Linear maps - default 1nm 380-780nm grid",
          "[linear_maps][slice15]") {
  check_grid(wl_uniform(380.0, 780.0, 1.0), "1nm-380-780");
}

TEST_CASE("Linear maps - uniform 5nm 380-780nm grid",
          "[linear_maps][slice15]") {
  check_grid(wl_uniform(380.0, 780.0, 5.0), "5nm-380-780");
}

TEST_CASE("Linear maps - 390-770nm step 2 (zero-fill both edges)",
          "[linear_maps][slice15]") {
  auto wl = wl_uniform(390.0, 770.0, 2.0);
  REQUIRE(wl.front() == 390.0);
  REQUIRE(wl.back() == 770.0);
  check_grid(wl, "2nm-390-770-zerofill");
}

TEST_CASE("Linear maps - 370-800nm step 1 (crop both edges)",
          "[linear_maps][slice15]") {
  auto wl = wl_uniform(370.0, 800.0, 1.0);
  REQUIRE(wl.front() == 370.0);
  REQUIRE(wl.back() == 800.0);
  check_grid(wl, "1nm-370-800-crop");

  // Columns for input wavelengths outside 380-780 nm must be exactly
  // zero in both W and every M[i] -- TM-30-20 S3.5 drops those samples
  // from the calculation entirely.
  auto &G = GlobalFixtures::instance();
  const std::vector<double> s = smooth_test_spd(wl);
  const Spd probe(wl, s);
  const ResampledTables tables = prepare_resampled_tables(
      probe.wavelengths(), G.cmf_2deg, G.cmf_10deg, G.ces, G.daylight_basis);
  const XyzLinearMaps maps = xyz_linear_maps(wl, tables);
  const std::size_t n_in = maps.n_in;

  for (std::size_t col = 0; col < n_in; ++col) {
    if (wl[col] >= 380.0 && wl[col] <= 780.0) {
      continue;
    }
    INFO("out-of-range column at wavelength " << wl[col]);
    for (std::size_t c = 0; c < 3; ++c) {
      REQUIRE(maps.source[c * n_in + col] == 0.0);
    }
    for (std::size_t i = 0; i < 99; ++i) {
      for (std::size_t c = 0; c < 3; ++c) {
        REQUIRE(maps.ces[(i * 3 + c) * n_in + col] == 0.0);
      }
    }
  }
}

} // namespace
} // namespace tm30::test
