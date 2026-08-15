// C++-level baseline benchmark for the batch TM-30 path.
// Measures try_evaluate_cached() per-call cost on this machine BEFORE the
// n_workers feature lands, so the Phase 1 timing-regression test can be
// calibrated against real numbers instead of guesses.
//
// Build (from repo root, after cmake -B build):
//   g++ -O3 -std=c++20 -Iinclude -DTM30_DATA_DIR=\"$PWD/data\" \
//       tools/bench_cpp_baseline.cpp build/libtm30-core.a -pthread \
//       -o /tmp/bench_cpp && /tmp/bench_cpp
#include "tm30/cct.hpp"
#include "tm30/csv_loader.hpp"
#include "tm30/pipeline.hpp"
#include "tm30/reference.hpp"
#include "tm30/resample.hpp"
#include "tm30/spd.hpp"
#include "tm30/tm30.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace tm30;

static std::string data_path(const std::string &f) {
  return std::string(TM30_DATA_DIR) + "/" + f;
}

// The bundled illuminant corpus, from data/illuminant_corpus.txt -- the
// single source of truth this tool, the benchmark scripts and
// tests/slice_13_parallel_test.cpp all read, so renaming an SPD is one edit.
static std::vector<std::string> load_spd_names() {
  std::ifstream in(data_path("illuminant_corpus.txt"));
  if (!in)
    throw std::runtime_error("cannot open data/illuminant_corpus.txt");
  std::vector<std::string> names;
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos)
      line.erase(hash);
    // Trims the \r of a CRLF checkout too (data/ is committed -text).
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

static CmfData load_cmf(const std::string &path) {
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

static CesData load_ces(const std::string &path) {
  CsvTable table = load_csv(path);
  CesData data;
  data.samples.resize(table.headers.size() - 1);
  for (const auto &row : table.rows) {
    data.wavelengths.push_back(row[0]);
    for (std::size_t c = 1; c < row.size(); ++c)
      data.samples[c - 1].push_back(row[c]);
  }
  return data;
}

static std::pair<std::vector<double>, std::vector<double>>
load_spd_csv(const std::string &path) {
  CsvTable table = load_csv(path);
  std::vector<double> wl, vals;
  for (const auto &row : table.rows) {
    wl.push_back(row[0]);
    vals.push_back(row[1]);
  }
  return {wl, vals};
}

struct Corpus {
  CmfData cmf_2deg, cmf_10deg;
  CesData ces;
  DaylightBasis basis;
  PlanckianLut lut;
  std::vector<std::vector<double>> wls, vals; // raw per-SPD data
};

static Corpus load_corpus() {
  Corpus c;
  c.cmf_2deg = load_cmf(data_path("cie_1931_2.csv"));
  c.cmf_10deg = load_cmf(data_path("cmf_1964_10.csv"));
  c.ces = load_ces(data_path("ces.csv"));
  c.basis = load_daylight_basis(data_path("daylight_basis.csv"));
  c.lut = load_planckian_lut(data_path("planckian_uv.csv"));
  for (const auto &n : load_spd_names()) {
    auto [wl, v] = load_spd_csv(data_path(n + ".csv"));
    c.wls.push_back(std::move(wl));
    c.vals.push_back(std::move(v));
  }
  return c;
}

using Clock = std::chrono::steady_clock;

static double median_ms(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2] * 1e3;
}

int main() {
  Corpus c = load_corpus();

  // Resample the 5 nm HP SPDs onto the common 1 nm grid so the batch has
  // a uniform width (same as benchmarks/benchmark_tm30.py).
  const std::vector<double> &wl1 = c.wls[0]; // 380..780 @ 1 nm (401 pts)
  std::vector<std::vector<double>> vals1 = c.vals;
  for (size_t i = 0; i < c.wls.size(); ++i) {
    if (c.wls[i].size() == wl1.size())
      continue;
    std::vector<double> out(wl1.size());
    for (size_t j = 0; j < wl1.size(); ++j) {
      const double x = wl1[j];
      size_t k = 0;
      while (k + 1 < c.wls[i].size() && c.wls[i][k + 1] < x)
        ++k;
      const double x0 = c.wls[i][k], x1 = c.wls[i][k + 1];
      const double y0 = c.vals[i][k], y1 = c.vals[i][k + 1];
      out[j] = (x1 == x0) ? y0 : y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }
    vals1[i] = std::move(out);
  }

  ResampledTables tables =
      prepare_resampled_tables(wl1, c.cmf_2deg, c.cmf_10deg, c.ces, c.basis);

  std::vector<Spd> owned;
  for (size_t i = 0; i < c.wls.size(); ++i)
    owned.emplace_back(wl1, vals1[i]);
  std::vector<SpdView> views19, views1;
  for (auto &s : owned)
    views19.push_back({s.wavelengths(), s.values()});
  views1.push_back(views19[0]);

  Tm30Request req{/*bins=*/true, /*samples=*/true};

  for (int i = 0; i < 5; ++i) {
    try_evaluate_cached(views19, tables, c.lut, req);
    try_evaluate_cached(views1, tables, c.lut, req);
  }

  {
    std::vector<double> t;
    for (int i = 0; i < 100; ++i) {
      auto t0 = Clock::now();
      auto r = try_evaluate_cached(views19, tables, c.lut, req);
      (void)r;
      auto t1 = Clock::now();
      t.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::printf("batch19  : median %.4f ms/call (%.4f ms/SPD)\n", median_ms(t),
                median_ms(t) / 19.0);
  }

  {
    std::vector<double> t;
    for (int i = 0; i < 300; ++i) {
      auto t0 = Clock::now();
      auto r = try_evaluate_cached(views1, tables, c.lut, req);
      (void)r;
      auto t1 = Clock::now();
      t.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::printf("batch1   : median %.4f ms/call\n", median_ms(t));
  }

  {
    std::vector<double> t;
    for (int i = 0; i < 1000; ++i) {
      auto t0 = Clock::now();
      std::thread th([] {});
      th.join();
      auto t1 = Clock::now();
      t.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::printf("spawn1   : median %.4f us\n", median_ms(t) * 1e3);
  }

  {
    std::vector<double> t;
    for (int i = 0; i < 200; ++i) {
      auto t0 = Clock::now();
      std::vector<std::thread> ths;
      for (int j = 0; j < 4; ++j)
        ths.emplace_back([] {});
      for (auto &th : ths)
        th.join();
      auto t1 = Clock::now();
      t.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::printf("spawn4   : median %.4f us\n", median_ms(t) * 1e3);
  }

  return 0;
}
