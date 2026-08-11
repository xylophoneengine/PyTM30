#pragma once

/// @file csv_loader.hpp
/// Simple runtime CSV loader for spectral data tables.
/// Reads data/ces.csv and data/cmf_1964_10.csv at 1 nm or 5 nm resolution.
///
/// TM-30-20 §3.5: CES reflectance functions provided in 1-nm increments.

#include <string>
#include <vector>

namespace tm30 {

/// Holds a single spectral data table loaded from CSV.
struct CsvTable {
  /// Column headers (first row of CSV).
  std::vector<std::string> headers;

  /// Data rows: each row is a vector of doubles matching the column count.
  std::vector<std::vector<double>> rows;
};

/// Load a CSV file with a header row.
/// Expects comma-separated values; handles \\r\\n and \\n line endings.
/// @throws std::runtime_error on file-not-found or parse failure.
CsvTable load_csv(const std::string &filepath);

} // namespace tm30
