// Simple CSV loader for spectral data tables.
#include "tm30/csv_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>

namespace tm30 {

CsvTable load_csv(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open CSV file: " + filepath);
  }

  CsvTable table;
  std::string line;

  // Read header line
  if (!std::getline(file, line)) {
    throw std::runtime_error("CSV file is empty: " + filepath);
  }

  // Strip trailing \r if present (Windows line endings)
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }

  {
    std::istringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      table.headers.push_back(cell);
    }
  }

  if (table.headers.empty()) {
    throw std::runtime_error("CSV file has no header columns: " + filepath);
  }

  // Read data rows
  while (std::getline(file, line)) {
    if (line.empty()) continue;

    if (line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) continue;

    std::istringstream ss(line);
    std::string cell;
    std::vector<double> row;
    row.reserve(table.headers.size());

    while (std::getline(ss, cell, ',')) {
      // Trim whitespace
      size_t start = 0;
      size_t end = cell.size();
      while (start < end && std::isspace(static_cast<unsigned char>(cell[start]))) ++start;
      while (end > start && std::isspace(static_cast<unsigned char>(cell[end - 1]))) --end;

      if (start == end) {
        throw std::runtime_error("Empty cell in CSV data row: " + filepath);
      }

      double val = std::stod(cell.substr(start, end - start));
      row.push_back(val);
    }

    if (row.size() != table.headers.size()) {
      throw std::runtime_error(
          "Column count mismatch in " + filepath
          + ": expected " + std::to_string(table.headers.size())
          + ", got " + std::to_string(row.size()));
    }

    table.rows.push_back(std::move(row));
  }

  return table;
}

}  // namespace tm30
