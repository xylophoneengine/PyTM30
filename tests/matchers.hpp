#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace tm30::test {

// ══════════════════════════════════════════════════════════════════════════
//  WithinTolerance - Catch2 matcher for absolute tolerance.
//
//  Usage:
//    REQUIRE_THAT(result, WithinTolerance(Tol_Xyz, 95.047));
//
//  On failure, prints expected, actual, delta, and tolerance.
// ══════════════════════════════════════════════════════════════════════════
template <typename T>
class WithinTolerance final : public Catch::Matchers::MatcherGenericBase {

public:
  explicit WithinTolerance(T tolerance, T expected)
      : tolerance_{tolerance}, expected_{expected} {}

  bool match(T const &actual) const {
    actual_ = actual;
    return std::abs(actual - expected_) <= tolerance_;
  }

  std::string describe() const override {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(10);

    oss << "\n  expected: " << expected_ << "\n  actual:   " << actual_
        << "\n  delta:    " << std::abs(actual_ - expected_)
        << "\n  tolerance: " << tolerance_;
    return oss.str();
  }

private:
  T tolerance_;
  T expected_;
  mutable T actual_{};
};

// ══════════════════════════════════════════════════════════════════════════
//  WithinRelTolerance - Catch2 matcher for relative tolerance.
//
//  Usage:
//    REQUIRE_THAT(result, WithinRelTolerance(1e-3, 100.0));
//
//  Passes when |actual - expected| / expected <= relative_tolerance.
//  On failure, prints expected, actual, relative delta, and tolerance.
// ══════════════════════════════════════════════════════════════════════════
template <typename T>
class WithinRelTolerance final : public Catch::Matchers::MatcherGenericBase {

public:
  explicit WithinRelTolerance(T tolerance, T expected)
      : tolerance_{tolerance}, expected_{expected} {}

  bool match(T const &actual) const {
    actual_ = actual;
    if (expected_ == T{}) {
      // Avoid division by zero - fall back to absolute comparison.
      return std::abs(actual_) <= tolerance_;
    }
    return std::abs(actual - expected_) / std::abs(expected_) <= tolerance_;
  }

  std::string describe() const override {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(10);

    T delta = std::abs(actual_ - expected_);
    T rel_delta = (expected_ != T{}) ? delta / std::abs(expected_) : delta;

    oss << "\n  expected:   " << expected_ << "\n  actual:     " << actual_
        << "\n  delta:      " << delta << "\n  rel delta:  " << rel_delta
        << "\n  rel tolerance: " << tolerance_;
    return oss.str();
  }

private:
  T tolerance_;
  T expected_;
  mutable T actual_{};
};

} // namespace tm30::test
