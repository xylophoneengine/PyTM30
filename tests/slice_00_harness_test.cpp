// Slice 0 - Harness validation.
//
// Purpose: prove that the CMake + Catch2 + tolerance-matcher stack
// correctly detects failures AND successes.  No colour science whatsoever.
//
// The deliberately-failing test below is commented out once the harness is
// verified; it stays in the file as documentation that the harness works.

#include "matchers.hpp"
#include <catch2/catch_test_macros.hpp>

namespace tm30::test {
namespace {

// ══════════════════════════════════════════════════════════════════════════
// Deliberately failing test - proves the harness can detect failures.
//
// After harness verification this is COMMENTED OUT, not deleted.
// It remains here so future agents can uncomment and re-verify the
// harness if the toolchain changes.
//
// ══════════════════════════════════════════════════════════════════════════
// TEST_CASE("Slice 0 - deliberately failing tolerance test", "[harness]") {
//     // 1.0 vs 2.0, tolerance 0.5 → should FAIL (delta = 1.0 > 0.5).
//     REQUIRE_THAT(1.0, WithinTolerance(0.5, 2.0));
// }

// ══════════════════════════════════════════════════════════════════════════
// Passing test - proves the matcher works for values within tolerance.
// ══════════════════════════════════════════════════════════════════════════
TEST_CASE("Slice 0 - passing tolerance test", "[harness]") {
  // 1.0 vs 1.0, tolerance 0.5 → should PASS (delta = 0.0 ≤ 0.5).
  REQUIRE_THAT(1.0, WithinTolerance(0.5, 1.0));
}

TEST_CASE("Slice 0 - near-boundary pass", "[harness]") {
  // 1.5 vs 1.0, tolerance 0.5 → should PASS (delta = 0.5 ≤ 0.5).
  REQUIRE_THAT(1.5, WithinTolerance(0.5, 1.0));
}

TEST_CASE("Slice 0 - relative tolerance pass", "[harness]") {
  // 101.0 vs 100.0, rel tolerance 0.02 → should PASS (rel delta = 0.01 ≤ 0.02).
  REQUIRE_THAT(101.0, WithinRelTolerance(0.02, 100.0));
}

// TEST_CASE("Slice 0 - relative tolerance fail", "[harness]") {
//     // 103.0 vs 100.0, rel tolerance 0.02 → should FAIL (rel delta = 0.03 >
//     0.02).
//     // REQUIRE_THAT(103.0, WithinRelTolerance(0.02, 100.0));
// }

} // namespace
} // namespace tm30::test
