#pragma once

#include <stdexcept>
#include <string>

namespace tm30 {

/// Base class for all TM-30 construction/validation errors.
/// Thrown at construction time; the hot path never throws.
class InvalidSpd : public std::runtime_error {
public:
  explicit InvalidSpd(const std::string &what) : std::runtime_error(what) {}
};

} // namespace tm30
