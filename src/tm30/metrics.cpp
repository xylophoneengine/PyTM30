// TM-30-20: Color difference (ΔE') and fidelity index (Rf).
//
// TM-30-20 §3.8: Color Difference Formula
// TM-30-20 §4.1: Fidelity Index (Rf)
#include "tm30/metrics.hpp"

#include <cmath>
#include <numeric>

namespace tm30 {

std::array<double, 99>
compute_delta_e(const std::array<Cam02Ucs, 99> &jab_test,
                const std::array<Cam02Ucs, 99> &jab_ref) {

  // TM-30-20 §3.8 Eq. (52)
  std::array<double, 99> delta_e{};

  for (std::size_t i = 0; i < 99; ++i) {
    // TM-30-20 §3.8 Eq. (52)
    const double dJ = jab_test[i].J_prime - jab_ref[i].J_prime;
    const double da = jab_test[i].a_prime - jab_ref[i].a_prime;
    const double db = jab_test[i].b_prime - jab_ref[i].b_prime;
    delta_e[i] = std::sqrt(dJ * dJ + da * da + db * db);
  }

  return delta_e;
}

RfResult compute_rf(const std::array<double, 99> &delta_e_array) {
  RfResult result{};

  // Step 1: arithmetic mean of 99 ΔE′ values
  // TM-30-20 §4.1
  const double sum =
      std::accumulate(delta_e_array.begin(), delta_e_array.end(), 0.0);
  result.delta_e_avg = sum / 99.0;

  // Step 2: scale by factor 6.73, subtract from 100
  // TM-30-20 §4.1 Eq. (53)
  result.Rf_prime = 100.0 - 6.73 * result.delta_e_avg;

  // Step 3: logarithmic rescaling to [0, 100]
  // TM-30-20 §4.1 Eq. (54)
  result.Rf = 10.0 * std::log(std::exp(result.Rf_prime / 10.0) + 1.0);

  return result;
}

std::array<double, 99>
compute_rf_cesi(const std::array<double, 99> &delta_e_array) {
  std::array<double, 99> result;
  for (std::size_t i = 0; i < 99; ++i) {
    // TM-30-20 §4.2 Eq. (55): Rf,CESi' = 100 - 6.73 · ΔE'_i
    const double rf_prime = 100.0 - 6.73 * delta_e_array[i];

    // TM-30-20 §4.2 Eq. (56): Rf,CESi = 10 · ln(exp(Rf,CESi'/10) + 1)
    result[i] = 10.0 * std::log(std::exp(rf_prime / 10.0) + 1.0);
  }
  return result;
}

double compute_rf_skin(const std::array<double, 99> &rf_cesi) {
  // TM-30-20 §4.2: Rf,skin = (Rf,CES15 + Rf,CES18) / 2
  // CES indices are 1-based: CES15 → index 14, CES18 → index 17
  return (rf_cesi[14] + rf_cesi[17]) / 2.0;
}

} // namespace tm30
