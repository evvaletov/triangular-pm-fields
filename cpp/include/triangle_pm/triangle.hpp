#pragma once

#include <array>
#include <string>
#include <vector>

namespace triangle_pm {

constexpr int kMaxOrder = 6;
constexpr int kMaxMono = 84;

enum Status : int {
  kOk = 0,
  kInvalidOrder = 5,
  kNonfiniteObserver = 6,
  kBadRadius = 7,
  kPhysicalEdgeVertex = 8,
  kDaOnChargedSheet = 9,
  kNonfiniteResult = 10,
  kInvalidGeometry = 11
};

inline int nmono(int order) {
  if (order < 0 || order > kMaxOrder) return 0;
  return (order + 1) * (order + 2) * (order + 3) / 6;
}

struct Result {
  int status = kInvalidGeometry;
  std::vector<std::array<int, 3>> exponents;
  std::array<std::vector<double>, 3> coefficients;
};

std::vector<std::array<int, 3>> monomials(int order);

// mu0 H = sigma/(4 pi) * integral_T (p-s)/|p-s|^3 dS.
// Inputs are binary64. Membership uses exact rationals of those values.
// Query and Taylor arithmetic use >=80 decimal digits, then binary64 output.
// Coefficients are Taylor (1/alpha!) terms of mu0 H(p + radius u).
Result evaluate(int order, double radius, double sigma,
                const std::array<double, 3>& a,
                const std::array<double, 3>& b,
                const std::array<double, 3>& c,
                const std::array<double, 3>& p);

Result evaluate_tokens(const std::vector<std::string>& tokens);

std::string to_json(const Result& result);

bool parse_binary64(const std::string& token, double& value);

}  // namespace triangle_pm
