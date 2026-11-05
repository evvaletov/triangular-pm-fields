#pragma once

#include "triangle_pm/triangle.hpp"

#include <array>
#include <string>
#include <vector>

namespace triangle_pm {

// Native client of the Fortran tan_mod C ABI (tan_load_c / tan_eval_c).
// Shares the Fortran physical kernel; this is not a second field formulation.
// The loaded cache is process-global in that kernel and is not thread-safe.
Result native_eval(const std::string& library, const std::string& cache,
                   const std::array<double, 3>& point, int mode, int order,
                   double radius);

Result native_eval_tokens(const std::vector<std::string>& tokens);

}  // namespace triangle_pm
