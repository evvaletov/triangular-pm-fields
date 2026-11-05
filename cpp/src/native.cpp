#include "triangle_pm/native.hpp"

#include <cmath>
#include <dlfcn.h>

#include <vector>

namespace triangle_pm {
namespace {

using tan_load_fn = void (*)(const char*, int*);
using tan_eval_fn = void (*)(const double*, int, int, double, double*, int*);

Result fail_native(int status) {
  Result r;
  r.status = status;
  return r;
}

bool finite_d(double x) { return std::isfinite(x); }

}  // namespace

Result native_eval(const std::string& library, const std::string& cache,
                   const std::array<double, 3>& point, int mode, int order,
                   double radius) {
  if (library.empty() || cache.empty() || library.find('\0') != std::string::npos ||
      cache.find('\0') != std::string::npos) return fail_native(kInvalidGeometry);
  if (!finite_d(point[0]) || !finite_d(point[1]) || !finite_d(point[2])) {
    return fail_native(kNonfiniteObserver);
  }
  if (!finite_d(radius) || radius <= 0.0) return fail_native(kBadRadius);

  void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) return fail_native(kInvalidGeometry);

  auto load = reinterpret_cast<tan_load_fn>(dlsym(handle, "tan_load_c"));
  auto eval = reinterpret_cast<tan_eval_fn>(dlsym(handle, "tan_eval_c"));
  if (!load || !eval) {
    dlclose(handle);
    return fail_native(kInvalidGeometry);
  }

  // Clear native allocations before unloading the library.
  auto close = [&]() {
    int ignored = 0;
    load("", &ignored);
    dlclose(handle);
  };

  int status = kInvalidGeometry;
  load(cache.c_str(), &status);
  if (status != kOk) {
    close();
    return fail_native(status);
  }

  // Fortran coeff(3,84) is component-fastest: Bx,By,Bz for each monomial.
  std::vector<double> coeff(3 * kMaxMono, 0.0);
  const double p[3] = {point[0], point[1], point[2]};
  eval(p, mode, order, radius, coeff.data(), &status);
  if (status != kOk) {
    close();
    return fail_native(status);
  }

  Result r;
  r.status = kOk;
  r.exponents = monomials(order);
  const int n = nmono(order);
  r.coefficients[0].resize(static_cast<std::size_t>(n));
  r.coefficients[1].resize(static_cast<std::size_t>(n));
  r.coefficients[2].resize(static_cast<std::size_t>(n));
  for (int k = 0; k < n; ++k) {
    r.coefficients[0][static_cast<std::size_t>(k)] = coeff[static_cast<std::size_t>(3 * k + 0)];
    r.coefficients[1][static_cast<std::size_t>(k)] = coeff[static_cast<std::size_t>(3 * k + 1)];
    r.coefficients[2][static_cast<std::size_t>(k)] = coeff[static_cast<std::size_t>(3 * k + 2)];
    if (!finite_d(r.coefficients[0][static_cast<std::size_t>(k)]) ||
        !finite_d(r.coefficients[1][static_cast<std::size_t>(k)]) ||
        !finite_d(r.coefficients[2][static_cast<std::size_t>(k)])) {
      close();
      return fail_native(kNonfiniteResult);
    }
  }
  close();
  return r;
}

Result native_eval_tokens(const std::vector<std::string>& tokens) {
  if (tokens.size() != 8) return fail_native(kInvalidGeometry);
  const std::string& library = tokens[0];
  const std::string& cache = tokens[1];
  double x = 0, y = 0, z = 0, mode_d = 0, order_d = 0, radius = 0;
  if (!parse_binary64(tokens[2], x) || !parse_binary64(tokens[3], y) ||
      !parse_binary64(tokens[4], z) || !parse_binary64(tokens[5], mode_d) ||
      !parse_binary64(tokens[6], order_d) || !parse_binary64(tokens[7], radius)) {
    return fail_native(kInvalidGeometry);
  }
  if (!finite_d(mode_d) || mode_d != std::floor(mode_d) || mode_d < 0 || mode_d > 2)
    return fail_native(4);
  if (!finite_d(order_d) || order_d != std::floor(order_d) || order_d < 0 || order_d > 6)
    return fail_native(kInvalidOrder);
  return native_eval(library, cache, {x, y, z}, static_cast<int>(mode_d),
                     static_cast<int>(order_d), radius);
}

}  // namespace triangle_pm
