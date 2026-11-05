#include "triangle_pm/triangle.hpp"

#include <iostream>

int main() {
  using triangle_pm::evaluate;
  const auto r = evaluate(0, 1.0, 0.8, {0.0, 0.0, 0.0}, {1.2, 0.0, 0.0},
                          {0.2, 0.9, 0.0}, {0.4, 0.3, 0.5});
  std::cout << "standalone status " << r.status << " Bx By Bz "
            << r.coefficients[0][0] << ' ' << r.coefficients[1][0] << ' '
            << r.coefficients[2][0] << '\n';
  return r.status;
}
