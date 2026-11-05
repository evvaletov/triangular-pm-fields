#include "triangle_pm/native.hpp"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
  const triangle_pm::Result r = triangle_pm::native_eval_tokens(args);
  std::cout << triangle_pm::to_json(r) << '\n';
  return r.status;
}
