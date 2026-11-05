#include "triangle_pm/triangle.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_ws(const std::string& line) {
  std::istringstream in(line);
  std::vector<std::string> tok;
  std::string s;
  while (in >> s) tok.push_back(s);
  return tok;
}

int emit(const triangle_pm::Result& r, int& exit_code) {
  std::cout << triangle_pm::to_json(r) << '\n';
  if (r.status != 0) exit_code = r.status;
  return r.status;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  bool batch = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--batch") {
      batch = true;
    } else {
      args.push_back(a);
    }
  }

  int exit_code = 0;
  if (batch) {
    if (!args.empty()) {
      triangle_pm::Result r;
      r.status = triangle_pm::kInvalidGeometry;
      emit(r, exit_code);
      return exit_code ? exit_code : 1;
    }
    std::string line;
    bool any = false;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      any = true;
      emit(triangle_pm::evaluate_tokens(split_ws(line)), exit_code);
    }
    if (!any) {
      triangle_pm::Result r;
      r.status = triangle_pm::kInvalidGeometry;
      emit(r, exit_code);
    }
    return exit_code;
  }

  emit(triangle_pm::evaluate_tokens(args), exit_code);
  return exit_code;
}
