#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
export PYTHONPATH="$root/python${PYTHONPATH:+:$PYTHONPATH}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
julia --startup-file=no --project=julia julia/test/runtests.jl
python3 tools/conformance.py --build build --output local-results/conformance.json
python3 examples/triangle.py --library build/libtan.so
python3 examples/python_correlated_fit.py
python3 validation/correlated/check_oracle.py \
  --candidate python/triangular_pm_fields/correlated_fit.py \
  --output local-results/correlated-oracle.json
python3 validation/correlated/pm_gaussian_check.py \
  --arrays validation/correlated/pm_fixture.npz \
  --candidate python/triangular_pm_fields/correlated_fit.py \
  --output local-results/correlated-gaussian.json
./build/example_fortran
./build/cpp/triangle_example
