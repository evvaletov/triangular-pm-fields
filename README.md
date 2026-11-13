# Triangular permanent-magnet fields

Author: Eremey Valetov.

Triangular permanent-magnet field calculations and Taylor maps. Citation
metadata are in [CITATION.cff](CITATION.cff).

The code evaluates the magnetic field of prescribed charged triangular
surfaces and normalized spatial Taylor coefficients through degree six.
It also combines coplanar triangles into a cached boundary representation.
This avoids repeating internal mesh-edge contributions during field queries.
The repository includes source, independent references, examples, and scripts
for reproducing the validation and timings.
See [LICENSE](LICENSE) for the terms that apply to this source release.

The [recorded qualification](validation/README.md) includes the executed
five-language checks and complete comparison arrays.

## Implementations

| Language | Included implementation | Additional requirements |
|---|---|---|
| Fortran | Cached assembly engine, scalar modes and complete degree-six Taylor maps | GNU Fortran with real128; GMP |
| COSY INFINITY | TRLOAD/TRFLD native bridge and project analytic COSYScript routines | Compatible licensed runtime and macro library, supplied separately |
| C++ | Standalone high-precision triangle evaluator and native C-ABI client | Boost headers; native library only for the native client |
| Python | Geometry preparation, persistent geometry cache, native client, analytic/jet baseline, correlated linear fitting | NumPy, mpmath; native library for the native client |
| Julia | Standalone high-precision triangle evaluator and native C-ABI client | Julia; native library only for the native client |

The native clients share one physical engine. Agreement between those clients
checks their APIs, data layout, and DA composition. Independent field evidence
comes from the Green-integral reference corpus. Standalone C++ and Julia are
separate implementations of the closed form, not separate physical formulas.
The optional [JAX adapter](adapters/jax_repair_20261103/README.md) has its own
name, dependency pins, supported domain, and checks.

## Physical and coefficient convention

For charge `sigma` in tesla and consistent length units, the returned quantity is

`mu0 H(p) = sigma/(4*pi) * integral_T (p-s)/|p-s|^3 dS`.

Inside a uniformly polarized closed magnet, recover `B` by adding the
polarization `J`. Reversing vertex order while holding charge fixed does not
change this single-sheet field. A closed-body surface charge is `J dot n_out`.
Geometry and charge stay fixed during a spatial Taylor expansion:

`mu0 H(p + radius*u) = sum_alpha c_alpha * u^alpha`, for `|alpha| <= order`.

Coefficients already include division by `alpha!`. Monomials are ordered by
total degree, then increasing x exponent, then increasing y exponent. There
are 84 monomials per component at degree six. Radius normalizes variables;
it does not certify a region of convergence.

Native and standalone C++/Julia boundary rules use exact predicates on the
binary64 input geometry. Scalar face-interior values use the average trace.
A charged physical edge or vertex is singular. An ordinary Taylor expansion
on a charged sheet is rejected. Exterior coplanar centers are valid. A zero
charge returns zero after validating geometry and the other inputs. The
Python analytic baseline and optional JAX adapter have narrower documented
domains. Their regular-center checks do not establish the native boundary policies.

## Build and run on Linux

Install a C/C++ compiler, GNU Fortran, CMake, GMP development headers, Boost
development headers, Python 3.12, and Julia. The recorded qualification
uses GNU compilers 14.3.1, Python 3.12.15, and Julia 1.12.6. Other platforms and
compiler combinations have not been qualified by this release.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install .
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
python examples/triangle.py --library build/libtan.so
./build/example_fortran
julia --startup-file=no --project=julia julia/test/runtests.jl
```

Installing the Python package does not build or bundle libtan. Select that
library explicitly when creating `NativeLibrary`. The native library has a
process-global active cache; direct callers must serialize load/evaluate
sequences. The Python client does so across its instances and reloads the
requested geometry for each batch. Use separate processes for independent
native contexts. See [docs/native-api.md](docs/native-api.md).

Standalone C++ and Julia accept the same command-line arguments:

```sh
./build/cpp/triangle_cli 6 .125 1.25 0 0 0 1 0 0 0 1 0 .25 .25 .75
julia --startup-file=no --project=julia julia/cli.jl 6 .125 1.25 0 0 0 1 0 0 0 1 0 .25 .25 .75
```

These emit JSON containing status, exponent triples, and a `3 x nmono`
coefficient array. See the [C++](cpp/README.txt) and [Julia](julia/README.md)
documentation for library APIs and native client usage.

## Reproduce validation

```sh
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/conformance.py --build build --output local-results/conformance.json
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/benchmark.py --build build --output local-results/benchmark.json
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 \
  python tools/benchmark_ports.py --build build --output local-results/ports.json
```

The [frozen corpus](corpus/README.md) contains 105 cases, including 48 complete
Taylor maps at orders 1, 2, 4, and 6. The independent surface-integral references
were generated before inspecting candidate port output. All components and all
coefficients are checked. Hard scalar references distinguish independent
integrals from shared closed-form high-precision checks. Boundary errors and
invalid inputs remain in the report. Native cache preparation cannot encode
invalid triangles; those cases are explicitly distinguished from executed
native queries.

The coefficient budget is `max|c-c_ref| / max(1,max|c_ref|) <= 1e-10`.
Scalar values also satisfy an L2 budget of `1e-12 T + 1e-10*|reference|`.
These are finite-corpus results, not universal fixed-precision error bounds.
Benchmark reports retain warmups, sample arrays, versions, workload sizes,
and preparation/query boundaries. Shared-host timings are not universal speed
guarantees. The precision choices differ between implementations.
`benchmark_ports.py` measures full C++/Julia CLI batches, including process
startup, Julia compilation, input parsing, and JSON output. Its results do not
represent warm library-kernel costs.

COSY execution requires the external licensed prerequisites. See
[cosy/README.md](cosy/README.md) for build and validation commands. CI checks
the open-tool consumers; licensed COSY is qualified separately and is never
reported as executed merely because saved output is present.

The standalone Python baseline has a separate
[API and domain description](docs/python-reference.md).

## Fitting with correlated errors

The Python package also provides a linear fit with a known joint covariance
for training and held-out observations. It carries shared errors through
parameter uncertainty and held-out residuals, including the cross covariance
between scans. See [the fitting contract](docs/correlated-fitting.md) and run
`python examples/python_correlated_fit.py` for a synthetic example.

The example and tests use synthetic observations. Physical magnet validation requires
measured scans, matching geometry, and independent calibration records.
