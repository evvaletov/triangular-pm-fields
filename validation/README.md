# Validation record — 4 November 2026

A clean source export built with GNU 14.3.1, Python 3.12.15, and Julia 1.12.6
passed the checks below. The Python wheel also installed in a separate virtual
environment and imported from site-packages. Dependency versions and preserved
native source hashes are in [PROVENANCE.json](../PROVENANCE.json).

| Consumer | Executed cases | Largest scaled coefficient error |
|---|---:|---:|
| python_analytic | 48 | 8.33e-17 |
| cpp_standalone | 105 | 9.71e-17 |
| julia_standalone | 105 | 9.71e-17 |
| fortran_native | 98 | 9.71e-17 |
| python_native | 87 | 9.71e-17 |
| cpp_native | 98 | 9.71e-17 |
| julia_native | 98 | 9.71e-17 |
| COSY native bridge | 87 | 9.72e-17 |

All listed cases passed. The standalone C++ and Julia counts include the full
105-case status contract. Invalid geometry cannot form a native cache. The
native clients therefore execute 98 applicable cases; the Python wrapper
rejects 11 of those in its own input validation, leaving 87 native executions.
The standalone Python baseline is checked on the 48 regular Taylor cases.
COSY executes the 80 successful cases and seven physical-boundary error cases.
The full [conformance report](conformance.json) records each output, comparison,
and excluded native call. [COSY results](cosy.json) include runtime and library
hashes; the accompanying `cosy-q*.txt` files are fresh interpreter output.

The coefficient limit is `max|c-c_ref|/max(1,max|c_ref|) <= 1e-10`.
Scalar outputs also pass `1e-12 T + 1e-10*|reference|` in L2 norm.
The independent oracle's maximum refinement difference is `4.03e-16`.
See [corpus/README.md](../corpus/README.md) for the reference methods and their
limits. All components and all requested coefficients are compared.

Additional checks passed:

- 119 native Fortran checks and the C-ABI regression program.
- 84 C++ checks, including concurrent first calls, the native client, and oversized integer inputs.
- 46 Python tests covering preparation, persistent-cache integrity, the
  independent corpus, ABI layout, concurrent wrapper calls, and failed loads.
- 218 Julia assertions, including oversized orders, map dimensions, native-client range, and failed reloads.
- 48 regular-center JAX maps and four manufactured jet controls; largest
  scaled error `1.25e-15`, with x64 CPU JAX/jaxlib 0.10.1.
- Actual COSY nonlinear fourth-variable composition, mixed scalar/DA inputs,
  aliases, output capacity, mode checks, and cache reloads. These composition
  controls share the native field engine; they do not establish independent
  physical accuracy. See [cosy-composition.json](cosy-composition.json).
- Python installation, all five language examples, the official
  CITATION.cff 1.2.0 schema, Ruff's F checks, and ShellCheck for `tools/ci.sh`.

[benchmark.json](benchmark.json) retains all CPU and wall-time samples.
For 8, 32, and 128 coplanar triangles, the baseline/optimized median CPU-time
ratios were 2.51, 2.68, and 4.25. Both constructors produced identical canonical
cache text. Each comparison used seven
samples after warmup. The separate native-query timings exclude cache loading
and use a regular 512-observer workload. These shared-host measurements are
limited to the recorded workloads.

[ports.json](ports.json) measures full C++/Julia CLI batches, including startup,
Julia compilation, parsing, and JSON. Those timings have different precision
and overhead from the native kernel measurements. They are not warm-query
speed comparisons. [execution.json](execution.json) records the clean-export
command groups and exit statuses. Reproduction commands are in the root README.

The open-tool GitHub workflow is included. This local qualification does not
claim a hosted Actions result. The licensed COSY runtime and macro library
are external prerequisites and are absent from the repository.

ThreadSanitizer could not link on the qualification host because its system
`libtsan` target was missing. No sanitizer result is claimed. The C++ first-use
regression ran eight concurrent callers, and shared table initialization uses
`std::call_once`.
