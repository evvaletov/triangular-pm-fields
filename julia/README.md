# Julia triangle μ0 H

Standalone local Taylor map of the charged-triangle kernel, plus a separately
named native client that shares the Fortran `tan_mod` physical kernel.

## Layout

- `src/TriangularPMFields.jl` — library
- `cli.jl` — standalone CLI
- `native_cli.jl` — Fortran C ABI client
- `examples/example.jl` — regular-point example
- `test/runtests.jl` — independent tests
- `JULIA_HANDOFF.txt` — lane commands, counts, and limits

Dependencies: Julia 1.10+ stdlib (`BigFloat`, `Rational`, `Libdl`, `Test`).
No package download. Native queries need a built `libtan.so` and GMP.

## Standalone

The standalone path implements the field and the degree-0..6 jet itself.
It does not call Python, Fortran, or another evaluator.

Inputs are parsed as binary64. Face/edge/vertex membership uses exact
`Rational{BigInt}` values of those binary64 numbers. Query and Taylor
arithmetic uses scoped 256-bit `BigFloat`; outputs are binary64.
Scoped precision requires serial calls. Use separate processes for concurrent
standalone evaluations, and avoid concurrent changes to BigFloat precision.

Coefficients mean

```
μ0 H(p + radius * u) = sum_{|α|≤q} c_α u^α
```

with `α!` already in `c_α`. Graded order, then x ascending, y ascending,
`z = degree-x-y`. `q=6` has 84 terms per component. Radius scales the
monomials; it is not a convergence-ball certificate.

Scalar exact face interiors use the average trace. A nonzero charged edge
or vertex is rejected. DA on the charged sheet is rejected. Exterior-plane
DA continues after correcting a known-zero plane numerator constant without
dropping higher coefficients. Zero charge, after geometry validation,
returns zero even on the support.

Limits: truncated local series through degree 6, corpus-bounded accuracy.
This is not a universal precision or speed claim.

```
julia --startup-file=no cli.jl ORDER RADIUS SIGMA AX AY AZ BX BY BZ CX CY CZ PX PY PZ
julia --startup-file=no cli.jl --batch   # one JSON object per stdin line
```

Success writes one finite JSON object `{status, exponents, coefficients}`
with `coefficients` shape `(3, nmono)`. Nonzero status: empty arrays, no
NaN/Inf, nonzero exit. `--batch` continues after a bad record and exits
nonzero if any record failed.

## Native client

`native_cli.jl` loads `libtan.so` and a `TAN_CACHE_V1` cache and calls
`tan_load_c` / `tan_eval_c`. It shares the Fortran physical kernel. Status
codes and modes follow `../docs/native-api.md`.
Julia stores the native 3×84 buffer component-first and returns 3 rows by
`nmono` columns. The Fortran cache is process-global; do not use it from
multiple threads.

```
julia --startup-file=no native_cli.jl LIBRARY CACHE X Y Z MODE ORDER RADIUS
```

## Tests

```
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
julia --startup-file=no test/runtests.jl
```

Tests use analytic identities, Gauss quadrature of the Green kernel, and
spherical excess. They do not take another language port as oracle.
