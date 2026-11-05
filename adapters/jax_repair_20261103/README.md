# jax_repair_20261103

This optional adapter evaluates the unscaled sheet integral `K`. Multiply its
output by `sigma/(4*pi)` to obtain `mu0 H`. For Taylor coefficients, also
multiply each column by `radius**sum(alpha)`. `compiled_scalar()` returns a
batched scalar function; `compiled_directional_coefficients(q)` returns the
complete spatial coefficients at orders 1 through 6 using directional jets.

The implementation uses local solid-angle and edge-logarithm charts compatible
with `jax.experimental.jet` and an outer `jax.jit`. Geometry and observer
coordinates remain runtime inputs. The named adapter is separate from the
stock magpylib_jax comparison baseline, pinned at
`6290916635693fb04a099205b48f845a396dffc0`. That baseline is not vendored or
modified by this package.

The supported domain is regular observation centers, including exterior
coplanar points and edge-line extensions outside finite source segments.
It supplies no scalar face-average policy or charged-sheet Taylor map.
Invalid geometry and physical edge/vertex centers are outside this adapter's
contract. Use the native interfaces for status-based boundary handling.

The recorded environment is CPU JAX/jaxlib 0.10.1 with binary64 enabled.
Install `requirements-tested.txt` from this directory in a separate environment
alongside the root Python package, then run from the repository root:

```sh
PYTHONPATH=python JAX_ENABLE_X64=true JAX_PLATFORMS=cpu \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/check_jax.py --output local-results/jax.json
```

The check covers all 48 regular Taylor maps in the frozen corpus and
manufactured exponential/polynomial jet controls. It retains complete
coefficient arrays. The first call for a new order can compile; the check's
elapsed times include compilation and are not warm-query benchmarks.
See the [JAX installation documentation](https://docs.jax.dev/en/latest/installation.html)
for platform-specific installation. GPU execution is unqualified here.
