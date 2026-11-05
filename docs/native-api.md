# Native assembly API

Prepare a `TAN_CACHE_V1` geometry file in Python:

```python
from triangular_pm_fields import PreparedCache
cache = PreparedCache(triangles, charges)  # (N,3,3) vertices; (N,) charges
cache.write("geometry.tan")
```

Preparation uses exact rational binary64 geometry. It groups planes, combines
signed charge differences, splits collinear overlapping segments and
T-junctions, coalesces equal-weight segments, and checks closed weighted chains.
Inputs are prescribed charges. This calculation does not solve a material
response or determine magnetization from permeability.

For repeated preparation across processes, use
`triangular_pm_fields.cache_store.load_or_prepare(triangles, charges, directory)`.
This saves text geometry and a JSON manifest under a content-derived key.
The key binds binary64 input, options, format, and constructor source hash.
Hits verify the content and manifest; stale or corrupt entries rebuild.
The store uses POSIX file locks and atomic file replacement. It addresses
accidental corruption and staleness, without authenticating hostile files.

## Fortran

```fortran
call tan_load(path, status)
call tan_eval(point, mode, order, radius, coefficients, status)
call tan_exponents(order, ex, ey, ez, n)
used_mode = tan_last_mode()
```

`point(3)`, `radius`, and `coefficients(3,84)` are binary64. Mode 0 is a
binary64 scalar calculation; mode 1 uses real128 and accepts orders 0–6;
mode 2 selects a scalar precision path using geometry and conditioning guards.
Modes 0 and 2 require order 0. `tan_last_mode()` reports the path actually used:
0, 1, or -1 when no kernel ran. Mode 2's guards have finite-corpus validation;
they do not provide an error bound for every possible input.

## C ABI

```c
void tan_load_c(const char *path, int *status);
void tan_eval_c(const double point[3], int mode, int order, double radius,
                double coefficients[252], int *status);
void tan_batch_c(int count, const double *points, int mode,
                 double *fields, int *statuses);
int tan_last_mode_c(void);
```

Coefficients have Fortran layout: component varies fastest, followed by
monomial. Batch points and fields have `3*count` entries with component fastest.
Batch queries are scalar at radius 1. All native calls use one process-global
cache and require external serialization. Every load clears the active context
before parsing; a failed load cannot leave a previously loaded cache active.
Check status before consuming output. Before allocating geometry, the Fortran
loader checks nonnegative counts against a conservative text-file byte bound
using int64 arithmetic and rejects counts that leave no room for one-past-end
indices. Impossible counts and truncated records return status 3. Explicit
Fortran loader allocations use `STAT=`; allocation failure returns status 2
and cleans up any partly allocated cache. These checks do not authenticate the
file or impose a fixed maximum memory size on plausible large geometry files.

| Status | Meaning |
|---|---|
| 0 | Success |
| 1 | No cache loaded |
| 2 | Cache file I/O or Fortran loader allocation failure |
| 3 | Cache format failure |
| 4 | Invalid mode or mode/order combination |
| 5 | Invalid order |
| 6 | Nonfinite observer |
| 7 | Nonfinite or nonpositive radius |
| 8 | Surviving physical edge or vertex |
| 9 | Taylor center on nonzero charged support |
| 10 | Nonfinite result |
| 11 | Domain failure |

Native support classification concerns the assembled signed density. Canceled
edges, holes, and T-junctions are covered by the inherited native regressions.
The standalone single-triangle interfaces use status 11 also for invalid
input geometry or charge, which never reaches the native cache loader.

The Python client accepts arrays of shape `(n,3)` and returns `(n,3,nmono)`.
It raises `ValueError` for wrapper input errors and `NativeError` for native
statuses, with status and observer index retained. A module-wide lock covers
each load and batch across all Python client instances. Calling the C ABI
concurrently through a different binding would bypass that lock.
