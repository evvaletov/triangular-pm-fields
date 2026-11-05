# Frozen independent triangle corpus

This corpus qualifies the finite triangle field in the release contract. It was
frozen before candidate implementation outputs were inspected. The case manifest
hash in `freeze.json` is authoritative. No C++, Julia, Fortran, C ABI, or native
kernel is used to generate these values.

There are 105 cases: 48 nonzero Taylor maps (12 centers at each of orders 1, 2,
4, and 6), 32 scalar/zero-charge cases, 7 exact charged boundary errors, and 18
invalid-input cases. Six of the 32 scalar/zero-charge cases return exact zero;
three of those request order-six maps on the source support. There are 80 success
cases and 25 expected errors. The scalar controls include reversal, length
scaling, translation, an oblique triangle, far field, heights down to 1e-12,
edge/vertex distances of 1e-8, and average traces. Invalid nonfinite values are
JSON strings `nan`/`inf` so the manifest remains standard JSON. Parse numeric
inputs to binary64 before constructing exact boundary predicates.

## Meaning and array schema

Every case specifies `id`, `vertices` (3 by 3, vertices in rows), `point`,
`sigma`, `order`, `radius`, `expected_status`, `suite`, and `reference_method`.
Charge is the actual surface charge in tesla; geometry and observation use the
same length unit. The output is mu0 H, not interior B:

    mu0 H(p + radius*u) = sigma/(4*pi) integral_T (p+radius*u-s)/|p+radius*u-s|^3 dS
                       = sum_alpha coefficients[:,alpha] u^alpha.

Coefficients include division by the multi-factorial. Exponents are graded by
total degree, then x ascending, then y ascending, with z=degree-x-y. Order six
has 84 monomials and 252 retained values. Positive radius normalizes variables;
it does not certify a convergence ball. Vertex winding leaves the field
unchanged for fixed sigma.

`reference.json` has a `results` list joined to cases by `id`. Successful entries
contain status 0, `exponents` of shape (nmono,3), and `coefficients` of shape
(3,nmono). Error entries have their expected status and empty arrays. Each entry
contains the evidence for its method and a `qualified` boolean. The top level
records source/case/generator/NPZ hashes, environment versions, and qualification
counts. High-precision values are retained as decimal strings.

`reference.npz` stores every binary64 array. Keys are `<id>__reference`,
`<id>__gauss_32`, `<id>__gauss_48`, `<id>__gauss_64`,
`<id>__closed_form_100`, `<id>__closed_form_140`, `<id>__integral_50`,
`<id>__integral_70`, or `<id>__trace_1e-14` / `<id>__trace_1e-18` as applicable.
All arrays are component-major (3,nmono), including scalar arrays (3,1).
There are no arrays for failed-status cases. All success cases have reference
arrays; none are silently skipped. `generation.jsonl` records the full run.

## Reference methods and independence

**Taylor maps:** the square-to-triangle Gauss-Legendre quadrature and binomial
Green-kernel coefficient expansion adapt `quadrature_nodes` and `quadrature_jet`
from the reference source identified in `freeze.json`. The generator contains its
own copy; no controller-relative module is imported. For d=p-s, it expands

    (d+delta) (d.d)^(-3/2)
      * sum_k binomial(-3/2,k) [(2*d.delta + delta.delta)/(d.d)]^k.

The fixed geometry is integrated at tensor orders 32, 48, and 64. Each result is
multiplied by sigma/(4*pi) and radius^degree. The entire 64-order array is the
reference, with both 32-to-48 and 48-to-64 changes retained. This integral does
not evaluate the edge logarithms or solid angle. Well-separated and exterior
plane centers were chosen so that this ordinary quadrature converges. It does
not establish a bound for arbitrary near-sheet Taylor centers.

**All nonzero scalar cases:** edge logarithms and the solid angle are evaluated
with mpmath at 100 and 140 decimal digits, beginning with the exact binary64
input values. This is high precision of the shared closed form, explicitly
not a formulation-independent reference. The 140-digit value is rounded to the
binary64 scalar reference. Regular and far scalar values are independently
checked by Green surface quadrature at orders 32 and 64. Constant coefficients
of all Taylor maps also receive a 140-digit shared-formula cross-check.

**Demanding scalars:** for the unit right triangle 0<=x<=1, 0<=y<=1-x, the y
integral of the Green kernel can be performed without the triangle closed form.
Let dx=px-x, a=dx^2+pz^2, l=-py, h=1-x-py, R(t)=sqrt(a+t^2), and
D=[h/R(h)-l/R(l)]/a. The remaining one-dimensional integrands are

    Kx: dx*D,     Ky: 1/R(h)-1/R(l),     Kz: pz*D.

The appropriate continuous limit is used at a=0 outside the y segment. mpmath
tanh-sinh integrates these functions over x at 50 and 70 digits, with geometric
splitting around px and 1-py to resolve the narrow features. This independently
integrates the Green kernel and does not reuse the edge/solid-angle formula.
Both precise integral values and both differences from the closed form are
retained. This reduced control is deliberately limited to the unit right
triangle; it is not claimed as a general independent arbitrary-triangle API.

**Average trace:** the closed form sets the exact planar normal component to
zero. Independent reduced integrals at positive and negative heights 1e-14 and
1e-18 give two symmetric averages; their refinement and agreement with the
exact trace are tested. Tests also check the normal jump equals sigma.
These averages are limiting controls, not a singular quadrature at height zero.

**Error statuses and zero charge:** a separate rational predicate uses Python
`Fraction` values representing exact binary64 geometry. It checks nonzero area,
exact coplanarity, and oriented edge signs. Nonzero charged edges/vertices give
8, charged face-interior Taylor centers give 9, and scalar interiors give the
average trace. Geometry, order, radius, observer, and charge validity precede
the zero-charge shortcut. There is no floating tolerance for these predicates.

## Acceptance and reproduction

The frozen coefficient budget is
max(abs(candidate-reference))/max(1,max(abs(reference))) <= 1e-10.
Independent refinement uses the same normalization with limit 1e-11.
The scalar per-observer budget is
norm(candidate-reference) <= 1e-12 T + 1e-10*norm(reference).
Every retained scalar control must pass the scalar budget, as well as the
refinement threshold. These are corpus-bounded numerical checks, not rigorous
interval enclosures. The corpus contains no unsupported reference case or
silent skip. Native assembly cancellation, holes/T-junctions, composition,
failed reload, and malformed API shapes remain separate release tests.

From the exported package root, with NumPy and mpmath installed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 8 prlimit --as=8589934592 timeout 600 python3 tools/generate_reference.py > corpus/generation.jsonl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 8 prlimit --as=8589934592 timeout 600 python3 -m unittest discover -s tests -p test_reference.py -v
```

`taskset` and `prlimit` are Linux resource controls; change the CPU number to an
available CPU on another host, or provide equivalent controls. Numerical
libraries should retain one thread. Generation is also available per bounded
worker using `--case ID`, followed by `--collect`; those commands write/read
`corpus/parts/` and verify case and generator hashes before collection. A failed
qualification causes nonzero exit after the full reference/report is saved.

The tests check artifact integrity, exact statuses, all coefficient shapes,
all recorded convergence results, winding/translation/scale/charge invariants,
zero fields, vacuum divergence and curl through every available degree, the
far-field monopole, the average trace jump, and Taylor evaluation against fresh
Green integration at displaced observers. They qualify the oracle itself and
do not replace the independent review or candidate conformance harness.
