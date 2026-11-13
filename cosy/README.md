# COSY INFINITY consumer

`triangle_assembly_proc.f` implements `TRLOAD` and `TRFLD` for a compatible
COSYFFAG-based COSY runtime. `analytic_triangle_routines.fox` contains the
project's analytic triangle routines. The runtime, its licensed sources, and
the COSY.bin macro library are external prerequisites and are not distributed
in this repository.

The native bridge composes the spatial Taylor coefficients with COSY DA input
polynomials. It supports nonlinear coordinate maps and additional DA variables.
The project source overlay expects the COSYFFAG build layout and intrinsic
dispatch table used by the recorded qualification. An arbitrary stock COSY
installation does not supply that layout or these two intrinsics.

## Build an isolated runtime

```sh
python cosy/build_runtime.py \
  --cosyffag-source /path/to/compatible/COSYFFAG \
  --pristine '/path/to/licensed/COSY 10.2/UNIX' \
  --output /tmp/triangle-cosy-runtime
```

The output must be new and isolated from the external source tree's production
executables. The builder uses this repository's native kernels and bridge,
with the external build infrastructure and licensed source. Its source
anchors are checked before patching the temporary build script.

## Execute and check

Prepare `triangle.tan` with `examples/triangle.py`. In an isolated working
directory, put that file and `examples/triangle.fox` alongside an authorized
COSY.bin macro library, then run the compatible runtime on `triangle.fox`.
The example prints the scalar field and first spatial coefficient.

For all representable corpus cases:

```sh
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/cosy_conformance.py \
  --runtime /tmp/triangle-cosy-runtime --library /path/to/COSY.bin \
  --output local-results/cosy.json
```

This command launches COSY, checks statuses, and exports every requested
coefficient. Reports retain runtime/library hashes and raw interpreter output.
Additional nonlinear composition, alias/capacity, mode, and reload checks are
available through `cosy/check_bridge.py --help`; pass a triangle cache and
the corresponding `3 x 84` native coefficient array at `(0.2,0.3,0.4)` to
test physical fields. Its optional stub mode tests interface behavior only.

The open-tool CI does not execute licensed COSY. The initial qualification
includes a local runtime build and actual interpreter runs; see the saved
validation summary. No runtime or macro-library binary is included with it.

## Analytic COSYScript routine domain

`SI_VFIELD_TRI_CHARGE` is a separate analytic entry point from `TRLOAD`/`TRFLD`.
It adds its complete triangle contribution to `SI_VFR_BX/BY/BZ` only after the
calculation succeeds. Exactly zero `SIGMA` leaves those outputs unchanged,
including at a vertex or for degenerate geometry. No nonzero charge is silently
rounded to zero.

For nonzero charge, this conservative implementation requires:

- `abs(SIGMA/(4*PI)) >= 1e-30`;
- the norm of `(B-A) cross (C-A)` to be at least `1e-30`;
- all three edge lengths to exceed `1e-15`;
- every constant-part observer-to-vertex distance to be at least `1e-15`;
- a nonzero constant part of the unnormalized observer-plane height.

The distance and area thresholds use the caller's coordinate units. Inputs
must be finite, and intermediate arithmetic must remain representable. These
conditions are a restricted numerical domain, not a statement that excluded
regular physical fields vanish. In particular, the unit triangle at
`(0,0,1e-16)` with nonzero charge has a regular nonzero field but is rejected by
this entry point. Coplanar observers are excluded even outside the triangle;
this analytic routine does not claim the native cache's exact-boundary policy.
For DA inputs these tests select a chart using the constant part; they do not
certify a finite Taylor convergence radius.

The charge, geometry, and observer checks precede field evaluation. The edge
helper also rejects nonpositive rationalization denominators or logarithm gaps.
Failures emit `TRI ANALYTIC UNSUPPORTED: ...` and invoke the existing COSY
`QUIT -1` mechanism. A failed call never returns a zero or partial triangle
contribution. COSY may return operating-system status zero after `QUIT`, so
consumers must check the interpreter diagnostic and expected completion marker.

To execute the released analytic source against an independent regular-point
surface quadrature and its first derivatives, and exercise near-vertex,
charge/geometry threshold, coplanar, and exact-zero-charge cases:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python cosy/check_analytic.py \
  --runtime /tmp/triangle-cosy-runtime --library /path/to/COSY.bin \
  --output local-results/cosy-analytic
```

The checker compiles these routines inside its test procedure, so it does not
silently test an older copy embedded in the external macro library. It retains
raw interpreter output and hashes of the source, runtime, and macro library.
