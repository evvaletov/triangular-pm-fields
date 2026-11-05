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
