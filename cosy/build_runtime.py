"""Build an isolated TRLOAD/TRFLD runtime from an external COSYFFAG source tree.

The licensed COSY sources and the compatible COSYFFAG build infrastructure are
external prerequisites. This is not a builder for arbitrary stock COSY trees.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cosyffag-source', type=Path, required=True)
    parser.add_argument('--pristine', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        parser.error('output already exists; choose a fresh isolated output path')
    if not (args.cosyffag_source / 'cosy_ext/build_cosy.sh').is_file():
        parser.error('expected the compatible external cosy_ext/build_cosy.sh')
    if not args.pristine.is_dir():
        parser.error('licensed pristine source directory is missing')
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', COMPILER='gfortran', OUTPUT=str(output),
               TRIANGLE_ROOT=str(args.cosyffag_source.resolve()),
               TRIANGLE_PRISTINE=str(args.pristine.resolve()),
               TRIANGLE_OUTPUT=str(output),
               TRIANGLE_SOURCE=str(root / 'fortran/native/tan_mod.f90'),
               TRIANGLE_EXACT_SOURCE=str(root / 'fortran/triangle_exact_geometry.c'),
               TRIANGLE_BRIDGE_SOURCE=str(root / 'cosy/triangle_assembly_proc.f'),
               TRIANGLE_DISPATCH_SCRIPT=str(root / 'cosy/source_overlay.py'))
    with tempfile.TemporaryDirectory(prefix='tan-cosy-build-') as temporary:
        script = Path(temporary) / 'build.sh'
        subprocess.run(['python3', str(root / 'cosy/source_overlay.py'),
                        'prepare', str(script)], env=env, check=True)
        subprocess.run(['bash', str(script)], env=env, check=True, timeout=900)
    if not output.is_file():
        raise RuntimeError('builder exited without producing the requested runtime')


if __name__ == '__main__':
    main()
