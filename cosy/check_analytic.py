"""Execute the released analytic FOX source, including explicit domain failures.

Licensed runtime and COSY.bin are external prerequisites. Source is nested in
RUN so calls resolve to these released routines, not a macro-library copy.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import re
import subprocess
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent


def quadrature(n):
    """Independent surface integral and spatial Jacobian, unit right triangle."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes, weights = (nodes + 1) / 2, weights / 2
    x, v = np.meshgrid(nodes, nodes, indexing='ij')
    y = (1-x)*v
    w = np.outer(weights, weights)*(1-x)/(4*math.pi)
    d = np.stack((.2-x, .3-y, np.full_like(x, .4)), axis=-1)
    r2 = np.sum(d*d, axis=-1)
    field = np.einsum('ij,ijk->k', w/r2**1.5, d)
    jacobian = np.eye(3)*np.sum(w/r2**1.5)
    jacobian -= 3*np.einsum('ij,ijk,ijl->kl', w/r2**2.5, d, d)
    return np.column_stack((field, jacobian))


def bounds():
    for kind, limit in ((resource.RLIMIT_AS, 8*1024**3), (resource.RLIMIT_CPU, 60)):
        soft, hard = resource.getrlimit(kind)
        resource.setrlimit(kind, (min(soft, limit) if soft != resource.RLIM_INFINITY else limit, hard))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime, library = args.runtime.resolve(strict=True), args.library.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    source = (HERE/'analytic_triangle_routines.fox').read_text()
    ordinary = '0 0 0 1 0 0 0 1 0'
    cases = [
        ('regular', ordinary, '1', ('.2+DA(1)', '.3+DA(2)', '.4+DA(3)'), None),
        ('near_vertex', ordinary, '1', ('0', '0', '1E-16'), 'vertex distance'),
        ('small_charge', ordinary, '1E-31', ('.2', '.3', '.4'), 'nonzero charge scale'),
        ('small_normal', '0 0 0 1E-16 0 0 0 1E-16 0', '1', ('.2', '.3', '.4'), 'face normal scale'),
        ('short_edge0', '0 0 0 1E-16 0 0 0 1 0', '1', ('.2', '.3', '.4'), 'edge length'),
        ('short_edge1', '0 1 0 0 0 0 1E-16 0 0', '1', ('.2', '.3', '.4'), 'edge length'),
        ('short_edge2', '0 0 0 0 1 0 1E-16 0 0', '1', ('.2', '.3', '.4'), 'edge length'),
        ('coplanar', ordinary, '1', ('.2', '.3', '0'), 'coplanar observer'),
        ('zero_charge_degenerate', '0 0 0 0 0 0 0 0 0', '0', ('0', '0', '0'), None),
    ]
    reference = quadrature(64)
    refinement = float(np.max(np.abs(reference-quadrature(32))))
    assert refinement < 1e-13, refinement
    records = []
    with tempfile.TemporaryDirectory(prefix='ta-') as temporary:
        work = Path(temporary)
        for name, vertices, sigma, point, failure in cases:
            directory = work/name
            directory.mkdir()
            (directory/'COSY.bin').symlink_to(library)
            lines = ["INCLUDE 'COSY' ;", 'PROCEDURE RUN ;',
                     'VARIABLE X 200 ; VARIABLE Y 200 ; VARIABLE Z 200 ;',
                     source, 'DEF ; OV 1 0 3 ; DAEPS 1E-250 ;',
                     'SI_VFR_BX := 10 ; SI_VFR_BY := 20 ; SI_VFR_BZ := 30 ;']
            lines += [f'{a} := {v} ;' for a, v in zip('XYZ', point)]
            lines += [f'SI_VFIELD_TRI_CHARGE {vertices} {sigma} X Y Z ;']
            for axis, a in enumerate('XYZ'):
                for code in range(4):
                    lines.append(f"WRITE 6 'VALUE {axis} {code} '&SF(COEF(SI_VFR_B{a},{code}),'(E26.17)') ;")
            lines += ["WRITE 6 'ANALYTIC COMPLETE' ;", 'ENDPROCEDURE ;', 'RUN ;', 'END ;']
            deck = '\n'.join(lines)+'\n'
            (directory/'run.fox').write_text(deck)
            result = subprocess.run([str(runtime), 'run.fox'], cwd=directory,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=75,
                                    preexec_fn=bounds,
                                    env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            text = result.stdout
            (args.output/f'{name}.txt').write_text(text)
            record = dict(name=name, vertices=vertices, sigma=sigma, observer=point,
                          expected_failure=failure, returncode=result.returncode,
                          deck_sha256=hashlib.sha256(deck.encode()).hexdigest())
            if failure:
                passed = (f'TRI ANALYTIC UNSUPPORTED: {failure}' in text
                          and 'ANALYTIC COMPLETE' not in text
                          and re.search(r'QUIT AT\s+-1\b', text) is not None)
            else:
                values = np.full((3, 4), np.nan)
                for line in text.splitlines():
                    parts = line.split()
                    if len(parts) == 4 and parts[0] == 'VALUE':
                        values[int(parts[1]), int(parts[2])] = float(parts[3])
                expected = reference.copy() if name == 'regular' else np.zeros((3, 4))
                expected[:, 0] += [10, 20, 30]
                error = float(np.max(np.abs(values-expected)))
                passed = (result.returncode == 0 and 'ANALYTIC COMPLETE' in text
                          and not any(s in text for s in ('### ERROR', '$$$ ERROR', '### QUIT', 'QUIT AT'))
                          and error < 2e-12)
                record.update(values=values.tolist() if np.isfinite(values).all() else None,
                              expected=expected.tolist(),
                              max_absolute_error=error if math.isfinite(error) else None)
            record['passed'] = bool(passed)
            records.append(record)
    report = dict(actually_executed=True, runtime=str(runtime), library=str(library),
                  runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  quadrature_refinement_max_absolute_change=refinement,
                  all_passed=all(r['passed'] for r in records), results=records)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f"Analytic COSY: {sum(r['passed'] for r in records)}/{len(records)} passed")
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
