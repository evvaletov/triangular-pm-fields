"""Optional pinned JAX adapter: all 48 regular-center corpus maps plus jet controls."""
import os
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import jax
import jaxlib
import numpy as np
from adapters.jax_repair_20261103.adapter import (
    compiled_directional_coefficients, polynomial_control, control_reference)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not jax.config.x64_enabled:
        raise RuntimeError('JAX x64 must be enabled')
    cases=json.loads((ROOT/'corpus/cases.json').read_text())['cases']
    reference=json.loads((ROOT/'corpus/reference.json').read_text())
    refs={r['id']:r for r in reference['results']}
    records=[]
    for q in (1,2,4,6):
        control,_=compiled_directional_coefficients(q,polynomial_control)
        expected=control_reference(q)
        actual=np.asarray(control(np.eye(3),np.zeros(3)).block_until_ready())
        err=float(np.max(np.abs(actual-expected))/max(1.,np.max(np.abs(expected))))
        records.append(dict(id=f'polynomial_control_q{q}',scaled_error=err,passed=bool(np.isfinite(actual).all() and err<=1e-10)))
        function,reconstruction=compiled_directional_coefficients(q)
        for c in cases:
            if c['suite']!='taylor' or c['order']!=q:continue
            start=time.perf_counter()
            raw=np.asarray(function(np.array(c['vertices']),np.array(c['point'])).block_until_ready())
            expected=np.asarray(refs[c['id']]['coefficients'])
            powers=refs[c['id']]['exponents']
            actual=raw*(float(c['sigma'])/(4*math.pi))*np.array([c['radius']**sum(e) for e in powers])
            err=float(np.max(np.abs(actual-expected))/max(1.,np.max(np.abs(expected))))
            passed=bool(np.isfinite(actual).all() and err<=1e-10)
            records.append(dict(id=c['id'],scaled_error=err,passed=passed,coefficients=actual.tolist(),
                                elapsed_including_possible_compile_s=time.perf_counter()-start))
            print(c['id'],'PASS' if passed else 'FAIL',err,flush=True)
    report=dict(adapter='jax_repair_20261103',jax=jax.__version__,jaxlib=jaxlib.__version__,
                x64=True,platform=jax.default_backend(),cases_sha256=reference['cases_sha256'],
                scope='48 regular-center maps and four manufactured jet controls; source boundaries excluded',
                all_passed=all(r['passed'] for r in records),results=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['all_passed'] else 1


if __name__=='__main__':sys.exit(main())
