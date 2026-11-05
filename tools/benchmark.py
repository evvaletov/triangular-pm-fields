"""Bounded single-process preparation and native-query benchmark.

Reports raw samples and both CPU and wall time. It verifies canonical cache
identity against the retained constructor baseline before recording a speedup.
Native mode1 is the accuracy reference for mode0 and mode2 within this regular
query workload. Independent accuracy qualification is tools/conformance.py.
"""
import argparse
import ctypes as ct
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import numpy as np
from triangular_pm_fields import PreparedCache
from triangular_pm_fields.cache_store import load_or_prepare

ROOT = Path(__file__).resolve().parents[1]


def measure(function, samples=7, repeats=1):
    function()  # excluded warmup
    cpu, wall = [], []
    for _ in range(samples):
        t, c = time.perf_counter(), time.process_time()
        for _ in range(repeats):
            function()
        wall.append((time.perf_counter()-t)/repeats)
        cpu.append((time.process_time()-c)/repeats)
    return dict(samples=samples,repeats_per_sample=repeats,excluded_warmups=1,
                cpu_s=cpu,wall_s=wall,median_cpu_s=statistics.median(cpu),
                median_wall_s=statistics.median(wall))


def mesh(n):
    triangles=[]
    for i in range(n):
        for j in range(n):
            a=[i/n,j/n,0];b=[(i+1)/n,j/n,0]
            c=[(i+1)/n,(j+1)/n,0];d=[i/n,(j+1)/n,0]
            triangles.extend([[a,b,c],[a,c,d]])
    return np.asarray(triangles),np.ones(2*n*n)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=ROOT/'build')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('retained_baseline',ROOT/'tests/baseline_prepare_cache.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records=[]
    with tempfile.TemporaryDirectory(prefix='tan-bench-') as temporary:
        for n in (2,4,8):
            triangles,charges=mesh(n)
            actual=PreparedCache(triangles,charges)
            expected=module.PreparedCache(triangles,charges)
            if actual.text != expected.text:
                raise RuntimeError('baseline/optimized canonical cache mismatch')
            baseline=measure(lambda:module.PreparedCache(triangles,charges))
            optimized=measure(lambda:PreparedCache(triangles,charges))
            store=Path(temporary)/f'store{n}'
            load_or_prepare(triangles,charges,store)
            warm=measure(lambda:load_or_prepare(triangles,charges,store),repeats=10)
            records.append(dict(triangles=len(triangles),canonical_bytes_equal=True,
                                baseline_prepare=baseline,optimized_prepare=optimized,
                                persistent_verified_hit=warm,
                                preparation_cpu_speedup=baseline['median_cpu_s']/optimized['median_cpu_s']))
        cache=Path(temporary)/'square.tan';actual.write(cache)
        lib=ct.CDLL(str((args.build/'libtan.so').resolve()))
        dp=ct.POINTER(ct.c_double);ip=ct.POINTER(ct.c_int)
        lib.tan_load_c.argtypes=[ct.c_char_p,ip];lib.tan_load_c.restype=None
        lib.tan_batch_c.argtypes=[ct.c_int,dp,ct.c_int,dp,ip];lib.tan_batch_c.restype=None
        status=ct.c_int();lib.tan_load_c(os.fsencode(cache),ct.byref(status))
        if status.value:raise RuntimeError(f'load status {status.value}')
        rng=np.random.default_rng(20261104)
        points=np.ascontiguousarray(rng.uniform([-.2,-.2,.3],[1.2,1.2,1.5],(512,3)))
        fields=np.empty_like(points);statuses=np.empty(len(points),dtype=np.int32)
        def query(mode):
            lib.tan_batch_c(len(points),points.ctypes.data_as(dp),mode,
                            fields.ctypes.data_as(dp),statuses.ctypes.data_as(ip))
            if np.any(statuses):raise RuntimeError('native batch error')
        query(1);reference=fields.copy();queries=[]
        for mode in (0,1,2):
            query(mode)
            ratios=np.linalg.norm(fields-reference,axis=1)/(1e-12+1e-10*np.linalg.norm(reference,axis=1))
            if not np.isfinite(fields).all() or ratios.max()>1:
                raise RuntimeError(f'mode {mode} missed accuracy budget')
            queries.append(dict(mode=mode,observers=len(points),cache_load_excluded=True,
                                max_scalar_budget_ratio=float(ratios.max()),
                                timing=measure(lambda:query(mode),repeats=5)))
    report=dict(schema_version=1,environment=dict(python=platform.python_version(),numpy=np.__version__,
                system=platform.system(),machine=platform.machine(),
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}),
                caveat='Shared host; bounded regular workload. Timing is not a universal speed guarantee.',
                preparation=records,native_queries=queries)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({r['triangles']:r['preparation_cpu_speedup'] for r in records},indent=2))


if __name__=='__main__':main()
