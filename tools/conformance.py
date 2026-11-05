"""Execute frozen independent cases against the built language consumers.

Standalone C++/Julia use their full boundary/input contract. Native consumers
use representable C-ABI cases; invalid geometry cannot be written as a cache.
No successful field is accepted unless every coefficient agrees with the
independent corpus. Raw outputs and every discrepancy remain in the report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from triangular_pm_fields import NativeLibrary, NativeError, PreparedCache
from triangular_pm_fields.native import exponents
from triangular_pm_fields.reference import field_jet

ROOT = Path(__file__).resolve().parents[1]


def execute(command):
    start = time.perf_counter()
    process = subprocess.run([str(x) for x in command], capture_output=True,
                             text=True, timeout=180)
    try:
        result = json.loads(process.stdout)
    except (ValueError, TypeError):
        result = {'status': -1, 'exponents': [], 'coefficients': []}
    result.update(exit_code=process.returncode, stderr=process.stderr,
                  elapsed_s=time.perf_counter()-start)
    if result['status'] == -1:
        result['stdout'] = process.stdout
    return result


def execute_batch(command, arguments):
    """Amortize Julia startup without treating a batch error as each row's exit."""
    process = subprocess.run([str(x) for x in command],
        input=''.join(' '.join(map(str,row))+'\n' for row in arguments),
        capture_output=True,text=True,timeout=600)
    lines = process.stdout.splitlines()
    if len(lines) != len(arguments):
        raise RuntimeError(f'batch returned {len(lines)} rows for {len(arguments)} inputs: '+process.stderr[-2000:])
    results = [json.loads(line) for line in lines]
    if (process.returncode == 0) != all(row['status']==0 for row in results):
        raise RuntimeError('batch exit code/status disagreement: '+process.stderr[-2000:])
    for row in results:
        row.update(batch_exit_code=process.returncode,stderr=process.stderr)
    return results


def compare(case, result, reference):
    problems = []
    if result.get('status') != case['expected_status']:
        problems.append(f"status {result.get('status')} != {case['expected_status']}")
    if 'exit_code' in result and (result['exit_code'] == 0) != (result.get('status') == 0):
        problems.append('exit code/status disagreement')
    if case['expected_status'] or result.get('status'):
        if result.get('exponents') or result.get('coefficients'):
            problems.append('error output is not empty')
        return problems, None
    if result.get('exponents') != reference['exponents']:
        problems.append('exponent layout mismatch')
    a = np.asarray(result.get('coefficients'), dtype=float)
    b = np.asarray(reference['coefficients'], dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all():
        return problems + ['nonfinite output or wrong coefficient shape'], None
    error = float(np.max(np.abs(a-b)) / max(1.0, np.max(np.abs(b))))
    if error > 1e-10:
        problems.append(f'coefficient error {error} exceeds 1e-10')
    if case['order'] == 0:
        ratio = float(np.linalg.norm(a-b) / (1e-12 + 1e-10*np.linalg.norm(b)))
        if ratio > 1:
            problems.append(f'scalar budget ratio {ratio} exceeds 1')
    return problems, error


def native_fortran(build, cache, case):
    process = subprocess.run([str(build/'tan_cli'), str(cache),
                              *map(str, case['point']), '1', str(case['order']),
                              str(case['radius'])], capture_output=True, text=True, timeout=90)
    rows = process.stdout.splitlines()
    status = int(rows[0].split()[1]) if rows and rows[0].startswith('status ') else -1
    powers, columns = [], []
    if status == 0:
        for row in rows[3:]:
            values = row.split()
            powers.append(list(map(int, values[1:4])))
            columns.append(list(map(float, values[4:7])))
    return dict(status=status, exponents=powers,
                coefficients=np.array(columns).T.tolist() if columns else [],
                exit_code=process.returncode, stderr=process.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'build')
    parser.add_argument('--julia', default='julia')
    parser.add_argument('--native-only', action='store_true', help='development subset, not release qualification')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    library = build/'libtan.so'
    cases_path = ROOT/'corpus/cases.json'
    reference_path = ROOT/'corpus/reference.json'
    specification = json.loads(cases_path.read_text())
    reference_doc = json.loads(reference_path.read_text())
    if hashlib.sha256(cases_path.read_bytes()).hexdigest() != reference_doc['cases_sha256']:
        raise RuntimeError('frozen reference/case hash mismatch')
    references = {x['id']: x for x in reference_doc['results']}
    native = NativeLibrary(library)
    rows, skipped = [], []
    with tempfile.TemporaryDirectory(prefix='tan-conformance-') as temporary:
        julia_standalone, julia_native = {}, {}
        if not args.native_only:
            batch = [[c['order'],c['radius'],c['sigma'],
                      *np.asarray(c['vertices']).ravel().tolist(),*c['point']]
                     for c in specification['cases']]
            command = [args.julia,'--startup-file=no','--project='+str(ROOT/'julia')]
            results = execute_batch([*command,ROOT/'julia/cli.jl','--batch'],batch)
            julia_standalone = dict(zip((c['id'] for c in specification['cases']),results))
            native_cases, batch = [], []
            for c in specification['cases']:
                path = Path(temporary)/(c['id']+'.tan')
                try:
                    PreparedCache([c['vertices']],[c['sigma']]).write(path)
                except (ValueError,TypeError,OverflowError):
                    continue
                native_cases.append(c)
                batch.append([library,path,*c['point'],1,c['order'],c['radius']])
            results = execute_batch([*command,ROOT/'julia/native_cli.jl','--batch'],batch)
            julia_native = dict(zip((c['id'] for c in native_cases),results))
        for case in specification['cases']:
            providers = {}
            if not args.native_only and case['suite']=='taylor':
                powers = references[case['id']]['exponents']
                jets = field_jet(case['vertices'],case['point'],case['order'])
                values = np.array([[j.c.get(tuple(e),0j) for e in powers] for j in jets])
                if np.max(np.abs(values.imag)) > 1e-12:
                    raise RuntimeError('Python analytic baseline returned complex field coefficients')
                coefficients = values.real*(case['sigma']/(4*math.pi))*np.array(
                    [case['radius']**sum(e) for e in powers])
                providers['python_analytic'] = dict(status=0,exponents=powers,
                                                    coefficients=coefficients.tolist())
            if not args.native_only:
                cli = [case['order'], case['radius'], case['sigma'],
                       *np.asarray(case['vertices']).ravel().tolist(), *case['point']]
                providers['cpp_standalone'] = execute([build/'cpp/triangle_cli', *cli])
                providers['julia_standalone'] = julia_standalone[case['id']]
            cache = Path(temporary)/(case['id']+'.tan')
            try:
                PreparedCache([case['vertices']], [case['sigma']]).write(cache)
            except (ValueError, TypeError, OverflowError) as exc:
                skipped.append({'id': case['id'], 'consumers': 'native',
                                'reason': 'geometry cannot form a native cache: '+str(exc)})
            else:
                providers['fortran_native'] = native_fortran(build, cache, case)
                # Invalid q/radius/observer belong to the low-level ABI tests;
                # the Python wrapper rejects these before its native call.
                try:
                    values = native.evaluate(cache, [case['point']], order=case['order'],
                                             radius=case['radius'])[0]
                    providers['python_native'] = dict(status=0,
                        exponents=[list(x) for x in exponents(case['order'])], coefficients=values.tolist())
                except NativeError as exc:
                    providers['python_native'] = dict(status=exc.status, exponents=[], coefficients=[])
                except ValueError as exc:
                    skipped.append({'id': case['id'], 'consumers': 'python_native',
                                    'reason': 'Python input validation: '+str(exc)})
                if not args.native_only:
                    native_args = [library, cache, *case['point'], 1, case['order'], case['radius']]
                    providers['cpp_native'] = execute([build/'cpp/native_cli', *native_args])
                    providers['julia_native'] = julia_native[case['id']]
            for provider, result in providers.items():
                problems, error = compare(case, result, references[case['id']])
                rows.append(dict(id=case['id'], provider=provider, passed=not problems,
                                 problems=problems, scaled_error=error, result=result))
            print(case['id'], 'PASS' if all(x['passed'] for x in rows if x['id']==case['id']) else 'FAIL', flush=True)
    report = dict(schema_version=1, cases_sha256=reference_doc['cases_sha256'],
                  reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                  native_only=args.native_only, all_passed=all(x['passed'] for x in rows),
                  executed_by_provider=dict(Counter(x['provider'] for x in rows)),
                  skipped=skipped, results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','skipped')}, indent=2))
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
