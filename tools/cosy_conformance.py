"""Run the licensed COSY bridge against every representable valid corpus case.

This launches COSY locally and exports all spatial coefficients. An absent
licensed runtime is a prerequisite failure, never a substituted saved export.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cosy.check_bridge import run
from conformance import compare
from triangular_pm_fields import PreparedCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime, library = args.runtime.resolve(strict=True), args.library.resolve(strict=True)
    cases = json.loads((ROOT/'corpus/cases.json').read_text())['cases']
    reference = json.loads((ROOT/'corpus/reference.json').read_text())
    refs = {r['id']: r for r in reference['results']}
    records = []
    # COSY path strings and parser buffers are short; a fresh short directory
    # also isolates the native global cache and all interpreter output files.
    with tempfile.TemporaryDirectory(prefix='tc-') as temporary:
        work = Path(temporary)
        for order in (0, 1, 2, 4, 6):
            selected = [c for c in cases if c['order']==order and c['expected_status'] in (0,8,9)]
            body = []
            for index, case in enumerate(selected):
                cache = work/f'{order}-{index}.tan'
                PreparedCache([case['vertices']], [case['sigma']]).write(cache)
                body += [f"TRLOAD '{cache}' STATUS ;", "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;"]
                for axis, coordinate in enumerate('XYZ'):
                    value = float(case['point'][axis])
                    expr = f'{value:.17E}'
                    if order:
                        expr += f'+({float(case["radius"]):.17E})*DA({axis+1})'
                    body.append(f'{coordinate} := {expr} ;')
                body += ['TRFLD X Y Z BX BY BZ 1 STATUS ;', "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;"]
                if case['expected_status']==0:
                    for component, name in enumerate(('BX','BY','BZ')):
                        for e in refs[case['id']]['exponents']:
                            code = ''.join(str(i+1)*n for i,n in enumerate(e)) or '0'
                            label = ' '.join(map(str,e))+' 0'
                            value = f'COEF({name},{code})' if order else name
                            body.append(f"WRITE 6 'VALUE {3*index+component} {label} '&SF({value},'(E26.17)') ;")
            directory = work/f'q{order}'
            values, statuses = run(runtime, library, directory, body, order=max(1,order))
            if len(statuses) != 2*len(selected):
                raise RuntimeError(f'COSY returned {len(statuses)} statuses for {len(selected)} cases')
            stdout = (directory/'stdout.txt').read_text()
            for index, case in enumerate(selected):
                if statuses[2*index] != 0:
                    raise RuntimeError(f'COSY load failed: {case["id"]}')
                status = statuses[2*index+1]
                powers = refs[case['id']]['exponents'] if status==0 else []
                coefficients = [[values[(3*index+a,tuple(e)+(0,))] for e in powers]
                                for a in range(3)] if status==0 else []
                result = dict(status=status,exponents=powers,coefficients=coefficients)
                problems, error = compare(case,result,refs[case['id']])
                records.append(dict(id=case['id'],passed=not problems,problems=problems,
                                    scaled_error=error,result=result))
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.with_name(args.output.stem+f'-q{order}.txt').write_text(stdout)
    report = dict(schema_version=1, actually_executed=True,
                  runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  cases_sha256=reference['cases_sha256'],
                  skipped_invalid_input_cases=[c['id'] for c in cases if c['expected_status'] not in (0,8,9)],
                  all_passed=all(r['passed'] for r in records),results=records)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"COSY: {sum(r['passed'] for r in records)}/{len(records)} passed")
    return 0 if report['all_passed'] else 1


if __name__=='__main__':
    sys.exit(main())
