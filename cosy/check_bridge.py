"""Bounded bridge checks; stub mode tests plumbing, not physical fields."""
import argparse
from decimal import Decimal, localcontext
import itertools
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ZERO = (0, 0, 0, 0)


def keys(n, q):
    return [e for d in range(q+1) for e in itertools.product(range(d+1), repeat=n)
            if sum(e) == d]


def multiply(a, b):
    out = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            k = tuple(x+y for x, y in zip(ka, kb))
            if sum(k) <= 6:
                out[k] = out.get(k, 0.) + va*vb
    return out


def compose(coefficients, deltas):
    powers = []
    for delta in deltas:
        row = [{ZERO: 1.}]
        for _ in range(6):
            row.append(multiply(row[-1], delta))
        powers.append(row)
    out = [dict() for _ in range(3)]
    for j, e in enumerate(keys(3, 6)):
        term = multiply(multiply(powers[0][e[0]], powers[1][e[1]]), powers[2][e[2]])
        for axis in range(3):
            for k, v in term.items():
                out[axis][k] = out[axis].get(k, 0.) + coefficients[axis][j]*v
    return out


def run(runtime, library, directory, body, order=6, declarations=''):
    directory.mkdir(parents=True, exist_ok=True)
    lib = directory/'COSY.bin'
    if not lib.exists():
        lib.symlink_to(library)
    deck = """INCLUDE 'COSY' ;
PROCEDURE RUN ;
 VARIABLE X 1000 ; VARIABLE Y 1000 ; VARIABLE Z 1000 ;
 VARIABLE BX 1000 ; VARIABLE BY 1000 ; VARIABLE BZ 1000 ;
 VARIABLE STATUS 1 ;
 VARIABLE SMALL 1 ;
 %s
 DEF ;
 OV %d 0 4 ;
 DAEPS 1E-250 ;
 %s
 WRITE 6 'BRIDGE COMPLETE' ;
ENDPROCEDURE ;
RUN ;
END ;
""" % (declarations, order, '\n '.join(body))
    (directory/'run.fox').write_text(deck)
    result = subprocess.run([str(runtime), 'run.fox'],
                            cwd=directory, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=90,
                            env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    (directory/'stdout.txt').write_text(result.stdout)
    assert result.returncode == 0 and 'BRIDGE COMPLETE' in result.stdout, result.stdout[-3000:]
    assert not any(x in result.stdout for x in ('### ERROR', '$$$ ERROR', '### QUIT')), result.stdout[-3000:]
    values = {}
    statuses = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == 'VALUE':
            values[(int(parts[1]), tuple(map(int, parts[2:6])))] = float(parts[6])
        if parts and parts[0] == 'STATUS':
            statuses.append(int(float(parts[1])))
    return values, statuses


def outputs(names=('BX', 'BY', 'BZ')):
    result = ["WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;"]
    for axis, name in enumerate(names):
        for e in keys(4, 6):
            code = ''.join(str(i+1)*n for i, n in enumerate(e)) or '0'
            label = ' '.join(map(str, e))
            result.append(f"WRITE 6 'VALUE {axis} {label} '&SF(COEF({name},{code}),'(E26.17)') ;")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--coefficients', type=Path, help='JSON 3 by 84 native coefficient array at (.2,.3,.4)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--automatic', action='store_true', help='Require native scalar mode2 support')
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    stub = args.cache is None
    cache = args.cache.resolve() if args.cache else target/'factor.txt'
    if stub:
        cache.write_text('1\n')
        coefficients = [[(100*(a+1)+j+1)/1000 for j in range(84)] for a in range(3)]
    else:
        coefficients = json.loads(args.coefficients.read_text())
    unit = [{tuple(int(i == j) for i in range(4)): 1.} for j in range(3)]
    nonlinear = [dict(p) for p in unit]
    nonlinear[0][(0, 1, 0, 1)] = .1
    nonlinear[1][(0, 0, 2, 0)] = .2
    nonlinear[2][(1, 0, 0, 1)] = .3
    cases = [('identity', ['.2+DA(1)', '.3+DA(2)', '.4+DA(3)'], unit, False),
             ('nonlinear', ['.2+DA(1)+.1*DA(2)*DA(4)', '.3+DA(2)+.2*DA(3)*DA(3)', '.4+DA(3)+.3*DA(1)*DA(4)'], nonlinear, False),
             ('mixed', ['.2', '.3+DA(2)', '.4'], [{}, unit[1], {}], False),
             ('alias', ['.2+DA(1)', '.3+DA(2)', '.4+DA(3)'], unit, True)]
    report = {}
    for label, xyz, deltas, alias in cases:
        body = [f"TRLOAD '{cache}' STATUS ;", "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;"]
        body += [f'{name} := {expr} ;' for name, expr in zip(('X','Y','Z'), xyz)]
        names = ('X','Y','Z') if alias else ('BX','BY','BZ')
        body += [f'TRFLD X Y Z {" ".join(names)} 1 STATUS ;'] + outputs(names)
        actual, statuses = run(args.runtime.resolve(), args.library.resolve(), target/label, body)
        assert statuses == [0, 0], (label, statuses)
        expected = compose(coefficients, deltas)
        errors = [abs(actual[a, k]-expected[a].get(k, 0.))/(1+abs(expected[a].get(k, 0.)))
                  for a in range(3) for k in keys(4, 6)]
        assert len(actual) == 630 and max(errors) < 2e-12, (label, max(errors))
        report[label] = {'coefficients_checked': len(actual), 'max_scaled_error': max(errors)}
    body = [f"TRLOAD '{cache}' STATUS ;", "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;",
            'BX := 123 ; BY := 124 ; BZ := 125 ;',
            'TRFLD .2 .3 .4 BX BY BZ 99 STATUS ;', "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;",
            'X := .2+DA(1) ;', 'TRFLD X .3 .4 BX BY BZ 0 STATUS ;',
            "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;", 'TRFLD X .3 .4 BX BY BZ 1 STATUS ;',
            "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;",
            "WRITE 6 'VALUE 0 0 0 0 0 '&SF(BX,'(E26.17)') ;"]
    actual, statuses = run(args.runtime.resolve(), args.library.resolve(), target/'invalid', body, order=7)
    assert statuses == [0,1002,1002,1003] and actual[0,ZERO] == 123, (statuses, actual)
    report['invalid'] = {'statuses': statuses, 'outputs_preserved': True}
    body = [f"TRLOAD '{cache}' STATUS ;", 'TRFLD .2 .3 .4 BX BY BZ 1 STATUS ;']
    body += ["WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;"]
    for a, name in enumerate(('BX','BY','BZ')):
        body += [f"WRITE 6 'VALUE {a} 0 0 0 0 '&SF({name},'(E26.17)') ;"]
    actual, statuses = run(args.runtime.resolve(), args.library.resolve(), target/'scalar', body)
    assert statuses == [0]
    assert all(abs(actual[a,ZERO]-coefficients[a][0]) < 1e-13*(1+abs(coefficients[a][0])) for a in range(3))
    report['scalar'] = {'status': statuses[0]}
    body = [f"TRLOAD '{cache}' STATUS ;", 'X := .2+DA(1) ;',
            'TRFLD X .3 .4 BX BY BZ 2 STATUS ;',
            "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;"]
    if args.automatic:
        body += ['TRFLD .2 .3 .4 BX BY BZ 2 STATUS ;',
                 "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;",
                 "WRITE 6 'VALUE 0 0 0 0 0 '&SF(BX,'(E26.17)') ;"]
    actual, statuses = run(args.runtime.resolve(), args.library.resolve(), target/'automatic', body)
    assert statuses == ([1002,0] if args.automatic else [1002]), statuses
    if args.automatic:
        assert abs(actual[0,ZERO]-coefficients[0][0])<1e-13*(1+abs(coefficients[0][0]))
    report['automatic'] = {'statuses':statuses, 'native_mode2_tested':args.automatic}
    body = [f"TRLOAD '{cache}' STATUS ;", 'X := .2+DA(1) ; Y := .3+DA(2) ; Z := .4+DA(3) ;',
            'SMALL := 123 ;', 'TRFLD X Y Z SMALL BY BZ 1 STATUS ;',
            "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;",
            "WRITE 6 'VALUE 0 0 0 0 0 '&SF(SMALL,'(E26.17)') ;",
            'TRFLD X Y Z BX BX BZ 1 STATUS ;',
            "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;"]
    actual, statuses = run(args.runtime.resolve(), args.library.resolve(), target/'capacity_alias', body)
    assert statuses == [1006,1004] and actual[0,ZERO] == 123, (statuses, actual)
    report['capacity_alias'] = {'statuses': statuses, 'outputs_preserved': True}
    second = target/'reload_cache.txt'
    if stub:
        second.write_text('2\n')
    else:
        lines = cache.read_text().splitlines()
        planes, edges = map(int, lines[1].split())
        with localcontext() as ctx:
            ctx.prec = 2000
            for j in range(2+planes, 2+planes+edges):
                cells = lines[j].split()
                cells[-1] = str(2*Decimal(cells[-1]))
                lines[j] = ' '.join(cells)
        second.write_text('\n'.join(lines)+'\n')
    body = [f"TRLOAD '{cache}' STATUS ;", 'TRFLD .2 .3 .4 BX BY BZ 0 STATUS ;',
            "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;",
            f"TRLOAD '{second}' STATUS ;", "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;",
            'TRFLD .2 .3 .4 BX BY BZ 0 STATUS ;',
            "WRITE 6 'STATUS '&SF(STATUS,'(F12.0)') ;", "WRITE 6 'VALUE 0 0 0 0 0 '&SF(BX,'(E26.17)') ;"]
    actual, statuses = run(args.runtime.resolve(), args.library.resolve(), target/'reload', body)
    assert statuses == [0,0,0] and abs(actual[0,ZERO]-2*coefficients[0][0]) < 1e-13*(1+abs(coefficients[0][0]))
    report['reload'] = {'statuses': statuses, 'factor2_value': actual[0,ZERO]}
    report['stub_only'] = stub
    (target/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
