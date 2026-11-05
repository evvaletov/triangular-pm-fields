#!/usr/bin/env python3
"""Reproduce the frozen corpus without calling any candidate implementation.

Green-kernel Gauss integration adapts quadrature_nodes/quadrature_jet from the
paper triangle.py at the source/hash recorded in corpus/freeze.json. Scalar
closed-form controls use that paper's shared edge/solid-angle formula. The
reduced one-dimensional Green integral is separately derived in corpus/README.md.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import mpmath as mp
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]


def indices(order):
    return tuple((a, b, degree-a-b) for degree in range(order+1)
                 for a in range(degree+1) for b in range(degree-a+1))


def quadrature_nodes(vertices, order):
    t, w = np.polynomial.legendre.leggauss(order)
    t, w = (t+1)/2, w/2
    u, v = np.meshgrid(t, t, indexing='ij')
    a, b = vertices[1]-vertices[0], vertices[2]-vertices[0]
    nodes = vertices[0]+u[..., None]*a+((1-u)*v)[..., None]*b
    weights = w[:, None]*w[None, :]*(1-u)*np.linalg.norm(np.cross(a, b))
    return nodes.reshape(-1, 3), weights.ravel()


def quadrature_jet(vertices, point, order=6, quadrature_order=64):
    """Expand (d+delta)/|d+delta|^3 under a surface Green integral."""
    vertices = np.asarray(vertices, float)
    nodes, weights = quadrature_nodes(vertices, quadrature_order)
    d = np.asarray(point)-nodes
    r2 = np.sum(d*d, axis=1)
    keys = indices(order)
    lookup = {k: i for i, k in enumerate(keys)}
    terms = []
    for axis in range(3):
        key = tuple(int(i == axis) for i in range(3))
        terms.extend([(key, 2*d[:, axis]/r2),
                      (tuple(2*x for x in key), 1/r2)])
    power = np.zeros((len(keys), len(nodes)))
    power[0] = 1
    series = power.copy()
    coefficient = 1.0
    for k in range(1, order+1):
        new = np.zeros_like(power)
        for i, a in enumerate(keys):
            for b, values in terms:
                target = tuple(x+y for x, y in zip(a, b))
                if sum(target) <= order:
                    new[lookup[target]] += power[i]*values
        power = new
        coefficient *= (-1.5-k+1)/k
        series += coefficient*power
    result = np.zeros((3, len(keys)))
    base_weights = weights/r2**1.5
    for axis in range(3):
        c = series*d[:, axis]
        for i, key in enumerate(keys):
            if key[axis] > 0:
                previous = list(key)
                previous[axis] -= 1
                c[i] += series[lookup[tuple(previous)]]
        result[axis] = c@base_weights
    return result


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def sub(a, b):
    return [x-y for x, y in zip(a, b)]


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]]


def classify(case):
    """Contract predicates on exact binary64 inputs, independent of ports."""
    q = case['order']
    if isinstance(q, bool) or not isinstance(q, int) or not 0 <= q <= 6:
        return 5
    radius = float(case['radius'])
    if not np.isfinite(radius) or radius <= 0:
        return 7
    p = np.asarray(case['point'], float)
    if p.shape != (3,) or not np.isfinite(p).all():
        return 6
    v = np.asarray(case['vertices'], float)
    sigma = float(case['sigma'])
    if v.shape != (3, 3) or not np.isfinite(v).all() or not np.isfinite(sigma):
        return 11
    vf = [[Fraction(float(x)) for x in row] for row in v]
    pf = [Fraction(float(x)) for x in p]
    n = cross(sub(vf[1], vf[0]), sub(vf[2], vf[0]))
    if not any(n):
        return 11
    if sigma == 0:
        return 0
    if dot(sub(pf, vf[0]), n) != 0:
        return 0
    sides = [dot(cross(sub(vf[(i+1)%3], vf[i]), sub(pf, vf[i])), n)
             for i in range(3)]
    if min(sides) >= 0:
        if min(sides) == 0:
            return 8
        if q > 0:
            return 9
    return 0


def field_mp(case, dps):
    """Shared closed form at arbitrary precision, NOT formulation-independent."""
    with mp.workdps(dps):
        v = [[mp.mpf(float(x)) for x in row] for row in case['vertices']]
        p = [mp.mpf(float(x)) for x in case['point']]
        norm = lambda a: mp.sqrt(dot(a, a))
        n = cross(sub(v[1], v[0]), sub(v[2], v[0]))
        area2 = norm(n)
        n = [x/area2 for x in n]
        r = [sub(a, p) for a in v]
        radii = [norm(a) for a in r]
        h = dot(sub(p, v[0]), n)
        num = -h*area2
        den = mp.fprod(radii)+sum(dot(r[i], r[(i+1)%3])*radii[(i+2)%3]
                                 for i in range(3))
        omega = 0 if h == 0 else 2*mp.atan2(num, den)
        result = [-omega*x for x in n]
        for i in range(3):
            j = (i+1)%3
            edge = sub(v[j], v[i])
            length = norm(edge)
            m = cross([x/length for x in edge], n)
            ll = mp.log((radii[i]+radii[j]+length)/(radii[i]+radii[j]-length))
            result = [x+y*ll for x, y in zip(result, m)]
        factor = mp.mpf(float(case['sigma']))/(4*mp.pi)
        return [mp.nstr(x*factor, dps) for x in result]


def reduced_integral(case, dps, height=None):
    """Unit right triangle: integrate y exactly, x with mp tanh-sinh.

    This is an integral of the Green kernel, not edge/solid-angle evaluation.
    Only the axis-aligned unit triangle (either winding) is supported here.
    A nonzero height override is used for independently controlled average trace.
    """
    vertices = sorted(tuple(row) for row in case['vertices'])
    assert vertices == [(0., 0., 0.), (0., 1., 0.), (1., 0., 0.)]
    with mp.workdps(dps):
        px, py, pz = [mp.mpf(float(x)) for x in case['point']]
        if height is not None:
            pz = mp.mpf(height)
        rho = mp.sqrt(pz*pz + max(-py, mp.mpf(0))**2)
        # At exterior planar points rho captures the closest projected distance.
        # Geometric split points resolve peaks without blindly increasing order.
        anchors = {mp.mpf(0), mp.mpf(1)}
        for center in (px, 1-py):
            if 0 < center < 1:
                anchors.add(center)
            offset = max(rho, mp.mpf('1e-30'))
            while offset < 2:
                for x in (center-offset, center+offset):
                    if 0 < x < 1:
                        anchors.add(x)
                offset *= 10
        breaks = sorted(anchors)
        def integrand(x, axis):
            dx = px-x
            a = dx*dx+pz*pz
            lo, hi = -py, 1-x-py
            rl, rh = mp.sqrt(a+lo*lo), mp.sqrt(a+hi*hi)
            if axis == 1:
                return 1/rh-1/rl
            # For a=0 outside the integration segment use the continuous limit.
            if a == 0:
                factor = mp.sign(lo)*(1/(2*lo*lo)-1/(2*hi*hi))
            else:
                factor = (hi/rh-lo/rl)/a
            return (dx if axis == 0 else pz)*factor
        factor = mp.mpf(float(case['sigma']))/(4*mp.pi)
        result = [mp.quad(lambda x: integrand(x, axis), breaks)*factor
                  for axis in range(3)]
        return [mp.nstr(x, dps) for x in result]


def array(values):
    return np.asarray(values, dtype=float).reshape(3, -1)


def scaled_max(a, b):
    return float(np.max(np.abs(a-b))/max(1., float(np.max(np.abs(b)))))


def scalar_budget_ratio(a, b):
    return float(np.linalg.norm(a-b)/(1e-12+1e-10*np.linalg.norm(b)))


def calculate(case):
    start = time.monotonic()
    status = classify(case)
    if status != case['expected_status']:
        raise AssertionError(f"frozen status mismatch: {case['id']}: {status}")
    result = dict(id=case['id'], status=status, exponents=[], coefficients=[],
                  method=case['reference_method'], qualified=True)
    arrays = {}
    if status:
        return result, arrays
    q = case['order']
    exponents = indices(q)
    result['exponents'] = exponents
    if case['sigma'] == 0:
        value = np.zeros((3, len(exponents)))
        result['coefficients'] = value.tolist()
        arrays['reference'] = value
        result['refinement'] = dict(method='exact_zero', scaled_max=0.)
        return result, arrays
    if case['suite'] == 'taylor':
        factor = float(case['sigma'])/(4*np.pi)
        scaling = np.array([float(case['radius'])**sum(a) for a in exponents])
        for n in (32, 48, 64):
            arrays[f'gauss_{n}'] = quadrature_jet(case['vertices'], case['point'], q, n)*factor*scaling
        value = arrays['gauss_64']
        refinements = [scaled_max(arrays['gauss_32'], arrays['gauss_48']),
                       scaled_max(arrays['gauss_48'], value)]
        result['refinement'] = dict(orders=[32, 48, 64], scaled_max=refinements,
                                    threshold=1e-11)
        result['qualified'] = max(refinements) <= 1e-11
        # Independent constant-term cross-control, with full MP strings retained.
        precise = field_mp(case, 140)
        result['constant_shared_formula_mp140'] = precise
        ratio = scalar_budget_ratio(value[:, :1], array(precise))
        result['constant_crosscheck_budget_ratio'] = ratio
        result['qualified'] &= ratio <= 1
    else:
        lo, hi = field_mp(case, 100), field_mp(case, 140)
        result['shared_formula_mp100'] = lo
        result['shared_formula_mp140'] = hi
        arrays['closed_form_100'] = array(lo)
        value = array(hi)
        arrays['closed_form_140'] = value
        refinement = scaled_max(array(lo), value)
        result['precision_refinement_scaled_max'] = refinement
        method = case['reference_method']
        if method == 'closed_form_mp_and_gauss':
            for n in (32, 64):
                arrays[f'gauss_{n}'] = quadrature_jet(case['vertices'], case['point'], 0, n)*float(case['sigma'])/(4*np.pi)
            controls = [arrays['gauss_32'], arrays['gauss_64']]
            metadata = dict(method='independent_Green_surface_Gauss', orders=[32, 64])
        elif method == 'closed_form_mp_and_reduced_integral':
            lo_i, hi_i = reduced_integral(case, 50), reduced_integral(case, 70)
            result['independent_integral_mp50'] = lo_i
            result['independent_integral_mp70'] = hi_i
            controls = [array(lo_i), array(hi_i)]
            arrays.update(integral_50=controls[0], integral_70=controls[1])
            metadata = dict(method='independent_reduced_Green_integral', dps=[50, 70])
        elif method == 'closed_form_mp_and_symmetric_trace':
            controls = []
            result['independent_trace_controls'] = []
            for height in ('1e-14', '1e-18'):
                above = reduced_integral(case, 70, height=height)
                below = reduced_integral(case, 70, height='-'+height)
                with mp.workdps(70):
                    average = [mp.nstr((mp.mpf(a)+mp.mpf(b))/2, 70) for a, b in zip(above, below)]
                controls.append(array(average))
                arrays['trace_'+height] = controls[-1]
                result['independent_trace_controls'].append(dict(height=height, above=above, below=below, average=average))
            metadata = dict(method='independent_symmetric_Green_trace', heights=['1e-14', '1e-18'], dps=70)
        else:
            raise ValueError('No frozen reference method for '+case['id'])
        error = scaled_max(controls[0], controls[1])
        ratios = [scalar_budget_ratio(c, value) for c in controls]
        result['refinement'] = dict(**metadata, scaled_max=error, threshold=1e-11,
                                    scalar_budget_ratios=ratios)
        result['qualified'] = refinement <= 1e-11 and error <= 1e-11 and max(ratios) <= 1
    arrays['reference'] = value
    result['coefficients'] = value.tolist()
    result['qualified'] = bool(result['qualified'] and np.isfinite(value).all())
    result['elapsed_seconds'] = time.monotonic()-start
    return result, arrays


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=PACKAGE/'corpus')
    parser.add_argument('--case', help='Generate one case into corpus/parts; combine with --collect.')
    parser.add_argument('--collect', action='store_true', help='Collect independently bounded case workers.')
    args = parser.parse_args()
    corpus = args.corpus
    cases_file = corpus/'cases.json'
    cases = json.loads(cases_file.read_text())['cases']
    frozen = json.loads((corpus/'freeze.json').read_text())
    assert sha(cases_file) == frozen['cases_sha256'], 'frozen case definitions changed'
    if args.case:
        cases = [c for c in cases if c['id'] == args.case]
        if not cases:
            parser.error('unknown case')
    results, arrays = [], {}
    parts = corpus/'parts'
    if args.case:
        parts.mkdir(exist_ok=True)
    for case in cases:
        if args.collect:
            payload = json.loads((parts/(case['id']+'.json')).read_text())
            assert payload['cases_sha256'] == sha(cases_file)
            assert payload['generator_sha256'] == sha(Path(__file__))
            result = payload['result']
            with np.load(parts/(case['id']+'.npz')) as archive:
                raw = {k: archive[k] for k in archive.files}
        else:
            result, raw = calculate(case)
        results.append(result)
        arrays.update({case['id']+'__'+k: v for k, v in raw.items()})
        print(json.dumps(dict(id=case['id'], qualified=result['qualified'],
                              seconds=result.get('elapsed_seconds', 0))), flush=True)
        if args.case:
            (parts/(case['id']+'.json')).write_text(json.dumps(dict(cases_sha256=sha(cases_file), generator_sha256=sha(Path(__file__)), result=result), indent=2, allow_nan=False)+'\n')
            np.savez_compressed(parts/(case['id']+'.npz'), **raw)
    if not args.case:
        report = dict(schema_version=1, cases_sha256=sha(cases_file),
                      generator_sha256=sha(Path(__file__)), provenance=frozen,
                      environment=dict(python=platform.python_version(), numpy=np.__version__, mpmath=mp.__version__),
                      all_qualified=all(r['qualified'] for r in results),
                      counts=dict(total=len(results), success=sum(r['status']==0 for r in results),
                                  errors=sum(r['status']!=0 for r in results)), results=results)
        np.savez_compressed(corpus/'reference.npz', **arrays)
        report['arrays_sha256'] = sha(corpus/'reference.npz')
        (corpus/'reference.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if not all(r['qualified'] for r in results):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
