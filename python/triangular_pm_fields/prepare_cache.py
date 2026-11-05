"""Exact binary64 geometry preparation for the compiled boundary evaluator.

This is setup code, never the numerical query backend. Fractions decide plane
identity, splitting, orientation, and cancellation. Primitive integer normals are serialized exactly. Native normalization rounds
only the numerical geometry; GMP retains the exact predicates. No tolerance merges geometry.

Identical signed directed raw edges are netted inside each exact plane before
subdivision. Every input vertex from a nonzero-charge face is retained, including
vertices of canceled raw edges, so T-junctions still split surviving segments.
Candidates on a remaining raw edge are the plane vertices inside that edge's
axis-aligned bounding box in original binary64 coordinates, then exact
cross-product collinearity. No supporting-line index and no approximate predicate.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import numpy as np

# Bound at import to the producer actually loaded into this process. Disk edits
# after import do not relabel this fingerprint; cache_store rejects that drift.
_LOADED_SOURCE_PATH = Path(__file__).resolve()
_LOADED_SOURCE_BYTES = _LOADED_SOURCE_PATH.read_bytes()
_LOADED_SOURCE_SHA256 = hashlib.sha256(_LOADED_SOURCE_BYTES).hexdigest()


def _source_fingerprint():
    """Private SHA-256 of the producer source loaded into this process."""
    return _LOADED_SOURCE_SHA256


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def qpoint(p):
    return tuple(F(float(x)) for x in p)


def on_segment(p, a, b):
    d, r = sub(b, a), sub(p, a)
    return not any(cross(d, r)) and 0 <= dot(r, d) <= dot(d, d)


def decimal(q):
    # All finite binary64 coordinates and their sums terminate in <=1074
    # fractional digits. 2000 significant digits also cover their integer part.
    with localcontext() as context:
        if q.denominator & (q.denominator-1):
            raise ValueError("cache records must be integer or dyadic values")
        context.prec = 2000
        return str(Decimal(q.numerator)/Decimal(q.denominator))


def _cached_qpoint(p, memo):
    key = (float(p[0]), float(p[1]), float(p[2]))
    cached = memo.get(key)
    if cached is None:
        cached = (F(key[0]), F(key[1]), F(key[2]))
        memo[key] = cached
    return cached


def _cached_charge(charge, memo):
    key = float(charge)
    cached = memo.get(key)
    if cached is None:
        cached = F(key)
        memo[key] = cached
    return cached


def _net_directed_edges(faces):
    directed = defaultdict(F)
    for vertices, charge, orientation in faces:
        weight = charge * orientation
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            directed[a, b] += weight
    remaining = {}
    consumed = set()
    for (a, b), weight in directed.items():
        if (a, b) in consumed:
            continue
        net = weight - directed.get((b, a), F(0))
        consumed.add((a, b))
        consumed.add((b, a))
        if net:
            remaining[a, b] = net
    return remaining


class PreparedCache:
    def __init__(self, triangles, charges, *, coalesce=True):
        tri = np.array(triangles, dtype=np.float64, copy=True)
        sigma = np.array(charges, dtype=np.float64, copy=True)
        if tri.size == 0:
            tri = np.empty((0, 3, 3))
        if tri.ndim != 3 or tri.shape[1:] != (3, 3) or sigma.shape != (len(tri),):
            raise ValueError('expected triangles(m,3,3), charges(m)')
        if not np.isfinite(tri).all() or not np.isfinite(sigma).all():
            raise ValueError('nonfinite input')
        vertex_memo = {}
        charge_memo = {}
        groups = {}
        for vertices, charge in zip(tri, sigma):
            v = tuple(_cached_qpoint(p, vertex_memo) for p in vertices)
            n = cross(sub(v[1], v[0]), sub(v[2], v[0]))
            pivot = next((x for x in n if x), None)
            if pivot is None:
                raise ValueError('degenerate triangle')
            if not charge:
                continue
            normal = tuple(x/pivot for x in n)
            key = (normal, dot(normal, v[0]))
            group = groups.setdefault(key, {'point': v[0], 'faces': [], 'vertices': set()})
            group['faces'].append((v, _cached_charge(charge, charge_memo),
                                   1 if pivot > 0 else -1))
            group['vertices'].update(v)
        self.planes = []
        self.edges = []
        for key, group in sorted(groups.items()):
            vertices = sorted(group['vertices'])
            vf = np.array([[float(x) for x in p] for p in vertices])
            edges = defaultdict(F)
            for (a, b), weight in _net_directed_edges(group['faces']).items():
                direction = sub(b, a)
                axis = next(i for i,x in enumerate(direction) if x)
                af, bf = np.array(a, dtype=float), np.array(b, dtype=float)
                candidates = np.flatnonzero(np.all((vf >= np.minimum(af, bf)) &
                                                  (vf <= np.maximum(af, bf)), axis=1))
                split = [vertices[i] for i in candidates
                         if not any(cross(direction, sub(vertices[i], a)))]
                split.sort(key=lambda p: p[axis], reverse=direction[axis]<0)
                for x, y in zip(split, split[1:]):
                    edge = (x, y) if x < y else (y, x)
                    edges[edge] += weight * (1 if x < y else -1)
            edges = {e: w for e, w in edges.items() if w}
            if coalesce:
                edges = self._coalesce(edges)
            if not edges:
                continue
            # An exact closed weighted chain is a construction invariant.
            balance = defaultdict(F)
            for (a, b), w in edges.items():
                balance[a] += w
                balance[b] -= w
            if any(balance.values()):
                raise ValueError('open weighted boundary chain')
            i = len(self.planes)
            self.planes.append((key, group['point']))
            self.edges.extend((i, a, b, w) for (a, b), w in sorted(edges.items()))
        self.triangles, self.charges = tri, sigma
        self.text = self._text()
        self.sha256 = hashlib.sha256(self.text.encode()).hexdigest()

    @staticmethod
    def _coalesce(edges):
        """Join adjacent collinear equal-density edges without changing a chain."""
        edges = dict(edges)
        while True:
            adjacency = defaultdict(list)
            for edge in edges:
                for v in edge:
                    adjacency[v].append(edge)
            changed = False
            for v, incident in adjacency.items():
                if len(incident) != 2:
                    continue
                e, f = incident
                a = e[0] if e[1] == v else e[1]
                b = f[0] if f[1] == v else f[1]
                if any(cross(sub(a, v), sub(b, v))) or dot(sub(a, v), sub(b, v)) >= 0:
                    continue
                incoming = edges[e] if e == (a, v) else -edges[e]
                outgoing = edges[f] if f == (v, b) else -edges[f]
                if incoming != outgoing:
                    continue
                del edges[e], edges[f]
                new = (a, b) if a < b else (b, a)
                edges[new] = edges.get(new, F(0)) + incoming*(1 if a < b else -1)
                if not edges[new]:
                    del edges[new]
                changed = True
                break
            if not changed:
                return edges

    def _text(self):
        lines = ['TAN_CACHE_V1', f'{len(self.planes)} {len(self.edges)}']
        for (normal, _), point in self.planes:
            denominator = math.lcm(*(x.denominator for x in normal))
            integer = [x.numerator*(denominator//x.denominator) for x in normal]
            divisor = math.gcd(*integer)
            exact_normal = tuple(F(x//divisor) for x in integer)
            lines.append(' '.join(decimal(x) for x in (*point, *exact_normal)))
        for i, a, b, weight in self.edges:
            lines.append(str(i+1)+' '+' '.join(decimal(x) for x in (*a, *b, weight)))
        return '\n'.join(lines)+'\n'

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text)
        return path

    def classify(self, point):
        """Exact regular/sheet/physical-edge classification of binary64 points.

        Winding works for signed overlaps, holes and canceled local support;
        triangle-union membership would incorrectly reject canceled interiors.
        """
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('point must contain three finite values')
        p = qpoint(point)
        for _, a, b, _ in self.edges:
            if on_segment(p, a, b):
                return 'edge'
        for i, ((normal, offset), _) in enumerate(self.planes):
            if dot(normal, p) != offset:
                continue
            axes = [j for j in range(3) if j != max(range(3), key=lambda k: abs(normal[k]))]
            x, y = axes
            winding = F(0)
            for plane, a, b, weight in self.edges:
                if plane != i:
                    continue
                orient = (b[x]-a[x])*(p[y]-a[y])-(b[y]-a[y])*(p[x]-a[x])
                if a[y] <= p[y] < b[y] and orient > 0:
                    winding += weight
                elif b[y] <= p[y] < a[y] and orient < 0:
                    winding -= weight
            if winding:
                return 'sheet'
        return 'regular'

    def metadata(self):
        return dict(input_triangles=len(self.triangles), planes=len(self.planes),
                    boundary_edges=len(self.edges), sha256=self.sha256,
                    exact_setup=True, query_geometry_precision='real128 or binary64')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSON with triangles and charges')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    cache = PreparedCache(data['triangles'], data['charges'])
    cache.write(args.output)
    print(json.dumps(cache.metadata()))


if __name__ == '__main__':
    main()
