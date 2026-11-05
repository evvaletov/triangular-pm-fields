"""Triangle surface integral K(p) = integral_T (p-s)/|p-s|^3 dS.

Length units cancel. Multiply K by (J dot n)/(4*pi) for the contribution
to mu0*H from a face of a uniformly polarized magnet. Add J inside the body
to obtain B. This is a standalone article implementation, not COSYFFAG code.
"""
from functools import lru_cache
import numpy as np


def geometry(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 3) or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a finite 3 by 3 array")
    lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    scale = float(max(lengths))
    if scale == 0:
        raise ValueError("zero-area triangle")
    v = (vertices - vertices[0]) / scale
    cross = np.cross(v[1], v[2])
    area2 = float(np.linalg.norm(cross))
    if area2 == 0:
        raise ValueError("zero-area triangle")
    return v, cross / area2, scale, area2


def field(vertices, point, stable=True):
    """Return K. Exact face-interior values use the average of the two traces.

    Exact edges and vertices are singular and raise ValueError. Stabilization
    removes the edge denominator subtraction; far-field cancellation remains.
    """
    v, n, scale, area2 = geometry(vertices)
    p = (np.asarray(point, dtype=float) - np.asarray(vertices)[0]) / scale
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("point must have three finite coordinates")
    r = v - p
    radii = np.linalg.norm(r, axis=1)
    if min(radii) == 0:
        raise ValueError("field is singular at a triangle vertex")
    h = float(p @ n)
    numerator = -h * area2
    denominator = (np.prod(radii) + (r[0] @ r[1]) * radii[2]
                   + (r[1] @ r[2]) * radii[0] + (r[2] @ r[0]) * radii[1])
    omega = 0.0 if h == 0.0 else 2 * np.arctan2(numerator, denominator)
    result = -omega * n
    for i in range(3):
        j = (i + 1) % 3
        edge = v[j] - v[i]
        length = float(np.linalg.norm(edge))
        tangent = edge / length
        outward = np.cross(tangent, n)
        u = float((p - v[i]) @ tangent)
        w = float((v[j] - p) @ tangent)
        offset = p - v[i] if abs(u) <= abs(w) else p - v[j]
        rho = float(np.linalg.norm(np.cross(offset, tangent)))
        if rho == 0.0 and u >= 0.0 and w >= 0.0:
            raise ValueError("field is singular on a triangle edge")
        if stable:
            # R_i+R_j-l = (|u|+|l-u|-l) + rho^2/(R_i+|u|)
            #                                      + rho^2/(R_j+|l-u|).
            # Compute the second projection from its own endpoint. Using l-u
            # here loses small distances to a vertex through rounding.
            baseline = 2 * (max(-u, 0.0) + max(-w, 0.0))
            correction = rho * (rho / (radii[i] + abs(u))
                                + rho / (radii[j] + abs(w)))
            gap = baseline + correction
            if gap > 0:
                logterm = np.logaddexp(0, np.log(2 * length) - np.log(gap))
            else:
                # Interior projection; protect against rho^2 underflow.
                loggap = 2 * np.log(rho) + np.log(1 / (radii[i] + abs(u))
                                                + 1 / (radii[j] + abs(w)))
                logterm = np.logaddexp(0, np.log(2 * length) - loggap)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                logterm = np.log((radii[i] + radii[j] + length)
                                 / (radii[i] + radii[j] - length))
        result = result + outward * logterm
    return result


def quadrature(vertices, point, order=64):
    """Independent Gauss-Legendre integration after a square-to-triangle map."""
    vertices = np.asarray(vertices, dtype=float)
    nodes, weights = quadrature_nodes(vertices, order)
    d = np.asarray(point) - nodes
    return np.sum(weights[:, None] * d / np.linalg.norm(d, axis=1)[:, None] ** 3, axis=0)


def quadrature_nodes(vertices, order):
    t, w = np.polynomial.legendre.leggauss(order)
    t, w = (t + 1) / 2, w / 2
    u, v = np.meshgrid(t, t, indexing="ij")
    a, b = vertices[1] - vertices[0], vertices[2] - vertices[0]
    nodes = vertices[0] + u[..., None] * a + ((1 - u) * v)[..., None] * b
    weights = w[:, None] * w[None, :] * (1 - u) * np.linalg.norm(np.cross(a, b))
    return nodes.reshape(-1, 3), weights.ravel()


def field_mp(vertices, point, dps=80):
    """High-precision evaluation of the same closed form (not an independent oracle)."""
    import mpmath as mp
    with mp.workdps(dps):
        v = [list(map(lambda x: mp.mpf(float(x)), row)) for row in vertices]
        p = [mp.mpf(float(x)) for x in point]
        dot = lambda a, b: sum(x * y for x, y in zip(a, b))
        cross = lambda a, b: [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
        sub = lambda a, b: [x-y for x, y in zip(a, b)]
        norm = lambda a: mp.sqrt(dot(a, a))
        n = cross(sub(v[1], v[0]), sub(v[2], v[0]))
        area2 = norm(n)
        n = [x/area2 for x in n]
        r = [sub(a, p) for a in v]
        radii = [norm(a) for a in r]
        h = dot(sub(p, v[0]), n)
        num = -h * area2
        den = mp.fprod(radii) + sum(dot(r[i], r[(i+1)%3])*radii[(i+2)%3] for i in range(3))
        omega = 0 if h == 0 else 2*mp.atan2(num, den)
        result = [-omega*x for x in n]
        for i in range(3):
            j = (i+1)%3
            e = sub(v[j], v[i]); length = norm(e)
            m = cross([x/length for x in e], n)
            ll = mp.log((radii[i]+radii[j]+length)/(radii[i]+radii[j]-length))
            result = [x+y*ll for x,y in zip(result,m)]
        return np.array([float(x) for x in result])


@lru_cache(None)
def indices(order):
    return tuple((a, b, degree-a-b) for degree in range(order+1)
                 for a in range(degree+1) for b in range(degree-a+1))


class Jet:
    """Truncated polynomial in three observation-coordinate increments.

    Coefficients are Taylor coefficients (derivatives divided by multi-factorial).
    Geometry and branch decisions are ordinary constants.
    """
    def __init__(self, value=0.0, order=6, coefficients=None):
        self.order = order
        self.c = dict(coefficients) if coefficients is not None else {(0, 0, 0): complex(value)}

    @property
    def constant(self):
        return self.c.get((0, 0, 0), 0j)

    @classmethod
    def variable(cls, value, axis, order):
        c = {(0, 0, 0): complex(value)}
        key = tuple(int(i == axis) for i in range(3))
        c[key] = 1.0
        return cls(order=order, coefficients=c)

    def coerce(self, other):
        return other if isinstance(other, Jet) else Jet(other, self.order)

    def __add__(self, other):
        other = self.coerce(other)
        c = self.c.copy()
        for k, v in other.c.items():
            c[k] = c.get(k, 0) + v
        return Jet(order=self.order, coefficients=c)

    __radd__ = __add__

    def __neg__(self):
        return Jet(order=self.order, coefficients={k: -v for k, v in self.c.items()})

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) + -self

    def __mul__(self, other):
        other = self.coerce(other)
        c = {}
        for a, x in self.c.items():
            if x == 0:
                continue
            for b, y in other.c.items():
                k = tuple(ai+bi for ai, bi in zip(a,b))
                if sum(k) <= self.order:
                    c[k] = c.get(k, 0) + x*y
        return Jet(order=self.order, coefficients=c)

    __rmul__ = __mul__

    def power(self, exponent):
        a = self.constant
        if a == 0:
            raise ValueError("Taylor expansion requires nonzero constant")
        u = (self - a) * (1/a)
        total, term, coeff = Jet(1, self.order), Jet(1, self.order), 1.0
        for k in range(1, self.order+1):
            term = term * u
            coeff *= (exponent-k+1)/k
            total = total + coeff*term
        return a**exponent * total

    def __truediv__(self, other):
        if not isinstance(other, Jet):
            return self * (1/other)
        return self * other.power(-1)

    def __rtruediv__(self, other):
        return self.power(-1) * other

    def log(self):
        a = self.constant
        if a == 0:
            raise ValueError("log Taylor expansion requires nonzero constant")
        u = (self-a)/a
        total, term = Jet(np.log(a), self.order), Jet(1, self.order)
        for k in range(1, self.order+1):
            term = term*u
            total = total + ((-1)**(k+1)/k)*term
        return total

    def imag(self):
        return Jet(order=self.order, coefficients={k: complex(v.imag) for k,v in self.c.items()})

    def evaluate(self, displacement, degree=None):
        degree = self.order if degree is None else degree
        return sum(value * np.prod(np.asarray(displacement)**np.array(k))
                   for k, value in self.c.items() if sum(k) <= degree).real

    def array(self):
        return np.array([self.c.get(k, 0).real for k in indices(self.order)])


def field_jet(vertices, point, order=6):
    """Local multivariate Taylor map; do not center a map on the source triangle."""
    v, n, scale, area2 = geometry(vertices)
    p0 = (np.asarray(point)-np.asarray(vertices)[0])/scale
    # Variables represent physical coordinate increments, not normalized ones.
    p = [Jet.variable(p0[i], i, order)/1.0 for i in range(3)]
    if scale != 1:
        p = [Jet(p0[i], order)+(p[i]-p0[i])/scale for i in range(3)]
    dot = lambda a,b: sum(x*y for x,y in zip(a,b))
    cross = lambda a,b: [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    r = [[v[i,j]-p[j] for j in range(3)] for i in range(3)]
    radii = [dot(x,x).power(0.5) for x in r]
    h = dot(p,n)
    num = -h*area2
    den = radii[0]*radii[1]*radii[2] + sum(dot(r[i],r[(i+1)%3])*radii[(i+2)%3] for i in range(3))
    if h.constant.real == 0 and den.constant.real <= 0:
        raise ValueError("Taylor center lies on the source triangle")
    z = den + 1j*num
    omega0 = 2*np.arctan2(num.constant.real,den.constant.real)
    # Recenter the complex logarithm at 1. Its constant is exactly zero;
    # all local derivatives are independent of atan2's global branch cut.
    omega = omega0 + 2*(1+(z-z.constant)/z.constant).log().imag()
    result = [-omega*x for x in n]
    for i in range(3):
        j=(i+1)%3
        e=v[j]-v[i]; length=np.linalg.norm(e); e=e/length
        m=np.cross(e,n)
        u=dot([p[k]-v[i,k] for k in range(3)],e)
        w=dot([v[j,k]-p[k] for k in range(3)],e)
        endpoint=i if abs(u.constant)<=abs(w.constant) else j
        rho_vec=cross([p[k]-v[endpoint,k] for k in range(3)],e)
        rho2=dot(rho_vec,rho_vec)
        # Choose a constant chart for |u| and |length-u|. These expressions
        # algebraically equal R_i+R_j-length within the selected neighborhood.
        su=1 if u.constant.real>=0 else -1
        sw=1 if w.constant.real>=0 else -1
        gap=(su-1)*u+(sw-1)*w+rho2*(1/(radii[i]+su*u)+1/(radii[j]+sw*w))
        ll=(1+2*length/gap).log()
        result=[x+mi*ll for x,mi in zip(result,m)]
    return result


def evaluate_map(coefficients, displacement, order=6):
    """Evaluate all three components with a shared array of monomials."""
    exponents=np.asarray(indices(order))
    monomials=np.prod(np.asarray(displacement)[None,:]**exponents,axis=1)
    return np.asarray(coefficients)@monomials


def quadrature_jet(vertices, point, order=6, quadrature_order=64):
    """Independent coefficient oracle: expand the Green kernel under the integral.

    Uses a binomial expansion of (r^2+2*d.delta+delta^2)^(-3/2),
    not the edge/solid-angle formula or the Jet implementation.
    """
    vertices=np.asarray(vertices,float)
    nodes,weights=quadrature_nodes(vertices,quadrature_order)
    d=np.asarray(point)-nodes
    r2=np.sum(d*d,axis=1)
    keys=indices(order); lookup={k:i for i,k in enumerate(keys)}
    q=[]
    for axis in range(3):
        key=tuple(int(i==axis) for i in range(3))
        q.append((key,2*d[:,axis]/r2))
        q.append((tuple(2*x for x in key),1/r2))
    power=np.zeros((len(keys),len(nodes))); power[0]=1
    series=power.copy(); coefficient=1.0
    for k in range(1,order+1):
        new=np.zeros_like(power)
        for i,a in enumerate(keys):
            for b,values in q:
                target=tuple(x+y for x,y in zip(a,b))
                if sum(target)<=order:
                    new[lookup[target]]+=power[i]*values
        power=new
        coefficient*=(-1.5-k+1)/k
        series+=coefficient*power
    result=np.zeros((3,len(keys)))
    base_weights=weights/r2**1.5
    for axis in range(3):
        c=series*d[:,axis]
        for i,key in enumerate(keys):
            if key[axis]>0:
                previous=list(key); previous[axis]-=1
                c[i]+=series[lookup[tuple(previous)]]
        result[axis]=c@base_weights
    return result


def body_field(vertices, faces, polarization, point):
    """Return the surface-charge contribution to mu0 H; outward faces required."""
    result=np.zeros(3)
    for face in faces:
        triangle=np.asarray(vertices)[list(face)]
        _,normal,_,_=geometry(triangle)
        charge=np.asarray(polarization)@normal
        if charge != 0:
            result+=charge*field(triangle,point)/(4*np.pi)
    return result
