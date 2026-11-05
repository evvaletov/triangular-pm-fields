"""Repaired pure-JAX kernel for the triangular-sheet integral K.

This is an explicit adapter kernel, not a magpylib_jax export and not a
monkeypatch of installed JAX primitives. Stock magpylib_jax
6290916635693fb04a099205b48f845a396dffc0 remains the unmodified baseline.

K(p) = integral_T (p-s)/|p-s|^3 dS. Length units cancel. Polarization
parallel to the unit normal converts library B to K by 4*pi/sigma.

The supported contract is regular observation centers: off the source
triangle, including the exterior plane and the infinite edge-line
extensions away from the finite segments. The scalar evaluation of this
closed form does not promise a boundary-average policy.

Charts, chosen to stay inside JAX (no Python extraction of tracer values):

1. Solid angle. Installed jax.experimental.jet `_atan2_taylor` (JAX 0.10.1
   `jax/experimental/jet.py`) binds `lax.atan2` from `jnp.arctan2(y, x)`
   and first forms the numpy-order ratio y/x by
   `jet2(lax.div, primals_in, series_in)`. That ratio is ill-conditioned
   when the second argument x approaches 0 with y nonzero. The local
   identity arg(z) = arg(z0) + arg(z/z0) is used instead. z0 is the
   expansion-center primal via lax.stop_gradient (COSY CONS). At the
   center, z/z0 = 1, so the increment chart is atan2(imag, real) about
   (0, 1), where the installed rule evaluates y/x = 0/1.

2. Edge logarithm. The algebraic identity
     R_i + R_j - L = (|u|+|L-u|-L) + rho^2/(R_i+|u|) + rho^2/(R_j+|L-u|)
   is evaluated with signs frozen at the expansion-center primal. The
   logarithm is log1p(2 L / gap). This is the same neighborhood chart as
   triangular_pm_fields.reference.field_jet.

3. Geometry. Unit normal and edge lengths use ordinary sqrt(sum squares).
   The stock magpylib_jax path uses _safe_norm, a custom_jvp whose jet
   unwrap leaked JetTracers under an outer jit, including with explicitly
   zero geometry increments.

Source-set distinctions, kept separate:

- Source-face interior is a jump surface. The two traces differ by the
  solid-angle jump; there is no unique ordinary Taylor jet that crosses
  the face. Python `field()` assigns the average of the traces (Omega=0)
  as a scalar convention only. This kernel evaluates the closed form as
  written and does not implement that average.
- Exact edge and vertex are physical singularities of K (logarithmic
  tangential field). They are not chart failures.
- D=0 with N nonzero is a regular atan2 chart test. Exterior coplanar
  observers, where N=0 and D>0, are another class of regular centers.

Singularities are not replaced by nan_to_num. Geometry and observer are
runtime arguments. Taylor charts freeze only the local analytic branch at
the current primal; they do not bake vertices or the observer into the
compiled module. A setup-fixed specialization of geometry is not used for
the default compiled export.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def euclid(vector, axis=-1, keepdims=False):
    """Ordinary Euclidean norm. Avoids custom_jvp helpers such as _safe_norm."""
    return jnp.sqrt(jnp.sum(vector * vector, axis=axis, keepdims=keepdims))


def atan2_recentered(y, x):
    """arg(x + i y) with a Taylor chart that stays regular when x0 = 0, y0 != 0.

    Constant term: atan2(y0, x0) of the expansion-center primal.
    Increments: atan2 of z/z0, whose primal is (real, imag) = (1, 0).
    Under a pure scalar evaluation stop_gradient is the identity, so the
    value equals atan2(y, x).
    """
    y0 = jax.lax.stop_gradient(y)
    x0 = jax.lax.stop_gradient(x)
    radius0 = x0 * x0 + y0 * y0
    real = (x * x0 + y * y0) / radius0
    imag = (y * x0 - x * y0) / radius0
    return jnp.arctan2(y0, x0) + jnp.arctan2(imag, real)


def k_point(vertices, point):
    """K at one observer. vertices (3, 3), point (3,) -> (3,).

    Internal length scaling is homogeneous: K is dimensionless, and physical
    observation increments reach the scaled chart through the chain rule.
    """
    vertices = jnp.asarray(vertices)
    point = jnp.asarray(point)
    edges = jnp.roll(vertices, -1, axis=0) - vertices
    scale = jnp.max(euclid(edges, axis=1))
    origin = vertices[0]
    v = (vertices - origin) / scale
    p = (point - origin) / scale
    cross_ab = jnp.cross(v[1] - v[0], v[2] - v[0])
    area2 = euclid(cross_ab)
    normal = cross_ab / area2
    radii_vectors = v - p
    radii = euclid(radii_vectors, axis=1)
    height = jnp.dot(p, normal)
    numerator = -height * area2
    denominator = (
        radii[0] * radii[1] * radii[2]
        + jnp.dot(radii_vectors[0], radii_vectors[1]) * radii[2]
        + jnp.dot(radii_vectors[1], radii_vectors[2]) * radii[0]
        + jnp.dot(radii_vectors[2], radii_vectors[0]) * radii[1]
    )
    omega = 2.0 * atan2_recentered(numerator, denominator)
    result = -omega * normal
    for i in range(3):
        j = (i + 1) % 3
        edge = v[j] - v[i]
        length = euclid(edge)
        tangent = edge / length
        outward = jnp.cross(tangent, normal)
        u = jnp.dot(p - v[i], tangent)
        w = jnp.dot(v[j] - p, tangent)
        u0 = jax.lax.stop_gradient(u)
        w0 = jax.lax.stop_gradient(w)
        offset = jnp.where(jnp.abs(u0) <= jnp.abs(w0), p - v[i], p - v[j])
        rho2 = jnp.sum(jnp.cross(offset, tangent) ** 2)
        sign_u = jnp.where(u0 >= 0, 1.0, -1.0)
        sign_w = jnp.where(w0 >= 0, 1.0, -1.0)
        gap = (
            (sign_u - 1.0) * u
            + (sign_w - 1.0) * w
            + rho2 * (1.0 / (radii[i] + sign_u * u) + 1.0 / (radii[j] + sign_w * w))
        )
        logterm = jnp.log1p(2.0 * length / gap)
        result = result + outward * logterm
    return result


def k_field(vertices, points):
    """K at many observers. vertices (3, 3), points (n, 3) -> (n, 3)."""
    return jax.vmap(lambda point: k_point(vertices, point))(points)
