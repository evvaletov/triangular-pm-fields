"""Compiled directional Taylor adapter for the repaired triangular K kernel.

This adapter is identified as jax_repair_20261103. It is not a stock
magpylib_jax API. Reconstruction directions, inverses, monomial index order,
and factorial convention follow the shared spatial Taylor contract: jet(..., factorial_scaled=False) supplies ordinary
univariate Taylor coefficients, and the homogeneous inverse is applied
without a further q! division.

Only observation coordinates carry polynomial increments. Vertices remain
runtime inputs of the compiled export. The local charts inside kernel.py
freeze branch signs at the expansion-center primal; that is a chart choice,
not a compiled-in geometry.

The directional Taylor contract is regular observation centers. The scalar
adapter evaluates the closed form as written and does not promise the
Python `field()` face-interior average-trace policy.
"""
from __future__ import annotations

import math

import numpy as np

from triangular_pm_fields.reference import indices
from .kernel import k_field, k_point


def reconstruction(order, count=64):
    """Fixed spherical directions; one linear inverse per homogeneous degree.

    Uses 64 deterministic directions and the shared graded monomial order.
    """
    k = np.arange(count, dtype=float)
    z = 1 - 2 * (k + .5) / count
    phi = k * np.pi * (3 - np.sqrt(5)) + .17
    directions = np.column_stack([
        np.sqrt(1 - z * z) * np.cos(phi),
        np.sqrt(1 - z * z) * np.sin(phi),
        z,
    ])
    inverses = []
    records = []
    for degree in range(1, order + 1):
        keys = np.asarray([a for a in indices(order) if sum(a) == degree])
        matrix = np.prod(directions[:, None, :] ** keys[None, :, :], axis=2)
        inverse = np.linalg.pinv(matrix, rcond=1e-14)
        error = float(np.max(np.abs(inverse @ matrix - np.eye(len(keys)))))
        condition = float(np.linalg.cond(matrix))
        if error > 1e-12 or condition > 1e4:
            raise ValueError(
                f'Unqualified reconstruction: degree {degree}, {error=}, {condition=}'
            )
        inverses.append(inverse)
        records.append({
            'degree': degree,
            'columns': len(keys),
            'condition_number': condition,
            'monomial_identity_error': error,
        })
    return directions, inverses, records


def polynomial_control(vertices, point):
    import jax.numpy as jnp
    x, y, z = point
    return jnp.stack([jnp.exp(x + 2 * y - .5 * z), x * x * y * y * y * z, x + y + z * z])


def control_reference(order):
    result = np.zeros((3, len(indices(order))))
    for i, alpha in enumerate(indices(order)):
        result[0, i] = math.prod(base ** a / math.factorial(a)
                                 for base, a in zip([1, 2, -.5], alpha))
        result[1, i] = float(alpha == (2, 3, 1))
        result[2, i] = float(alpha in [(1, 0, 0), (0, 1, 0), (0, 0, 2)])
    return result


def directional_coefficients(order, function=None):
    """Complete spatial coefficients from normalized directional Taylor jets.

    Returns (evaluate, reconstruction_records). evaluate(vertices, point)
    has shape (3, C(order+3, 3)) in the documented monomial order.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental.jet import jet
    function = function or k_point
    directions, inverses, records = reconstruction(order)
    directions = jnp.asarray(directions)
    inverses = [jnp.asarray(a) for a in inverses]

    def evaluate(vertices, point):
        def one(direction):
            series = (direction,) + (jnp.zeros_like(point),) * (order - 1)
            value, terms = jet(
                lambda p: function(vertices, p),
                (point,),
                (series,),
                factorial_scaled=False,
            )
            return value, jnp.stack(terms, axis=0)
        values, terms = jax.vmap(one)(directions)
        columns = [values[0, :, None]]
        for degree, inverse in enumerate(inverses, 1):
            columns.append((inverse @ terms[:, degree - 1, :]).T)
        return jnp.concatenate(columns, axis=1)

    return evaluate, records


_COMPILED_DIRECTIONAL = {}
_COMPILED_SCALAR = None


def compiled_directional_coefficients(order, function=None):
    """Outer-jit export. Vertices and observer remain runtime arguments.

    The compiled module is cached by (order, kernel identity). Geometry is
    not part of the cache key and remains a runtime argument.
    """
    import jax
    function = function or k_point
    key = (order, function)
    if key not in _COMPILED_DIRECTIONAL:
        evaluate, records = directional_coefficients(order, function=function)
        _COMPILED_DIRECTIONAL[key] = (jax.jit(evaluate), records)
    return _COMPILED_DIRECTIONAL[key]


def compiled_scalar():
    """Outer-jit batched scalar K. Vertices and observers remain runtime.

    Regular centers are the supported contract. Face-interior, edge, and
    vertex samples are recorded; this export does not promise a
    boundary-average policy.
    """
    import jax
    global _COMPILED_SCALAR
    if _COMPILED_SCALAR is None:
        _COMPILED_SCALAR = jax.jit(k_field)
    return _COMPILED_SCALAR


def jacfwd_coefficients(order, function=None):
    """Unique mixed coefficients from composed Jacobians, divided by alpha!."""
    import jax
    import jax.numpy as jnp
    function = function or k_point
    derivatives = [function]
    for _ in range(order):
        derivatives.append(jax.jacfwd(derivatives[-1], argnums=1))

    def evaluate(vertices, point):
        tensors = [f(vertices, point) for f in derivatives]
        columns = []
        for alpha in indices(order):
            axes = tuple(axis for axis, count in enumerate(alpha) for _ in range(count))
            value = tensors[sum(alpha)][(slice(None),) + axes]
            columns.append(value / math.prod(math.factorial(a) for a in alpha))
        return jnp.stack(columns, axis=1)

    return evaluate
