"""Serialized Python client for the process-global native TAN cache.

Each call loads a cache and evaluates the complete requested batch under one
module-wide lock. Do not call the same native library through another binding
concurrently. Separate processes provide independent native contexts.
"""
from __future__ import annotations

import ctypes as ct
import math
import operator
import os
from pathlib import Path
import threading

import numpy as np

_LOCK = threading.RLock()


class NativeError(RuntimeError):
    def __init__(self, status, operation, index=None):
        self.status = int(status)
        self.operation = operation
        self.index = index
        super().__init__(f'{operation}: native status {status}' +
                         (f' at point {index}' if index is not None else ''))


def exponents(order):
    """Graded x-ascending/y-ascending monomial triples through order 6."""
    try:
        q = operator.index(order)
    except TypeError as exc:
        raise ValueError('order must be an integer from 0 through 6') from exc
    if isinstance(order, (bool, np.bool_)) or not 0 <= q <= 6:
        raise ValueError('order must be an integer from 0 through 6')
    return [(x, y, d-x-y) for d in range(q+1)
            for x in range(d+1) for y in range(d-x+1)]


class NativeLibrary:
    """Load an explicitly selected libtan shared library.

    evaluate returns shape (points, 3, monomials), with coefficients of
    mu0 H(point + radius*u). No cache remains associated with this object.
    Every call reloads its named geometry; native kernel timings should use
    the C or Fortran API when excluding this loading overhead.
    """
    def __init__(self, library):
        self.path = Path(library).resolve(strict=True)
        self._lib = ct.CDLL(str(self.path))
        ptr = ct.POINTER(ct.c_double)
        self._lib.tan_load_c.argtypes = [ct.c_char_p, ct.POINTER(ct.c_int)]
        self._lib.tan_load_c.restype = None
        self._lib.tan_eval_c.argtypes = [ptr, ct.c_int, ct.c_int, ct.c_double,
                                       ptr, ct.POINTER(ct.c_int)]
        self._lib.tan_eval_c.restype = None

    def evaluate(self, cache, points, *, mode=1, order=0, radius=1.0):
        powers = exponents(order)
        try:
            mode_i = operator.index(mode)
        except TypeError as exc:
            raise ValueError('mode must be integer 0, 1, or 2') from exc
        if isinstance(mode, (bool, np.bool_)) or mode_i not in (0, 1, 2):
            raise ValueError('mode must be integer 0, 1, or 2')
        if order and mode_i != 1:
            raise ValueError('Taylor coefficients require mode=1')
        radius = float(radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError('radius must be positive and finite')
        xyz = np.asarray(points, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('points must be a finite array of shape (n,3)')
        xyz = np.ascontiguousarray(xyz)
        path = os.fsencode(Path(cache).absolute())
        if b'\x00' in path:
            raise ValueError('cache path contains a NUL byte')
        result = np.empty((len(xyz), 3, len(powers)), dtype=np.float64)
        ptr = ct.POINTER(ct.c_double)
        with _LOCK:
            status = ct.c_int()
            self._lib.tan_load_c(path, ct.byref(status))
            if status.value:
                raise NativeError(status.value, 'load')
            for i, point in enumerate(xyz):
                coefficients = np.zeros((3, 84), dtype=np.float64, order='F')
                self._lib.tan_eval_c(point.ctypes.data_as(ptr), mode_i,
                                     int(order), radius,
                                     coefficients.ctypes.data_as(ptr), ct.byref(status))
                if status.value:
                    raise NativeError(status.value, 'evaluate', i)
                result[i] = coefficients[:, :len(powers)]
        return result
