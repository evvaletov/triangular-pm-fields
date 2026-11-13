"""Linear fitting with a fixed covariance across training and held-out data.

The measurement model is

    y_t = A_t theta + e_t
    y_h = A_h theta + e_h

with a caller-supplied dense symmetric positive-definite joint covariance C
for [e_t, e_h] in that row order. Geometry and calibration enter through the
fixed design and covariance supplied by the caller.

The GLS estimator of theta is formed from A_t and the training block C_tt
only: Cholesky-whiten A_t, column-scale, SVD. Holdout values never enter the
estimator. The unconditional holdout residual covariance is computed from the
joint factor C = L L^T as C_r = F F^T with F = [-K, I] L and K = A_h H.

Dense-memory limitation
-----------------------
C, its Cholesky factor, and the holdout factor F are dense. For n = n_t + n_h,
one covariance array occupies 8 n^2 bytes. Construction keeps several arrays
of this size and temporary copies, as well as an n_h by n factor. There is no
sparse or out-of-core path. The intended use
is small synthetic designs and modest assembled observation lists, not a
full-field scan map.

This is statistical software with synthetic checks. It is not experimental
validation of a physical PM scan and it does not calibrate instruments.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np

__all__ = [
    'CorrelatedLinearFit',
    'CorrelatedLinearFitResult',
    'RankDeficientError',
]

_REAL_KINDS = frozenset('iuf')
_EPS = np.finfo(np.float64).eps
_TINY = np.finfo(np.float64).tiny


class RankDeficientError(ValueError):
    """Whitened, column-scaled training design is rank-deficient."""

    def __init__(self, message, *, singular_values=None, rank=None, threshold=None):
        super().__init__(message)
        self.singular_values = singular_values
        self.rank = rank
        self.threshold = threshold


def _as_real_float64(value, name, *, ndim, shape=None):
    """Copy a real finite array to C-order float64. Complex inputs are rejected."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f'{name} must be a real array') from exc
    if array.dtype.kind == 'c' or np.iscomplexobj(array):
        raise TypeError(f'{name} must be real; complex inputs are rejected')
    if array.dtype.kind not in _REAL_KINDS:
        raise TypeError(f'{name} must be a real numeric array, got dtype {array.dtype}')
    if array.ndim != ndim:
        raise ValueError(f'{name} must have ndim {ndim}, got {array.ndim}')
    out = np.array(array, dtype=np.float64, copy=True, order='C')
    if shape is not None and out.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {out.shape}')
    if not np.isfinite(out).all():
        raise ValueError(f'{name} must be finite')
    return out


@dataclass(frozen=True, slots=True)
class _ArraySnapshot:
    data: bytes
    shape: tuple[int, ...]

    def view(self):
        return np.frombuffer(self.data, dtype=np.float64).reshape(self.shape)


def _frozen_array(value):
    """Keep both the values and their interpretation outside mutable ndarray headers."""
    if isinstance(value, _ArraySnapshot):
        return value
    array = np.array(value, dtype=np.float64, copy=True, order='C')
    return _ArraySnapshot(array.tobytes(), array.shape)


class _ArrayAccess:
    __slots__ = ()

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        # A read-only ndarray still permits dtype/shape changes. Each access
        # receives its own header; its bytes and the stored shape are immutable.
        return value.view() if isinstance(value, _ArraySnapshot) else value


def _require_finite(array, name):
    if not np.isfinite(array).all():
        raise ValueError(f'{name} is not finite')


def _require_normal_or_zero(array, name):
    if np.any((array != 0.0) & (np.abs(array) < _TINY)):
        raise ValueError(f'{name} is outside the normal float64 range; rescale units')


def _symmetrize_joint_covariance(covariance):
    """Check symmetry in correlation units, so observation units do not hide errors."""
    diagonal = np.diag(covariance)
    if np.any(diagonal <= 0.0):
        raise ValueError('joint_covariance must have positive diagonal variances')
    standard_deviations = np.sqrt(diagonal)
    scales = standard_deviations[:, None] * standard_deviations[None, :]
    limit = (covariance.shape[0] * _EPS) * scales
    with np.errstate(over='ignore'):
        asymmetry = np.abs(covariance - covariance.T)
    if np.any(asymmetry > limit):
        raise ValueError('joint_covariance asymmetry exceeds roundoff in correlation units')
    return 0.5 * (covariance + covariance.T)


@dataclass(frozen=True, eq=False, slots=True)
class CorrelatedLinearFitResult(_ArrayAccess, Mapping):
    """Immutable GLS result. Holdout residual fields are filled by evaluate_holdout."""

    parameters: np.ndarray
    fitted_training: np.ndarray
    predicted_holdout: np.ndarray
    training_residual: np.ndarray
    training_chi_square: float
    training_dof: int
    holdout_residual: np.ndarray | None = None
    holdout_chi_square: float | None = None
    holdout_dof: int | None = None

    def __post_init__(self):
        for name in ('parameters', 'fitted_training', 'predicted_holdout',
                     'training_residual', 'holdout_residual'):
            value = object.__getattribute__(self, name)
            if value is not None:
                object.__setattr__(self, name, _frozen_array(value))

    def __getstate__(self):
        return tuple(object.__getattribute__(self, field.name) for field in fields(self))

    def __setstate__(self, state):
        # Frozen-slot dataclasses otherwise restore the exposed ndarray views.
        for field, value in zip(fields(self), state, strict=True):
            object.__setattr__(self, field.name, value)
        self.__post_init__()

    def __getitem__(self, key):
        if not isinstance(key, str) or key not in self.__dataclass_fields__:
            raise KeyError(key)
        try:
            value = getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self):
        for field in fields(self):
            if getattr(self, field.name) is not None:
                yield field.name

    def __len__(self):
        return sum(1 for _ in self)

    def __repr__(self):
        names = ', '.join(self)
        return f'{self.__class__.__name__}({names})'

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return (set(self) == set(other) and
                all(np.array_equal(self[key], other[key]) for key in self))

    __hash__ = None


class CorrelatedLinearFit(_ArrayAccess):
    """GLS estimator H y_t with known joint covariance of training and holdout.

    Parameters
    ----------
    training_design : array (n_t, p)
        A_t. At least one row and one parameter. n_t >= p is required.
    holdout_design : array (n_h, p)
        A_h with the same parameter columns. At least one row.
    joint_covariance : array (n_t + n_h, n_t + n_h)
        Known SPD covariance of [e_t, e_h]. Roundoff-sized asymmetry is
        symmetrized; larger asymmetry is rejected. No unit floor.
    parameter_scales : array (p,), optional
        Positive scales that multiply whitened training columns before SVD.
        Default ones. Fitted parameters are recovered as scale_j * phi_j.

    Notes
    -----
    H is determined by A_t and C_tt. A_h and the holdout blocks of C enter
    K = A_h H and the residual covariance C_r, never the estimator of theta.
    ``prediction_covariance`` is A_h Cov(theta_hat) A_h^T. It omits
    held-out measurement noise and is not experimental accuracy.
    """

    __slots__ = (
        'training_design',
        'holdout_design',
        'joint_covariance',
        'parameter_scales',
        'estimator',
        'parameter_covariance',
        'prediction_covariance',
        'holdout_residual_covariance',
        'singular_values',
        'rank',
        'rank_threshold',
        'condition_number',
        'training_dof',
        'holdout_dof',
        'n_training',
        'n_holdout',
        'n_parameters',
        '_chol_training',
        '_holdout_residual_factor',
    )

    def __setattr__(self, name, value):
        raise AttributeError(f'{self.__class__.__name__} attributes are frozen')

    def __delattr__(self, name):
        raise AttributeError(f'{self.__class__.__name__} attributes are frozen')

    def __init__(self, training_design, holdout_design, joint_covariance,
                 parameter_scales=None):
        A_t = _as_real_float64(training_design, 'training_design', ndim=2)
        n_t, p = int(A_t.shape[0]), int(A_t.shape[1])
        if n_t < 1:
            raise ValueError('training_design must have at least one row')
        if p < 1:
            raise ValueError('training_design must have at least one parameter column')
        if n_t < p:
            raise ValueError(
                f'training design is underdetermined: n_training={n_t} < n_parameters={p}'
            )
        A_h = _as_real_float64(holdout_design, 'holdout_design', ndim=2)
        if A_h.shape[1] != p:
            raise ValueError(
                f'holdout_design must have shape (n_holdout, {p}), got {A_h.shape}'
            )
        n_h = int(A_h.shape[0])
        if n_h < 1:
            raise ValueError('holdout_design must have at least one row')
        n_tot = n_t + n_h
        C = _as_real_float64(
            joint_covariance, 'joint_covariance', ndim=2, shape=(n_tot, n_tot),
        )
        C_sym = _symmetrize_joint_covariance(C)
        _require_finite(C_sym, 'symmetrized joint_covariance')
        try:
            L = np.linalg.cholesky(C_sym)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                'joint_covariance is not symmetric positive-definite'
            ) from exc
        _require_finite(L, 'Cholesky factor of joint_covariance')
        if parameter_scales is None:
            scales = np.ones(p, dtype=np.float64)
        else:
            scales = _as_real_float64(
                parameter_scales, 'parameter_scales', ndim=1, shape=(p,),
            )
            if np.any(scales <= 0.0):
                raise ValueError('parameter_scales must be strictly positive')
        _require_normal_or_zero(scales, 'parameter_scales')
        L_tt = np.array(L[:n_t, :n_t], dtype=np.float64, copy=True, order='C')
        try:
            whitened = np.linalg.solve(L_tt, A_t)
        except np.linalg.LinAlgError as exc:
            raise ValueError('training covariance block is not usable for whitening') from exc
        _require_finite(whitened, 'whitened training design')
        scaled = whitened * scales
        _require_finite(scaled, 'column-scaled whitened training design')
        _require_normal_or_zero(scaled, 'column-scaled whitened training design')
        if np.any((whitened != 0.0) & (scaled == 0.0)):
            raise ValueError('column scaling underflows to zero; rescale units')
        U, singular, Vt = np.linalg.svd(scaled, full_matrices=False)
        _require_finite(U, 'SVD U')
        _require_finite(singular, 'singular_values')
        _require_normal_or_zero(singular, 'singular_values')
        _require_finite(Vt, 'SVD Vt')
        peak = float(singular[0]) if singular.size else 0.0
        threshold = float(max(n_t, p) * _EPS * peak)
        rank = int(np.count_nonzero(singular > threshold))
        if rank < p:
            raise RankDeficientError(
                f'whitened scaled training rank {rank} is below {p}',
                singular_values=np.array(singular, dtype=np.float64, copy=True),
                rank=rank,
                threshold=threshold,
            )
        # G = D V diag(1/s); theta = G U^T L_tt^{-1} y_t, Cov(theta) = G G^T.
        G = scales[:, None] * Vt.T / singular
        _require_finite(G, 'scaled SVD coefficient factor')
        try:
            estimator = np.linalg.solve(L_tt.T, U @ G.T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError('failed to form the GLS estimator from the training factor') from exc
        _require_finite(estimator, 'estimator')
        parameter_covariance = G @ G.T
        parameter_covariance = 0.5 * (parameter_covariance + parameter_covariance.T)
        _require_finite(parameter_covariance, 'parameter_covariance')
        if np.any(np.diag(parameter_covariance) < _TINY):
            raise ValueError('parameter covariance underflows the supported float64 range')
        prediction_factor = A_h @ G
        _require_finite(prediction_factor, 'prediction factor')
        prediction = prediction_factor @ prediction_factor.T
        prediction = 0.5 * (prediction + prediction.T)
        _require_finite(prediction, 'prediction_covariance')
        active_prediction = np.any(A_h != 0.0, axis=1)
        if np.any(active_prediction & (np.diag(prediction) < _TINY)):
            raise ValueError('prediction_covariance underflows the supported float64 range')
        K = A_h @ estimator
        _require_finite(K, 'K = A_h H')
        F = L[n_t:, :] - K @ L[:n_t, :]
        _require_finite(F, 'holdout residual factor F')
        residual_covariance = F @ F.T
        residual_covariance = 0.5 * (residual_covariance + residual_covariance.T)
        _require_finite(residual_covariance, 'holdout_residual_covariance')
        if np.any(np.diag(residual_covariance) < _TINY):
            raise ValueError('holdout residual covariance underflows the supported float64 range')
        try:
            _q, residual_factor = np.linalg.qr(F.T, mode='reduced')
        except np.linalg.LinAlgError as exc:
            raise ValueError('QR of the holdout residual factor failed') from exc
        _require_finite(residual_factor, 'QR factor of F')
        if np.any(np.abs(np.diag(residual_factor)) == 0.0):
            raise ValueError('holdout residual covariance is numerically rank-deficient')
        smallest = float(singular[-1])
        condition = float(peak / smallest) if smallest > 0.0 else float('inf')
        if not np.isfinite(condition):
            raise ValueError('condition_number is not finite')
        object.__setattr__(self, 'training_design', _frozen_array(A_t))
        object.__setattr__(self, 'holdout_design', _frozen_array(A_h))
        object.__setattr__(self, 'joint_covariance', _frozen_array(C_sym))
        object.__setattr__(self, 'parameter_scales', _frozen_array(scales))
        object.__setattr__(self, 'estimator', _frozen_array(estimator))
        object.__setattr__(self, 'parameter_covariance', _frozen_array(parameter_covariance))
        object.__setattr__(self, 'prediction_covariance', _frozen_array(prediction))
        object.__setattr__(self, 'holdout_residual_covariance', _frozen_array(residual_covariance))
        object.__setattr__(self, 'singular_values', _frozen_array(singular))
        object.__setattr__(self, 'rank', int(rank))
        object.__setattr__(self, 'rank_threshold', float(threshold))
        object.__setattr__(self, 'condition_number', float(condition))
        object.__setattr__(self, 'training_dof', int(n_t - p))
        object.__setattr__(self, 'holdout_dof', int(n_h))
        object.__setattr__(self, 'n_training', n_t)
        object.__setattr__(self, 'n_holdout', n_h)
        object.__setattr__(self, 'n_parameters', p)
        object.__setattr__(self, '_chol_training', _frozen_array(L_tt))
        object.__setattr__(self, '_holdout_residual_factor', _frozen_array(residual_factor))

    def _observation(self, value, length, name):
        return _as_real_float64(value, name, ndim=1, shape=(length,))

    def _training_chi_square(self, residual):
        whitened = np.linalg.solve(self._chol_training, residual)
        return float(whitened @ whitened)

    def _holdout_chi_square(self, residual):
        # C_r = R^T R from the reduced QR of F^T, so chi^2 = ||R^{-T} r||^2.
        solved = np.linalg.solve(self._holdout_residual_factor.T, residual)
        return float(solved @ solved)

    def fit(self, y_train):
        """Estimate theta from training observations only.

        The holdout prediction is A_h theta_hat. It is not a residual-conditioned
        update that uses training residuals or holdout values.
        """
        y_t = self._observation(y_train, self.n_training, 'y_train')
        parameters = self.estimator @ y_t
        _require_finite(parameters, 'parameters')
        fitted = self.training_design @ parameters
        predicted = self.holdout_design @ parameters
        residual = y_t - fitted
        _require_finite(fitted, 'fitted_training')
        _require_finite(predicted, 'predicted_holdout')
        _require_finite(residual, 'training_residual')
        chi_square = self._training_chi_square(residual)
        if not np.isfinite(chi_square):
            raise ValueError('training_chi_square is not finite')
        return CorrelatedLinearFitResult(
            parameters=_frozen_array(parameters),
            fitted_training=_frozen_array(fitted),
            predicted_holdout=_frozen_array(predicted),
            training_residual=_frozen_array(residual),
            training_chi_square=float(chi_square),
            training_dof=int(self.training_dof),
        )

    def evaluate_holdout(self, y_train, y_holdout):
        """Fit from y_train, then score holdout residuals against C_r.

        holdout_dof is n_h under the stated full-rank known-Gaussian-error
        model. No chi-square distribution is claimed for non-Gaussian errors,
        learned covariance, or regularization. y_holdout does not affect
        theta_hat.
        """
        fitted = self.fit(y_train)
        y_h = self._observation(y_holdout, self.n_holdout, 'y_holdout')
        residual = y_h - fitted.predicted_holdout
        _require_finite(residual, 'holdout_residual')
        chi_square = self._holdout_chi_square(residual)
        if not np.isfinite(chi_square):
            raise ValueError('holdout_chi_square is not finite')
        return CorrelatedLinearFitResult(
            parameters=fitted.parameters,
            fitted_training=fitted.fitted_training,
            predicted_holdout=fitted.predicted_holdout,
            training_residual=fitted.training_residual,
            training_chi_square=fitted.training_chi_square,
            training_dof=fitted.training_dof,
            holdout_residual=_frozen_array(residual),
            holdout_chi_square=float(chi_square),
            holdout_dof=int(self.holdout_dof),
        )
