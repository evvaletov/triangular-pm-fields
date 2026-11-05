"""Synthetic regressions for the known-fixed-joint-covariance linear fit.

No physical PM scan is used. Designs, covariances, and Gaussian draws are
constructed in this file. Ensemble seed, counts, covariance, and acceptance
were frozen before looking at ensemble output.
"""
from __future__ import annotations

import unittest

import mpmath as mp
import numpy as np

from triangular_pm_fields.correlated_fit import (
    CorrelatedLinearFit,
    CorrelatedLinearFitResult,
    RankDeficientError,
)

# Frozen 5 November 2026 before looking at ensemble output. Do not retune.
ENSEMBLE_SEED = 2026110501
ENSEMBLE_COUNT = 512
ENSEMBLE_A_TRAIN = np.array([
    [1.00, 0.00],
    [1.00, 0.50],
    [1.00, 1.00],
    [1.00, 1.50],
    [1.00, 2.00],
    [0.50, 2.50],
    [0.00, 3.00],
    [-0.50, 3.50],
], dtype=np.float64)
ENSEMBLE_A_HOLD = np.array([
    [1.00, 0.25],
    [1.00, 1.25],
    [0.50, 2.25],
    [0.00, 3.25],
    [-0.50, 4.00],
], dtype=np.float64)
ENSEMBLE_THETA = np.array([1.25, -0.40], dtype=np.float64)
ENSEMBLE_SIGMA = 0.12
ENSEMBLE_RHO = 0.35
ENSEMBLE_MEAN_Z = 4.0
ENSEMBLE_CHI2_Z = 4.0
ENSEMBLE_COVERAGE_Z = 4.0
ENSEMBLE_COV_FROB_REL = 0.30
NORMAL_Z95 = 1.959963984540054


def ar1_covariance(n, sigma, rho):
    index = np.arange(n, dtype=np.float64)
    return (sigma ** 2) * (rho ** np.abs(index[:, None] - index[None, :]))


def compound_symmetry(n, independent, shared):
    return independent * np.eye(n, dtype=np.float64) + shared * np.ones((n, n), dtype=np.float64)


def block_joint(C_tt, C_th, C_hh):
    n_t = C_tt.shape[0]
    n_h = C_hh.shape[0]
    C = np.empty((n_t + n_h, n_t + n_h), dtype=np.float64)
    C[:n_t, :n_t] = C_tt
    C[:n_t, n_t:] = C_th
    C[n_t:, :n_t] = C_th.T
    C[n_t:, n_t:] = C_hh
    return 0.5 * (C + C.T)


def four_term_residual_covariance(C, K, n_t):
    C_tt = C[:n_t, :n_t]
    C_th = C[:n_t, n_t:]
    C_ht = C[n_t:, :n_t]
    C_hh = C[n_t:, n_t:]
    return C_hh + K @ C_tt @ K.T - C_ht @ K.T - K @ C_th


def mp_cholesky(A):
    n = A.rows
    L = mp.zeros(n)
    for i in range(n):
        for j in range(i + 1):
            total = sum((L[i, k] * L[j, k] for k in range(j)), mp.mpf('0'))
            delta = A[i, j] - total
            if i == j:
                L[i, j] = mp.sqrt(delta)
            else:
                L[i, j] = delta / L[j, j]
    return L


def mp_forward(L, B):
    n, cols = L.rows, B.cols
    X = mp.zeros(n, cols)
    for col in range(cols):
        for i in range(n):
            total = sum((L[i, j] * X[j, col] for j in range(i)), mp.mpf('0'))
            X[i, col] = (B[i, col] - total) / L[i, i]
    return X


def mp_back_upper(R, rhs):
    n = R.rows
    x = mp.zeros(n, 1)
    for i in range(n - 1, -1, -1):
        total = sum((R[i, j] * x[j, 0] for j in range(i + 1, n)), mp.mpf('0'))
        x[i, 0] = (rhs[i, 0] - total) / R[i, i]
    return x


def mp_thin_qr(A):
    n, p = A.rows, A.cols
    Q = mp.zeros(n, p)
    R = mp.zeros(p)
    for k in range(p):
        v = mp.matrix(n, 1)
        for i in range(n):
            v[i, 0] = A[i, k]
        for j in range(k):
            R[j, k] = sum((Q[i, j] * v[i, 0] for i in range(n)), mp.mpf('0'))
            for i in range(n):
                v[i, 0] -= R[j, k] * Q[i, j]
        R[k, k] = mp.sqrt(sum((v[i, 0] ** 2 for i in range(n)), mp.mpf('0')))
        for i in range(n):
            Q[i, k] = v[i, 0] / R[k, k]
    return Q, R


def high_precision_gls(A, y, C_tt, dps=64):
    """Independent GLS via high-precision Cholesky and thin QR, not SVD."""
    with mp.workdps(dps):
        A_m = mp.matrix(np.asarray(A, dtype=np.float64).tolist())
        y_m = mp.matrix(np.asarray(y, dtype=np.float64).reshape(-1, 1).tolist())
        C_m = mp.matrix(np.asarray(C_tt, dtype=np.float64).tolist())
        L = mp_cholesky(C_m)
        A_w = mp_forward(L, A_m)
        y_w = mp_forward(L, y_m)
        Q, R = mp_thin_qr(A_w)
        theta_m = mp_back_upper(R, Q.T * y_w)
        gram = R.T * R
        cov_m = mp.inverse(gram)
        theta = np.array([float(theta_m[i, 0]) for i in range(A_m.cols)], dtype=np.float64)
        cov = np.array(
            [[float(cov_m[i, j]) for j in range(A_m.cols)] for i in range(A_m.cols)],
            dtype=np.float64,
        )
    return theta, 0.5 * (cov + cov.T)


def numpy_qr_gls(A, y, C_tt):
    L = np.linalg.cholesky(C_tt)
    A_w = np.linalg.solve(L, A)
    y_w = np.linalg.solve(L, y)
    Q, R = np.linalg.qr(A_w, mode='reduced')
    theta = np.linalg.solve(R, Q.T @ y_w)
    cov = np.linalg.inv(R.T @ R)
    return theta, 0.5 * (cov + cov.T)


def toy_full_rank(n_t=6, n_h=4, p=2, seed=11):
    rng = np.random.default_rng(seed)
    raw_t, _ = np.linalg.qr(rng.normal(size=(n_t, n_t)), mode='reduced')
    raw_h, _ = np.linalg.qr(rng.normal(size=(n_h, n_h)), mode='reduced')
    singular = np.linspace(3.0, 1.2, p)
    A_t = raw_t[:, :p] * singular
    A_h = raw_h[:, :p] * np.linspace(2.5, 1.1, p)
    C = ar1_covariance(n_t + n_h, 0.2, 0.3)
    C = C + 0.04 * np.ones_like(C)
    C = 0.5 * (C + C.T)
    return A_t, A_h, C


def assert_bytes_frozen(test, array, name):
    test.assertFalse(array.flags.writeable, msg=name)
    with test.assertRaises((ValueError, AttributeError)):
        array.flags.writeable = True
    with test.assertRaises((ValueError, TypeError)):
        array.ravel()[0] = 1.0


class ValidationTests(unittest.TestCase):
    def test_rejects_complex_without_coercion(self):
        A_t, A_h, C = toy_full_rank()
        complex_A = A_t.astype(np.complex128)
        complex_A.imag[:] = 0.0
        with self.assertRaises(TypeError):
            CorrelatedLinearFit(complex_A, A_h, C)
        with self.assertRaises(TypeError):
            CorrelatedLinearFit(A_t, A_h.astype(np.complex128), C)
        with self.assertRaises(TypeError):
            CorrelatedLinearFit(A_t, A_h, C.astype(np.complex128))
        fit = CorrelatedLinearFit(A_t, A_h, C)
        y = np.ones(A_t.shape[0], dtype=np.complex128)
        with self.assertRaises(TypeError):
            fit.fit(y)

    def test_rejects_shapes_nonfinite_and_nonpositive_scales(self):
        A_t, A_h, C = toy_full_rank()
        CorrelatedLinearFit(A_t, A_h, C)
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(A_t.reshape(-1), A_h, C)
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(np.zeros((0, 2)), A_h, C)
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(A_t, np.zeros((0, 2)), C)
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(A_t[:, :0], A_h[:, :0], np.eye(A_t.shape[0] + A_h.shape[0]))
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(A_t, A_h[:, :1], C)
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(A_t, A_h, C[:-1, :-1])
        bad = A_t.copy()
        bad[0, 0] = np.inf
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(bad, A_h, C)
        nan_C = C.copy()
        nan_C[1, 1] = np.nan
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(A_t, A_h, nan_C)
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(A_t, A_h, C, parameter_scales=[1.0, 0.0])
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(A_t, A_h, C, parameter_scales=[-1.0, 1.0])
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(A_t, A_h, C, parameter_scales=[1.0])
        fit = CorrelatedLinearFit(A_t, A_h, C)
        with self.assertRaises(ValueError):
            fit.fit(np.ones(A_t.shape[0] - 1))
        with self.assertRaises(ValueError):
            fit.fit(np.ones((A_t.shape[0], 1)))
        with self.assertRaises(ValueError):
            fit.evaluate_holdout(np.ones(A_t.shape[0]), np.ones((A_h.shape[0], 1)))

    def test_rejects_underdetermined_before_svd(self):
        A_t = np.array([[1.0, 0.0, 0.5]], dtype=np.float64)
        A_h = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float64)
        C = np.eye(3, dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            CorrelatedLinearFit(A_t, A_h, C)
        self.assertIn('underdetermined', str(ctx.exception))


class CovarianceValidationTests(unittest.TestCase):
    def test_rejects_indefinite_asymmetric_and_accepts_roundoff(self):
        A_t = np.array([[1.0], [1.0]], dtype=np.float64)
        A_h = np.array([[1.0]], dtype=np.float64)
        indefinite = np.diag([1.0, 1.0, -0.2])
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(A_t, A_h, indefinite)
        asymmetric = np.array([
            [1.0, 0.4, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        with self.assertRaises(ValueError):
            CorrelatedLinearFit(A_t, A_h, asymmetric)
        C = np.eye(3, dtype=np.float64)
        C[0, 1] = np.finfo(np.float64).eps
        fit = CorrelatedLinearFit(A_t, A_h, C)
        np.testing.assert_allclose(fit.joint_covariance, fit.joint_covariance.T)
        self.assertEqual(fit.rank, 1)


class ImmutabilityTests(unittest.TestCase):
    def test_copies_inputs_and_freezes_owned_arrays(self):
        A_t, A_h, C = toy_full_rank()
        scales = np.array([1.5, 0.4], dtype=np.float64)
        A_t_src, A_h_src, C_src, s_src = A_t.copy(), A_h.copy(), C.copy(), scales.copy()
        fit = CorrelatedLinearFit(A_t_src, A_h_src, C_src, parameter_scales=s_src)
        A_t_src[:] = 0.0
        A_h_src[:] = 0.0
        C_src[:] = 0.0
        s_src[:] = 9.0
        self.assertGreater(np.linalg.norm(fit.training_design), 0.0)
        self.assertGreater(np.linalg.norm(fit.holdout_design), 0.0)
        self.assertGreater(np.linalg.norm(fit.joint_covariance), 0.0)
        np.testing.assert_allclose(fit.parameter_scales, scales)
        for name in (
            'training_design', 'holdout_design', 'joint_covariance',
            'parameter_scales', 'estimator', 'parameter_covariance',
            'prediction_covariance', 'holdout_residual_covariance',
            'singular_values',
        ):
            assert_bytes_frozen(self, getattr(fit, name), name)
        with self.assertRaises(AttributeError):
            fit.rank = 0
        y = A_t @ np.array([0.3, -0.2])
        result = fit.fit(y)
        self.assertIsInstance(result, CorrelatedLinearFitResult)
        for name in (
            'parameters', 'fitted_training', 'predicted_holdout', 'training_residual',
        ):
            assert_bytes_frozen(self, result[name], name)
        with self.assertRaises(AttributeError):
            result.training_dof = -1
        y_mut = y.copy()
        held = result.parameters.copy()
        y_mut[:] = 99.0
        np.testing.assert_allclose(result.parameters, held)
        self.assertNotIn('holdout_residual', result)
        self.assertEqual(len(result), 6)


class RecoveryAndGLSTests(unittest.TestCase):
    def test_noiseless_recovery_and_estimator_identity(self):
        A_t, A_h, C = toy_full_rank()
        theta = np.array([0.7, -1.1], dtype=np.float64)
        y_t = A_t @ theta
        y_h = A_h @ theta
        fit = CorrelatedLinearFit(A_t, A_h, C)
        result = fit.evaluate_holdout(y_t, y_h)
        np.testing.assert_allclose(result.parameters, theta, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.predicted_holdout, y_h, rtol=1e-12, atol=1e-12)
        self.assertLess(result.training_chi_square, 1e-18)
        self.assertLess(result.holdout_chi_square, 1e-18)
        self.assertEqual(result.training_dof, A_t.shape[0] - A_t.shape[1])
        self.assertEqual(result.holdout_dof, A_h.shape[0])
        np.testing.assert_allclose(fit.estimator @ A_t, np.eye(A_t.shape[1]), atol=1e-12)

    def test_matches_independent_high_precision_qr_and_numpy_qr(self):
        A_t, A_h, C = toy_full_rank(n_t=5, n_h=3, p=2, seed=21)
        theta = np.array([1.05, 0.25], dtype=np.float64)
        rng = np.random.default_rng(22)
        L = np.linalg.cholesky(C)
        noise = L @ rng.standard_normal(C.shape[0])
        y_t = A_t @ theta + noise[:A_t.shape[0]]
        fit = CorrelatedLinearFit(A_t, A_h, C)
        result = fit.fit(y_t)
        hp_theta, hp_cov = high_precision_gls(A_t, y_t, C[:A_t.shape[0], :A_t.shape[0]], dps=64)
        qr_theta, qr_cov = numpy_qr_gls(A_t, y_t, C[:A_t.shape[0], :A_t.shape[0]])
        np.testing.assert_allclose(result.parameters, hp_theta, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.parameters, qr_theta, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fit.parameter_covariance, hp_cov, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(fit.parameter_covariance, qr_cov, rtol=1e-11, atol=1e-14)

    def test_diagonal_c_matches_weighted_svd_and_lstsq(self):
        rng = np.random.default_rng(31)
        A_t = rng.normal(size=(9, 3))
        A_h = rng.normal(size=(4, 3))
        sigma = np.array([0.05, 0.08, 0.04, 0.07, 0.06, 0.09, 0.03, 0.11, 0.05], dtype=np.float64)
        C = np.diag(np.concatenate((sigma ** 2, np.full(4, 0.04))))
        theta = np.array([0.4, -0.2, 1.1], dtype=np.float64)
        y_t = A_t @ theta + sigma * rng.standard_normal(9)
        fit = CorrelatedLinearFit(A_t, A_h, C)
        result = fit.fit(y_t)
        weighted_A = A_t / sigma[:, None]
        weighted_y = y_t / sigma
        U, s, Vt = np.linalg.svd(weighted_A, full_matrices=False)
        svd_theta = Vt.T @ ((U.T @ weighted_y) / s)
        lstsq_theta, *_ = np.linalg.lstsq(weighted_A, weighted_y, rcond=None)
        np.testing.assert_allclose(result.parameters, svd_theta, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.parameters, lstsq_theta, rtol=1e-12, atol=1e-12)

    def test_holdout_values_do_not_enter_estimator(self):
        A_t, A_h, C = toy_full_rank()
        A_h_other = A_h * 1.7 + 0.3
        C_other = C.copy()
        C_other[A_t.shape[0]:, :] *= 0.0
        C_other[:, A_t.shape[0]:] *= 0.0
        C_other[A_t.shape[0]:, A_t.shape[0]:] = np.eye(A_h.shape[0])
        C_other = 0.5 * (C_other + C_other.T)
        # Keep C_tt identical; scramble holdout blocks and A_h.
        C_other[:A_t.shape[0], :A_t.shape[0]] = C[:A_t.shape[0], :A_t.shape[0]]
        fit_a = CorrelatedLinearFit(A_t, A_h, C)
        fit_b = CorrelatedLinearFit(A_t, A_h_other, C_other)
        np.testing.assert_allclose(fit_a.estimator, fit_b.estimator, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            fit_a.parameter_covariance, fit_b.parameter_covariance, rtol=1e-12, atol=1e-12,
        )
        y_t = A_t @ np.array([0.2, 0.9])
        y_h_1 = np.ones(A_h.shape[0])
        y_h_2 = 50.0 * np.arange(A_h.shape[0], dtype=np.float64)
        r1 = fit_a.evaluate_holdout(y_t, y_h_1)
        r2 = fit_a.evaluate_holdout(y_t, y_h_2)
        np.testing.assert_array_equal(np.array(r1.parameters), np.array(r2.parameters))
        self.assertNotEqual(r1.holdout_chi_square, r2.holdout_chi_square)

    def test_residual_covariance_matches_four_term_on_generic_spd(self):
        A_t, A_h, C = toy_full_rank(seed=41)
        fit = CorrelatedLinearFit(A_t, A_h, C)
        K = A_h @ fit.estimator
        explicit = four_term_residual_covariance(fit.joint_covariance, K, A_t.shape[0])
        np.testing.assert_allclose(
            fit.holdout_residual_covariance, explicit, rtol=1e-12, atol=1e-14,
        )
        predicted = A_h @ fit.parameter_covariance @ A_h.T
        np.testing.assert_allclose(
            fit.prediction_covariance, predicted, rtol=1e-12, atol=1e-14,
        )
        self.assertGreater(
            np.linalg.norm(fit.holdout_residual_covariance - fit.prediction_covariance),
            1e-6,
        )


class SharedGainTests(unittest.TestCase):
    def test_analytic_variance_floor_and_cross_term_cancellation(self):
        n_t, n_h = 6, 4
        sigma2, tau2 = 0.04, 0.25
        A_t = np.ones((n_t, 1), dtype=np.float64)
        A_h = np.ones((n_h, 1), dtype=np.float64)
        C = compound_symmetry(n_t + n_h, sigma2, tau2)
        fit = CorrelatedLinearFit(A_t, A_h, C)
        analytic_var = sigma2 / n_t + tau2
        np.testing.assert_allclose(fit.parameter_covariance, [[analytic_var]], rtol=1e-12)
        analytic_Cr = sigma2 * np.eye(n_h) + (sigma2 / n_t) * np.ones((n_h, n_h))
        np.testing.assert_allclose(fit.holdout_residual_covariance, analytic_Cr, rtol=1e-12, atol=1e-14)
        C_disc = block_joint(
            C[:n_t, :n_t], np.zeros((n_t, n_h)), C[n_t:, n_t:],
        )
        discarded = CorrelatedLinearFit(A_t, A_h, C_disc)
        np.testing.assert_allclose(discarded.parameter_covariance, fit.parameter_covariance)
        self.assertGreater(
            np.linalg.norm(discarded.holdout_residual_covariance - fit.holdout_residual_covariance),
            0.1,
        )
        C_diag = (sigma2 + tau2) * np.eye(n_t + n_h)
        diagonal = CorrelatedLinearFit(A_t, A_h, C_diag)
        self.assertLess(float(diagonal.parameter_covariance[0, 0]), 0.5 * analytic_var)
        y_t = np.full(n_t, 2.0)
        y_h = np.full(n_h, 2.0)
        scored = fit.evaluate_holdout(y_t, y_h)
        self.assertEqual(scored.holdout_dof, n_h)
        self.assertLess(scored.holdout_chi_square, 1e-18)


class ScalingAndRankTests(unittest.TestCase):
    def test_parameter_scale_and_unit_invariance(self):
        A_t, A_h, C = toy_full_rank(seed=51)
        theta = np.array([0.55, -0.8], dtype=np.float64)
        y_t = A_t @ theta
        y_h = A_h @ theta
        base = CorrelatedLinearFit(A_t, A_h, C)
        scaled = CorrelatedLinearFit(A_t, A_h, C, parameter_scales=[2.0, 0.25])
        np.testing.assert_allclose(
            base.fit(y_t).parameters, scaled.fit(y_t).parameters, rtol=1e-12, atol=1e-12,
        )
        alpha = 4.0
        unit = CorrelatedLinearFit(alpha * A_t, alpha * A_h, (alpha ** 2) * C)
        np.testing.assert_allclose(
            unit.fit(alpha * y_t).parameters, theta, rtol=1e-12, atol=1e-12,
        )
        D = np.diag([2.0, 0.5])
        reparam = CorrelatedLinearFit(A_t @ D, A_h @ D, C)
        recovered = reparam.evaluate_holdout(y_t, y_h).parameters
        np.testing.assert_allclose(D @ recovered, theta, rtol=1e-12, atol=1e-12)

    def test_rejects_rank_deficiency_and_keeps_singular_values(self):
        A_t, A_h, C = toy_full_rank(n_t=8, n_h=3, p=2, seed=61)
        A_t[:, 1] = 2.0 * A_t[:, 0]
        A_h[:, 1] = 2.0 * A_h[:, 0]
        with self.assertRaises(RankDeficientError) as ctx:
            CorrelatedLinearFit(A_t, A_h, C)
        err = ctx.exception
        self.assertEqual(err.rank, 1)
        self.assertEqual(len(err.singular_values), 2)
        self.assertGreater(err.threshold, 0.0)
        self.assertLessEqual(err.singular_values[1], err.threshold)

    def test_square_determined_design_has_zero_training_dof(self):
        rng = np.random.default_rng(71)
        A_t, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        A_h = rng.normal(size=(2, 3))
        C = np.eye(5, dtype=np.float64)
        fit = CorrelatedLinearFit(A_t, A_h, C)
        self.assertEqual(fit.training_dof, 0)
        self.assertEqual(fit.rank, 3)
        theta = np.array([0.1, 0.2, -0.3])
        result = fit.fit(A_t @ theta)
        np.testing.assert_allclose(result.parameters, theta, rtol=1e-12, atol=1e-12)
        self.assertLess(result.training_chi_square, 1e-18)


class EnsembleTests(unittest.TestCase):
    def test_unbiased_gaussian_scan_matches_frozen_acceptance(self):
        n_t = ENSEMBLE_A_TRAIN.shape[0]
        n_h = ENSEMBLE_A_HOLD.shape[0]
        C = ar1_covariance(n_t + n_h, ENSEMBLE_SIGMA, ENSEMBLE_RHO)
        fit = CorrelatedLinearFit(ENSEMBLE_A_TRAIN, ENSEMBLE_A_HOLD, C)
        L = np.linalg.cholesky(C)
        rng = np.random.default_rng(ENSEMBLE_SEED)
        z = rng.standard_normal((ENSEMBLE_COUNT, C.shape[0]))
        errors = z @ L.T
        y_t = ENSEMBLE_A_TRAIN @ ENSEMBLE_THETA + errors[:, :n_t]
        y_h = ENSEMBLE_A_HOLD @ ENSEMBLE_THETA + errors[:, n_t:]
        parameters = y_t @ fit.estimator.T
        mean_error = parameters.mean(axis=0) - ENSEMBLE_THETA
        se = np.sqrt(np.diag(fit.parameter_covariance) / ENSEMBLE_COUNT)
        self.assertTrue(np.all(np.abs(mean_error) <= ENSEMBLE_MEAN_Z * se))
        sample_cov = np.cov(parameters, rowvar=False, ddof=1)
        rel = np.linalg.norm(sample_cov - fit.parameter_covariance) / np.linalg.norm(
            fit.parameter_covariance
        )
        self.assertLessEqual(rel, ENSEMBLE_COV_FROB_REL)
        std = np.sqrt(np.diag(fit.parameter_covariance))
        coverage = np.mean(np.abs(parameters - ENSEMBLE_THETA) <= NORMAL_Z95 * std, axis=0)
        coverage_se = np.sqrt(0.95 * 0.05 / ENSEMBLE_COUNT)
        self.assertTrue(np.all(np.abs(coverage - 0.95) <= ENSEMBLE_COVERAGE_Z * coverage_se))
        train_chi = np.empty(ENSEMBLE_COUNT)
        hold_chi = np.empty(ENSEMBLE_COUNT)
        for i in range(ENSEMBLE_COUNT):
            scored = fit.evaluate_holdout(y_t[i], y_h[i])
            train_chi[i] = scored.training_chi_square
            hold_chi[i] = scored.holdout_chi_square
            np.testing.assert_allclose(scored.parameters, parameters[i])
        train_se = np.sqrt(2.0 * fit.training_dof / ENSEMBLE_COUNT)
        hold_se = np.sqrt(2.0 * fit.holdout_dof / ENSEMBLE_COUNT)
        self.assertLessEqual(abs(train_chi.mean() - fit.training_dof), ENSEMBLE_CHI2_Z * train_se)
        self.assertLessEqual(abs(hold_chi.mean() - fit.holdout_dof), ENSEMBLE_CHI2_Z * hold_se)


if __name__ == '__main__':
    unittest.main()
