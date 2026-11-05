"""Synthetic shared-offset correlated linear fit.

No physical magnet scan is used. Training and holdout observe one scalar
parameter through all-ones design rows. Each observation is that parameter
plus a shared random calibration offset g and independent noise. The known
joint covariance

    C = sigma_e^2 I + tau^2 1 1^T

is supplied by the caller. This script does not calibrate an instrument and
is not experimental validation.

The GLS variance of the fitted parameter is sigma_e^2 / n_train + tau^2. The
shared offset therefore sets a floor that extra training rows cannot remove.
Holdout minus training-mean residuals cancel g, so the holdout residual
covariance does not contain tau^2. Zeroing the train/holdout cross blocks
keeps the parameter variance and inflates the residual covariance; that
truncation is the wrong uncertainty for this model.
"""
from __future__ import annotations

import numpy as np

from triangular_pm_fields.correlated_fit import CorrelatedLinearFit


def compound_symmetry(n, independent, shared):
    return independent * np.eye(n) + shared * np.ones((n, n))


def main():
    n_train, n_hold = 6, 4
    sigma_e, tau = 0.20, 0.50
    theta = 1.35
    A_t = np.ones((n_train, 1))
    A_h = np.ones((n_hold, 1))
    C = compound_symmetry(n_train + n_hold, sigma_e ** 2, tau ** 2)
    fit = CorrelatedLinearFit(A_t, A_h, C)

    analytic_var = sigma_e ** 2 / n_train + tau ** 2
    analytic_Cr = (sigma_e ** 2) * np.eye(n_hold) + (sigma_e ** 2 / n_train) * np.ones((n_hold, n_hold))
    C_no_cross = C.copy()
    C_no_cross[:n_train, n_train:] = 0.0
    C_no_cross[n_train:, :n_train] = 0.0
    discarded = CorrelatedLinearFit(A_t, A_h, 0.5 * (C_no_cross + C_no_cross.T))

    y_t = np.full(n_train, theta)
    y_h = np.full(n_hold, theta)
    noiseless = fit.evaluate_holdout(y_t, y_h)

    rng = np.random.default_rng(2026110503)
    L = np.linalg.cholesky(C)
    sample = np.concatenate((A_t @ [theta], A_h @ [theta])) + L @ rng.standard_normal(n_train + n_hold)
    noisy = fit.evaluate_holdout(sample[:n_train], sample[n_train:])

    print('CorrelatedLinearFit shared-offset toy (synthetic, known C)')
    print(f'  n_train={n_train} n_hold={n_hold} sigma_e={sigma_e} tau={tau}')
    print(f'  rank={fit.rank} training_dof={fit.training_dof} holdout_dof={fit.holdout_dof}')
    print(f'  analytic parameter variance {analytic_var:.12f}')
    print(f'  fitted parameter variance   {fit.parameter_covariance[0, 0]:.12f}')
    print(f'  parameter prediction cov    {fit.prediction_covariance[0, 0]:.12f}')
    print(f'  holdout residual diag mean  {np.mean(np.diag(fit.holdout_residual_covariance)):.12f}')
    print(f'  analytic residual diag mean {np.mean(np.diag(analytic_Cr)):.12f}')
    print(f'  discarded-cross residual diag mean '
          f'{np.mean(np.diag(discarded.holdout_residual_covariance)):.12f}')
    print(f'  noiseless parameters {noiseless.parameters[0]:.12f} '
          f'train_chi2={noiseless.training_chi_square:.3e} '
          f'hold_chi2={noiseless.holdout_chi_square:.3e}')
    print(f'  one noisy draw parameters {noisy.parameters[0]:.12f} '
          f'train_chi2={noisy.training_chi_square:.6f} '
          f'hold_chi2={noisy.holdout_chi_square:.6f}')
    print('  The example uses a specified covariance and synthetic observations.')


if __name__ == '__main__':
    main()
