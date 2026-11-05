"""Fixed synthetic PM Gaussian verification; see PROTOCOL.txt for frozen gates.

Supply --arrays field_fit_20261103/references/arrays.npz; optionally --candidate
correlated_fit.py. The independent QR baseline runs even without a candidate.
No field evaluator, geometry optimizer or learned covariance is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from check_oracle import load_candidate

N, SEED, NT, P = 4096, 2026110507, 96, 6


def prepare(path):
    with np.load(path, allow_pickle=False) as data:
        a = data['nominal_holdout'][:64].reshape(192, 6).copy()
        fine = data['geometry_sensitivity_fine']
        coarse = data['geometry_sensitivity_coarse']
        points = data['holdout_points'][:64]
        np.testing.assert_array_equal(points, data['tolerance_points'])
        assert fine.shape == coarse.shape == (64, 3, 18)
        assert np.isfinite(fine).all() and np.isfinite(coarse).all() and np.isfinite(a).all()
        sensitivity = -(fine[:, :, :3]+fine[:, :, 9:12]).reshape(192, 3)
        sig = np.tile([50e-6]*3+[1e-3]*3+[50e-6]*3, 2)
        delta = np.linalg.norm(((fine-coarse)*sig).reshape(-1, 18), axis=0)
        bound = 1e-9+1e-3*np.linalg.norm((fine*sig).reshape(-1, 18), axis=0)
        assert np.all(delta <= bound), 'Existing sensitivity stencil qualification failed'
    theta = np.array([0., 1.2, 0., 0., 1.2, 0.])
    white = np.tile(np.array([10., 15., 20.])*1e-6, 64)
    axis_offsets = np.tile(np.eye(3), (64, 1))*np.array([20., 25., 30.])*1e-6
    u = np.column_stack((.001*(a@theta), axis_offsets,
                         sensitivity*np.array([25., 35., 45.])*1e-6))
    c = np.diag(white**2)+u@u.T
    return a, theta, white, u, c, float(np.max(delta/bound))


def qr_reference(a, c):
    at, ah = a[:NT], a[NT:]
    lt = np.linalg.cholesky(c[:NT, :NT])
    q, r = np.linalg.qr(np.linalg.solve(lt, at), mode='reduced')
    h = np.linalg.solve(r, np.linalg.solve(lt.T, q).T)
    inv_r = np.linalg.solve(r, np.eye(P))
    covariance = inv_r@inv_r.T
    k = ah@h
    # Explicit cross-block formula, independent of candidate F F^T construction.
    cr = c[NT:, NT:]+k@c[:NT, :NT]@k.T-c[NT:, :NT]@k.T-k@c[:NT, NT:]
    cr = (cr+cr.T)/2
    return h, covariance, cr, lt


def max_covariance_z(values):
    dimension = values.shape[1]
    identity = np.eye(dimension)
    empirical = np.cov(values, rowvar=False, ddof=1)
    return float(np.max(np.abs(empirical-identity)/np.sqrt((1+identity)/(N-1))))


def chi_metrics(values, dof):
    variance = 2*dof
    se_var = np.sqrt((12*dof**2+48*dof-(N-3)/(N-1)*variance**2)/N)
    return dict(dof=dof, mean=float(np.mean(values)), variance=float(np.var(values, ddof=1)),
                mean_z=float((np.mean(values)-dof)/np.sqrt(variance/N)),
                variance_z=float((np.var(values, ddof=1)-variance)/se_var))


def compare_array(actual, expected, absolute=0):
    np.testing.assert_allclose(actual, expected, rtol=2e-10,
                               atol=max(absolute, 2e-12*np.max(np.abs(expected))))


def run(arrays, candidate=None, raw_output=None):
    started = time.monotonic()
    a, theta, white, u, c, refinement = prepare(arrays)
    h, covariance, cr, lt = qr_reference(a, c)
    rng = np.random.default_rng(SEED)
    errors = rng.standard_normal((N, 192))*white
    errors += rng.standard_normal((N, 7))@u.T
    observations = a@theta+errors
    parameters = observations[:, :NT]@h.T
    rt = observations[:, :NT]-parameters@a[:NT].T
    rh = observations[:, NT:]-parameters@a[NT:].T
    qt = np.sum(np.linalg.solve(lt, rt.T)**2, axis=0)
    lr = np.linalg.cholesky(cr)
    qh = np.sum(np.linalg.solve(lr, rh.T)**2, axis=0)
    candidate_info = None
    if candidate is not None:
        fit = load_candidate(candidate)(a[:NT], a[NT:], c)
        for actual, expected in [(fit.estimator, h), (fit.parameter_covariance, covariance),
                                 (fit.holdout_residual_covariance, cr),
                                 (fit.prediction_covariance, a[NT:]@covariance@a[NT:].T)]:
            compare_array(actual, expected)
        for i, obs in enumerate(observations):
            result = fit.evaluate_holdout(obs[:NT], obs[NT:])
            compare_array(result['parameters'], parameters[i])
            compare_array(result['training_residual'], rt[i], 2e-12)
            compare_array(result['holdout_residual'], rh[i], 2e-12)
            compare_array(result['predicted_holdout'], a[NT:]@parameters[i], 2e-12)
            compare_array(result['training_chi_square'], qt[i])
            compare_array(result['holdout_chi_square'], qh[i])
            assert result['training_dof'] == NT-P and result['holdout_dof'] == 192-NT
        candidate_info = dict(sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                              complete_scans_compared=N, passed=True)
    zp = np.linalg.solve(np.linalg.cholesky(covariance), (parameters-theta).T).T
    zr = np.linalg.solve(lr, rh.T).T
    cover = np.mean(np.abs(parameters-theta) <= 1.959963984540054*np.sqrt(np.diag(covariance)), axis=0)
    statistics = dict(parameter_mean_bias_T=(parameters.mean(axis=0)-theta).tolist(),
                      parameter_mean_bias_z=((parameters.mean(axis=0)-theta)/np.sqrt(np.diag(covariance)/N)).tolist(),
                      parameter_mean_max_abs_z=float(np.max(np.abs(zp.mean(axis=0)))*np.sqrt(N)),
                      parameter_covariance_max_abs_z=max_covariance_z(zp),
                      holdout_residual_covariance_max_abs_z=max_covariance_z(zr),
                      parameter_95pct_coverage=cover.tolist(),
                      parameter_coverage_max_abs_z=float(np.max(np.abs(cover-.95))/np.sqrt(.95*.05/N)),
                      training_chi_square=chi_metrics(qt, NT-P),
                      holdout_chi_square=chi_metrics(qh, 192-NT))
    # Descriptive cross-covariance omission diagnostic, not an additional gate.
    # Both marginal covariances are unchanged. The residuals retain their true
    # joint generative model, making the omitted-cross quadratic miscalibrated.
    k = a[NT:]@h
    wrong_cr = c[NT:, NT:]+k@c[:NT, :NT]@k.T
    wrong_qh = np.sum(np.linalg.solve(np.linalg.cholesky(wrong_cr), rh.T)**2, axis=0)
    omission = dict(expected_chi_square_mean=float(np.trace(np.linalg.solve(wrong_cr, cr))),
                    observed_chi_square_mean=float(np.mean(wrong_qh)),
                    correct_degrees_of_freedom=192-NT,
                    residual_covariance_relative_frobenius_change=float(np.linalg.norm(wrong_cr-cr)/np.linalg.norm(cr)),
                    interpretation='descriptive negative control; marginal covariances fixed, true cross covariance incorrectly omitted')
    gates = [statistics[key] for key in ['parameter_mean_max_abs_z',
             'parameter_covariance_max_abs_z', 'holdout_residual_covariance_max_abs_z',
             'parameter_coverage_max_abs_z']]
    for key in ('training_chi_square', 'holdout_chi_square'):
        gates.extend(abs(statistics[key][x]) for x in ('mean_z', 'variance_z'))
    if raw_output:
        np.savez_compressed(raw_output, parameters=parameters, training_chi_square=qt,
                            holdout_chi_square=qh, holdout_residual=rh,
                            parameter_covariance=covariance, holdout_residual_covariance=cr)
    result = dict(passed=bool(max(gates) <= 6), seed=SEED, scans=N,
                  fixed_covariance_model='white + shared gain + axis offsets + fixed global position sensitivity',
                  interpretation='synthetic fixed known joint Gaussian covariance; unconditional holdout residual',
                  arrays_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256(Path(__file__).with_name('PROTOCOL.txt').read_bytes()).hexdigest(),
                  existing_sensitivity_max_bound_ratio=refinement,
                  statistics=statistics, cross_covariance_omission=omission, candidate=candidate_info,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   affinity=sorted(os.sched_getaffinity(0)),
                                   OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),
                                   elapsed_s=time.monotonic()-started))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrays', required=True, type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--raw-output', type=Path)
    args = parser.parse_args()
    result = run(args.arrays, args.candidate, args.raw_output)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
