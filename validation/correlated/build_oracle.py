"""Independent 80-digit oracle; no candidate imports. Run to regenerate fixtures."""
import json
from pathlib import Path
import mpmath as mp

mp.mp.dps = 80


def solve(a, b):
    return mp.matrix([[mp.lu_solve(a, b[:, j])[i] for j in range(b.cols)]
                      for i in range(a.rows)])


def encode(a):
    if isinstance(a, mp.matrix):
        return [[mp.nstr(a[i, j], 80) for j in range(a.cols)] for i in range(a.rows)]
    return mp.nstr(a, 80)


def case(name, at, ah, c, yt, yh, scales=None):
    nt, nh, p = at.rows, ah.rows, at.cols
    ct, ch, cht = c[:nt, :nt], c[nt:, nt:], c[nt:, :nt]
    precision_a = solve(ct, at)
    covariance = solve(at.T*precision_a, mp.eye(p))
    h = covariance*precision_a.T
    theta = h*yt
    k = ah*h
    cr = ch + k*ct*k.T - cht*k.T - k*cht.T
    rt, rh = yt-at*theta, yh-ah*theta
    return dict(name=name, inputs={key: encode(value) for key, value in
                dict(training_design=at, holdout_design=ah, joint_covariance=c,
                     y_train=yt, y_holdout=yh).items()},
                parameter_scales=scales,
                expected={key: encode(value) for key, value in dict(
                    H=h, parameters=theta, parameter_covariance=covariance,
                    prediction_covariance=ah*covariance*ah.T,
                    holdout_residual_covariance=cr, fitted_training=at*theta,
                    predicted_holdout=ah*theta, training_residual=rt,
                    holdout_residual=rh,
                    training_chi_square=(rt.T*solve(ct, rt))[0],
                    holdout_chi_square=(rh.T*solve(cr, rh))[0]).items()},
                training_dof=nt-p, holdout_dof=nh)


def build():
    at = mp.matrix([[1, -2], [1, 0], [1, 1], [1, 3]])
    ah = mp.matrix([[1, -1], [1, 2]])
    # Rational SPD covariance with nontrivial cross-block correlations.
    u = mp.matrix([[1, 2], [2, -1], [0, 1], [-1, 2], [3, 1], [-2, 3]])/10
    c = mp.diag([mp.mpf(i)/10 for i in [4, 7, 5, 9, 6, 8]])+u*u.T
    yt, yh = mp.matrix([1, 2, 4, 5]), mp.matrix([2, 5])
    cases = [case('correlated_rational', at, ah, c, yt, yh),
             case('diagonal', at, ah, mp.diag([1, 2, 3, 4, 5, 6]), yt, yh),
             case('noiseless', at, ah, c, at*mp.matrix([2, -1]), ah*mp.matrix([2, -1]))]
    order = [2, 0, 3, 1, 5, 4]
    perm = mp.matrix([[int(j == order[i]) for j in range(6)] for i in range(6)])
    all_a, all_y = mp.matrix(list(at.tolist())+list(ah.tolist())), mp.matrix([*yt, *yh])
    a2, y2, c2 = perm*all_a, perm*all_y, perm*c*perm.T
    cases.append(case('reordered', a2[:4, :], a2[4:, :], c2, y2[:4, :], y2[4:, :]))
    row_scale, param_scale = mp.diag([2, 3, 1, 4, 5, 2]), mp.diag([10, mp.mpf('0.1')])
    a2, y2, c2 = row_scale*all_a*param_scale, row_scale*all_y, row_scale*c*row_scale
    cases.append(case('row_units_and_parameter_coordinates', a2[:4, :], a2[4:, :],
                      c2, y2[:4, :], y2[4:, :], ['0.1', '10']))
    x = mp.matrix([1, 2, 3, 4, 5])
    gain_c = mp.eye(5)/100 + mp.mpf('0.16')*x*x.T
    cases.append(case('shared_scalar_gain', x[:3, :], x[3:, :], gain_c,
                      mp.matrix(['2.1', '3.9', '6.2']), mp.matrix(['8.1', '10.3'])))
    gain = cases[-1]['expected']
    h = mp.matrix([[mp.mpf(v) for v in row] for row in gain['H']])
    assert mp.norm(h-x[:3, :].T/14) < mp.mpf('1e-75')
    assert abs(mp.mpf(gain['parameter_covariance'][0][0])-(mp.mpf('0.16')+mp.mpf('0.01')/14)) < mp.mpf('1e-75')
    cr = mp.matrix([[mp.mpf(v) for v in row] for row in gain['holdout_residual_covariance']])
    assert mp.norm(cr-(mp.eye(2)+x[3:, :]*x[3:, :].T/14)/100) < mp.mpf('1e-75')
    negatives = []
    for name, bad_at, bad_c in [
        ('rank_deficient', mp.matrix([[1, 1]]*4), c),
        ('underdetermined', mp.matrix([[1, 0, 0, 0, 0]]*4), c),
        ('singular_joint', at, mp.matrix([[1]*6]*6)),
        ('indefinite_joint', at, mp.diag([1, 1, 1, 1, 1, -1])),
    ]:
        negatives.append(dict(name=name, training_design=encode(bad_at),
                              holdout_design=encode(mp.matrix([[1]*bad_at.cols]*2)),
                              joint_covariance=encode(bad_c)))
    asymmetric = c.copy(); asymmetric[0, 1] += mp.mpf('0.1')
    negatives.append(dict(name='asymmetric_joint', training_design=encode(at),
                          holdout_design=encode(ah), joint_covariance=encode(asymmetric)))
    return dict(precision_decimal_digits=80, method='mpmath lu_solve, tiny normal equations',
                positive_cases=cases, negative_cases=negatives,
                gain_floor=encode(mp.mpf('0.16')),
                gain_cross_discarded_excess=encode(mp.mpf('0.32')*x[3:, :]*x[3:, :].T))


if __name__ == '__main__':
    path = Path(__file__).with_name('oracle_fixtures.json')
    path.write_text(json.dumps(build(), indent=2)+'\n')
    print(f'Wrote {path}: six positive, five rejection fixtures; analytic gain assertions passed')
