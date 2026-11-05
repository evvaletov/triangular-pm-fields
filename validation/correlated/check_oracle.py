"""Portable comparison against the committed independent 80-digit fixtures.

python check_oracle.py --candidate /path/to/correlated_fit.py --output result.json
No mpmath is needed to check the committed JSON; only NumPy and the candidate.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def load_candidate(path):
    spec = importlib.util.spec_from_file_location('correlated_fit_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.CorrelatedLinearFit


def arr(value):
    return np.asarray(value, dtype=np.float64)


def check(cls, fixtures):
    checked = []
    vector_keys = {'parameters', 'fitted_training', 'predicted_holdout',
                   'training_residual', 'holdout_residual'}
    attribute_keys = {'H', 'parameter_covariance', 'prediction_covariance',
                      'holdout_residual_covariance'}
    for case in fixtures['positive_cases']:
        inputs = {k: arr(v) for k, v in case['inputs'].items()}
        yt, yh = inputs.pop('y_train').ravel(), inputs.pop('y_holdout').ravel()
        if case['parameter_scales'] is not None:
            inputs['parameter_scales'] = arr(case['parameter_scales'])
        fit = cls(**inputs)
        result = fit.evaluate_holdout(yt, yh)
        fit_only = fit.fit(yt)
        for key, values in case['expected'].items():
            expected = arr(values)
            if key in vector_keys:
                expected = expected.ravel()
            attribute = 'estimator' if key == 'H' else key
            actual = getattr(fit, attribute) if key in attribute_keys else result[key]
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12,
                                       err_msg=f'{case["name"]}: {key}')
            if key not in attribute_keys and key in fit_only:
                np.testing.assert_allclose(fit_only[key], expected, rtol=2e-11, atol=2e-12)
        assert result['training_dof'] == case['training_dof']
        assert result['holdout_dof'] == case['holdout_dof']
        assert fit.rank == inputs['training_design'].shape[1]
        checked.append(case['name'])
    rejected = []
    for case in fixtures['negative_cases']:
        try:
            cls(**{k: arr(v) for k, v in case.items() if k != 'name'})
        except (ValueError, np.linalg.LinAlgError):
            rejected.append(case['name'])
        else:
            raise AssertionError(f'Accepted negative control {case["name"]}')
    # Keep each marginal block; deleting only cross-block covariance must change
    # the gain toy residual covariance by the independently derived exact term.
    toy = next(x for x in fixtures['positive_cases'] if x['name'] == 'shared_scalar_gain')
    inputs = {k: arr(v) for k, v in toy['inputs'].items() if k not in ('y_train', 'y_holdout')}
    c = inputs['joint_covariance'].copy(); c[:3, 3:] = 0; c[3:, :3] = 0
    inputs['joint_covariance'] = c
    altered = cls(**inputs)
    excess = altered.holdout_residual_covariance-arr(toy['expected']['holdout_residual_covariance'])
    np.testing.assert_allclose(excess, arr(fixtures['gain_cross_discarded_excess']), rtol=2e-11, atol=2e-12)
    return dict(passed=True, positive_cases=checked, rejected_cases=rejected,
                shared_gain_cross_covariance_negative_control=True,
                gain_parameter_variance_floor=float(fixtures['gain_floor']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).with_name('oracle_fixtures.json'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    result = check(load_candidate(args.candidate), fixtures)
    result.update(candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256(args.fixtures.read_bytes()).hexdigest())
    rendered = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')
