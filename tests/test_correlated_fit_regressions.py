"""Independent oracle and range regressions for the correlated fit."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from triangular_pm_fields import CorrelatedLinearFit


class CorrelatedFitRegressionTests(unittest.TestCase):
    def test_destructive_subnormal_scaling_is_rejected(self):
        tiny = np.finfo(float).tiny
        cases = [([[1.5]], [np.nextafter(0., 1.)]),
                 ([[1.5e-10]], [tiny]),
                 ([[tiny]], [tiny]),
                 ([[1., 1.], [1., 1.+1e-8]], [tiny, tiny])]
        for design, scales in cases:
            design = np.asarray(design)
            with self.subTest(design=design.tolist(), scales=scales):
                with self.assertRaisesRegex(ValueError, 'range|underflows'):
                    CorrelatedLinearFit(design, np.ones((1, design.shape[1])),
                                        np.eye(len(design)+1), parameter_scales=scales)

    def test_normal_scale_boundary_preserves_noiseless_fit(self):
        fit = CorrelatedLinearFit([[1.5]], [[1.]], np.eye(2),
                                  parameter_scales=[np.finfo(float).tiny])
        np.testing.assert_allclose(fit.fit([1.5])['parameters'], [1.], rtol=1e-14)
        np.testing.assert_allclose(fit.parameter_covariance, [[4./9.]], rtol=1e-14)

    def test_overflowed_prediction_is_rejected(self):
        fit = CorrelatedLinearFit([[1.]], [[1e200]], np.eye(2)*1e-300)
        with np.errstate(over='ignore', invalid='ignore'):
            with self.assertRaisesRegex(ValueError, 'predicted_holdout'):
                fit.fit([1e200])

    def test_underflowed_parameter_variance_is_rejected(self):
        # The exact parameter variance is 1e-400, which float64 cannot store.
        # Returning zero would also erase a representable 1e-200 prediction variance.
        with self.assertRaisesRegex(ValueError, 'underflows'):
            CorrelatedLinearFit([[1e100]], [[1e100]], np.eye(2)*1e-200)

    def test_result_mapping_only_exposes_result_fields(self):
        result = CorrelatedLinearFit([[1.]], [[1.]], np.eye(2)).fit([2.])
        self.assertEqual(set(result), {
            'parameters', 'fitted_training', 'predicted_holdout',
            'training_residual', 'training_chi_square', 'training_dof',
        })
        for key in ('__class__', 'items', 'missing', 'holdout_residual'):
            with self.subTest(key=key), self.assertRaises(KeyError):
                result[key]

    def test_independent_frozen_high_precision_fixtures(self):
        reference = Path(__file__).resolve().parents[1]/'validation/correlated'
        spec = importlib.util.spec_from_file_location('fixed_gls_oracle', reference/'check_oracle.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.check(CorrelatedLinearFit, json.loads((reference/'oracle_fixtures.json').read_text()))
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['positive_cases']), 6)
        self.assertEqual(len(report['rejected_cases']), 5)


if __name__ == '__main__':
    unittest.main()
