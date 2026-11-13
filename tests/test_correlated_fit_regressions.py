"""Independent oracle and range regressions for the correlated fit."""
import importlib.util
import copy
import json
import pickle
from pathlib import Path
import unittest

import numpy as np

from triangular_pm_fields import CorrelatedLinearFit
from triangular_pm_fields.correlated_fit import CorrelatedLinearFitResult


class CorrelatedFitRegressionTests(unittest.TestCase):
    def test_prepared_array_metadata_cannot_change_future_fits(self):
        model = CorrelatedLinearFit([[1.], [2.]], [[3.], [4.]], np.eye(4))
        expected = model.evaluate_holdout([2., 4.], [6., 8.])
        names = ('training_design', 'holdout_design', 'joint_covariance',
                 'parameter_scales', 'estimator', 'parameter_covariance',
                 'prediction_covariance', 'holdout_residual_covariance',
                 'singular_values')
        for name in names:
            original = getattr(model, name).copy()
            with self.subTest(name=name):
                exposed = getattr(model, name)
                exposed.dtype = np.int64
                exposed.shape = (exposed.size,)
                exposed.base.dtype = np.uint8
                np.testing.assert_array_equal(getattr(model, name), original)
                self.assertEqual(getattr(model, name).dtype, np.dtype('float64'))
                self.assertEqual(getattr(model, name).shape, original.shape)
                actual = model.evaluate_holdout([2., 4.], [6., 8.])
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key])

    def test_result_metadata_and_input_aliases_cannot_change_saved_values(self):
        result = CorrelatedLinearFit([[1.], [2.]], [[3.], [4.]], np.eye(4))
        result = result.evaluate_holdout([2., 4.], [6., 8.])
        for name in ('parameters', 'fitted_training', 'predicted_holdout',
                     'training_residual', 'holdout_residual'):
            original = result[name].copy()
            with self.subTest(name=name):
                exposed = result[name]
                exposed.dtype = np.int64
                exposed.shape = (1, exposed.size)
                exposed.base.dtype = np.uint8
                np.testing.assert_array_equal(result[name], original)
                self.assertEqual(getattr(result, name).dtype, np.dtype('float64'))
                self.assertEqual(getattr(result, name).shape, original.shape)
        supplied = np.array([2.])
        direct = CorrelatedLinearFitResult(supplied, supplied, supplied,
                                          supplied, 0., 0)
        supplied[0] = 7.
        np.testing.assert_array_equal(direct.parameters, [2.])
        with self.assertRaises(ValueError):
            direct.parameters.flags.writeable = True

    def test_result_copy_and_pickle_preserve_immutable_snapshots(self):
        result = CorrelatedLinearFit([[1.], [2.]], [[3.], [4.]], np.eye(4))
        result = result.evaluate_holdout([2., 4.], [6., 8.])
        for restored in (copy.copy(result), copy.deepcopy(result),
                         pickle.loads(pickle.dumps(result))):
            for name in result:
                np.testing.assert_array_equal(restored[name], result[name])
                if isinstance(restored[name], np.ndarray):
                    exposed = restored[name]
                    with self.assertRaises(ValueError):
                        exposed.flags.writeable = True
                    exposed.dtype = np.int64
                    exposed.shape = (1, exposed.size)
                    exposed.base.dtype = np.uint8
                    np.testing.assert_array_equal(restored[name], result[name])

    def test_small_channels_cannot_hide_asymmetric_covariance(self):
        covariance = np.diag([1e12, 1e-12, 1e-12])
        covariance[1, 2] = 5e-13
        for units in (np.ones(3), np.array([1e-6, 1e6, 1e6])):
            scaled = covariance * units[:, None] * units[None, :]
            with self.subTest(units=units), self.assertRaisesRegex(ValueError, 'asymmetry'):
                CorrelatedLinearFit([[1.], [1.]], [[1.]], scaled)

    def test_subnormal_variances_and_erased_prediction_are_rejected(self):
        cases = (
            ([[3e161]], [[1.]], np.eye(2), 'parameter covariance'),
            ([[1e150]], [[1e-20]], np.eye(2), 'prediction_covariance'),
            ([[1.]], [[0.]], np.diag([1., 1e-320]), 'holdout residual covariance'),
        )
        for training, holdout, covariance, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                CorrelatedLinearFit(training, holdout, covariance)
        model = CorrelatedLinearFit([[1.]], [[0.]], np.eye(2))
        np.testing.assert_array_equal(model.prediction_covariance, [[0.]])
        tiny = np.finfo(float).tiny
        boundary = CorrelatedLinearFit([[1.]], [[np.sqrt(tiny)]], np.eye(2))
        np.testing.assert_array_equal(boundary.prediction_covariance, [[tiny]])

    def test_result_equality_handles_array_values(self):
        model = CorrelatedLinearFit([[1., 0.], [0., 1.]], [[1., 1.]], np.eye(3))
        first = model.fit([2., 3.])
        self.assertEqual(first, model.fit([2., 3.]))
        self.assertEqual(first, dict(first))
        self.assertNotEqual(first, model.fit([2., 4.]))
        self.assertNotEqual(first, None)

    def test_accepted_ill_conditioned_design_against_high_precision_qr(self):
        from test_correlated_fit import high_precision_gls
        spacing = 2.**-40
        training = np.array([[1., 1.], [1., 1.+spacing], [1., 1.-spacing]])
        observations = np.array([1., 2., 3.])
        model = CorrelatedLinearFit(training, np.eye(2), np.eye(5))
        theta, covariance = high_precision_gls(training, observations, np.eye(3), dps=80)
        self.assertGreater(model.condition_number, 1e12)
        # A condition-scaled check for this declared matrix, not a universal bound.
        tolerance = 8 * model.condition_number * np.finfo(float).eps
        np.testing.assert_allclose(model.fit(observations).parameters, theta, rtol=tolerance)
        np.testing.assert_allclose(model.parameter_covariance, covariance, rtol=tolerance)

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
