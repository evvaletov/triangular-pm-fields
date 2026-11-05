"""Qualification of the frozen oracle, not tests of any candidate port."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest
import mpmath as mp
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
CORPUS = PACKAGE/'corpus'
SPEC = importlib.util.spec_from_file_location('generate_reference', PACKAGE/'tools/generate_reference.py')
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definition = json.loads((CORPUS/'cases.json').read_text())
        cls.cases = {c['id']: c for c in cls.definition['cases']}
        cls.report = json.loads((CORPUS/'reference.json').read_text())
        cls.results = {r['id']: r for r in cls.report['results']}
        cls.arrays = np.load(CORPUS/'reference.npz')

    @classmethod
    def tearDownClass(cls):
        cls.arrays.close()

    def test_integrity_and_complete_coverage(self):
        for key, path in [('cases_sha256', CORPUS/'cases.json'),
                          ('generator_sha256', PACKAGE/'tools/generate_reference.py'),
                          ('arrays_sha256', CORPUS/'reference.npz')]:
            self.assertEqual(self.report[key], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(set(self.cases), set(self.results))
        self.assertEqual(len(self.cases), 105)
        self.assertTrue(self.report['all_qualified'])
        for id, case in self.cases.items():
            with self.subTest(id=id):
                result = self.results[id]
                self.assertTrue(result['qualified'])
                self.assertEqual(result['status'], case['expected_status'])
                self.assertEqual(GEN.classify(case), case['expected_status'])
                if result['status']:
                    self.assertEqual(result['coefficients'], [])
                    self.assertEqual(result['exponents'], [])
                    continue
                exponents = GEN.indices(case['order'])
                value = np.array(result['coefficients'])
                self.assertEqual(value.shape, (3, math.comb(case['order']+3, 3)))
                self.assertTrue(np.isfinite(value).all())
                self.assertEqual(result['exponents'], [list(x) for x in exponents])
                np.testing.assert_array_equal(value, self.arrays[id+'__reference'])

    def test_all_refinement_and_scalar_controls(self):
        for id, result in self.results.items():
            if result['status']:
                continue
            with self.subTest(id=id):
                refinement = result['refinement']
                self.assertLessEqual(np.max(refinement['scaled_max']), 1e-11)
                for ratio in refinement.get('scalar_budget_ratios', []):
                    self.assertLessEqual(ratio, 1)
                self.assertLessEqual(result.get('constant_crosscheck_budget_ratio', 0), 1)
                case = self.cases[id]
                if case['suite'] == 'taylor':
                    for n in (32, 48, 64):
                        self.assertIn(id+'__gauss_'+str(n), self.arrays.files)
                    ref = self.arrays[id+'__reference']
                    self.assertLessEqual(GEN.scaled_max(self.arrays[id+'__gauss_32'], self.arrays[id+'__gauss_48']), 1e-11)
                    self.assertLessEqual(GEN.scaled_max(self.arrays[id+'__gauss_48'], ref), 1e-11)
                elif case['sigma'] != 0:
                    self.assertEqual(len(result['shared_formula_mp100']), 3)
                    self.assertEqual(len(result['shared_formula_mp140']), 3)
                    self.assertLessEqual(result['precision_refinement_scaled_max'], 1e-11)
                    # Also examine retained decimal values before binary64
                    # projection, which otherwise hides their fine differences.
                    with mp.workdps(160):
                        lo = list(map(mp.mpf, result['shared_formula_mp100']))
                        hi = list(map(mp.mpf, result['shared_formula_mp140']))
                        error = max(abs(a-b) for a, b in zip(lo, hi))
                        self.assertLess(error, mp.mpf('1e-70')*max(1, *map(abs, hi)))

    def test_charge_length_translation_and_winding_invariance(self):
        for q in (1, 2, 4, 6):
            baseline = self.arrays[f'regular_above_q{q}__reference']
            for name in ('translated', 'scaled_small', 'scaled_large', 'reversed'):
                with self.subTest(q=q, name=name):
                    np.testing.assert_allclose(self.arrays[f'{name}_q{q}__reference'], baseline, atol=3e-15, rtol=3e-13)
        for id, case in self.cases.items():
            if case['expected_status'] == 0 and case['sigma'] == 0:
                np.testing.assert_array_equal(self.arrays[id+'__reference'], 0.)
        np.testing.assert_allclose(self.arrays['average_trace__reference'], self.arrays['average_trace_reversed__reference'], atol=1e-15, rtol=0)
        # Corresponding scalar uses +1.25 and Taylor uses -0.75.
        np.testing.assert_allclose(self.arrays['negative_charge_q6__reference'][:, :1], -.6*self.arrays['scalar_negative_charge__reference'], atol=2e-15, rtol=1e-13)

    def test_vacuum_divergence_and_curl_all_retained_degrees(self):
        for id, case in self.cases.items():
            if case['suite'] != 'taylor':
                continue
            q = case['order']
            value = self.arrays[id+'__reference']
            keys = GEN.indices(q)
            lookup = {k: i for i, k in enumerate(keys)}
            for alpha in GEN.indices(q-1):
                derivative = np.zeros((3, 3))
                for axis in range(3):
                    next_alpha = list(alpha)
                    next_alpha[axis] += 1
                    derivative[:, axis] = (alpha[axis]+1)*value[:, lookup[tuple(next_alpha)]]
                with self.subTest(id=id, alpha=alpha):
                    self.assertLess(abs(np.trace(derivative)), 2e-13)
                    np.testing.assert_allclose(derivative, derivative.T, atol=2e-13, rtol=0)

    def test_far_monopole_and_average_trace(self):
        case = self.cases['far_1e6']
        delta = np.array(case['point'])-np.array([1/3, 1/3, 0])
        monopole = case['sigma']*.5/(4*np.pi)*delta/np.linalg.norm(delta)**3
        ref = self.arrays['far_1e6__reference'][:, 0]
        self.assertLess(np.linalg.norm(monopole-ref)/np.linalg.norm(ref), 1e-12)
        trace = self.arrays['average_trace__reference'][:, 0]
        self.assertEqual(trace[2], 0.)
        for control in self.results['average_trace']['independent_trace_controls']:
            above, below = np.array(control['above'], float), np.array(control['below'], float)
            np.testing.assert_allclose((above+below)/2, trace, atol=3e-14, rtol=0)
            self.assertAlmostEqual(above[2]-below[2], self.cases['average_trace']['sigma'], delta=2e-13)

    def test_map_against_displaced_green_integral(self):
        # A fresh scalar quadrature at an off-center observer tests radius powers,
        # factorial normalization and the full graded coefficient interpretation.
        u = np.array([.02, -.03, .01])
        for id in ('regular_above_q6', 'exterior_plane_q6', 'oblique_q6', 'scaled_small_q6'):
            case = self.cases[id]
            exponents = np.array(GEN.indices(6))
            polynomial = self.arrays[id+'__reference'] @ np.prod(u[None, :]**exponents, axis=1)
            point = np.array(case['point'])+case['radius']*u
            integral = GEN.quadrature_jet(case['vertices'], point, 0, 80)[:, 0]*case['sigma']/(4*np.pi)
            with self.subTest(id=id):
                np.testing.assert_allclose(polynomial, integral, atol=3e-14, rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
