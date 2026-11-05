"""Standalone preparation/CLI regressions; run after ``make test``.

Requires Python 3 and NumPy; uses bundled source and fixtures.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from triangular_pm_fields.prepare_cache import PreparedCache


HERE = Path(__file__).resolve().parent
BUILD = Path(os.environ.get('TAN_BUILD_DIR', str(HERE.parent / 'build')))
SOURCE = HERE.parent / 'python/triangular_pm_fields'
RECTANGLE = np.array([[[-1, -1, 0], [1, -1, 0], [1, 1, 0]],
                      [[-1, -1, 0], [1, 1, 0], [-1, 1, 0]]], dtype=float)


class PreparationCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'rectangle.tan'
        self.cache = PreparedCache(RECTANGLE, [1, 1])
        self.cache.write(self.path)

    def evaluate(self, point, mode=1, order=0, status=0):
        result = subprocess.run([str(BUILD / 'tan_cli'), str(self.path),
                                 *map(str, point), str(mode), str(order)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0 if status == 0 else 1,
                         result.stdout + result.stderr)
        self.assertIn(f'status {status}\n', result.stdout)
        if status:
            return None
        rows = [line.split() for line in result.stdout.splitlines()[3:]]
        self.assertEqual(len(rows), math.comb(order + 3, 3))
        return np.array([[float(v) for v in row[4:]] for row in rows])

    def test_preparation_cli_matches_api(self):
        source = self.path.with_suffix('.json')
        source.write_text(json.dumps(dict(triangles=RECTANGLE.tolist(), charges=[1, 1])))
        target = self.path.with_name('from_cli.tan')
        result = subprocess.run([sys.executable, str(SOURCE / 'prepare_cache.py'),
                                 str(source), str(target)], check=True,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(target.read_bytes(), self.path.read_bytes())
        self.assertEqual(json.loads(result.stdout)['boundary_edges'], 4)

    def test_rectangle_axis_matches_solid_angle(self):
        # Uniform square: Omega = 4 atan(a*b/(h*sqrt(h^2+a^2+b^2))).
        expected = [0, 0, math.atan(1 / math.sqrt(3)) / math.pi]
        for mode in (0, 1, 2):
            with self.subTest(mode=mode):
                np.testing.assert_allclose(self.evaluate([0, 0, 1], mode)[0],
                                           expected, rtol=2e-13, atol=2e-15)

    def test_internal_diagonal_is_not_a_physical_edge(self):
        self.assertEqual(self.cache.classify([0, 0, 0]), 'sheet')
        np.testing.assert_allclose(self.evaluate([0, 0, 0])[0], 0, atol=1e-15)
        self.evaluate([0, 0, 0], order=1, status=9)
        self.assertEqual(self.cache.classify([1, 0, 0]), 'edge')
        self.evaluate([1, 0, 0], status=8)

    def test_exact_opposite_charges_cancel_all_support(self):
        cache = PreparedCache(np.concatenate([RECTANGLE, RECTANGLE]), [1, 1, -1, -1])
        self.assertEqual((len(cache.planes), len(cache.edges)), (0, 0))
        self.assertEqual(cache.classify([1, 0, 0]), 'regular')
        cache.write(self.path)
        np.testing.assert_array_equal(self.evaluate([1, 0, 0], order=6), 0)

    def test_winding_does_not_change_physical_charge(self):
        reference = self.evaluate([0.2, 0.3, 0.4], order=2)
        PreparedCache(RECTANGLE[:, ::-1], [1, 1]).write(self.path)
        np.testing.assert_allclose(self.evaluate([0.2, 0.3, 0.4], order=2),
                                   reference, rtol=1e-14, atol=1e-15)

    def test_invalid_inputs_and_unsupported_order_fail(self):
        for triangles, charges in ((RECTANGLE, [1]), (RECTANGLE * np.nan, [1, 1]),
                                   (np.zeros((1, 3, 3)), [1])):
            with self.assertRaises(ValueError):
                PreparedCache(triangles, charges)
        self.evaluate([0, 0, 1], order=7, status=5)
        self.evaluate([0, 0, 1], mode=0, order=1, status=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
