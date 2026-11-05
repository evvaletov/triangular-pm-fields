"""Exact-preparation regressions against the pinned baseline constructor.

Compares canonical TAN_CACHE_V1 bytes and classifications. Run from this
directory after building the native library. Does not modify the baseline module.
"""
from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np

from triangular_pm_fields.prepare_cache import PreparedCache


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE_PATH = HERE / 'baseline_prepare_cache.py'
SEED = 20261104
RECTANGLE = np.array([[[-1, -1, 0], [1, -1, 0], [1, 1, 0]],
                      [[-1, -1, 0], [1, 1, 0], [-1, 1, 0]]], dtype=np.float64)


def load_baseline_class():
    spec = importlib.util.spec_from_file_location('baseline_prepare_cache', BASELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PreparedCache


BaselinePreparedCache = load_baseline_class()


def rectangle_mesh(n, *, bounds=(-1.0, 1.0, -0.75, 0.75), z=0.0, sigma=0.8, diagonal=0):
    xs = np.linspace(bounds[0], bounds[1], n + 1, dtype=np.float64)
    ys = np.linspace(bounds[2], bounds[3], n + 1, dtype=np.float64)
    triangles = []
    for i in range(n):
        for j in range(n):
            a = np.array([xs[i], ys[j], z], dtype=np.float64)
            b = np.array([xs[i + 1], ys[j], z], dtype=np.float64)
            c = np.array([xs[i + 1], ys[j + 1], z], dtype=np.float64)
            d = np.array([xs[i], ys[j + 1], z], dtype=np.float64)
            if diagonal == 0:
                triangles.extend(([a, b, c], [a, c, d]))
            else:
                triangles.extend(([a, b, d], [b, c, d]))
    tri = np.array(triangles, dtype=np.float64)
    return tri, np.full(len(tri), sigma, dtype=np.float64)


def compare_caches(optimized, baseline, points=()):
    if optimized.text != baseline.text:
        return 'text mismatch'
    if optimized.sha256 != baseline.sha256:
        return 'sha256 mismatch'
    if len(optimized.planes) != len(baseline.planes):
        return 'plane count mismatch'
    if len(optimized.edges) != len(baseline.edges):
        return 'edge count mismatch'
    if optimized.metadata() != baseline.metadata():
        return 'metadata mismatch'
    for point in points:
        got = optimized.classify(point)
        expected = baseline.classify(point)
        if got != expected:
            return f'classify {point}: {got} != {expected}'
    return None


def sample_points():
    coords = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    points = []
    for x in coords:
        for y in coords:
            for z in (0.0, 1.0, -1.0, np.nextafter(0.0, 1.0)):
                points.append((x, y, z))
    return points


class OptimizedMatchesBaseline(unittest.TestCase):
    points = sample_points()

    def assert_match(self, triangles, charges, *, coalesce=True, points=None):
        optimized = PreparedCache(triangles, charges, coalesce=coalesce)
        baseline = BaselinePreparedCache(triangles, charges, coalesce=coalesce)
        mismatch = compare_caches(optimized, baseline,
                                  self.points if points is None else points)
        self.assertIsNone(mismatch, mismatch)
        return optimized, baseline

    def test_baseline_module_is_the_pinned_file(self):
        self.assertTrue(BASELINE_PATH.is_file())
        self.assertNotEqual(BASELINE_PATH.resolve(), (HERE / 'prepare_cache.py').resolve())
        self.assertEqual(BaselinePreparedCache.__module__, 'baseline_prepare_cache')

    def test_rectangle_known_chain_both_coalesce(self):
        for coalesce, n_edges in ((True, 4), (False, 4)):
            with self.subTest(coalesce=coalesce):
                cache, _ = self.assert_match(RECTANGLE, [1, 1], coalesce=coalesce)
                self.assertEqual((len(cache.planes), len(cache.edges)), (1, n_edges))
                self.assertEqual(cache.classify([0, 0, 0]), 'sheet')
                self.assertEqual(cache.classify([1, 0, 0]), 'edge')
                self.assertEqual(cache.classify([0, 0, 1]), 'regular')

    def test_refined_rectangle_meshes(self):
        for n in (1, 2, 4, 8):
            for diagonal in (0, 1):
                for coalesce in (True, False):
                    with self.subTest(n=n, diagonal=diagonal, coalesce=coalesce):
                        tri, charges = rectangle_mesh(n, diagonal=diagonal)
                        cache, _ = self.assert_match(tri, charges, coalesce=coalesce)
                        self.assertEqual(len(cache.planes), 1)
                        if coalesce:
                            self.assertEqual(len(cache.edges), 4)
                        else:
                            self.assertEqual(len(cache.edges), 4 * n)

    def test_overlap_identical_faces(self):
        tri = np.concatenate([RECTANGLE, RECTANGLE])
        charges = np.array([1, 1, 1, 1], dtype=np.float64)
        cache, _ = self.assert_match(tri, charges)
        self.assertEqual((len(cache.planes), len(cache.edges)), (1, 4))
        self.assertEqual(cache.classify([0, 0, 0]), 'sheet')

    def test_t_junction_splits_surviving_edge(self):
        large = np.array([[[0, 0, 0], [4, 0, 0], [0, 4, 0]]], dtype=np.float64)
        small = np.array([[[1, 0, 0], [2, 0, 0], [1, 1, 0]]], dtype=np.float64)
        tri = np.concatenate([large, small])
        for coalesce in (True, False):
            with self.subTest(coalesce=coalesce):
                cache, _ = self.assert_match(tri, [1, 1], coalesce=coalesce)
                self.assertEqual(cache.classify([1, 0, 0]), 'edge')
                self.assertEqual(cache.classify([2, 0, 0]), 'edge')
                self.assertEqual(cache.classify([0.5, 0, 0]), 'edge')
                self.assertEqual(cache.classify([3, 0, 0]), 'edge')

    def test_canceled_raw_edge_still_supplies_t_junction_vertex(self):
        # Opposite copies of (0,0)-(2,0)-(1,1) cancel. The surviving large
        # triangle does not include (2,0,0); that vertex must still split
        # (0,0,0)-(4,0,0) when coalesce is off.
        canceled = np.array([[[0, 0, 0], [2, 0, 0], [1, 1, 0]],
                             [[0, 0, 0], [2, 0, 0], [1, 1, 0]]], dtype=np.float64)
        large = np.array([[[0, 0, 0], [4, 0, 0], [0, 4, 0]]], dtype=np.float64)
        tri = np.concatenate([canceled, large])
        charges = np.array([1.0, -1.0, 1.0], dtype=np.float64)
        for coalesce in (True, False):
            with self.subTest(coalesce=coalesce):
                cache, _ = self.assert_match(tri, charges, coalesce=coalesce)
                self.assertEqual(cache.classify([2, 0, 0]), 'edge')
                if not coalesce:
                    vertices = {(e[1], e[2]) for e in cache.edges}
                    self.assertTrue(any(p[0] == 2 and p[1] == 0 and p[2] == 0
                                        for edge in vertices for p in edge))

    def test_hole_signed_winding(self):
        outer, outer_c = rectangle_mesh(1, bounds=(-2, 2, -2, 2), sigma=1.0)
        inner, inner_c = rectangle_mesh(1, bounds=(-1, 1, -1, 1), sigma=-1.0)
        tri = np.concatenate([outer, inner])
        charges = np.concatenate([outer_c, inner_c])
        for coalesce in (True, False):
            with self.subTest(coalesce=coalesce):
                cache, _ = self.assert_match(tri, charges, coalesce=coalesce)
                self.assertEqual(len(cache.planes), 1)
                self.assertEqual(len(cache.edges), 8)
                self.assertEqual(cache.classify([0, 0, 0]), 'regular')
                self.assertEqual(cache.classify([1.5, 0, 0]), 'sheet')
                self.assertEqual(cache.classify([1, 0, 0]), 'edge')
                self.assertEqual(cache.classify([2, 0, 0]), 'edge')

    def test_unequal_and_reversed_winding(self):
        reversed_rect = RECTANGLE[:, ::-1]
        for coalesce in (True, False):
            with self.subTest(kind='all-reversed', coalesce=coalesce):
                self.assert_match(reversed_rect, [1, 1], coalesce=coalesce)
            mixed = np.stack([RECTANGLE[0], reversed_rect[1]])
            with self.subTest(kind='mixed-reverse', coalesce=coalesce):
                self.assert_match(mixed, [1, 1], coalesce=coalesce)
            with self.subTest(kind='unequal-charge', coalesce=coalesce):
                self.assert_match(RECTANGLE, [1.0, -0.25], coalesce=coalesce)

    def test_tiny_charges(self):
        tiny = 2.0 ** -60
        for coalesce in (True, False):
            with self.subTest(kind='same-sign', coalesce=coalesce):
                cache, _ = self.assert_match(RECTANGLE, [tiny, tiny], coalesce=coalesce)
                self.assertEqual(len(cache.planes), 1)
                self.assertEqual(cache.classify([0, 0, 0]), 'sheet')
            with self.subTest(kind='canceled-copies', coalesce=coalesce):
                doubled = np.concatenate([RECTANGLE, RECTANGLE])
                cache, _ = self.assert_match(doubled, [tiny, tiny, -tiny, -tiny],
                                            coalesce=coalesce)
                self.assertEqual((len(cache.planes), len(cache.edges)), (0, 0))
                self.assertEqual(cache.classify([0, 0, 0]), 'regular')
            with self.subTest(kind='opposite-halves', coalesce=coalesce):
                cache, _ = self.assert_match(RECTANGLE, [tiny, -tiny], coalesce=coalesce)
                self.assertEqual(len(cache.planes), 1)

    def test_single_ulp_plane_separation(self):
        z = np.nextafter(0.0, 1.0)
        lower, lower_c = rectangle_mesh(1, z=0.0, sigma=1.0)
        upper, upper_c = rectangle_mesh(1, z=z, sigma=1.0)
        tri = np.concatenate([lower, upper])
        charges = np.concatenate([lower_c, upper_c])
        for coalesce in (True, False):
            with self.subTest(coalesce=coalesce):
                cache, _ = self.assert_match(tri, charges, coalesce=coalesce)
                self.assertEqual(len(cache.planes), 2)
                self.assertEqual(cache.classify([0, 0, 0]), 'sheet')
                self.assertEqual(cache.classify([0, 0, z]), 'sheet')

    def test_zero_charge_degeneracy_and_skip(self):
        degenerate = np.zeros((1, 3, 3), dtype=np.float64)
        for cls in (PreparedCache, BaselinePreparedCache):
            with self.subTest(cls=cls.__module__):
                with self.assertRaises(ValueError):
                    cls(degenerate, [0.0])
                with self.assertRaises(ValueError):
                    cls(degenerate, [1.0])
        valid_zero = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
        cache, _ = self.assert_match(valid_zero, [0.0])
        self.assertEqual((len(cache.planes), len(cache.edges)), (0, 0))
        mixed = np.concatenate([RECTANGLE, valid_zero])
        self.assert_match(mixed, [1.0, 1.0, 0.0])

    def test_empty_arrays(self):
        for triangles, charges in (
                (np.empty((0, 3, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)),
                ([], []),
        ):
            with self.subTest(type=type(triangles).__name__):
                cache, _ = self.assert_match(triangles, charges)
                self.assertEqual(cache.text, 'TAN_CACHE_V1\n0 0\n')
                self.assertEqual(cache.classify([0, 0, 0]), 'regular')

    def test_negative_and_canceled_weights(self):
        doubled = np.concatenate([RECTANGLE, RECTANGLE])
        cache, _ = self.assert_match(doubled, [1, 1, -1, -1])
        self.assertEqual((len(cache.planes), len(cache.edges)), (0, 0))
        self.assertEqual(cache.classify([1, 0, 0]), 'regular')
        self.assert_match(RECTANGLE, [-1, -1])
        self.assert_match(np.concatenate([RECTANGLE, RECTANGLE]), [0.5, 0.5, -0.25, -0.25])

    def test_invalid_inputs_match_baseline(self):
        cases = (
            (RECTANGLE, [1]),
            (RECTANGLE * np.nan, [1, 1]),
            (np.zeros((1, 3, 3)), [1]),
            (RECTANGLE, [np.inf, 1]),
            (np.ones((2, 3)), [1, 1]),
        )
        for triangles, charges in cases:
            with self.subTest(shape=np.shape(triangles)):
                with self.assertRaises(ValueError):
                    PreparedCache(triangles, charges)
                with self.assertRaises(ValueError):
                    BaselinePreparedCache(triangles, charges)

    def test_seeded_adversarial_corpus(self):
        rng = np.random.default_rng(SEED)
        mismatches = []
        compared = 0
        for trial in range(40):
            n_tri = int(rng.integers(1, 8))
            if trial % 2 == 0:
                pts = rng.integers(-6, 7, size=(n_tri, 3, 3)).astype(np.float64)
                charges = rng.choice([-2.0, -1.0, -0.5, 0.0, 0.25, 1.0, 2.0], size=n_tri)
                # Force some shared-plane structure.
                pts[:, :, 2] = 0.0
            else:
                pts = rng.integers(-4, 5, size=(n_tri, 3, 3)).astype(np.float64)
                charges = rng.standard_normal(n_tri).astype(np.float64)
            coalesce = bool(trial % 3)
            try:
                optimized = PreparedCache(pts, charges, coalesce=coalesce)
                baseline = BaselinePreparedCache(pts, charges, coalesce=coalesce)
            except ValueError:
                with self.assertRaises(ValueError):
                    PreparedCache(pts, charges, coalesce=coalesce)
                with self.assertRaises(ValueError):
                    BaselinePreparedCache(pts, charges, coalesce=coalesce)
                continue
            compared += 1
            mismatch = compare_caches(optimized, baseline, self.points)
            if mismatch:
                mismatches.append((trial, mismatch))
        self.assertGreaterEqual(compared, 10)
        self.assertEqual(mismatches, [])

    def test_fine_mesh_n16_and_disjoint_m64_match_baseline(self):
        tri, charges = rectangle_mesh(16)
        cache, _ = self.assert_match(tri, charges, coalesce=True)
        self.assertEqual((len(cache.planes), len(cache.edges)), (1, 4))
        squares = []
        sigmas = []
        for i in range(64):
            x0 = float(i % 8)
            y0 = float(i // 8)
            z = float(i)
            squares.append([[x0, y0, z], [x0 + 1, y0, z], [x0 + 1, y0 + 1, z]])
            squares.append([[x0, y0, z], [x0 + 1, y0 + 1, z], [x0, y0 + 1, z]])
            sigmas.extend([0.5, 0.5])
        tri64 = np.array(squares, dtype=np.float64)
        charges64 = np.array(sigmas, dtype=np.float64)
        cache64, _ = self.assert_match(tri64, charges64, coalesce=True)
        self.assertEqual(len(cache64.planes), 64)
        self.assertEqual(len(cache64.edges), 256)

    def test_attrs_write_and_classify_remain(self):
        cache = PreparedCache(RECTANGLE, [1, 1])
        self.assertTrue(hasattr(cache, 'triangles'))
        self.assertTrue(hasattr(cache, 'charges'))
        self.assertTrue(hasattr(cache, 'planes'))
        self.assertTrue(hasattr(cache, 'edges'))
        self.assertTrue(hasattr(cache, 'text'))
        self.assertTrue(hasattr(cache, 'sha256'))
        self.assertEqual(cache.text[:12], 'TAN_CACHE_V1')
        metadata = cache.metadata()
        self.assertEqual(metadata['input_triangles'], 2)
        self.assertEqual(metadata['exact_setup'], True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
