"""Python ABI layout, global-context isolation, and failed-load regressions."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from triangular_pm_fields import NativeLibrary, NativeError, PreparedCache
from triangular_pm_fields.native import exponents


class PythonNative(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [self.root/'one.tan', self.root/'two.tan']
        for scale, path in enumerate(self.paths,1):
            PreparedCache([[[0,0,0],[1,0,0],[0,1,0]]],[scale]).write(path)
        build = Path(os.environ.get('TAN_BUILD_DIR', str(Path(__file__).resolve().parents[1]/'build')))
        self.libraries = [NativeLibrary(build/'libtan.so') for _ in range(2)]
        self.points = [[.25,.25,.75], [2.,3.,0.]]

    def test_layout_and_radius_powers(self):
        lib = self.libraries[0]
        a = lib.evaluate(self.paths[0],self.points,order=6,radius=.125)
        b = lib.evaluate(self.paths[0],self.points,order=6,radius=.25)
        self.assertEqual(a.shape,(2,3,84))
        np.testing.assert_allclose(b,a*np.array([2.**sum(e) for e in exponents(6)]),rtol=1e-13,atol=1e-15)
        np.testing.assert_allclose(a[:,:,0],lib.evaluate(self.paths[0],self.points)[:,:,0],rtol=1e-13,atol=1e-15)

    def test_alternating_and_threaded_contexts(self):
        expected = self.libraries[0].evaluate(self.paths[0],self.points)
        def query(i):
            factor = i%2+1
            actual = self.libraries[i%2].evaluate(self.paths[i%2],self.points)
            np.testing.assert_allclose(actual,factor*expected,rtol=1e-13,atol=1e-15)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(query,range(40)))

    def test_failed_load_then_recovery(self):
        lib = self.libraries[0]
        expected = lib.evaluate(self.paths[0],self.points)
        with self.assertRaises(NativeError) as context:
            lib.evaluate(self.root/'missing.tan',self.points)
        self.assertEqual(context.exception.status,2)
        np.testing.assert_array_equal(lib.evaluate(self.paths[0],self.points),expected)

    def test_input_errors_and_boundary_status(self):
        lib = self.libraries[0]
        for options in ({'order':7},{'order':True},{'order':1.5},{'mode':3},
                        {'mode':0,'order':1},{'radius':0},{'radius':float('nan')}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                lib.evaluate(self.paths[0],self.points,**options)
        for points in ([1,2,3],[[1,2]],[[0,float('inf'),0]]):
            with self.assertRaises(ValueError):
                lib.evaluate(self.paths[0],points)
        for point,order,status in (([.5,0,0],0,8),([.25,.25,0],1,9)):
            with self.assertRaises(NativeError) as context:
                lib.evaluate(self.paths[0],[point],order=order)
            self.assertEqual(context.exception.status,status)


if __name__=='__main__':
    unittest.main()
