"""Prepare one charged triangle, then evaluate its scalar field and q=6 map."""
import argparse
from pathlib import Path
from triangular_pm_fields import NativeLibrary, PreparedCache

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--library', type=Path, default=Path('build/libtan.so'))
parser.add_argument('--cache', type=Path, default=Path('triangle.tan'))
args = parser.parse_args()
PreparedCache([[[0,0,0],[1,0,0],[0,1,0]]],[1.25]).write(args.cache)
library = NativeLibrary(args.library)
coefficients = library.evaluate(args.cache, [[.25,.25,.75]], order=6, radius=.125)
print('mu0 H [T]:', coefficients[0,:,0])
print('Taylor array shape:', coefficients.shape)
print('Cache:', args.cache)
