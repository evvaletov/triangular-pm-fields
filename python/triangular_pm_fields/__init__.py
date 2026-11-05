"""Triangular permanent-magnet field preparation and native evaluation."""
from .prepare_cache import PreparedCache
from .native import NativeLibrary, NativeError
from .correlated_fit import CorrelatedLinearFit
__version__ = "0.2.0"
__all__ = ["PreparedCache", "NativeLibrary", "NativeError", "CorrelatedLinearFit"]
