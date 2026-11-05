"""Triangular permanent-magnet field preparation and native evaluation."""
from .prepare_cache import PreparedCache
from .native import NativeLibrary, NativeError
__version__ = "0.1.0"
__all__ = ["PreparedCache", "NativeLibrary", "NativeError"]
