"""Andypymeta - A Python metadata library with comprehensive testing and debugging.

This package provides utilities for handling metadata in Python applications
with built-in testing and debugging capabilities.
"""

__version__ = "0.1.0"
__author__ = "Andypymeta Team"
__license__ = "Apache-2.0"

from .core import MetadataHandler
from .debug import DebugManager
from .utils import get_version, validate_metadata

__all__ = [
    "MetadataHandler",
    "DebugManager",
    "get_version",
    "validate_metadata",
]
