"""
PyMeta: A comprehensive Python package for meta-analysis.

This package provides tools for conducting meta-analyses, including:
- Effect size calculations for various data types
- Multiple meta-analytic models (fixed-effect, random-effects, multivariate)
- Publication bias detection and correction methods
- Comprehensive visualization tools
- Living review capabilities
- Command-line interface
"""

from ._version import __version__
from .api.public import *

# Package metadata
__author__ = "PyMeta Development Team"
__email__ = "pymeta@example.com"
__license__ = "Apache-2.0"
__url__ = "https://github.com/mahmood789/Andypymeta"

# Import main API functions for convenience
# These will be implemented in the api.public module
# from .api.public import (
#     meta_analysis,
#     effect_sizes,
#     forest_plot,
#     funnel_plot,
#     egger_test,
#     begg_test,
#     trim_and_fill,
#     load_example_data
# )