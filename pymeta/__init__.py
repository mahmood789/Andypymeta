"""PyMeta: A comprehensive modular meta-analysis package.

Version 4.1-modular - Complete modular structure with registries.
"""

__version__ = "4.1-modular"
__author__ = "PyMeta Development Team"

# Core imports
from .config import config, MetaAnalysisConfig, PlotStyle
from .errors import (
    PyMetaError, DataError, ConvergenceError, ModelError,
    EstimationError, ValidationError, PlottingError,
    BiasTestError, TSAError, RegistryError
)
from .typing import MetaPoint, MetaResults, BiasTestResult, TSAResult
from .registries import (
    register_estimator, register_model, register_plot, register_bias_test,
    get_estimator, get_model, get_plot, get_bias_test,
    list_estimators, list_models, list_plots, list_bias_tests
)

# Import submodules to trigger registrations
from . import estimators
from . import models
from . import effects
from . import bias
from . import plots
from . import tsa

# Main suite
from .suite import PyMeta

# Import key model classes for direct access
from .models.fixed_effects import FixedEffects
from .models.random_effects import RandomEffects
from .models.glmm_binomial import GLMMBinomial

# High-level API exports
__all__ = [
    # Version and metadata
    '__version__',
    
    # Core classes and configuration
    'config', 'MetaAnalysisConfig', 'PlotStyle',
    'MetaPoint', 'MetaResults', 'BiasTestResult', 'TSAResult',
    
    # Exceptions
    'PyMetaError', 'DataError', 'ConvergenceError', 'ModelError',
    'EstimationError', 'ValidationError', 'PlottingError',
    'BiasTestError', 'TSAError', 'RegistryError',
    
    # Registry functions
    'register_estimator', 'register_model', 'register_plot', 'register_bias_test',
    'get_estimator', 'get_model', 'get_plot', 'get_bias_test',
    'list_estimators', 'list_models', 'list_plots', 'list_bias_tests',
    
    # Main suite
    'PyMeta',
    
    # Model classes
    'FixedEffects', 'RandomEffects', 'GLMMBinomial',
    
    # Submodules
    'estimators', 'models', 'effects', 'bias', 'plots', 'tsa'
]


def about():
    """Print information about PyMeta."""
    print(f"""
PyMeta {__version__}
====================

A comprehensive modular meta-analysis package for Python.

Features:
- Multiple tau² estimators: {', '.join(list_estimators())}
- Model types: {', '.join(list_models())}
- Bias tests: {', '.join(list_bias_tests())}
- Plot types: {', '.join(list_plots())}
- Trial Sequential Analysis (TSA)
- GLMM Binomial models with graceful fallback
- Publication-quality plotting with multiple styles
- Command-line interface
- Streamlit GUI application

Documentation: https://pymeta.readthedocs.io
Source: https://github.com/mahmood789/Andypymeta
""")


def quick_start():
    """Quick start guide."""
    print("""
Quick Start Guide
=================

1. Basic Meta-Analysis:
   ```python
   import pymeta
   
   # Create data points
   points = [
       pymeta.MetaPoint(effect=0.5, variance=0.1, label="Study 1"),
       pymeta.MetaPoint(effect=0.3, variance=0.15, label="Study 2"),
   ]
   
   # Perform analysis
   meta = pymeta.PyMeta(points)
   results = meta.analyze()
   print(results.summary_dict)
   ```

2. With Plotting:
   ```python
   meta.plot_forest()
   meta.plot_funnel()
   meta.plot_baujat()
   ```

3. Bias Detection:
   ```python
   bias_result = meta.test_bias('egger')
   print(f"Egger test p-value: {bias_result.p_value}")
   ```

4. Trial Sequential Analysis:
   ```python
   tsa_result = meta.perform_tsa(delta=0.2)
   meta.plot_tsa(tsa_result)
   ```

For more examples, see the documentation.
""")


# Set default plot style on import
config.set_plot_style('default')