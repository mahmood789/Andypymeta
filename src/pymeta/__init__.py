"""
pymeta: A comprehensive meta-analysis toolkit for Python

A modern Python package for conducting meta-analyses with support for:
- Fixed and random effects models
- Binary and continuous outcomes
- Bias detection and correction
- Visualization tools
- Bayesian approaches
- Robust variance estimation
"""

__version__ = "0.0.1"
__author__ = "pymeta contributors"

# Core functionality imports
from .effects import binary_effects
from .models import fixed_effects, random_effects
from .estimators import tau2_estimators
from .inference import prediction_intervals
from .bias import egger_test
from .viz import forest_plot, funnel_plot

__all__ = [
    "binary_effects",
    "fixed_effects", 
    "random_effects",
    "tau2_estimators",
    "prediction_intervals",
    "egger_test",
    "forest_plot",
    "funnel_plot",
]