"""
PyMeta - Comprehensive Meta-Analysis Package

A Python package for meta-analysis with advanced features including:
- HKSJ (Hartung-Knapp-Sidik-Jonkman) variance adjustment
- Influence diagnostics and leave-one-out analysis
- Contour-enhanced funnel plots
- Living meta-analysis with scheduling
"""

from dataclasses import dataclass
from typing import Optional, List, Union
import numpy as np
import pandas as pd

__version__ = "0.1.0"

# Core data structures
@dataclass
class MetaPoint:
    """Represents a single study point in meta-analysis."""
    effect: float
    variance: float
    study_id: str
    n: Optional[int] = None
    
    @property
    def se(self) -> float:
        """Standard error of the effect."""
        return np.sqrt(self.variance)
    
    @property
    def weight(self) -> float:
        """Weight for fixed effects model."""
        return 1.0 / self.variance if self.variance > 0 else 0.0


@dataclass
class MetaResults:
    """Results from meta-analysis."""
    effect: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    tau2: float
    i2: float
    h2: float
    q_stat: float
    q_p_value: float
    method: str
    use_hksj: bool = False
    df: Optional[int] = None
    points: Optional[List[MetaPoint]] = None
    
    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.ci_upper - self.ci_lower


@dataclass 
class MetaAnalysisConfig:
    """Configuration for meta-analysis."""
    model: str = "RE"  # "FE" or "RE"
    tau2_method: str = "REML"  # "DL", "REML", "PM", "ML"
    use_hksj: bool = False
    alpha: float = 0.05
    
    def __post_init__(self):
        """Validate configuration."""
        if self.model not in ["FE", "RE"]:
            raise ValueError("Model must be 'FE' or 'RE'")
        if self.tau2_method not in ["DL", "REML", "PM", "ML"]:
            raise ValueError("tau2_method must be one of: DL, REML, PM, ML")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")


# Main analysis functions
def analyze_data(
    effects: np.ndarray,
    variances: np.ndarray,
    study_ids: Optional[List[str]] = None,
    config: Optional[MetaAnalysisConfig] = None
) -> MetaResults:
    """
    Perform meta-analysis on effect sizes and variances.
    
    Args:
        effects: Array of effect sizes
        variances: Array of variances
        study_ids: Optional study identifiers
        config: Configuration object
        
    Returns:
        MetaResults object with analysis results
    """
    if config is None:
        config = MetaAnalysisConfig()
        
    if study_ids is None:
        study_ids = [f"Study_{i+1}" for i in range(len(effects))]
        
    # Create MetaPoint objects
    points = [
        MetaPoint(effect=e, variance=v, study_id=sid)
        for e, v, sid in zip(effects, variances, study_ids)
    ]
    
    # Import model classes (will be implemented next)
    if config.model == "FE":
        from .models.fixed_effects import FixedEffects
        model = FixedEffects()
    else:
        from .models.random_effects import RandomEffects
        model = RandomEffects(tau2_method=config.tau2_method, use_hksj=config.use_hksj)
    
    return model.fit(points, alpha=config.alpha)


def analyze_csv(
    filepath: str,
    effect_col: str = "effect",
    variance_col: str = "variance", 
    study_col: str = "study",
    config: Optional[MetaAnalysisConfig] = None
) -> MetaResults:
    """
    Perform meta-analysis from CSV file.
    
    Args:
        filepath: Path to CSV file
        effect_col: Name of effect size column
        variance_col: Name of variance column
        study_col: Name of study ID column
        config: Configuration object
        
    Returns:
        MetaResults object with analysis results
    """
    df = pd.read_csv(filepath)
    
    if effect_col not in df.columns:
        raise ValueError(f"Effect column '{effect_col}' not found in CSV")
    if variance_col not in df.columns:
        raise ValueError(f"Variance column '{variance_col}' not found in CSV")
    if study_col not in df.columns:
        raise ValueError(f"Study column '{study_col}' not found in CSV")
        
    effects = df[effect_col].values
    variances = df[variance_col].values
    study_ids = df[study_col].astype(str).tolist()
    
    return analyze_data(effects, variances, study_ids, config)


# Import and expose main modules
from . import config
from . import stats
from . import models
from . import plots
from . import diagnostics

# Import specific functions for convenience
from .plots.forest import plot_forest
from .plots.funnel import plot_funnel

__all__ = [
    "MetaPoint",
    "MetaResults", 
    "MetaAnalysisConfig",
    "analyze_data",
    "analyze_csv",
    "plot_forest",
    "plot_funnel",
    "config",
    "stats", 
    "models",
    "plots",
    "diagnostics",
]