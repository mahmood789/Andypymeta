"""Core dataclasses and type definitions for PyMeta."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


@dataclass
class MetaPoint:
    """Individual meta-analysis data point."""
    effect: float
    variance: float
    weight: Optional[float] = None
    label: Optional[str] = None
    study_id: Optional[str] = None
    
    def __post_init__(self):
        """Calculate weight if not provided."""
        if self.weight is None:
            self.weight = 1.0 / self.variance if self.variance > 0 else 0.0


@dataclass
class MetaResults:
    """Results from a meta-analysis."""
    pooled_effect: float
    pooled_variance: float
    confidence_interval: Tuple[float, float]
    z_score: float
    p_value: float
    tau2: float = 0.0
    i_squared: float = 0.0
    q_statistic: float = 0.0
    q_p_value: float = 0.0
    n_studies: int = 0
    method: str = "unknown"
    heterogeneity_test: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set default heterogeneity test if not provided."""
        if self.heterogeneity_test is None:
            self.heterogeneity_test = {}
    
    @property
    def pooled_se(self) -> float:
        """Standard error of pooled effect."""
        return np.sqrt(self.pooled_variance)
    
    @property
    def summary_dict(self) -> Dict[str, Any]:
        """Summary dictionary of results."""
        return {
            'pooled_effect': self.pooled_effect,
            'pooled_se': self.pooled_se,
            'ci_lower': self.confidence_interval[0],
            'ci_upper': self.confidence_interval[1],
            'z_score': self.z_score,
            'p_value': self.p_value,
            'tau2': self.tau2,
            'i_squared': self.i_squared,
            'q_statistic': self.q_statistic,
            'q_p_value': self.q_p_value,
            'n_studies': self.n_studies,
            'method': self.method
        }


@dataclass
class BiasTestResult:
    """Results from publication bias test."""
    test_name: str
    statistic: float
    p_value: float
    interpretation: str
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set default details if not provided."""
        if self.details is None:
            self.details = {}


@dataclass
class TSAResult:
    """Trial Sequential Analysis results."""
    cumulative_z: List[float]
    boundaries: Dict[str, List[float]]
    information_fraction: List[float]
    required_information_size: float
    futility_reached: bool = False
    superiority_reached: bool = False
    
    @property
    def monitoring_boundary_reached(self) -> bool:
        """Check if any monitoring boundary was reached."""
        return self.futility_reached or self.superiority_reached