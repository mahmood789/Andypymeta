"""
Type definitions and custom types for PyMeta.

This module defines type aliases and data structures used throughout the package.
"""

from typing import Union, Tuple, List, Dict, Any, Optional, Callable, Protocol
from dataclasses import dataclass
import numpy as np
import pandas as pd


# Basic type aliases
Numeric = Union[int, float, np.number]
Array = Union[List[Numeric], np.ndarray, pd.Series]
DataFrame = pd.DataFrame

# Statistical types
EffectSize = float
Variance = float
StandardError = float
PValue = float
ConfidenceInterval = Tuple[float, float]
Weight = float

# Model types
ModelType = str  # 'fixed', 'random', 'multivariate'
Tau2Method = str  # 'dl', 'ml', 'reml', 'pm', 'hs', 'eb', 'sj'
LinkFunction = str  # 'identity', 'log', 'logit', 'cloglog'

# Plot types
PlotStyle = str  # 'classic', 'modern', 'minimal', 'publication'
ColorMap = Union[str, List[str], Dict[str, str]]


@dataclass
class EffectSizeResult:
    """Result of effect size calculation."""
    effect_size: EffectSize
    variance: Variance
    standard_error: StandardError
    ci_lower: float
    ci_upper: float
    sample_size: Optional[int] = None
    method: Optional[str] = None


@dataclass
class MetaAnalysisResult:
    """Result of meta-analysis."""
    overall_effect: EffectSize
    overall_variance: Variance
    ci_lower: float
    ci_upper: float
    pvalue: PValue
    q_statistic: float
    q_pvalue: PValue
    i_squared: float
    tau_squared: float
    tau_squared_se: float
    n_studies: int
    model: ModelType
    tau2_method: Optional[Tau2Method] = None
    weights: Optional[Array] = None
    fitted_values: Optional[Array] = None
    residuals: Optional[Array] = None
    studies: Optional[List[str]] = None


@dataclass
class SubgroupResult:
    """Result of subgroup analysis."""
    subgroups: Dict[str, MetaAnalysisResult]
    overall: MetaAnalysisResult
    between_groups_q: float
    between_groups_pvalue: PValue
    within_groups_q: float
    within_groups_pvalue: PValue
    grouping_variable: str


@dataclass
class MetaRegressionResult:
    """Result of meta-regression analysis."""
    coefficients: Dict[str, float]
    standard_errors: Dict[str, float]
    pvalues: Dict[str, PValue]
    ci_lower: Dict[str, float]
    ci_upper: Dict[str, float]
    r_squared: float
    tau_squared: float
    q_residual: float
    q_residual_pvalue: PValue
    fitted_values: Array
    residuals: Array
    moderators: List[str]


@dataclass
class BiasTestResult:
    """Result of publication bias test."""
    test_name: str
    statistic: float
    pvalue: PValue
    interpretation: str
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class EggerTestResult(BiasTestResult):
    """Result of Egger's regression test."""
    bias: float
    bias_se: float
    t_statistic: float
    ci_lower: float
    ci_upper: float


@dataclass
class BeggTestResult(BiasTestResult):
    """Result of Begg's rank correlation test."""
    tau: float
    z_statistic: float


@dataclass
class TrimFillResult:
    """Result of trim-and-fill analysis."""
    n_trimmed: int
    n_filled: int
    original_effect: EffectSize
    adjusted_effect: EffectSize
    original_ci: ConfidenceInterval
    adjusted_ci: ConfidenceInterval
    filled_studies: Optional[DataFrame] = None


@dataclass
class InfluenceResult:
    """Result of influence analysis."""
    study_id: str
    influence_measure: float
    effect_without: EffectSize
    se_without: StandardError
    ci_without: ConfidenceInterval
    dfbetas: float
    cooks_distance: float


@dataclass
class LeaveOneOutResult:
    """Result of leave-one-out analysis."""
    influences: List[InfluenceResult]
    min_effect: EffectSize
    max_effect: EffectSize
    effect_range: float
    most_influential_study: str
    max_influence: float


@dataclass
class CumulativeResult:
    """Result of cumulative meta-analysis."""
    cumulative_effects: Array
    cumulative_ci_lower: Array
    cumulative_ci_upper: Array
    cumulative_pvalues: Array
    study_order: List[str]
    order_variable: str


@dataclass
class HeterogeneityMeasures:
    """Heterogeneity measures."""
    q_statistic: float
    q_pvalue: PValue
    i_squared: float
    h_squared: float
    tau_squared: float
    tau_squared_se: float
    tau_squared_ci: ConfidenceInterval


@dataclass
class ModelFit:
    """Model fit statistics."""
    aic: float
    bic: float
    log_likelihood: float
    deviance: float
    degrees_freedom: int


class EstimatorProtocol(Protocol):
    """Protocol for tau-squared estimators."""
    
    def estimate(self, effects: Array, variances: Array, weights: Array) -> float:
        """Estimate tau-squared."""
        ...


class ModelProtocol(Protocol):
    """Protocol for meta-analysis models."""
    
    def fit(self, data: DataFrame) -> MetaAnalysisResult:
        """Fit the model to data."""
        ...
    
    def predict(self, data: DataFrame) -> Array:
        """Make predictions."""
        ...


class PlotProtocol(Protocol):
    """Protocol for plotting functions."""
    
    def __call__(self, result: MetaAnalysisResult, **kwargs) -> Any:
        """Create plot."""
        ...


# Configuration types
@dataclass
class PlotConfig:
    """Plot configuration."""
    style: PlotStyle = 'classic'
    figure_size: Tuple[float, float] = (10, 8)
    dpi: int = 300
    font_size: int = 12
    colors: Optional[ColorMap] = None


@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    model: ModelType = 'random'
    tau2_method: Tau2Method = 'dl'
    alpha: float = 0.05
    digits: int = 4
    confidence_level: float = 0.95


@dataclass
class LivingReviewConfig:
    """Living review configuration."""
    review_id: str
    search_strategy: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    data_sources: List[str]
    update_frequency: str = 'monthly'
    auto_search: bool = True
    notification_methods: List[str] = None


# Data validation types
class ValidatedDataFrame(DataFrame):
    """DataFrame with validation."""
    
    def __init__(self, data, required_columns=None, **kwargs):
        super().__init__(data, **kwargs)
        if required_columns:
            self._validate_columns(required_columns)
    
    def _validate_columns(self, required_columns):
        """Validate required columns are present."""
        missing = [col for col in required_columns if col not in self.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


# Search and screening types
@dataclass
class SearchResult:
    """Result from literature search."""
    title: str
    authors: List[str]
    abstract: str
    doi: Optional[str]
    pmid: Optional[str]
    source: str
    publication_date: Optional[str]
    relevance_score: Optional[float] = None


@dataclass
class ScreeningResult:
    """Result from study screening."""
    study_id: str
    include: bool
    exclusion_reason: Optional[str]
    reviewer: str
    screening_date: str
    confidence: Optional[float] = None


# Network meta-analysis types (for future implementation)
@dataclass
class NetworkNode:
    """Node in treatment network."""
    treatment_id: str
    treatment_name: str
    n_studies: int
    n_participants: int


@dataclass
class NetworkEdge:
    """Edge in treatment network."""
    treatment_a: str
    treatment_b: str
    n_studies: int
    n_participants: int
    direct_evidence: bool


@dataclass
class NetworkGeometry:
    """Network geometry measures."""
    n_treatments: int
    n_comparisons: int
    n_studies: int
    connectivity: float
    transitivity: float


# Export/import types
FileFormat = str  # 'csv', 'excel', 'json', 'stata', 'spss'
ExportFormat = str  # 'csv', 'excel', 'json', 'html', 'pdf'

# Function type aliases
Transformer = Callable[[Array], Array]
Validator = Callable[[Any], bool]
Processor = Callable[[DataFrame], DataFrame]