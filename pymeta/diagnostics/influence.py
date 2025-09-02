"""
Influence diagnostics and leave-one-out analysis for meta-analysis.

This module provides tools for identifying influential studies and assessing
the stability of meta-analysis results through leave-one-out analysis.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from .. import MetaPoint, MetaResults, MetaAnalysisConfig, analyze_data


@dataclass
class InfluenceResult:
    """Results from influence analysis for a single study."""
    study_id: str
    effect: float
    variance: float
    weight: float
    standardized_residual: float
    leverage: float
    cook_distance: float
    dffits: float
    dfbetas: float
    studentized_residual: float


@dataclass
class LeaveOneOutResult:
    """Results from leave-one-out analysis."""
    original_result: MetaResults
    loo_results: List[MetaResults]
    study_ids: List[str]
    effect_changes: List[float]
    se_changes: List[float]
    i2_changes: List[float]
    tau2_changes: List[float]
    
    @property
    def max_effect_change(self) -> float:
        """Maximum change in pooled effect when removing a study."""
        return max(abs(change) for change in self.effect_changes)
    
    @property
    def most_influential_study(self) -> str:
        """Study ID of the most influential study (largest effect change)."""
        max_idx = np.argmax([abs(change) for change in self.effect_changes])
        return self.study_ids[max_idx]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for easy analysis."""
        data = []
        for i, (study_id, loo_result) in enumerate(zip(self.study_ids, self.loo_results)):
            data.append({
                'study_id': study_id,
                'original_effect': self.original_result.effect,
                'loo_effect': loo_result.effect,
                'effect_change': self.effect_changes[i],
                'original_se': self.original_result.se,
                'loo_se': loo_result.se,
                'se_change': self.se_changes[i],
                'original_i2': self.original_result.i2,
                'loo_i2': loo_result.i2,
                'i2_change': self.i2_changes[i],
                'original_tau2': self.original_result.tau2,
                'loo_tau2': loo_result.tau2,
                'tau2_change': self.tau2_changes[i],
                'loo_ci_lower': loo_result.ci_lower,
                'loo_ci_upper': loo_result.ci_upper,
                'loo_p_value': loo_result.p_value
            })
        return pd.DataFrame(data)


def leave_one_out_analysis(
    points: List[MetaPoint],
    config: Optional[MetaAnalysisConfig] = None
) -> LeaveOneOutResult:
    """
    Perform leave-one-out analysis to assess study influence.
    
    This function systematically removes each study and re-runs the meta-analysis
    to assess how each study influences the overall results.
    
    Args:
        points: List of MetaPoint objects
        config: Configuration for meta-analysis
        
    Returns:
        LeaveOneOutResult with original and leave-one-out results
    """
    if config is None:
        config = MetaAnalysisConfig()
        
    if len(points) < 3:
        raise ValueError("Need at least 3 studies for meaningful leave-one-out analysis")
    
    # Get original results with all studies
    effects = np.array([p.effect for p in points])
    variances = np.array([p.variance for p in points])
    study_ids = [p.study_id for p in points]
    
    original_result = analyze_data(effects, variances, study_ids, config)
    
    # Perform leave-one-out analysis
    loo_results = []
    for i in range(len(points)):
        # Create subset excluding study i
        loo_points = [p for j, p in enumerate(points) if j != i]
        loo_effects = np.array([p.effect for p in loo_points])
        loo_variances = np.array([p.variance for p in loo_points])
        loo_study_ids = [p.study_id for p in loo_points]
        
        # Run meta-analysis without study i
        loo_result = analyze_data(loo_effects, loo_variances, loo_study_ids, config)
        loo_results.append(loo_result)
    
    # Calculate changes
    effect_changes = [loo.effect - original_result.effect for loo in loo_results]
    se_changes = [loo.se - original_result.se for loo in loo_results]
    i2_changes = [loo.i2 - original_result.i2 for loo in loo_results]
    tau2_changes = [loo.tau2 - original_result.tau2 for loo in loo_results]
    
    return LeaveOneOutResult(
        original_result=original_result,
        loo_results=loo_results,
        study_ids=study_ids,
        effect_changes=effect_changes,
        se_changes=se_changes,
        i2_changes=i2_changes,
        tau2_changes=tau2_changes
    )


def influence_measures(
    points: List[MetaPoint],
    meta_result: MetaResults,
    config: Optional[MetaAnalysisConfig] = None
) -> List[InfluenceResult]:
    """
    Calculate influence measures for each study in the meta-analysis.
    
    Computes various influence diagnostics including standardized residuals,
    leverage, Cook's distance, DFFITS, DFBETAS, and studentized residuals.
    
    Args:
        points: List of MetaPoint objects
        meta_result: Results from meta-analysis
        config: Configuration for meta-analysis
        
    Returns:
        List of InfluenceResult objects, one for each study
    """
    if config is None:
        config = MetaAnalysisConfig()
        
    if len(points) < 2:
        raise ValueError("Need at least 2 studies for influence analysis")
    
    effects = np.array([p.effect for p in points])
    variances = np.array([p.variance for p in points])
    k = len(points)
    
    # Calculate weights based on model type
    if config.model == "FE":
        weights = 1.0 / variances
    else:
        # Random effects weights
        weights = 1.0 / (variances + meta_result.tau2)
    
    sum_weights = np.sum(weights)
    pooled_effect = meta_result.effect
    
    # Calculate influence measures for each study
    influence_results = []
    
    for i, point in enumerate(points):
        # Raw residual
        residual = point.effect - pooled_effect
        
        # Leverage (hat value)
        leverage = weights[i] / sum_weights
        
        # Standardized residual
        residual_variance = variances[i] + meta_result.tau2 if config.model == "RE" else variances[i]
        standardized_residual = residual / np.sqrt(residual_variance)
        
        # Studentized residual (external)
        # This is a simplified calculation
        mse = np.sum(weights * (effects - pooled_effect) ** 2) / (k - 1)
        studentized_residual = residual / np.sqrt(mse * (1 - leverage))
        
        # Cook's distance
        cook_distance = (standardized_residual ** 2 * leverage) / (1 - leverage)
        
        # DFFITS (difference in fits)
        dffits = studentized_residual * np.sqrt(leverage / (1 - leverage))
        
        # DFBETAS (change in coefficient)
        dfbetas = studentized_residual * np.sqrt(leverage)
        
        influence_results.append(InfluenceResult(
            study_id=point.study_id,
            effect=point.effect,
            variance=point.variance,
            weight=weights[i],
            standardized_residual=standardized_residual,
            leverage=leverage,
            cook_distance=cook_distance,
            dffits=dffits,
            dfbetas=dfbetas,
            studentized_residual=studentized_residual
        ))
    
    return influence_results


def identify_outliers(
    influence_results: List[InfluenceResult],
    cook_threshold: float = 1.0,
    studentized_threshold: float = 2.0,
    leverage_threshold: Optional[float] = None
) -> Dict[str, List[str]]:
    """
    Identify potentially influential or outlying studies based on diagnostic criteria.
    
    Args:
        influence_results: List of InfluenceResult objects
        cook_threshold: Threshold for Cook's distance
        studentized_threshold: Threshold for studentized residuals
        leverage_threshold: Threshold for leverage (if None, uses 2*k/n rule)
        
    Returns:
        Dictionary with lists of study IDs flagged by each criterion
    """
    k = len(influence_results)
    
    if leverage_threshold is None:
        leverage_threshold = 2.0 / k  # Common rule of thumb
    
    outliers = {
        'high_cook': [],
        'high_studentized': [], 
        'high_leverage': [],
        'high_dffits': [],
        'any_flag': []
    }
    
    dffits_threshold = 2 * np.sqrt(1.0 / k)  # Common threshold for DFFITS
    
    for result in influence_results:
        study_id = result.study_id
        flagged = False
        
        if result.cook_distance > cook_threshold:
            outliers['high_cook'].append(study_id)
            flagged = True
            
        if abs(result.studentized_residual) > studentized_threshold:
            outliers['high_studentized'].append(study_id)
            flagged = True
            
        if result.leverage > leverage_threshold:
            outliers['high_leverage'].append(study_id)
            flagged = True
            
        if abs(result.dffits) > dffits_threshold:
            outliers['high_dffits'].append(study_id)
            flagged = True
            
        if flagged:
            outliers['any_flag'].append(study_id)
    
    return outliers