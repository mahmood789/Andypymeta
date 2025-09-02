"""
Fixed effects meta-analysis model.
"""

from typing import List
import numpy as np
from scipy.stats import norm

from ..stats.hksj import hksj_se, hksj_confidence_interval, hksj_p_value
from .. import MetaPoint, MetaResults


class FixedEffects:
    """Fixed effects meta-analysis model."""
    
    def __init__(self):
        """Initialize fixed effects model."""
        pass
    
    def fit(self, points: List[MetaPoint], alpha: float = 0.05) -> MetaResults:
        """
        Fit fixed effects meta-analysis model.
        
        Args:
            points: List of MetaPoint objects
            alpha: Significance level
            
        Returns:
            MetaResults object with analysis results
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 studies for meta-analysis")
            
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        
        # Fixed effects weights (inverse variance)
        weights = 1.0 / variances
        sum_weights = np.sum(weights)
        
        # Pooled effect estimate
        pooled_effect = np.sum(weights * effects) / sum_weights
        
        # Standard error of pooled effect
        pooled_se = np.sqrt(1.0 / sum_weights)
        
        # Confidence interval (normal distribution)
        zcrit = norm.ppf(1 - alpha/2)
        ci_lower = pooled_effect - zcrit * pooled_se
        ci_upper = pooled_effect + zcrit * pooled_se
        
        # P-value (two-sided)
        z_stat = pooled_effect / pooled_se
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        
        # Heterogeneity statistics
        q_stat = np.sum(weights * (effects - pooled_effect) ** 2)
        df = len(points) - 1
        q_p_value = 1 - norm.cdf(q_stat) if df > 0 else 1.0
        
        # I² and H² (set to 0 for fixed effects model)
        i2 = 0.0
        h2 = 1.0
        tau2 = 0.0  # No between-study variance in fixed effects
        
        return MetaResults(
            effect=pooled_effect,
            se=pooled_se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            tau2=tau2,
            i2=i2,
            h2=h2,
            q_stat=q_stat,
            q_p_value=q_p_value,
            method="Fixed Effects",
            use_hksj=False,
            points=points
        )