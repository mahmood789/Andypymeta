"""
Hartung-Knapp-Sidik-Jonkman (HKSJ) variance adjustment implementation.

The HKSJ method provides a more robust variance estimation for random effects
meta-analysis by using a t-distribution instead of normal distribution and
adjusting the variance estimates.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from scipy.stats import t


@dataclass
class HKSJResult:
    """Result from HKSJ variance adjustment."""
    se_hk: float  # HKSJ-adjusted standard error
    df: int      # Degrees of freedom for t-distribution
    tcrit: float # Critical t-value for confidence intervals
    
    @property
    def variance_hk(self) -> float:
        """HKSJ-adjusted variance."""
        return self.se_hk ** 2


def hksj_se(
    effects: np.ndarray,
    variances: np.ndarray, 
    tau2: float,
    alpha: float = 0.05
) -> HKSJResult:
    """
    Calculate HKSJ-adjusted standard error and degrees of freedom.
    
    The HKSJ method adjusts the variance of the pooled effect estimate
    to account for uncertainty in the between-study variance (tau²).
    It uses a t-distribution with adjusted degrees of freedom instead
    of the normal distribution.
    
    Args:
        effects: Array of study effect sizes
        variances: Array of study variances  
        tau2: Between-study variance estimate
        alpha: Significance level for confidence intervals
        
    Returns:
        HKSJResult with adjusted standard error, degrees of freedom, and critical value
        
    References:
        Hartung, J., & Knapp, G. (2001). A refined method for the meta‐analysis 
        of controlled clinical trials with binary outcome. Statistics in Medicine, 
        20(24), 3875-3889.
        
        Sidik, K., & Jonkman, J. N. (2002). A simple confidence interval for 
        meta‐analysis. Statistics in Medicine, 21(21), 3153-3159.
    """
    if len(effects) != len(variances):
        raise ValueError("effects and variances must have same length")
        
    if len(effects) < 2:
        raise ValueError("Need at least 2 studies for HKSJ adjustment")
        
    k = len(effects)  # Number of studies
    
    # Calculate weights for random effects model
    weights = 1.0 / (variances + tau2)
    sum_weights = np.sum(weights)
    
    # Calculate pooled effect estimate  
    pooled_effect = np.sum(weights * effects) / sum_weights
    
    # Calculate Q statistic (measure of heterogeneity)
    q_stat = np.sum(weights * (effects - pooled_effect) ** 2)
    
    # Standard variance for pooled effect
    var_pooled = 1.0 / sum_weights
    
    # HKSJ variance adjustment factor
    if k > 2:
        # Adjustment factor based on Q statistic
        adjustment_factor = q_stat / (k - 1)
        
        # Apply bounds to prevent extreme adjustments
        adjustment_factor = max(1.0, adjustment_factor)
        
        # HKSJ-adjusted variance
        var_hk = var_pooled * adjustment_factor
    else:
        # For k=2, use standard variance 
        var_hk = var_pooled
        
    se_hk = np.sqrt(var_hk)
    
    # Degrees of freedom for t-distribution
    # Use conservative estimate: k - 1
    df = k - 1
    
    # Critical t-value for confidence intervals
    tcrit = t.ppf(1 - alpha/2, df)
    
    return HKSJResult(se_hk=se_hk, df=df, tcrit=tcrit)


def hksj_confidence_interval(
    pooled_effect: float,
    hksj_result: HKSJResult
) -> tuple[float, float]:
    """
    Calculate confidence interval using HKSJ adjustment.
    
    Args:
        pooled_effect: Pooled effect estimate
        hksj_result: Result from hksj_se function
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    margin = hksj_result.tcrit * hksj_result.se_hk
    return (pooled_effect - margin, pooled_effect + margin)


def hksj_p_value(
    pooled_effect: float,
    hksj_result: HKSJResult
) -> float:
    """
    Calculate two-sided p-value using HKSJ adjustment.
    
    Args:
        pooled_effect: Pooled effect estimate
        hksj_result: Result from hksj_se function
        
    Returns:
        Two-sided p-value
    """
    t_stat = pooled_effect / hksj_result.se_hk
    p_value = 2 * (1 - t.cdf(abs(t_stat), hksj_result.df))
    return p_value