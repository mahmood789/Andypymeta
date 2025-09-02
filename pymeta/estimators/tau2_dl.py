"""DerSimonian-Laird tau² estimator."""

import numpy as np
from typing import List
from ..typing import MetaPoint
from ..registries import register_estimator
from ..errors import EstimationError


@register_estimator("DL")
def tau2_dersimonian_laird(points: List[MetaPoint]) -> float:
    """DerSimonian-Laird estimator for between-study variance (tau²).
    
    This is the most commonly used estimator, though it can be biased
    downward especially with few studies.
    
    Args:
        points: List of MetaPoint objects
        
    Returns:
        Estimated tau² value
        
    Raises:
        EstimationError: If estimation fails
    
    References:
        DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
        Controlled Clinical Trials, 7(3), 177-188.
    """
    if len(points) < 2:
        return 0.0
    
    try:
        # Extract effects and weights
        effects = np.array([p.effect for p in points])
        weights = np.array([p.weight for p in points])
        
        # Calculate Q statistic
        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        q_statistic = np.sum(weights * (effects - weighted_mean) ** 2)
        
        # Degrees of freedom
        df = len(points) - 1
        
        # Calculate C (correction factor)
        sum_weights = np.sum(weights)
        sum_weights_squared = np.sum(weights ** 2)
        c = sum_weights - (sum_weights_squared / sum_weights)
        
        # DerSimonian-Laird estimate
        if c > 0:
            tau2 = max(0, (q_statistic - df) / c)
        else:
            tau2 = 0.0
        
        return tau2
        
    except Exception as e:
        raise EstimationError(f"DerSimonian-Laird estimation failed: {e}")


@register_estimator("DL_simple")
def tau2_dersimonian_laird_simple(points: List[MetaPoint]) -> float:
    """Simplified DerSimonian-Laird estimator using inverse variance weights.
    
    This version uses simple inverse variance weights (1/variance) rather than
    the iterative weights, making it faster but potentially less accurate.
    
    Args:
        points: List of MetaPoint objects
        
    Returns:
        Estimated tau² value
    """
    if len(points) < 2:
        return 0.0
    
    try:
        # Use inverse variance weights
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        weights = 1.0 / variances
        
        # Calculate weighted mean
        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        
        # Calculate Q statistic
        q_statistic = np.sum(weights * (effects - weighted_mean) ** 2)
        
        # Degrees of freedom
        df = len(points) - 1
        
        # Calculate C (correction factor)
        sum_weights = np.sum(weights)
        sum_weights_squared = np.sum(weights ** 2)
        c = sum_weights - (sum_weights_squared / sum_weights)
        
        # DL estimate
        if c > 0 and q_statistic > df:
            tau2 = (q_statistic - df) / c
        else:
            tau2 = 0.0
        
        return tau2
        
    except Exception as e:
        raise EstimationError(f"Simple DerSimonian-Laird estimation failed: {e}")


def _calculate_q_statistic(points: List[MetaPoint], weights: np.ndarray = None) -> float:
    """Calculate Cochran's Q statistic.
    
    Args:
        points: List of MetaPoint objects
        weights: Optional weights array
        
    Returns:
        Q statistic value
    """
    effects = np.array([p.effect for p in points])
    
    if weights is None:
        weights = np.array([p.weight for p in points])
    
    weighted_mean = np.sum(weights * effects) / np.sum(weights)
    q_statistic = np.sum(weights * (effects - weighted_mean) ** 2)
    
    return q_statistic


def _calculate_heterogeneity_stats(points: List[MetaPoint], tau2: float) -> dict:
    """Calculate heterogeneity statistics.
    
    Args:
        points: List of MetaPoint objects
        tau2: Between-study variance estimate
        
    Returns:
        Dictionary with heterogeneity statistics
    """
    from scipy import stats
    
    # Basic statistics
    k = len(points)
    df = k - 1
    
    # Calculate Q with original weights
    weights = np.array([p.weight for p in points])
    q_statistic = _calculate_q_statistic(points, weights)
    
    # P-value for Q test
    q_p_value = 1 - stats.chi2.cdf(q_statistic, df) if df > 0 else 1.0
    
    # I² statistic
    if q_statistic > df and df > 0:
        i_squared = 100 * (q_statistic - df) / q_statistic
    else:
        i_squared = 0.0
    
    # H² statistic
    h_squared = q_statistic / df if df > 0 else 1.0
    
    return {
        'q_statistic': q_statistic,
        'q_p_value': q_p_value,
        'i_squared': i_squared,
        'h_squared': h_squared,
        'tau2': tau2,
        'tau': np.sqrt(tau2),
        'df': df
    }