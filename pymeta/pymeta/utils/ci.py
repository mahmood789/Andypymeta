"""
Confidence interval utilities for PyMeta.

This module provides functions for calculating confidence intervals for various effect sizes.
"""

import numpy as np
import scipy.stats as stats
from typing import Tuple, Optional
from ..types import Array


def normal_ci(estimate: Array, se: Array, alpha: float = 0.05) -> Tuple[Array, Array]:
    """
    Normal approximation confidence interval.
    
    Parameters
    ----------
    estimate : array-like
        Point estimates
    se : array-like
        Standard errors
    alpha : float, default 0.05
        Significance level
    
    Returns
    -------
    tuple of arrays
        Lower and upper confidence bounds
    """
    estimate = np.asarray(estimate)
    se = np.asarray(se)
    
    z = stats.norm.ppf(1 - alpha / 2)
    margin = z * se
    
    return estimate - margin, estimate + margin


def t_ci(estimate: Array, se: Array, df: Array, alpha: float = 0.05) -> Tuple[Array, Array]:
    """
    t-distribution confidence interval.
    
    Parameters
    ----------
    estimate : array-like
        Point estimates
    se : array-like
        Standard errors
    df : array-like
        Degrees of freedom
    alpha : float, default 0.05
        Significance level
    
    Returns
    -------
    tuple of arrays
        Lower and upper confidence bounds
    """
    estimate = np.asarray(estimate)
    se = np.asarray(se)
    df = np.asarray(df)
    
    t = stats.t.ppf(1 - alpha / 2, df)
    margin = t * se
    
    return estimate - margin, estimate + margin


def log_normal_ci(log_estimate: Array, log_se: Array, alpha: float = 0.05) -> Tuple[Array, Array]:
    """
    Confidence interval for log-transformed estimates (exponentiated back).
    
    Parameters
    ----------
    log_estimate : array-like
        Log-transformed point estimates
    log_se : array-like
        Standard errors on log scale
    alpha : float, default 0.05
        Significance level
    
    Returns
    -------
    tuple of arrays
        Lower and upper confidence bounds (exponentiated)
    """
    log_lower, log_upper = normal_ci(log_estimate, log_se, alpha)
    return np.exp(log_lower), np.exp(log_upper)


def bootstrap_ci(data: Array, statistic_func, n_bootstrap: int = 1000, 
                alpha: float = 0.05, method: str = 'percentile') -> Tuple[float, float]:
    """
    Bootstrap confidence interval.
    
    Parameters
    ----------
    data : array-like
        Original data
    statistic_func : callable
        Function to compute statistic
    n_bootstrap : int, default 1000
        Number of bootstrap samples
    alpha : float, default 0.05
        Significance level
    method : str, default 'percentile'
        Bootstrap method ('percentile', 'bias_corrected', 'bca')
    
    Returns
    -------
    tuple
        Lower and upper confidence bounds
    """
    data = np.asarray(data)
    n = len(data)
    
    # Bootstrap samples
    bootstrap_stats = []
    rng = np.random.RandomState(42)  # For reproducibility
    
    for _ in range(n_bootstrap):
        boot_sample = rng.choice(data, size=n, replace=True)
        boot_stat = statistic_func(boot_sample)
        bootstrap_stats.append(boot_stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    if method == 'percentile':
        lower_p = 100 * (alpha / 2)
        upper_p = 100 * (1 - alpha / 2)
        return np.percentile(bootstrap_stats, [lower_p, upper_p])
    
    elif method == 'bias_corrected':
        # Bias-corrected method
        original_stat = statistic_func(data)
        p0 = np.mean(bootstrap_stats < original_stat)
        z0 = stats.norm.ppf(p0)
        
        z_alpha = stats.norm.ppf(alpha / 2)
        z_1_alpha = stats.norm.ppf(1 - alpha / 2)
        
        alpha_1 = stats.norm.cdf(2 * z0 + z_alpha)
        alpha_2 = stats.norm.cdf(2 * z0 + z_1_alpha)
        
        lower_p = 100 * alpha_1
        upper_p = 100 * alpha_2
        
        return np.percentile(bootstrap_stats, [lower_p, upper_p])
    
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")


def profile_likelihood_ci(log_likelihood_func, mle: float, alpha: float = 0.05,
                         bounds: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
    """
    Profile likelihood confidence interval.
    
    Parameters
    ----------
    log_likelihood_func : callable
        Log-likelihood function
    mle : float
        Maximum likelihood estimate
    alpha : float, default 0.05
        Significance level
    bounds : tuple, optional
        Search bounds (lower, upper)
    
    Returns
    -------
    tuple
        Lower and upper confidence bounds
    """
    from scipy.optimize import minimize_scalar
    
    # Critical value from chi-square distribution
    critical_value = stats.chi2.ppf(1 - alpha, 1) / 2
    
    # Maximum log-likelihood
    max_ll = log_likelihood_func(mle)
    
    # Target log-likelihood for CI bounds
    target_ll = max_ll - critical_value
    
    def objective(x):
        return (log_likelihood_func(x) - target_ll) ** 2
    
    if bounds is None:
        # Default bounds around MLE
        range_size = abs(mle) if mle != 0 else 1.0
        bounds = (mle - 10 * range_size, mle + 10 * range_size)
    
    # Find lower bound
    result_lower = minimize_scalar(objective, bounds=(bounds[0], mle), method='bounded')
    lower_bound = result_lower.x
    
    # Find upper bound
    result_upper = minimize_scalar(objective, bounds=(mle, bounds[1]), method='bounded')
    upper_bound = result_upper.x
    
    return lower_bound, upper_bound