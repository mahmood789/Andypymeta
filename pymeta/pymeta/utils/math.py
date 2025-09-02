"""
Mathematical utility functions for PyMeta.

This module provides mathematical functions and calculations used in meta-analysis.
"""

import numpy as np
import scipy.stats as stats
from scipy.special import digamma, polygamma
from typing import Union, Tuple, Optional
from ..types import Array, Numeric


def safe_divide(numerator: Array, denominator: Array, 
               fill_value: float = 0.0) -> Array:
    """
    Safely divide arrays, handling division by zero.
    
    Parameters
    ----------
    numerator : array-like
        Numerator values
    denominator : array-like
        Denominator values
    fill_value : float, default 0.0
        Value to use when denominator is zero
    
    Returns
    -------
    numpy.ndarray
        Result of division
    """
    numerator = np.asarray(numerator)
    denominator = np.asarray(denominator)
    
    result = np.full_like(numerator, fill_value, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    
    return result


def safe_log(x: Array, fill_value: float = np.nan) -> Array:
    """
    Safely compute natural logarithm, handling non-positive values.
    
    Parameters
    ----------
    x : array-like
        Input values
    fill_value : float, default np.nan
        Value to use for non-positive inputs
    
    Returns
    -------
    numpy.ndarray
        Natural logarithm of inputs
    """
    x = np.asarray(x)
    result = np.full_like(x, fill_value, dtype=float)
    mask = x > 0
    result[mask] = np.log(x[mask])
    
    return result


def safe_sqrt(x: Array, fill_value: float = 0.0) -> Array:
    """
    Safely compute square root, handling negative values.
    
    Parameters
    ----------
    x : array-like
        Input values
    fill_value : float, default 0.0
        Value to use for negative inputs
    
    Returns
    -------
    numpy.ndarray
        Square root of inputs
    """
    x = np.asarray(x)
    result = np.full_like(x, fill_value, dtype=float)
    mask = x >= 0
    result[mask] = np.sqrt(x[mask])
    
    return result


def logit(p: Array) -> Array:
    """
    Compute logit transformation.
    
    Parameters
    ----------
    p : array-like
        Probabilities (must be between 0 and 1)
    
    Returns
    -------
    numpy.ndarray
        Logit-transformed values
    """
    p = np.asarray(p)
    return safe_log(p / (1 - p))


def expit(x: Array) -> Array:
    """
    Compute inverse logit (logistic) transformation.
    
    Parameters
    ----------
    x : array-like
        Input values
    
    Returns
    -------
    numpy.ndarray
        Inverse logit-transformed values
    """
    x = np.asarray(x)
    return 1 / (1 + np.exp(-x))


def cloglog(p: Array) -> Array:
    """
    Compute complementary log-log transformation.
    
    Parameters
    ----------
    p : array-like
        Probabilities (must be between 0 and 1)
    
    Returns
    -------
    numpy.ndarray
        Complementary log-log transformed values
    """
    p = np.asarray(p)
    return safe_log(-safe_log(1 - p))


def inv_cloglog(x: Array) -> Array:
    """
    Compute inverse complementary log-log transformation.
    
    Parameters
    ----------
    x : array-like
        Input values
    
    Returns
    -------
    numpy.ndarray
        Inverse complementary log-log transformed values
    """
    x = np.asarray(x)
    return 1 - np.exp(-np.exp(x))


def fisher_z_transform(r: Array) -> Array:
    """
    Fisher's z-transformation for correlations.
    
    Parameters
    ----------
    r : array-like
        Correlation coefficients
    
    Returns
    -------
    numpy.ndarray
        Fisher's z-transformed values
    """
    r = np.asarray(r)
    # Clip to valid range to avoid numerical issues
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * safe_log((1 + r) / (1 - r))


def inv_fisher_z_transform(z: Array) -> Array:
    """
    Inverse Fisher's z-transformation.
    
    Parameters
    ----------
    z : array-like
        Fisher's z values
    
    Returns
    -------
    numpy.ndarray
        Correlation coefficients
    """
    z = np.asarray(z)
    exp_2z = np.exp(2 * z)
    return (exp_2z - 1) / (exp_2z + 1)


def freeman_tukey_transform(k: Array, n: Array) -> Array:
    """
    Freeman-Tukey double arcsine transformation for proportions.
    
    Parameters
    ----------
    k : array-like
        Number of events
    n : array-like
        Total number of observations
    
    Returns
    -------
    numpy.ndarray
        Freeman-Tukey transformed values
    """
    k = np.asarray(k)
    n = np.asarray(n)
    
    return np.arcsin(safe_sqrt(k / (n + 1))) + np.arcsin(safe_sqrt((k + 1) / (n + 1)))


def inv_freeman_tukey_transform(y: Array, n: Array) -> Array:
    """
    Inverse Freeman-Tukey transformation.
    
    Parameters
    ----------
    y : array-like
        Freeman-Tukey transformed values
    n : array-like
        Total number of observations
    
    Returns
    -------
    numpy.ndarray
        Proportions
    """
    y = np.asarray(y)
    n = np.asarray(n)
    
    # This is an approximation - exact inverse requires numerical methods
    sin_half_y = np.sin(y / 2)
    return n * sin_half_y**2


def hedges_g_correction(n: Array) -> Array:
    """
    Hedges' bias correction factor for standardized mean differences.
    
    Parameters
    ----------
    n : array-like
        Total sample sizes
    
    Returns
    -------
    numpy.ndarray
        Correction factors
    """
    n = np.asarray(n)
    df = n - 2
    
    # Small sample correction
    correction = 1 - 3 / (4 * df - 1)
    
    # For very small samples, use more accurate formula
    small_sample_mask = df < 10
    if np.any(small_sample_mask):
        df_small = df[small_sample_mask]
        correction[small_sample_mask] = (
            np.exp(stats.loggamma(df_small / 2) - np.log(safe_sqrt(df_small / 2)) 
                  - stats.loggamma((df_small - 1) / 2))
        )
    
    return correction


def glass_delta_variance(n1: Array, n2: Array, delta: Array) -> Array:
    """
    Variance of Glass's delta effect size.
    
    Parameters
    ----------
    n1 : array-like
        Sample size of group 1
    n2 : array-like
        Sample size of group 2
    delta : array-like
        Effect size
    
    Returns
    -------
    numpy.ndarray
        Variances
    """
    n1 = np.asarray(n1)
    n2 = np.asarray(n2)
    delta = np.asarray(delta)
    
    return (n1 + n2) / (n1 * n2) + delta**2 / (2 * n2)


def cohen_d_variance(n1: Array, n2: Array, d: Array) -> Array:
    """
    Variance of Cohen's d effect size.
    
    Parameters
    ----------
    n1 : array-like
        Sample size of group 1
    n2 : array-like
        Sample size of group 2
    d : array-like
        Effect size
    
    Returns
    -------
    numpy.ndarray
        Variances
    """
    n1 = np.asarray(n1)
    n2 = np.asarray(n2)
    d = np.asarray(d)
    
    return (n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2))


def log_odds_ratio_variance(a: Array, b: Array, c: Array, d: Array) -> Array:
    """
    Variance of log odds ratio.
    
    Parameters
    ----------
    a : array-like
        Treatment successes
    b : array-like
        Treatment failures
    c : array-like
        Control successes
    d : array-like
        Control failures
    
    Returns
    -------
    numpy.ndarray
        Variances
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    d = np.asarray(d)
    
    return safe_divide(1, a) + safe_divide(1, b) + safe_divide(1, c) + safe_divide(1, d)


def log_risk_ratio_variance(a: Array, b: Array, c: Array, d: Array) -> Array:
    """
    Variance of log risk ratio.
    
    Parameters
    ----------
    a : array-like
        Treatment successes
    b : array-like
        Treatment failures
    c : array-like
        Control successes
    d : array-like
        Control failures
    
    Returns
    -------
    numpy.ndarray
        Variances
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    d = np.asarray(d)
    
    n1 = a + b
    n2 = c + d
    
    return safe_divide(b, a * n1) + safe_divide(d, c * n2)


def risk_difference_variance(a: Array, b: Array, c: Array, d: Array) -> Array:
    """
    Variance of risk difference.
    
    Parameters
    ----------
    a : array-like
        Treatment successes
    b : array-like
        Treatment failures
    c : array-like
        Control successes
    d : array-like
        Control failures
    
    Returns
    -------
    numpy.ndarray
        Variances
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    d = np.asarray(d)
    
    n1 = a + b
    n2 = c + d
    
    p1 = safe_divide(a, n1)
    p2 = safe_divide(c, n2)
    
    return safe_divide(p1 * (1 - p1), n1) + safe_divide(p2 * (1 - p2), n2)


def correlation_variance(r: Array, n: Array) -> Array:
    """
    Variance of correlation coefficient.
    
    Parameters
    ----------
    r : array-like
        Correlation coefficients
    n : array-like
        Sample sizes
    
    Returns
    -------
    numpy.ndarray
        Variances
    """
    r = np.asarray(r)
    n = np.asarray(n)
    
    return (1 - r**2)**2 / (n - 1)


def fisher_z_variance(n: Array) -> Array:
    """
    Variance of Fisher's z-transformed correlation.
    
    Parameters
    ----------
    n : array-like
        Sample sizes
    
    Returns
    -------
    numpy.ndarray
        Variances
    """
    n = np.asarray(n)
    return 1 / (n - 3)


def q_statistic(effects: Array, variances: Array, weights: Array) -> float:
    """
    Compute Cochran's Q statistic for heterogeneity.
    
    Parameters
    ----------
    effects : array-like
        Effect sizes
    variances : array-like
        Variances
    weights : array-like
        Weights
    
    Returns
    -------
    float
        Q statistic
    """
    effects = np.asarray(effects)
    weights = np.asarray(weights)
    
    # Weighted mean
    weighted_mean = np.sum(weights * effects) / np.sum(weights)
    
    # Q statistic
    q = np.sum(weights * (effects - weighted_mean)**2)
    
    return q


def i_squared(q: float, df: int) -> float:
    """
    Compute I² statistic from Q and degrees of freedom.
    
    Parameters
    ----------
    q : float
        Q statistic
    df : int
        Degrees of freedom
    
    Returns
    -------
    float
        I² statistic (percentage)
    """
    if df <= 0 or q <= df:
        return 0.0
    
    return 100 * (q - df) / q


def h_squared(q: float, df: int) -> float:
    """
    Compute H² statistic from Q and degrees of freedom.
    
    Parameters
    ----------
    q : float
        Q statistic
    df : int
        Degrees of freedom
    
    Returns
    -------
    float
        H² statistic
    """
    if df <= 0:
        return 1.0
    
    return max(1.0, q / df)


def tau_squared_moments(q: float, weights: Array) -> float:
    """
    Method of moments estimator for tau².
    
    Parameters
    ----------
    q : float
        Q statistic
    weights : array-like
        Weights
    
    Returns
    -------
    float
        Tau² estimate
    """
    weights = np.asarray(weights)
    k = len(weights)
    
    if k <= 1:
        return 0.0
    
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights**2)
    
    c = sum_w - sum_w2 / sum_w
    
    if c <= 0:
        return 0.0
    
    tau2 = max(0.0, (q - (k - 1)) / c)
    
    return tau2


def continuity_correction(a: Array, b: Array, c: Array, d: Array, 
                         correction: float = 0.5) -> Tuple[Array, Array, Array, Array]:
    """
    Apply continuity correction to 2x2 table data.
    
    Parameters
    ----------
    a, b, c, d : array-like
        Cell counts
    correction : float, default 0.5
        Correction value to add
    
    Returns
    -------
    tuple of arrays
        Corrected cell counts
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    
    # Apply correction where there are zero cells
    zero_mask = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    
    a[zero_mask] += correction
    b[zero_mask] += correction
    c[zero_mask] += correction
    d[zero_mask] += correction
    
    return a, b, c, d


def wilson_score_interval(k: Array, n: Array, alpha: float = 0.05) -> Tuple[Array, Array]:
    """
    Wilson score confidence interval for proportions.
    
    Parameters
    ----------
    k : array-like
        Number of successes
    n : array-like
        Number of trials
    alpha : float, default 0.05
        Significance level
    
    Returns
    -------
    tuple of arrays
        Lower and upper confidence bounds
    """
    k = np.asarray(k)
    n = np.asarray(n)
    
    z = stats.norm.ppf(1 - alpha / 2)
    z2 = z**2
    
    p = k / n
    
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    margin = z * safe_sqrt(p * (1 - p) / n + z2 / (4 * n**2)) / denominator
    
    lower = center - margin
    upper = center + margin
    
    return np.clip(lower, 0, 1), np.clip(upper, 0, 1)


def clopper_pearson_interval(k: Array, n: Array, alpha: float = 0.05) -> Tuple[Array, Array]:
    """
    Clopper-Pearson exact confidence interval for proportions.
    
    Parameters
    ----------
    k : array-like
        Number of successes
    n : array-like
        Number of trials
    alpha : float, default 0.05
        Significance level
    
    Returns
    -------
    tuple of arrays
        Lower and upper confidence bounds
    """
    k = np.asarray(k)
    n = np.asarray(n)
    
    lower = np.zeros_like(k, dtype=float)
    upper = np.ones_like(k, dtype=float)
    
    # Lower bound
    mask = k > 0
    if np.any(mask):
        lower[mask] = stats.beta.ppf(alpha / 2, k[mask], n[mask] - k[mask] + 1)
    
    # Upper bound
    mask = k < n
    if np.any(mask):
        upper[mask] = stats.beta.ppf(1 - alpha / 2, k[mask] + 1, n[mask] - k[mask])
    
    return lower, upper