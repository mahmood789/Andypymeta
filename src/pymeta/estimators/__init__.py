"""Tau-squared estimation methods for random effects meta-analysis."""

import numpy as np
import scipy.stats as stats


def tau2_estimators(effect_sizes, variances, method="DL"):
    """
    Estimate between-study variance (tau-squared) using various methods.
    
    Parameters
    ----------
    effect_sizes : array_like
        Study effect sizes
    variances : array_like
        Within-study variances
    method : str, default "DL"
        Estimation method: "DL" (DerSimonian-Laird), "REML", "ML", "HE", "HS"
        
    Returns
    -------
    dict
        Dictionary with tau2 estimate and method information
    """
    effect_sizes = np.asarray(effect_sizes)
    variances = np.asarray(variances)
    k = len(effect_sizes)
    
    if method == "DL":
        # DerSimonian-Laird method
        weights = 1 / variances
        sum_weights = np.sum(weights)
        weighted_mean = np.sum(weights * effect_sizes) / sum_weights
        
        Q = np.sum(weights * (effect_sizes - weighted_mean)**2)
        C = sum_weights - np.sum(weights**2) / sum_weights
        
        tau2_dl = max(0, (Q - (k - 1)) / C)
        
        return {
            'tau2': tau2_dl,
            'method': 'DerSimonian-Laird',
            'Q': Q,
            'df': k - 1,
            'p_value': 1 - stats.chi2.cdf(Q, k - 1) if Q > 0 else 1.0,
            'I2': max(0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
        }
        
    elif method == "HE":
        # Hedges estimator
        weights = 1 / variances
        sum_weights = np.sum(weights)
        weighted_mean = np.sum(weights * effect_sizes) / sum_weights
        
        Q = np.sum(weights * (effect_sizes - weighted_mean)**2)
        tau2_he = max(0, (Q - (k - 1)) / (k - 1))
        
        return {
            'tau2': tau2_he,
            'method': 'Hedges',
            'Q': Q,
            'df': k - 1,
            'p_value': 1 - stats.chi2.cdf(Q, k - 1) if Q > 0 else 1.0,
            'I2': max(0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
        }
        
    elif method == "HS":
        # Hunter-Schmidt estimator
        weights = 1 / variances
        sum_weights = np.sum(weights)
        weighted_mean = np.sum(weights * effect_sizes) / sum_weights
        
        var_observed = np.sum(weights * (effect_sizes - weighted_mean)**2) / sum_weights
        mean_within_var = np.sum(weights * variances) / sum_weights
        tau2_hs = max(0, var_observed - mean_within_var)
        
        return {
            'tau2': tau2_hs,
            'method': 'Hunter-Schmidt',
            'var_observed': var_observed,
            'mean_within_var': mean_within_var
        }
        
    else:
        raise ValueError(f"Unknown method: {method}")


def reference_tau2_dl(effect_sizes, variances):
    """Reference implementation of DerSimonian-Laird tau2 for testing."""
    effect_sizes = np.asarray(effect_sizes)
    variances = np.asarray(variances)
    k = len(effect_sizes)
    
    weights = 1 / variances
    sum_weights = np.sum(weights)
    weighted_mean = np.sum(weights * effect_sizes) / sum_weights
    
    Q = np.sum(weights * (effect_sizes - weighted_mean)**2)
    C = sum_weights - np.sum(weights**2) / sum_weights
    
    return max(0, (Q - (k - 1)) / C)