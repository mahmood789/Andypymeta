"""Fixed and random effects meta-analysis models."""

import numpy as np
import scipy.stats as stats
from ..estimators import tau2_estimators


def fixed_effects(effect_sizes, variances):
    """
    Conduct fixed effects meta-analysis.
    
    Parameters
    ----------
    effect_sizes : array_like
        Study effect sizes
    variances : array_like
        Within-study variances
        
    Returns
    -------
    dict
        Dictionary with pooled effect size and statistics
    """
    effect_sizes = np.asarray(effect_sizes)
    variances = np.asarray(variances)
    
    weights = 1 / variances
    pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
    pooled_variance = 1 / np.sum(weights)
    pooled_se = np.sqrt(pooled_variance)
    
    # Test statistics
    z_value = pooled_effect / pooled_se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
    
    # Confidence interval
    ci_lower = pooled_effect - 1.96 * pooled_se
    ci_upper = pooled_effect + 1.96 * pooled_se
    
    # Heterogeneity test
    Q = np.sum(weights * (effect_sizes - pooled_effect)**2)
    df = len(effect_sizes) - 1
    Q_p_value = 1 - stats.chi2.cdf(Q, df) if Q > 0 else 1.0
    
    return {
        'pooled_effect': pooled_effect,
        'se': pooled_se,
        'variance': pooled_variance,
        'z_value': z_value,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'Q': Q,
        'Q_df': df,
        'Q_p_value': Q_p_value,
        'method': 'Fixed Effects'
    }


def random_effects(effect_sizes, variances, tau2_method="DL"):
    """
    Conduct random effects meta-analysis.
    
    Parameters
    ----------
    effect_sizes : array_like
        Study effect sizes
    variances : array_like
        Within-study variances
    tau2_method : str, default "DL"
        Method for estimating tau-squared
        
    Returns
    -------
    dict
        Dictionary with pooled effect size and statistics
    """
    effect_sizes = np.asarray(effect_sizes)
    variances = np.asarray(variances)
    
    # Estimate tau-squared
    tau2_result = tau2_estimators(effect_sizes, variances, method=tau2_method)
    tau2 = tau2_result['tau2']
    
    # Random effects weights
    weights = 1 / (variances + tau2)
    pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
    pooled_variance = 1 / np.sum(weights)
    pooled_se = np.sqrt(pooled_variance)
    
    # Test statistics
    z_value = pooled_effect / pooled_se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
    
    # Confidence interval
    ci_lower = pooled_effect - 1.96 * pooled_se
    ci_upper = pooled_effect + 1.96 * pooled_se
    
    # I-squared statistic
    Q = tau2_result.get('Q', 0)
    I2 = tau2_result.get('I2', 0)
    
    return {
        'pooled_effect': pooled_effect,
        'se': pooled_se,
        'variance': pooled_variance,
        'tau2': tau2,
        'tau': np.sqrt(tau2),
        'z_value': z_value,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'Q': Q,
        'Q_df': len(effect_sizes) - 1,
        'Q_p_value': tau2_result.get('p_value', 1.0),
        'I2': I2,
        'method': f'Random Effects ({tau2_method})'
    }


def reference_re_summary(effect_sizes, variances, tau2):
    """Reference implementation for random effects summary for testing."""
    weights = 1 / (variances + tau2)
    pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
    pooled_variance = 1 / np.sum(weights)
    return pooled_effect, pooled_variance