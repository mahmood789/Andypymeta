"""Prediction intervals and inference methods."""

import numpy as np
import scipy.stats as stats


def prediction_intervals(pooled_effect, pooled_se, tau2, k, alpha=0.05):
    """
    Calculate prediction intervals for a new study.
    
    Parameters
    ----------
    pooled_effect : float
        Pooled effect size from meta-analysis
    pooled_se : float
        Standard error of pooled effect
    tau2 : float
        Between-study variance
    k : int
        Number of studies in meta-analysis
    alpha : float, default 0.05
        Significance level
        
    Returns
    -------
    dict
        Dictionary with prediction interval bounds
    """
    if k <= 2:
        return {
            'pi_lower': np.nan,
            'pi_upper': np.nan,
            'warning': 'Prediction intervals require k > 2 studies'
        }
    
    # Prediction standard error
    pred_se = np.sqrt(pooled_se**2 + tau2)
    
    # t-distribution critical value
    df = k - 2
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # Prediction interval
    pi_lower = pooled_effect - t_crit * pred_se
    pi_upper = pooled_effect + t_crit * pred_se
    
    return {
        'pi_lower': pi_lower,
        'pi_upper': pi_upper,
        'pred_se': pred_se,
        't_crit': t_crit,
        'df': df,
        'alpha': alpha
    }


def heterogeneity_tests(effect_sizes, variances, pooled_effect=None):
    """
    Perform tests for heterogeneity.
    
    Parameters
    ----------
    effect_sizes : array_like
        Study effect sizes
    variances : array_like
        Within-study variances  
    pooled_effect : float, optional
        Pooled effect size (calculated if not provided)
        
    Returns
    -------
    dict
        Dictionary with heterogeneity test statistics
    """
    effect_sizes = np.asarray(effect_sizes)
    variances = np.asarray(variances)
    k = len(effect_sizes)
    
    if pooled_effect is None:
        weights = 1 / variances
        pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
    
    weights = 1 / variances
    Q = np.sum(weights * (effect_sizes - pooled_effect)**2)
    df = k - 1
    
    # Cochran's Q test
    Q_p_value = 1 - stats.chi2.cdf(Q, df) if Q > 0 else 1.0
    
    # I-squared
    I2 = max(0, (Q - df) / Q) if Q > 0 else 0.0
    
    # H-squared  
    H2 = Q / df if df > 0 else 1.0
    
    # Tau-squared (DL estimator)
    sum_weights = np.sum(weights)
    C = sum_weights - np.sum(weights**2) / sum_weights
    tau2 = max(0, (Q - df) / C) if C > 0 else 0.0
    
    return {
        'Q': Q,
        'df': df,
        'Q_p_value': Q_p_value,
        'I2': I2,
        'H2': H2,
        'tau2': tau2,
        'tau': np.sqrt(tau2)
    }