"""Robust variance estimation methods."""

import numpy as np
import warnings

# Optional dependencies  
try:
    from scipy.cluster.hierarchy import linkage, fcluster
    HAS_SCIPY_CLUSTER = True
except ImportError:
    HAS_SCIPY_CLUSTER = False


def robust_variance_estimation(effect_sizes, variances, cluster_ids=None, 
                              small_sample_correction=True):
    """
    Robust variance estimation for dependent effect sizes.
    
    Useful when studies contribute multiple effect sizes or there are
    other dependencies between studies.
    
    Parameters
    ----------
    effect_sizes : array_like
        Effect sizes
    variances : array_like
        Within-study variances
    cluster_ids : array_like, optional
        Cluster identifiers for dependent effects
    small_sample_correction : bool, default True
        Apply small sample correction
        
    Returns
    -------
    dict
        Dictionary with RVE results
    """
    effect_sizes = np.asarray(effect_sizes)
    variances = np.asarray(variances)
    k = len(effect_sizes)
    
    if cluster_ids is None:
        cluster_ids = np.arange(k)  # Each effect size is its own cluster
    else:
        cluster_ids = np.asarray(cluster_ids)
    
    # Weights
    weights = 1 / variances
    
    # Weighted mean
    pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
    
    # Residuals
    residuals = effect_sizes - pooled_effect
    
    # Calculate robust variance
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    
    # Cluster-robust variance calculation
    cluster_contributions = []
    
    for cluster in unique_clusters:
        cluster_mask = cluster_ids == cluster
        cluster_weights = weights[cluster_mask]
        cluster_residuals = residuals[cluster_mask]
        
        # Sum of weighted residuals within cluster
        cluster_sum = np.sum(cluster_weights * cluster_residuals)
        cluster_contributions.append(cluster_sum)
    
    cluster_contributions = np.array(cluster_contributions)
    
    # Robust variance
    meat = np.sum(cluster_contributions**2)
    bread = np.sum(weights)**2
    robust_variance = meat / bread
    
    # Small sample correction
    if small_sample_correction and n_clusters > 1:
        correction_factor = (n_clusters / (n_clusters - 1)) * (k - 1) / (k - 1)
        robust_variance *= correction_factor
    
    robust_se = np.sqrt(robust_variance)
    
    # Test statistics
    t_value = pooled_effect / robust_se
    df = n_clusters - 1
    
    # Approximate p-value (would need t-distribution for exact)
    from scipy import stats
    p_value = 2 * (1 - stats.t.cdf(abs(t_value), df))
    
    return {
        'pooled_effect': pooled_effect,
        'robust_se': robust_se,
        'robust_variance': robust_variance,
        't_value': t_value,
        'df': df,
        'p_value': p_value,
        'ci_lower': pooled_effect - stats.t.ppf(0.975, df) * robust_se,
        'ci_upper': pooled_effect + stats.t.ppf(0.975, df) * robust_se,
        'n_clusters': n_clusters,
        'n_effects': k,
        'method': 'Robust Variance Estimation'
    }


def hierarchical_effects_model(effect_sizes, variances, level_1_ids, level_2_ids):
    """
    Three-level meta-analysis for hierarchical effect sizes.
    
    Parameters
    ----------
    effect_sizes : array_like
        Effect sizes
    variances : array_like
        Sampling variances
    level_1_ids : array_like
        Level 1 identifiers (e.g., effect size within study)
    level_2_ids : array_like
        Level 2 identifiers (e.g., study within cluster)
        
    Returns
    -------
    dict
        Dictionary with hierarchical model results
    """
    effect_sizes = np.asarray(effect_sizes)
    variances = np.asarray(variances)
    level_1_ids = np.asarray(level_1_ids)
    level_2_ids = np.asarray(level_2_ids)
    
    # Simplified implementation - would need proper multilevel modeling
    # For now, use RVE with level 2 clusters
    return robust_variance_estimation(effect_sizes, variances, 
                                    cluster_ids=level_2_ids)


def sensitivity_analysis_rve(effect_sizes, variances, cluster_ids=None, 
                            rho_values=[0, 0.2, 0.5, 0.8]):
    """
    Sensitivity analysis for different correlation assumptions in RVE.
    
    Parameters
    ----------
    effect_sizes : array_like
        Effect sizes
    variances : array_like
        Within-study variances
    cluster_ids : array_like, optional
        Cluster identifiers
    rho_values : list, default [0, 0.2, 0.5, 0.8]
        Correlation values to test
        
    Returns
    -------
    dict
        Dictionary with sensitivity analysis results
    """
    results = {}
    
    for rho in rho_values:
        # This is a simplified implementation
        # Would need to adjust variances based on correlation assumption
        
        adjusted_variances = variances  # Placeholder
        
        result = robust_variance_estimation(effect_sizes, adjusted_variances, 
                                          cluster_ids=cluster_ids)
        
        results[f'rho_{rho}'] = {
            'pooled_effect': result['pooled_effect'],
            'robust_se': result['robust_se'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper']
        }
    
    return {
        'sensitivity_results': results,
        'method': 'RVE Sensitivity Analysis',
        'note': 'Simplified implementation for testing'
    }