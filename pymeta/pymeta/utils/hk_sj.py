"""
Hunter-Schmidt and Sidik-Jonkman tau-squared estimators.

This module implements specialized tau-squared estimation methods.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from typing import Optional
from ..types import Array


def hunter_schmidt_tau2(effects: Array, variances: Array, weights: Array) -> float:
    """
    Hunter-Schmidt tau-squared estimator.
    
    This estimator is popular in psychometric meta-analysis and uses
    the sample variance of observed effect sizes.
    
    Parameters
    ----------
    effects : array-like
        Effect sizes
    variances : array-like
        Within-study variances
    weights : array-like
        Weights (typically 1/variance)
    
    Returns
    -------
    float
        Tau-squared estimate
    """
    effects = np.asarray(effects)
    variances = np.asarray(variances)
    k = len(effects)
    
    if k <= 1:
        return 0.0
    
    # Sample variance of observed effect sizes
    var_observed = np.var(effects, ddof=1)
    
    # Average within-study variance
    avg_within_var = np.mean(variances)
    
    # Hunter-Schmidt estimator
    tau2 = max(0.0, var_observed - avg_within_var)
    
    return tau2


def sidik_jonkman_tau2(effects: Array, variances: Array, weights: Array) -> float:
    """
    Sidik-Jonkman tau-squared estimator.
    
    This estimator uses the variance of standardized effect sizes.
    
    Parameters
    ----------
    effects : array-like
        Effect sizes
    variances : array-like
        Within-study variances
    weights : array-like
        Weights (typically 1/variance)
    
    Returns
    -------
    float
        Tau-squared estimate
    """
    effects = np.asarray(effects)
    variances = np.asarray(variances)
    weights = np.asarray(weights)
    k = len(effects)
    
    if k <= 1:
        return 0.0
    
    # Weighted mean
    sum_weights = np.sum(weights)
    weighted_mean = np.sum(weights * effects) / sum_weights
    
    # Standardized effect sizes
    se = np.sqrt(variances)
    z_scores = (effects - weighted_mean) / se
    
    # Sample variance of z-scores
    var_z = np.var(z_scores, ddof=1)
    
    # SJ estimator
    tau2 = max(0.0, (var_z - 1) * np.mean(variances))
    
    return tau2


def empirical_bayes_tau2(effects: Array, variances: Array, weights: Array,
                        max_iter: int = 100, tol: float = 1e-6) -> float:
    """
    Empirical Bayes tau-squared estimator.
    
    This estimator maximizes the marginal likelihood.
    
    Parameters
    ----------
    effects : array-like
        Effect sizes
    variances : array-like
        Within-study variances
    weights : array-like
        Weights (typically 1/variance)
    max_iter : int, default 100
        Maximum number of iterations
    tol : float, default 1e-6
        Convergence tolerance
    
    Returns
    -------
    float
        Tau-squared estimate
    """
    effects = np.asarray(effects)
    variances = np.asarray(variances)
    k = len(effects)
    
    if k <= 1:
        return 0.0
    
    def marginal_log_likelihood(tau2):
        """Marginal log-likelihood function."""
        total_var = variances + tau2
        weights_new = 1.0 / total_var
        
        # Weighted mean
        sum_weights = np.sum(weights_new)
        weighted_mean = np.sum(weights_new * effects) / sum_weights
        
        # Log-likelihood components
        ll = -0.5 * np.sum(np.log(2 * np.pi * total_var))
        ll -= 0.5 * np.sum(weights_new * (effects - weighted_mean)**2)
        
        return ll
    
    # Optimize tau-squared
    result = minimize_scalar(
        lambda x: -marginal_log_likelihood(x),
        bounds=(0, 10 * np.var(effects)),
        method='bounded'
    )
    
    return max(0.0, result.x)


def mandel_paule_tau2(effects: Array, variances: Array, weights: Array,
                     max_iter: int = 100, tol: float = 1e-6) -> float:
    """
    Mandel-Paule tau-squared estimator.
    
    This estimator solves an equation iteratively.
    
    Parameters
    ----------
    effects : array-like
        Effect sizes
    variances : array-like
        Within-study variances
    weights : array-like
        Weights (typically 1/variance)
    max_iter : int, default 100
        Maximum number of iterations
    tol : float, default 1e-6
        Convergence tolerance
    
    Returns
    -------
    float
        Tau-squared estimate
    """
    effects = np.asarray(effects)
    variances = np.asarray(variances)
    k = len(effects)
    
    if k <= 1:
        return 0.0
    
    def mp_equation(tau2):
        """Mandel-Paule equation to solve."""
        weights_new = 1.0 / (variances + tau2)
        sum_weights = np.sum(weights_new)
        weighted_mean = np.sum(weights_new * effects) / sum_weights
        
        q = np.sum(weights_new * (effects - weighted_mean)**2)
        return q - (k - 1)
    
    # Initial estimate using DerSimonian-Laird
    weights_initial = 1.0 / variances
    sum_weights = np.sum(weights_initial)
    weighted_mean = np.sum(weights_initial * effects) / sum_weights
    q_initial = np.sum(weights_initial * (effects - weighted_mean)**2)
    
    c = sum_weights - np.sum(weights_initial**2) / sum_weights
    tau2_initial = max(0.0, (q_initial - (k - 1)) / c)
    
    # Iterative solution
    tau2 = tau2_initial
    
    for i in range(max_iter):
        # Compute new weights
        weights_new = 1.0 / (variances + tau2)
        sum_weights_new = np.sum(weights_new)
        weighted_mean_new = np.sum(weights_new * effects) / sum_weights_new
        
        # Compute Q
        q_new = np.sum(weights_new * (effects - weighted_mean_new)**2)
        
        # Update tau2
        c_new = sum_weights_new - np.sum(weights_new**2) / sum_weights_new
        tau2_new = max(0.0, (q_new - (k - 1)) / c_new)
        
        # Check convergence
        if abs(tau2_new - tau2) < tol:
            break
        
        tau2 = tau2_new
    
    return tau2


def restricted_ml_tau2(effects: Array, variances: Array, weights: Array) -> float:
    """
    Restricted Maximum Likelihood (REML) tau-squared estimator.
    
    Parameters
    ----------
    effects : array-like
        Effect sizes
    variances : array-like
        Within-study variances
    weights : array-like
        Weights (typically 1/variance)
    
    Returns
    -------
    float
        Tau-squared estimate
    """
    effects = np.asarray(effects)
    variances = np.asarray(variances)
    k = len(effects)
    
    if k <= 1:
        return 0.0
    
    def reml_objective(tau2):
        """REML objective function (negative log-likelihood)."""
        total_var = variances + tau2
        weights_new = 1.0 / total_var
        
        # Weighted mean
        sum_weights = np.sum(weights_new)
        weighted_mean = np.sum(weights_new * effects) / sum_weights
        
        # REML log-likelihood
        ll = -0.5 * np.sum(np.log(total_var))
        ll -= 0.5 * np.log(sum_weights)
        ll -= 0.5 * np.sum(weights_new * (effects - weighted_mean)**2)
        
        return -ll
    
    # Optimize tau-squared
    result = minimize_scalar(
        reml_objective,
        bounds=(0, 10 * np.var(effects)),
        method='bounded'
    )
    
    return max(0.0, result.x)


def maximum_likelihood_tau2(effects: Array, variances: Array, weights: Array) -> float:
    """
    Maximum Likelihood (ML) tau-squared estimator.
    
    Parameters
    ----------
    effects : array-like
        Effect sizes
    variances : array-like
        Within-study variances
    weights : array-like
        Weights (typically 1/variance)
    
    Returns
    -------
    float
        Tau-squared estimate
    """
    effects = np.asarray(effects)
    variances = np.asarray(variances)
    k = len(effects)
    
    if k <= 1:
        return 0.0
    
    def ml_objective(tau2):
        """ML objective function (negative log-likelihood)."""
        total_var = variances + tau2
        weights_new = 1.0 / total_var
        
        # Weighted mean
        sum_weights = np.sum(weights_new)
        weighted_mean = np.sum(weights_new * effects) / sum_weights
        
        # ML log-likelihood
        ll = -0.5 * k * np.log(2 * np.pi)
        ll -= 0.5 * np.sum(np.log(total_var))
        ll -= 0.5 * np.sum(weights_new * (effects - weighted_mean)**2)
        
        return -ll
    
    # Optimize tau-squared
    result = minimize_scalar(
        ml_objective,
        bounds=(0, 10 * np.var(effects)),
        method='bounded'
    )
    
    return max(0.0, result.x)