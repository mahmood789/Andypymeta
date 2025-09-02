"""REML (Restricted Maximum Likelihood) tau² estimator."""

import numpy as np
from typing import List, Optional
from scipy.optimize import minimize_scalar, OptimizeResult
from ..typing import MetaPoint
from ..registries import register_estimator
from ..errors import EstimationError, ConvergenceError
from ..config import config


@register_estimator("REML")
def tau2_reml(points: List[MetaPoint], 
              max_iterations: int = None,
              tolerance: float = None) -> float:
    """REML estimator for between-study variance (tau²).
    
    The REML estimator tends to be less biased than DerSimonian-Laird,
    especially with small numbers of studies.
    
    Args:
        points: List of MetaPoint objects
        max_iterations: Maximum number of iterations (unused, kept for compatibility)
        tolerance: Convergence tolerance
        
    Returns:
        Estimated tau² value
        
    Raises:
        EstimationError: If estimation fails
        ConvergenceError: If optimization fails to converge
    
    References:
        Viechtbauer, W. (2005). Bias and efficiency of meta-analytic variance estimators
        in the random-effects model. Journal of Educational and Behavioral Statistics, 30(3), 261-293.
    """
    if len(points) < 2:
        return 0.0
    
    if tolerance is None:
        tolerance = config.tolerance
    
    try:
        # Extract effects and variances
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        
        # Define REML objective function
        def reml_objective(tau2):
            """REML log-likelihood (negative for minimization)."""
            return -_reml_log_likelihood(effects, variances, tau2)
        
        # Find optimal tau² using scalar minimization
        result = minimize_scalar(
            reml_objective,
            bounds=(0, 100),  # Reasonable bounds for tau²
            method='bounded',
            options={'xatol': tolerance}
        )
        
        if not result.success:
            raise ConvergenceError(f"REML optimization failed: {result.message}")
        
        tau2_estimate = result.x
        return max(0, tau2_estimate)
        
    except ConvergenceError:
        raise
    except Exception as e:
        raise EstimationError(f"REML estimation failed: {e}")


def _reml_log_likelihood(effects: np.ndarray, variances: np.ndarray, tau2: float) -> float:
    """Calculate REML log-likelihood for given tau².
    
    Args:
        effects: Effect sizes
        variances: Effect variances  
        tau2: Between-study variance parameter
        
    Returns:
        REML log-likelihood value
    """
    n = len(effects)
    
    # Total variances
    total_variances = variances + tau2
    
    # Weights
    weights = 1.0 / total_variances
    
    # Weighted mean
    sum_weights = np.sum(weights)
    weighted_mean = np.sum(weights * effects) / sum_weights
    
    # Log-likelihood components
    # 1. Data likelihood part
    data_ll = -0.5 * np.sum(np.log(total_variances)) - \
              0.5 * np.sum(weights * (effects - weighted_mean) ** 2)
    
    # 2. REML correction (penalty for estimating fixed effect)
    reml_correction = -0.5 * np.log(sum_weights)
    
    # Total REML log-likelihood
    reml_ll = data_ll + reml_correction
    
    return reml_ll


@register_estimator("REML_alt")
def tau2_reml_alternative(points: List[MetaPoint]) -> float:
    """Alternative REML implementation using profile likelihood.
    
    This implementation uses a different optimization approach that may
    be more stable in some cases.
    
    Args:
        points: List of MetaPoint objects
        
    Returns:
        Estimated tau² value
    """
    if len(points) < 2:
        return 0.0
    
    try:
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        
        # Grid search for initial value
        tau2_candidates = np.logspace(-4, 2, 50)  # From 0.0001 to 100
        best_ll = -np.inf
        best_tau2 = 0.0
        
        for tau2 in tau2_candidates:
            ll = _reml_log_likelihood(effects, variances, tau2)
            if ll > best_ll:
                best_ll = ll
                best_tau2 = tau2
        
        # Refine with local optimization around best candidate
        def objective(tau2):
            return -_reml_log_likelihood(effects, variances, tau2)
        
        result = minimize_scalar(
            objective,
            bounds=(max(0, best_tau2 - 1), best_tau2 + 1),
            method='bounded'
        )
        
        if result.success:
            return max(0, result.x)
        else:
            return max(0, best_tau2)
        
    except Exception as e:
        raise EstimationError(f"Alternative REML estimation failed: {e}")


def _reml_information_matrix(effects: np.ndarray, variances: np.ndarray, tau2: float) -> float:
    """Calculate Fisher information for tau² at REML estimate.
    
    Args:
        effects: Effect sizes
        variances: Effect variances
        tau2: Tau² estimate
        
    Returns:
        Fisher information value
    """
    total_variances = variances + tau2
    weights = 1.0 / total_variances
    
    # Second derivative of log-likelihood w.r.t. tau²
    # This is an approximation of the Fisher information
    
    # Components of the Hessian
    sum_weights = np.sum(weights)
    sum_weights_sq = np.sum(weights ** 2)
    weighted_mean = np.sum(weights * effects) / sum_weights
    
    # Fisher information (negative expected second derivative)
    fisher_info = 0.5 * sum_weights_sq - \
                  0.5 * sum_weights_sq ** 2 / sum_weights + \
                  np.sum(weights ** 2 * (effects - weighted_mean) ** 2)
    
    return fisher_info


def reml_confidence_interval(points: List[MetaPoint], 
                           tau2_estimate: float,
                           confidence_level: float = 0.95) -> tuple:
    """Calculate confidence interval for REML tau² estimate.
    
    Args:
        points: List of MetaPoint objects
        tau2_estimate: REML tau² estimate
        confidence_level: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    effects = np.array([p.effect for p in points])
    variances = np.array([p.variance for p in points])
    
    # Calculate Fisher information
    fisher_info = _reml_information_matrix(effects, variances, tau2_estimate)
    
    if fisher_info <= 0:
        # Fallback to simple bounds
        return (0, tau2_estimate * 3)
    
    # Standard error
    se_tau2 = 1.0 / np.sqrt(fisher_info)
    
    # Critical value
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha / 2)
    
    # Confidence interval
    lower = max(0, tau2_estimate - z_critical * se_tau2)
    upper = tau2_estimate + z_critical * se_tau2
    
    return (lower, upper)


@register_estimator("ML")
def tau2_maximum_likelihood(points: List[MetaPoint]) -> float:
    """Maximum Likelihood estimator for tau² (without REML correction).
    
    This is the standard ML estimator, which tends to be biased downward
    compared to REML.
    
    Args:
        points: List of MetaPoint objects
        
    Returns:
        Estimated tau² value
    """
    if len(points) < 2:
        return 0.0
    
    try:
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        
        def ml_objective(tau2):
            """ML log-likelihood (negative for minimization)."""
            total_variances = variances + tau2
            weights = 1.0 / total_variances
            weighted_mean = np.sum(weights * effects) / np.sum(weights)
            
            # ML log-likelihood (without REML correction)
            ml_ll = -0.5 * np.sum(np.log(total_variances)) - \
                    0.5 * np.sum(weights * (effects - weighted_mean) ** 2)
            
            return -ml_ll
        
        result = minimize_scalar(
            ml_objective,
            bounds=(0, 100),
            method='bounded'
        )
        
        if result.success:
            return max(0, result.x)
        else:
            # Fallback to DL estimate
            from .tau2_dl import tau2_dersimonian_laird_simple
            return tau2_dersimonian_laird_simple(points)
        
    except Exception as e:
        raise EstimationError(f"ML estimation failed: {e}")