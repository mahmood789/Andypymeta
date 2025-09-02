"""Paule-Mandel tau² estimator."""

import numpy as np
from typing import List
from ..typing import MetaPoint
from ..registries import register_estimator
from ..errors import EstimationError, ConvergenceError
from ..config import config


@register_estimator("PM")
def tau2_paule_mandel(points: List[MetaPoint], 
                     max_iterations: int = None,
                     tolerance: float = None) -> float:
    """Paule-Mandel estimator for between-study variance (tau²).
    
    The Paule-Mandel estimator is an iterative method that tends to perform
    better than DerSimonian-Laird, especially with few studies.
    
    Args:
        points: List of MetaPoint objects
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Estimated tau² value
        
    Raises:
        EstimationError: If estimation fails
        ConvergenceError: If algorithm fails to converge
    
    References:
        Paule, R. C., & Mandel, J. (1982). Consensus values and weighting factors.
        Journal of Research of the National Bureau of Standards, 87(5), 377-385.
    """
    if len(points) < 2:
        return 0.0
    
    if max_iterations is None:
        max_iterations = config.max_iterations
    if tolerance is None:
        tolerance = config.tolerance
    
    try:
        # Extract effects and variances
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        k = len(points)
        
        # Initial estimate using DerSimonian-Laird
        from .tau2_dl import tau2_dersimonian_laird_simple
        tau2 = tau2_dersimonian_laird_simple(points)
        
        # Iterative estimation
        for iteration in range(max_iterations):
            tau2_old = tau2
            
            # Update weights
            weights = 1.0 / (variances + tau2)
            
            # Calculate weighted mean
            weighted_mean = np.sum(weights * effects) / np.sum(weights)
            
            # Calculate Q statistic
            q_statistic = np.sum(weights * (effects - weighted_mean) ** 2)
            
            # Update tau² using Paule-Mandel equation
            # E[Q] = k - 1, so we solve for tau² such that Q = k - 1
            if k > 1:
                # Use bisection method to solve for tau²
                tau2 = _solve_paule_mandel_equation(effects, variances, k - 1)
            else:
                tau2 = 0.0
            
            # Check convergence
            if abs(tau2 - tau2_old) < tolerance:
                break
        else:
            raise ConvergenceError(f"Paule-Mandel failed to converge after {max_iterations} iterations")
        
        return max(0, tau2)
        
    except ConvergenceError:
        raise
    except Exception as e:
        raise EstimationError(f"Paule-Mandel estimation failed: {e}")


def _solve_paule_mandel_equation(effects: np.ndarray, 
                                variances: np.ndarray,
                                target_q: float,
                                max_iterations: int = 100,
                                tolerance: float = 1e-6) -> float:
    """Solve the Paule-Mandel equation using bisection method.
    
    We need to find tau² such that Q(tau²) = target_q, where:
    Q(tau²) = Σ w_i(tau²) * (y_i - ȳ(tau²))²
    w_i(tau²) = 1 / (v_i + tau²)
    ȳ(tau²) = Σ w_i(tau²) * y_i / Σ w_i(tau²)
    
    Args:
        effects: Effect sizes
        variances: Effect variances
        target_q: Target Q value (usually k-1)
        max_iterations: Maximum iterations for bisection
        tolerance: Convergence tolerance
        
    Returns:
        tau² value that achieves target Q
    """
    def q_function(tau2):
        """Calculate Q statistic for given tau²."""
        weights = 1.0 / (variances + tau2)
        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        return np.sum(weights * (effects - weighted_mean) ** 2)
    
    # Initial bounds
    tau2_lower = 0.0
    tau2_upper = 10.0  # Start with reasonable upper bound
    
    # Expand upper bound if needed
    while q_function(tau2_upper) > target_q and tau2_upper < 1000:
        tau2_upper *= 2
    
    # Bisection method
    for _ in range(max_iterations):
        tau2_mid = (tau2_lower + tau2_upper) / 2
        q_mid = q_function(tau2_mid)
        
        if abs(q_mid - target_q) < tolerance:
            return tau2_mid
        
        if q_mid > target_q:
            tau2_lower = tau2_mid
        else:
            tau2_upper = tau2_mid
        
        if tau2_upper - tau2_lower < tolerance:
            return tau2_mid
    
    # Return midpoint if not converged
    return (tau2_lower + tau2_upper) / 2


@register_estimator("PM_simple")
def tau2_paule_mandel_simple(points: List[MetaPoint]) -> float:
    """Simplified Paule-Mandel estimator with fixed number of iterations.
    
    This version uses a fixed small number of iterations for faster computation
    at the cost of some accuracy.
    
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
        k = len(points)
        
        # Start with DL estimate
        from .tau2_dl import tau2_dersimonian_laird_simple
        tau2 = tau2_dersimonian_laird_simple(points)
        
        # Fixed number of PM iterations
        for _ in range(5):  # Usually converges quickly
            weights = 1.0 / (variances + tau2)
            weighted_mean = np.sum(weights * effects) / np.sum(weights)
            q_statistic = np.sum(weights * (effects - weighted_mean) ** 2)
            
            # Simple PM update
            if k > 1:
                # Approximate solution
                weights_sum = np.sum(weights)
                weights_sq_sum = np.sum(weights ** 2)
                c = weights_sum - (weights_sq_sum / weights_sum)
                
                if c > 0:
                    tau2_new = max(0, (q_statistic - (k - 1)) / c)
                    tau2 = 0.5 * tau2 + 0.5 * tau2_new  # Damped update
                else:
                    break
        
        return max(0, tau2)
        
    except Exception as e:
        raise EstimationError(f"Simple Paule-Mandel estimation failed: {e}")


def _paule_mandel_variance(points: List[MetaPoint], tau2: float) -> float:
    """Calculate variance of Paule-Mandel tau² estimate.
    
    Args:
        points: List of MetaPoint objects
        tau2: Estimated tau² value
        
    Returns:
        Estimated variance of tau² estimate
    """
    # This is an approximation - exact variance is complex
    variances = np.array([p.variance for p in points])
    weights = 1.0 / (variances + tau2)
    
    # Approximate variance using inverse Fisher information
    # This is a rough approximation
    n = len(points)
    if n > 2:
        var_tau2 = 2 * tau2 ** 2 / (n - 1)
    else:
        var_tau2 = tau2
    
    return var_tau2