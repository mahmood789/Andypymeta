"""O'Brien-Fleming and Pocock boundary functions for TSA."""

import numpy as np
from typing import List, Union
from scipy import stats

from ..errors import TSAError
from ..config import config


def obrien_fleming_boundary(information_fractions: Union[List[float], np.ndarray],
                           alpha: float = None) -> np.ndarray:
    """Calculate O'Brien-Fleming monitoring boundaries.
    
    Args:
        information_fractions: Fraction of total information at each analysis
        alpha: Overall Type I error rate
        
    Returns:
        Array of boundary Z-values
        
    References:
        O'Brien, P. C., & Fleming, T. R. (1979). A multiple testing procedure for
        clinical trials. Biometrics, 35(3), 549-556.
    """
    if alpha is None:
        alpha = config.tsa_alpha
    
    information_fractions = np.array(information_fractions)
    
    if not np.all((information_fractions > 0) & (information_fractions <= 1)):
        raise TSAError("Information fractions must be between 0 and 1")
    
    if not np.all(np.diff(information_fractions) >= 0):
        raise TSAError("Information fractions must be non-decreasing")
    
    try:
        # O'Brien-Fleming spending function: α(t) = 2{1 - Φ(z_{α/2}/√t)}
        # Boundary: z_k = z_{α/2} / √(t_k)
        z_alpha_half = stats.norm.ppf(1 - alpha / 2)
        boundaries = z_alpha_half / np.sqrt(information_fractions)
        
        return boundaries
        
    except Exception as e:
        raise TSAError(f"Failed to calculate O'Brien-Fleming boundaries: {e}")


def pocock_boundary(information_fractions: Union[List[float], np.ndarray],
                   alpha: float = None) -> np.ndarray:
    """Calculate Pocock monitoring boundaries.
    
    Args:
        information_fractions: Fraction of total information at each analysis
        alpha: Overall Type I error rate
        
    Returns:
        Array of boundary Z-values
        
    References:
        Pocock, S. J. (1977). Group sequential methods in the design and analysis
        of clinical trials. Biometrika, 64(2), 191-199.
    """
    if alpha is None:
        alpha = config.tsa_alpha
    
    information_fractions = np.array(information_fractions)
    
    if not np.all((information_fractions > 0) & (information_fractions <= 1)):
        raise TSAError("Information fractions must be between 0 and 1")
    
    try:
        # Pocock boundaries are constant across analyses
        # Need to solve for boundary value c such that overall α is maintained
        
        k = len(information_fractions)  # Number of analyses
        
        # Approximate Pocock boundary (exact requires numerical integration)
        if k == 1:
            boundary_value = stats.norm.ppf(1 - alpha / 2)
        else:
            # Approximate formula for Pocock boundary
            boundary_value = stats.norm.ppf(1 - alpha / (2 * k)) * np.sqrt(1 + 1/(2*k))
        
        boundaries = np.full(len(information_fractions), boundary_value)
        
        return boundaries
        
    except Exception as e:
        raise TSAError(f"Failed to calculate Pocock boundaries: {e}")


def lan_demets_boundary(information_fractions: Union[List[float], np.ndarray],
                       alpha: float = None,
                       spending_function: str = "obrien_fleming") -> np.ndarray:
    """Calculate Lan-DeMets boundaries with spending functions.
    
    Args:
        information_fractions: Fraction of total information at each analysis
        alpha: Overall Type I error rate
        spending_function: Type of spending function ("obrien_fleming", "pocock")
        
    Returns:
        Array of boundary Z-values
        
    References:
        Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for
        clinical trials. Biometrika, 70(3), 659-663.
    """
    if alpha is None:
        alpha = config.tsa_alpha
    
    information_fractions = np.array(information_fractions)
    
    if spending_function == "obrien_fleming":
        return obrien_fleming_boundary(information_fractions, alpha)
    elif spending_function == "pocock":
        return pocock_boundary(information_fractions, alpha)
    else:
        raise TSAError(f"Unknown spending function: {spending_function}")


def calculate_information_size(delta: float,
                              sigma: float = 1.0,
                              alpha: float = None,
                              beta: float = None) -> float:
    """Calculate required information size for meta-analysis.
    
    Args:
        delta: Effect size to detect
        sigma: Standard deviation (default 1.0 for standardized effects)
        alpha: Type I error rate
        beta: Type II error rate (1 - power)
        
    Returns:
        Required information size (effective sample size)
    """
    if alpha is None:
        alpha = config.tsa_alpha
    if beta is None:
        beta = config.tsa_beta
    
    if delta == 0:
        raise TSAError("Effect size delta cannot be zero")
    
    try:
        # Standard sample size formula for two-sided test
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(1 - beta)
        
        # Information size = (Z_α + Z_β)² * σ² / δ²
        information_size = ((z_alpha + z_beta) ** 2) * (sigma ** 2) / (delta ** 2)
        
        return information_size
        
    except Exception as e:
        raise TSAError(f"Failed to calculate information size: {e}")


def futility_boundary(information_fractions: Union[List[float], np.ndarray],
                     beta: float = None) -> np.ndarray:
    """Calculate futility boundaries for early stopping.
    
    Args:
        information_fractions: Fraction of total information at each analysis
        beta: Type II error rate
        
    Returns:
        Array of futility boundary Z-values (negative values)
    """
    if beta is None:
        beta = config.tsa_beta
    
    information_fractions = np.array(information_fractions)
    
    try:
        # Simple futility boundary based on conditional power
        # This is a basic implementation
        z_beta = stats.norm.ppf(beta)
        
        # Scale by information fraction
        futility_boundaries = z_beta / np.sqrt(information_fractions)
        
        return futility_boundaries
        
    except Exception as e:
        raise TSAError(f"Failed to calculate futility boundaries: {e}")


def boundary_summary(information_fractions: Union[List[float], np.ndarray],
                    alpha: float = None,
                    beta: float = None) -> dict:
    """Calculate summary of all boundary types.
    
    Args:
        information_fractions: Fraction of total information at each analysis
        alpha: Type I error rate
        beta: Type II error rate
        
    Returns:
        Dictionary with all boundary types
    """
    if alpha is None:
        alpha = config.tsa_alpha
    if beta is None:
        beta = config.tsa_beta
    
    try:
        boundaries = {
            'information_fractions': np.array(information_fractions),
            'obrien_fleming': obrien_fleming_boundary(information_fractions, alpha),
            'pocock': pocock_boundary(information_fractions, alpha),
            'futility': futility_boundary(information_fractions, beta),
            'alpha': alpha,
            'beta': beta,
            'n_analyses': len(information_fractions)
        }
        
        return boundaries
        
    except Exception as e:
        raise TSAError(f"Failed to calculate boundary summary: {e}")


def adjust_boundaries_for_heterogeneity(boundaries: np.ndarray,
                                       diversity_adjustment: float) -> np.ndarray:
    """Adjust monitoring boundaries for heterogeneity (diversity).
    
    Args:
        boundaries: Original boundary values
        diversity_adjustment: Diversity adjustment factor (1 + I²/100 for example)
        
    Returns:
        Adjusted boundary values
    """
    if diversity_adjustment <= 0:
        raise TSAError("Diversity adjustment must be positive")
    
    # Scale boundaries by square root of diversity adjustment
    adjusted_boundaries = boundaries * np.sqrt(diversity_adjustment)
    
    return adjusted_boundaries