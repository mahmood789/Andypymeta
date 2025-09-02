"""Egger regression test for publication bias."""

import numpy as np
from typing import List
from scipy import stats

from ..typing import MetaPoint, BiasTestResult
from ..registries import register_bias_test
from ..errors import BiasTestError


@register_bias_test("egger")
def egger_regression_test(points: List[MetaPoint]) -> BiasTestResult:
    """Perform Egger's regression test for publication bias.
    
    Tests for small-study effects by regressing standardized effect sizes
    against their precision (1/standard error).
    
    Args:
        points: List of MetaPoint objects
        
    Returns:
        BiasTestResult object
        
    Raises:
        BiasTestError: If test cannot be performed
    
    References:
        Egger, M., et al. (1997). Bias in meta-analysis detected by a simple,
        graphical test. BMJ, 315(7109), 629-634.
    """
    if len(points) < 3:
        raise BiasTestError("At least 3 studies required for Egger test")
    
    try:
        # Extract data
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        standard_errors = np.sqrt(variances)
        
        # Calculate precision (1/SE)
        precision = 1.0 / standard_errors
        
        # Standardized effect sizes (effect / SE)
        standardized_effects = effects / standard_errors
        
        # Linear regression: standardized_effect = a + b * precision
        # The intercept 'a' represents bias
        
        # Design matrix
        X = np.column_stack([np.ones(len(points)), precision])
        y = standardized_effects
        
        # Weighted least squares (weights = precision²)
        weights = precision ** 2
        W = np.diag(weights)
        
        # Calculate coefficients: β = (X'WX)⁻¹X'Wy
        XTW = X.T @ W
        XTWX = XTW @ X
        XTWy = XTW @ y
        
        # Check for singularity
        if np.linalg.cond(XTWX) > 1e12:
            raise BiasTestError("Singular matrix in Egger regression")
        
        beta = np.linalg.solve(XTWX, XTWy)
        intercept, slope = beta
        
        # Calculate standard errors
        # Residuals
        y_pred = X @ beta
        residuals = y - y_pred
        
        # Weighted residual sum of squares
        rss = residuals.T @ W @ residuals
        df = len(points) - 2
        
        # Covariance matrix
        if df > 0:
            cov_matrix = np.linalg.inv(XTWX) * (rss / df)
            se_intercept = np.sqrt(cov_matrix[0, 0])
            se_slope = np.sqrt(cov_matrix[1, 1])
        else:
            raise BiasTestError("Insufficient degrees of freedom for Egger test")
        
        # Test statistic for intercept (bias)
        t_statistic = intercept / se_intercept if se_intercept > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
        
        # Interpretation
        interpretation = _interpret_egger_test(p_value, intercept)
        
        # Additional details
        details = {
            'intercept': intercept,
            'intercept_se': se_intercept,
            'slope': slope,
            'slope_se': se_slope,
            't_statistic': t_statistic,
            'degrees_of_freedom': df,
            'r_squared': _calculate_r_squared(standardized_effects, y_pred, weights),
            'n_studies': len(points)
        }
        
        return BiasTestResult(
            test_name="Egger's regression test",
            statistic=t_statistic,
            p_value=p_value,
            interpretation=interpretation,
            details=details
        )
        
    except Exception as e:
        raise BiasTestError(f"Egger test failed: {e}")


def _interpret_egger_test(p_value: float, intercept: float, alpha: float = 0.10) -> str:
    """Interpret Egger test results.
    
    Args:
        p_value: Test p-value
        intercept: Regression intercept
        alpha: Significance level (default 0.10 for bias tests)
        
    Returns:
        Interpretation string
    """
    significant = p_value < alpha
    
    if significant:
        direction = "positive" if intercept > 0 else "negative"
        return (f"Evidence of publication bias detected (p={p_value:.4f} < {alpha}). "
                f"The {direction} intercept suggests small-study effects.")
    else:
        return (f"No significant evidence of publication bias (p={p_value:.4f} ≥ {alpha}). "
                "However, this does not rule out bias.")


def _calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Calculate weighted R² for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        weights: Regression weights
        
    Returns:
        Weighted R² value
    """
    # Weighted mean
    weighted_mean = np.sum(weights * y_true) / np.sum(weights)
    
    # Total sum of squares (weighted)
    tss = np.sum(weights * (y_true - weighted_mean) ** 2)
    
    # Residual sum of squares (weighted)
    rss = np.sum(weights * (y_true - y_pred) ** 2)
    
    # R²
    if tss > 0:
        r_squared = 1 - (rss / tss)
    else:
        r_squared = 0.0
    
    return max(0, r_squared)  # Ensure non-negative


@register_bias_test("begg")
def begg_mazumdar_test(points: List[MetaPoint]) -> BiasTestResult:
    """Perform Begg and Mazumdar rank correlation test.
    
    Tests for correlation between effect sizes and their variances
    using Kendall's tau rank correlation.
    
    Args:
        points: List of MetaPoint objects
        
    Returns:
        BiasTestResult object
    """
    if len(points) < 3:
        raise BiasTestError("At least 3 studies required for Begg test")
    
    try:
        # Extract data
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        
        # Calculate Kendall's tau correlation
        tau_statistic, p_value = stats.kendalltau(effects, variances)
        
        # Interpretation
        if p_value < 0.10:
            interpretation = (f"Evidence of publication bias detected using rank correlation "
                            f"(τ={tau_statistic:.4f}, p={p_value:.4f} < 0.10)")
        else:
            interpretation = (f"No significant evidence of publication bias using rank correlation "
                            f"(τ={tau_statistic:.4f}, p={p_value:.4f} ≥ 0.10)")
        
        details = {
            'kendall_tau': tau_statistic,
            'n_studies': len(points),
            'test_type': 'rank_correlation'
        }
        
        return BiasTestResult(
            test_name="Begg and Mazumdar test",
            statistic=tau_statistic,
            p_value=p_value,
            interpretation=interpretation,
            details=details
        )
        
    except Exception as e:
        raise BiasTestError(f"Begg test failed: {e}")


def funnel_plot_asymmetry_test(points: List[MetaPoint]) -> dict:
    """Test for funnel plot asymmetry using multiple methods.
    
    Args:
        points: List of MetaPoint objects
        
    Returns:
        Dictionary with results from multiple asymmetry tests
    """
    results = {}
    
    # Egger test
    try:
        results['egger'] = egger_regression_test(points)
    except Exception as e:
        results['egger'] = {'error': str(e)}
    
    # Begg test
    try:
        results['begg'] = begg_mazumdar_test(points)
    except Exception as e:
        results['begg'] = {'error': str(e)}
    
    # Combined interpretation
    egger_significant = (isinstance(results.get('egger'), BiasTestResult) and 
                        results['egger'].p_value < 0.10)
    begg_significant = (isinstance(results.get('begg'), BiasTestResult) and 
                       results['begg'].p_value < 0.10)
    
    if egger_significant and begg_significant:
        combined_interpretation = "Strong evidence of publication bias (both tests significant)"
    elif egger_significant or begg_significant:
        combined_interpretation = "Moderate evidence of publication bias (one test significant)"
    else:
        combined_interpretation = "No strong evidence of publication bias"
    
    results['combined_interpretation'] = combined_interpretation
    
    return results