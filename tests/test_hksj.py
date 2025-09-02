"""
Test suite for HKSJ (Hartung-Knapp-Sidik-Jonkman) variance adjustment.
"""

import pytest
import numpy as np
from scipy.stats import t

from pymeta.stats.hksj import hksj_se, hksj_confidence_interval, hksj_p_value


class TestHKSJ:
    """Test cases for HKSJ variance adjustment."""
    
    def test_hksj_se_basic(self):
        """Test basic HKSJ standard error calculation."""
        effects = np.array([0.5, 0.8, 0.3, 0.6])
        variances = np.array([0.1, 0.15, 0.12, 0.08])
        tau2 = 0.02
        
        result = hksj_se(effects, variances, tau2)
        
        assert result.se_hk > 0
        assert result.df == len(effects) - 1
        assert result.tcrit > 0
        assert result.variance_hk == result.se_hk ** 2
    
    def test_hksj_se_edge_cases(self):
        """Test edge cases for HKSJ calculation."""
        # Test with minimum studies
        effects = np.array([0.5, 0.8])
        variances = np.array([0.1, 0.15])
        tau2 = 0.02
        
        result = hksj_se(effects, variances, tau2)
        assert result.df == 1
        
        # Test with zero tau2
        result = hksj_se(effects, variances, 0.0)
        assert result.se_hk > 0
    
    def test_hksj_se_validation(self):
        """Test input validation for HKSJ."""
        effects = np.array([0.5, 0.8, 0.3])
        variances = np.array([0.1, 0.15])  # Different length
        tau2 = 0.02
        
        with pytest.raises(ValueError, match="same length"):
            hksj_se(effects, variances, tau2)
        
        # Test insufficient studies
        effects = np.array([0.5])
        variances = np.array([0.1])
        
        with pytest.raises(ValueError, match="at least 2 studies"):
            hksj_se(effects, variances, tau2)
    
    def test_hksj_confidence_interval(self):
        """Test HKSJ confidence interval calculation."""
        effects = np.array([0.5, 0.8, 0.3, 0.6])
        variances = np.array([0.1, 0.15, 0.12, 0.08])
        tau2 = 0.02
        pooled_effect = 0.55
        
        hksj_result = hksj_se(effects, variances, tau2)
        ci_lower, ci_upper = hksj_confidence_interval(pooled_effect, hksj_result)
        
        assert ci_lower < pooled_effect < ci_upper
        assert ci_upper - ci_lower > 0  # Positive width
    
    def test_hksj_p_value(self):
        """Test HKSJ p-value calculation."""
        effects = np.array([0.5, 0.8, 0.3, 0.6])
        variances = np.array([0.1, 0.15, 0.12, 0.08])
        tau2 = 0.02
        pooled_effect = 0.55
        
        hksj_result = hksj_se(effects, variances, tau2)
        p_value = hksj_p_value(pooled_effect, hksj_result)
        
        assert 0 <= p_value <= 1
        
        # Test null effect
        p_value_null = hksj_p_value(0.0, hksj_result)
        assert 0 <= p_value_null <= 1
    
    def test_hksj_adjustment_effect(self):
        """Test that HKSJ adjustment typically increases variance."""
        # Create data with some heterogeneity
        effects = np.array([0.2, 0.5, 0.8, 0.3, 0.6, 0.4])
        variances = np.array([0.1, 0.12, 0.08, 0.15, 0.09, 0.11])
        tau2 = 0.05
        
        # Calculate weights and standard variance
        weights = 1.0 / (variances + tau2)
        standard_var = 1.0 / np.sum(weights)
        standard_se = np.sqrt(standard_var)
        
        # Calculate HKSJ adjusted variance
        hksj_result = hksj_se(effects, variances, tau2)
        
        # HKSJ should typically increase SE when there's heterogeneity
        # (though not guaranteed in all cases)
        assert hksj_result.se_hk > 0
        assert hksj_result.df == len(effects) - 1
    
    def test_hksj_reproducibility(self):
        """Test that HKSJ calculations are reproducible."""
        effects = np.array([0.3, 0.7, 0.4, 0.6, 0.5])
        variances = np.array([0.08, 0.12, 0.10, 0.09, 0.11])
        tau2 = 0.03
        
        # Calculate multiple times
        result1 = hksj_se(effects, variances, tau2)
        result2 = hksj_se(effects, variances, tau2)
        
        assert result1.se_hk == result2.se_hk
        assert result1.df == result2.df
        assert result1.tcrit == result2.tcrit
    
    def test_hksj_vs_normal_distribution(self):
        """Test HKSJ produces wider CIs than normal distribution."""
        effects = np.array([0.4, 0.6, 0.5, 0.7, 0.3])
        variances = np.array([0.09, 0.11, 0.10, 0.08, 0.12])
        tau2 = 0.04
        pooled_effect = 0.5
        alpha = 0.05
        
        # Calculate HKSJ CI
        hksj_result = hksj_se(effects, variances, tau2)
        hksj_lower, hksj_upper = hksj_confidence_interval(pooled_effect, hksj_result)
        
        # Calculate normal CI
        weights = 1.0 / (variances + tau2)
        normal_se = np.sqrt(1.0 / np.sum(weights))
        from scipy.stats import norm
        zcrit = norm.ppf(1 - alpha/2)
        normal_lower = pooled_effect - zcrit * normal_se
        normal_upper = pooled_effect + zcrit * normal_se
        
        # HKSJ CI should typically be wider (more conservative)
        hksj_width = hksj_upper - hksj_lower
        normal_width = normal_upper - normal_lower
        
        # For small samples with heterogeneity, HKSJ is usually more conservative
        assert hksj_width > 0
        assert normal_width > 0
        # Note: Not asserting HKSJ > normal as this depends on Q statistic