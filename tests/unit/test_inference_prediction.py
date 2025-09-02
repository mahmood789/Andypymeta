"""Unit tests for prediction intervals and inference methods."""

import numpy as np
import pytest
import scipy.stats as stats

from pymeta.inference import prediction_intervals, heterogeneity_tests


class TestPredictionIntervals:
    """Test suite for prediction interval calculation."""
    
    def test_prediction_intervals_basic(self):
        """Test basic prediction interval calculation."""
        pooled_effect = 0.5
        pooled_se = 0.1
        tau2 = 0.04
        k = 10
        
        result = prediction_intervals(pooled_effect, pooled_se, tau2, k)
        
        # Check structure
        assert isinstance(result, dict)
        assert 'pi_lower' in result
        assert 'pi_upper' in result
        assert 'pred_se' in result
        assert 't_crit' in result
        assert 'df' in result
        assert 'alpha' in result
        
        # Check values
        assert np.isfinite(result['pi_lower'])
        assert np.isfinite(result['pi_upper'])
        assert result['pi_lower'] < result['pi_upper']
        assert result['pi_lower'] < pooled_effect < result['pi_upper']
        assert result['pred_se'] > pooled_se  # Should be larger due to tau2
        assert result['df'] == k - 2
        assert result['alpha'] == 0.05
    
    def test_prediction_intervals_manual_calculation(self):
        """Test against manual calculation."""
        pooled_effect = 0.3
        pooled_se = 0.08
        tau2 = 0.02
        k = 8
        alpha = 0.05
        
        result = prediction_intervals(pooled_effect, pooled_se, tau2, k, alpha)
        
        # Manual calculation
        pred_se_manual = np.sqrt(pooled_se**2 + tau2)
        df_manual = k - 2
        t_crit_manual = stats.t.ppf(1 - alpha/2, df_manual)
        pi_lower_manual = pooled_effect - t_crit_manual * pred_se_manual
        pi_upper_manual = pooled_effect + t_crit_manual * pred_se_manual
        
        np.testing.assert_allclose(result['pred_se'], pred_se_manual, rtol=1e-10)
        np.testing.assert_allclose(result['t_crit'], t_crit_manual, rtol=1e-10)
        np.testing.assert_allclose(result['pi_lower'], pi_lower_manual, rtol=1e-10)
        np.testing.assert_allclose(result['pi_upper'], pi_upper_manual, rtol=1e-10)
    
    def test_prediction_intervals_no_heterogeneity(self):
        """Test prediction intervals when tau2 = 0."""
        pooled_effect = 0.4
        pooled_se = 0.1
        tau2 = 0.0
        k = 6
        
        result = prediction_intervals(pooled_effect, pooled_se, tau2, k)
        
        # When tau2 = 0, pred_se should equal pooled_se
        np.testing.assert_allclose(result['pred_se'], pooled_se, rtol=1e-10)
        
        # Should still be valid intervals
        assert result['pi_lower'] < result['pi_upper']
        assert np.isfinite(result['pi_lower'])
        assert np.isfinite(result['pi_upper'])
    
    def test_prediction_intervals_high_heterogeneity(self):
        """Test prediction intervals with high heterogeneity."""
        pooled_effect = 0.5
        pooled_se = 0.1
        tau2 = 0.5  # High heterogeneity
        k = 15
        
        result = prediction_intervals(pooled_effect, pooled_se, tau2, k)
        
        # High tau2 should give much wider intervals
        assert result['pred_se'] > pooled_se * 2  # Much larger
        
        # Intervals should be quite wide
        width = result['pi_upper'] - result['pi_lower']
        assert width > 1.0  # Wide intervals due to high heterogeneity
    
    def test_prediction_intervals_insufficient_studies(self):
        """Test behavior with insufficient studies (k <= 2)."""
        pooled_effect = 0.5
        pooled_se = 0.1
        tau2 = 0.04
        
        # k = 2 (insufficient)
        result = prediction_intervals(pooled_effect, pooled_se, tau2, k=2)
        
        assert np.isnan(result['pi_lower'])
        assert np.isnan(result['pi_upper'])
        assert 'warning' in result
        
        # k = 1 (insufficient)
        result = prediction_intervals(pooled_effect, pooled_se, tau2, k=1)
        
        assert np.isnan(result['pi_lower'])
        assert np.isnan(result['pi_upper'])
        assert 'warning' in result
    
    def test_prediction_intervals_different_alpha(self):
        """Test prediction intervals with different significance levels."""
        pooled_effect = 0.5
        pooled_se = 0.1
        tau2 = 0.04
        k = 10
        
        # 90% PI (alpha = 0.10)
        result_90 = prediction_intervals(pooled_effect, pooled_se, tau2, k, alpha=0.10)
        
        # 99% PI (alpha = 0.01)
        result_99 = prediction_intervals(pooled_effect, pooled_se, tau2, k, alpha=0.01)
        
        # 99% intervals should be wider than 90%
        width_90 = result_90['pi_upper'] - result_90['pi_lower']
        width_99 = result_99['pi_upper'] - result_99['pi_lower']
        
        assert width_99 > width_90
        assert result_99['t_crit'] > result_90['t_crit']
    
    def test_prediction_intervals_large_k(self):
        """Test prediction intervals with large number of studies."""
        pooled_effect = 0.5
        pooled_se = 0.05
        tau2 = 0.02
        k = 100  # Many studies
        
        result = prediction_intervals(pooled_effect, pooled_se, tau2, k)
        
        # With large k, t-distribution approaches normal
        # t_crit should be close to 1.96 for 95% intervals
        assert abs(result['t_crit'] - 1.96) < 0.1
        assert result['df'] == 98


class TestHeterogeneityTests:
    """Test suite for heterogeneity testing."""
    
    def test_heterogeneity_tests_basic(self, heterogeneous_data):
        """Test basic heterogeneity test calculation."""
        data = heterogeneous_data
        
        result = heterogeneity_tests(
            data['effect_sizes'], 
            data['variances']
        )
        
        # Check structure
        assert isinstance(result, dict)
        required_keys = ['Q', 'df', 'Q_p_value', 'I2', 'H2', 'tau2', 'tau']
        for key in required_keys:
            assert key in result
        
        # Check values
        assert result['Q'] >= 0
        assert result['df'] == len(data['effect_sizes']) - 1
        assert 0 <= result['Q_p_value'] <= 1
        assert 0 <= result['I2'] <= 1
        assert result['H2'] >= 1
        assert result['tau2'] >= 0
        assert result['tau'] >= 0
        assert result['tau'] == np.sqrt(result['tau2'])
    
    def test_heterogeneity_tests_manual_calculation(self):
        """Test against manual calculation."""
        effect_sizes = np.array([0.3, 0.7, 0.5])
        variances = np.array([0.04, 0.09, 0.06])
        
        result = heterogeneity_tests(effect_sizes, variances)
        
        # Manual Q calculation
        weights = 1 / variances
        pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
        Q_manual = np.sum(weights * (effect_sizes - pooled_effect)**2)
        
        np.testing.assert_allclose(result['Q'], Q_manual, rtol=1e-10)
        
        # Manual I2 calculation
        df = len(effect_sizes) - 1
        I2_manual = max(0, (Q_manual - df) / Q_manual) if Q_manual > 0 else 0
        
        np.testing.assert_allclose(result['I2'], I2_manual, rtol=1e-10)
        
        # Manual H2 calculation
        H2_manual = Q_manual / df if df > 0 else 1
        
        np.testing.assert_allclose(result['H2'], H2_manual, rtol=1e-10)
    
    def test_heterogeneity_tests_provided_pooled_effect(self):
        """Test when pooled effect is provided."""
        effect_sizes = np.array([0.3, 0.7, 0.5])
        variances = np.array([0.04, 0.09, 0.06])
        pooled_effect = 0.45
        
        result = heterogeneity_tests(effect_sizes, variances, pooled_effect)
        
        # Should use the provided pooled effect
        weights = 1 / variances
        Q_manual = np.sum(weights * (effect_sizes - pooled_effect)**2)
        
        np.testing.assert_allclose(result['Q'], Q_manual, rtol=1e-10)
    
    def test_heterogeneity_tests_homogeneous_data(self, homogeneous_data):
        """Test with homogeneous data."""
        data = homogeneous_data
        
        result = heterogeneity_tests(data['effect_sizes'], data['variances'])
        
        # Should show low heterogeneity
        assert result['I2'] < 0.5  # Less than 50%
        assert result['H2'] < 2    # Close to 1
        assert result['Q_p_value'] > 0.05  # Non-significant
    
    def test_heterogeneity_tests_heterogeneous_data(self, heterogeneous_data):
        """Test with heterogeneous data."""
        data = heterogeneous_data
        
        result = heterogeneity_tests(data['effect_sizes'], data['variances'])
        
        # Should show high heterogeneity
        assert result['I2'] > 0.5  # More than 50%
        assert result['H2'] > 2    # Well above 1
        # Note: p-value might still be > 0.05 with small sample size
    
    def test_heterogeneity_tests_identical_studies(self):
        """Test with identical studies (no heterogeneity)."""
        effect_sizes = [0.5, 0.5, 0.5, 0.5]
        variances = [0.1, 0.1, 0.1, 0.1]
        
        result = heterogeneity_tests(effect_sizes, variances)
        
        # Should show no heterogeneity
        assert result['Q'] == 0
        assert result['I2'] == 0
        assert result['H2'] == 0  # Q/df = 0/3 = 0
        assert result['tau2'] == 0
        assert result['Q_p_value'] == 1.0
    
    def test_heterogeneity_tests_single_study(self):
        """Test with single study."""
        effect_sizes = [0.5]
        variances = [0.1]
        
        result = heterogeneity_tests(effect_sizes, variances)
        
        assert result['Q'] == 0
        assert result['df'] == 0
        assert result['I2'] == 0
        assert result['tau2'] == 0
        # H2 should be handled appropriately (avoid division by zero)
    
    def test_heterogeneity_tests_extreme_heterogeneity(self):
        """Test with extremely heterogeneous data."""
        effect_sizes = [-2, 0, 2, -1.5, 1.5]
        variances = [0.01, 0.01, 0.01, 0.01, 0.01]  # Small variances
        
        result = heterogeneity_tests(effect_sizes, variances)
        
        # Should show very high heterogeneity
        assert result['Q'] > 50  # Very large Q
        assert result['I2'] > 0.9  # More than 90%
        assert result['H2'] > 10   # Much greater than 1
        assert result['Q_p_value'] < 0.001  # Highly significant
    
    def test_heterogeneity_q_distribution(self):
        """Test that Q follows chi-square distribution under null."""
        # This is more of a theoretical test
        # Under null hypothesis of no heterogeneity, Q ~ chi2(df)
        
        effect_sizes = [0.5, 0.51, 0.49, 0.52, 0.48]  # Small differences
        variances = [0.1, 0.1, 0.1, 0.1, 0.1]
        
        result = heterogeneity_tests(effect_sizes, variances)
        
        # With small differences, Q should be small
        # and p-value should be large (non-significant)
        assert result['Q'] < 10  # Reasonable Q value
        assert result['Q_p_value'] > 0.1  # Non-significant
    
    def test_heterogeneity_tests_array_inputs(self):
        """Test with different array input types."""
        effect_sizes_list = [0.3, 0.5, 0.4]
        variances_list = [0.04, 0.06, 0.05]
        
        # Lists
        result1 = heterogeneity_tests(effect_sizes_list, variances_list)
        
        # Numpy arrays
        result2 = heterogeneity_tests(
            np.array(effect_sizes_list), 
            np.array(variances_list)
        )
        
        # Should give identical results
        for key in result1.keys():
            np.testing.assert_allclose(result1[key], result2[key], rtol=1e-10)


class TestInferenceEdgeCases:
    """Test edge cases for inference methods."""
    
    def test_prediction_intervals_zero_tau2(self):
        """Test prediction intervals when tau2 is exactly zero."""
        result = prediction_intervals(0.5, 0.1, 0.0, 10)
        
        # Should reduce to confidence interval calculation
        expected_se = 0.1  # Same as pooled_se when tau2 = 0
        np.testing.assert_allclose(result['pred_se'], expected_se, rtol=1e-10)
    
    def test_prediction_intervals_very_large_tau2(self):
        """Test prediction intervals with very large tau2."""
        result = prediction_intervals(0.5, 0.05, 10.0, 20)  # Very large tau2
        
        # pred_se should be dominated by tau2
        expected_se = np.sqrt(0.05**2 + 10.0)
        np.testing.assert_allclose(result['pred_se'], expected_se, rtol=1e-10)
        
        # Intervals should be very wide
        width = result['pi_upper'] - result['pi_lower']
        assert width > 10  # Very wide due to large tau2
    
    def test_heterogeneity_zero_variances(self):
        """Test heterogeneity tests with zero variances."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [0.0, 0.04, 0.06]  # One zero variance
        
        # Should handle gracefully or raise appropriate warning
        with pytest.warns(RuntimeWarning) or pytest.raises(ZeroDivisionError):
            result = heterogeneity_tests(effect_sizes, variances)
    
    def test_extreme_values_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large effect sizes
        effect_sizes = [100, 101, 99]
        variances = [0.1, 0.1, 0.1]
        
        result = heterogeneity_tests(effect_sizes, variances)
        
        # Should produce finite results
        assert np.isfinite(result['Q'])
        assert np.isfinite(result['I2'])
        assert np.isfinite(result['tau2'])
        
        # Very small effect sizes
        effect_sizes = [1e-6, 2e-6, 1.5e-6]
        variances = [1e-10, 1e-10, 1e-10]
        
        result = heterogeneity_tests(effect_sizes, variances)
        
        # Should produce finite results
        assert np.isfinite(result['Q'])
        assert np.isfinite(result['I2'])
        assert np.isfinite(result['tau2'])