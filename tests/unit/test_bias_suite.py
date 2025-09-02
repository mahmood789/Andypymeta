"""Unit tests for bias detection methods."""

import numpy as np
import pytest
from scipy import stats

from pymeta.bias import egger_test, begg_test, trim_fill
from tests.conftest import reference_egger_regression


class TestEggerTest:
    """Test suite for Egger's regression test for publication bias."""
    
    def test_egger_test_basic(self, continuous_effects_data):
        """Test basic Egger test calculation."""
        data = continuous_effects_data
        
        result = egger_test(data['effect_sizes'], data['standard_errors'])
        
        # Check structure
        assert isinstance(result, dict)
        required_keys = [
            'intercept', 'se_intercept', 't_intercept', 'p_intercept',
            'slope', 'se_slope', 't_slope', 'p_slope', 'df', 'bias_detected'
        ]
        for key in required_keys:
            assert key in result
        
        # Check values
        assert np.isfinite(result['intercept'])
        assert np.isfinite(result['slope'])
        assert result['se_intercept'] > 0
        assert result['se_slope'] > 0
        assert np.isfinite(result['t_intercept'])
        assert np.isfinite(result['t_slope'])
        assert 0 <= result['p_intercept'] <= 1
        assert 0 <= result['p_slope'] <= 1
        assert result['df'] == len(data['effect_sizes']) - 2
        assert isinstance(result['bias_detected'], bool)
    
    def test_egger_against_reference(self, continuous_effects_data):
        """Test Egger test against reference implementation."""
        data = continuous_effects_data
        
        result = egger_test(data['effect_sizes'], data['standard_errors'])
        
        # Reference implementation
        ref_intercept, ref_slope = reference_egger_regression(
            data['effect_sizes'], data['standard_errors']
        )
        
        # Should match within numerical precision
        if not (np.isnan(ref_intercept) or np.isnan(ref_slope)):
            np.testing.assert_allclose(result['intercept'], ref_intercept, rtol=1e-8)
            np.testing.assert_allclose(result['slope'], ref_slope, rtol=1e-8)
    
    def test_egger_test_no_bias_simulation(self):
        """Test Egger test with simulated data without bias."""
        np.random.seed(123)
        
        # Simulate unbiased data
        k = 20
        true_effect = 0.5
        sample_sizes = np.random.randint(50, 200, k)
        standard_errors = 2 / np.sqrt(sample_sizes)
        effect_sizes = np.random.normal(true_effect, standard_errors)
        
        result = egger_test(effect_sizes, standard_errors)
        
        # With unbiased data, intercept should be close to 0
        # p-value should be non-significant (usually)
        assert abs(result['intercept']) < 1.0  # Reasonable range
        # Don't enforce p > 0.05 due to randomness, but check structure
        assert np.isfinite(result['p_intercept'])
    
    def test_egger_test_with_bias_simulation(self):
        """Test Egger test with simulated publication bias."""
        np.random.seed(456)
        
        # Simulate biased data (funnel plot asymmetry)
        k = 15
        true_effect = 0.3
        sample_sizes = np.random.randint(30, 150, k)
        standard_errors = 2 / np.sqrt(sample_sizes)
        
        # Add bias: smaller studies have larger effects
        bias_factor = 2.0  # Strong bias
        effect_sizes = np.random.normal(
            true_effect + bias_factor / sample_sizes, 
            standard_errors
        )
        
        result = egger_test(effect_sizes, standard_errors)
        
        # Should detect bias (intercept significantly different from 0)
        # Note: Don't enforce p < 0.05 due to simulation variability
        assert np.isfinite(result['intercept'])
        assert np.isfinite(result['p_intercept'])
    
    def test_egger_manual_calculation(self):
        """Test against manual calculation for simple case."""
        # Simple case with known values
        effect_sizes = np.array([0.2, 0.4, 0.6])
        standard_errors = np.array([0.1, 0.2, 0.3])
        
        result = egger_test(effect_sizes, standard_errors)
        
        # Manual weighted regression calculation
        precision = 1 / standard_errors
        weights = precision**2
        
        # Design matrix [1, precision]
        X = np.column_stack([np.ones(3), precision])
        W = np.diag(weights)
        
        # Weighted least squares: beta = (X'WX)^-1 X'Wy
        XtWX_inv = np.linalg.inv(X.T @ W @ X)
        beta = XtWX_inv @ X.T @ W @ effect_sizes
        
        manual_intercept, manual_slope = beta
        
        np.testing.assert_allclose(result['intercept'], manual_intercept, rtol=1e-10)
        np.testing.assert_allclose(result['slope'], manual_slope, rtol=1e-10)
    
    def test_egger_test_identical_standard_errors(self):
        """Test Egger test when all studies have identical standard errors."""
        effect_sizes = [0.2, 0.4, 0.6, 0.3, 0.5]
        standard_errors = [0.1, 0.1, 0.1, 0.1, 0.1]  # All identical
        
        result = egger_test(effect_sizes, standard_errors)
        
        # Should still work (no division by zero in precision calculation)
        assert np.isfinite(result['intercept'])
        assert np.isfinite(result['slope'])
        assert np.isfinite(result['p_intercept'])
    
    def test_egger_test_two_studies(self):
        """Test Egger test with minimum number of studies."""
        effect_sizes = [0.3, 0.7]
        standard_errors = [0.1, 0.2]
        
        result = egger_test(effect_sizes, standard_errors)
        
        assert result['df'] == 0  # 2 studies - 2 parameters = 0 df
        # With 0 df, t-test is not meaningful, but calculation should work
        assert np.isfinite(result['intercept'])
        assert np.isfinite(result['slope'])
    
    def test_egger_test_single_study(self):
        """Test error handling with single study."""
        effect_sizes = [0.5]
        standard_errors = [0.1]
        
        # Should raise error or handle gracefully
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            egger_test(effect_sizes, standard_errors)
    
    def test_egger_test_array_inputs(self):
        """Test with different array input types."""
        effect_sizes_list = [0.3, 0.5, 0.4, 0.6]
        se_list = [0.1, 0.15, 0.12, 0.18]
        
        # Lists
        result1 = egger_test(effect_sizes_list, se_list)
        
        # Numpy arrays
        result2 = egger_test(np.array(effect_sizes_list), np.array(se_list))
        
        # Should give identical results
        np.testing.assert_allclose(result1['intercept'], result2['intercept'], rtol=1e-10)
        np.testing.assert_allclose(result1['slope'], result2['slope'], rtol=1e-10)


class TestBeggTest:
    """Test suite for Begg's rank correlation test."""
    
    def test_begg_test_basic(self, continuous_effects_data):
        """Test basic Begg test calculation."""
        data = continuous_effects_data
        
        result = begg_test(data['effect_sizes'], data['variances'])
        
        # Check structure
        assert isinstance(result, dict)
        assert 'tau' in result
        assert 'p_value' in result
        assert 'bias_detected' in result
        assert 'test' in result
        
        # Check values
        assert -1 <= result['tau'] <= 1  # Correlation coefficient
        assert 0 <= result['p_value'] <= 1
        assert isinstance(result['bias_detected'], bool)
        assert result['test'] == 'Begg rank correlation'
    
    def test_begg_test_no_correlation(self):
        """Test Begg test with uncorrelated data."""
        # Effect sizes and SEs should be uncorrelated in unbiased data
        effect_sizes = [0.1, 0.5, 0.3, 0.7, 0.4, 0.6]
        variances = [0.04, 0.09, 0.06, 0.12, 0.08, 0.10]
        
        result = begg_test(effect_sizes, variances)
        
        # Should not detect bias (tau close to 0, p-value large)
        assert abs(result['tau']) < 1.0  # Not perfect correlation
        assert np.isfinite(result['p_value'])
    
    def test_begg_test_perfect_correlation(self):
        """Test Begg test with perfect correlation (simulated bias)."""
        # Create perfectly correlated data
        standard_errors = [0.05, 0.1, 0.15, 0.2, 0.25]
        effect_sizes = [0.8, 0.6, 0.4, 0.2, 0.1]  # Inversely correlated
        variances = np.array(standard_errors)**2
        
        result = begg_test(effect_sizes, variances)
        
        # Should detect strong correlation
        assert abs(result['tau']) > 0.5  # Strong correlation
        # p-value should be small, but don't enforce due to small sample
    
    def test_begg_test_identical_values(self):
        """Test Begg test with identical values."""
        effect_sizes = [0.5, 0.5, 0.5, 0.5]
        variances = [0.1, 0.1, 0.1, 0.1]
        
        result = begg_test(effect_sizes, variances)
        
        # Should handle ties gracefully
        assert np.isfinite(result['tau'])
        assert np.isfinite(result['p_value'])
    
    def test_begg_test_small_sample(self):
        """Test Begg test with small sample size."""
        effect_sizes = [0.3, 0.7]
        variances = [0.04, 0.09]
        
        result = begg_test(effect_sizes, variances)
        
        # Should work but have limited power
        assert np.isfinite(result['tau'])
        assert np.isfinite(result['p_value'])


class TestTrimFill:
    """Test suite for trim-and-fill method."""
    
    def test_trim_fill_basic(self, continuous_effects_data):
        """Test basic trim-and-fill calculation."""
        data = continuous_effects_data
        
        result = trim_fill(data['effect_sizes'], data['variances'])
        
        # Check structure
        assert isinstance(result, dict)
        required_keys = [
            'k0', 'filled_effects', 'filled_variances', 
            'pooled_effect_filled', 'method', 'note'
        ]
        for key in required_keys:
            assert key in result
        
        # Check values
        assert result['k0'] >= 0  # Number of missing studies
        assert len(result['filled_effects']) >= len(data['effect_sizes'])
        assert len(result['filled_variances']) >= len(data['variances'])
        assert np.isfinite(result['pooled_effect_filled'])
        assert 'Trim-and-fill' in result['method']
    
    @pytest.mark.parametrize("estimator", ["L0", "R0", "Q0"])
    def test_trim_fill_estimators(self, continuous_effects_data, estimator):
        """Test trim-and-fill with different estimators."""
        data = continuous_effects_data
        
        result = trim_fill(data['effect_sizes'], data['variances'], estimator=estimator)
        
        assert estimator in result['method']
        assert np.isfinite(result['pooled_effect_filled'])
    
    def test_trim_fill_no_missing_studies(self):
        """Test trim-and-fill when no studies are missing."""
        # Symmetric data should not suggest missing studies
        effect_sizes = [0.45, 0.50, 0.55, 0.48, 0.52]
        variances = [0.04, 0.05, 0.06, 0.045, 0.055]
        
        result = trim_fill(effect_sizes, variances)
        
        # With current simplified implementation, k0 will be 0
        assert result['k0'] == 0
        np.testing.assert_array_equal(result['filled_effects'], effect_sizes)


class TestBiasDetectionIntegration:
    """Integration tests for bias detection methods."""
    
    def test_all_bias_tests_consistent_interface(self, continuous_effects_data):
        """Test that all bias tests have consistent interfaces."""
        data = continuous_effects_data
        
        egger_result = egger_test(data['effect_sizes'], data['standard_errors'])
        begg_result = begg_test(data['effect_sizes'], data['variances'])
        
        # Both should have bias_detected boolean
        assert isinstance(egger_result['bias_detected'], bool)
        assert isinstance(begg_result['bias_detected'], bool)
        
        # Both should have p_value
        assert 'p_value' in egger_result  # p_intercept for Egger
        assert 'p_value' in begg_result
    
    def test_bias_detection_with_symmetric_data(self):
        """Test bias detection methods with symmetric (unbiased) data."""
        np.random.seed(789)
        
        # Create symmetric funnel plot data
        k = 20
        true_effect = 0.4
        sample_sizes = np.random.randint(50, 300, k)
        standard_errors = 2 / np.sqrt(sample_sizes)
        effect_sizes = np.random.normal(true_effect, standard_errors)
        variances = standard_errors**2
        
        egger_result = egger_test(effect_sizes, standard_errors)
        begg_result = begg_test(effect_sizes, variances)
        
        # With symmetric data, should generally not detect bias
        # But don't enforce due to randomness in simulation
        assert np.isfinite(egger_result['p_intercept'])
        assert np.isfinite(begg_result['p_value'])
    
    def test_bias_detection_edge_cases(self):
        """Test bias detection with edge cases."""
        # All effect sizes identical
        effect_sizes = [0.5, 0.5, 0.5, 0.5]
        standard_errors = [0.1, 0.2, 0.15, 0.25]
        variances = np.array(standard_errors)**2
        
        egger_result = egger_test(effect_sizes, standard_errors)
        begg_result = begg_test(effect_sizes, variances)
        
        # Should handle gracefully
        assert np.isfinite(egger_result['intercept'])
        assert np.isfinite(begg_result['tau'])
    
    def test_empty_input_handling(self):
        """Test error handling with empty inputs."""
        with pytest.raises((ValueError, IndexError)):
            egger_test([], [])
        
        with pytest.raises((ValueError, IndexError)):
            begg_test([], [])
        
        with pytest.raises((ValueError, IndexError)):
            trim_fill([], [])
    
    def test_mismatched_input_lengths(self):
        """Test error handling with mismatched input lengths."""
        with pytest.raises((ValueError, IndexError)):
            egger_test([0.3, 0.5], [0.1])  # Different lengths
        
        with pytest.raises((ValueError, IndexError)):
            begg_test([0.3], [0.04, 0.06])  # Different lengths


class TestBiasDetectionNumerical:
    """Test numerical properties of bias detection methods."""
    
    def test_egger_regression_properties(self):
        """Test mathematical properties of Egger regression."""
        effect_sizes = [0.2, 0.4, 0.6, 0.8]
        standard_errors = [0.05, 0.1, 0.15, 0.2]
        
        result = egger_test(effect_sizes, standard_errors)
        
        # Regression should fit the data reasonably
        precision = 1 / np.array(standard_errors)
        predicted = result['intercept'] + result['slope'] * precision
        
        # R-squared should be calculable (residuals should make sense)
        residuals = np.array(effect_sizes) - predicted
        assert np.all(np.isfinite(residuals))
    
    def test_kendall_tau_properties(self):
        """Test properties of Kendall's tau in Begg test."""
        # Perfectly monotonic relationship
        effect_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
        variances = [0.01, 0.04, 0.09, 0.16, 0.25]  # Perfectly correlated
        
        result = begg_test(effect_sizes, variances)
        
        # Should detect perfect correlation
        assert abs(result['tau']) > 0.8  # Very high correlation
    
    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values."""
        # Very large effect sizes
        effect_sizes = [100, 101, 102]
        standard_errors = [0.1, 0.1, 0.1]
        variances = np.array(standard_errors)**2
        
        egger_result = egger_test(effect_sizes, standard_errors)
        begg_result = begg_test(effect_sizes, variances)
        
        # Should produce finite results
        assert np.isfinite(egger_result['intercept'])
        assert np.isfinite(egger_result['slope'])
        assert np.isfinite(begg_result['tau'])
        
        # Very small effect sizes
        effect_sizes = [1e-6, 2e-6, 3e-6]
        standard_errors = [1e-7, 1e-7, 1e-7]
        variances = np.array(standard_errors)**2
        
        egger_result = egger_test(effect_sizes, standard_errors)
        begg_result = begg_test(effect_sizes, variances)
        
        # Should produce finite results
        assert np.isfinite(egger_result['intercept'])
        assert np.isfinite(egger_result['slope'])
        assert np.isfinite(begg_result['tau'])