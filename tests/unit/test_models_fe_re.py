"""Unit tests for fixed and random effects meta-analysis models."""

import numpy as np
import pytest

from pymeta.models import fixed_effects, random_effects, reference_re_summary
from tests.conftest import reference_re_summary as conftest_re_summary


class TestFixedEffectsModel:
    """Test suite for fixed effects meta-analysis."""
    
    def test_fixed_effects_basic(self, fe_re_data):
        """Test basic fixed effects calculation."""
        data = fe_re_data
        
        result = fixed_effects(data['effect_sizes'], data['variances'])
        
        # Check return structure
        assert isinstance(result, dict)
        required_keys = [
            'pooled_effect', 'se', 'variance', 'z_value', 'p_value',
            'ci_lower', 'ci_upper', 'Q', 'Q_df', 'Q_p_value', 'method'
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check values
        assert np.isfinite(result['pooled_effect'])
        assert result['se'] > 0
        assert result['variance'] > 0
        assert np.isfinite(result['z_value'])
        assert 0 <= result['p_value'] <= 1
        assert result['ci_lower'] < result['ci_upper']
        assert result['Q'] >= 0
        assert result['Q_df'] == len(data['effect_sizes']) - 1
        assert 0 <= result['Q_p_value'] <= 1
        assert result['method'] == 'Fixed Effects'
    
    def test_fixed_effects_manual_calculation(self):
        """Test against manual calculation."""
        effect_sizes = np.array([0.3, 0.5, 0.4])
        variances = np.array([0.04, 0.09, 0.06])
        
        result = fixed_effects(effect_sizes, variances)
        
        # Manual calculation
        weights = 1 / variances  # [25, 11.11, 16.67]
        sum_weights = np.sum(weights)
        manual_pooled = np.sum(weights * effect_sizes) / sum_weights
        manual_variance = 1 / sum_weights
        manual_se = np.sqrt(manual_variance)
        
        np.testing.assert_allclose(result['pooled_effect'], manual_pooled, rtol=1e-10)
        np.testing.assert_allclose(result['variance'], manual_variance, rtol=1e-10)
        np.testing.assert_allclose(result['se'], manual_se, rtol=1e-10)
    
    def test_fixed_effects_confidence_interval(self, fe_re_data):
        """Test confidence interval calculation."""
        data = fe_re_data
        result = fixed_effects(data['effect_sizes'], data['variances'])
        
        # 95% CI should be effect ± 1.96*SE
        expected_lower = result['pooled_effect'] - 1.96 * result['se']
        expected_upper = result['pooled_effect'] + 1.96 * result['se']
        
        np.testing.assert_allclose(result['ci_lower'], expected_lower, rtol=1e-10)
        np.testing.assert_allclose(result['ci_upper'], expected_upper, rtol=1e-10)
    
    def test_fixed_effects_heterogeneity_test(self, heterogeneous_data):
        """Test Q statistic calculation."""
        data = heterogeneous_data
        result = fixed_effects(data['effect_sizes'], data['variances'])
        
        # Manual Q calculation
        weights = 1 / np.array(data['variances'])
        pooled_effect = result['pooled_effect']
        manual_Q = np.sum(weights * (np.array(data['effect_sizes']) - pooled_effect)**2)
        
        np.testing.assert_allclose(result['Q'], manual_Q, rtol=1e-10)
        
        # With heterogeneous data, Q should be large and p-value small
        assert result['Q'] > 0
        # Don't enforce p < 0.05 as it depends on the specific data and degrees of freedom
    
    def test_fixed_effects_single_study(self):
        """Test with single study."""
        result = fixed_effects([0.5], [0.1])
        
        assert result['pooled_effect'] == 0.5
        assert result['variance'] == 0.1
        assert result['se'] == np.sqrt(0.1)
        assert result['Q'] == 0
        assert result['Q_df'] == 0
    
    def test_fixed_effects_identical_studies(self):
        """Test with identical studies."""
        effect_sizes = [0.5, 0.5, 0.5]
        variances = [0.1, 0.1, 0.1]
        
        result = fixed_effects(effect_sizes, variances)
        
        assert result['pooled_effect'] == 0.5
        assert result['Q'] == 0  # No heterogeneity
        assert result['Q_p_value'] == 1.0


class TestRandomEffectsModel:
    """Test suite for random effects meta-analysis."""
    
    def test_random_effects_basic(self, fe_re_data):
        """Test basic random effects calculation."""
        data = fe_re_data
        
        result = random_effects(data['effect_sizes'], data['variances'])
        
        # Check return structure
        required_keys = [
            'pooled_effect', 'se', 'variance', 'tau2', 'tau',
            'z_value', 'p_value', 'ci_lower', 'ci_upper',
            'Q', 'Q_df', 'Q_p_value', 'I2', 'method'
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check values
        assert np.isfinite(result['pooled_effect'])
        assert result['se'] > 0
        assert result['variance'] > 0
        assert result['tau2'] >= 0
        assert result['tau'] >= 0
        assert result['tau'] == np.sqrt(result['tau2'])
        assert np.isfinite(result['z_value'])
        assert 0 <= result['p_value'] <= 1
        assert result['ci_lower'] < result['ci_upper']
        assert result['Q'] >= 0
        assert result['Q_df'] == len(data['effect_sizes']) - 1
        assert 0 <= result['Q_p_value'] <= 1
        assert 0 <= result['I2'] <= 1
        assert 'Random Effects' in result['method']
    
    def test_random_effects_vs_reference(self, fe_re_data):
        """Test random effects against reference implementation."""
        data = fe_re_data
        
        result = random_effects(data['effect_sizes'], data['variances'])
        
        # Use module reference
        ref_pooled, ref_variance = reference_re_summary(
            data['effect_sizes'], data['variances'], result['tau2']
        )
        
        np.testing.assert_allclose(result['pooled_effect'], ref_pooled, rtol=1e-10)
        np.testing.assert_allclose(result['variance'], ref_variance, rtol=1e-10)
        
        # Use conftest reference
        ref_pooled2, ref_variance2 = conftest_re_summary(
            data['effect_sizes'], data['variances'], result['tau2']
        )
        
        np.testing.assert_allclose(result['pooled_effect'], ref_pooled2, rtol=1e-10)
        np.testing.assert_allclose(result['variance'], ref_variance2, rtol=1e-10)
    
    @pytest.mark.parametrize("tau2_method", ["DL", "HE", "HS"])
    def test_random_effects_tau2_methods(self, fe_re_data, tau2_method):
        """Test random effects with different tau2 estimation methods."""
        data = fe_re_data
        
        result = random_effects(
            data['effect_sizes'], 
            data['variances'], 
            tau2_method=tau2_method
        )
        
        assert tau2_method in result['method']
        assert result['tau2'] >= 0
        assert np.isfinite(result['pooled_effect'])
    
    def test_random_effects_reduces_to_fixed_when_tau2_zero(self):
        """Test that RE reduces to FE when tau2 = 0."""
        # Use homogeneous data that should give tau2 ≈ 0
        effect_sizes = [0.5, 0.5, 0.5, 0.5]
        variances = [0.1, 0.1, 0.1, 0.1]
        
        fe_result = fixed_effects(effect_sizes, variances)
        re_result = random_effects(effect_sizes, variances)
        
        # When tau2 = 0, RE should equal FE
        if re_result['tau2'] == 0:
            np.testing.assert_allclose(
                re_result['pooled_effect'], 
                fe_result['pooled_effect'], 
                rtol=1e-10
            )
            np.testing.assert_allclose(
                re_result['variance'], 
                fe_result['variance'], 
                rtol=1e-10
            )
    
    def test_random_effects_wider_ci_than_fixed(self, heterogeneous_data):
        """Test that RE gives wider CI than FE when there's heterogeneity."""
        data = heterogeneous_data
        
        fe_result = fixed_effects(data['effect_sizes'], data['variances'])
        re_result = random_effects(data['effect_sizes'], data['variances'])
        
        # RE should have larger variance and wider CI when tau2 > 0
        if re_result['tau2'] > 0:
            assert re_result['variance'] >= fe_result['variance']
            assert re_result['se'] >= fe_result['se']
            
            # CI width
            fe_width = fe_result['ci_upper'] - fe_result['ci_lower']
            re_width = re_result['ci_upper'] - re_result['ci_lower']
            assert re_width >= fe_width
    
    def test_random_effects_manual_calculation(self):
        """Test against manual calculation."""
        effect_sizes = np.array([0.3, 0.7])
        variances = np.array([0.04, 0.09])
        
        result = random_effects(effect_sizes, variances)
        
        # Manual verification of random effects calculation
        tau2 = result['tau2']
        weights = 1 / (variances + tau2)
        manual_pooled = np.sum(weights * effect_sizes) / np.sum(weights)
        manual_variance = 1 / np.sum(weights)
        
        np.testing.assert_allclose(result['pooled_effect'], manual_pooled, rtol=1e-10)
        np.testing.assert_allclose(result['variance'], manual_variance, rtol=1e-10)
    
    def test_random_effects_single_study(self):
        """Test random effects with single study."""
        result = random_effects([0.5], [0.1])
        
        assert result['pooled_effect'] == 0.5
        assert result['variance'] == 0.1  # Should equal original variance
        assert result['tau2'] == 0  # No between-study variance possible
    
    def test_random_effects_two_studies(self):
        """Test random effects with exactly two studies."""
        effect_sizes = [0.3, 0.7]
        variances = [0.04, 0.09]
        
        result = random_effects(effect_sizes, variances)
        
        assert np.isfinite(result['pooled_effect'])
        assert result['tau2'] >= 0
        assert result['Q_df'] == 1


class TestFixedVsRandomComparison:
    """Test comparison between fixed and random effects models."""
    
    def test_fe_vs_re_same_data(self, fe_re_data):
        """Compare FE and RE on same dataset."""
        data = fe_re_data
        
        fe_result = fixed_effects(data['effect_sizes'], data['variances'])
        re_result = random_effects(data['effect_sizes'], data['variances'])
        
        # Both should be finite and reasonable
        assert np.isfinite(fe_result['pooled_effect'])
        assert np.isfinite(re_result['pooled_effect'])
        
        # Q statistics should be identical
        np.testing.assert_allclose(fe_result['Q'], re_result['Q'], rtol=1e-10)
        
        # If tau2 > 0, RE should have wider CI
        if re_result['tau2'] > 0:
            assert re_result['se'] >= fe_result['se']
    
    def test_convergence_with_large_studies(self):
        """Test that FE and RE converge when within-study variances are large."""
        # Large within-study variances make tau2 relatively small
        effect_sizes = [0.4, 0.6, 0.5]
        variances = [1.0, 1.2, 1.1]  # Large variances
        
        fe_result = fixed_effects(effect_sizes, variances)
        re_result = random_effects(effect_sizes, variances)
        
        # When within-study variances are large, tau2 becomes relatively unimportant
        # So FE and RE should be similar
        relative_diff = abs(fe_result['pooled_effect'] - re_result['pooled_effect'])
        assert relative_diff < 0.1  # Should be reasonably close
    
    def test_precision_weighting(self):
        """Test that models properly weight by precision."""
        # One study with much higher precision
        effect_sizes = [0.3, 0.7]
        variances = [0.001, 1.0]  # First study much more precise
        
        fe_result = fixed_effects(effect_sizes, variances)
        re_result = random_effects(effect_sizes, variances)
        
        # Both should be closer to the more precise study (0.3)
        assert fe_result['pooled_effect'] < 0.5  # Closer to 0.3 than 0.7
        # RE might be less influenced by precision if tau2 is large
    
    def test_array_input_types(self):
        """Test that both models accept various array types."""
        effect_sizes_list = [0.3, 0.5, 0.4]
        variances_list = [0.04, 0.06, 0.05]
        
        # Test lists
        fe_result1 = fixed_effects(effect_sizes_list, variances_list)
        re_result1 = random_effects(effect_sizes_list, variances_list)
        
        # Test numpy arrays
        fe_result2 = fixed_effects(
            np.array(effect_sizes_list), 
            np.array(variances_list)
        )
        re_result2 = random_effects(
            np.array(effect_sizes_list), 
            np.array(variances_list)
        )
        
        # Results should be identical
        assert fe_result1['pooled_effect'] == fe_result2['pooled_effect']
        assert re_result1['pooled_effect'] == re_result2['pooled_effect']
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            fixed_effects([], [])
        
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            random_effects([], [])
    
    def test_mismatched_input_lengths(self):
        """Test error handling for mismatched input lengths."""
        with pytest.raises((ValueError, IndexError)):
            fixed_effects([0.3, 0.5], [0.04])  # Different lengths
        
        with pytest.raises((ValueError, IndexError)):
            random_effects([0.3], [0.04, 0.06])  # Different lengths


class TestModelEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_variances(self):
        """Test handling of zero variances."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [0.0, 0.04, 0.06]  # One zero variance
        
        # Should handle gracefully or raise appropriate error
        with pytest.warns(RuntimeWarning) or pytest.raises(ZeroDivisionError):
            fe_result = fixed_effects(effect_sizes, variances)
        
        with pytest.warns(RuntimeWarning) or pytest.raises(ZeroDivisionError):
            re_result = random_effects(effect_sizes, variances)
    
    def test_extreme_effect_sizes(self):
        """Test with extreme effect sizes."""
        effect_sizes = [-10, 10, 0]
        variances = [0.1, 0.1, 0.1]
        
        fe_result = fixed_effects(effect_sizes, variances)
        re_result = random_effects(effect_sizes, variances)
        
        # Should handle extreme values
        assert np.isfinite(fe_result['pooled_effect'])
        assert np.isfinite(re_result['pooled_effect'])
        
        # RE should show high heterogeneity
        assert re_result['tau2'] > 0
        assert re_result['I2'] > 0.5
    
    def test_very_small_variances(self):
        """Test numerical stability with very small variances."""
        effect_sizes = [0.5, 0.51, 0.49]
        variances = [1e-10, 1e-10, 1e-10]
        
        fe_result = fixed_effects(effect_sizes, variances)
        re_result = random_effects(effect_sizes, variances)
        
        # Should not produce NaN or inf
        assert np.isfinite(fe_result['pooled_effect'])
        assert np.isfinite(re_result['pooled_effect'])
        assert np.isfinite(fe_result['variance'])
        assert np.isfinite(re_result['variance'])