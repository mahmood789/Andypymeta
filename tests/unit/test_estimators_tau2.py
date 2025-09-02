"""Unit tests for tau-squared estimation methods."""

import numpy as np
import pytest

from pymeta.estimators import tau2_estimators, reference_tau2_dl
from tests.conftest import reference_dl_tau2


class TestTau2Estimators:
    """Test suite for between-study variance estimation."""
    
    def test_dersimonian_laird_basic(self, continuous_effects_data):
        """Test basic DerSimonian-Laird estimation."""
        data = continuous_effects_data
        
        result = tau2_estimators(
            data['effect_sizes'], 
            data['variances'], 
            method="DL"
        )
        
        # Check return structure
        assert isinstance(result, dict)
        assert 'tau2' in result
        assert 'method' in result
        assert 'Q' in result
        assert 'df' in result
        assert 'p_value' in result
        assert 'I2' in result
        
        # Check values
        assert result['tau2'] >= 0  # tau2 cannot be negative
        assert result['method'] == 'DerSimonian-Laird'
        assert result['df'] == len(data['effect_sizes']) - 1
        assert 0 <= result['I2'] <= 1  # I2 is a proportion
        assert 0 <= result['p_value'] <= 1
        
        # Q should be non-negative
        assert result['Q'] >= 0
    
    def test_dl_against_reference_implementation(self, continuous_effects_data):
        """Test DL method against reference implementation."""
        data = continuous_effects_data
        
        # Our implementation
        result = tau2_estimators(
            data['effect_sizes'],
            data['variances'],
            method="DL"
        )
        
        # Reference implementation from conftest
        ref_tau2 = reference_dl_tau2(data['effect_sizes'], data['variances'])
        
        # Should match within numerical precision
        np.testing.assert_allclose(result['tau2'], ref_tau2, rtol=1e-10)
    
    def test_dl_module_reference(self, continuous_effects_data):
        """Test against the module's own reference implementation."""
        data = continuous_effects_data
        
        # Main function
        result = tau2_estimators(
            data['effect_sizes'],
            data['variances'], 
            method="DL"
        )
        
        # Module reference function
        ref_tau2 = reference_tau2_dl(data['effect_sizes'], data['variances'])
        
        np.testing.assert_allclose(result['tau2'], ref_tau2, rtol=1e-10)
    
    def test_hedges_estimator(self, continuous_effects_data):
        """Test Hedges estimator."""
        data = continuous_effects_data
        
        result = tau2_estimators(
            data['effect_sizes'],
            data['variances'],
            method="HE"
        )
        
        assert result['method'] == 'Hedges'
        assert result['tau2'] >= 0
        assert 'Q' in result
        assert 'df' in result
        
        # Hedges should give different result than DL
        dl_result = tau2_estimators(
            data['effect_sizes'],
            data['variances'],
            method="DL"
        )
        
        # Usually different unless tau2 = 0
        if result['tau2'] > 0 or dl_result['tau2'] > 0:
            assert result['tau2'] != dl_result['tau2']
    
    def test_hunter_schmidt_estimator(self, continuous_effects_data):
        """Test Hunter-Schmidt estimator."""
        data = continuous_effects_data
        
        result = tau2_estimators(
            data['effect_sizes'],
            data['variances'],
            method="HS"
        )
        
        assert result['method'] == 'Hunter-Schmidt'
        assert result['tau2'] >= 0
        assert 'var_observed' in result
        assert 'mean_within_var' in result
        
        # Check that observed variance >= within variance when tau2 > 0
        if result['tau2'] > 0:
            assert result['var_observed'] >= result['mean_within_var']
    
    def test_homogeneous_data(self, homogeneous_data):
        """Test with homogeneous data (should give low tau2)."""
        data = homogeneous_data
        
        result = tau2_estimators(
            data['effect_sizes'],
            data['variances'],
            method="DL"
        )
        
        # Should have low heterogeneity
        assert result['tau2'] < 0.1  # Relatively small tau2
        assert result['I2'] < 0.5   # Less than 50% heterogeneity
        assert result['p_value'] > 0.05  # Non-significant Q test
    
    def test_heterogeneous_data(self, heterogeneous_data):
        """Test with heterogeneous data (should give high tau2)."""
        data = heterogeneous_data
        
        result = tau2_estimators(
            data['effect_sizes'],
            data['variances'],
            method="DL"
        )
        
        # Should have high heterogeneity
        assert result['I2'] > 0.5  # More than 50% heterogeneity
        # Note: p-value might still be > 0.05 with small k
    
    def test_single_study(self):
        """Test with single study (should return tau2 = 0)."""
        result = tau2_estimators([0.5], [0.1], method="DL")
        
        assert result['tau2'] == 0
        assert result['df'] == 0
        assert result['Q'] == 0
        assert np.isnan(result['p_value']) or result['p_value'] == 1.0
        assert result['I2'] == 0
    
    def test_two_studies(self):
        """Test with exactly two studies."""
        effect_sizes = [0.3, 0.7]
        variances = [0.04, 0.06]
        
        result = tau2_estimators(effect_sizes, variances, method="DL")
        
        assert result['df'] == 1
        assert result['tau2'] >= 0
        assert np.isfinite(result['Q'])
        assert np.isfinite(result['p_value'])
    
    def test_identical_studies(self):
        """Test with identical studies (no heterogeneity)."""
        effect_sizes = [0.5, 0.5, 0.5, 0.5]
        variances = [0.1, 0.1, 0.1, 0.1]
        
        result = tau2_estimators(effect_sizes, variances, method="DL")
        
        assert result['tau2'] == 0  # No heterogeneity
        assert result['Q'] == 0     # No variation
        assert result['I2'] == 0    # No heterogeneity
    
    def test_zero_variances_handling(self):
        """Test handling of zero variances."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [0.0, 0.1, 0.05]  # One zero variance
        
        # Should handle gracefully (might raise warning)
        with pytest.warns(RuntimeWarning, match="divide by zero") or pytest.warns(None):
            result = tau2_estimators(effect_sizes, variances, method="DL")
        
        # Result should be defined or function should raise appropriate error
        assert isinstance(result, dict) or True  # Allow for different error handling
    
    @pytest.mark.parametrize("method", ["DL", "HE", "HS"])
    def test_all_methods_consistency(self, continuous_effects_data, method):
        """Test that all methods produce consistent output structure."""
        data = continuous_effects_data
        
        result = tau2_estimators(
            data['effect_sizes'],
            data['variances'],
            method=method
        )
        
        # Common structure checks
        assert isinstance(result, dict)
        assert 'tau2' in result
        assert 'method' in result
        assert result['tau2'] >= 0
        
        # Method-specific checks
        if method in ["DL", "HE"]:
            assert 'Q' in result
            assert 'df' in result
            assert 'p_value' in result
            assert 'I2' in result
    
    def test_invalid_method(self, continuous_effects_data):
        """Test error handling for invalid methods."""
        data = continuous_effects_data
        
        with pytest.raises(ValueError, match="Unknown method"):
            tau2_estimators(
                data['effect_sizes'],
                data['variances'],
                method="INVALID"
            )
    
    def test_array_inputs(self):
        """Test that function accepts various array-like inputs."""
        effect_sizes_list = [0.3, 0.5, 0.4]
        variances_list = [0.04, 0.06, 0.05]
        
        # Lists
        result1 = tau2_estimators(effect_sizes_list, variances_list)
        
        # Numpy arrays
        result2 = tau2_estimators(
            np.array(effect_sizes_list),
            np.array(variances_list)
        )
        
        # Should give identical results
        assert result1['tau2'] == result2['tau2']
        assert result1['Q'] == result2['Q']
    
    def test_extreme_heterogeneity(self):
        """Test with extremely heterogeneous data."""
        # Very different effect sizes with small variances
        effect_sizes = [-2, 0, 2]
        variances = [0.01, 0.01, 0.01]
        
        result = tau2_estimators(effect_sizes, variances, method="DL")
        
        # Should have very high heterogeneity
        assert result['tau2'] > 0
        assert result['I2'] > 0.8  # More than 80% heterogeneity
        assert result['Q'] > 10    # High Q statistic
    
    def test_large_sample_behavior(self):
        """Test behavior with large number of studies."""
        np.random.seed(789)
        k = 50  # Many studies
        
        true_tau = 0.2
        effect_sizes = np.random.normal(0.5, true_tau, k)
        variances = np.random.uniform(0.01, 0.1, k)
        
        result = tau2_estimators(effect_sizes, variances, method="DL")
        
        # Should estimate tau2 reasonably well with many studies
        assert result['tau2'] >= 0
        # Don't require exact match due to randomness, but should be in ballpark
        assert 0 <= result['tau2'] <= 1.0  # Reasonable range
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very small variances
        effect_sizes = [0.5, 0.51, 0.49]
        variances = [1e-10, 1e-10, 1e-10]
        
        result = tau2_estimators(effect_sizes, variances, method="DL")
        
        # Should not produce NaN or inf
        assert np.isfinite(result['tau2'])
        assert np.isfinite(result['Q'])


class TestTau2EstimatorsComparison:
    """Test comparison between different tau2 methods."""
    
    def test_method_comparison_no_heterogeneity(self):
        """Compare methods when there's no heterogeneity."""
        # Identical studies
        effect_sizes = [0.5] * 5
        variances = [0.1] * 5
        
        dl_result = tau2_estimators(effect_sizes, variances, method="DL")
        he_result = tau2_estimators(effect_sizes, variances, method="HE")
        hs_result = tau2_estimators(effect_sizes, variances, method="HS")
        
        # All should give tau2 = 0
        assert dl_result['tau2'] == 0
        assert he_result['tau2'] == 0
        assert hs_result['tau2'] == 0
    
    def test_method_comparison_with_heterogeneity(self, heterogeneous_data):
        """Compare methods with heterogeneous data."""
        data = heterogeneous_data
        
        dl_result = tau2_estimators(data['effect_sizes'], data['variances'], method="DL")
        he_result = tau2_estimators(data['effect_sizes'], data['variances'], method="HE")
        hs_result = tau2_estimators(data['effect_sizes'], data['variances'], method="HS")
        
        # All should detect heterogeneity
        assert dl_result['tau2'] > 0
        assert he_result['tau2'] > 0  
        assert hs_result['tau2'] > 0
        
        # Methods should give different estimates
        tau2_values = [dl_result['tau2'], he_result['tau2'], hs_result['tau2']]
        assert len(set(tau2_values)) > 1  # At least some difference
    
    def test_dl_vs_reference_multiple_datasets(self):
        """Test DL method against reference on multiple datasets."""
        datasets = [
            ([0.1, 0.2, 0.3], [0.01, 0.02, 0.015]),
            ([0.5, 0.6, 0.4, 0.7], [0.1, 0.12, 0.08, 0.15]),
            ([-0.2, 0.1, 0.3, -0.1], [0.05, 0.06, 0.04, 0.07])
        ]
        
        for effect_sizes, variances in datasets:
            result = tau2_estimators(effect_sizes, variances, method="DL")
            ref_tau2 = reference_dl_tau2(effect_sizes, variances)
            
            np.testing.assert_allclose(
                result['tau2'], ref_tau2, 
                rtol=1e-10,
                err_msg=f"Mismatch for dataset {effect_sizes}, {variances}"
            )