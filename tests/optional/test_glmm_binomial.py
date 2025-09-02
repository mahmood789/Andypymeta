"""Tests for GLMM binomial models (marked as slow/optional)."""

import numpy as np
import pytest

from pymeta.glmm import glmm_binomial, glmm_continuous
from tests.conftest import requires_statsmodels


@pytest.mark.slow
@pytest.mark.optional
class TestGLMMBinomial:
    """Test suite for GLMM binomial models (computationally intensive)."""
    
    def test_glmm_binomial_basic(self, binary_table_2x2):
        """Test basic GLMM binomial analysis."""
        data = binary_table_2x2
        
        result = glmm_binomial(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control']
        )
        
        # Check return structure
        assert isinstance(result, dict)
        required_keys = [
            'log_or', 'se_log_or', 'ci_lower', 'ci_upper',
            'tau2', 'tau', 'converged', 'method'
        ]
        for key in required_keys:
            assert key in result
        
        # Check values
        assert np.isfinite(result['log_or'])
        assert result['se_log_or'] > 0
        assert result['ci_lower'] < result['ci_upper']
        assert result['tau2'] >= 0
        assert result['tau'] >= 0
        assert result['tau'] == np.sqrt(result['tau2'])
        assert isinstance(result['converged'], bool)
        assert 'GLMM' in result['method']
    
    def test_glmm_binomial_with_study_ids(self, binary_table_2x2):
        """Test GLMM with custom study identifiers."""
        data = binary_table_2x2
        study_ids = ['RCT_A', 'RCT_B', 'RCT_C', 'RCT_D']
        
        result = glmm_binomial(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control'],
            study_ids=study_ids
        )
        
        assert np.isfinite(result['log_or'])
        assert result['converged']
    
    def test_glmm_binomial_convergence_settings(self, binary_table_2x2):
        """Test GLMM with different convergence settings."""
        data = binary_table_2x2
        
        result = glmm_binomial(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control'],
            max_iter=50  # Reduced iterations
        )
        
        # Should still converge or handle gracefully
        assert np.isfinite(result['log_or'])
        assert 'n_iter' in result or 'converged' in result
    
    def test_glmm_binomial_with_zeros(self, binary_table_with_zeros):
        """Test GLMM with zero events (challenging case)."""
        data = binary_table_with_zeros
        
        result = glmm_binomial(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control']
        )
        
        # Should handle zero events
        assert np.isfinite(result['log_or'])
        assert result['se_log_or'] > 0
    
    @requires_statsmodels
    def test_glmm_binomial_real_implementation(self, binary_table_2x2):
        """Test GLMM with actual statsmodels (if available)."""
        data = binary_table_2x2
        
        result = glmm_binomial(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control']
        )
        
        # With real implementation, should have more detailed results
        assert 'log_or' in result
        assert 'tau2' in result
    
    def test_glmm_binomial_single_study(self):
        """Test GLMM with single study."""
        result = glmm_binomial([10], [50], [5], [50])
        
        # Should handle single study
        assert np.isfinite(result['log_or'])
        # tau2 might be 0 or unidentifiable with single study
    
    def test_glmm_binomial_large_dataset(self):
        """Test GLMM performance with larger dataset."""
        np.random.seed(678)
        k = 20  # Larger number of studies
        
        # Simulate binomial data
        n_treatment = np.random.randint(50, 200, k)
        n_control = np.random.randint(50, 200, k)
        
        # Simulate events with some heterogeneity
        p_treatment = np.random.beta(2, 3, k)  # Variable event rates
        p_control = p_treatment * np.random.uniform(0.5, 0.9, k)
        
        events_treatment = np.random.binomial(n_treatment, p_treatment)
        events_control = np.random.binomial(n_control, p_control)
        
        result = glmm_binomial(events_treatment, n_treatment, events_control, n_control)
        
        # Should handle larger dataset
        assert np.isfinite(result['log_or'])
        assert result['tau2'] >= 0
    
    def test_glmm_binomial_extreme_event_rates(self):
        """Test GLMM with extreme event rates."""
        # Very rare events
        events_treatment = [1, 2, 0, 1]
        n_treatment = [100, 120, 90, 110]
        events_control = [0, 1, 1, 0]
        n_control = [100, 100, 95, 105]
        
        result = glmm_binomial(events_treatment, n_treatment, events_control, n_control)
        
        # Should handle rare events
        assert np.isfinite(result['log_or'])
    
    def test_glmm_binomial_array_inputs(self):
        """Test GLMM with different array input types."""
        events_t_list = [10, 15, 8]
        n_t_list = [50, 60, 45]
        events_c_list = [5, 8, 6]
        n_c_list = [50, 55, 40]
        
        # Lists
        result1 = glmm_binomial(events_t_list, n_t_list, events_c_list, n_c_list)
        
        # Numpy arrays
        result2 = glmm_binomial(
            np.array(events_t_list), np.array(n_t_list),
            np.array(events_c_list), np.array(n_c_list)
        )
        
        # Should give similar results (allowing for randomness in fitting)
        assert np.isfinite(result1['log_or'])
        assert np.isfinite(result2['log_or'])


@pytest.mark.optional
class TestGLMMContinuous:
    """Test suite for GLMM continuous outcomes."""
    
    def test_glmm_continuous_basic(self, continuous_effects_data):
        """Test basic GLMM for continuous outcomes."""
        data = continuous_effects_data
        
        result = glmm_continuous(data['effect_sizes'], data['variances'])
        
        # Check structure
        assert isinstance(result, dict)
        required_keys = ['pooled_effect', 'se', 'ci_lower', 'ci_upper', 'method']
        for key in required_keys:
            assert key in result
        
        # Check values
        assert np.isfinite(result['pooled_effect'])
        assert result['se'] > 0
        assert result['ci_lower'] < result['ci_upper']
        assert 'GLMM' in result['method']
    
    def test_glmm_continuous_with_covariates(self, continuous_effects_data):
        """Test GLMM with study-level covariates."""
        data = continuous_effects_data
        
        # Create some dummy covariates
        k = len(data['effect_sizes'])
        covariates = np.random.normal(0, 1, k)
        
        result = glmm_continuous(
            data['effect_sizes'],
            data['variances'],
            covariates=covariates
        )
        
        assert np.isfinite(result['pooled_effect'])
        assert 'GLMM' in result['method']
    
    def test_glmm_continuous_vs_simple_meta(self, fe_re_data):
        """Compare GLMM continuous with simple meta-analysis."""
        data = fe_re_data
        
        # Simple weighted average
        weights = 1 / np.array(data['variances'])
        simple_pooled = np.sum(weights * data['effect_sizes']) / np.sum(weights)
        
        # GLMM result
        glmm_result = glmm_continuous(data['effect_sizes'], data['variances'])
        
        # Should be reasonably close for simple case
        assert abs(glmm_result['pooled_effect'] - simple_pooled) < 1.0
    
    def test_glmm_continuous_single_study(self):
        """Test GLMM continuous with single study."""
        result = glmm_continuous([0.5], [0.1])
        
        # Should return the single study result
        assert abs(result['pooled_effect'] - 0.5) < 0.1
        assert result['se'] > 0


@pytest.mark.slow
@pytest.mark.optional
class TestGLMMPerformance:
    """Test performance and computational aspects of GLMM."""
    
    def test_glmm_computational_time(self):
        """Test that GLMM completes in reasonable time."""
        import time
        
        # Medium-sized dataset
        np.random.seed(789)
        k = 15
        
        events_treatment = np.random.poisson(10, k)
        n_treatment = events_treatment + np.random.poisson(40, k)
        events_control = np.random.poisson(8, k)
        n_control = events_control + np.random.poisson(35, k)
        
        start_time = time.time()
        result = glmm_binomial(events_treatment, n_treatment, events_control, n_control)
        end_time = time.time()
        
        # Should complete reasonably quickly (allowing for simplified implementation)
        assert end_time - start_time < 30  # 30 seconds max
        assert np.isfinite(result['log_or'])
    
    def test_glmm_memory_usage(self):
        """Test GLMM memory usage with larger datasets."""
        # This is more of a smoke test
        k = 50  # Larger dataset
        
        events_treatment = np.random.randint(5, 25, k)
        n_treatment = events_treatment + np.random.randint(25, 75, k)
        events_control = np.random.randint(3, 20, k)
        n_control = events_control + np.random.randint(25, 75, k)
        
        # Should not crash or use excessive memory
        result = glmm_binomial(events_treatment, n_treatment, events_control, n_control)
        
        assert np.isfinite(result['log_or'])
    
    def test_glmm_convergence_monitoring(self):
        """Test GLMM convergence monitoring."""
        # Use challenging data that might have convergence issues
        events_treatment = [0, 50, 1, 49]  # Extreme values
        n_treatment = [50, 50, 50, 50]
        events_control = [1, 1, 49, 1]
        n_control = [50, 50, 50, 50]
        
        result = glmm_binomial(events_treatment, n_treatment, events_control, n_control)
        
        # Should report convergence status
        assert 'converged' in result
        # Even if not converged, should return some result
        assert 'log_or' in result


@pytest.mark.optional
class TestGLMMEdgeCases:
    """Test edge cases for GLMM methods."""
    
    def test_glmm_empty_input(self):
        """Test GLMM with empty inputs."""
        with pytest.raises((ValueError, IndexError)):
            glmm_binomial([], [], [], [])
        
        with pytest.raises((ValueError, IndexError)):
            glmm_continuous([], [])
    
    def test_glmm_mismatched_lengths(self):
        """Test GLMM with mismatched input lengths."""
        with pytest.raises((ValueError, IndexError)):
            glmm_binomial([10, 15], [50], [5, 8], [50, 55])
    
    def test_glmm_invalid_data(self):
        """Test GLMM with invalid data."""
        # Events exceeding sample size
        with pytest.raises((ValueError, RuntimeError)):
            glmm_binomial([60], [50], [10], [50])  # 60 > 50
    
    def test_glmm_zero_variances(self):
        """Test GLMM continuous with zero variances."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [0.0, 0.04, 0.06]  # One zero variance
        
        # Should handle or raise appropriate error
        try:
            result = glmm_continuous(effect_sizes, variances)
            assert np.isfinite(result['pooled_effect'])
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise error
            pass
    
    def test_glmm_negative_variances(self):
        """Test GLMM continuous with negative variances."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [-0.04, 0.06, 0.05]  # One negative variance
        
        # Should raise error
        with pytest.raises(ValueError):
            glmm_continuous(effect_sizes, variances)


@pytest.mark.slow
@pytest.mark.optional
class TestGLMMIntegration:
    """Integration tests for GLMM methods."""
    
    def test_glmm_vs_standard_meta_analysis(self, binary_table_2x2):
        """Compare GLMM results with standard meta-analysis."""
        from pymeta.effects import binary_effects
        from pymeta.models import random_effects
        
        data = binary_table_2x2
        
        # Standard meta-analysis
        binary_results = binary_effects(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control']
        )
        re_results = random_effects(binary_results['effect_size'], binary_results['variance'])
        
        # GLMM analysis
        glmm_results = glmm_binomial(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control']
        )
        
        # Should be in the same ballpark
        log_or_diff = abs(glmm_results['log_or'] - re_results['pooled_effect'])
        assert log_or_diff < 2.0  # Reasonable agreement (allowing for method differences)
    
    def test_glmm_reproducibility(self, binary_table_2x2):
        """Test GLMM reproducibility."""
        data = binary_table_2x2
        
        # Run same analysis twice
        result1 = glmm_binomial(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control']
        )
        
        result2 = glmm_binomial(
            data['events_treatment'],
            data['n_treatment'],
            data['events_control'],
            data['n_control']
        )
        
        # Should give identical results (deterministic)
        np.testing.assert_allclose(result1['log_or'], result2['log_or'], rtol=1e-10)
        np.testing.assert_allclose(result1['tau2'], result2['tau2'], rtol=1e-10)
    
    def test_glmm_with_missing_optional_deps(self):
        """Test GLMM behavior when optional dependencies are missing."""
        # This tests the fallback behavior in our implementation
        # The actual function should handle missing statsmodels gracefully
        
        # Import the function to test fallback
        from pymeta.glmm import glmm_binomial
        
        result = glmm_binomial([10], [50], [5], [50])
        
        # Should return some result even if using simplified implementation
        assert 'log_or' in result
        assert 'method' in result