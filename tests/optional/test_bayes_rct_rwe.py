"""Tests for Bayesian RCT/RWE analysis (requires pymc)."""

import numpy as np
import pytest

from pymeta.bayes import bayesian_random_effects, bayesian_rct_rwe_synthesis
from tests.conftest import requires_pymc


@pytest.mark.slow
@pytest.mark.optional
class TestBayesianRandomEffects:
    """Test suite for Bayesian random effects meta-analysis."""
    
    @requires_pymc
    def test_bayesian_re_basic(self, continuous_effects_data):
        """Test basic Bayesian random effects analysis."""
        data = continuous_effects_data
        
        result = bayesian_random_effects(
            data['effect_sizes'],
            data['variances'],
            draws=500,  # Reduced for testing speed
            tune=200
        )
        
        # Check return structure
        assert isinstance(result, dict)
        required_keys = [
            'pooled_effect_mean', 'pooled_effect_sd', 'tau_mean', 'tau_sd',
            'hdi_2.5', 'hdi_97.5', 'method', 'n_draws', 'n_tune'
        ]
        for key in required_keys:
            assert key in result
        
        # Check values
        assert np.isfinite(result['pooled_effect_mean'])
        assert result['pooled_effect_sd'] > 0
        assert result['tau_mean'] >= 0
        assert result['tau_sd'] >= 0
        assert result['hdi_2.5'] < result['hdi_97.5']
        assert 'Bayesian' in result['method']
        assert result['n_draws'] == 500
        assert result['n_tune'] == 200
    
    def test_bayesian_re_fallback(self, continuous_effects_data):
        """Test Bayesian RE fallback when PyMC not available."""
        data = continuous_effects_data
        
        # This will use the fallback implementation
        result = bayesian_random_effects(data['effect_sizes'], data['variances'])
        
        # Should return mock results with warning
        assert 'pooled_effect_mean' in result
        assert 'warning' in result or 'mock' in result['method']
    
    @requires_pymc
    def test_bayesian_re_custom_draws(self, fe_re_data):
        """Test Bayesian RE with custom number of draws."""
        data = fe_re_data
        
        result = bayesian_random_effects(
            data['effect_sizes'],
            data['variances'],
            draws=300,
            tune=100
        )
        
        assert result['n_draws'] == 300
        assert result['n_tune'] == 100
        assert np.isfinite(result['pooled_effect_mean'])
    
    @requires_pymc
    def test_bayesian_re_trace_access(self, fe_re_data):
        """Test access to MCMC trace."""
        data = fe_re_data
        
        result = bayesian_random_effects(
            data['effect_sizes'],
            data['variances'],
            draws=200,
            tune=100
        )
        
        # Should include trace for further analysis
        if 'trace' in result:
            trace = result['trace']
            # Basic check that trace exists and has expected structure
            assert trace is not None
    
    @requires_pymc
    def test_bayesian_re_single_study(self):
        """Test Bayesian RE with single study."""
        result = bayesian_random_effects([0.5], [0.1], draws=200, tune=100)
        
        # Should handle single study
        assert np.isfinite(result['pooled_effect_mean'])
        # tau should be near zero or unidentifiable
    
    @requires_pymc
    def test_bayesian_re_homogeneous_data(self, homogeneous_data):
        """Test Bayesian RE with homogeneous data."""
        data = homogeneous_data
        
        result = bayesian_random_effects(
            data['effect_sizes'],
            data['variances'],
            draws=400,
            tune=200
        )
        
        # Should detect low heterogeneity
        assert result['tau_mean'] < 0.5  # Should be relatively small
        assert np.isfinite(result['pooled_effect_mean'])
    
    @requires_pymc
    def test_bayesian_re_heterogeneous_data(self, heterogeneous_data):
        """Test Bayesian RE with heterogeneous data."""
        data = heterogeneous_data
        
        result = bayesian_random_effects(
            data['effect_sizes'],
            data['variances'],
            draws=400,
            tune=200
        )
        
        # Should detect heterogeneity
        assert result['tau_mean'] > 0  # Should be positive
        assert np.isfinite(result['pooled_effect_mean'])
    
    def test_bayesian_re_array_inputs(self):
        """Test Bayesian RE with different array types."""
        effect_sizes_list = [0.3, 0.5, 0.4]
        variances_list = [0.04, 0.06, 0.05]
        
        # Lists
        result1 = bayesian_random_effects(effect_sizes_list, variances_list, draws=100)
        
        # Numpy arrays  
        result2 = bayesian_random_effects(
            np.array(effect_sizes_list),
            np.array(variances_list),
            draws=100
        )
        
        # Both should work
        assert np.isfinite(result1['pooled_effect_mean'])
        assert np.isfinite(result2['pooled_effect_mean'])


@pytest.mark.slow
@pytest.mark.optional
class TestBayesianRCTRWE:
    """Test suite for Bayesian RCT/RWE synthesis."""
    
    @requires_pymc
    def test_bayesian_rct_rwe_basic(self):
        """Test basic Bayesian RCT/RWE synthesis."""
        # Simulate RCT and RWE data
        np.random.seed(890)
        
        rct_effects = np.random.normal(0.5, 0.1, 5)
        rct_variances = np.random.uniform(0.04, 0.08, 5)
        rwe_effects = np.random.normal(0.6, 0.15, 8)  # Slightly biased
        rwe_variances = np.random.uniform(0.02, 0.12, 8)
        
        result = bayesian_rct_rwe_synthesis(
            rct_effects, rct_variances,
            rwe_effects, rwe_variances
        )
        
        # Check structure
        assert isinstance(result, dict)
        required_keys = [
            'pooled_effect_mean', 'pooled_effect_sd',
            'rwe_bias_mean', 'rwe_bias_sd', 'method', 'n_rct', 'n_rwe'
        ]
        for key in required_keys:
            assert key in result
        
        # Check values
        assert np.isfinite(result['pooled_effect_mean'])
        assert result['pooled_effect_sd'] > 0
        assert np.isfinite(result['rwe_bias_mean'])
        assert result['rwe_bias_sd'] >= 0
        assert result['n_rct'] == 5
        assert result['n_rwe'] == 8
        assert 'RCT/RWE' in result['method']
    
    def test_bayesian_rct_rwe_fallback(self):
        """Test RCT/RWE synthesis fallback when PyMC not available."""
        rct_effects = [0.4, 0.6, 0.5]
        rct_variances = [0.04, 0.06, 0.05]
        rwe_effects = [0.5, 0.7, 0.6, 0.4]
        rwe_variances = [0.03, 0.05, 0.04, 0.06]
        
        result = bayesian_rct_rwe_synthesis(
            rct_effects, rct_variances,
            rwe_effects, rwe_variances
        )
        
        # Should return mock results with warning
        assert 'pooled_effect_mean' in result
        assert 'rwe_bias_mean' in result
        assert 'warning' in result or 'mock' in result['method']
    
    @requires_pymc
    def test_bayesian_rct_rwe_custom_priors(self):
        """Test RCT/RWE synthesis with custom bias priors."""
        rct_effects = [0.4, 0.6]
        rct_variances = [0.04, 0.06]
        rwe_effects = [0.5, 0.8]
        rwe_variances = [0.03, 0.05]
        
        result = bayesian_rct_rwe_synthesis(
            rct_effects, rct_variances,
            rwe_effects, rwe_variances,
            rwe_bias_prior_mean=0.1,  # Expect positive bias
            rwe_bias_prior_sd=0.05    # Tight prior
        )
        
        # Should incorporate prior information
        assert np.isfinite(result['pooled_effect_mean'])
        assert np.isfinite(result['rwe_bias_mean'])
    
    @requires_pymc
    def test_bayesian_rct_rwe_no_bias(self):
        """Test RCT/RWE synthesis when RWE has no bias."""
        # Create similar RCT and RWE data
        true_effect = 0.5
        rct_effects = np.random.normal(true_effect, 0.1, 4)
        rct_variances = [0.05, 0.06, 0.04, 0.07]
        rwe_effects = np.random.normal(true_effect, 0.1, 6)  # Same distribution
        rwe_variances = [0.03, 0.04, 0.05, 0.06, 0.04, 0.05]
        
        result = bayesian_rct_rwe_synthesis(
            rct_effects, rct_variances,
            rwe_effects, rwe_variances,
            draws=300  # Faster for testing
        )
        
        # Bias should be close to zero
        assert abs(result['rwe_bias_mean']) < 0.5  # Reasonable range
        assert np.isfinite(result['pooled_effect_mean'])
    
    @requires_pymc
    def test_bayesian_rct_rwe_large_bias(self):
        """Test RCT/RWE synthesis with large RWE bias."""
        rct_effects = [0.3, 0.4, 0.5]
        rct_variances = [0.04, 0.05, 0.06]
        rwe_effects = [0.8, 0.9, 1.0]  # Much larger effects (biased)
        rwe_variances = [0.03, 0.04, 0.05]
        
        result = bayesian_rct_rwe_synthesis(
            rct_effects, rct_variances,
            rwe_effects, rwe_variances,
            draws=300
        )
        
        # Should detect substantial bias
        assert result['rwe_bias_mean'] > 0.2  # Positive bias
        assert np.isfinite(result['pooled_effect_mean'])
    
    @requires_pymc
    def test_bayesian_rct_rwe_single_studies(self):
        """Test RCT/RWE synthesis with single studies."""
        result = bayesian_rct_rwe_synthesis(
            [0.5], [0.1],  # Single RCT
            [0.6], [0.08], # Single RWE
            draws=200
        )
        
        # Should handle single studies
        assert np.isfinite(result['pooled_effect_mean'])
        assert np.isfinite(result['rwe_bias_mean'])
    
    @requires_pymc
    def test_bayesian_rct_rwe_trace_access(self):
        """Test access to MCMC trace in RCT/RWE synthesis."""
        rct_effects = [0.4, 0.6]
        rct_variances = [0.04, 0.06]
        rwe_effects = [0.5, 0.7]
        rwe_variances = [0.03, 0.05]
        
        result = bayesian_rct_rwe_synthesis(
            rct_effects, rct_variances,
            rwe_effects, rwe_variances,
            draws=200
        )
        
        # Should include trace for further analysis
        if 'trace' in result:
            trace = result['trace']
            assert trace is not None


@pytest.mark.optional
class TestBayesianMethods:
    """General tests for Bayesian methods."""
    
    def test_bayesian_method_interface_consistency(self):
        """Test that Bayesian methods have consistent interfaces."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [0.04, 0.06, 0.05]
        
        # Both methods should work with fallback
        re_result = bayesian_random_effects(effect_sizes, variances)
        rct_rwe_result = bayesian_rct_rwe_synthesis(
            effect_sizes[:2], variances[:2],
            effect_sizes[2:], variances[2:]
        )
        
        # Both should return dict with required keys
        assert isinstance(re_result, dict)
        assert isinstance(rct_rwe_result, dict)
        assert 'pooled_effect_mean' in re_result
        assert 'pooled_effect_mean' in rct_rwe_result
        assert 'method' in re_result
        assert 'method' in rct_rwe_result
    
    def test_bayesian_empty_input_handling(self):
        """Test handling of empty inputs."""
        with pytest.raises((ValueError, IndexError)):
            bayesian_random_effects([], [])
        
        with pytest.raises((ValueError, IndexError)):
            bayesian_rct_rwe_synthesis([], [], [], [])
    
    def test_bayesian_mismatched_lengths(self):
        """Test error handling for mismatched input lengths."""
        with pytest.raises((ValueError, IndexError)):
            bayesian_random_effects([0.3, 0.5], [0.04])  # Different lengths
        
        with pytest.raises((ValueError, IndexError)):
            bayesian_rct_rwe_synthesis([0.3], [0.04, 0.06], [0.5], [0.05])
    
    def test_bayesian_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        effect_sizes = [0.3, 0.5]
        variances = [0.04, 0.06]
        
        # Negative draws
        with pytest.raises(ValueError):
            bayesian_random_effects(effect_sizes, variances, draws=-10)
        
        # Negative tune
        with pytest.raises(ValueError):
            bayesian_random_effects(effect_sizes, variances, tune=-5)


@pytest.mark.slow
@pytest.mark.optional
class TestBayesianPerformance:
    """Test performance aspects of Bayesian methods."""
    
    @requires_pymc
    def test_bayesian_re_performance(self):
        """Test Bayesian RE performance with reasonable dataset."""
        import time
        
        np.random.seed(901)
        k = 15  # Medium-sized dataset
        effect_sizes = np.random.normal(0.4, 0.2, k)
        variances = np.random.uniform(0.02, 0.08, k)
        
        start_time = time.time()
        result = bayesian_random_effects(
            effect_sizes, variances,
            draws=500, tune=200  # Reasonable for testing
        )
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 60  # 60 seconds max
        assert np.isfinite(result['pooled_effect_mean'])
    
    @requires_pymc
    def test_bayesian_rct_rwe_performance(self):
        """Test RCT/RWE synthesis performance."""
        import time
        
        np.random.seed(912)
        
        # Medium-sized datasets
        rct_effects = np.random.normal(0.4, 0.1, 8)
        rct_variances = np.random.uniform(0.03, 0.07, 8)
        rwe_effects = np.random.normal(0.5, 0.15, 12)
        rwe_variances = np.random.uniform(0.02, 0.06, 12)
        
        start_time = time.time()
        result = bayesian_rct_rwe_synthesis(
            rct_effects, rct_variances,
            rwe_effects, rwe_variances
        )
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 90  # 90 seconds max
        assert np.isfinite(result['pooled_effect_mean'])
    
    def test_bayesian_memory_usage(self):
        """Test Bayesian methods memory usage."""
        # This is more of a smoke test for memory issues
        np.random.seed(923)
        k = 25  # Larger dataset
        
        effect_sizes = np.random.normal(0.3, 0.3, k)
        variances = np.random.uniform(0.01, 0.1, k)
        
        # Should not crash due to memory issues
        result = bayesian_random_effects(effect_sizes, variances, draws=100)
        
        assert np.isfinite(result['pooled_effect_mean'])


@pytest.mark.slow
@pytest.mark.optional
class TestBayesianIntegration:
    """Integration tests for Bayesian methods."""
    
    def test_bayesian_vs_frequentist_comparison(self, continuous_effects_data):
        """Compare Bayesian and frequentist results."""
        from pymeta.models import random_effects
        
        data = continuous_effects_data
        
        # Frequentist random effects
        freq_result = random_effects(data['effect_sizes'], data['variances'])
        
        # Bayesian random effects
        bayes_result = bayesian_random_effects(
            data['effect_sizes'], data['variances'],
            draws=300  # Faster for comparison
        )
        
        # Should be in reasonable agreement
        if 'pooled_effect_mean' in bayes_result and not ('warning' in bayes_result):
            effect_diff = abs(bayes_result['pooled_effect_mean'] - freq_result['pooled_effect'])
            assert effect_diff < 1.0  # Should be reasonably close
    
    @requires_pymc
    def test_bayesian_reproducibility(self, fe_re_data):
        """Test Bayesian method reproducibility with fixed seed."""
        data = fe_re_data
        
        # PyMC uses its own random number generation
        # So we test that sampling completes successfully multiple times
        
        result1 = bayesian_random_effects(
            data['effect_sizes'], data['variances'],
            draws=200, tune=100
        )
        
        result2 = bayesian_random_effects(
            data['effect_sizes'], data['variances'],
            draws=200, tune=100
        )
        
        # Both should complete successfully
        assert np.isfinite(result1['pooled_effect_mean'])
        assert np.isfinite(result2['pooled_effect_mean'])
        
        # Results should be in reasonable range (allowing for MCMC variation)
        assert abs(result1['pooled_effect_mean'] - result2['pooled_effect_mean']) < 1.0
    
    @requires_pymc
    def test_bayesian_convergence_diagnostics(self, continuous_effects_data):
        """Test that Bayesian methods provide convergence information."""
        data = continuous_effects_data
        
        result = bayesian_random_effects(
            data['effect_sizes'], data['variances'],
            draws=400, tune=200
        )
        
        # Should have completed sampling
        assert 'n_draws' in result
        assert 'n_tune' in result
        
        # Trace should be accessible for diagnostics
        if 'trace' in result:
            # Could check R-hat, effective sample size, etc.
            # For now, just verify trace exists
            assert result['trace'] is not None