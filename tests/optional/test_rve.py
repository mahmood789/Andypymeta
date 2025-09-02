"""Tests for robust variance estimation methods."""

import numpy as np
import pytest

from pymeta.rve import robust_variance_estimation, hierarchical_effects_model, sensitivity_analysis_rve


@pytest.mark.optional
class TestRobustVarianceEstimation:
    """Test suite for robust variance estimation."""
    
    def test_rve_basic(self, continuous_effects_data):
        """Test basic RVE calculation."""
        data = continuous_effects_data
        
        # Create some cluster structure
        k = len(data['effect_sizes'])
        cluster_ids = np.repeat(range(k//2), 2)[:k]  # Pairs of studies
        
        result = robust_variance_estimation(
            data['effect_sizes'],
            data['variances'],
            cluster_ids=cluster_ids
        )
        
        # Check structure
        assert isinstance(result, dict)
        required_keys = [
            'pooled_effect', 'robust_se', 'robust_variance',
            't_value', 'df', 'p_value', 'ci_lower', 'ci_upper',
            'n_clusters', 'n_effects', 'method'
        ]
        for key in required_keys:
            assert key in result
        
        # Check values
        assert np.isfinite(result['pooled_effect'])
        assert result['robust_se'] > 0
        assert result['robust_variance'] > 0
        assert np.isfinite(result['t_value'])
        assert 0 <= result['p_value'] <= 1
        assert result['ci_lower'] < result['ci_upper']
        assert result['n_clusters'] > 0
        assert result['n_effects'] == k
        assert 'Robust Variance' in result['method']
    
    def test_rve_no_clustering(self, continuous_effects_data):
        """Test RVE with no clustering (each study independent)."""
        data = continuous_effects_data
        
        result = robust_variance_estimation(
            data['effect_sizes'],
            data['variances']
            # No cluster_ids provided - each study is its own cluster
        )
        
        # Should work with independent studies
        assert np.isfinite(result['pooled_effect'])
        assert result['n_clusters'] == len(data['effect_sizes'])
        assert result['n_effects'] == len(data['effect_sizes'])
    
    def test_rve_vs_standard_meta_analysis(self, fe_re_data):
        """Compare RVE with standard meta-analysis when no clustering."""
        from pymeta.models import fixed_effects
        
        data = fe_re_data
        
        # Standard fixed effects
        fe_result = fixed_effects(data['effect_sizes'], data['variances'])
        
        # RVE with no clustering
        rve_result = robust_variance_estimation(
            data['effect_sizes'],
            data['variances']
        )
        
        # Should be similar when no clustering
        np.testing.assert_allclose(
            rve_result['pooled_effect'], 
            fe_result['pooled_effect'], 
            rtol=1e-10
        )
        
        # RVE SE might be slightly different due to small sample correction
        assert rve_result['robust_se'] > 0
    
    def test_rve_with_clustering(self):
        """Test RVE with actual clustering structure."""
        # Create clustered data - multiple effect sizes per study
        effect_sizes = [0.3, 0.35, 0.5, 0.55, 0.4, 0.42]
        variances = [0.04, 0.05, 0.06, 0.07, 0.05, 0.06]
        cluster_ids = [1, 1, 2, 2, 3, 3]  # 3 clusters, 2 effects each
        
        result = robust_variance_estimation(
            effect_sizes, variances, cluster_ids=cluster_ids
        )
        
        assert result['n_clusters'] == 3
        assert result['n_effects'] == 6
        assert result['df'] == 2  # n_clusters - 1
        assert np.isfinite(result['pooled_effect'])
    
    def test_rve_small_sample_correction(self):
        """Test small sample correction in RVE."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [0.04, 0.06, 0.05]
        cluster_ids = [1, 2, 3]  # Each effect is its own cluster
        
        # With correction
        result_with = robust_variance_estimation(
            effect_sizes, variances, cluster_ids,
            small_sample_correction=True
        )
        
        # Without correction
        result_without = robust_variance_estimation(
            effect_sizes, variances, cluster_ids,
            small_sample_correction=False
        )
        
        # With correction should have larger SE
        assert result_with['robust_se'] >= result_without['robust_se']
    
    def test_rve_single_cluster(self):
        """Test RVE with single cluster."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [0.04, 0.06, 0.05]
        cluster_ids = [1, 1, 1]  # All in same cluster
        
        result = robust_variance_estimation(
            effect_sizes, variances, cluster_ids
        )
        
        assert result['n_clusters'] == 1
        assert result['df'] == 0  # 1 - 1 = 0
        # With 0 df, t-test is not meaningful, but calculation should work
        assert np.isfinite(result['pooled_effect'])
    
    def test_rve_unbalanced_clusters(self):
        """Test RVE with unbalanced cluster sizes."""
        effect_sizes = [0.3, 0.5, 0.4, 0.6, 0.45]
        variances = [0.04, 0.06, 0.05, 0.07, 0.055]
        cluster_ids = [1, 1, 2, 3, 3]  # Clusters: 2, 1, 2 effects
        
        result = robust_variance_estimation(
            effect_sizes, variances, cluster_ids
        )
        
        assert result['n_clusters'] == 3
        assert result['n_effects'] == 5
        assert np.isfinite(result['pooled_effect'])
    
    def test_rve_array_inputs(self):
        """Test RVE with different array input types."""
        effect_sizes_list = [0.3, 0.5, 0.4]
        variances_list = [0.04, 0.06, 0.05]
        cluster_ids_list = [1, 2, 3]
        
        # Lists
        result1 = robust_variance_estimation(
            effect_sizes_list, variances_list, cluster_ids_list
        )
        
        # Numpy arrays
        result2 = robust_variance_estimation(
            np.array(effect_sizes_list),
            np.array(variances_list),
            np.array(cluster_ids_list)
        )
        
        # Should give identical results
        np.testing.assert_allclose(
            result1['pooled_effect'], result2['pooled_effect'], rtol=1e-10
        )
        np.testing.assert_allclose(
            result1['robust_se'], result2['robust_se'], rtol=1e-10
        )


@pytest.mark.optional
class TestHierarchicalEffectsModel:
    """Test suite for three-level hierarchical models."""
    
    def test_hierarchical_basic(self):
        """Test basic hierarchical effects model."""
        effect_sizes = [0.3, 0.35, 0.5, 0.55, 0.4, 0.42]
        variances = [0.04, 0.05, 0.06, 0.07, 0.05, 0.06]
        level_1_ids = [1, 2, 3, 4, 5, 6]  # Effect size IDs
        level_2_ids = [1, 1, 2, 2, 3, 3]  # Study IDs
        
        result = hierarchical_effects_model(
            effect_sizes, variances, level_1_ids, level_2_ids
        )
        
        # Currently delegates to RVE
        assert isinstance(result, dict)
        assert 'pooled_effect' in result
        assert 'robust_se' in result
        assert 'method' in result
    
    def test_hierarchical_three_levels(self):
        """Test hierarchical model with clear three-level structure."""
        # Multiple outcomes per study, multiple studies per cluster
        effect_sizes = [0.2, 0.25, 0.4, 0.45, 0.3, 0.35, 0.6, 0.65]
        variances = [0.03, 0.04, 0.05, 0.06, 0.04, 0.05, 0.07, 0.08]
        level_1_ids = [1, 2, 3, 4, 5, 6, 7, 8]  # Outcome IDs
        level_2_ids = [1, 1, 2, 2, 3, 3, 4, 4]  # Study IDs (2 outcomes per study)
        
        result = hierarchical_effects_model(
            effect_sizes, variances, level_1_ids, level_2_ids
        )
        
        assert np.isfinite(result['pooled_effect'])
        assert result['robust_se'] > 0
    
    def test_hierarchical_single_level(self):
        """Test hierarchical model reducing to single level."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [0.04, 0.06, 0.05]
        level_1_ids = [1, 2, 3]
        level_2_ids = [1, 2, 3]  # Each effect is its own study
        
        result = hierarchical_effects_model(
            effect_sizes, variances, level_1_ids, level_2_ids
        )
        
        # Should reduce to standard meta-analysis
        assert np.isfinite(result['pooled_effect'])


@pytest.mark.optional
class TestSensitivityAnalysisRVE:
    """Test suite for RVE sensitivity analysis."""
    
    def test_sensitivity_analysis_basic(self):
        """Test basic RVE sensitivity analysis."""
        effect_sizes = [0.3, 0.35, 0.5, 0.55]
        variances = [0.04, 0.05, 0.06, 0.07]
        cluster_ids = [1, 1, 2, 2]
        
        result = sensitivity_analysis_rve(
            effect_sizes, variances, cluster_ids
        )
        
        # Check structure
        assert isinstance(result, dict)
        assert 'sensitivity_results' in result
        assert 'method' in result
        
        # Should have results for different rho values
        sensitivity_results = result['sensitivity_results']
        assert isinstance(sensitivity_results, dict)
        
        # Default rho values
        expected_rhos = ['rho_0', 'rho_0.2', 'rho_0.5', 'rho_0.8']
        for rho_key in expected_rhos:
            assert rho_key in sensitivity_results
            rho_result = sensitivity_results[rho_key]
            assert 'pooled_effect' in rho_result
            assert 'robust_se' in rho_result
            assert 'p_value' in rho_result
    
    def test_sensitivity_analysis_custom_rhos(self):
        """Test sensitivity analysis with custom rho values."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [0.04, 0.06, 0.05]
        custom_rhos = [0, 0.3, 0.7]
        
        result = sensitivity_analysis_rve(
            effect_sizes, variances,
            rho_values=custom_rhos
        )
        
        sensitivity_results = result['sensitivity_results']
        
        # Should have results for custom rho values
        assert 'rho_0' in sensitivity_results
        assert 'rho_0.3' in sensitivity_results
        assert 'rho_0.7' in sensitivity_results
        assert len(sensitivity_results) == 3
    
    def test_sensitivity_analysis_no_clustering(self):
        """Test sensitivity analysis with no clustering."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [0.04, 0.06, 0.05]
        
        result = sensitivity_analysis_rve(effect_sizes, variances)
        
        # Should work even without clustering
        assert 'sensitivity_results' in result
        sensitivity_results = result['sensitivity_results']
        
        # All rho values should give similar results when no clustering
        pooled_effects = [res['pooled_effect'] for res in sensitivity_results.values()]
        assert np.std(pooled_effects) < 0.1  # Should be very similar


@pytest.mark.optional
class TestRVEEdgeCases:
    """Test edge cases for RVE methods."""
    
    def test_rve_empty_input(self):
        """Test RVE with empty inputs."""
        with pytest.raises((ValueError, IndexError)):
            robust_variance_estimation([], [], [])
    
    def test_rve_mismatched_lengths(self):
        """Test RVE with mismatched input lengths."""
        with pytest.raises((ValueError, IndexError)):
            robust_variance_estimation([0.3, 0.5], [0.04], [1, 2])
    
    def test_rve_zero_variances(self):
        """Test RVE with zero variances."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [0.0, 0.06, 0.05]  # One zero variance
        cluster_ids = [1, 2, 3]
        
        # Should handle gracefully or raise appropriate error
        try:
            result = robust_variance_estimation(effect_sizes, variances, cluster_ids)
            assert np.isfinite(result['pooled_effect'])
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise error
            pass
    
    def test_rve_negative_variances(self):
        """Test RVE with negative variances."""
        effect_sizes = [0.3, 0.5, 0.4]
        variances = [-0.04, 0.06, 0.05]  # One negative variance
        cluster_ids = [1, 2, 3]
        
        # Should raise error
        with pytest.raises(ValueError):
            robust_variance_estimation(effect_sizes, variances, cluster_ids)
    
    def test_rve_extreme_clustering(self):
        """Test RVE with extreme clustering scenarios."""
        # All effects in one cluster
        effect_sizes = [0.3, 0.5, 0.4, 0.6]
        variances = [0.04, 0.06, 0.05, 0.07]
        cluster_ids = [1, 1, 1, 1]
        
        result = robust_variance_estimation(effect_sizes, variances, cluster_ids)
        
        assert result['n_clusters'] == 1
        assert result['df'] == 0
        assert np.isfinite(result['pooled_effect'])
    
    def test_rve_string_cluster_ids(self):
        """Test RVE with string cluster identifiers."""
        effect_sizes = [0.3, 0.5, 0.4, 0.6]
        variances = [0.04, 0.06, 0.05, 0.07]
        cluster_ids = ['A', 'A', 'B', 'B']
        
        result = robust_variance_estimation(effect_sizes, variances, cluster_ids)
        
        assert result['n_clusters'] == 2
        assert np.isfinite(result['pooled_effect'])


@pytest.mark.optional
class TestRVEIntegration:
    """Integration tests for RVE methods."""
    
    def test_rve_vs_standard_when_independent(self, continuous_effects_data):
        """Compare RVE with standard meta-analysis for independent studies."""
        from pymeta.models import fixed_effects
        
        data = continuous_effects_data
        
        # Standard meta-analysis
        fe_result = fixed_effects(data['effect_sizes'], data['variances'])
        
        # RVE with each study as its own cluster
        rve_result = robust_variance_estimation(
            data['effect_sizes'],
            data['variances'],
            small_sample_correction=False  # Closer to standard
        )
        
        # Should be very similar
        np.testing.assert_allclose(
            rve_result['pooled_effect'],
            fe_result['pooled_effect'],
            rtol=1e-8
        )
    
    def test_rve_wider_intervals_with_clustering(self):
        """Test that RVE gives wider CIs with clustering."""
        effect_sizes = [0.3, 0.32, 0.5, 0.52, 0.4, 0.42]
        variances = [0.04, 0.04, 0.06, 0.06, 0.05, 0.05]
        
        # Independent analysis
        rve_independent = robust_variance_estimation(
            effect_sizes, variances,
            cluster_ids=[1, 2, 3, 4, 5, 6]  # All independent
        )
        
        # Clustered analysis
        rve_clustered = robust_variance_estimation(
            effect_sizes, variances,
            cluster_ids=[1, 1, 2, 2, 3, 3]  # 3 clusters
        )
        
        # Clustered should have wider confidence intervals
        independent_width = (rve_independent['ci_upper'] - 
                           rve_independent['ci_lower'])
        clustered_width = (rve_clustered['ci_upper'] - 
                          rve_clustered['ci_lower'])
        
        assert clustered_width >= independent_width
    
    def test_rve_degrees_freedom_calculation(self):
        """Test degrees of freedom calculation in different scenarios."""
        effect_sizes = [0.3, 0.5, 0.4, 0.6, 0.45, 0.55]
        variances = [0.04, 0.06, 0.05, 0.07, 0.055, 0.065]
        
        # 3 clusters
        cluster_ids_3 = [1, 1, 2, 2, 3, 3]
        result_3 = robust_variance_estimation(effect_sizes, variances, cluster_ids_3)
        assert result_3['df'] == 2  # 3 - 1
        
        # 6 clusters (independent)
        cluster_ids_6 = [1, 2, 3, 4, 5, 6]
        result_6 = robust_variance_estimation(effect_sizes, variances, cluster_ids_6)
        assert result_6['df'] == 5  # 6 - 1
    
    def test_rve_numerical_stability(self):
        """Test RVE numerical stability."""
        # Very small variances
        effect_sizes = [0.5, 0.51, 0.49]
        variances = [1e-8, 1e-8, 1e-8]
        cluster_ids = [1, 2, 3]
        
        result = robust_variance_estimation(effect_sizes, variances, cluster_ids)
        
        # Should not produce NaN or inf
        assert np.isfinite(result['pooled_effect'])
        assert np.isfinite(result['robust_se'])
        assert np.isfinite(result['t_value'])
        
        # Very large effect sizes
        effect_sizes = [100, 101, 99]
        variances = [0.1, 0.1, 0.1]
        
        result = robust_variance_estimation(effect_sizes, variances, cluster_ids)
        
        assert np.isfinite(result['pooled_effect'])
        assert np.isfinite(result['robust_se'])