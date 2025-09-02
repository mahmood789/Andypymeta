"""Tests for meta-analysis models."""

import pytest
import numpy as np
from pymeta.models.fixed_effects import FixedEffects
from pymeta.models.random_effects import RandomEffects
from pymeta.models.glmm_binomial import GLMMBinomial
from pymeta.io.datasets import create_example_data
from pymeta.effects.binary import calculate_from_2x2_tables
from pymeta.errors import ValidationError, ModelError


class TestModels:
    """Test meta-analysis models."""
    
    @pytest.fixture
    def sample_points(self):
        """Create sample data for testing."""
        return create_example_data(n_studies=6, true_effect=0.3, tau2=0.05, seed=42)
    
    def test_fixed_effects_basic(self, sample_points):
        """Test basic fixed effects model."""
        model = FixedEffects(sample_points)
        results = model.fit()
        
        assert hasattr(results, 'pooled_effect')
        assert hasattr(results, 'pooled_variance')
        assert hasattr(results, 'confidence_interval')
        assert results.tau2 == 0.0  # Fixed effects assumes no heterogeneity
        assert results.n_studies == len(sample_points)
        assert np.isfinite(results.pooled_effect)
        assert results.pooled_variance > 0
    
    def test_random_effects_basic(self, sample_points):
        """Test basic random effects model."""
        model = RandomEffects(sample_points)
        results = model.fit()
        
        assert hasattr(results, 'pooled_effect')
        assert hasattr(results, 'pooled_variance')
        assert hasattr(results, 'confidence_interval')
        assert results.tau2 >= 0  # Random effects estimates between-study variance
        assert results.n_studies == len(sample_points)
        assert np.isfinite(results.pooled_effect)
        assert results.pooled_variance > 0
    
    def test_random_effects_different_estimators(self, sample_points):
        """Test random effects with different tau² estimators."""
        estimators = ['DL', 'PM', 'REML']
        
        results_by_estimator = {}
        for estimator in estimators:
            model = RandomEffects(sample_points, tau2_estimator=estimator)
            results = model.fit()
            results_by_estimator[estimator] = results
            
            # Basic checks
            assert np.isfinite(results.pooled_effect)
            assert results.pooled_variance > 0
            assert results.tau2 >= 0
        
        # All should give reasonable results
        effects = [r.pooled_effect for r in results_by_estimator.values()]
        assert max(effects) - min(effects) < 2.0  # Shouldn't vary wildly
    
    def test_glmm_binomial_fallback(self):
        """Test GLMM binomial model with fallback."""
        # Create 2x2 table data
        tables = [
            np.array([[10, 5], [20, 15]]),
            np.array([[8, 12], [18, 22]]),
            np.array([[15, 8], [25, 17]])
        ]
        
        model = GLMMBinomial(tables_2x2=tables)
        results = model.fit()
        
        # Should work even without statsmodels (fallback mode)
        assert hasattr(results, 'pooled_effect')
        assert np.isfinite(results.pooled_effect)
        assert results.n_studies == len(tables)
    
    def test_model_validation_errors(self):
        """Test model validation for invalid input."""
        from pymeta.typing import MetaPoint
        
        # Empty data
        with pytest.raises(ValidationError):
            FixedEffects([])
        
        # Single study
        with pytest.raises(ValidationError):
            FixedEffects([MetaPoint(effect=0.5, variance=0.1)])
        
        # Invalid effect size
        with pytest.raises(ValidationError):
            points = [
                MetaPoint(effect=np.nan, variance=0.1),
                MetaPoint(effect=0.3, variance=0.2)
            ]
            FixedEffects(points)
        
        # Invalid variance
        with pytest.raises(ValidationError):
            points = [
                MetaPoint(effect=0.5, variance=-0.1),
                MetaPoint(effect=0.3, variance=0.2)
            ]
            FixedEffects(points)
    
    def test_model_comparison(self, sample_points):
        """Compare fixed vs random effects models."""
        fe_model = FixedEffects(sample_points)
        fe_results = fe_model.fit()
        
        re_model = RandomEffects(sample_points, tau2_estimator='DL')
        re_results = re_model.fit()
        
        # Fixed effects should have smaller variance (more precision)
        assert fe_results.pooled_variance <= re_results.pooled_variance
        
        # Effects should be reasonably close
        assert abs(fe_results.pooled_effect - re_results.pooled_effect) < 1.0
        
        # Random effects should have tau² >= 0
        assert re_results.tau2 >= 0
        assert fe_results.tau2 == 0
    
    def test_heterogeneity_statistics(self, sample_points):
        """Test heterogeneity statistic calculations."""
        model = RandomEffects(sample_points)
        results = model.fit()
        
        # Check heterogeneity statistics
        assert 0 <= results.i_squared <= 100
        assert results.q_statistic >= 0
        assert 0 <= results.q_p_value <= 1
        
        # Q statistic should have appropriate degrees of freedom
        expected_df = len(sample_points) - 1
        assert results.heterogeneity_test['degrees_of_freedom'] == expected_df
    
    def test_confidence_intervals(self, sample_points):
        """Test confidence interval calculations."""
        model = FixedEffects(sample_points)
        results = model.fit()
        
        # CI should be symmetric around point estimate
        ci_lower, ci_upper = results.confidence_interval
        point_estimate = results.pooled_effect
        
        # Check symmetry (approximately)
        lower_dist = point_estimate - ci_lower
        upper_dist = ci_upper - point_estimate
        assert abs(lower_dist - upper_dist) < 1e-10
        
        # CI should contain point estimate
        assert ci_lower <= point_estimate <= ci_upper
        
        # CI width should be reasonable
        ci_width = ci_upper - ci_lower
        assert ci_width > 0
        assert ci_width < 10  # Shouldn't be extremely wide
    
    def test_influence_analysis(self, sample_points):
        """Test leave-one-out influence analysis."""
        if len(sample_points) < 3:
            pytest.skip("Need at least 3 studies for influence analysis")
        
        model = RandomEffects(sample_points)
        influence_results = model.influence_analysis()
        
        assert 'original_result' in influence_results
        assert 'influence_results' in influence_results
        assert len(influence_results['influence_results']) == len(sample_points)
        
        # Each influence result should have required fields
        for inf_result in influence_results['influence_results']:
            assert 'study_index' in inf_result
            assert 'effect_change' in inf_result
            assert 'tau2_change' in inf_result
            assert np.isfinite(inf_result['effect_change'])
    
    def test_cumulative_analysis(self, sample_points):
        """Test cumulative meta-analysis."""
        if len(sample_points) < 3:
            pytest.skip("Need at least 3 studies for cumulative analysis")
        
        model = RandomEffects(sample_points)
        cumulative_results = model.cumulative_analysis()
        
        # Should have one result for each subset starting from 2 studies
        expected_results = len(sample_points) - 1
        assert len(cumulative_results) == expected_results
        
        # Each result should be valid
        for result in cumulative_results:
            assert np.isfinite(result.pooled_effect)
            assert result.pooled_variance > 0
            assert result.n_studies >= 2
    
    def test_model_registry_integration(self):
        """Test that models are properly registered."""
        from pymeta.registries import get_model, list_models
        
        model_names = list_models()
        assert 'fixed_effects' in model_names
        assert 'random_effects' in model_names
        assert 'glmm_binomial' in model_names
        
        # Test retrieval
        fe_class = get_model('fixed_effects')
        assert fe_class == FixedEffects
        
        re_class = get_model('random_effects')
        assert re_class == RandomEffects
    
    def test_prediction_intervals(self, sample_points):
        """Test prediction interval calculations for random effects."""
        model = RandomEffects(sample_points)
        results = model.fit()
        
        if results.tau2 > 0:
            pred_interval = model.prediction_interval()
            
            # Prediction interval should be wider than confidence interval
            ci_width = results.confidence_interval[1] - results.confidence_interval[0]
            pred_width = pred_interval[1] - pred_interval[0]
            assert pred_width >= ci_width
            
            # Should contain the pooled effect
            assert pred_interval[0] <= results.pooled_effect <= pred_interval[1]


if __name__ == '__main__':
    pytest.main([__file__])