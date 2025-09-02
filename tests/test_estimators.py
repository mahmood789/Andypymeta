"""Tests for tau² estimators."""

import pytest
import numpy as np
from pymeta.estimators.tau2_dl import tau2_dersimonian_laird
from pymeta.estimators.tau2_pm import tau2_paule_mandel
from pymeta.estimators.tau2_reml import tau2_reml
from pymeta.io.datasets import create_example_data
from pymeta.errors import EstimationError, ConvergenceError


class TestTau2Estimators:
    """Test tau² estimators."""
    
    @pytest.fixture
    def sample_points(self):
        """Create sample data for testing."""
        return create_example_data(n_studies=8, true_effect=0.5, tau2=0.1, seed=42)
    
    def test_dersimonian_laird_basic(self, sample_points):
        """Test basic DerSimonian-Laird estimation."""
        tau2 = tau2_dersimonian_laird(sample_points)
        
        assert tau2 >= 0  # Tau² should be non-negative
        assert np.isfinite(tau2)
    
    def test_dersimonian_laird_no_heterogeneity(self):
        """Test DL with no heterogeneity (identical studies)."""
        from pymeta.typing import MetaPoint
        
        # Create identical studies (no heterogeneity)
        points = [
            MetaPoint(effect=0.5, variance=0.1, label=f"Study {i}")
            for i in range(5)
        ]
        
        tau2 = tau2_dersimonian_laird(points)
        assert tau2 == 0.0  # Should be zero with no heterogeneity
    
    def test_dersimonian_laird_insufficient_studies(self):
        """Test DL with insufficient studies."""
        from pymeta.typing import MetaPoint
        
        points = [MetaPoint(effect=0.5, variance=0.1)]
        tau2 = tau2_dersimonian_laird(points)
        assert tau2 == 0.0  # Should return 0 with <2 studies
    
    def test_paule_mandel_basic(self, sample_points):
        """Test basic Paule-Mandel estimation."""
        tau2 = tau2_paule_mandel(sample_points)
        
        assert tau2 >= 0
        assert np.isfinite(tau2)
    
    def test_paule_mandel_convergence(self, sample_points):
        """Test Paule-Mandel convergence."""
        # Should converge for reasonable data
        tau2 = tau2_paule_mandel(sample_points, max_iterations=100, tolerance=1e-6)
        
        assert tau2 >= 0
        assert np.isfinite(tau2)
    
    def test_reml_basic(self, sample_points):
        """Test basic REML estimation."""
        tau2 = tau2_reml(sample_points)
        
        assert tau2 >= 0
        assert np.isfinite(tau2)
    
    def test_estimator_comparison(self, sample_points):
        """Compare different estimators on same data."""
        tau2_dl = tau2_dersimonian_laird(sample_points)
        tau2_pm = tau2_paule_mandel(sample_points)
        tau2_reml = tau2_reml(sample_points)
        
        # All should be non-negative and finite
        estimators = [tau2_dl, tau2_pm, tau2_reml]
        for tau2 in estimators:
            assert tau2 >= 0
            assert np.isfinite(tau2)
        
        # REML typically gives higher estimates than DL
        # (this is a general tendency, not a strict rule)
        # So we just check they're in reasonable range
        assert all(0 <= tau2 <= 10 for tau2 in estimators)
    
    def test_high_heterogeneity_data(self):
        """Test estimators with high heterogeneity."""
        # Create data with high between-study variance
        points = create_example_data(n_studies=6, true_effect=0.5, tau2=1.0, seed=123)
        
        tau2_dl = tau2_dersimonian_laird(points)
        tau2_pm = tau2_paule_mandel(points)
        tau2_reml = tau2_reml(points)
        
        # All should detect substantial heterogeneity
        for tau2 in [tau2_dl, tau2_pm, tau2_reml]:
            assert tau2 > 0.1  # Should detect substantial heterogeneity
            assert np.isfinite(tau2)
    
    def test_estimator_error_handling(self):
        """Test error handling in estimators."""
        from pymeta.typing import MetaPoint
        
        # Create problematic data (zero variance)
        points = [
            MetaPoint(effect=0.5, variance=0.0, label="Study 1"),
            MetaPoint(effect=0.3, variance=0.0, label="Study 2")
        ]
        
        # Should handle gracefully or raise appropriate errors
        with pytest.raises((EstimationError, ZeroDivisionError, ValueError)):
            tau2_dersimonian_laird(points)
    
    def test_registry_integration(self):
        """Test that estimators are properly registered."""
        from pymeta.registries import get_estimator, list_estimators
        
        estimator_names = list_estimators()
        assert 'DL' in estimator_names
        assert 'PM' in estimator_names
        assert 'REML' in estimator_names
        
        # Test retrieval
        dl_func = get_estimator('DL')
        assert callable(dl_func)
        
        pm_func = get_estimator('PM')
        assert callable(pm_func)
        
        reml_func = get_estimator('REML')
        assert callable(reml_func)
    
    def test_reproducibility(self):
        """Test that estimators give reproducible results."""
        # Create same data twice
        points1 = create_example_data(n_studies=6, seed=999)
        points2 = create_example_data(n_studies=6, seed=999)
        
        # Should give identical results
        tau2_dl_1 = tau2_dersimonian_laird(points1)
        tau2_dl_2 = tau2_dersimonian_laird(points2)
        assert abs(tau2_dl_1 - tau2_dl_2) < 1e-10
        
        tau2_pm_1 = tau2_paule_mandel(points1)
        tau2_pm_2 = tau2_paule_mandel(points2)
        assert abs(tau2_pm_1 - tau2_pm_2) < 1e-6  # Slightly larger tolerance for iterative method
        
        tau2_reml_1 = tau2_reml(points1)
        tau2_reml_2 = tau2_reml(points2)
        assert abs(tau2_reml_1 - tau2_reml_2) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__])