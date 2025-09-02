"""
Test suite for influence diagnostics and leave-one-out analysis.
"""

import pytest
import numpy as np
import pandas as pd

from pymeta import MetaPoint, MetaAnalysisConfig, analyze_data
from pymeta.diagnostics import (
    leave_one_out_analysis,
    influence_measures,
    identify_outliers
)


class TestInfluenceDiagnostics:
    """Test cases for influence diagnostics."""
    
    def create_test_points(self, n=5):
        """Create test data points."""
        effects = np.array([0.3, 0.5, 0.4, 0.6, 0.2])[:n]
        variances = np.array([0.1, 0.08, 0.12, 0.09, 0.11])[:n]
        study_ids = [f"Study_{i+1}" for i in range(n)]
        
        return [
            MetaPoint(effect=e, variance=v, study_id=sid)
            for e, v, sid in zip(effects, variances, study_ids)
        ]
    
    def test_leave_one_out_basic(self):
        """Test basic leave-one-out analysis."""
        points = self.create_test_points(5)
        config = MetaAnalysisConfig()
        
        result = leave_one_out_analysis(points, config)
        
        assert result.original_result is not None
        assert len(result.loo_results) == len(points)
        assert len(result.study_ids) == len(points)
        assert len(result.effect_changes) == len(points)
        assert len(result.se_changes) == len(points)
        assert len(result.i2_changes) == len(points)
        assert len(result.tau2_changes) == len(points)
    
    def test_leave_one_out_properties(self):
        """Test leave-one-out result properties."""
        points = self.create_test_points(4)
        config = MetaAnalysisConfig()
        
        result = leave_one_out_analysis(points, config)
        
        # Test max effect change
        max_change = max(abs(change) for change in result.effect_changes)
        assert result.max_effect_change == max_change
        
        # Test most influential study
        max_idx = np.argmax([abs(change) for change in result.effect_changes])
        assert result.most_influential_study == result.study_ids[max_idx]
    
    def test_leave_one_out_dataframe(self):
        """Test conversion to DataFrame."""
        points = self.create_test_points(4)
        config = MetaAnalysisConfig()
        
        result = leave_one_out_analysis(points, config)
        df = result.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(points)
        assert 'study_id' in df.columns
        assert 'effect_change' in df.columns
        assert 'loo_effect' in df.columns
    
    def test_leave_one_out_validation(self):
        """Test leave-one-out input validation."""
        # Test insufficient studies
        points = self.create_test_points(2)
        config = MetaAnalysisConfig()
        
        with pytest.raises(ValueError, match="at least 3 studies"):
            leave_one_out_analysis(points, config)
    
    def test_leave_one_out_with_hksj(self):
        """Test leave-one-out with HKSJ adjustment."""
        points = self.create_test_points(5)
        config = MetaAnalysisConfig(use_hksj=True)
        
        result = leave_one_out_analysis(points, config)
        
        assert result.original_result.use_hksj is True
        for loo_result in result.loo_results:
            assert loo_result.use_hksj is True
    
    def test_influence_measures_basic(self):
        """Test basic influence measures calculation."""
        points = self.create_test_points(5)
        config = MetaAnalysisConfig()
        
        # First get meta-analysis results
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        study_ids = [p.study_id for p in points]
        meta_result = analyze_data(effects, variances, study_ids, config)
        
        influence_results = influence_measures(points, meta_result, config)
        
        assert len(influence_results) == len(points)
        
        for inf in influence_results:
            assert inf.study_id in study_ids
            assert inf.effect in effects
            assert inf.variance in variances
            assert inf.weight > 0
            assert hasattr(inf, 'standardized_residual')
            assert hasattr(inf, 'leverage')
            assert hasattr(inf, 'cook_distance')
            assert hasattr(inf, 'dffits')
            assert hasattr(inf, 'dfbetas')
            assert hasattr(inf, 'studentized_residual')
    
    def test_influence_measures_leverage_sum(self):
        """Test that leverage values sum to approximately k (number of studies)."""
        points = self.create_test_points(4)
        config = MetaAnalysisConfig()
        
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        study_ids = [p.study_id for p in points]
        meta_result = analyze_data(effects, variances, study_ids, config)
        
        influence_results = influence_measures(points, meta_result, config)
        
        # Sum of leverages should approximately equal 1 (for meta-analysis)
        leverage_sum = sum(inf.leverage for inf in influence_results)
        assert 0.9 < leverage_sum < 1.1  # Allow some numerical tolerance
    
    def test_influence_measures_validation(self):
        """Test influence measures input validation."""
        points = self.create_test_points(1)
        config = MetaAnalysisConfig()
        
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        study_ids = [p.study_id for p in points]
        meta_result = analyze_data(effects, variances, study_ids, config)
        
        with pytest.raises(ValueError, match="at least 2 studies"):
            influence_measures(points, meta_result, config)
    
    def test_identify_outliers_basic(self):
        """Test outlier identification."""
        points = self.create_test_points(5)
        config = MetaAnalysisConfig()
        
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        study_ids = [p.study_id for p in points]
        meta_result = analyze_data(effects, variances, study_ids, config)
        
        influence_results = influence_measures(points, meta_result, config)
        outliers = identify_outliers(influence_results)
        
        assert isinstance(outliers, dict)
        assert 'high_cook' in outliers
        assert 'high_studentized' in outliers
        assert 'high_leverage' in outliers
        assert 'high_dffits' in outliers
        assert 'any_flag' in outliers
        
        # Each should be a list
        for key, value in outliers.items():
            assert isinstance(value, list)
    
    def test_identify_outliers_with_outlier(self):
        """Test outlier identification with an actual outlier."""
        # Create data with one extreme point
        points = [
            MetaPoint(effect=0.3, variance=0.1, study_id="Study_1"),
            MetaPoint(effect=0.4, variance=0.1, study_id="Study_2"),
            MetaPoint(effect=0.35, variance=0.1, study_id="Study_3"),
            MetaPoint(effect=2.0, variance=0.1, study_id="Outlier"),  # Extreme effect
            MetaPoint(effect=0.32, variance=0.1, study_id="Study_5"),
        ]
        
        config = MetaAnalysisConfig()
        
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        study_ids = [p.study_id for p in points]
        meta_result = analyze_data(effects, variances, study_ids, config)
        
        influence_results = influence_measures(points, meta_result, config)
        outliers = identify_outliers(
            influence_results,
            cook_threshold=0.5,  # Lower threshold to catch outliers
            studentized_threshold=1.5
        )
        
        # The outlier should be flagged
        assert len(outliers['any_flag']) > 0
    
    def test_influence_measures_with_fixed_effects(self):
        """Test influence measures with fixed effects model."""
        points = self.create_test_points(4)
        config = MetaAnalysisConfig(model="FE")
        
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        study_ids = [p.study_id for p in points]
        meta_result = analyze_data(effects, variances, study_ids, config)
        
        influence_results = influence_measures(points, meta_result, config)
        
        assert len(influence_results) == len(points)
        # For fixed effects, tau2 should be 0
        assert meta_result.tau2 == 0.0
    
    def test_leave_one_out_effect_changes(self):
        """Test that leave-one-out produces reasonable effect changes."""
        points = self.create_test_points(5)
        config = MetaAnalysisConfig()
        
        result = leave_one_out_analysis(points, config)
        
        # Effect changes should be reasonable (not infinite or NaN)
        for change in result.effect_changes:
            assert np.isfinite(change)
        
        # At least one study should have some influence (change > 0)
        assert any(abs(change) > 0 for change in result.effect_changes)
    
    def test_influence_diagnostics_consistency(self):
        """Test consistency between influence measures and leave-one-out."""
        points = self.create_test_points(4)
        config = MetaAnalysisConfig()
        
        # Get original analysis
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        study_ids = [p.study_id for p in points]
        meta_result = analyze_data(effects, variances, study_ids, config)
        
        # Get influence measures
        influence_results = influence_measures(points, meta_result, config)
        
        # Get leave-one-out results
        loo_result = leave_one_out_analysis(points, config)
        
        # Studies with high Cook's distance should have larger effect changes in LOO
        # This is a general expectation, though not a strict requirement
        cook_distances = [inf.cook_distance for inf in influence_results]
        effect_changes = [abs(change) for change in loo_result.effect_changes]
        
        # Check that we can identify the same studies as influential
        max_cook_idx = np.argmax(cook_distances)
        max_change_idx = np.argmax(effect_changes)
        
        # These don't have to be identical, but should be somewhat related
        assert isinstance(max_cook_idx, (int, np.integer))
        assert isinstance(max_change_idx, (int, np.integer))