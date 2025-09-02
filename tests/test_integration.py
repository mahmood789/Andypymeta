"""
Integration tests for PyMeta package functionality.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from pymeta import (
    MetaAnalysisConfig,
    MetaPoint,
    analyze_data,
    analyze_csv
)
from pymeta.plots import plot_forest, plot_funnel, plot_funnel_contour
from pymeta.diagnostics import leave_one_out_analysis, influence_measures


class TestPyMetaIntegration:
    """Integration tests for complete PyMeta workflows."""
    
    def create_test_csv(self):
        """Create a test CSV file."""
        data = pd.DataFrame({
            'study': [f'Study_{i+1}' for i in range(8)],
            'effect': [0.25, 0.31, 0.18, 0.42, 0.28, 0.15, 0.35, 0.22],
            'variance': [0.04, 0.03, 0.05, 0.02, 0.04, 0.06, 0.03, 0.05]
        })
        
        fd, filepath = tempfile.mkstemp(suffix='.csv')
        try:
            data.to_csv(filepath, index=False)
            return filepath
        finally:
            os.close(fd)
    
    def test_complete_workflow_standard(self):
        """Test complete workflow with standard random effects."""
        csv_file = self.create_test_csv()
        
        try:
            # Configuration
            config = MetaAnalysisConfig(
                model="RE",
                tau2_method="REML",
                use_hksj=False,
                alpha=0.05
            )
            
            # Analysis
            results = analyze_csv(csv_file, config=config)
            
            # Verify results
            assert results is not None
            assert results.effect is not None
            assert results.se > 0
            assert results.ci_lower < results.ci_upper
            assert 0 <= results.p_value <= 1
            assert results.tau2 >= 0
            assert 0 <= results.i2 <= 100
            assert results.method == "Random Effects (REML)"
            assert results.use_hksj is False
            
            # Generate plots
            forest_fig = plot_forest(results)
            assert forest_fig is not None
            forest_fig.savefig(tempfile.mktemp(suffix='.png'))
            
            funnel_fig = plot_funnel(results)
            assert funnel_fig is not None
            funnel_fig.savefig(tempfile.mktemp(suffix='.png'))
            
            contour_fig = plot_funnel_contour(results)
            assert contour_fig is not None
            contour_fig.savefig(tempfile.mktemp(suffix='.png'))
            
            # Diagnostics
            if len(results.points) >= 3:
                loo_results = leave_one_out_analysis(results.points, config)
                assert loo_results is not None
                assert len(loo_results.loo_results) == len(results.points)
                
                influence_results = influence_measures(results.points, results, config)
                assert len(influence_results) == len(results.points)
            
        finally:
            os.unlink(csv_file)
    
    def test_complete_workflow_hksj(self):
        """Test complete workflow with HKSJ adjustment."""
        csv_file = self.create_test_csv()
        
        try:
            # Configuration with HKSJ
            config = MetaAnalysisConfig(
                model="RE",
                tau2_method="REML",
                use_hksj=True,
                alpha=0.05
            )
            
            # Analysis
            results = analyze_csv(csv_file, config=config)
            
            # Verify HKSJ-specific results
            assert results.use_hksj is True
            assert "HKSJ" in results.method
            assert results.df is not None
            assert results.df > 0
            
            # HKSJ should typically produce wider CIs
            assert results.ci_width > 0
            
            # Generate plots with HKSJ
            forest_fig = plot_forest(results)
            assert forest_fig is not None
            
            # Plot title should indicate HKSJ
            title = forest_fig.axes[0].get_title()
            assert "HKSJ" in title
            
            # Diagnostics with HKSJ
            if len(results.points) >= 3:
                loo_results = leave_one_out_analysis(results.points, config)
                assert all(res.use_hksj for res in loo_results.loo_results)
            
        finally:
            os.unlink(csv_file)
    
    def test_fixed_effects_workflow(self):
        """Test complete workflow with fixed effects model."""
        csv_file = self.create_test_csv()
        
        try:
            config = MetaAnalysisConfig(model="FE")
            results = analyze_csv(csv_file, config=config)
            
            # Fixed effects specific checks
            assert results.tau2 == 0.0
            assert results.method == "Fixed Effects"
            assert results.use_hksj is False  # HKSJ not applicable to FE
            
            # Should still be able to generate plots
            forest_fig = plot_forest(results)
            assert forest_fig is not None
            
        finally:
            os.unlink(csv_file)
    
    def test_different_tau2_methods(self):
        """Test different tau² estimation methods."""
        csv_file = self.create_test_csv()
        
        try:
            methods = ["DL", "REML", "PM", "ML"]
            results_by_method = {}
            
            for method in methods:
                config = MetaAnalysisConfig(tau2_method=method)
                results = analyze_csv(csv_file, config=config)
                results_by_method[method] = results
                
                assert method in results.method
                assert results.tau2 >= 0
            
            # Different methods should potentially give different results
            tau2_values = [res.tau2 for res in results_by_method.values()]
            # Not all methods will give identical results
            
        finally:
            os.unlink(csv_file)
    
    def test_data_validation_workflow(self):
        """Test workflow with data validation scenarios."""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'study': ['Study_1', 'Study_2'],
            'effect': [0.3, 0.5],
            'variance': [0.1, 0.08]
        })
        
        fd, minimal_file = tempfile.mkstemp(suffix='.csv')
        try:
            minimal_data.to_csv(minimal_file, index=False)
            
            config = MetaAnalysisConfig()
            results = analyze_csv(minimal_file, config=config)
            
            assert results is not None
            assert len(results.points) == 2
            
        finally:
            os.close(fd)
            os.unlink(minimal_file)
        
        # Test with missing columns
        bad_data = pd.DataFrame({
            'wrong_effect': [0.3, 0.5],
            'wrong_variance': [0.1, 0.08],
            'study': ['S1', 'S2']
        })
        
        fd, bad_file = tempfile.mkstemp(suffix='.csv')
        try:
            bad_data.to_csv(bad_file, index=False)
            
            with pytest.raises(ValueError):
                analyze_csv(bad_file, config=config)
                
        finally:
            os.close(fd)
            os.unlink(bad_file)
    
    def test_array_analysis_workflow(self):
        """Test workflow using array-based analysis."""
        effects = np.array([0.25, 0.31, 0.18, 0.42, 0.28])
        variances = np.array([0.04, 0.03, 0.05, 0.02, 0.04])
        study_ids = [f'Study_{i+1}' for i in range(5)]
        
        # Standard analysis
        config = MetaAnalysisConfig()
        results = analyze_data(effects, variances, study_ids, config)
        
        assert results is not None
        assert len(results.points) == 5
        
        # HKSJ analysis
        config_hksj = MetaAnalysisConfig(use_hksj=True)
        results_hksj = analyze_data(effects, variances, study_ids, config_hksj)
        
        assert results_hksj.use_hksj is True
        
        # Compare results (HKSJ might have wider CI)
        assert results.effect == results_hksj.effect  # Point estimate same
        # Standard errors might differ due to HKSJ adjustment
    
    def test_plotting_integration(self):
        """Test integration of all plotting functions."""
        effects = np.array([0.2, 0.4, 0.3, 0.5, 0.1])
        variances = np.array([0.05, 0.03, 0.04, 0.02, 0.06])
        study_ids = [f'Study_{i+1}' for i in range(5)]
        
        config = MetaAnalysisConfig(use_hksj=True)
        results = analyze_data(effects, variances, study_ids, config)
        
        # Test all plot types work together
        plots = {
            'forest': plot_forest(results),
            'funnel': plot_funnel(results),
            'contour': plot_funnel_contour(results)
        }
        
        for plot_name, fig in plots.items():
            assert fig is not None
            # Save to temporary file to ensure complete rendering
            fig.savefig(tempfile.mktemp(suffix=f'_{plot_name}.png'))
    
    def test_diagnostics_integration(self):
        """Test integration of diagnostic functions."""
        effects = np.array([0.25, 0.35, 0.15, 0.45, 0.30, 0.20])
        variances = np.array([0.04, 0.03, 0.05, 0.02, 0.04, 0.05])
        study_ids = [f'Study_{i+1}' for i in range(6)]
        
        config = MetaAnalysisConfig()
        results = analyze_data(effects, variances, study_ids, config)
        points = results.points
        
        # Leave-one-out analysis
        loo_results = leave_one_out_analysis(points, config)
        assert len(loo_results.loo_results) == len(points)
        
        # Influence measures
        influence_results = influence_measures(points, results, config)
        assert len(influence_results) == len(points)
        
        # Convert LOO to DataFrame
        loo_df = loo_results.to_dataframe()
        assert isinstance(loo_df, pd.DataFrame)
        assert len(loo_df) == len(points)
        
        # Check that most influential study makes sense
        most_influential = loo_results.most_influential_study
        assert most_influential in study_ids
    
    def test_configuration_edge_cases(self):
        """Test configuration with edge cases."""
        effects = np.array([0.3, 0.5])
        variances = np.array([0.1, 0.08])
        study_ids = ['S1', 'S2']
        
        # Test extreme alpha values
        config_low_alpha = MetaAnalysisConfig(alpha=0.001)
        results_low = analyze_data(effects, variances, study_ids, config_low_alpha)
        
        config_high_alpha = MetaAnalysisConfig(alpha=0.1)
        results_high = analyze_data(effects, variances, study_ids, config_high_alpha)
        
        # Lower alpha should give wider CI
        assert results_low.ci_width > results_high.ci_width
        
        # Test HKSJ with minimal studies
        config_hksj = MetaAnalysisConfig(use_hksj=True)
        results_hksj = analyze_data(effects, variances, study_ids, config_hksj)
        
        assert results_hksj.use_hksj is True
        assert results_hksj.df == 1  # n-1 = 2-1 = 1
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create test data
        csv_file = self.create_test_csv()
        
        try:
            # Step 1: Load and analyze with different configurations
            configs = [
                MetaAnalysisConfig(model="RE", use_hksj=False),
                MetaAnalysisConfig(model="RE", use_hksj=True),
                MetaAnalysisConfig(model="FE")
            ]
            
            all_results = []
            for config in configs:
                results = analyze_csv(csv_file, config=config)
                all_results.append(results)
                
                # Step 2: Generate all plots for each analysis
                plot_forest(results)
                plot_funnel(results)
                plot_funnel_contour(results)
                
                # Step 3: Run diagnostics (if enough studies)
                if len(results.points) >= 3:
                    loo_results = leave_one_out_analysis(results.points, config)
                    influence_results = influence_measures(results.points, results, config)
                    
                    # Step 4: Export diagnostic results
                    loo_df = loo_results.to_dataframe()
                    assert isinstance(loo_df, pd.DataFrame)
            
            # Step 5: Compare results across methods
            assert len(all_results) == 3
            
            # Fixed effects should have tau2 = 0
            fe_result = next(r for r in all_results if not r.use_hksj and r.tau2 == 0)
            assert fe_result.method == "Fixed Effects"
            
            # HKSJ result should have df specified
            hksj_result = next(r for r in all_results if r.use_hksj)
            assert hksj_result.df is not None
            
        finally:
            os.unlink(csv_file)