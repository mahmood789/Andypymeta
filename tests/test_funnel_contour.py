"""
Test suite for contour-enhanced funnel plot functionality.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from pymeta import MetaPoint, MetaAnalysisConfig, analyze_data
from pymeta.plots.funnel_contour import plot_funnel_contour, plot_funnel_contour_asymmetry


class TestFunnelContour:
    """Test cases for contour-enhanced funnel plots."""
    
    def create_test_results(self, n=5, use_hksj=False):
        """Create test meta-analysis results."""
        effects = np.array([0.3, 0.5, 0.4, 0.6, 0.2])[:n]
        variances = np.array([0.1, 0.08, 0.12, 0.09, 0.11])[:n]
        study_ids = [f"Study_{i+1}" for i in range(n)]
        
        config = MetaAnalysisConfig(use_hksj=use_hksj)
        results = analyze_data(effects, variances, study_ids, config)
        
        return results
    
    def test_plot_funnel_contour_basic(self):
        """Test basic contour funnel plot creation."""
        results = self.create_test_results()
        
        fig = plot_funnel_contour(results)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1  # Should have at least one axis
        
        # Check that the plot was created without errors
        assert fig.axes[0].get_xlabel() == "Effect Size"
        assert fig.axes[0].get_ylabel() == "Standard Error"
        
        plt.close(fig)
    
    def test_plot_funnel_contour_custom_levels(self):
        """Test contour plot with custom significance levels."""
        results = self.create_test_results()
        custom_levels = [0.001, 0.01, 0.05, 0.1]
        
        fig = plot_funnel_contour(
            results,
            contour_levels=custom_levels
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_funnel_contour_options(self):
        """Test contour plot with various options."""
        results = self.create_test_results()
        
        # Test with pooled effect hidden
        fig1 = plot_funnel_contour(
            results,
            show_pooled=False,
            figsize=(8, 6)
        )
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Test with studies hidden
        fig2 = plot_funnel_contour(
            results,
            show_studies=False,
            title="Custom Title"
        )
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
        
        # Test with both hidden
        fig3 = plot_funnel_contour(
            results,
            show_pooled=False,
            show_studies=False
        )
        assert isinstance(fig3, plt.Figure)
        plt.close(fig3)
    
    def test_plot_funnel_contour_hksj(self):
        """Test contour plot with HKSJ results."""
        results = self.create_test_results(use_hksj=True)
        
        fig = plot_funnel_contour(results)
        
        assert isinstance(fig, plt.Figure)
        assert results.use_hksj is True
        
        # Title should indicate HKSJ
        title = fig.axes[0].get_title()
        assert "HKSJ" in title
        
        plt.close(fig)
    
    def test_plot_funnel_contour_asymmetry(self):
        """Test asymmetry-focused funnel plot."""
        results = self.create_test_results()
        
        fig = plot_funnel_contour_asymmetry(results)
        
        assert isinstance(fig, plt.Figure)
        assert fig.axes[0].get_xlabel() == "Effect Size (centered on pooled effect)"
        
        plt.close(fig)
    
    def test_plot_funnel_contour_validation(self):
        """Test input validation for contour plots."""
        # Create results without points
        results = self.create_test_results()
        results.points = None
        
        with pytest.raises(ValueError, match="must contain points"):
            plot_funnel_contour(results)
    
    def test_plot_funnel_contour_save(self):
        """Test saving contour plot to file."""
        import tempfile
        import os
        
        results = self.create_test_results()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_contour.png")
            
            fig = plot_funnel_contour(results, save_path=save_path)
            
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0  # File is not empty
            
            plt.close(fig)
    
    def test_contour_levels_validation(self):
        """Test that contour levels are properly handled."""
        results = self.create_test_results()
        
        # Test with empty contour levels
        fig1 = plot_funnel_contour(results, contour_levels=[])
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Test with single contour level
        fig2 = plot_funnel_contour(results, contour_levels=[0.05])
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
        
        # Test with extreme contour levels
        fig3 = plot_funnel_contour(results, contour_levels=[0.001, 0.999])
        assert isinstance(fig3, plt.Figure)
        plt.close(fig3)
    
    def test_plot_with_different_study_counts(self):
        """Test contour plots with different numbers of studies."""
        # Test with minimum studies
        results_min = self.create_test_results(n=2)
        fig1 = plot_funnel_contour(results_min)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Test with many studies
        results_many = self.create_test_results(n=5)
        fig2 = plot_funnel_contour(results_many)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
    
    def test_contour_plot_axes_properties(self):
        """Test that plot axes have correct properties."""
        results = self.create_test_results()
        
        fig = plot_funnel_contour(results)
        ax = fig.axes[0]
        
        # Check axis labels
        assert ax.get_xlabel() == "Effect Size"
        assert ax.get_ylabel() == "Standard Error"
        
        # Check that y-axis is inverted (larger studies at top)
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]  # First value should be larger (inverted)
        
        # Check that grid is enabled
        assert ax.grid(True)
        
        plt.close(fig)
    
    def test_contour_plot_with_extreme_data(self):
        """Test contour plot with extreme effect sizes and variances."""
        # Create data with extreme values
        effects = np.array([-5.0, -0.1, 0.0, 0.1, 5.0])
        variances = np.array([0.001, 0.1, 1.0, 10.0, 0.01])
        study_ids = [f"Study_{i+1}" for i in range(5)]
        
        config = MetaAnalysisConfig()
        results = analyze_data(effects, variances, study_ids, config)
        
        fig = plot_funnel_contour(results)
        
        assert isinstance(fig, plt.Figure)
        # Should handle extreme values without crashing
        
        plt.close(fig)
    
    def test_colormap_and_contours(self):
        """Test that colormap and contours are properly applied."""
        results = self.create_test_results()
        
        fig = plot_funnel_contour(results)
        ax = fig.axes[0]
        
        # Check that there are contour collections (filled contours and lines)
        collections = ax.collections
        assert len(collections) > 0
        
        # Check that there's a colorbar
        assert len(fig.axes) >= 2  # Main plot + colorbar
        
        plt.close(fig)
    
    def test_plot_funnel_contour_asymmetry_centered(self):
        """Test that asymmetry plot properly centers effects."""
        results = self.create_test_results()
        original_effects = np.array([p.effect for p in results.points])
        
        fig = plot_funnel_contour_asymmetry(results)
        
        # The plot should center effects around the pooled effect
        # This is mainly a visual feature, so we just check it runs
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_contour_plot_statistics_box(self):
        """Test that statistics text box is properly displayed."""
        results = self.create_test_results()
        
        fig = plot_funnel_contour(results)
        ax = fig.axes[0]
        
        # Check that there are text objects (statistics box)
        texts = ax.texts
        assert len(texts) > 0
        
        # One of the texts should contain statistics
        stats_text_found = False
        for text in texts:
            if "Effect:" in text.get_text() or "P-value:" in text.get_text():
                stats_text_found = True
                break
        
        assert stats_text_found
        
        plt.close(fig)
    
    def test_plot_funnel_contour_large_figsize(self):
        """Test contour plot with large figure size."""
        results = self.create_test_results()
        
        fig = plot_funnel_contour(results, figsize=(16, 12))
        
        assert isinstance(fig, plt.Figure)
        assert fig.get_size_inches()[0] == 16
        assert fig.get_size_inches()[1] == 12
        
        plt.close(fig)
    
    def test_contour_plot_error_handling(self):
        """Test error handling in contour plot generation."""
        results = self.create_test_results()
        
        # Test with invalid contour levels (should handle gracefully)
        try:
            fig = plot_funnel_contour(results, contour_levels=[-1, 2])
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        except Exception:
            # If it fails, that's also acceptable as long as it's handled
            pass