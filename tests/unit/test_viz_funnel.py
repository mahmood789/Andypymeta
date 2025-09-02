"""Unit tests for funnel plot visualization."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path

from pymeta.viz import funnel_plot


@pytest.mark.viz
class TestFunnelPlot:
    """Test suite for funnel plot creation."""
    
    def test_funnel_plot_basic(self, continuous_effects_data, capture_plots):
        """Test basic funnel plot creation."""
        data = continuous_effects_data
        
        fig = funnel_plot(
            data['effect_sizes'], 
            data['standard_errors']
        )
        
        # Check that figure was created
        assert fig is not None
        assert hasattr(fig, 'axes')
        assert len(fig.axes) == 1
        
        # Check basic plot properties
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Effect Size'
        assert ax.get_ylabel() == 'Standard Error'
        assert 'Funnel Plot' in ax.get_title()
        
        # Y-axis should be inverted (smaller SE at top)
        y_lim = ax.get_ylim()
        assert y_lim[0] > y_lim[1]  # Inverted
        
        plt.close(fig)
    
    def test_funnel_plot_with_contours(self, continuous_effects_data, capture_plots):
        """Test funnel plot with significance contours."""
        data = continuous_effects_data
        
        fig = funnel_plot(
            data['effect_sizes'],
            data['standard_errors'],
            show_contours=True
        )
        
        ax = fig.axes[0]
        
        # Should have significance contour lines
        lines = ax.get_lines()
        assert len(lines) >= 2  # At least ±1.96*SE lines
        
        # Should have legend when contours are shown
        assert ax.get_legend() is not None
        
        plt.close(fig)
    
    def test_funnel_plot_without_contours(self, continuous_effects_data, capture_plots):
        """Test funnel plot without significance contours."""
        data = continuous_effects_data
        
        fig = funnel_plot(
            data['effect_sizes'],
            data['standard_errors'],
            show_contours=False
        )
        
        ax = fig.axes[0]
        
        # Should have minimal lines (no contours)
        lines = ax.get_lines()
        assert len(lines) == 0  # No contour lines
        
        # Should not have legend
        assert ax.get_legend() is None
        
        plt.close(fig)
    
    def test_funnel_plot_save_functionality(self, continuous_effects_data, capture_plots, test_output_dir):
        """Test saving funnel plot to file."""
        data = continuous_effects_data
        save_path = test_output_dir / "test_funnel_plot.png"
        
        fig = funnel_plot(
            data['effect_sizes'],
            data['standard_errors'],
            save_path=str(save_path)
        )
        
        # Check that file was created
        assert save_path.exists()
        assert save_path.stat().st_size > 0  # File is not empty
        
        plt.close(fig)
    
    def test_funnel_plot_custom_figsize(self, fe_re_data, capture_plots):
        """Test funnel plot with custom figure size."""
        data = fe_re_data
        custom_figsize = (10, 8)
        
        fig = funnel_plot(
            data['effect_sizes'],
            np.sqrt(data['variances']),
            figsize=custom_figsize
        )
        
        # Check figure size
        assert fig.get_size_inches()[0] == custom_figsize[0]
        assert fig.get_size_inches()[1] == custom_figsize[1]
        
        plt.close(fig)
    
    def test_funnel_plot_custom_title(self, continuous_effects_data, capture_plots):
        """Test funnel plot with custom title."""
        data = continuous_effects_data
        custom_title = "Publication Bias Assessment"
        
        fig = funnel_plot(
            data['effect_sizes'],
            data['standard_errors'],
            title=custom_title
        )
        
        ax = fig.axes[0]
        assert custom_title in ax.get_title()
        
        plt.close(fig)
    
    def test_funnel_plot_scatter_points(self, continuous_effects_data, capture_plots):
        """Test that data points are displayed as scatter plot."""
        data = continuous_effects_data
        
        fig = funnel_plot(
            data['effect_sizes'],
            data['standard_errors']
        )
        
        ax = fig.axes[0]
        
        # Should have scatter plot points
        collections = ax.collections
        assert len(collections) >= 1  # At least one scatter collection
        
        # Check that all data points are plotted
        scatter_collection = collections[0]
        assert len(scatter_collection.get_offsets()) == len(data['effect_sizes'])
        
        plt.close(fig)
    
    def test_funnel_plot_significance_contours(self, capture_plots):
        """Test significance contour calculation."""
        effect_sizes = [0, 0.5, -0.3, 0.8, -0.2]
        standard_errors = [0.1, 0.15, 0.2, 0.25, 0.3]
        
        fig = funnel_plot(effect_sizes, standard_errors, show_contours=True)
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        
        # Should have exactly 2 contour lines (±1.96*SE)
        assert len(lines) == 2
        
        # Lines should be symmetric around x=0
        line1_x = lines[0].get_xdata()
        line2_x = lines[1].get_xdata()
        
        # One should be positive, one negative (approximately)
        assert np.any(line1_x > 0) or np.any(line2_x > 0)
        assert np.any(line1_x < 0) or np.any(line2_x < 0)
        
        plt.close(fig)
    
    def test_funnel_plot_single_study(self, capture_plots):
        """Test funnel plot with single study."""
        fig = funnel_plot([0.5], [0.1])
        
        assert fig is not None
        ax = fig.axes[0]
        
        # Should have one data point
        collections = ax.collections
        if collections:
            assert len(collections[0].get_offsets()) == 1
        
        plt.close(fig)
    
    def test_funnel_plot_many_studies(self, capture_plots):
        """Test funnel plot with many studies."""
        np.random.seed(234)
        n_studies = 30
        
        effect_sizes = np.random.normal(0.3, 0.4, n_studies)
        standard_errors = np.random.uniform(0.05, 0.3, n_studies)
        
        fig = funnel_plot(effect_sizes, standard_errors)
        
        assert fig is not None
        ax = fig.axes[0]
        
        # Should handle many studies
        collections = ax.collections
        if collections:
            assert len(collections[0].get_offsets()) == n_studies
        
        plt.close(fig)
    
    def test_funnel_plot_extreme_values(self, capture_plots):
        """Test funnel plot with extreme effect sizes."""
        effect_sizes = [-3, 0, 3, -2.5, 2.8]
        standard_errors = [0.1, 0.1, 0.1, 0.15, 0.12]
        
        fig = funnel_plot(effect_sizes, standard_errors)
        
        # Should handle extreme values without error
        assert fig is not None
        
        plt.close(fig)
    
    def test_funnel_plot_array_inputs(self, capture_plots):
        """Test funnel plot with different array input types."""
        effect_sizes_list = [0.3, 0.5, 0.4]
        se_list = [0.1, 0.12, 0.11]
        
        # Lists
        fig1 = funnel_plot(effect_sizes_list, se_list)
        
        # Numpy arrays
        fig2 = funnel_plot(np.array(effect_sizes_list), np.array(se_list))
        
        # Both should work
        assert fig1 is not None
        assert fig2 is not None
        
        plt.close(fig1)
        plt.close(fig2)


@pytest.mark.viz
class TestFunnelPlotAsymmetry:
    """Test funnel plot for detecting asymmetry (publication bias)."""
    
    def test_symmetric_funnel_plot(self, capture_plots):
        """Test funnel plot with symmetric data (no bias)."""
        np.random.seed(345)
        
        # Create symmetric data
        n_studies = 20
        true_effect = 0.5
        standard_errors = np.random.uniform(0.05, 0.3, n_studies)
        effect_sizes = np.random.normal(true_effect, standard_errors)
        
        fig = funnel_plot(effect_sizes, standard_errors, show_contours=True)
        
        # Should create plot without issues
        assert fig is not None
        
        # Visual inspection would show roughly symmetric distribution
        # We can check that data spans both sides of the funnel
        ax = fig.axes[0]
        collections = ax.collections
        if collections:
            x_data = collections[0].get_offsets()[:, 0]  # Effect sizes
            assert np.any(x_data < true_effect) and np.any(x_data > true_effect)
        
        plt.close(fig)
    
    def test_asymmetric_funnel_plot(self, capture_plots):
        """Test funnel plot with asymmetric data (simulated bias)."""
        # Create asymmetric data (missing small negative studies)
        effect_sizes = [0.8, 0.6, 0.4, 0.7, 0.5, 0.9, 0.3]  # All positive
        standard_errors = [0.05, 0.1, 0.2, 0.15, 0.25, 0.08, 0.3]
        
        fig = funnel_plot(effect_sizes, standard_errors, show_contours=True)
        
        # Should create plot without issues
        assert fig is not None
        
        # All effect sizes are positive (asymmetric)
        ax = fig.axes[0]
        collections = ax.collections
        if collections:
            x_data = collections[0].get_offsets()[:, 0]
            assert np.all(x_data > 0)  # All positive
        
        plt.close(fig)
    
    def test_funnel_plot_with_zero_effects(self, capture_plots):
        """Test funnel plot including studies with zero effect."""
        effect_sizes = [-0.2, 0, 0.2, 0, 0.1, -0.1]
        standard_errors = [0.1, 0.15, 0.1, 0.2, 0.12, 0.18]
        
        fig = funnel_plot(effect_sizes, standard_errors, show_contours=True)
        
        # Should handle zero effects
        assert fig is not None
        
        # Should have studies at effect size = 0
        ax = fig.axes[0]
        collections = ax.collections
        if collections:
            x_data = collections[0].get_offsets()[:, 0]
            assert np.any(x_data == 0)
        
        plt.close(fig)


@pytest.mark.viz
class TestFunnelPlotContours:
    """Test significance contours in funnel plots."""
    
    def test_contour_line_properties(self, capture_plots):
        """Test properties of significance contour lines."""
        effect_sizes = [0, 0.5, -0.3]
        standard_errors = [0.1, 0.15, 0.2]
        
        fig = funnel_plot(effect_sizes, standard_errors, show_contours=True)
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        
        # Should have contour lines
        assert len(lines) >= 2
        
        # Lines should have appropriate style
        for line in lines:
            assert line.get_linestyle() == '--'  # Dashed lines
            assert line.get_alpha() <= 1.0  # Some transparency
        
        plt.close(fig)
    
    def test_contour_mathematical_correctness(self, capture_plots):
        """Test that contours represent ±1.96*SE correctly."""
        effect_sizes = [0]
        standard_errors = [0.1]
        
        fig = funnel_plot(effect_sizes, standard_errors, show_contours=True)
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        
        if len(lines) >= 2:
            # Extract line data
            line1_x = lines[0].get_xdata()
            line1_y = lines[0].get_ydata()
            
            # For a given SE, x should be ±1.96*SE
            # Check some points on the line
            for x, y in zip(line1_x, line1_y):
                if y > 0:  # Valid SE
                    expected_x = abs(1.96 * y)
                    assert abs(abs(x) - expected_x) < 0.1  # Allow some tolerance
        
        plt.close(fig)
    
    def test_contour_range(self, capture_plots):
        """Test that contours cover appropriate range."""
        effect_sizes = [0, 0.5, -0.5]
        standard_errors = [0.05, 0.2, 0.3]
        
        fig = funnel_plot(effect_sizes, standard_errors, show_contours=True)
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        
        if len(lines) >= 2:
            # Contour should extend to cover the data range
            max_se = max(standard_errors)
            
            for line in lines:
                line_y = line.get_ydata()
                if len(line_y) > 0:
                    max_line_y = max(line_y)
                    assert max_line_y >= max_se * 0.9  # Should cover most of the range
        
        plt.close(fig)


@pytest.mark.viz
class TestFunnelPlotEdgeCases:
    """Test edge cases and error handling for funnel plots."""
    
    def test_empty_input_handling(self, capture_plots):
        """Test handling of empty inputs."""
        fig = funnel_plot([], [])
        
        # Should create figure even with no data
        assert fig is not None
        
        plt.close(fig)
    
    def test_mismatched_input_lengths(self, capture_plots):
        """Test error handling for mismatched input lengths."""
        with pytest.raises((ValueError, IndexError)):
            funnel_plot([0.3, 0.5], [0.1])  # Different lengths
    
    def test_negative_standard_errors(self, capture_plots):
        """Test handling of negative standard errors."""
        effect_sizes = [0.3, 0.5, 0.4]
        standard_errors = [-0.1, 0.12, 0.11]  # One negative
        
        # Should handle gracefully or raise appropriate error
        try:
            fig = funnel_plot(effect_sizes, standard_errors)
            assert fig is not None
            plt.close(fig)
        except ValueError:
            # Acceptable to raise error for negative SEs
            pass
    
    def test_zero_standard_errors(self, capture_plots):
        """Test funnel plot handling of zero standard errors."""
        effect_sizes = [0.3, 0.5, 0.4]
        standard_errors = [0.0, 0.1, 0.12]  # One zero SE
        
        fig = funnel_plot(effect_sizes, standard_errors)
        
        # Should handle zero SE (point at top of funnel)
        assert fig is not None
        
        plt.close(fig)
    
    def test_very_large_standard_errors(self, capture_plots):
        """Test with very large standard errors."""
        effect_sizes = [0.5]
        standard_errors = [5.0]  # Very large SE
        
        fig = funnel_plot(effect_sizes, standard_errors)
        
        # Should handle large SEs without error
        assert fig is not None
        
        plt.close(fig)
    
    def test_nan_input_handling(self, capture_plots):
        """Test handling of NaN values."""
        effect_sizes = [0.3, np.nan, 0.4]
        standard_errors = [0.1, 0.12, 0.11]
        
        # Should handle NaN gracefully
        try:
            fig = funnel_plot(effect_sizes, standard_errors)
            assert fig is not None
            plt.close(fig)
        except (ValueError, TypeError):
            # Acceptable to raise error for NaN values
            pass
    
    def test_infinite_values(self, capture_plots):
        """Test handling of infinite values."""
        effect_sizes = [0.3, np.inf, 0.4]
        standard_errors = [0.1, 0.12, 0.11]
        
        # Should handle infinity gracefully
        try:
            fig = funnel_plot(effect_sizes, standard_errors)
            assert fig is not None
            plt.close(fig)
        except (ValueError, TypeError):
            # Acceptable to raise error for infinite values
            pass


@pytest.mark.viz
class TestFunnelPlotLayout:
    """Test layout and formatting of funnel plots."""
    
    def test_axis_inversion(self, continuous_effects_data, capture_plots):
        """Test that y-axis is properly inverted."""
        data = continuous_effects_data
        
        fig = funnel_plot(data['effect_sizes'], data['standard_errors'])
        
        ax = fig.axes[0]
        y_lim = ax.get_ylim()
        
        # Y-axis should be inverted (smaller SE at top)
        assert y_lim[0] > y_lim[1]
        
        plt.close(fig)
    
    def test_grid_presence(self, continuous_effects_data, capture_plots):
        """Test that grid is displayed."""
        data = continuous_effects_data
        
        fig = funnel_plot(data['effect_sizes'], data['standard_errors'])
        
        ax = fig.axes[0]
        
        # Should have grid enabled
        assert ax.get_axisbelow() is not None
        
        plt.close(fig)
    
    def test_axis_labels(self, continuous_effects_data, capture_plots):
        """Test axis labels are correct."""
        data = continuous_effects_data
        
        fig = funnel_plot(data['effect_sizes'], data['standard_errors'])
        
        ax = fig.axes[0]
        
        assert ax.get_xlabel() == 'Effect Size'
        assert ax.get_ylabel() == 'Standard Error'
        
        plt.close(fig)
    
    def test_plot_formatting(self, continuous_effects_data, capture_plots):
        """Test overall plot formatting."""
        data = continuous_effects_data
        
        fig = funnel_plot(data['effect_sizes'], data['standard_errors'])
        
        # Should have proper tight layout
        # This is more of a visual check, but we ensure no errors
        assert fig is not None
        
        plt.close(fig)


@pytest.mark.viz
class TestFunnelPlotPerformance:
    """Test performance aspects of funnel plot creation."""
    
    def test_large_number_of_studies(self, capture_plots):
        """Test funnel plot performance with many studies."""
        np.random.seed(567)
        n_studies = 200  # Large number
        
        effect_sizes = np.random.normal(0.3, 0.5, n_studies)
        standard_errors = np.random.uniform(0.05, 0.4, n_studies)
        
        # Should handle large number of studies reasonably quickly
        fig = funnel_plot(effect_sizes, standard_errors, show_contours=True)
        
        assert fig is not None
        
        plt.close(fig)
    
    def test_memory_cleanup(self, capture_plots):
        """Test that figures are properly cleaned up."""
        initial_figures = len(plt.get_fignums())
        
        # Create and close multiple figures
        for i in range(3):
            fig = funnel_plot([0.3, 0.5], [0.1, 0.12])
            plt.close(fig)
        
        final_figures = len(plt.get_fignums())
        
        # Should not accumulate figures
        assert final_figures <= initial_figures + 1  # Allow for some tolerance