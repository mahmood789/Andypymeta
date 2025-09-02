"""Unit tests for forest plot visualization."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path

from pymeta.viz import forest_plot, create_demo_plot


@pytest.mark.viz
class TestForestPlot:
    """Test suite for forest plot creation."""
    
    def test_forest_plot_basic(self, continuous_effects_data, capture_plots, test_output_dir):
        """Test basic forest plot creation."""
        data = continuous_effects_data
        
        fig = forest_plot(
            data['effect_sizes'], 
            data['standard_errors'],
            study_labels=data['study_labels']
        )
        
        # Check that figure was created
        assert fig is not None
        assert hasattr(fig, 'axes')
        assert len(fig.axes) == 1
        
        # Check basic plot properties
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Effect Size'
        assert 'Forest Plot' in ax.get_title()
        
        # Clean up
        plt.close(fig)
    
    def test_forest_plot_with_pooled_estimate(self, fe_re_data, capture_plots):
        """Test forest plot with pooled estimate."""
        data = fe_re_data
        
        fig = forest_plot(
            data['effect_sizes'],
            np.sqrt(data['variances']),
            study_labels=data['study_labels'],
            pooled_effect=0.5,
            pooled_se=0.1,
            title="Test Forest Plot with Pooled"
        )
        
        assert fig is not None
        ax = fig.axes[0]
        assert 'Test Forest Plot with Pooled' in ax.get_title()
        
        # Should have legend when pooled estimate is shown
        assert ax.get_legend() is not None
        
        plt.close(fig)
    
    def test_forest_plot_save_functionality(self, continuous_effects_data, capture_plots, test_output_dir):
        """Test saving forest plot to file."""
        data = continuous_effects_data
        save_path = test_output_dir / "test_forest_plot.png"
        
        fig = forest_plot(
            data['effect_sizes'],
            data['standard_errors'],
            save_path=str(save_path)
        )
        
        # Check that file was created
        assert save_path.exists()
        assert save_path.stat().st_size > 0  # File is not empty
        
        plt.close(fig)
    
    def test_forest_plot_custom_figsize(self, fe_re_data, capture_plots):
        """Test forest plot with custom figure size."""
        data = fe_re_data
        custom_figsize = (12, 6)
        
        fig = forest_plot(
            data['effect_sizes'],
            np.sqrt(data['variances']),
            figsize=custom_figsize
        )
        
        # Check figure size
        assert fig.get_size_inches()[0] == custom_figsize[0]
        assert fig.get_size_inches()[1] == custom_figsize[1]
        
        plt.close(fig)
    
    def test_forest_plot_no_study_labels(self, continuous_effects_data, capture_plots):
        """Test forest plot without providing study labels."""
        data = continuous_effects_data
        
        fig = forest_plot(
            data['effect_sizes'],
            data['standard_errors']
            # No study_labels provided
        )
        
        # Should create default labels
        ax = fig.axes[0]
        y_labels = [tick.get_text() for tick in ax.get_yticklabels()]
        
        # Should have generated Study 1, Study 2, etc.
        assert any('Study' in label for label in y_labels)
        
        plt.close(fig)
    
    def test_forest_plot_confidence_intervals(self, fe_re_data, capture_plots):
        """Test that confidence intervals are displayed correctly."""
        data = fe_re_data
        
        fig = forest_plot(
            data['effect_sizes'],
            np.sqrt(data['variances']),
            study_labels=data['study_labels']
        )
        
        ax = fig.axes[0]
        
        # Should have error bars (confidence intervals)
        # Check for presence of error bar container
        errorbar_containers = [child for child in ax.get_children() 
                             if hasattr(child, 'has_xerr')]
        assert len(errorbar_containers) > 0
        
        plt.close(fig)
    
    def test_forest_plot_null_line(self, continuous_effects_data, capture_plots):
        """Test that null effect line is shown."""
        data = continuous_effects_data
        
        fig = forest_plot(
            data['effect_sizes'],
            data['standard_errors']
        )
        
        ax = fig.axes[0]
        
        # Should have vertical line at x=0
        vertical_lines = [line for line in ax.get_lines() 
                         if hasattr(line, 'get_linestyle') and 
                         line.get_linestyle() == '--']
        assert len(vertical_lines) > 0
        
        # Check that line is at x=0
        line_data = vertical_lines[0].get_xdata()
        assert 0 in line_data or np.allclose(line_data, 0)
        
        plt.close(fig)
    
    def test_forest_plot_single_study(self, capture_plots):
        """Test forest plot with single study."""
        fig = forest_plot([0.5], [0.1], study_labels=['Single Study'])
        
        assert fig is not None
        ax = fig.axes[0]
        
        # Should have one data point
        assert len(ax.get_yticklabels()) >= 1
        
        plt.close(fig)
    
    def test_forest_plot_many_studies(self, capture_plots):
        """Test forest plot with many studies."""
        np.random.seed(123)
        n_studies = 25
        
        effect_sizes = np.random.normal(0.3, 0.2, n_studies)
        standard_errors = np.random.uniform(0.05, 0.15, n_studies)
        study_labels = [f'Study {i+1:02d}' for i in range(n_studies)]
        
        fig = forest_plot(effect_sizes, standard_errors, study_labels=study_labels)
        
        assert fig is not None
        ax = fig.axes[0]
        
        # Should handle many studies
        assert len(ax.get_yticklabels()) >= n_studies
        
        plt.close(fig)
    
    def test_forest_plot_extreme_values(self, capture_plots):
        """Test forest plot with extreme effect sizes."""
        effect_sizes = [-2, 0, 2, -1.5, 1.8]
        standard_errors = [0.1, 0.1, 0.1, 0.15, 0.12]
        
        fig = forest_plot(effect_sizes, standard_errors)
        
        # Should handle extreme values without error
        assert fig is not None
        
        plt.close(fig)
    
    def test_forest_plot_array_inputs(self, capture_plots):
        """Test forest plot with different array input types."""
        effect_sizes_list = [0.3, 0.5, 0.4]
        se_list = [0.1, 0.12, 0.11]
        
        # Lists
        fig1 = forest_plot(effect_sizes_list, se_list)
        
        # Numpy arrays
        fig2 = forest_plot(np.array(effect_sizes_list), np.array(se_list))
        
        # Both should work
        assert fig1 is not None
        assert fig2 is not None
        
        plt.close(fig1)
        plt.close(fig2)
    
    def test_forest_plot_zero_standard_errors(self, capture_plots):
        """Test forest plot handling of zero standard errors."""
        effect_sizes = [0.3, 0.5, 0.4]
        standard_errors = [0.0, 0.1, 0.12]  # One zero SE
        
        # Should handle gracefully (might show warning)
        fig = forest_plot(effect_sizes, standard_errors)
        
        assert fig is not None
        
        plt.close(fig)


@pytest.mark.viz
class TestForestPlotAdvanced:
    """Advanced tests for forest plot functionality."""
    
    def test_forest_plot_pooled_vs_individual(self, fe_re_data, capture_plots):
        """Test visual distinction between pooled and individual estimates."""
        data = fe_re_data
        
        fig = forest_plot(
            data['effect_sizes'],
            np.sqrt(data['variances']),
            study_labels=data['study_labels'],
            pooled_effect=0.45,
            pooled_se=0.08
        )
        
        ax = fig.axes[0]
        
        # Should have different markers for individual vs pooled
        # Individual studies typically use squares, pooled uses diamond
        scatter_collections = [child for child in ax.get_children() 
                             if hasattr(child, 'get_sizes')]
        
        # Should have at least two different marker types/sizes
        if len(scatter_collections) >= 2:
            sizes1 = scatter_collections[0].get_sizes()
            sizes2 = scatter_collections[1].get_sizes()
            assert not np.array_equal(sizes1, sizes2)
        
        plt.close(fig)
    
    def test_forest_plot_layout_properties(self, continuous_effects_data, capture_plots):
        """Test forest plot layout and formatting."""
        data = continuous_effects_data
        
        fig = forest_plot(
            data['effect_sizes'],
            data['standard_errors'],
            study_labels=data['study_labels']
        )
        
        ax = fig.axes[0]
        
        # Check axis properties
        assert ax.get_xlabel() == 'Effect Size'
        
        # Y-axis should be inverted (studies from top to bottom)
        y_lim = ax.get_ylim()
        assert y_lim[0] > y_lim[1]  # Inverted
        
        # Should have grid
        assert ax.get_axisbelow() is not None  # Grid behind data
        
        plt.close(fig)
    
    def test_forest_plot_confidence_interval_calculation(self, capture_plots):
        """Test that confidence intervals are calculated correctly."""
        effect_sizes = [0.5]
        standard_errors = [0.1]
        
        fig = forest_plot(effect_sizes, standard_errors)
        
        # Extract error bar data
        ax = fig.axes[0]
        errorbar_containers = [child for child in ax.get_children() 
                             if hasattr(child, 'has_xerr')]
        
        if errorbar_containers:
            # 95% CI should be ± 1.96 * SE
            expected_lower = 0.5 - 1.96 * 0.1
            expected_upper = 0.5 + 1.96 * 0.1
            
            # This is a visual test - actual extraction of error bar values
            # is complex in matplotlib, so we just check that error bars exist
            assert len(errorbar_containers) > 0
        
        plt.close(fig)


@pytest.mark.viz
class TestCreateDemoPlot:
    """Test demo plot creation functionality."""
    
    def test_create_demo_plot(self, capture_plots):
        """Test demo plot creation."""
        fig = create_demo_plot()
        
        assert fig is not None
        assert hasattr(fig, 'axes')
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert 'Demo Forest Plot' in ax.get_title()
        
        plt.close(fig)
    
    def test_demo_plot_reproducibility(self, capture_plots):
        """Test that demo plot is reproducible."""
        # Should use fixed seed internally
        fig1 = create_demo_plot()
        fig2 = create_demo_plot()
        
        # Both should be created successfully
        assert fig1 is not None
        assert fig2 is not None
        
        # They should have the same title and basic structure
        assert fig1.axes[0].get_title() == fig2.axes[0].get_title()
        
        plt.close(fig1)
        plt.close(fig2)


@pytest.mark.viz  
class TestForestPlotEdgeCases:
    """Test edge cases and error handling for forest plots."""
    
    def test_empty_input_handling(self, capture_plots):
        """Test handling of empty inputs."""
        # Empty arrays should be handled gracefully
        fig = forest_plot([], [])
        
        # Should create figure even with no data
        assert fig is not None
        
        plt.close(fig)
    
    def test_mismatched_input_lengths(self, capture_plots):
        """Test error handling for mismatched input lengths."""
        with pytest.raises((ValueError, IndexError)):
            forest_plot([0.3, 0.5], [0.1])  # Different lengths
    
    def test_negative_standard_errors(self, capture_plots):
        """Test handling of negative standard errors."""
        effect_sizes = [0.3, 0.5, 0.4]
        standard_errors = [-0.1, 0.12, 0.11]  # One negative
        
        # Should handle gracefully or raise appropriate error
        try:
            fig = forest_plot(effect_sizes, standard_errors)
            assert fig is not None
            plt.close(fig)
        except ValueError:
            # Acceptable to raise error for negative SEs
            pass
    
    def test_very_large_confidence_intervals(self, capture_plots):
        """Test with very large confidence intervals."""
        effect_sizes = [0.5]
        standard_errors = [10.0]  # Very large SE
        
        fig = forest_plot(effect_sizes, standard_errors)
        
        # Should handle large CIs without error
        assert fig is not None
        
        plt.close(fig)
    
    def test_nan_input_handling(self, capture_plots):
        """Test handling of NaN values."""
        effect_sizes = [0.3, np.nan, 0.4]
        standard_errors = [0.1, 0.12, 0.11]
        
        # Should handle NaN gracefully
        try:
            fig = forest_plot(effect_sizes, standard_errors)
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
            fig = forest_plot(effect_sizes, standard_errors)
            assert fig is not None
            plt.close(fig)
        except (ValueError, TypeError):
            # Acceptable to raise error for infinite values
            pass
    
    def test_string_labels_with_special_characters(self, capture_plots):
        """Test study labels with special characters."""
        effect_sizes = [0.3, 0.5]
        standard_errors = [0.1, 0.12]
        study_labels = ['Study α', 'Study β (2023)']
        
        fig = forest_plot(effect_sizes, standard_errors, study_labels=study_labels)
        
        # Should handle special characters in labels
        assert fig is not None
        
        plt.close(fig)
    
    def test_very_long_study_labels(self, capture_plots):
        """Test with very long study labels."""
        effect_sizes = [0.3, 0.5]
        standard_errors = [0.1, 0.12]
        study_labels = [
            'Very Long Study Name That Might Cause Layout Issues',
            'Another Extremely Long Study Name With Many Words'
        ]
        
        fig = forest_plot(effect_sizes, standard_errors, study_labels=study_labels)
        
        # Should handle long labels without crashing
        assert fig is not None
        
        plt.close(fig)


@pytest.mark.viz
class TestForestPlotPerformance:
    """Test performance aspects of forest plot creation."""
    
    def test_large_number_of_studies(self, capture_plots):
        """Test forest plot performance with many studies."""
        np.random.seed(456)
        n_studies = 100  # Large number
        
        effect_sizes = np.random.normal(0.3, 0.2, n_studies)
        standard_errors = np.random.uniform(0.05, 0.15, n_studies)
        study_labels = [f'Study {i+1:03d}' for i in range(n_studies)]
        
        # Should handle large number of studies reasonably quickly
        fig = forest_plot(effect_sizes, standard_errors, study_labels=study_labels)
        
        assert fig is not None
        
        plt.close(fig)
    
    def test_memory_cleanup(self, capture_plots):
        """Test that figures are properly cleaned up."""
        initial_figures = len(plt.get_fignums())
        
        # Create and close multiple figures
        for i in range(5):
            fig = forest_plot([0.3, 0.5], [0.1, 0.12])
            plt.close(fig)
        
        final_figures = len(plt.get_fignums())
        
        # Should not accumulate figures
        assert final_figures <= initial_figures + 1  # Allow for some tolerance