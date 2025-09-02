"""
Contour-enhanced funnel plot with significance contours for meta-analysis.

This module provides funnel plots with contour lines showing different
significance levels, helping to visualize publication bias and the 
distribution of study effects.
"""

from typing import Optional, Tuple, List, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .. import MetaResults, MetaPoint


def plot_funnel_contour(
    results: MetaResults,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    contour_levels: Optional[List[float]] = None,
    show_pooled: bool = True,
    show_studies: bool = True,
    save_path: Optional[str] = None,
    **kwargs: Any
) -> plt.Figure:
    """
    Create a contour-enhanced funnel plot for meta-analysis results.
    
    This plot shows significance contours for different p-value levels,
    helping to identify areas where studies would be considered significant
    and potentially revealing publication bias patterns.
    
    Args:
        results: MetaResults object with analysis results
        title: Plot title (auto-generated if None)
        figsize: Figure size as (width, height)
        contour_levels: P-value levels for contours (default: [0.01, 0.05, 0.10])
        show_pooled: Whether to show pooled effect line
        show_studies: Whether to show individual study points
        save_path: Path to save the plot (optional)
        **kwargs: Additional arguments passed to matplotlib
        
    Returns:
        matplotlib Figure object
    """
    if results.points is None:
        raise ValueError("MetaResults must contain points for funnel plot")
    
    if contour_levels is None:
        contour_levels = [0.01, 0.05, 0.10]
    
    points = results.points
    
    # Extract data
    effects = np.array([p.effect for p in points])
    ses = np.array([p.se for p in points])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine plot bounds
    effect_min = min(effects.min(), results.effect) - 2 * ses.max()
    effect_max = max(effects.max(), results.effect) + 2 * ses.max()
    se_max = ses.max() * 1.2
    
    # Create grid for contour calculation
    effect_grid = np.linspace(effect_min, effect_max, 200)
    se_grid = np.linspace(0.001, se_max, 100)  # Start from small positive value
    
    Effect_grid, SE_grid = np.meshgrid(effect_grid, se_grid)
    
    # Calculate z-scores for each point in the grid
    # Assuming null hypothesis of effect = 0
    Z_grid = np.abs(Effect_grid) / SE_grid
    
    # Convert z-scores to two-sided p-values
    from scipy.stats import norm
    P_grid = 2 * (1 - norm.cdf(Z_grid))
    
    # Create custom colormap
    colors = ['#f7f7f7', '#cccccc', '#969696', '#636363']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('significance', colors, N=n_bins)
    
    # Plot contours
    contour_colors = ['red', 'orange', 'yellow']
    contour_labels = [f'p = {level}' for level in contour_levels]
    
    # Plot filled contours for background
    contour_fill = ax.contourf(Effect_grid, SE_grid, P_grid, 
                              levels=[0] + contour_levels + [1.0],
                              cmap=cmap, alpha=0.6, extend='max')
    
    # Plot contour lines
    contour_lines = ax.contour(Effect_grid, SE_grid, P_grid,
                              levels=contour_levels,
                              colors=contour_colors,
                              linewidths=2,
                              linestyles='--')
    
    # Add contour labels
    ax.clabel(contour_lines, contour_levels, inline=True, fontsize=10,
              fmt=lambda x: f'p = {x:.2f}')
    
    # Plot individual studies
    if show_studies:
        scatter = ax.scatter(effects, ses, c='blue', s=60, alpha=0.8, 
                           edgecolors='darkblue', linewidths=1, zorder=5,
                           label='Studies')
        
        # Optionally add study labels for influential points
        for i, point in enumerate(points):
            if point.se < se_max * 0.1 or abs(point.effect) > 2 * results.se:
                ax.annotate(point.study_id, (point.effect, point.se),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
    
    # Plot pooled effect line
    if show_pooled:
        ax.axvline(x=results.effect, color='red', linestyle='-', linewidth=3,
                   label=f'Pooled Effect = {results.effect:.3f}', zorder=4)
        
        # Add confidence interval as shaded region
        se_range = np.linspace(0, se_max, 100)
        ci_lower = results.effect - 1.96 * se_range
        ci_upper = results.effect + 1.96 * se_range
        ax.fill_betweenx(se_range, ci_lower, ci_upper, 
                        alpha=0.2, color='red', label='95% CI')
    
    # Plot null effect line
    ax.axvline(x=0, color='black', linestyle=':', linewidth=1,
               alpha=0.7, label='Null Effect')
    
    # Customize plot
    ax.set_xlabel('Effect Size', fontsize=12)
    ax.set_ylabel('Standard Error', fontsize=12)
    ax.invert_yaxis()  # Larger studies (smaller SE) at top
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Set title
    if title is None:
        title = f"Contour-Enhanced Funnel Plot - {results.method}"
        if results.use_hksj:
            title += " (HKSJ)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add statistics text box
    stats_text = (
        f"Effect: {results.effect:.3f} [{results.ci_lower:.3f}, {results.ci_upper:.3f}]\n"
        f"P-value: {results.p_value:.3f}\n"
        f"τ² = {results.tau2:.3f}, I² = {results.i2:.1f}%\n"
        f"Studies: {len(points)}"
    )
    
    if results.use_hksj and results.df is not None:
        stats_text += f"\ndf = {results.df} (HKSJ)"
        
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='white', alpha=0.9), fontsize=10)
    
    # Add colorbar for p-value interpretation
    cbar = plt.colorbar(contour_fill, ax=ax, shrink=0.8)
    cbar.set_label('Two-sided p-value', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_funnel_contour_asymmetry(
    results: MetaResults,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None,
    **kwargs: Any
) -> plt.Figure:
    """
    Create a funnel plot specifically designed to assess asymmetry/publication bias.
    
    This variant emphasizes the symmetry around the pooled effect and makes
    it easier to spot asymmetrical patterns that might indicate publication bias.
    
    Args:
        results: MetaResults object with analysis results
        title: Plot title (auto-generated if None)
        figsize: Figure size as (width, height)
        save_path: Path to save the plot (optional)
        **kwargs: Additional arguments passed to matplotlib
        
    Returns:
        matplotlib Figure object
    """
    if results.points is None:
        raise ValueError("MetaResults must contain points for funnel plot")
    
    points = results.points
    
    # Transform effects to center around pooled effect
    effects = np.array([p.effect for p in points])
    ses = np.array([p.se for p in points])
    centered_effects = effects - results.effect
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create symmetric contours around zero (pooled effect)
    se_max = ses.max() * 1.2
    effect_range = 4 * se_max  # Symmetric range
    
    effect_grid = np.linspace(-effect_range, effect_range, 200)
    se_grid = np.linspace(0.001, se_max, 100)
    
    Effect_grid, SE_grid = np.meshgrid(effect_grid, se_grid)
    
    # Calculate significance contours
    from scipy.stats import norm
    Z_grid = np.abs(Effect_grid) / SE_grid
    P_grid = 2 * (1 - norm.cdf(Z_grid))
    
    # Plot symmetric significance regions
    contour_levels = [0.01, 0.05, 0.10]
    colors = ['darkred', 'red', 'orange']
    
    for i, (level, color) in enumerate(zip(contour_levels, colors)):
        contour = ax.contour(Effect_grid, SE_grid, P_grid,
                           levels=[level], colors=[color],
                           linewidths=2, linestyles='--')
        ax.clabel(contour, [level], inline=True, fontsize=10,
                 fmt=lambda x: f'p = {x:.2f}')
    
    # Plot individual studies (centered)
    ax.scatter(centered_effects, ses, c='blue', s=60, alpha=0.8,
              edgecolors='darkblue', linewidths=1, zorder=5)
    
    # Add vertical line at zero (pooled effect)
    ax.axvline(x=0, color='red', linestyle='-', linewidth=3,
               label='Pooled Effect', zorder=4)
    
    # Customize plot
    ax.set_xlabel('Effect Size (centered on pooled effect)', fontsize=12)
    ax.set_ylabel('Standard Error', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set title
    if title is None:
        title = "Funnel Plot - Publication Bias Assessment"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig