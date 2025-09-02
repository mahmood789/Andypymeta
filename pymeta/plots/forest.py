"""
Forest plot implementation for meta-analysis results.
"""

from typing import Optional, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .. import MetaResults, MetaPoint


def plot_forest(
    results: MetaResults,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    show_weights: bool = True,
    show_stats: bool = True,
    save_path: Optional[str] = None,
    **kwargs: Any
) -> plt.Figure:
    """
    Create a forest plot for meta-analysis results.
    
    Args:
        results: MetaResults object with analysis results
        title: Plot title (auto-generated if None)
        figsize: Figure size as (width, height)
        show_weights: Whether to show study weights
        show_stats: Whether to show heterogeneity statistics
        save_path: Path to save the plot (optional)
        **kwargs: Additional arguments passed to matplotlib
        
    Returns:
        matplotlib Figure object
    """
    if results.points is None:
        raise ValueError("MetaResults must contain points for forest plot")
    
    points = results.points
    k = len(points)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate positions
    y_positions = np.arange(k + 2)  # +2 for overall result and spacing
    
    # Plot individual studies
    for i, point in enumerate(points):
        y_pos = y_positions[i]
        
        # Calculate confidence interval
        se = point.se
        ci_lower = point.effect - 1.96 * se
        ci_upper = point.effect + 1.96 * se
        
        # Plot point estimate
        ax.plot(point.effect, y_pos, 'o', markersize=8, color='blue', alpha=0.7)
        
        # Plot confidence interval
        ax.plot([ci_lower, ci_upper], [y_pos, y_pos], '-', color='blue', alpha=0.7)
        
        # Add study label
        label = point.study_id
        if show_weights and hasattr(point, 'weight'):
            label += f" ({point.weight:.1f}%)"
        ax.text(-0.1, y_pos, label, ha='right', va='center', 
                transform=ax.get_yaxis_transform())
    
    # Plot overall result
    overall_y = y_positions[k + 1]
    
    # Plot diamond for overall effect
    diamond_width = 0.3
    diamond_x = [
        results.ci_lower,
        results.effect,
        results.ci_upper,
        results.effect,
        results.ci_lower
    ]
    diamond_y = [
        overall_y,
        overall_y + diamond_width,
        overall_y,
        overall_y - diamond_width,
        overall_y
    ]
    
    diamond = patches.Polygon(
        list(zip(diamond_x, diamond_y)),
        closed=True,
        facecolor='red',
        edgecolor='darkred',
        alpha=0.7
    )
    ax.add_patch(diamond)
    
    # Add overall label
    overall_label = "Overall"
    if results.use_hksj:
        overall_label += " (HKSJ)"
    ax.text(-0.1, overall_y, overall_label, ha='right', va='center',
            weight='bold', transform=ax.get_yaxis_transform())
    
    # Add vertical line at null effect
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Studies')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Set title
    if title is None:
        title = f"Forest Plot - {results.method}"
        if results.use_hksj:
            title += " with HKSJ"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add statistics text
    if show_stats:
        stats_text = (
            f"Effect: {results.effect:.3f} "
            f"[{results.ci_lower:.3f}, {results.ci_upper:.3f}]\n"
            f"P-value: {results.p_value:.3f}\n"
            f"τ² = {results.tau2:.3f}, I² = {results.i2:.1f}%\n"
            f"Q = {results.q_stat:.2f}, p = {results.q_p_value:.3f}"
        )
        
        if results.use_hksj and results.df is not None:
            stats_text += f"\ndf = {results.df}"
            
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig