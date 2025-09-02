"""
Basic funnel plot implementation for meta-analysis results.
"""

from typing import Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt

from .. import MetaResults, MetaPoint


def plot_funnel(
    results: MetaResults,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    show_ci: bool = True,
    save_path: Optional[str] = None,
    **kwargs: Any
) -> plt.Figure:
    """
    Create a funnel plot for meta-analysis results.
    
    Args:
        results: MetaResults object with analysis results
        title: Plot title (auto-generated if None)
        figsize: Figure size as (width, height)
        show_ci: Whether to show confidence interval lines
        save_path: Path to save the plot (optional)
        **kwargs: Additional arguments passed to matplotlib
        
    Returns:
        matplotlib Figure object
    """
    if results.points is None:
        raise ValueError("MetaResults must contain points for funnel plot")
    
    points = results.points
    
    # Extract data
    effects = np.array([p.effect for p in points])
    ses = np.array([p.se for p in points])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual studies
    ax.scatter(effects, ses, alpha=0.7, s=50, edgecolors='black', linewidths=0.5)
    
    # Plot pooled effect line
    ax.axvline(x=results.effect, color='red', linestyle='-', linewidth=2,
               label=f'Pooled Effect = {results.effect:.3f}')
    
    # Plot confidence interval lines if requested
    if show_ci:
        # Create range for CI lines
        se_range = np.linspace(0, max(ses) * 1.1, 100)
        
        # 95% CI lines
        ci_lower = results.effect - 1.96 * se_range
        ci_upper = results.effect + 1.96 * se_range
        
        ax.plot(ci_lower, se_range, '--', color='gray', alpha=0.7, label='95% CI')
        ax.plot(ci_upper, se_range, '--', color='gray', alpha=0.7)
    
    # Customize plot
    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Standard Error')
    ax.set_title(title or 'Funnel Plot')
    ax.invert_yaxis()  # Larger studies (smaller SE) at top
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig