"""Basic funnel plot for publication bias assessment."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from scipy import stats

from ..typing import MetaPoint, MetaResults
from ..registries import register_plot
from ..errors import PlottingError
from ..config import config


@register_plot("funnel")
def funnel_plot(points: List[MetaPoint],
               results: Optional[MetaResults] = None,
               title: Optional[str] = None,
               style: Optional[str] = None,
               show_confidence_lines: bool = True,
               show_effect_line: bool = True,
               figsize: Optional[Tuple[float, float]] = None,
               save_path: Optional[str] = None) -> plt.Figure:
    """Create funnel plot for publication bias assessment.
    
    Args:
        points: List of MetaPoint objects
        results: MetaResults from meta-analysis (for reference line)
        title: Plot title
        style: Plot style name
        show_confidence_lines: Whether to show pseudo-confidence region
        show_effect_line: Whether to show pooled effect line
        figsize: Figure size tuple
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Get plot style
        plot_style = config.get_plot_style(style)
        if figsize is None:
            figsize = plot_style.figure_size
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=plot_style.dpi)
        
        # Extract data
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        standard_errors = np.sqrt(variances)
        
        # Use precision (1/SE) for y-axis (inverted funnel)
        precision = 1.0 / standard_errors
        
        # Plot points
        colors = plot_style.color_palette
        point_color = colors[0] if colors else 'blue'
        
        ax.scatter(effects, precision, c=point_color, alpha=0.7, s=50,
                  edgecolors='black', linewidths=0.5)
        
        # Add pooled effect line
        if show_effect_line and results:
            pooled_effect = results.pooled_effect
            ax.axvline(x=pooled_effect, color='red', linestyle='-', 
                      linewidth=2, alpha=0.8, label=f'Pooled Effect ({pooled_effect:.3f})')
        
        # Add null effect line
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, 
                  alpha=0.5, label='No Effect')
        
        # Add pseudo-confidence region
        if show_confidence_lines:
            # Create range for confidence lines
            effect_range = np.linspace(effects.min() - 0.5, effects.max() + 0.5, 100)
            max_precision = precision.max()
            precision_range = np.linspace(0, max_precision * 1.1, 100)
            
            # Use pooled effect as center, or 0 if no results
            center_effect = results.pooled_effect if results else 0
            
            # 95% pseudo-confidence lines
            z_95 = 1.96
            for prec in precision_range:
                if prec > 0:
                    se = 1.0 / prec
                    ci_width = z_95 * se
                    left_bound = center_effect - ci_width
                    right_bound = center_effect + ci_width
                    
                    # Only plot if within reasonable range
                    if abs(left_bound) < 10 and abs(right_bound) < 10:
                        ax.plot([left_bound, right_bound], [prec, prec], 
                               color='gray', alpha=0.3, linewidth=0.5)
        
        # Labels and formatting
        ax.set_xlabel('Effect Size', fontsize=plot_style.font_size)
        ax.set_ylabel('Precision (1/SE)', fontsize=plot_style.font_size)
        
        if title:
            ax.set_title(title, fontsize=plot_style.font_size + 2, fontweight='bold')
        else:
            ax.set_title('Funnel Plot', fontsize=plot_style.font_size + 2, fontweight='bold')
        
        # Grid
        if plot_style.grid:
            ax.grid(True, alpha=plot_style.grid_alpha)
        
        # Legend
        ax.legend(loc='upper right')
        
        # Spine styling
        if plot_style.spine_style == 'minimal':
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=plot_style.dpi, bbox_inches='tight')
        
        return fig
        
    except Exception as e:
        raise PlottingError(f"Funnel plot creation failed: {e}")


@register_plot("funnel_se")
def funnel_plot_standard_error(points: List[MetaPoint],
                              results: Optional[MetaResults] = None,
                              title: Optional[str] = None,
                              style: Optional[str] = None,
                              figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
    """Create funnel plot with standard error on y-axis (traditional orientation).
    
    Args:
        points: List of MetaPoint objects
        results: MetaResults from meta-analysis
        title: Plot title
        style: Plot style name
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Get plot style
        plot_style = config.get_plot_style(style)
        if figsize is None:
            figsize = plot_style.figure_size
        
        fig, ax = plt.subplots(figsize=figsize, dpi=plot_style.dpi)
        
        # Extract data
        effects = np.array([p.effect for p in points])
        standard_errors = np.sqrt([p.variance for p in points])
        
        # Plot points
        colors = plot_style.color_palette
        point_color = colors[0] if colors else 'blue'
        
        ax.scatter(effects, standard_errors, c=point_color, alpha=0.7, s=50,
                  edgecolors='black', linewidths=0.5)
        
        # Add reference lines
        if results:
            ax.axvline(x=results.pooled_effect, color='red', linestyle='-',
                      linewidth=2, alpha=0.8, label=f'Pooled Effect ({results.pooled_effect:.3f})')
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1,
                  alpha=0.5, label='No Effect')
        
        # Invert y-axis (smaller SE at top)
        ax.invert_yaxis()
        
        # Labels
        ax.set_xlabel('Effect Size', fontsize=plot_style.font_size)
        ax.set_ylabel('Standard Error', fontsize=plot_style.font_size)
        
        if title:
            ax.set_title(title, fontsize=plot_style.font_size + 2, fontweight='bold')
        else:
            ax.set_title('Funnel Plot (Standard Error)', 
                        fontsize=plot_style.font_size + 2, fontweight='bold')
        
        # Grid and legend
        if plot_style.grid:
            ax.grid(True, alpha=plot_style.grid_alpha)
        
        ax.legend()
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        raise PlottingError(f"SE funnel plot creation failed: {e}")


def interpret_funnel_plot(points: List[MetaPoint]) -> dict:
    """Provide interpretation of funnel plot asymmetry.
    
    Args:
        points: List of MetaPoint objects
        
    Returns:
        Dictionary with interpretation
    """
    # Basic asymmetry assessment
    effects = np.array([p.effect for p in points])
    standard_errors = np.sqrt([p.variance for p in points])
    
    # Split by precision
    median_precision = np.median(1.0 / standard_errors)
    high_precision = effects[standard_errors <= 1.0 / median_precision]
    low_precision = effects[standard_errors > 1.0 / median_precision]
    
    # Compare means
    if len(high_precision) > 0 and len(low_precision) > 0:
        high_mean = np.mean(high_precision)
        low_mean = np.mean(low_precision)
        asymmetry = abs(high_mean - low_mean)
    else:
        asymmetry = 0
    
    # Interpretation
    if asymmetry > 0.5:
        interpretation = "High asymmetry suggests possible publication bias"
    elif asymmetry > 0.2:
        interpretation = "Moderate asymmetry, possible small-study effects"
    else:
        interpretation = "Low asymmetry, no strong evidence of bias"
    
    return {
        'asymmetry_measure': asymmetry,
        'high_precision_mean': np.mean(high_precision) if len(high_precision) > 0 else None,
        'low_precision_mean': np.mean(low_precision) if len(low_precision) > 0 else None,
        'interpretation': interpretation,
        'n_high_precision': len(high_precision),
        'n_low_precision': len(low_precision)
    }