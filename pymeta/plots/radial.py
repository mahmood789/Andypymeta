"""Radial (Galbraith) plot for meta-analysis."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from ..typing import MetaPoint, MetaResults
from ..registries import register_plot
from ..errors import PlottingError
from ..config import config


@register_plot("radial")
def radial_plot(points: List[MetaPoint],
               results: Optional[MetaResults] = None,
               title: Optional[str] = None,
               style: Optional[str] = None,
               show_regression_line: bool = True,
               figsize: Optional[Tuple[float, float]] = None,
               save_path: Optional[str] = None) -> plt.Figure:
    """Create radial (Galbraith) plot for meta-analysis.
    
    The radial plot shows standardized effects vs precision,
    useful for detecting outliers and heterogeneity patterns.
    
    Args:
        points: List of MetaPoint objects
        results: Optional MetaResults from meta-analysis
        title: Plot title
        style: Plot style name
        show_regression_line: Whether to show regression line
        figsize: Figure size tuple
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure object
        
    References:
        Galbraith, R. F. (1988). A note on graphical presentation of estimated
        odds ratios from several clinical trials. Statistics in Medicine, 7(8), 889-894.
    """
    try:
        # Get plot style
        plot_style = config.get_plot_style(style)
        if figsize is None:
            figsize = plot_style.figure_size
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=plot_style.dpi)
        
        # Calculate radial plot coordinates
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        standard_errors = np.sqrt(variances)
        
        # Precision (1/SE)
        precision = 1.0 / standard_errors
        
        # Standardized effects (effect/SE)
        standardized_effects = effects / standard_errors
        
        # Plot points
        colors = plot_style.color_palette
        point_color = colors[0] if colors else 'blue'
        
        ax.scatter(precision, standardized_effects, c=point_color, alpha=0.7, s=50,
                  edgecolors='black', linewidths=0.5)
        
        # Add reference lines
        if results:
            # Line for pooled effect
            pooled_effect = results.pooled_effect
            max_precision = precision.max()
            precision_range = np.linspace(0, max_precision * 1.1, 100)
            
            # Pooled effect line: standardized_effect = pooled_effect * precision
            pooled_line = pooled_effect * precision_range
            ax.plot(precision_range, pooled_line, color='red', linewidth=2,
                   alpha=0.8, label=f'Pooled Effect ({pooled_effect:.3f})')
        
        # Add confidence bounds
        z_95 = 1.96
        max_precision = precision.max()
        precision_range = np.linspace(0, max_precision * 1.1, 100)
        
        # Upper and lower 95% bounds
        upper_bound = z_95 * np.ones_like(precision_range)
        lower_bound = -z_95 * np.ones_like(precision_range)
        
        ax.plot(precision_range, upper_bound, color='gray', linestyle='--',
               alpha=0.5, label='95% Confidence Bounds')
        ax.plot(precision_range, lower_bound, color='gray', linestyle='--', alpha=0.5)
        
        # Add regression line if requested
        if show_regression_line and len(points) >= 3:
            # Weighted regression
            weights = precision ** 2
            
            # Design matrix for regression: standardized_effect = a + b * precision
            X = np.column_stack([np.ones(len(precision)), precision])
            y = standardized_effects
            W = np.diag(weights)
            
            try:
                # Weighted least squares
                XTW = X.T @ W
                beta = np.linalg.solve(XTW @ X, XTW @ y)
                
                # Plot regression line
                regression_line = X @ beta
                sort_idx = np.argsort(precision)
                ax.plot(precision[sort_idx], regression_line[sort_idx],
                       color='green', linewidth=2, alpha=0.8,
                       label=f'Regression Line (slope={beta[1]:.3f})')
                
            except np.linalg.LinAlgError:
                pass  # Skip if singular matrix
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Add study labels for outliers
        for i, (prec, std_eff) in enumerate(zip(precision, standardized_effects)):
            if abs(std_eff) > 2.5:  # Outlier threshold
                label = points[i].label or f"Study {i+1}"
                ax.annotate(label, (prec, std_eff), xytext=(5, 5),
                          textcoords='offset points', fontsize=plot_style.font_size - 2,
                          alpha=0.8)
        
        # Labels and formatting
        ax.set_xlabel('Precision (1/SE)', fontsize=plot_style.font_size)
        ax.set_ylabel('Standardized Effect (Effect/SE)', fontsize=plot_style.font_size)
        
        if title:
            ax.set_title(title, fontsize=plot_style.font_size + 2, fontweight='bold')
        else:
            ax.set_title('Radial Plot', fontsize=plot_style.font_size + 2, fontweight='bold')
        
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
        raise PlottingError(f"Radial plot creation failed: {e}")


def detect_outliers_radial(points: List[MetaPoint], threshold: float = 2.0) -> dict:
    """Detect outliers using radial plot criteria.
    
    Args:
        points: List of MetaPoint objects
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Dictionary with outlier information
    """
    try:
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        standard_errors = np.sqrt(variances)
        
        # Standardized effects
        standardized_effects = effects / standard_errors
        
        # Identify outliers
        outliers = []
        for i, std_eff in enumerate(standardized_effects):
            if abs(std_eff) > threshold:
                outliers.append({
                    'study_index': i,
                    'label': points[i].label or f"Study {i+1}",
                    'effect': effects[i],
                    'standard_error': standard_errors[i],
                    'standardized_effect': std_eff,
                    'z_score': abs(std_eff),
                    'outlier_direction': 'positive' if std_eff > 0 else 'negative'
                })
        
        return {
            'outliers': outliers,
            'n_outliers': len(outliers),
            'threshold': threshold,
            'max_abs_standardized_effect': max(abs(standardized_effects)),
            'outlier_indices': [o['study_index'] for o in outliers]
        }
        
    except Exception as e:
        raise PlottingError(f"Radial outlier detection failed: {e}")