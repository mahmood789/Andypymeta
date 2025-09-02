"""Baujat plot for heterogeneity and influence detection."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any

from ..typing import MetaPoint, MetaResults
from ..models.random_effects import RandomEffects
from ..registries import register_plot
from ..errors import PlottingError
from ..config import config


@register_plot("baujat")
def baujat_plot(points: List[MetaPoint],
               results: Optional[MetaResults] = None,
               title: Optional[str] = None,
               style: Optional[str] = None,
               show_labels: bool = True,
               highlight_threshold: float = 2.0,
               figsize: Optional[Tuple[float, float]] = None,
               save_path: Optional[str] = None) -> plt.Figure:
    """Create Baujat plot for heterogeneity and influence analysis.
    
    The Baujat plot shows the contribution of each study to overall heterogeneity
    (x-axis) versus its influence on the overall result (y-axis).
    
    Args:
        points: List of MetaPoint objects
        results: Optional MetaResults from meta-analysis
        title: Plot title
        style: Plot style name
        show_labels: Whether to show study labels
        highlight_threshold: Threshold for highlighting influential studies
        figsize: Figure size tuple
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure object
        
    References:
        Baujat, B., et al. (2002). A graphical method for exploring heterogeneity
        in meta-analyses. Statistics in Medicine, 21(18), 2641-2652.
    """
    try:
        # Get plot style
        plot_style = config.get_plot_style(style)
        if figsize is None:
            figsize = plot_style.figure_size
        
        # Calculate Baujat statistics
        baujat_stats = calculate_baujat_statistics(points, results)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=plot_style.dpi)
        
        # Extract data
        heterogeneity_contributions = baujat_stats['heterogeneity_contributions']
        influence_measures = baujat_stats['influence_measures']
        study_labels = [p.label or f"Study {i+1}" for i, p in enumerate(points)]
        
        # Plot points
        colors = plot_style.color_palette
        point_color = colors[0] if colors else 'blue'
        
        scatter = ax.scatter(heterogeneity_contributions, influence_measures,
                           c=point_color, alpha=0.7, s=60,
                           edgecolors='black', linewidths=0.5)
        
        # Highlight influential studies
        max_heterogeneity = max(heterogeneity_contributions)
        max_influence = max(influence_measures)
        
        # Define thresholds for highlighting
        het_threshold = max_heterogeneity / highlight_threshold
        inf_threshold = max_influence / highlight_threshold
        
        highlighted_indices = []
        for i, (het, inf) in enumerate(zip(heterogeneity_contributions, influence_measures)):
            if het > het_threshold or inf > inf_threshold:
                highlighted_indices.append(i)
                # Highlight with different color
                highlight_color = colors[1] if len(colors) > 1 else 'red'
                ax.scatter(het, inf, c=highlight_color, s=80, alpha=0.9,
                         edgecolors='black', linewidths=1, marker='o')
        
        # Add study labels
        if show_labels:
            for i, (het, inf, label) in enumerate(zip(heterogeneity_contributions,
                                                     influence_measures, study_labels)):
                # Only label highlighted studies or if few studies
                if i in highlighted_indices or len(points) <= 10:
                    ax.annotate(label, (het, inf), xytext=(5, 5),
                              textcoords='offset points', fontsize=plot_style.font_size - 2,
                              alpha=0.8)
        
        # Add reference lines
        if len(heterogeneity_contributions) > 3:
            # Median lines
            median_het = np.median(heterogeneity_contributions)
            median_inf = np.median(influence_measures)
            
            ax.axvline(x=median_het, color='gray', linestyle='--', alpha=0.5,
                      label=f'Median Heterogeneity ({median_het:.3f})')
            ax.axhline(y=median_inf, color='gray', linestyle='--', alpha=0.5,
                      label=f'Median Influence ({median_inf:.3f})')
        
        # Labels and formatting
        ax.set_xlabel('Contribution to Heterogeneity', fontsize=plot_style.font_size)
        ax.set_ylabel('Influence on Overall Result', fontsize=plot_style.font_size)
        
        if title:
            ax.set_title(title, fontsize=plot_style.font_size + 2, fontweight='bold')
        else:
            ax.set_title('Baujat Plot', fontsize=plot_style.font_size + 2, fontweight='bold')
        
        # Grid
        if plot_style.grid:
            ax.grid(True, alpha=plot_style.grid_alpha)
        
        # Legend if reference lines were added
        if len(heterogeneity_contributions) > 3:
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
        raise PlottingError(f"Baujat plot creation failed: {e}")


def calculate_baujat_statistics(points: List[MetaPoint],
                              results: Optional[MetaResults] = None) -> Dict[str, Any]:
    """Calculate Baujat plot statistics.
    
    Args:
        points: List of MetaPoint objects
        results: Optional MetaResults from meta-analysis
        
    Returns:
        Dictionary with Baujat statistics
    """
    if len(points) < 3:
        raise PlottingError("At least 3 studies required for Baujat plot")
    
    try:
        # Fit overall model if results not provided
        if results is None:
            model = RandomEffects(points)
            results = model.fit()
        
        n_studies = len(points)
        heterogeneity_contributions = []
        influence_measures = []
        
        # Calculate for each study
        for i in range(n_studies):
            # Leave-one-out analysis
            loo_points = [p for j, p in enumerate(points) if j != i]
            
            if len(loo_points) >= 2:
                # Fit model without study i
                loo_model = RandomEffects(loo_points)
                loo_result = loo_model.fit()
                
                # Heterogeneity contribution (reduction in Q when study removed)
                q_contribution = results.q_statistic - loo_result.q_statistic
                heterogeneity_contributions.append(max(0, q_contribution))
                
                # Influence measure (squared standardized difference in pooled effect)
                effect_difference = results.pooled_effect - loo_result.pooled_effect
                pooled_se = np.sqrt(results.pooled_variance)
                
                if pooled_se > 0:
                    standardized_difference = effect_difference / pooled_se
                    influence = standardized_difference ** 2
                else:
                    influence = 0
                
                influence_measures.append(influence)
            else:
                # Not enough studies for leave-one-out
                heterogeneity_contributions.append(0)
                influence_measures.append(0)
        
        # Study information
        study_info = []
        for i, point in enumerate(points):
            study_info.append({
                'study_index': i,
                'label': point.label or f"Study {i+1}",
                'effect': point.effect,
                'variance': point.variance,
                'weight': point.weight,
                'heterogeneity_contribution': heterogeneity_contributions[i],
                'influence_measure': influence_measures[i]
            })
        
        # Sort by combined influence (heterogeneity + influence)
        combined_influence = [h + inf for h, inf in zip(heterogeneity_contributions, influence_measures)]
        most_influential_idx = np.argmax(combined_influence)
        
        return {
            'heterogeneity_contributions': heterogeneity_contributions,
            'influence_measures': influence_measures,
            'study_info': study_info,
            'most_influential_study': {
                'index': most_influential_idx,
                'label': points[most_influential_idx].label or f"Study {most_influential_idx + 1}",
                'heterogeneity_contribution': heterogeneity_contributions[most_influential_idx],
                'influence_measure': influence_measures[most_influential_idx],
                'combined_influence': combined_influence[most_influential_idx]
            },
            'summary': {
                'max_heterogeneity_contribution': max(heterogeneity_contributions),
                'max_influence_measure': max(influence_measures),
                'mean_heterogeneity_contribution': np.mean(heterogeneity_contributions),
                'mean_influence_measure': np.mean(influence_measures)
            }
        }
        
    except Exception as e:
        raise PlottingError(f"Baujat statistics calculation failed: {e}")


def identify_outliers_baujat(points: List[MetaPoint],
                           heterogeneity_threshold: float = 2.0,
                           influence_threshold: float = 2.0) -> Dict[str, Any]:
    """Identify outlying studies using Baujat plot criteria.
    
    Args:
        points: List of MetaPoint objects
        heterogeneity_threshold: Threshold for heterogeneity contribution (as multiple of median)
        influence_threshold: Threshold for influence measure (as multiple of median)
        
    Returns:
        Dictionary with outlier information
    """
    baujat_stats = calculate_baujat_statistics(points)
    
    het_contributions = baujat_stats['heterogeneity_contributions']
    influence_measures = baujat_stats['influence_measures']
    
    # Calculate thresholds
    median_het = np.median(het_contributions)
    median_inf = np.median(influence_measures)
    
    het_cutoff = median_het * heterogeneity_threshold
    inf_cutoff = median_inf * influence_threshold
    
    # Identify outliers
    outliers = []
    for i, (het, inf) in enumerate(zip(het_contributions, influence_measures)):
        is_outlier = het > het_cutoff or inf > inf_cutoff
        
        if is_outlier:
            outliers.append({
                'study_index': i,
                'label': points[i].label or f"Study {i+1}",
                'heterogeneity_contribution': het,
                'influence_measure': inf,
                'outlier_reason': []
            })
            
            # Specify reasons
            if het > het_cutoff:
                outliers[-1]['outlier_reason'].append('high_heterogeneity_contribution')
            if inf > inf_cutoff:
                outliers[-1]['outlier_reason'].append('high_influence')
    
    return {
        'outliers': outliers,
        'n_outliers': len(outliers),
        'thresholds': {
            'heterogeneity_cutoff': het_cutoff,
            'influence_cutoff': inf_cutoff,
            'heterogeneity_threshold_multiplier': heterogeneity_threshold,
            'influence_threshold_multiplier': influence_threshold
        },
        'summary_stats': {
            'median_heterogeneity_contribution': median_het,
            'median_influence_measure': median_inf,
            'max_heterogeneity_contribution': max(het_contributions),
            'max_influence_measure': max(influence_measures)
        }
    }