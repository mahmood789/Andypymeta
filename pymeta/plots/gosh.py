"""GOSH (Graphical Display of Study Heterogeneity) plot with fast subset sampling."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
from itertools import combinations
import random

from ..typing import MetaPoint, MetaResults
from ..models.random_effects import RandomEffects
from ..registries import register_plot
from ..errors import PlottingError
from ..config import config


@register_plot("gosh")
def gosh_plot(points: List[MetaPoint],
             max_subsets: Optional[int] = None,
             min_studies: int = 3,
             title: Optional[str] = None,
             style: Optional[str] = None,
             color_by: str = "effect",
             figsize: Optional[Tuple[float, float]] = None,
             seed: Optional[int] = None,
             save_path: Optional[str] = None) -> plt.Figure:
    """Create GOSH plot with fast subset sampling.
    
    GOSH plots show meta-analysis results from all possible subsets of studies,
    useful for detecting influential studies and heterogeneity patterns.
    
    Args:
        points: List of MetaPoint objects
        max_subsets: Maximum number of subsets to sample (default from config)
        min_studies: Minimum number of studies per subset
        title: Plot title
        style: Plot style name
        color_by: Color points by ("effect", "tau2", "i_squared")
        figsize: Figure size tuple
        seed: Random seed for reproducible sampling
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure object
        
    References:
        Olkin, I., et al. (2012). GOSH–a graphical display of study heterogeneity.
        Research Synthesis Methods, 3(3), 214-223.
    """
    try:
        # Get configuration
        if max_subsets is None:
            max_subsets = config.gosh_max_subsets
        if seed is None:
            seed = config.gosh_seed
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Get plot style
        plot_style = config.get_plot_style(style)
        if figsize is None:
            figsize = plot_style.figure_size
        
        # Generate subset results
        subset_results = generate_gosh_subsets(points, max_subsets, min_studies)
        
        if not subset_results:
            raise PlottingError("No valid subsets generated for GOSH plot")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=plot_style.dpi)
        
        # Extract data for plotting
        effects = [r['pooled_effect'] for r in subset_results]
        i_squared_values = [r['i_squared'] for r in subset_results]
        
        # Color mapping
        if color_by == "effect":
            color_values = effects
            color_label = "Pooled Effect"
        elif color_by == "tau2":
            color_values = [r['tau2'] for r in subset_results]
            color_label = "Tau²"
        elif color_by == "i_squared":
            color_values = i_squared_values
            color_label = "I² (%)"
        else:
            color_values = effects
            color_label = "Pooled Effect"
        
        # Create scatter plot
        scatter = ax.scatter(effects, i_squared_values, c=color_values,
                           alpha=0.6, s=20, cmap='viridis',
                           edgecolors='none')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_label, fontsize=plot_style.font_size)
        
        # Highlight original full meta-analysis result
        try:
            full_model = RandomEffects(points)
            full_result = full_model.fit()
            
            ax.scatter(full_result.pooled_effect, full_result.i_squared,
                      c='red', s=100, marker='*', alpha=0.9,
                      edgecolors='black', linewidths=1,
                      label=f'Full Analysis (n={len(points)})')
        except:
            pass  # Skip if full analysis fails
        
        # Add reference lines
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1,
                  label='I² = 50%')
        ax.axhline(y=75, color='gray', linestyle=':', alpha=0.5, linewidth=1,
                  label='I² = 75%')
        
        # Labels and formatting
        ax.set_xlabel('Pooled Effect Size', fontsize=plot_style.font_size)
        ax.set_ylabel('I² (%)', fontsize=plot_style.font_size)
        
        if title:
            ax.set_title(title, fontsize=plot_style.font_size + 2, fontweight='bold')
        else:
            ax.set_title(f'GOSH Plot ({len(subset_results)} subsets)',
                        fontsize=plot_style.font_size + 2, fontweight='bold')
        
        # Grid
        if plot_style.grid:
            ax.grid(True, alpha=plot_style.grid_alpha)
        
        # Legend
        ax.legend(loc='upper right')
        
        # Set I² limits
        ax.set_ylim(-5, 105)
        
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
        raise PlottingError(f"GOSH plot creation failed: {e}")


def generate_gosh_subsets(points: List[MetaPoint],
                         max_subsets: int = 10000,
                         min_studies: int = 3) -> List[Dict[str, Any]]:
    """Generate meta-analysis results for GOSH subsets with efficient sampling.
    
    Args:
        points: List of MetaPoint objects
        max_subsets: Maximum number of subsets to generate
        min_studies: Minimum number of studies per subset
        
    Returns:
        List of dictionaries with subset results
    """
    if len(points) < min_studies:
        raise PlottingError(f"Need at least {min_studies} studies for GOSH analysis")
    
    n_studies = len(points)
    subset_results = []
    
    # Calculate total possible subsets
    total_possible = sum(
        len(list(combinations(range(n_studies), k)))
        for k in range(min_studies, n_studies + 1)
    )
    
    # Decide on sampling strategy
    if total_possible <= max_subsets:
        # Generate all possible subsets
        subset_indices = []
        for k in range(min_studies, n_studies + 1):
            subset_indices.extend(combinations(range(n_studies), k))
    else:
        # Random sampling of subsets
        subset_indices = []
        
        # Ensure we sample across different subset sizes
        for k in range(min_studies, n_studies + 1):
            k_possible = len(list(combinations(range(n_studies), k)))
            k_proportion = k_possible / total_possible
            k_samples = min(int(max_subsets * k_proportion), k_possible)
            
            if k_samples > 0:
                all_k_subsets = list(combinations(range(n_studies), k))
                sampled_k_subsets = random.sample(all_k_subsets, k_samples)
                subset_indices.extend(sampled_k_subsets)
    
    # Limit to max_subsets
    if len(subset_indices) > max_subsets:
        subset_indices = random.sample(subset_indices, max_subsets)
    
    # Generate results for each subset
    for indices in subset_indices:
        try:
            # Create subset
            subset_points = [points[i] for i in indices]
            
            # Fit random effects model
            model = RandomEffects(subset_points)
            result = model.fit()
            
            subset_results.append({
                'subset_indices': list(indices),
                'subset_size': len(indices),
                'pooled_effect': result.pooled_effect,
                'pooled_variance': result.pooled_variance,
                'tau2': result.tau2,
                'i_squared': result.i_squared,
                'q_statistic': result.q_statistic,
                'q_p_value': result.q_p_value,
                'confidence_interval': result.confidence_interval,
                'p_value': result.p_value
            })
            
        except Exception:
            # Skip subsets that fail to converge
            continue
    
    return subset_results


def gosh_outlier_detection(points: List[MetaPoint],
                          max_subsets: int = None,
                          outlier_threshold: float = 2.0) -> Dict[str, Any]:
    """Detect outlying studies using GOSH analysis.
    
    Args:
        points: List of MetaPoint objects
        max_subsets: Maximum number of subsets
        outlier_threshold: Standard deviation threshold for outlier detection
        
    Returns:
        Dictionary with outlier analysis results
    """
    if max_subsets is None:
        max_subsets = config.gosh_max_subsets
    
    # Generate subset results
    subset_results = generate_gosh_subsets(points, max_subsets)
    
    if not subset_results:
        raise PlottingError("No valid subsets for GOSH outlier detection")
    
    # Analyze effect size distribution
    effects = np.array([r['pooled_effect'] for r in subset_results])
    effect_mean = np.mean(effects)
    effect_std = np.std(effects)
    
    # Identify outlying subsets
    outlying_subsets = []
    for result in subset_results:
        z_score = abs(result['pooled_effect'] - effect_mean) / effect_std
        if z_score > outlier_threshold:
            outlying_subsets.append({
                'subset_indices': result['subset_indices'],
                'pooled_effect': result['pooled_effect'],
                'z_score': z_score,
                'deviation_direction': 'positive' if result['pooled_effect'] > effect_mean else 'negative'
            })
    
    # Count study appearances in outlying subsets
    study_outlier_counts = {i: 0 for i in range(len(points))}
    for outlying_subset in outlying_subsets:
        for study_idx in outlying_subset['subset_indices']:
            study_outlier_counts[study_idx] += 1
    
    # Identify potential outlier studies
    total_outlying_subsets = len(outlying_subsets)
    outlier_studies = []
    
    if total_outlying_subsets > 0:
        for study_idx, count in study_outlier_counts.items():
            outlier_proportion = count / total_outlying_subsets
            if outlier_proportion > 0.5:  # Appears in >50% of outlying subsets
                outlier_studies.append({
                    'study_index': study_idx,
                    'label': points[study_idx].label or f"Study {study_idx + 1}",
                    'outlier_subset_count': count,
                    'outlier_proportion': outlier_proportion,
                    'effect': points[study_idx].effect
                })
    
    return {
        'outlying_subsets': outlying_subsets,
        'outlier_studies': outlier_studies,
        'n_outlying_subsets': len(outlying_subsets),
        'n_total_subsets': len(subset_results),
        'outlier_threshold': outlier_threshold,
        'effect_distribution': {
            'mean': effect_mean,
            'std': effect_std,
            'min': np.min(effects),
            'max': np.max(effects),
            'range': np.max(effects) - np.min(effects)
        },
        'study_outlier_counts': study_outlier_counts
    }