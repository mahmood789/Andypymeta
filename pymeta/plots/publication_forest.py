"""Publication-quality forest plots with style system."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional, Dict, Any, Tuple
from scipy import stats

from ..typing import MetaPoint, MetaResults
from ..registries import register_plot
from ..errors import PlottingError
from ..config import config


@register_plot("forest")
def forest_plot(points: List[MetaPoint],
               results: MetaResults,
               title: Optional[str] = None,
               study_labels: Optional[List[str]] = None,
               effect_label: str = "Effect Size",
               style: Optional[str] = None,
               show_weights: bool = True,
               show_confidence_intervals: bool = True,
               show_prediction_interval: bool = False,
               prediction_interval: Optional[Tuple[float, float]] = None,
               figsize: Optional[Tuple[float, float]] = None,
               save_path: Optional[str] = None) -> plt.Figure:
    """Create publication-quality forest plot.
    
    Args:
        points: List of MetaPoint objects
        results: MetaResults from meta-analysis
        title: Plot title
        study_labels: Custom study labels
        effect_label: Label for effect size axis
        style: Plot style name
        show_weights: Whether to show study weights
        show_confidence_intervals: Whether to show study CIs
        show_prediction_interval: Whether to show prediction interval
        prediction_interval: Custom prediction interval bounds
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
        
        # Prepare data
        n_studies = len(points)
        study_names = study_labels or [p.label or f"Study {i+1}" for i, p in enumerate(points)]
        
        # Calculate study-level confidence intervals
        alpha = 1 - config.confidence_level
        z_critical = stats.norm.ppf(1 - alpha / 2)
        
        study_effects = [p.effect for p in points]
        study_ses = [np.sqrt(p.variance) for p in points]
        study_weights = [p.weight for p in points]
        
        study_ci_lower = [eff - z_critical * se for eff, se in zip(study_effects, study_ses)]
        study_ci_upper = [eff + z_critical * se for eff, se in zip(study_effects, study_ses)]
        
        # Calculate relative weights for symbol sizing
        total_weight = sum(study_weights)
        relative_weights = [100 * w / total_weight for w in study_weights]
        
        # Y positions for studies (reverse order for top-to-bottom display)
        y_positions = list(range(n_studies + 2))  # Extra space for overall result
        study_y_positions = y_positions[1:-1][::-1]  # Reverse for top-to-bottom
        overall_y_position = y_positions[0]
        
        # Plot study-level results
        colors = plot_style.color_palette
        study_color = colors[0] if colors else 'blue'
        
        for i, (y_pos, effect, ci_lower, ci_upper, weight) in enumerate(zip(
                study_y_positions, study_effects, study_ci_lower, study_ci_upper, relative_weights)):
            
            # Symbol size based on weight
            symbol_size = 20 + weight * 3  # Base size + weight scaling
            
            # Plot point estimate
            ax.scatter(effect, y_pos, s=symbol_size, c=study_color, 
                      marker='s', alpha=0.7, edgecolors='black', linewidths=0.5)
            
            # Plot confidence interval
            if show_confidence_intervals:
                ax.plot([ci_lower, ci_upper], [y_pos, y_pos], 
                       color=study_color, linewidth=1.5, alpha=0.7)
                # CI caps
                cap_height = 0.1
                ax.plot([ci_lower, ci_lower], [y_pos - cap_height, y_pos + cap_height],
                       color=study_color, linewidth=1.5, alpha=0.7)
                ax.plot([ci_upper, ci_upper], [y_pos - cap_height, y_pos + cap_height],
                       color=study_color, linewidth=1.5, alpha=0.7)
        
        # Plot overall result
        overall_color = colors[1] if len(colors) > 1 else 'red'
        overall_size = 150  # Larger diamond
        
        # Diamond shape for overall estimate
        diamond_width = 0.3
        diamond_x = results.pooled_effect
        diamond_y = overall_y_position
        
        # Diamond coordinates
        diamond = patches.RegularPolygon(
            (diamond_x, diamond_y), 4, radius=diamond_width,
            orientation=np.pi/4, facecolor=overall_color, 
            edgecolor='black', alpha=0.8, linewidth=2
        )
        ax.add_patch(diamond)
        
        # Overall confidence interval
        ax.plot([results.confidence_interval[0], results.confidence_interval[1]], 
               [diamond_y, diamond_y], color=overall_color, linewidth=3, alpha=0.8)
        
        # Add prediction interval if requested
        if show_prediction_interval and prediction_interval:
            pred_y = diamond_y - 0.5
            ax.plot([prediction_interval[0], prediction_interval[1]], 
                   [pred_y, pred_y], color='gray', linewidth=2, linestyle='--',
                   alpha=0.6, label='Prediction Interval')
        
        # Add vertical line at null effect
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Customize y-axis
        y_labels = study_names + ['Overall']
        ax.set_yticks(y_positions[:-1])
        ax.set_yticklabels(y_labels[::-1])  # Reverse to match positions
        
        # Add study information on the right
        if show_weights:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(y_positions[:-1])
            
            # Format weight labels
            weight_labels = [f"{w:.1f}%" for w in relative_weights[::-1]]  # Reverse order
            weight_labels.append(f"100.0%")  # Overall
            
            ax2.set_yticklabels(weight_labels)
            ax2.set_ylabel("Weight (%)", fontsize=plot_style.font_size)
            ax2.tick_params(axis='y', labelsize=plot_style.font_size - 2)
        
        # Labels and title
        ax.set_xlabel(effect_label, fontsize=plot_style.font_size)
        ax.set_ylabel("Study", fontsize=plot_style.font_size)
        
        if title:
            ax.set_title(title, fontsize=plot_style.font_size + 2, fontweight='bold')
        else:
            ax.set_title(f"Forest Plot - {results.method}", 
                        fontsize=plot_style.font_size + 2, fontweight='bold')
        
        # Add statistics text box
        stats_text = _create_stats_text(results)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
               fontsize=plot_style.font_size - 2)
        
        # Grid
        if plot_style.grid:
            ax.grid(True, alpha=plot_style.grid_alpha, axis='x')
        
        # Spine styling
        if plot_style.spine_style == 'minimal':
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        elif plot_style.spine_style == 'publication':
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=plot_style.dpi, bbox_inches='tight')
        
        return fig
        
    except Exception as e:
        raise PlottingError(f"Forest plot creation failed: {e}")


def _create_stats_text(results: MetaResults) -> str:
    """Create statistics text for forest plot."""
    lines = [
        f"Effect: {results.pooled_effect:.3f} ({results.confidence_interval[0]:.3f}, {results.confidence_interval[1]:.3f})",
        f"Z: {results.z_score:.2f}, p: {results.p_value:.4f}",
        f"I²: {results.i_squared:.1f}%",
        f"τ²: {results.tau2:.4f}",
        f"Studies: {results.n_studies}"
    ]
    return "\n".join(lines)


@register_plot("forest_subgroup")
def forest_plot_subgroup(point_groups: Dict[str, List[MetaPoint]],
                        result_groups: Dict[str, MetaResults],
                        overall_result: Optional[MetaResults] = None,
                        title: Optional[str] = None,
                        style: Optional[str] = None,
                        figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
    """Create forest plot with subgroup analysis.
    
    Args:
        point_groups: Dictionary of subgroup name -> MetaPoint lists
        result_groups: Dictionary of subgroup name -> MetaResults
        overall_result: Overall meta-analysis result across all groups
        title: Plot title
        style: Plot style name
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Get plot style
        plot_style = config.get_plot_style(style)
        if figsize is None:
            # Adjust height for subgroups
            n_total_studies = sum(len(points) for points in point_groups.values())
            n_subgroups = len(point_groups)
            height = max(8, (n_total_studies + n_subgroups * 2) * 0.4)
            figsize = (plot_style.figure_size[0], height)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=plot_style.dpi)
        
        colors = plot_style.color_palette
        y_position = 0
        y_labels = []
        y_positions = []
        
        # Plot each subgroup
        for group_idx, (group_name, points) in enumerate(point_groups.items()):
            result = result_groups[group_name]
            group_color = colors[group_idx % len(colors)]
            
            # Subgroup header
            y_position += 1
            y_positions.append(y_position)
            y_labels.append(f"{group_name} (n={len(points)})")
            
            # Plot studies in this subgroup
            for point in points:
                y_position += 1
                y_positions.append(y_position)
                
                # Study effect and CI
                se = np.sqrt(point.variance)
                ci_lower = point.effect - 1.96 * se
                ci_upper = point.effect + 1.96 * se
                
                # Plot study
                ax.scatter(point.effect, y_position, s=50, c=group_color,
                          marker='s', alpha=0.7, edgecolors='black')
                ax.plot([ci_lower, ci_upper], [y_position, y_position],
                       color=group_color, linewidth=1.5, alpha=0.7)
                
                study_label = point.label or f"Study {len(y_labels)}"
                y_labels.append(f"  {study_label}")
            
            # Subgroup summary
            y_position += 1
            y_positions.append(y_position)
            y_labels.append(f"  Subtotal")
            
            # Plot subgroup diamond
            diamond = patches.RegularPolygon(
                (result.pooled_effect, y_position), 4, radius=0.2,
                orientation=np.pi/4, facecolor=group_color,
                edgecolor='black', alpha=0.8
            )
            ax.add_patch(diamond)
            ax.plot([result.confidence_interval[0], result.confidence_interval[1]],
                   [y_position, y_position], color=group_color, linewidth=2)
            
            # Add spacing
            y_position += 1
        
        # Overall result if provided
        if overall_result:
            y_position += 1
            y_positions.append(y_position)
            y_labels.append("Overall")
            
            overall_diamond = patches.RegularPolygon(
                (overall_result.pooled_effect, y_position), 4, radius=0.3,
                orientation=np.pi/4, facecolor='red',
                edgecolor='black', alpha=0.8, linewidth=2
            )
            ax.add_patch(overall_diamond)
            ax.plot([overall_result.confidence_interval[0], overall_result.confidence_interval[1]],
                   [y_position, y_position], color='red', linewidth=3)
        
        # Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()  # Top to bottom
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel("Effect Size", fontsize=plot_style.font_size)
        
        if title:
            ax.set_title(title, fontsize=plot_style.font_size + 2, fontweight='bold')
        else:
            ax.set_title("Subgroup Forest Plot", fontsize=plot_style.font_size + 2, fontweight='bold')
        
        if plot_style.grid:
            ax.grid(True, alpha=plot_style.grid_alpha, axis='x')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        raise PlottingError(f"Subgroup forest plot creation failed: {e}")