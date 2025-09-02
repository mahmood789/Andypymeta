"""Visualization tools for meta-analysis."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Union


def forest_plot(effect_sizes, standard_errors, study_labels=None, 
                pooled_effect=None, pooled_se=None, 
                title="Forest Plot", figsize=(10, 8), save_path=None):
    """
    Create a forest plot for meta-analysis results.
    
    Parameters
    ----------
    effect_sizes : array_like
        Study effect sizes
    standard_errors : array_like
        Standard errors of effect sizes
    study_labels : list, optional
        Labels for studies
    pooled_effect : float, optional
        Pooled effect size
    pooled_se : float, optional
        Standard error of pooled effect
    title : str, default "Forest Plot"
        Plot title
    figsize : tuple, default (10, 8)
        Figure size
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    effect_sizes = np.asarray(effect_sizes)
    standard_errors = np.asarray(standard_errors)
    n_studies = len(effect_sizes)
    
    if study_labels is None:
        study_labels = [f"Study {i+1}" for i in range(n_studies)]
    
    # Calculate confidence intervals
    ci_lower = effect_sizes - 1.96 * standard_errors
    ci_upper = effect_sizes + 1.96 * standard_errors
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual studies
    y_positions = np.arange(n_studies)
    ax.errorbar(effect_sizes, y_positions, xerr=1.96*standard_errors, 
                fmt='s', markersize=8, capsize=5, capthick=2)
    
    # Add study labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(study_labels)
    
    # Add pooled estimate if provided
    if pooled_effect is not None and pooled_se is not None:
        pooled_ci_lower = pooled_effect - 1.96 * pooled_se
        pooled_ci_upper = pooled_effect + 1.96 * pooled_se
        
        ax.errorbar(pooled_effect, -1, xerr=1.96*pooled_se, 
                   fmt='D', markersize=12, capsize=8, capthick=3, 
                   color='red', label='Pooled')
        ax.set_yticks(list(y_positions) + [-1])
        ax.set_yticklabels(list(study_labels) + ['Pooled'])
    
    # Add vertical line at null effect
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Effect Size')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Studies from top to bottom
    
    if pooled_effect is not None:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def funnel_plot(effect_sizes, standard_errors, title="Funnel Plot", 
                figsize=(8, 6), save_path=None, show_contours=True):
    """
    Create a funnel plot for assessing publication bias.
    
    Parameters
    ----------
    effect_sizes : array_like
        Study effect sizes
    standard_errors : array_like
        Standard errors of effect sizes
    title : str, default "Funnel Plot"
        Plot title
    figsize : tuple, default (8, 6)
        Figure size
    save_path : str, optional
        Path to save the plot
    show_contours : bool, default True
        Whether to show significance contours
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    effect_sizes = np.asarray(effect_sizes)
    standard_errors = np.asarray(standard_errors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot studies
    ax.scatter(effect_sizes, standard_errors, alpha=0.7, s=50)
    
    # Add significance contours if requested
    if show_contours:
        # Create grid for contours
        x_range = np.linspace(effect_sizes.min() - 1, effect_sizes.max() + 1, 100)
        y_range = np.linspace(0, standard_errors.max() * 1.1, 100)
        
        # Add 95% significance contours (effect = ±1.96 * SE)
        se_line = np.linspace(0, standard_errors.max() * 1.1, 100)
        ax.plot(1.96 * se_line, se_line, '--', color='gray', alpha=0.5, label='95% CI')
        ax.plot(-1.96 * se_line, se_line, '--', color='gray', alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Standard Error')
    ax.set_title(title)
    ax.invert_yaxis()  # Smaller SE at top
    ax.grid(True, alpha=0.3)
    
    if show_contours:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_demo_plot():
    """Create a demo plot for testing purposes."""
    np.random.seed(42)
    n_studies = 10
    true_effect = 0.5
    
    # Simulate some meta-analysis data
    study_sizes = np.random.randint(50, 500, n_studies)
    standard_errors = 1 / np.sqrt(study_sizes)
    effect_sizes = np.random.normal(true_effect, standard_errors)
    
    fig = forest_plot(effect_sizes, standard_errors, 
                     pooled_effect=true_effect, pooled_se=0.1,
                     title="Demo Forest Plot")
    return fig