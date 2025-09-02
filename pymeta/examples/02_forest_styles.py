#!/usr/bin/env python3
"""
02_forest_styles.py - Demonstration of forest plot styles and customization

This example shows different forest plot styles available in PyMeta.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymeta as pm

def create_sample_data():
    """Create sample meta-analysis data."""
    np.random.seed(123)
    
    studies = [
        'Anderson et al. (2020)',
        'Brown et al. (2021)', 
        'Chen et al. (2019)',
        'Davis et al. (2022)',
        'Evans et al. (2020)',
        'Foster et al. (2021)',
        'Garcia et al. (2023)',
        'Harris et al. (2022)'
    ]
    
    data = pd.DataFrame({
        'study': studies,
        'effect_size': np.random.normal(0.4, 0.2, len(studies)),
        'variance': np.random.uniform(0.01, 0.08, len(studies)),
        'sample_size': np.random.randint(80, 300, len(studies)),
        'intervention': np.random.choice(['Treatment A', 'Treatment B'], len(studies))
    })
    
    return data

def demonstrate_forest_styles():
    """Demonstrate different forest plot styles."""
    
    data = create_sample_data()
    
    # Perform meta-analysis
    result = pm.meta_analysis(data, model='random', tau2_method='reml')
    
    # Create different styles
    styles = ['classic', 'modern', 'minimal', 'publication']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, style in enumerate(styles):
        ax = axes[i]
        
        forest_plot = pm.forest_plot(
            result,
            style=style,
            title=f'Forest Plot - {style.title()} Style',
            ax=ax
        )
        
    plt.tight_layout()
    plt.savefig('forest_plot_styles.png', dpi=300, bbox_inches='tight')
    print("Forest plot styles saved as 'forest_plot_styles.png'")

def demonstrate_customization():
    """Demonstrate forest plot customization options."""
    
    data = create_sample_data()
    result = pm.meta_analysis(data, model='random')
    
    # Customized forest plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    forest_plot = pm.forest_plot(
        result,
        ax=ax,
        # Customization options
        square_size_scale=2.0,
        diamond_color='red',
        line_color='blue',
        text_size=12,
        show_weights=True,
        show_confidence_intervals=True,
        title='Customized Forest Plot',
        xlabel='Effect Size (Cohen\'s d)',
        study_labels='left',
        overall_label='Overall Effect (Random-Effects)'
    )
    
    plt.savefig('forest_plot_custom.png', dpi=300, bbox_inches='tight')
    print("Customized forest plot saved as 'forest_plot_custom.png'")

def demonstrate_subgroup_forest():
    """Demonstrate subgroup forest plots."""
    
    data = create_sample_data()
    
    # Subgroup meta-analysis
    result = pm.subgroup_analysis(
        data, 
        grouping_var='intervention',
        model='random'
    )
    
    # Subgroup forest plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    forest_plot = pm.subgroup_forest_plot(
        result,
        ax=ax,
        title='Subgroup Forest Plot by Intervention Type',
        show_overall=True,
        show_subgroup_overall=True
    )
    
    plt.savefig('forest_plot_subgroups.png', dpi=300, bbox_inches='tight')
    print("Subgroup forest plot saved as 'forest_plot_subgroups.png'")

def main():
    """Run all forest plot examples."""
    
    print("PyMeta Forest Plot Styles Example")
    print("=" * 40)
    
    print("\n1. Demonstrating different forest plot styles...")
    demonstrate_forest_styles()
    
    print("\n2. Demonstrating customization options...")
    demonstrate_customization()
    
    print("\n3. Demonstrating subgroup forest plots...")
    demonstrate_subgroup_forest()
    
    print("\nAll examples completed!")
    print("Check the generated PNG files for results.")

if __name__ == "__main__":
    main()