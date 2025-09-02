#!/usr/bin/env python3
"""
01_quickstart_re.py - Quick start example for random-effects meta-analysis

This example demonstrates basic random-effects meta-analysis using PyMeta.
"""

import numpy as np
import pandas as pd
import pymeta as pm

def main():
    """Run quickstart random-effects meta-analysis example."""
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'study': [f'Study_{i}' for i in range(1, 11)],
        'effect_size': np.random.normal(0.5, 0.3, 10),
        'variance': np.random.uniform(0.02, 0.15, 10),
        'sample_size': np.random.randint(50, 200, 10),
        'year': np.random.randint(2010, 2024, 10)
    })
    
    print("Sample data:")
    print(data.head())
    print()
    
    # Perform random-effects meta-analysis
    print("Performing random-effects meta-analysis...")
    result = pm.meta_analysis(
        data, 
        model='random',
        tau2_method='dl'
    )
    
    print(f"Overall effect size: {result.overall_effect:.3f}")
    print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    print(f"Tau-squared: {result.tau_squared:.3f}")
    print(f"I-squared: {result.i_squared:.1f}%")
    print()
    
    # Create forest plot
    print("Creating forest plot...")
    forest_plot = pm.forest_plot(result, title="Random-Effects Meta-Analysis")
    
    # Save plot
    forest_plot.savefig('quickstart_forest_plot.png', dpi=300, bbox_inches='tight')
    print("Forest plot saved as 'quickstart_forest_plot.png'")
    
    # Test for heterogeneity
    print(f"Q-statistic: {result.q_statistic:.2f}")
    print(f"Q p-value: {result.q_pvalue:.3f}")
    
    if result.q_pvalue < 0.05:
        print("Significant heterogeneity detected (p < 0.05)")
    else:
        print("No significant heterogeneity (p >= 0.05)")

if __name__ == "__main__":
    main()