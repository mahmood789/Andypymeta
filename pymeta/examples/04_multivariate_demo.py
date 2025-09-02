#!/usr/bin/env python3
"""
04_multivariate_demo.py - Multivariate meta-analysis demonstration

This example shows how to conduct multivariate meta-analysis with multiple outcomes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymeta as pm

def create_multivariate_data():
    """Create sample multivariate meta-analysis data."""
    np.random.seed(789)
    
    # Study characteristics
    studies = [f'Study_{i}' for i in range(1, 16)]
    n_studies = len(studies)
    
    # Two correlated outcomes
    # True correlation between outcomes
    true_correlation = 0.6
    
    # Generate correlated effect sizes
    mean_effects = [0.4, 0.3]  # True effects for outcome 1 and 2
    cov_matrix = np.array([[0.2, 0.12], [0.12, 0.15]])  # Covariance matrix
    
    effects = np.random.multivariate_normal(mean_effects, cov_matrix, n_studies)
    
    # Generate sampling variances (inversely related to sample size)
    sample_sizes = np.random.randint(50, 300, n_studies)
    base_variance = 1 / np.sqrt(sample_sizes)
    
    # Add some random variation to variances
    var1 = base_variance * np.random.uniform(0.8, 1.2, n_studies)
    var2 = base_variance * np.random.uniform(0.8, 1.2, n_studies)
    
    # Calculate sampling correlation (typically lower than true correlation)
    sampling_correlations = np.random.uniform(0.3, 0.7, n_studies)
    
    # Create long format data
    data_rows = []
    for i, study in enumerate(studies):
        # Outcome 1
        data_rows.append({
            'study': study,
            'outcome': 'Outcome_1',
            'effect_size': effects[i, 0],
            'variance': var1[i]**2,
            'sample_size': sample_sizes[i],
            'outcome_type': 'Primary'
        })
        
        # Outcome 2
        data_rows.append({
            'study': study,
            'outcome': 'Outcome_2', 
            'effect_size': effects[i, 1],
            'variance': var2[i]**2,
            'sample_size': sample_sizes[i],
            'outcome_type': 'Secondary'
        })
    
    data = pd.DataFrame(data_rows)
    
    # Add correlation matrix information
    correlation_data = []
    for i, study in enumerate(studies):
        correlation_data.append({
            'study': study,
            'outcome1': 'Outcome_1',
            'outcome2': 'Outcome_2',
            'correlation': sampling_correlations[i]
        })
    
    correlations = pd.DataFrame(correlation_data)
    
    return data, correlations

def univariate_vs_multivariate():
    """Compare univariate and multivariate meta-analysis results."""
    
    data, correlations = create_multivariate_data()
    
    print("Univariate vs Multivariate Meta-Analysis")
    print("=" * 45)
    print(f"Number of studies: {len(data['study'].unique())}")
    print(f"Number of outcomes per study: {len(data['outcome'].unique())}")
    print()
    
    # Separate data by outcome
    outcome1_data = data[data['outcome'] == 'Outcome_1'].copy()
    outcome2_data = data[data['outcome'] == 'Outcome_2'].copy()
    
    # Univariate meta-analyses
    print("Univariate Meta-Analyses (ignoring correlation):")
    print("-" * 50)
    
    result1_uni = pm.meta_analysis(outcome1_data, model='random')
    print(f"Outcome 1:")
    print(f"  Effect: {result1_uni.overall_effect:.3f}")
    print(f"  95% CI: [{result1_uni.ci_lower:.3f}, {result1_uni.ci_upper:.3f}]")
    print(f"  Tau²: {result1_uni.tau_squared:.3f}")
    print()
    
    result2_uni = pm.meta_analysis(outcome2_data, model='random')
    print(f"Outcome 2:")
    print(f"  Effect: {result2_uni.overall_effect:.3f}")
    print(f"  95% CI: [{result2_uni.ci_lower:.3f}, {result2_uni.ci_upper:.3f}]")
    print(f"  Tau²: {result2_uni.tau_squared:.3f}")
    print()
    
    # Multivariate meta-analysis
    print("Multivariate Meta-Analysis (accounting for correlation):")
    print("-" * 55)
    
    mv_result = pm.multivariate_meta_analysis(
        data, 
        correlation_matrix=correlations,
        model='random'
    )
    
    print(f"Outcome 1:")
    print(f"  Effect: {mv_result.effects['Outcome_1']:.3f}")
    print(f"  95% CI: [{mv_result.ci_lower['Outcome_1']:.3f}, {mv_result.ci_upper['Outcome_1']:.3f}]")
    print()
    
    print(f"Outcome 2:")
    print(f"  Effect: {mv_result.effects['Outcome_2']:.3f}")
    print(f"  95% CI: [{mv_result.ci_lower['Outcome_2']:.3f}, {mv_result.ci_upper['Outcome_2']:.3f}]")
    print()
    
    print(f"Between-study correlation: {mv_result.tau_correlation:.3f}")
    print(f"Multivariate Q-statistic: {mv_result.q_statistic:.2f}")
    print(f"Multivariate Q p-value: {mv_result.q_pvalue:.3f}")
    
    return result1_uni, result2_uni, mv_result

def visualize_multivariate_results(result1_uni, result2_uni, mv_result):
    """Create visualizations for multivariate results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Forest plots for each outcome
    pm.forest_plot(result1_uni, ax=axes[0, 0], title='Outcome 1 - Univariate')
    pm.forest_plot(result2_uni, ax=axes[0, 1], title='Outcome 2 - Univariate')
    
    # Multivariate forest plot
    pm.multivariate_forest_plot(mv_result, ax=axes[1, 0], title='Multivariate Results')
    
    # Correlation plot
    pm.outcome_correlation_plot(mv_result, ax=axes[1, 1], title='Between-Study Correlations')
    
    plt.tight_layout()
    plt.savefig('multivariate_analysis.png', dpi=300, bbox_inches='tight')
    print("\nMultivariate analysis plots saved as 'multivariate_analysis.png'")

def network_comparison():
    """Demonstrate comparison with network meta-analysis approach."""
    
    data, correlations = create_multivariate_data()
    
    print("\nNetwork Meta-Analysis Comparison:")
    print("-" * 35)
    
    # Note: This would be a placeholder for future network meta-analysis functionality
    print("Network meta-analysis functionality planned for future release.")
    print("Current multivariate approach handles multiple outcomes per study.")
    print("Network approach would handle multiple treatments across studies.")

def dose_response_example():
    """Example of dose-response multivariate meta-analysis."""
    
    print("\nDose-Response Meta-Analysis:")
    print("-" * 30)
    
    # Generate dose-response data
    np.random.seed(999)
    
    studies = [f'Study_{i}' for i in range(1, 11)]
    doses = [10, 20, 50, 100]  # mg/day
    
    dose_data = []
    for study in studies:
        # Each study tests multiple doses
        n_doses = np.random.randint(2, 5)  # Each study tests 2-4 doses
        study_doses = np.random.choice(doses, n_doses, replace=False)
        
        for dose in study_doses:
            # Effect increases with log-dose
            log_dose_effect = 0.2 * np.log(dose) + np.random.normal(0, 0.1)
            
            dose_data.append({
                'study': study,
                'dose': dose,
                'log_dose': np.log(dose),
                'effect_size': log_dose_effect,
                'variance': np.random.uniform(0.01, 0.05),
                'sample_size': np.random.randint(30, 100)
            })
    
    dose_df = pd.DataFrame(dose_data)
    
    # Dose-response meta-regression
    dr_result = pm.dose_response_analysis(
        dose_df,
        dose_variable='log_dose',
        model='random'
    )
    
    print(f"Dose-response slope: {dr_result.slope:.3f}")
    print(f"95% CI: [{dr_result.slope_ci_lower:.3f}, {dr_result.slope_ci_upper:.3f}]")
    print(f"p-value: {dr_result.slope_pvalue:.3f}")
    
    # Plot dose-response
    fig, ax = plt.subplots(figsize=(10, 6))
    pm.dose_response_plot(dr_result, ax=ax, title='Dose-Response Meta-Analysis')
    plt.savefig('dose_response.png', dpi=300, bbox_inches='tight')
    print("Dose-response plot saved as 'dose_response.png'")

def main():
    """Run multivariate meta-analysis examples."""
    
    print("PyMeta Multivariate Meta-Analysis Examples")
    print("=" * 45)
    
    # Compare univariate vs multivariate
    result1, result2, mv_result = univariate_vs_multivariate()
    
    # Visualize results
    visualize_multivariate_results(result1, result2, mv_result)
    
    # Network comparison
    network_comparison()
    
    # Dose-response example
    dose_response_example()
    
    print("\nAll multivariate examples completed!")
    print("Check the generated PNG files for results.")

if __name__ == "__main__":
    main()