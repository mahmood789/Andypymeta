#!/usr/bin/env python3
"""
05_glmm_binomial_poisson.py - GLMM for binomial and Poisson data

This example demonstrates Generalized Linear Mixed Models for exact likelihood 
meta-analysis of binomial and Poisson outcomes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymeta as pm

def create_binomial_data():
    """Create sample binomial (binary outcome) data."""
    np.random.seed(101)
    
    studies = [f'Study_{i}' for i in range(1, 21)]
    
    data = []
    for study in studies:
        # Treatment group
        treat_n = np.random.randint(50, 200)
        treat_p = np.random.beta(2, 5)  # Base probability
        treat_events = np.random.binomial(treat_n, treat_p)
        
        # Control group  
        control_n = np.random.randint(50, 200)
        control_p = treat_p * np.random.uniform(0.6, 0.9)  # Lower probability
        control_events = np.random.binomial(control_n, control_p)
        
        data.append({
            'study': study,
            'treat_events': treat_events,
            'treat_total': treat_n,
            'control_events': control_events,
            'control_total': control_n,
            'year': np.random.randint(2010, 2024),
            'quality': np.random.choice(['High', 'Medium', 'Low'])
        })
    
    return pd.DataFrame(data)

def create_poisson_data():
    """Create sample Poisson (count outcome) data."""
    np.random.seed(202)
    
    studies = [f'Study_{i}' for i in range(1, 16)]
    
    data = []
    for study in studies:
        # Treatment group
        treat_time = np.random.uniform(100, 1000)  # Person-time
        treat_rate = np.random.gamma(2, 0.1)  # Events per person-time
        treat_events = np.random.poisson(treat_rate * treat_time)
        
        # Control group
        control_time = np.random.uniform(100, 1000)
        control_rate = treat_rate * np.random.uniform(1.2, 2.0)  # Higher rate
        control_events = np.random.poisson(control_rate * control_time)
        
        data.append({
            'study': study,
            'treat_events': treat_events,
            'treat_time': treat_time,
            'control_events': control_events,
            'control_time': control_time,
            'setting': np.random.choice(['Hospital', 'Community', 'Clinic']),
            'duration': np.random.randint(6, 36)  # months
        })
    
    return pd.DataFrame(data)

def binomial_glmm_analysis():
    """Demonstrate GLMM analysis for binomial data."""
    
    data = create_binomial_data()
    
    print("Binomial GLMM Analysis")
    print("=" * 25)
    print(f"Number of studies: {len(data)}")
    print()
    
    # Traditional approach with normal approximation
    print("1. Traditional approach (normal approximation):")
    print("-" * 50)
    
    # Calculate log odds ratios
    traditional_data = pm.calculate_log_odds_ratio(data)
    traditional_result = pm.meta_analysis(traditional_data, model='random')
    
    print(f"Pooled log OR: {traditional_result.overall_effect:.3f}")
    print(f"Pooled OR: {np.exp(traditional_result.overall_effect):.3f}")
    print(f"95% CI: [{np.exp(traditional_result.ci_lower):.3f}, {np.exp(traditional_result.ci_upper):.3f}]")
    print(f"Tau²: {traditional_result.tau_squared:.3f}")
    print()
    
    # GLMM approach with exact likelihood
    print("2. GLMM approach (exact likelihood):")
    print("-" * 40)
    
    # Binomial GLMM with logit link
    glmm_logit_result = pm.binomial_glmm(
        data,
        link='logit',
        method='ml'
    )
    
    print(f"GLMM log OR (logit): {glmm_logit_result.overall_effect:.3f}")
    print(f"GLMM OR (logit): {np.exp(glmm_logit_result.overall_effect):.3f}")
    print(f"95% CI: [{np.exp(glmm_logit_result.ci_lower):.3f}, {np.exp(glmm_logit_result.ci_upper):.3f}]")
    print(f"Tau²: {glmm_logit_result.tau_squared:.3f}")
    print()
    
    # Binomial GLMM with complementary log-log link
    glmm_cloglog_result = pm.binomial_glmm(
        data,
        link='cloglog',
        method='ml'
    )
    
    print(f"GLMM log OR (cloglog): {glmm_cloglog_result.overall_effect:.3f}")
    print(f"GLMM OR (cloglog): {np.exp(glmm_cloglog_result.overall_effect):.3f}")
    print(f"95% CI: [{np.exp(glmm_cloglog_result.ci_lower):.3f}, {np.exp(glmm_cloglog_result.ci_upper):.3f}]")
    print(f"Tau²: {glmm_cloglog_result.tau_squared:.3f}")
    print()
    
    return traditional_result, glmm_logit_result, glmm_cloglog_result

def poisson_glmm_analysis():
    """Demonstrate GLMM analysis for Poisson data."""
    
    data = create_poisson_data()
    
    print("Poisson GLMM Analysis")
    print("=" * 22)
    print(f"Number of studies: {len(data)}")
    print()
    
    # Traditional approach with normal approximation
    print("1. Traditional approach (normal approximation):")
    print("-" * 50)
    
    # Calculate log rate ratios
    traditional_data = pm.calculate_log_rate_ratio(data)
    traditional_result = pm.meta_analysis(traditional_data, model='random')
    
    print(f"Pooled log RR: {traditional_result.overall_effect:.3f}")
    print(f"Pooled RR: {np.exp(traditional_result.overall_effect):.3f}")
    print(f"95% CI: [{np.exp(traditional_result.ci_lower):.3f}, {np.exp(traditional_result.ci_upper):.3f}]")
    print(f"Tau²: {traditional_result.tau_squared:.3f}")
    print()
    
    # GLMM approach with exact likelihood
    print("2. GLMM approach (exact likelihood):")
    print("-" * 40)
    
    glmm_result = pm.poisson_glmm(
        data,
        link='log',
        method='ml'
    )
    
    print(f"GLMM log RR: {glmm_result.overall_effect:.3f}")
    print(f"GLMM RR: {np.exp(glmm_result.overall_effect):.3f}")
    print(f"95% CI: [{np.exp(glmm_result.ci_lower):.3f}, {np.exp(glmm_result.ci_upper):.3f}]")
    print(f"Tau²: {glmm_result.tau_squared:.3f}")
    print()
    
    return traditional_result, glmm_result

def compare_methods():
    """Compare traditional vs GLMM approaches."""
    
    print("Method Comparison Summary")
    print("=" * 28)
    
    print("Traditional approach advantages:")
    print("- Fast computation")
    print("- Well-established methods")
    print("- Wide software support")
    print()
    
    print("Traditional approach limitations:")
    print("- Normal approximation may be poor with rare events")
    print("- Cannot handle zero events in one arm naturally")
    print("- May give biased results with small samples")
    print()
    
    print("GLMM approach advantages:")
    print("- Exact likelihood, no normal approximation")
    print("- Handles zero events naturally")
    print("- Better performance with rare events")
    print("- Can include study-level covariates easily")
    print()
    
    print("GLMM approach limitations:")
    print("- More computationally intensive")
    print("- May have convergence issues")
    print("- Less familiar to some researchers")
    print()

def visualize_results(binomial_results, poisson_results):
    """Create visualizations comparing methods."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Binomial results comparison
    traditional_bin, glmm_logit, glmm_cloglog = binomial_results
    
    # Forest plots
    pm.forest_plot(traditional_bin, ax=axes[0, 0], title='Binomial: Traditional')
    pm.forest_plot(glmm_logit, ax=axes[0, 1], title='Binomial: GLMM Logit')
    
    # Poisson results comparison  
    traditional_pois, glmm_pois = poisson_results
    
    pm.forest_plot(traditional_pois, ax=axes[1, 0], title='Poisson: Traditional')
    pm.forest_plot(glmm_pois, ax=axes[1, 1], title='Poisson: GLMM')
    
    plt.tight_layout()
    plt.savefig('glmm_comparison.png', dpi=300, bbox_inches='tight')
    print("GLMM comparison plots saved as 'glmm_comparison.png'")

def special_cases():
    """Demonstrate handling of special cases with GLMM."""
    
    print("\nSpecial Cases with GLMM")
    print("=" * 25)
    
    # Data with zero events in one arm
    print("1. Handling zero events:")
    print("-" * 25)
    
    zero_data = pd.DataFrame({
        'study': ['A', 'B', 'C', 'D'],
        'treat_events': [0, 2, 0, 5],
        'treat_total': [50, 60, 40, 80],
        'control_events': [3, 8, 1, 12],
        'control_total': [55, 65, 45, 85]
    })
    
    print("Traditional approach (with continuity correction):")
    traditional_zero = pm.meta_analysis_with_zeros(zero_data, correction=0.5)
    print(f"OR: {np.exp(traditional_zero.overall_effect):.3f}")
    
    print("\nGLMM approach (exact likelihood):")
    glmm_zero = pm.binomial_glmm(zero_data, link='logit')
    print(f"OR: {np.exp(glmm_zero.overall_effect):.3f}")
    print()
    
    # Rare events
    print("2. Rare events scenario:")
    print("-" * 25)
    print("GLMM particularly beneficial when:")
    print("- Event rates < 1%")
    print("- Small sample sizes")
    print("- Many studies with zero events")
    print("- Need to avoid continuity corrections")

def main():
    """Run GLMM examples for binomial and Poisson data."""
    
    print("PyMeta GLMM Examples")
    print("=" * 20)
    print("Generalized Linear Mixed Models for exact likelihood meta-analysis")
    print()
    
    # Binomial GLMM analysis
    binomial_results = binomial_glmm_analysis()
    
    print("\n" + "="*60 + "\n")
    
    # Poisson GLMM analysis
    poisson_results = poisson_glmm_analysis()
    
    print("\n" + "="*60 + "\n")
    
    # Compare methods
    compare_methods()
    
    # Visualize results
    visualize_results(binomial_results, poisson_results)
    
    # Special cases
    special_cases()
    
    print("\nAll GLMM examples completed!")
    print("GLMM provides exact likelihood inference for count and binary data.")

if __name__ == "__main__":
    main()