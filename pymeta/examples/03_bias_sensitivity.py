#!/usr/bin/env python3
"""
03_bias_sensitivity.py - Publication bias detection and sensitivity analysis

This example demonstrates publication bias assessment methods in PyMeta.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymeta as pm

def create_biased_data():
    """Create sample data with publication bias."""
    np.random.seed(456)
    
    # Generate true effect sizes
    true_effects = np.random.normal(0.3, 0.1, 20)
    
    # Generate standard errors (larger studies have smaller SEs)
    sample_sizes = np.random.randint(30, 500, 20)
    standard_errors = 1 / np.sqrt(sample_sizes) * np.random.uniform(0.8, 1.2, 20)
    
    # Simulate observed effects with publication bias
    # Smaller studies with non-significant results are more likely to be missing
    observed_effects = []
    observed_ses = []
    study_ids = []
    
    for i, (true_eff, se) in enumerate(zip(true_effects, standard_errors)):
        # Add measurement error
        observed_eff = np.random.normal(true_eff, se)
        
        # Publication bias: smaller studies with non-sig results less likely to be published
        t_stat = abs(observed_eff / se)
        prob_publish = min(1.0, 0.3 + 0.7 * (t_stat / 2.0))
        
        if np.random.random() < prob_publish:
            observed_effects.append(observed_eff)
            observed_ses.append(se)
            study_ids.append(f'Study_{i+1}')
    
    return pd.DataFrame({
        'study': study_ids,
        'effect_size': observed_effects,
        'standard_error': observed_ses,
        'variance': [se**2 for se in observed_ses]
    })

def assess_publication_bias():
    """Demonstrate publication bias assessment methods."""
    
    data = create_biased_data()
    
    print("Publication Bias Assessment")
    print("=" * 30)
    print(f"Number of studies: {len(data)}")
    print()
    
    # Basic meta-analysis
    result = pm.meta_analysis(data, model='random')
    print(f"Random-effects estimate: {result.overall_effect:.3f}")
    print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    print()
    
    # Funnel plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Basic funnel plot
    ax = axes[0, 0]
    pm.funnel_plot(result, ax=ax, title='Basic Funnel Plot')
    
    # Funnel plot with contour lines
    ax = axes[0, 1]
    pm.funnel_plot(result, ax=ax, contours=True, title='Funnel Plot with Contours')
    
    # Egger's test
    egger_result = pm.egger_test(data)
    print(f"Egger's test:")
    print(f"  Bias coefficient: {egger_result.bias:.3f}")
    print(f"  95% CI: [{egger_result.ci_lower:.3f}, {egger_result.ci_upper:.3f}]")
    print(f"  p-value: {egger_result.pvalue:.3f}")
    print()
    
    # Begg's test
    begg_result = pm.begg_test(data)
    print(f"Begg's test:")
    print(f"  Kendall's tau: {begg_result.tau:.3f}")
    print(f"  p-value: {begg_result.pvalue:.3f}")
    print()
    
    # PET-PEESE
    pet_result = pm.pet_analysis(data)
    peese_result = pm.peese_analysis(data)
    
    print(f"PET analysis:")
    print(f"  Intercept: {pet_result.intercept:.3f} (p = {pet_result.pvalue:.3f})")
    print(f"PET-PEESE conditional estimate: {peese_result.conditional_estimate:.3f}")
    print()
    
    # Trim-and-fill
    tf_result = pm.trim_and_fill(data)
    print(f"Trim-and-fill:")
    print(f"  Number of imputed studies: {tf_result.n_imputed}")
    print(f"  Adjusted estimate: {tf_result.adjusted_effect:.3f}")
    print(f"  95% CI: [{tf_result.ci_lower:.3f}, {tf_result.ci_upper:.3f}]")
    print()
    
    # Plot trim-and-fill results
    ax = axes[1, 0]
    pm.trim_fill_plot(tf_result, ax=ax, title='Trim-and-Fill Plot')
    
    # Selection model
    try:
        sel_result = pm.selection_model(data)
        print(f"Selection model:")
        print(f"  Adjusted estimate: {sel_result.adjusted_effect:.3f}")
        print(f"  95% CI: [{sel_result.ci_lower:.3f}, {sel_result.ci_upper:.3f}]")
        print()
    except Exception as e:
        print(f"Selection model failed: {e}")
        print()
    
    # P-curve analysis (if applicable)
    try:
        pcurve_result = pm.pcurve_analysis(data)
        ax = axes[1, 1]
        pm.pcurve_plot(pcurve_result, ax=ax, title='P-Curve Analysis')
        
        print(f"P-curve analysis:")
        print(f"  Evidential value test: p = {pcurve_result.evidential_pvalue:.3f}")
        print(f"  Inadequate evidence test: p = {pcurve_result.inadequate_pvalue:.3f}")
        print()
    except Exception as e:
        print(f"P-curve analysis not applicable: {e}")
        axes[1, 1].text(0.5, 0.5, 'P-curve\nNot Applicable', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('P-Curve Analysis')
    
    plt.tight_layout()
    plt.savefig('bias_assessment.png', dpi=300, bbox_inches='tight')
    print("Bias assessment plots saved as 'bias_assessment.png'")

def sensitivity_analysis():
    """Demonstrate sensitivity analysis methods."""
    
    data = create_biased_data()
    
    print("\nSensitivity Analysis")
    print("=" * 20)
    
    # Leave-one-out analysis
    loo_result = pm.leave_one_out(data)
    
    print("Leave-one-out analysis:")
    print(f"  Range of estimates: {loo_result.min_effect:.3f} to {loo_result.max_effect:.3f}")
    print(f"  Most influential study: {loo_result.most_influential_study}")
    print()
    
    # Cumulative meta-analysis
    cum_result = pm.cumulative_analysis(data, order_by='year')
    
    print("Cumulative meta-analysis shows evolution of evidence over time.")
    
    # Plot sensitivity analyses
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Leave-one-out plot
    pm.influence_plot(loo_result, ax=axes[0], title='Leave-One-Out Analysis')
    
    # Cumulative plot
    pm.cumulative_plot(cum_result, ax=axes[1], title='Cumulative Meta-Analysis')
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print("Sensitivity analysis plots saved as 'sensitivity_analysis.png'")

def main():
    """Run publication bias and sensitivity analysis examples."""
    
    print("PyMeta Publication Bias and Sensitivity Analysis")
    print("=" * 50)
    
    # Assess publication bias
    assess_publication_bias()
    
    # Sensitivity analysis
    sensitivity_analysis()
    
    print("\nExample completed!")
    print("Check the generated PNG files for results.")

if __name__ == "__main__":
    main()