#!/usr/bin/env python3
"""
06_tsa_living_review.py - Trial Sequential Analysis and Living Reviews

This example demonstrates TSA for living systematic reviews and 
automated monitoring of evidence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pymeta as pm

def create_sequential_data():
    """Create sample data simulating studies published over time."""
    np.random.seed(303)
    
    # Generate publication dates over 10 years
    start_date = datetime(2014, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    n_studies = 25
    pub_dates = []
    for i in range(n_studies):
        # More studies published in recent years
        year_offset = int(np.random.exponential(3))
        if year_offset > 10:
            year_offset = 10
        
        pub_date = start_date + timedelta(days=year_offset * 365 + np.random.randint(0, 365))
        pub_dates.append(pub_date)
    
    pub_dates.sort()
    
    # Generate effect sizes with true underlying effect
    true_effect = 0.3
    effect_sizes = []
    variances = []
    sample_sizes = []
    
    for i, date in enumerate(pub_dates):
        # Sample size increases over time (better funding)
        base_n = 100 + (date.year - 2014) * 20
        n = int(np.random.normal(base_n, 50))
        if n < 50:
            n = 50
        
        # Variance inversely related to sample size
        variance = (2.0 / n) * np.random.uniform(0.8, 1.2)
        
        # Observed effect with measurement error
        observed_effect = np.random.normal(true_effect, np.sqrt(variance))
        
        effect_sizes.append(observed_effect)
        variances.append(variance)
        sample_sizes.append(n)
    
    data = pd.DataFrame({
        'study': [f'Study_{i+1}' for i in range(n_studies)],
        'effect_size': effect_sizes,
        'variance': variances,
        'sample_size': sample_sizes,
        'publication_date': pub_dates,
        'year': [d.year for d in pub_dates],
        'cumulative_n': np.cumsum(sample_sizes)
    })
    
    return data

def trial_sequential_analysis():
    """Demonstrate Trial Sequential Analysis (TSA)."""
    
    data = create_sequential_data()
    
    print("Trial Sequential Analysis (TSA)")
    print("=" * 35)
    print(f"Total studies: {len(data)}")
    print(f"Publication period: {data['year'].min()} - {data['year'].max()}")
    print(f"Total sample size: {data['sample_size'].sum():,}")
    print()
    
    # TSA parameters
    alpha = 0.05  # Type I error
    beta = 0.20   # Type II error (80% power)
    delta = 0.25  # Clinically relevant effect size
    
    print(f"TSA Parameters:")
    print(f"- Alpha: {alpha}")
    print(f"- Beta: {beta} (Power: {1-beta:.0%})")
    print(f"- Delta (MIRES): {delta}")
    print()
    
    # Perform TSA
    tsa_result = pm.trial_sequential_analysis(
        data,
        alpha=alpha,
        beta=beta,
        delta=delta,
        order_by='publication_date'
    )
    
    print("TSA Results:")
    print("-" * 12)
    print(f"Required Information Size (RIS): {tsa_result.required_information_size:,}")
    print(f"Actual cumulative sample size: {data['sample_size'].sum():,}")
    print(f"Information fraction: {tsa_result.information_fraction:.2%}")
    print()
    
    if tsa_result.futility_boundary_crossed:
        print("✓ Futility boundary crossed - unlikely to find significant effect")
    elif tsa_result.efficacy_boundary_crossed:
        print("✓ Efficacy boundary crossed - sufficient evidence for effect")
    elif tsa_result.required_information_reached:
        print("✓ Required information size reached")
    else:
        print("→ More studies needed to reach definitive conclusion")
    
    print(f"Conclusive evidence reached at study: {tsa_result.conclusion_study}")
    print()
    
    return tsa_result

def living_review_simulation():
    """Simulate a living systematic review process."""
    
    data = create_sequential_data()
    
    print("Living Systematic Review Simulation")
    print("=" * 38)
    
    # Initialize living review
    living_review = pm.LivingReview(
        review_id="example_review",
        search_strategy="intervention AND outcome",
        update_frequency="monthly",
        alpha_spending_function="lan_demets"
    )
    
    # Simulate monthly updates
    update_dates = pd.date_range(
        start="2014-01-01",
        end="2024-01-01", 
        freq='3MS'  # Quarterly updates
    )
    
    results_history = []
    
    for update_date in update_dates:
        # Get studies available at this date
        available_studies = data[data['publication_date'] <= update_date].copy()
        
        if len(available_studies) == 0:
            continue
            
        # Perform meta-analysis
        try:
            result = pm.meta_analysis(available_studies, model='random')
            
            # Check for significant change since last update
            if len(results_history) > 0:
                last_result = results_history[-1]
                change_magnitude = abs(result.overall_effect - last_result['effect'])
                significant_change = change_magnitude > 0.1  # Threshold
            else:
                significant_change = True
            
            # TSA check
            tsa_check = pm.tsa_monitoring(
                available_studies,
                alpha=0.05,
                beta=0.20,
                delta=0.25
            )
            
            update_record = {
                'date': update_date,
                'n_studies': len(available_studies),
                'total_n': available_studies['sample_size'].sum(),
                'effect': result.overall_effect,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper,
                'significant_change': significant_change,
                'tsa_conclusive': tsa_check.is_conclusive,
                'tsa_boundary': tsa_check.boundary_type if tsa_check.is_conclusive else None
            }
            
            results_history.append(update_record)
            
            # Alert conditions
            if significant_change or tsa_check.is_conclusive:
                print(f"📧 ALERT {update_date.strftime('%Y-%m-%d')}: ", end="")
                if significant_change:
                    print(f"Significant change detected (Δ={change_magnitude:.3f})")
                if tsa_check.is_conclusive:
                    print(f"TSA conclusive: {tsa_check.boundary_type}")
        
        except Exception as e:
            print(f"Update {update_date.strftime('%Y-%m-%d')} failed: {e}")
    
    print(f"\nLiving review completed:")
    print(f"- Total updates: {len(results_history)}")
    print(f"- Final effect estimate: {results_history[-1]['effect']:.3f}")
    print(f"- Final 95% CI: [{results_history[-1]['ci_lower']:.3f}, {results_history[-1]['ci_upper']:.3f}]")
    
    return pd.DataFrame(results_history)

def automated_search_demo():
    """Demonstrate automated search and screening."""
    
    print("\nAutomated Search and Screening")
    print("=" * 35)
    
    # Configure search monitors
    search_config = {
        'pubmed': {
            'query': '(meta-analysis OR systematic review) AND intervention',
            'frequency': 'daily',
            'email_alerts': True
        },
        'arxiv': {
            'query': 'meta-analysis intervention',
            'frequency': 'weekly',
            'categories': ['stat.AP', 'stat.ME']
        }
    }
    
    print("Search monitors configured:")
    for source, config in search_config.items():
        print(f"- {source.upper()}: {config['query']} ({config['frequency']})")
    
    # Simulate new study detection
    print("\nSimulated new study detection:")
    new_studies = [
        {
            'title': 'A New RCT of Intervention X vs Control',
            'authors': 'Smith J, Jones A, Brown B',
            'source': 'pubmed',
            'relevance_score': 0.85,
            'auto_include': True
        },
        {
            'title': 'Machine Learning in Meta-Analysis', 
            'authors': 'Chen L, Wang Y',
            'source': 'arxiv',
            'relevance_score': 0.65,
            'auto_include': False
        }
    ]
    
    for study in new_studies:
        print(f"📄 {study['title']}")
        print(f"   Relevance: {study['relevance_score']:.2f}")
        print(f"   Auto-include: {'Yes' if study['auto_include'] else 'Manual review needed'}")
        print()

def visualize_tsa_and_living_review(tsa_result, living_history):
    """Create visualizations for TSA and living review."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # TSA plot
    pm.tsa_plot(tsa_result, ax=axes[0, 0], title='Trial Sequential Analysis')
    
    # Living review evolution
    pm.living_review_plot(living_history, ax=axes[0, 1], title='Living Review Evolution')
    
    # Cumulative effect plot
    pm.cumulative_effect_plot(tsa_result, ax=axes[1, 0], title='Cumulative Effect Size')
    
    # Information growth
    pm.information_plot(tsa_result, ax=axes[1, 1], title='Information Accrual')
    
    plt.tight_layout()
    plt.savefig('tsa_living_review.png', dpi=300, bbox_inches='tight')
    print("TSA and living review plots saved as 'tsa_living_review.png'")

def monitoring_dashboard():
    """Demonstrate monitoring dashboard for living reviews."""
    
    print("\nLiving Review Monitoring Dashboard")
    print("=" * 40)
    print("📊 Dashboard Components:")
    print("- Real-time effect estimate tracker")
    print("- TSA boundary monitoring")
    print("- Search result feeds")
    print("- Quality assessment alerts")
    print("- Automated report generation")
    print()
    
    print("🔔 Alert System:")
    print("- Email notifications for new relevant studies")
    print("- Slack integration for team updates")
    print("- RSS feeds for stakeholders")
    print("- Automated PROSPERO updates")
    print()
    
    print("📈 Analytics:")
    print("- Study accumulation rate")
    print("- Effect size stability")
    print("- Heterogeneity trends")
    print("- Quality score evolution")

def main():
    """Run TSA and living review examples."""
    
    print("PyMeta TSA and Living Review Examples")
    print("=" * 40)
    print("Trial Sequential Analysis for adaptive meta-analysis")
    print()
    
    # Trial Sequential Analysis
    tsa_result = trial_sequential_analysis()
    
    print("\n" + "="*60 + "\n")
    
    # Living review simulation
    living_history = living_review_simulation()
    
    # Automated search demo
    automated_search_demo()
    
    # Visualizations
    visualize_tsa_and_living_review(tsa_result, living_history)
    
    # Monitoring dashboard
    monitoring_dashboard()
    
    print("\nTSA and Living Review examples completed!")
    print("These methods enable adaptive, evidence-based decision making.")

if __name__ == "__main__":
    main()