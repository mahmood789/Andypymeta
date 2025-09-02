"""Command line interface for pymeta."""

import argparse
import sys
import pandas as pd
import numpy as np
from ..models import fixed_effects, random_effects
from ..effects import binary_effects
from ..viz import forest_plot, funnel_plot


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='pymeta: Meta-analysis toolkit')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Meta-analysis command
    meta_parser = subparsers.add_parser('meta', help='Conduct meta-analysis')
    meta_parser.add_argument('--input', '-i', required=True, 
                           help='Input CSV file with study data')
    meta_parser.add_argument('--output', '-o', 
                           help='Output file for results')
    meta_parser.add_argument('--method', '-m', default='RE',
                           choices=['FE', 'RE'], 
                           help='Analysis method: FE (fixed effects) or RE (random effects)')
    meta_parser.add_argument('--effect-col', default='effect_size',
                           help='Column name for effect sizes')
    meta_parser.add_argument('--se-col', default='standard_error', 
                           help='Column name for standard errors')
    meta_parser.add_argument('--plot', action='store_true',
                           help='Generate forest plot')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version')
    
    args = parser.parse_args()
    
    if args.command == 'meta':
        run_meta_analysis(args)
    elif args.command == 'version':
        print("pymeta version 0.0.1")
    else:
        parser.print_help()


def run_meta_analysis(args):
    """Run meta-analysis from command line arguments."""
    try:
        # Load data
        data = pd.read_csv(args.input)
        
        # Extract effect sizes and standard errors
        effect_sizes = data[args.effect_col].values
        standard_errors = data[args.se_col].values
        variances = standard_errors**2
        
        # Run analysis
        if args.method == 'FE':
            results = fixed_effects(effect_sizes, variances)
        else:
            results = random_effects(effect_sizes, variances)
        
        # Print results
        print(f"\nMeta-analysis Results ({results['method']})")
        print("=" * 50)
        print(f"Pooled effect size: {results['pooled_effect']:.4f}")
        print(f"Standard error: {results['se']:.4f}")
        print(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
        print(f"Z-value: {results['z_value']:.4f}")
        print(f"P-value: {results['p_value']:.4f}")
        
        if 'Q' in results:
            print(f"\nHeterogeneity:")
            print(f"Q = {results['Q']:.4f}, df = {results['Q_df']}, p = {results['Q_p_value']:.4f}")
            if 'I2' in results:
                print(f"I² = {results['I2']*100:.1f}%")
            if 'tau2' in results:
                print(f"τ² = {results['tau2']:.4f}")
        
        # Save results if requested
        if args.output:
            results_df = pd.DataFrame([results])
            results_df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
        
        # Generate plot if requested
        if args.plot:
            study_labels = data.get('study_id', [f"Study {i+1}" for i in range(len(effect_sizes))])
            fig = forest_plot(effect_sizes, standard_errors, 
                            study_labels=study_labels,
                            pooled_effect=results['pooled_effect'],
                            pooled_se=results['se'])
            
            plot_path = args.input.replace('.csv', '_forest.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Forest plot saved to {plot_path}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()