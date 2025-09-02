"""
Publication bias assessment command for PyMeta CLI.

Handles bias detection and correction methods.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, List

import pandas as pd
import matplotlib.pyplot as plt
import pymeta as pm


def add_parser(subparsers: Any) -> None:
    """Add bias assessment parser to subparsers."""
    parser = subparsers.add_parser(
        'bias',
        help='Assess publication bias',
        description='Detect and correct for publication bias'
    )
    
    # Input/output options
    parser.add_argument(
        '--data', '-d',
        required=True,
        type=Path,
        help='Input data file (CSV, Excel, JSON)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('.'),
        help='Output directory (default: current directory)'
    )
    
    # Bias tests
    parser.add_argument(
        '--tests',
        nargs='+',
        choices=['egger', 'begg', 'pet-peese', 'trimfill', 'selection', 'pcurve', 'all'],
        default=['egger', 'begg'],
        help='Bias tests to perform (default: egger begg)'
    )
    
    # Specific test options
    parser.add_argument(
        '--trimfill-side',
        choices=['left', 'right', 'auto'],
        default='auto',
        help='Side for trim-and-fill (default: auto)'
    )
    
    parser.add_argument(
        '--selection-model',
        choices=['step', 'linear', 'half-normal'],
        default='step',
        help='Selection model type (default: step)'
    )
    
    # Output options
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate bias assessment plots'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate comprehensive bias assessment report'
    )
    
    parser.add_argument(
        '--format',
        choices=['csv', 'html', 'json'],
        nargs='+',
        default=['csv'],
        help='Output formats (default: csv)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )


def load_data(file_path: Path) -> pd.DataFrame:
    """Load data from file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.csv':
        return pd.read_csv(file_path)
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif suffix == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def run_egger_test(data: pd.DataFrame) -> Any:
    """Run Egger's regression test."""
    print("Running Egger's regression test...")
    return pm.egger_test(data)


def run_begg_test(data: pd.DataFrame) -> Any:
    """Run Begg's rank correlation test."""
    print("Running Begg's rank correlation test...")
    return pm.begg_test(data)


def run_pet_peese(data: pd.DataFrame) -> Any:
    """Run PET-PEESE analysis."""
    print("Running PET-PEESE analysis...")
    pet_result = pm.pet_analysis(data)
    peese_result = pm.peese_analysis(data)
    
    return {
        'pet': pet_result,
        'peese': peese_result,
        'conditional_estimate': peese_result.conditional_estimate if pet_result.pvalue < 0.05 else pet_result.intercept
    }


def run_trim_fill(data: pd.DataFrame, side: str) -> Any:
    """Run trim-and-fill analysis."""
    print(f"Running trim-and-fill analysis (side: {side})...")
    return pm.trim_and_fill(data, side=side)


def run_selection_model(data: pd.DataFrame, model_type: str) -> Any:
    """Run selection model analysis."""
    print(f"Running selection model analysis ({model_type})...")
    return pm.selection_model(data, model_type=model_type)


def run_pcurve(data: pd.DataFrame) -> Any:
    """Run p-curve analysis."""
    print("Running p-curve analysis...")
    return pm.pcurve_analysis(data)


def print_test_results(test_name: str, result: Any, alpha: float) -> None:
    """Print test results."""
    print(f"\n{test_name} Results:")
    print("-" * (len(test_name) + 9))
    
    if test_name == "Egger's Test":
        print(f"Bias coefficient: {result.bias:.4f}")
        print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"t-statistic: {result.t_statistic:.4f}")
        print(f"p-value: {result.pvalue:.4f}")
        
        if result.pvalue < alpha:
            print("✓ Significant asymmetry detected (potential bias)")
        else:
            print("→ No significant asymmetry")
    
    elif test_name == "Begg's Test":
        print(f"Kendall's tau: {result.tau:.4f}")
        print(f"z-statistic: {result.z_statistic:.4f}")
        print(f"p-value: {result.pvalue:.4f}")
        
        if result.pvalue < alpha:
            print("✓ Significant rank correlation (potential bias)")
        else:
            print("→ No significant rank correlation")
    
    elif test_name == "PET-PEESE":
        print(f"PET intercept: {result['pet'].intercept:.4f} (p = {result['pet'].pvalue:.4f})")
        print(f"PEESE estimate: {result['peese'].effect:.4f}")
        print(f"Conditional estimate: {result['conditional_estimate']:.4f}")
        
        if result['pet'].pvalue < alpha:
            print("✓ PET significant - using PEESE estimate")
        else:
            print("→ PET not significant - using PET estimate")
    
    elif test_name == "Trim-and-Fill":
        print(f"Studies trimmed: {result.n_trimmed}")
        print(f"Studies filled: {result.n_filled}")
        print(f"Original estimate: {result.original_effect:.4f}")
        print(f"Adjusted estimate: {result.adjusted_effect:.4f}")
        print(f"Adjustment: {result.adjusted_effect - result.original_effect:.4f}")
        
        if result.n_filled > 0:
            print(f"✓ {result.n_filled} missing studies imputed")
        else:
            print("→ No missing studies detected")
    
    elif test_name == "Selection Model":
        print(f"Original estimate: {result.original_effect:.4f}")
        print(f"Adjusted estimate: {result.adjusted_effect:.4f}")
        print(f"Selection intensity: {result.selection_intensity:.4f}")
        print(f"Model convergence: {'Yes' if result.converged else 'No'}")
    
    elif test_name == "P-Curve":
        print(f"Evidential value test: p = {result.evidential_pvalue:.4f}")
        print(f"Inadequate evidence test: p = {result.inadequate_pvalue:.4f}")
        
        if result.evidential_pvalue < 0.05:
            print("✓ Evidential value present")
        elif result.inadequate_pvalue < 0.05:
            print("⚠ Inadequate evidence")
        else:
            print("→ Inconclusive evidence")


def generate_bias_plots(data: pd.DataFrame, results: dict, output_dir: Path) -> None:
    """Generate bias assessment plots."""
    print("Generating bias assessment plots...")
    
    # Run basic meta-analysis for plots
    meta_result = pm.meta_analysis(data, model='random')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Funnel plot
    pm.funnel_plot(meta_result, ax=axes[0, 0], title='Funnel Plot')
    
    # Egger's regression plot
    if 'egger' in results:
        pm.egger_plot(results['egger'], ax=axes[0, 1], title="Egger's Regression")
    
    # Trim-and-fill plot
    if 'trimfill' in results:
        pm.trim_fill_plot(results['trimfill'], ax=axes[1, 0], title='Trim-and-Fill')
    
    # P-curve plot
    if 'pcurve' in results:
        pm.pcurve_plot(results['pcurve'], ax=axes[1, 1], title='P-Curve')
    
    plt.tight_layout()
    plot_path = output_dir / 'bias_assessment_plots.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Bias plots saved to {plot_path}")
    plt.close(fig)


def save_results(results: dict, args: argparse.Namespace) -> None:
    """Save bias assessment results."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile results into summary
    summary = {}
    
    for test_name, result in results.items():
        if test_name == 'egger':
            summary[f'{test_name}_bias'] = result.bias
            summary[f'{test_name}_pvalue'] = result.pvalue
        elif test_name == 'begg':
            summary[f'{test_name}_tau'] = result.tau
            summary[f'{test_name}_pvalue'] = result.pvalue
        elif test_name == 'pet-peese':
            summary['pet_intercept'] = result['pet'].intercept
            summary['pet_pvalue'] = result['pet'].pvalue
            summary['peese_estimate'] = result['peese'].effect
            summary['conditional_estimate'] = result['conditional_estimate']
        elif test_name == 'trimfill':
            summary['trimfill_n_filled'] = result.n_filled
            summary['trimfill_adjusted_effect'] = result.adjusted_effect
        elif test_name == 'selection':
            summary['selection_adjusted_effect'] = result.adjusted_effect
            summary['selection_intensity'] = result.selection_intensity
        elif test_name == 'pcurve':
            summary['pcurve_evidential_p'] = result.evidential_pvalue
            summary['pcurve_inadequate_p'] = result.inadequate_pvalue
    
    # Save in requested formats
    for fmt in args.format:
        if fmt == 'csv':
            pd.DataFrame([summary]).to_csv(output_dir / 'bias_assessment.csv', index=False)
        elif fmt == 'json':
            import json
            with open(output_dir / 'bias_assessment.json', 'w') as f:
                json.dump(summary, f, indent=2)
        elif fmt == 'html':
            # Create HTML report
            create_html_report(results, output_dir / 'bias_assessment.html')
    
    print(f"Results saved to {output_dir}")


def create_html_report(results: dict, output_path: Path) -> None:
    """Create HTML bias assessment report."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Publication Bias Assessment Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            h2 { color: #666; }
            .result { margin: 20px 0; padding: 10px; border-left: 3px solid #007acc; }
            .significant { border-left-color: #dc3545; }
            .not-significant { border-left-color: #28a745; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Publication Bias Assessment Report</h1>
    """
    
    for test_name, result in results.items():
        html_content += f"<div class='result'><h2>{test_name.title()} Test</h2>"
        # Add test-specific content here
        html_content += "</div>"
    
    html_content += "</body></html>"
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def run(args: argparse.Namespace) -> int:
    """Run bias assessment command."""
    try:
        # Load data
        print(f"Loading data from {args.data}...")
        data = load_data(args.data)
        print(f"Loaded {len(data)} studies")
        
        # Expand 'all' tests
        if 'all' in args.tests:
            args.tests = ['egger', 'begg', 'pet-peese', 'trimfill', 'selection', 'pcurve']
        
        # Run bias tests
        results = {}
        
        for test in args.tests:
            try:
                if test == 'egger':
                    results['egger'] = run_egger_test(data)
                    print_test_results("Egger's Test", results['egger'], args.alpha)
                
                elif test == 'begg':
                    results['begg'] = run_begg_test(data)
                    print_test_results("Begg's Test", results['begg'], args.alpha)
                
                elif test == 'pet-peese':
                    results['pet-peese'] = run_pet_peese(data)
                    print_test_results("PET-PEESE", results['pet-peese'], args.alpha)
                
                elif test == 'trimfill':
                    results['trimfill'] = run_trim_fill(data, args.trimfill_side)
                    print_test_results("Trim-and-Fill", results['trimfill'], args.alpha)
                
                elif test == 'selection':
                    results['selection'] = run_selection_model(data, args.selection_model)
                    print_test_results("Selection Model", results['selection'], args.alpha)
                
                elif test == 'pcurve':
                    results['pcurve'] = run_pcurve(data)
                    print_test_results("P-Curve", results['pcurve'], args.alpha)
            
            except Exception as e:
                print(f"Warning: {test} test failed: {e}", file=sys.stderr)
        
        # Generate plots if requested
        if args.plots:
            generate_bias_plots(data, results, Path(args.output))
        
        # Save results
        save_results(results, args)
        
        print("\nBias assessment completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error in bias assessment: {e}", file=sys.stderr)
        return 1