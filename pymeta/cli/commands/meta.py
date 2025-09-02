"""
Meta-analysis command for PyMeta CLI.

Handles basic meta-analysis operations.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pymeta as pm


def add_parser(subparsers: Any) -> None:
    """Add meta-analysis parser to subparsers."""
    parser = subparsers.add_parser(
        'meta',
        help='Perform meta-analysis',
        description='Conduct meta-analysis with various models and options'
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
    
    # Model options
    parser.add_argument(
        '--model', '-m',
        choices=['fixed', 'random'],
        default='random',
        help='Meta-analysis model (default: random)'
    )
    
    parser.add_argument(
        '--tau2-method',
        choices=['dl', 'ml', 'reml', 'pm', 'hs', 'eb', 'sj'],
        default='dl',
        help='Tau-squared estimation method (default: dl)'
    )
    
    # Effect size options
    parser.add_argument(
        '--effect-column',
        default='effect_size',
        help='Column name for effect sizes (default: effect_size)'
    )
    
    parser.add_argument(
        '--variance-column',
        default='variance',
        help='Column name for variances (default: variance)'
    )
    
    parser.add_argument(
        '--se-column',
        default='standard_error',
        help='Column name for standard errors (alternative to variance)'
    )
    
    # Analysis options
    parser.add_argument(
        '--subgroup',
        help='Column name for subgroup analysis'
    )
    
    parser.add_argument(
        '--moderators',
        nargs='+',
        help='Column names for meta-regression moderators'
    )
    
    # Output options
    parser.add_argument(
        '--format',
        choices=['csv', 'excel', 'json', 'html'],
        nargs='+',
        default=['csv'],
        help='Output formats (default: csv)'
    )
    
    parser.add_argument(
        '--plots',
        choices=['forest', 'funnel', 'all'],
        nargs='+',
        help='Generate plots'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file (YAML)'
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


def validate_data(data: pd.DataFrame, args: argparse.Namespace) -> None:
    """Validate input data."""
    required_columns = [args.effect_column]
    
    if args.variance_column in data.columns:
        required_columns.append(args.variance_column)
    elif args.se_column in data.columns:
        required_columns.append(args.se_column)
    else:
        raise ValueError("Data must contain either variance or standard error column")
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for missing values
    if data[required_columns].isnull().any().any():
        print("Warning: Missing values detected in required columns", file=sys.stderr)


def run_meta_analysis(data: pd.DataFrame, args: argparse.Namespace) -> Any:
    """Run the meta-analysis."""
    # Prepare data
    if args.se_column in data.columns and args.variance_column not in data.columns:
        data[args.variance_column] = data[args.se_column] ** 2
    
    # Basic meta-analysis
    if args.subgroup:
        if args.subgroup not in data.columns:
            raise ValueError(f"Subgroup column '{args.subgroup}' not found in data")
        
        result = pm.subgroup_analysis(
            data,
            grouping_var=args.subgroup,
            model=args.model,
            tau2_method=args.tau2_method
        )
    elif args.moderators:
        missing_moderators = [mod for mod in args.moderators if mod not in data.columns]
        if missing_moderators:
            raise ValueError(f"Moderator columns not found: {missing_moderators}")
        
        result = pm.meta_regression(
            data,
            moderators=args.moderators,
            model=args.model,
            tau2_method=args.tau2_method
        )
    else:
        result = pm.meta_analysis(
            data,
            model=args.model,
            tau2_method=args.tau2_method
        )
    
    return result


def save_results(result: Any, args: argparse.Namespace) -> None:
    """Save analysis results."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate base filename
    base_name = f"meta_analysis_{args.model}"
    if args.subgroup:
        base_name += f"_subgroup_{args.subgroup}"
    elif args.moderators:
        base_name += f"_regression"
    
    # Save in requested formats
    for fmt in args.format:
        if fmt == 'csv':
            result.to_csv(output_dir / f"{base_name}.csv")
        elif fmt == 'excel':
            result.to_excel(output_dir / f"{base_name}.xlsx")
        elif fmt == 'json':
            result.to_json(output_dir / f"{base_name}.json")
        elif fmt == 'html':
            result.to_html(output_dir / f"{base_name}.html")
    
    print(f"Results saved to {output_dir}")


def generate_plots(result: Any, data: pd.DataFrame, args: argparse.Namespace) -> None:
    """Generate requested plots."""
    if not args.plots:
        return
    
    output_dir = Path(args.output)
    
    for plot_type in args.plots:
        if plot_type == 'forest' or plot_type == 'all':
            fig = pm.forest_plot(result, title='Meta-Analysis Forest Plot')
            fig.savefig(output_dir / 'forest_plot.png', dpi=300, bbox_inches='tight')
            print(f"Forest plot saved to {output_dir / 'forest_plot.png'}")
        
        if plot_type == 'funnel' or plot_type == 'all':
            fig = pm.funnel_plot(result, title='Funnel Plot')
            fig.savefig(output_dir / 'funnel_plot.png', dpi=300, bbox_inches='tight')
            print(f"Funnel plot saved to {output_dir / 'funnel_plot.png'}")


def print_summary(result: Any, args: argparse.Namespace) -> None:
    """Print analysis summary."""
    print("\nMeta-Analysis Results")
    print("=" * 30)
    print(f"Model: {args.model.title()}-effect")
    
    if hasattr(result, 'overall_effect'):
        print(f"Overall effect: {result.overall_effect:.4f}")
        print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"p-value: {result.pvalue:.4f}")
        
        if args.model == 'random':
            print(f"Tau²: {result.tau_squared:.4f}")
            print(f"I²: {result.i_squared:.1f}%")
            print(f"Q-statistic: {result.q_statistic:.2f} (p = {result.q_pvalue:.4f})")


def run(args: argparse.Namespace) -> int:
    """Run meta-analysis command."""
    try:
        # Load configuration if provided
        if args.config:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            # Update args with config values (command line takes precedence)
            for key, value in config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
        
        # Load and validate data
        print(f"Loading data from {args.data}...")
        data = load_data(args.data)
        print(f"Loaded {len(data)} studies")
        
        validate_data(data, args)
        
        # Run analysis
        print(f"Running {args.model}-effect meta-analysis...")
        result = run_meta_analysis(data, args)
        
        # Print summary
        print_summary(result, args)
        
        # Save results
        save_results(result, args)
        
        # Generate plots
        generate_plots(result, data, args)
        
        print("\nMeta-analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error in meta-analysis: {e}", file=sys.stderr)
        return 1