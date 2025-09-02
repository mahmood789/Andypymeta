"""
Plotting command for PyMeta CLI.

Handles visualization generation.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import pymeta as pm


def add_parser(subparsers: Any) -> None:
    """Add plots parser to subparsers."""
    parser = subparsers.add_parser(
        'plots',
        help='Generate meta-analysis plots',
        description='Create various visualizations for meta-analysis'
    )
    
    # Subcommands for different plot types
    plot_subparsers = parser.add_subparsers(
        dest='plot_type',
        help='Plot type to generate',
        metavar='PLOT'
    )
    
    # Forest plot
    forest_parser = plot_subparsers.add_parser('forest', help='Generate forest plot')
    add_common_args(forest_parser)
    forest_parser.add_argument('--style', choices=['classic', 'modern', 'minimal', 'publication'], 
                              default='classic', help='Plot style')
    forest_parser.add_argument('--show-weights', action='store_true', help='Show study weights')
    
    # Funnel plot
    funnel_parser = plot_subparsers.add_parser('funnel', help='Generate funnel plot')
    add_common_args(funnel_parser)
    funnel_parser.add_argument('--contours', action='store_true', help='Add significance contours')
    funnel_parser.add_argument('--test', choices=['egger', 'begg'], help='Add bias test results')
    
    # Baujat plot
    baujat_parser = plot_subparsers.add_parser('baujat', help='Generate Baujat plot')
    add_common_args(baujat_parser)
    
    # Radial plot
    radial_parser = plot_subparsers.add_parser('radial', help='Generate radial plot')
    add_common_args(radial_parser)
    
    # All plots
    all_parser = plot_subparsers.add_parser('all', help='Generate all standard plots')
    add_common_args(all_parser)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to plot parsers."""
    parser.add_argument('--data', '-d', required=True, type=Path, 
                       help='Input data file (CSV, Excel, JSON)')
    parser.add_argument('--output', '-o', type=Path, 
                       help='Output file (if not specified, will display)')
    parser.add_argument('--title', help='Plot title')
    parser.add_argument('--width', type=float, default=10, help='Plot width in inches')
    parser.add_argument('--height', type=float, default=8, help='Plot height in inches')
    parser.add_argument('--dpi', type=int, default=300, help='Resolution for saved plots')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg', 'eps'], 
                       default='png', help='Output format')
    parser.add_argument('--model', choices=['fixed', 'random'], default='random',
                       help='Meta-analysis model')


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


def run_meta_analysis(data: pd.DataFrame, model: str) -> Any:
    """Run meta-analysis for plotting."""
    return pm.meta_analysis(data, model=model)


def generate_forest_plot(result: Any, args: argparse.Namespace) -> None:
    """Generate forest plot."""
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    
    pm.forest_plot(
        result,
        ax=ax,
        style=getattr(args, 'style', 'classic'),
        title=args.title or 'Forest Plot',
        show_weights=getattr(args, 'show_weights', False)
    )
    
    save_or_show_plot(fig, args)


def generate_funnel_plot(result: Any, args: argparse.Namespace) -> None:
    """Generate funnel plot."""
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    
    pm.funnel_plot(
        result,
        ax=ax,
        title=args.title or 'Funnel Plot',
        contours=getattr(args, 'contours', False)
    )
    
    # Add bias test if requested
    if hasattr(args, 'test') and args.test:
        data = result.data  # Assuming result has access to original data
        if args.test == 'egger':
            test_result = pm.egger_test(data)
            ax.text(0.02, 0.98, f"Egger's test: p = {test_result.pvalue:.3f}",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        elif args.test == 'begg':
            test_result = pm.begg_test(data)
            ax.text(0.02, 0.98, f"Begg's test: p = {test_result.pvalue:.3f}",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    save_or_show_plot(fig, args)


def generate_baujat_plot(result: Any, args: argparse.Namespace) -> None:
    """Generate Baujat plot."""
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    
    pm.baujat_plot(
        result,
        ax=ax,
        title=args.title or 'Baujat Plot'
    )
    
    save_or_show_plot(fig, args)


def generate_radial_plot(result: Any, args: argparse.Namespace) -> None:
    """Generate radial plot."""
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    
    pm.radial_plot(
        result,
        ax=ax,
        title=args.title or 'Radial Plot'
    )
    
    save_or_show_plot(fig, args)


def generate_all_plots(result: Any, args: argparse.Namespace) -> None:
    """Generate all standard plots."""
    plot_types = ['forest', 'funnel', 'baujat', 'radial']
    
    if args.output:
        output_dir = args.output.parent if args.output.is_file() else args.output
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = args.output.stem if args.output.is_file() else 'plot'
    else:
        output_dir = Path('.')
        base_name = 'plot'
    
    for plot_type in plot_types:
        # Create temporary args for each plot type
        temp_args = argparse.Namespace(**vars(args))
        temp_args.output = output_dir / f"{base_name}_{plot_type}.{args.format}"
        temp_args.title = f"{plot_type.title()} Plot"
        
        if plot_type == 'forest':
            generate_forest_plot(result, temp_args)
        elif plot_type == 'funnel':
            generate_funnel_plot(result, temp_args)
        elif plot_type == 'baujat':
            generate_baujat_plot(result, temp_args)
        elif plot_type == 'radial':
            generate_radial_plot(result, temp_args)
    
    print(f"All plots saved to {output_dir}")


def save_or_show_plot(fig: plt.Figure, args: argparse.Namespace) -> None:
    """Save plot to file or display."""
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(
            output_path,
            dpi=args.dpi,
            format=args.format,
            bbox_inches='tight'
        )
        print(f"Plot saved to {output_path}")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def run(args: argparse.Namespace) -> int:
    """Run plots command."""
    try:
        if not args.plot_type:
            print("Error: No plot type specified", file=sys.stderr)
            return 1
        
        # Load data
        print(f"Loading data from {args.data}...")
        data = load_data(args.data)
        print(f"Loaded {len(data)} studies")
        
        # Run meta-analysis
        print(f"Running {args.model}-effect meta-analysis...")
        result = run_meta_analysis(data, args.model)
        
        # Generate requested plot
        print(f"Generating {args.plot_type} plot...")
        
        if args.plot_type == 'forest':
            generate_forest_plot(result, args)
        elif args.plot_type == 'funnel':
            generate_funnel_plot(result, args)
        elif args.plot_type == 'baujat':
            generate_baujat_plot(result, args)
        elif args.plot_type == 'radial':
            generate_radial_plot(result, args)
        elif args.plot_type == 'all':
            generate_all_plots(result, args)
        else:
            print(f"Error: Unknown plot type: {args.plot_type}", file=sys.stderr)
            return 1
        
        print("Plot generation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error generating plots: {e}", file=sys.stderr)
        return 1