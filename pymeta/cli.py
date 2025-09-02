"""
Command-line interface for PyMeta package.

Provides comprehensive CLI tools for meta-analysis including:
- Basic analysis with HKSJ support
- Plotting capabilities
- Leave-one-out analysis
- Living meta-analysis scheduler
"""

import sys
import os
from pathlib import Path
from typing import Optional, List
import click
import pandas as pd

from . import (
    MetaAnalysisConfig, 
    analyze_csv,
    MetaResults
)
from .plots import plot_forest, plot_funnel, plot_funnel_contour
from .diagnostics import leave_one_out_analysis
from .living import start_living_analysis


@click.group()
@click.version_option(version="0.1.0")
def main():
    """PyMeta - Comprehensive Meta-Analysis Package with HKSJ Support."""
    pass


@main.command()
@click.option('--csv', 'csv_file', required=True, type=click.Path(exists=True),
              help='Path to CSV file containing meta-analysis data')
@click.option('--effect-col', default='effect', 
              help='Name of effect size column (default: effect)')
@click.option('--variance-col', default='variance',
              help='Name of variance column (default: variance)')
@click.option('--study-col', default='study',
              help='Name of study ID column (default: study)')
@click.option('--model', type=click.Choice(['FE', 'RE']), default='RE',
              help='Meta-analysis model (default: RE)')
@click.option('--tau2', type=click.Choice(['DL', 'REML', 'PM', 'ML']), default='REML',
              help='Tau² estimation method (default: REML)')
@click.option('--hksj/--no-hksj', default=False,
              help='Use HKSJ variance adjustment (default: no-hksj)')
@click.option('--alpha', type=float, default=0.05,
              help='Significance level (default: 0.05)')
@click.option('--output', type=click.Path(),
              help='Output file for results (optional)')
@click.option('--verbose', is_flag=True,
              help='Show detailed output')
def analyze(csv_file: str, effect_col: str, variance_col: str, study_col: str,
           model: str, tau2: str, hksj: bool, alpha: float, 
           output: Optional[str], verbose: bool):
    """Perform meta-analysis on CSV data."""
    
    try:
        # Create configuration
        config = MetaAnalysisConfig(
            model=model,
            tau2_method=tau2,
            use_hksj=hksj,
            alpha=alpha
        )
        
        if verbose:
            click.echo(f"Loading data from: {csv_file}")
            click.echo(f"Configuration: {config}")
        
        # Run analysis
        results = analyze_csv(
            filepath=csv_file,
            effect_col=effect_col,
            variance_col=variance_col,
            study_col=study_col,
            config=config
        )
        
        # Display results
        click.echo("\n" + "=" * 60)
        click.echo("META-ANALYSIS RESULTS")
        click.echo("=" * 60)
        click.echo(f"Method: {results.method}")
        click.echo(f"Effect Size: {results.effect:.6f}")
        click.echo(f"Standard Error: {results.se:.6f}")
        click.echo(f"95% CI: [{results.ci_lower:.6f}, {results.ci_upper:.6f}]")
        click.echo(f"P-value: {results.p_value:.6f}")
        click.echo(f"CI Width: {results.ci_width:.6f}")
        
        if results.use_hksj and results.df is not None:
            click.echo(f"Degrees of Freedom (HKSJ): {results.df}")
        
        click.echo("\nHeterogeneity Statistics:")
        click.echo(f"Tau²: {results.tau2:.6f}")
        click.echo(f"I²: {results.i2:.2f}%") 
        click.echo(f"H²: {results.h2:.6f}")
        click.echo(f"Q: {results.q_stat:.6f} (p = {results.q_p_value:.6f})")
        
        if results.points:
            click.echo(f"\nNumber of Studies: {len(results.points)}")
        
        # Save results if requested
        if output:
            _save_results_to_file(results, output, verbose)
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option('--csv', 'csv_file', required=True, type=click.Path(exists=True),
              help='Path to CSV file containing meta-analysis data')
@click.option('--which', type=click.Choice(['forest', 'funnel', 'funnel-contour']),
              default='forest', help='Type of plot to generate (default: forest)')
@click.option('--effect-col', default='effect',
              help='Name of effect size column (default: effect)')
@click.option('--variance-col', default='variance',
              help='Name of variance column (default: variance)')
@click.option('--study-col', default='study',
              help='Name of study ID column (default: study)')
@click.option('--model', type=click.Choice(['FE', 'RE']), default='RE',
              help='Meta-analysis model (default: RE)')
@click.option('--tau2', type=click.Choice(['DL', 'REML', 'PM', 'ML']), default='REML',
              help='Tau² estimation method (default: REML)')
@click.option('--hksj/--no-hksj', default=False,
              help='Use HKSJ variance adjustment (default: no-hksj)')
@click.option('--output', type=click.Path(),
              help='Output file for plot (default: auto-generated)')
@click.option('--title', help='Custom plot title')
@click.option('--figsize', nargs=2, type=float, default=[10.0, 8.0],
              help='Figure size as width height (default: 10 8)')
@click.option('--dpi', type=int, default=300,
              help='DPI for saved plot (default: 300)')
@click.option('--show', is_flag=True,
              help='Display plot interactively')
def plot(csv_file: str, which: str, effect_col: str, variance_col: str,
         study_col: str, model: str, tau2: str, hksj: bool,
         output: Optional[str], title: Optional[str], figsize: List[float],
         dpi: int, show: bool):
    """Generate plots for meta-analysis results."""
    
    try:
        # Create configuration
        config = MetaAnalysisConfig(
            model=model,
            tau2_method=tau2,
            use_hksj=hksj
        )
        
        click.echo(f"Analyzing data for {which} plot...")
        
        # Run analysis
        results = analyze_csv(
            filepath=csv_file,
            effect_col=effect_col,
            variance_col=variance_col,
            study_col=study_col,
            config=config
        )
        
        # Generate plot
        if which == 'forest':
            fig = plot_forest(results, title=title, figsize=tuple(figsize))
        elif which == 'funnel':
            fig = plot_funnel(results, title=title, figsize=tuple(figsize))
        elif which == 'funnel-contour':
            fig = plot_funnel_contour(results, title=title, figsize=tuple(figsize))
        
        # Save plot
        if output is None:
            # Auto-generate filename
            base_name = Path(csv_file).stem
            output = f"{base_name}_{which}_plot.png"
        
        fig.savefig(output, dpi=dpi, bbox_inches='tight')
        click.echo(f"Plot saved to: {output}")
        
        # Show plot if requested
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            import matplotlib.pyplot as plt
            plt.close(fig)
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option('--csv', 'csv_file', required=True, type=click.Path(exists=True),
              help='Path to CSV file containing meta-analysis data')
@click.option('--effect-col', default='effect',
              help='Name of effect size column (default: effect)')
@click.option('--variance-col', default='variance',
              help='Name of variance column (default: variance)')
@click.option('--study-col', default='study',
              help='Name of study ID column (default: study)')
@click.option('--model', type=click.Choice(['FE', 'RE']), default='RE',
              help='Meta-analysis model (default: RE)')
@click.option('--tau2', type=click.Choice(['DL', 'REML', 'PM', 'ML']), default='REML',
              help='Tau² estimation method (default: REML)')
@click.option('--hksj/--no-hksj', default=False,
              help='Use HKSJ variance adjustment (default: no-hksj)')
@click.option('--output', type=click.Path(),
              help='Output file for LOO results (CSV format)')
@click.option('--plot', is_flag=True,
              help='Generate influence plot')
@click.option('--verbose', is_flag=True,
              help='Show detailed output')
def loo(csv_file: str, effect_col: str, variance_col: str, study_col: str,
        model: str, tau2: str, hksj: bool, output: Optional[str], 
        plot: bool, verbose: bool):
    """Perform leave-one-out analysis to assess study influence."""
    
    try:
        # Create configuration
        config = MetaAnalysisConfig(
            model=model,
            tau2_method=tau2,
            use_hksj=hksj
        )
        
        if verbose:
            click.echo(f"Loading data from: {csv_file}")
            click.echo("Performing leave-one-out analysis...")
        
        # Load data and create points
        df = pd.read_csv(csv_file)
        from . import MetaPoint
        points = [
            MetaPoint(
                effect=row[effect_col],
                variance=row[variance_col],
                study_id=str(row[study_col])
            )
            for _, row in df.iterrows()
        ]
        
        # Perform leave-one-out analysis
        loo_results = leave_one_out_analysis(points, config)
        
        # Display summary
        click.echo("\n" + "=" * 60)
        click.echo("LEAVE-ONE-OUT ANALYSIS RESULTS")
        click.echo("=" * 60)
        click.echo(f"Original Effect: {loo_results.original_result.effect:.6f}")
        click.echo(f"Maximum Effect Change: {loo_results.max_effect_change:.6f}")
        click.echo(f"Most Influential Study: {loo_results.most_influential_study}")
        
        if verbose:
            click.echo("\nDetailed Results:")
            for i, (study_id, effect_change) in enumerate(
                zip(loo_results.study_ids, loo_results.effect_changes)
            ):
                loo_result = loo_results.loo_results[i]
                click.echo(
                    f"  {study_id}: Effect = {loo_result.effect:.6f} "
                    f"(Δ = {effect_change:+.6f})"
                )
        
        # Save results if requested
        if output:
            df_results = loo_results.to_dataframe()
            df_results.to_csv(output, index=False)
            click.echo(f"LOO results saved to: {output}")
        
        # Generate plot if requested
        if plot:
            _plot_leave_one_out(loo_results, csv_file)
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option('--csv', 'csv_file', required=True, type=click.Path(exists=True),
              help='Path to CSV file containing meta-analysis data')
@click.option('--seconds', type=int, default=3600,
              help='Update interval in seconds (default: 3600 = 1 hour)')
@click.option('--effect-col', default='effect',
              help='Name of effect size column (default: effect)')
@click.option('--variance-col', default='variance',
              help='Name of variance column (default: variance)')
@click.option('--study-col', default='study',
              help='Name of study ID column (default: study)')
@click.option('--model', type=click.Choice(['FE', 'RE']), default='RE',
              help='Meta-analysis model (default: RE)')
@click.option('--tau2', type=click.Choice(['DL', 'REML', 'PM', 'ML']), default='REML',
              help='Tau² estimation method (default: REML)')
@click.option('--hksj/--no-hksj', default=False,
              help='Use HKSJ variance adjustment (default: no-hksj)')
@click.option('--output-dir', type=click.Path(),
              help='Directory to save results and plots')
@click.option('--change-threshold', type=float, default=0.1,
              help='Minimum effect change to trigger notification (default: 0.1)')
@click.option('--max-updates', type=int, default=None,
              help='Maximum number of updates (default: unlimited)')
def live(csv_file: str, seconds: int, effect_col: str, variance_col: str,
         study_col: str, model: str, tau2: str, hksj: bool,
         output_dir: Optional[str], change_threshold: float,
         max_updates: Optional[int]):
    """Start living meta-analysis with periodic updates."""
    
    try:
        click.echo(f"Starting living meta-analysis...")
        click.echo(f"Data source: {csv_file}")
        click.echo(f"Update interval: {seconds} seconds")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            click.echo(f"Output directory: {output_dir}")
        
        # Start living analysis
        live_analysis = start_living_analysis(
            data_source=csv_file,
            update_interval_seconds=seconds,
            use_hksj=hksj,
            output_dir=output_dir
        )
        
        click.echo("Living meta-analysis started successfully!")
        click.echo("Press Ctrl+C to stop...")
        
        # Monitor updates
        updates_count = 0
        try:
            while True:
                import time
                time.sleep(10)  # Check every 10 seconds
                
                status = live_analysis.get_status()
                if status['update_count'] > updates_count:
                    updates_count = status['update_count']
                    if status['last_effect'] is not None:
                        click.echo(
                            f"Update #{updates_count}: "
                            f"Effect = {status['last_effect']:.6f}"
                        )
                    
                    if max_updates and updates_count >= max_updates:
                        click.echo(f"Reached maximum updates ({max_updates}), stopping...")
                        break
                        
        except KeyboardInterrupt:
            click.echo("\nStopping living meta-analysis...")
            
        finally:
            live_analysis.stop()
            click.echo("Living meta-analysis stopped.")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


def _save_results_to_file(results: MetaResults, filepath: str, verbose: bool):
    """Save meta-analysis results to a text file."""
    try:
        with open(filepath, 'w') as f:
            f.write("PyMeta Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Method: {results.method}\n")
            f.write(f"Effect Size: {results.effect}\n")
            f.write(f"Standard Error: {results.se}\n")
            f.write(f"95% CI: [{results.ci_lower}, {results.ci_upper}]\n")
            f.write(f"P-value: {results.p_value}\n")
            f.write(f"Tau²: {results.tau2}\n")
            f.write(f"I²: {results.i2}%\n")
            f.write(f"H²: {results.h2}\n")
            f.write(f"Q: {results.q_stat} (p = {results.q_p_value})\n")
            
            if results.use_hksj and results.df is not None:
                f.write(f"Degrees of Freedom (HKSJ): {results.df}\n")
                
            if results.points:
                f.write(f"Number of Studies: {len(results.points)}\n")
        
        if verbose:
            click.echo(f"Results saved to: {filepath}")
            
    except Exception as e:
        click.echo(f"Warning: Could not save results to file: {str(e)}", err=True)


def _plot_leave_one_out(loo_results, csv_file: str):
    """Generate a plot showing leave-one-out analysis results."""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot effect changes
        studies = loo_results.study_ids
        effect_changes = loo_results.effect_changes
        
        ax1.barh(range(len(studies)), effect_changes)
        ax1.set_yticks(range(len(studies)))
        ax1.set_yticklabels(studies)
        ax1.set_xlabel('Change in Effect Size')
        ax1.set_title('Leave-One-Out: Effect Size Changes')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax1.grid(True, alpha=0.3)
        
        # Plot I² changes
        i2_changes = loo_results.i2_changes
        
        ax2.barh(range(len(studies)), i2_changes)
        ax2.set_yticks(range(len(studies)))
        ax2.set_yticklabels(studies)
        ax2.set_xlabel('Change in I² (%)')
        ax2.set_title('Leave-One-Out: I² Changes')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        base_name = Path(csv_file).stem
        plot_path = f"{base_name}_loo_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        click.echo(f"LOO plot saved to: {plot_path}")
        plt.close()
        
    except Exception as e:
        click.echo(f"Warning: Could not generate LOO plot: {str(e)}", err=True)


if __name__ == '__main__':
    main()