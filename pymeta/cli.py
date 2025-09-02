"""Enhanced CLI interface for PyMeta."""

import click
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .suite import PyMeta
from .io.datasets import create_meta_points_from_dataframe, create_example_data
from .config import config


@click.group()
@click.version_option(version="4.1-modular", prog_name="pymeta")
def main():
    """PyMeta: Comprehensive meta-analysis toolkit."""
    pass


@main.command()
@click.option('--input', '-i', type=click.Path(exists=True), 
              help='Input CSV file with meta-analysis data')
@click.option('--model', '-m', default='random_effects',
              type=click.Choice(['fixed_effects', 'random_effects', 'glmm_binomial']),
              help='Meta-analysis model type')
@click.option('--tau2', default='DL',
              type=click.Choice(['DL', 'PM', 'REML', 'ML']),
              help='Tau² estimator for random effects')
@click.option('--style', default='default',
              type=click.Choice(['default', 'publication', 'presentation']),
              help='Plot style')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for plots and results')
@click.option('--alpha', default=0.05, type=float,
              help='Significance level')
@click.option('--effect-col', default='effect',
              help='Column name for effect sizes')
@click.option('--variance-col', default='variance',
              help='Column name for variances')
@click.option('--se-col', help='Column name for standard errors (alternative to variance)')
@click.option('--label-col', help='Column name for study labels')
@click.option('--bias-tests/--no-bias-tests', default=True,
              help='Perform publication bias tests')
@click.option('--plots/--no-plots', default=True,
              help='Generate plots')
def analyze(input, model, tau2, style, output, alpha, effect_col, variance_col, 
           se_col, label_col, bias_tests, plots):
    """Perform meta-analysis on input data."""
    
    # Set plot style
    config.set_plot_style(style)
    
    try:
        # Load data
        if input:
            click.echo(f"Loading data from {input}")
            df = pd.read_csv(input)
            
            # Create MetaPoint objects
            points = create_meta_points_from_dataframe(
                df, 
                effect_col=effect_col,
                variance_col=variance_col,
                se_col=se_col,
                label_col=label_col
            )
        else:
            click.echo("Using example data (no input file specified)")
            points = create_example_data(n_studies=10, seed=42)
        
        # Create PyMeta instance
        meta = PyMeta(points, model_type=model, tau2_estimator=tau2, alpha=alpha)
        
        # Perform analysis
        click.echo(f"\nPerforming {model} meta-analysis...")
        results = meta.analyze()
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("META-ANALYSIS RESULTS")
        click.echo("="*50)
        click.echo(f"Model: {results.method}")
        click.echo(f"Studies: {results.n_studies}")
        click.echo(f"Pooled Effect: {results.pooled_effect:.4f} "
                  f"(95% CI: {results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f})")
        click.echo(f"Standard Error: {results.pooled_se:.4f}")
        click.echo(f"Z-score: {results.z_score:.4f}")
        click.echo(f"P-value: {results.p_value:.4f}")
        click.echo(f"\nHeterogeneity:")
        click.echo(f"  Tau²: {results.tau2:.4f}")
        click.echo(f"  I²: {results.i_squared:.1f}%")
        click.echo(f"  Q: {results.q_statistic:.4f} (df={results.n_studies-1}, p={results.q_p_value:.4f})")
        
        # Publication bias tests
        if bias_tests:
            click.echo(f"\nPublication Bias Tests:")
            try:
                egger_result = meta.test_bias('egger')
                click.echo(f"  Egger's test: p={egger_result.p_value:.4f}")
                click.echo(f"    {egger_result.interpretation}")
                
                begg_result = meta.test_bias('begg')
                click.echo(f"  Begg's test: p={begg_result.p_value:.4f}")
                click.echo(f"    {begg_result.interpretation}")
            except Exception as e:
                click.echo(f"  Bias tests failed: {e}")
        
        # Generate plots
        if plots:
            output_dir = Path(output) if output else Path.cwd()
            output_dir.mkdir(exist_ok=True)
            
            click.echo(f"\nGenerating plots in {output_dir}...")
            
            try:
                # Forest plot
                fig = meta.plot_forest()
                fig.savefig(output_dir / 'forest_plot.png', dpi=300, bbox_inches='tight')
                click.echo("  ✓ Forest plot saved")
                
                # Funnel plot
                fig = meta.plot_funnel()
                fig.savefig(output_dir / 'funnel_plot.png', dpi=300, bbox_inches='tight')
                click.echo("  ✓ Funnel plot saved")
                
                # Baujat plot (if enough studies)
                if len(points) >= 3:
                    fig = meta.plot_baujat()
                    fig.savefig(output_dir / 'baujat_plot.png', dpi=300, bbox_inches='tight')
                    click.echo("  ✓ Baujat plot saved")
                
            except Exception as e:
                click.echo(f"  Plot generation failed: {e}")
        
        # Save results to file
        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
            
            # Save summary report
            report = meta.summary_report()
            with open(output_dir / 'analysis_report.txt', 'w') as f:
                f.write(report)
            click.echo(f"\n✓ Analysis report saved to {output_dir / 'analysis_report.txt'}")
        
        click.echo("\nAnalysis completed successfully!")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input CSV file with meta-analysis data')
@click.option('--plot-type', '-t', 
              type=click.Choice(['forest', 'funnel', 'baujat', 'radial', 'gosh']),
              required=True, help='Type of plot to generate')
@click.option('--output', '-o', type=click.Path(),
              help='Output file path for plot')
@click.option('--style', default='default',
              type=click.Choice(['default', 'publication', 'presentation']),
              help='Plot style')
@click.option('--effect-col', default='effect',
              help='Column name for effect sizes')
@click.option('--variance-col', default='variance',
              help='Column name for variances')
@click.option('--se-col', help='Column name for standard errors')
@click.option('--label-col', help='Column name for study labels')
@click.option('--gosh-subsets', default=1000, type=int,
              help='Number of subsets for GOSH plot')
@click.option('--gosh-seed', type=int,
              help='Random seed for GOSH sampling')
def plot(input, plot_type, output, style, effect_col, variance_col, 
         se_col, label_col, gosh_subsets, gosh_seed):
    """Generate specific plot types."""
    
    # Set plot style
    config.set_plot_style(style)
    
    try:
        # Load data
        click.echo(f"Loading data from {input}")
        df = pd.read_csv(input)
        
        # Create MetaPoint objects
        points = create_meta_points_from_dataframe(
            df,
            effect_col=effect_col,
            variance_col=variance_col,
            se_col=se_col,
            label_col=label_col
        )
        
        # Create PyMeta instance
        meta = PyMeta(points)
        
        # Generate requested plot
        click.echo(f"Generating {plot_type} plot...")
        
        if plot_type == 'forest':
            results = meta.analyze()
            fig = meta.plot_forest()
        elif plot_type == 'funnel':
            fig = meta.plot_funnel()
        elif plot_type == 'baujat':
            if len(points) < 3:
                raise click.ClickException("Baujat plot requires at least 3 studies")
            fig = meta.plot_baujat()
        elif plot_type == 'radial':
            fig = meta.plot_radial()
        elif plot_type == 'gosh':
            if len(points) < 4:
                raise click.ClickException("GOSH plot requires at least 4 studies")
            fig = meta.plot_gosh(max_subsets=gosh_subsets, seed=gosh_seed)
        
        # Save plot
        if output:
            fig.savefig(output, dpi=300, bbox_inches='tight')
            click.echo(f"✓ Plot saved to {output}")
        else:
            # Show plot
            import matplotlib.pyplot as plt
            plt.show()
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option('--n-studies', default=10, type=int,
              help='Number of studies to simulate')
@click.option('--true-effect', default=0.5, type=float,
              help='True underlying effect size')
@click.option('--tau2', default=0.1, type=float,
              help='Between-study variance')
@click.option('--seed', type=int,
              help='Random seed for reproducibility')
@click.option('--output', '-o', type=click.Path(),
              help='Output CSV file for simulated data')
def simulate(n_studies, true_effect, tau2, seed, output):
    """Generate simulated meta-analysis data."""
    
    try:
        click.echo(f"Simulating {n_studies} studies...")
        click.echo(f"True effect: {true_effect}")
        click.echo(f"Between-study variance (tau²): {tau2}")
        
        # Generate data
        points = create_example_data(n_studies, true_effect, tau2, seed)
        
        # Convert to DataFrame
        data = []
        for i, point in enumerate(points):
            data.append({
                'study_id': point.study_id,
                'study_label': point.label,
                'effect': point.effect,
                'variance': point.variance,
                'standard_error': np.sqrt(point.variance),
                'weight': point.weight
            })
        
        df = pd.DataFrame(data)
        
        # Display summary
        click.echo(f"\nSimulated Data Summary:")
        click.echo(f"  Mean effect: {df['effect'].mean():.4f}")
        click.echo(f"  Effect range: {df['effect'].min():.4f} to {df['effect'].max():.4f}")
        click.echo(f"  Mean SE: {df['standard_error'].mean():.4f}")
        
        # Save to file
        if output:
            df.to_csv(output, index=False)
            click.echo(f"✓ Simulated data saved to {output}")
        else:
            click.echo("\nFirst 5 rows:")
            click.echo(df.head().to_string(index=False))
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@main.command()
def info():
    """Display package information and available options."""
    
    click.echo(f"PyMeta {config.version}")
    click.echo("="*30)
    click.echo("A comprehensive modular meta-analysis package")
    click.echo()
    
    click.echo("Available Models:")
    for model in ['fixed_effects', 'random_effects', 'glmm_binomial']:
        click.echo(f"  • {model}")
    click.echo()
    
    click.echo("Available Tau² Estimators:")
    for estimator in ['DL', 'PM', 'REML', 'ML']:
        click.echo(f"  • {estimator}")
    click.echo()
    
    click.echo("Available Plot Styles:")
    for style in ['default', 'publication', 'presentation']:
        click.echo(f"  • {style}")
    click.echo()
    
    click.echo("Available Plot Types:")
    for plot_type in ['forest', 'funnel', 'baujat', 'radial', 'gosh']:
        click.echo(f"  • {plot_type}")
    click.echo()
    
    click.echo("Examples:")
    click.echo("  # Analyze data with random effects model")
    click.echo("  pymeta analyze -i data.csv -m random_effects --tau2 REML")
    click.echo()
    click.echo("  # Generate forest plot")
    click.echo("  pymeta plot -i data.csv -t forest -o forest.png")
    click.echo()
    click.echo("  # Simulate example data")
    click.echo("  pymeta simulate --n-studies 15 --output simulated.csv")


if __name__ == '__main__':
    main()