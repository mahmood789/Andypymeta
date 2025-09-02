"""
PyMeta suite module for comprehensive meta-analysis integration.

This module wires together all components of the PyMeta package including:
- HKSJ variance adjustment
- Influence diagnostics
- Advanced plotting
- Living meta-analysis
- Configuration management
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from . import (
    MetaAnalysisConfig,
    MetaPoint,
    MetaResults,
    analyze_data,
    analyze_csv
)
from .plots import plot_forest, plot_funnel, plot_funnel_contour
from .diagnostics import leave_one_out_analysis, influence_measures, identify_outliers
from .living import start_living_analysis, LiveMetaAnalysis, SchedulerConfig
from .config import get_config, set_config, GlobalConfig


class MetaAnalysisSuite:
    """
    Comprehensive meta-analysis suite with integrated functionality.
    
    This class provides a high-level interface for conducting complete
    meta-analyses with all available features including HKSJ adjustment,
    influence diagnostics, advanced plotting, and results export.
    """
    
    def __init__(self, config: Optional[MetaAnalysisConfig] = None):
        """
        Initialize meta-analysis suite.
        
        Args:
            config: Meta-analysis configuration (uses defaults if None)
        """
        self.config = config or MetaAnalysisConfig()
        self.results: Optional[MetaResults] = None
        self.data: Optional[pd.DataFrame] = None
        self.points: Optional[List[MetaPoint]] = None
    
    def load_csv(self, filepath: str, effect_col: str = "effect", 
                 variance_col: str = "variance", study_col: str = "study") -> None:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            effect_col: Name of effect size column
            variance_col: Name of variance column
            study_col: Name of study ID column
        """
        self.data = pd.read_csv(filepath)
        
        # Validate columns
        required_cols = [effect_col, variance_col, study_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
        
        # Create MetaPoint objects
        self.points = [
            MetaPoint(
                effect=row[effect_col],
                variance=row[variance_col],
                study_id=str(row[study_col])
            )
            for _, row in self.data.iterrows()
        ]
    
    def load_arrays(self, effects: np.ndarray, variances: np.ndarray,
                   study_ids: Optional[List[str]] = None) -> None:
        """
        Load data from arrays.
        
        Args:
            effects: Array of effect sizes
            variances: Array of variances
            study_ids: Optional study identifiers
        """
        if study_ids is None:
            study_ids = [f"Study_{i+1}" for i in range(len(effects))]
        
        self.points = [
            MetaPoint(effect=e, variance=v, study_id=sid)
            for e, v, sid in zip(effects, variances, study_ids)
        ]
        
        # Create DataFrame for convenience
        self.data = pd.DataFrame({
            'study': study_ids,
            'effect': effects,
            'variance': variances
        })
    
    def analyze(self, config: Optional[MetaAnalysisConfig] = None) -> MetaResults:
        """
        Perform meta-analysis.
        
        Args:
            config: Optional configuration override
            
        Returns:
            MetaResults object
        """
        if self.points is None:
            raise ValueError("No data loaded. Use load_csv() or load_arrays() first.")
        
        config = config or self.config
        
        effects = np.array([p.effect for p in self.points])
        variances = np.array([p.variance for p in self.points])
        study_ids = [p.study_id for p in self.points]
        
        self.results = analyze_data(effects, variances, study_ids, config)
        return self.results
    
    def plot_forest(self, **kwargs) -> Any:
        """Generate forest plot."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        return plot_forest(self.results, **kwargs)
    
    def plot_funnel(self, **kwargs) -> Any:
        """Generate funnel plot."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        return plot_funnel(self.results, **kwargs)
    
    def plot_funnel_contour(self, **kwargs) -> Any:
        """Generate contour-enhanced funnel plot."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        return plot_funnel_contour(self.results, **kwargs)
    
    def plot_all(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate all available plots.
        
        Args:
            save_dir: Optional directory to save plots
            
        Returns:
            Dictionary of plot figures
        """
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        plots = {
            'forest': self.plot_forest(),
            'funnel': self.plot_funnel(),
            'funnel_contour': self.plot_funnel_contour()
        }
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            for plot_name, fig in plots.items():
                save_path = os.path.join(save_dir, f"{plot_name}.png")
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plots
    
    def leave_one_out(self, config: Optional[MetaAnalysisConfig] = None):
        """
        Perform leave-one-out analysis.
        
        Args:
            config: Optional configuration override
            
        Returns:
            LeaveOneOutResult object
        """
        if self.points is None:
            raise ValueError("No data loaded")
        
        config = config or self.config
        return leave_one_out_analysis(self.points, config)
    
    def influence_analysis(self, config: Optional[MetaAnalysisConfig] = None):
        """
        Perform influence analysis.
        
        Args:
            config: Optional configuration override
            
        Returns:
            List of InfluenceResult objects
        """
        if self.points is None or self.results is None:
            raise ValueError("Run analyze() first")
        
        config = config or self.config
        return influence_measures(self.points, self.results, config)
    
    def identify_outliers(self, **kwargs):
        """
        Identify potential outliers.
        
        Args:
            **kwargs: Arguments passed to identify_outliers function
            
        Returns:
            Dictionary of outlier classifications
        """
        influence_results = self.influence_analysis()
        return identify_outliers(influence_results, **kwargs)
    
    def comprehensive_analysis(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis including all available methods.
        
        Args:
            save_dir: Optional directory to save results and plots
            
        Returns:
            Dictionary containing all analysis results
        """
        if self.points is None:
            raise ValueError("No data loaded")
        
        # Main analysis
        results = self.analyze()
        
        # Generate plots
        plots = self.plot_all(save_dir)
        
        # Diagnostic analyses
        diagnostics = {}
        
        if len(self.points) >= 3:
            diagnostics['leave_one_out'] = self.leave_one_out()
            diagnostics['influence'] = self.influence_analysis()
            diagnostics['outliers'] = self.identify_outliers()
        
        # Compare with/without HKSJ if using RE model
        comparisons = {}
        if self.config.model == "RE":
            # Standard RE
            config_standard = MetaAnalysisConfig(
                model=self.config.model,
                tau2_method=self.config.tau2_method,
                use_hksj=False,
                alpha=self.config.alpha
            )
            
            # HKSJ RE
            config_hksj = MetaAnalysisConfig(
                model=self.config.model,
                tau2_method=self.config.tau2_method,
                use_hksj=True,
                alpha=self.config.alpha
            )
            
            comparisons['standard_re'] = self.analyze(config_standard)
            comparisons['hksj_re'] = self.analyze(config_hksj)
        
        # Save results if directory specified
        if save_dir:
            self._save_comprehensive_results(save_dir, results, diagnostics, comparisons)
        
        return {
            'main_results': results,
            'plots': plots,
            'diagnostics': diagnostics,
            'comparisons': comparisons
        }
    
    def _save_comprehensive_results(self, save_dir: str, results: MetaResults,
                                   diagnostics: Dict[str, Any], 
                                   comparisons: Dict[str, MetaResults]) -> None:
        """Save comprehensive results to files."""
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save main results
        with open(os.path.join(save_dir, "main_results.txt"), 'w') as f:
            f.write(f"PyMeta Comprehensive Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Method: {results.method}\n")
            f.write(f"Effect: {results.effect:.6f}\n")
            f.write(f"SE: {results.se:.6f}\n")
            f.write(f"95% CI: [{results.ci_lower:.6f}, {results.ci_upper:.6f}]\n")
            f.write(f"P-value: {results.p_value:.6f}\n")
            f.write(f"Tau²: {results.tau2:.6f}\n")
            f.write(f"I²: {results.i2:.2f}%\n")
            f.write(f"Q: {results.q_stat:.6f} (p = {results.q_p_value:.6f})\n")
            
            if results.use_hksj and results.df is not None:
                f.write(f"HKSJ df: {results.df}\n")
        
        # Save diagnostic results
        if 'leave_one_out' in diagnostics:
            loo_df = diagnostics['leave_one_out'].to_dataframe()
            loo_df.to_csv(os.path.join(save_dir, "leave_one_out.csv"), index=False)
        
        if 'influence' in diagnostics:
            influence_data = []
            for inf in diagnostics['influence']:
                influence_data.append({
                    'study_id': inf.study_id,
                    'effect': inf.effect,
                    'variance': inf.variance,
                    'weight': inf.weight,
                    'std_residual': inf.standardized_residual,
                    'leverage': inf.leverage,
                    'cook_distance': inf.cook_distance,
                    'dffits': inf.dffits,
                    'dfbetas': inf.dfbetas
                })
            
            influence_df = pd.DataFrame(influence_data)
            influence_df.to_csv(os.path.join(save_dir, "influence_measures.csv"), index=False)
        
        # Save comparisons
        if comparisons:
            with open(os.path.join(save_dir, "method_comparison.txt"), 'w') as f:
                f.write("Method Comparison\n")
                f.write("=" * 30 + "\n\n")
                
                for method_name, method_results in comparisons.items():
                    f.write(f"{method_name.upper()}:\n")
                    f.write(f"  Effect: {method_results.effect:.6f}\n")
                    f.write(f"  SE: {method_results.se:.6f}\n")
                    f.write(f"  95% CI: [{method_results.ci_lower:.6f}, {method_results.ci_upper:.6f}]\n")
                    f.write(f"  CI Width: {method_results.ci_width:.6f}\n")
                    f.write(f"  P-value: {method_results.p_value:.6f}\n\n")
    
    def start_living_analysis(self, csv_path: str, 
                             update_interval_seconds: int = 3600,
                             output_dir: Optional[str] = None,
                             **kwargs) -> LiveMetaAnalysis:
        """
        Start a living meta-analysis.
        
        Args:
            csv_path: Path to CSV file for monitoring
            update_interval_seconds: Update frequency
            output_dir: Directory for saving results
            **kwargs: Additional scheduler configuration
            
        Returns:
            LiveMetaAnalysis instance
        """
        return start_living_analysis(
            data_source=csv_path,
            update_interval_seconds=update_interval_seconds,
            use_hksj=self.config.use_hksj,
            output_dir=output_dir,
            **kwargs
        )
    
    def summary(self) -> str:
        """
        Generate a text summary of current analysis.
        
        Returns:
            Formatted summary string
        """
        if self.results is None:
            return "No analysis results available. Run analyze() first."
        
        summary = f"""
PyMeta Analysis Summary
======================

Method: {self.results.method}
Studies: {len(self.results.points) if self.results.points else 'N/A'}

Effect Size: {self.results.effect:.4f} [{self.results.ci_lower:.4f}, {self.results.ci_upper:.4f}]
P-value: {self.results.p_value:.4f}

Heterogeneity:
- Tau²: {self.results.tau2:.4f}
- I²: {self.results.i2:.1f}%
- Q: {self.results.q_stat:.2f} (p = {self.results.q_p_value:.4f})

"""
        
        if self.results.use_hksj and self.results.df is not None:
            summary += f"HKSJ Adjustment: df = {self.results.df}\n"
        
        # Interpretation
        if self.results.p_value < 0.05:
            summary += "\nInterpretation: Statistically significant effect detected.\n"
        else:
            summary += "\nInterpretation: No statistically significant effect detected.\n"
        
        if self.results.i2 < 25:
            summary += "Heterogeneity: Low (I² < 25%)\n"
        elif self.results.i2 < 75:
            summary += "Heterogeneity: Moderate (25% ≤ I² < 75%)\n"
        else:
            summary += "Heterogeneity: High (I² ≥ 75%)\n"
        
        return summary


# Convenience functions for quick analysis
def quick_analysis(csv_path: str, use_hksj: bool = False, 
                  save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform quick comprehensive analysis from CSV file.
    
    Args:
        csv_path: Path to CSV file
        use_hksj: Whether to use HKSJ adjustment
        save_dir: Optional directory to save results
        
    Returns:
        Dictionary with all analysis results
    """
    config = MetaAnalysisConfig(use_hksj=use_hksj)
    suite = MetaAnalysisSuite(config)
    suite.load_csv(csv_path)
    
    return suite.comprehensive_analysis(save_dir)


def compare_methods(csv_path: str, save_dir: Optional[str] = None) -> Dict[str, MetaResults]:
    """
    Compare different meta-analysis methods.
    
    Args:
        csv_path: Path to CSV file
        save_dir: Optional directory to save results
        
    Returns:
        Dictionary with results from different methods
    """
    results = {}
    
    # Different configurations to compare
    configs = {
        'fixed_effects': MetaAnalysisConfig(model="FE"),
        'random_effects_dl': MetaAnalysisConfig(model="RE", tau2_method="DL"),
        'random_effects_reml': MetaAnalysisConfig(model="RE", tau2_method="REML"),
        'random_effects_hksj': MetaAnalysisConfig(model="RE", tau2_method="REML", use_hksj=True)
    }
    
    for method_name, config in configs.items():
        suite = MetaAnalysisSuite(config)
        suite.load_csv(csv_path)
        results[method_name] = suite.analyze()
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save comparison table
        comparison_data = []
        for method, result in results.items():
            comparison_data.append({
                'Method': result.method,
                'Effect': result.effect,
                'SE': result.se,
                'CI_Lower': result.ci_lower,
                'CI_Upper': result.ci_upper,
                'CI_Width': result.ci_width,
                'P_Value': result.p_value,
                'Tau2': result.tau2,
                'I2': result.i2
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(os.path.join(save_dir, "method_comparison.csv"), index=False)
    
    return results