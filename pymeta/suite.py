"""PyMeta orchestrator with integrated functionality."""

import numpy as np
import warnings
from typing import List, Optional, Dict, Any, Union, Tuple
import matplotlib.pyplot as plt

from .typing import MetaPoint, MetaResults, BiasTestResult, TSAResult
from .models.fixed_effects import FixedEffects
from .models.random_effects import RandomEffects
from .models.glmm_binomial import GLMMBinomial
from .bias.egger import egger_regression_test, begg_mazumdar_test, funnel_plot_asymmetry_test
from .tsa.cumulative import perform_tsa
from .plots.publication_forest import forest_plot
from .plots.funnel import funnel_plot
from .plots.baujat import baujat_plot
from .plots.radial import radial_plot
from .plots.gosh import gosh_plot
from .io.datasets import validate_meta_data, create_example_data
from .registries import get_model, list_models, list_estimators, list_bias_tests
from .errors import PyMetaError, ModelError, ValidationError
from .config import config


class PyMeta:
    """Main PyMeta orchestrator for comprehensive meta-analysis.
    
    Provides a high-level interface for conducting meta-analyses with
    integrated model fitting, bias detection, TSA, and plotting.
    """
    
    def __init__(self, 
                 data: Union[List[MetaPoint], np.ndarray] = None,
                 model_type: str = "random_effects",
                 tau2_estimator: str = None,
                 alpha: float = None):
        """Initialize PyMeta suite.
        
        Args:
            data: Meta-analysis data (MetaPoints, array, or None for examples)
            model_type: Type of meta-analysis model
            tau2_estimator: Tau² estimator for random effects models
            alpha: Significance level
        """
        self.alpha = alpha or config.alpha
        self.model_type = model_type
        self.tau2_estimator = tau2_estimator or config.default_tau2_estimator
        
        # Data handling
        if data is None:
            # Create example data
            self.points = create_example_data(n_studies=8, seed=42)
        else:
            self.points = validate_meta_data(data)
        
        # Analysis results
        self._model = None
        self._results: Optional[MetaResults] = None
        self._bias_tests: Dict[str, BiasTestResult] = {}
        self._tsa_result: Optional[TSAResult] = None
        
        # Generated advice
        self._advice: List[str] = []
    
    def analyze(self, 
               model_type: str = None,
               tau2_estimator: str = None,
               return_model: bool = False) -> Union[MetaResults, Tuple[MetaResults, Any]]:
        """Perform meta-analysis with specified model.
        
        Args:
            model_type: Override default model type
            tau2_estimator: Override default tau² estimator
            return_model: Whether to return model object along with results
            
        Returns:
            MetaResults object, optionally with model
        """
        # Use provided parameters or defaults
        model_type = model_type or self.model_type
        tau2_estimator = tau2_estimator or self.tau2_estimator
        
        try:
            # Create appropriate model
            if model_type in ["fixed_effects", "FE"]:
                self._model = FixedEffects(self.points, alpha=self.alpha)
            elif model_type in ["random_effects", "RE", "DL", "REML", "PM"]:
                # Handle specific estimator models
                if model_type in ["DL", "REML", "PM"]:
                    tau2_estimator = model_type
                    model_type = "random_effects"
                self._model = RandomEffects(self.points, tau2_estimator, self.alpha)
            elif model_type in ["glmm_binomial", "GLMM"]:
                self._model = GLMMBinomial(self.points, alpha=self.alpha)
            else:
                # Try registry
                try:
                    model_class = get_model(model_type)
                    self._model = model_class(self.points, alpha=self.alpha)
                except:
                    raise ModelError(f"Unknown model type: {model_type}")
            
            # Fit model
            self._results = self._model.fit()
            
            # Generate advice
            self._generate_advice()
            
            if return_model:
                return self._results, self._model
            return self._results
            
        except Exception as e:
            raise PyMetaError(f"Analysis failed: {e}")
    
    def test_bias(self, 
                  test_name: str = "egger",
                  **kwargs) -> BiasTestResult:
        """Perform publication bias test.
        
        Args:
            test_name: Name of bias test ("egger", "begg", "all")
            **kwargs: Additional arguments for bias test
            
        Returns:
            BiasTestResult object or dict for "all"
        """
        try:
            if test_name == "egger":
                result = egger_regression_test(self.points, **kwargs)
                self._bias_tests["egger"] = result
                return result
            elif test_name == "begg":
                result = begg_mazumdar_test(self.points, **kwargs)
                self._bias_tests["begg"] = result
                return result
            elif test_name == "all":
                return funnel_plot_asymmetry_test(self.points, **kwargs)
            else:
                # Try registry
                from .registries import get_bias_test
                test_func = get_bias_test(test_name)
                result = test_func(self.points, **kwargs)
                self._bias_tests[test_name] = result
                return result
                
        except Exception as e:
            raise PyMetaError(f"Bias test failed: {e}")
    
    def perform_tsa(self,
                   delta: float,
                   alpha: float = None,
                   beta: float = None,
                   model_type: str = None,
                   boundary_type: str = "obrien_fleming") -> TSAResult:
        """Perform Trial Sequential Analysis.
        
        Args:
            delta: Clinically relevant effect size
            alpha: Type I error rate
            beta: Type II error rate  
            model_type: Model for cumulative analysis
            boundary_type: Monitoring boundary type
            
        Returns:
            TSAResult object
        """
        try:
            alpha = alpha or self.alpha
            beta = beta or config.tsa_beta
            model_type = model_type or "fixed_effects"
            
            self._tsa_result = perform_tsa(
                self.points, delta, alpha, beta, model_type, boundary_type
            )
            
            return self._tsa_result
            
        except Exception as e:
            raise PyMetaError(f"TSA failed: {e}")
    
    def plot_forest(self, **kwargs) -> plt.Figure:
        """Create forest plot."""
        if self._results is None:
            self.analyze()
        
        return forest_plot(self.points, self._results, **kwargs)
    
    def plot_funnel(self, **kwargs) -> plt.Figure:
        """Create funnel plot."""
        return funnel_plot(self.points, self._results, **kwargs)
    
    def plot_baujat(self, **kwargs) -> plt.Figure:
        """Create Baujat plot."""
        return baujat_plot(self.points, self._results, **kwargs)
    
    def plot_radial(self, **kwargs) -> plt.Figure:
        """Create radial plot."""
        return radial_plot(self.points, self._results, **kwargs)
    
    def plot_gosh(self, **kwargs) -> plt.Figure:
        """Create GOSH plot."""
        return gosh_plot(self.points, **kwargs)
    
    def plot_tsa(self, tsa_result: TSAResult = None, **kwargs) -> plt.Figure:
        """Create TSA plot."""
        if tsa_result is None:
            tsa_result = self._tsa_result
        
        if tsa_result is None:
            raise PyMetaError("No TSA results available. Run perform_tsa() first.")
        
        return self._plot_tsa_implementation(tsa_result, **kwargs)
    
    def _plot_tsa_implementation(self, tsa_result: TSAResult, **kwargs) -> plt.Figure:
        """Implementation of TSA plotting."""
        from .tsa.cumulative import tsa_plot_data
        
        # Get plot style
        plot_style = config.get_plot_style(kwargs.get('style'))
        figsize = kwargs.get('figsize', plot_style.figure_size)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=plot_style.dpi)
        
        # Get plot data
        plot_data = tsa_plot_data(tsa_result)
        
        # Plot cumulative Z-scores
        info_fractions = plot_data['information_fractions']
        cumulative_z = plot_data['cumulative_z']
        
        ax.plot(info_fractions, cumulative_z, 'b-o', linewidth=2, markersize=6,
               label='Cumulative Z-score')
        
        # Plot boundaries
        boundaries = plot_data['boundaries']
        if 'superiority_upper' in boundaries:
            ax.plot(info_fractions, boundaries['superiority_upper'], 'r--',
                   linewidth=2, label='Superiority Boundary')
            ax.plot(info_fractions, [-b for b in boundaries['superiority_upper']], 'r--',
                   linewidth=2)
        
        if 'futility' in boundaries:
            ax.plot(info_fractions, boundaries['futility'], 'g:',
                   linewidth=2, label='Futility Boundary')
        
        # Add reference lines
        ax.axhline(y=1.96, color='gray', linestyle='-', alpha=0.5, label='Conventional Significance')
        ax.axhline(y=-1.96, color='gray', linestyle='-', alpha=0.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Labels and formatting
        ax.set_xlabel('Information Fraction', fontsize=plot_style.font_size)
        ax.set_ylabel('Cumulative Z-score', fontsize=plot_style.font_size)
        ax.set_title('Trial Sequential Analysis', fontsize=plot_style.font_size + 2, fontweight='bold')
        
        # Grid and legend
        if plot_style.grid:
            ax.grid(True, alpha=plot_style.grid_alpha)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def summary_report(self, include_plots: bool = False) -> str:
        """Generate comprehensive summary report.
        
        Args:
            include_plots: Whether to create and save plots
            
        Returns:
            Formatted summary report string
        """
        if self._results is None:
            self.analyze()
        
        lines = [
            "PyMeta Analysis Report",
            "=" * 50,
            "",
            f"Dataset: {len(self.points)} studies",
            f"Model: {self._results.method}",
            f"Analysis Date: {self._get_current_date()}",
            "",
            "Results:",
            f"  Pooled Effect: {self._results.pooled_effect:.4f} "
            f"(95% CI: {self._results.confidence_interval[0]:.4f}, {self._results.confidence_interval[1]:.4f})",
            f"  Z-score: {self._results.z_score:.4f}",
            f"  P-value: {self._results.p_value:.4f}",
            "",
            "Heterogeneity:",
            f"  Tau²: {self._results.tau2:.4f}",
            f"  I²: {self._results.i_squared:.1f}%",
            f"  Q: {self._results.q_statistic:.4f} (df={self._results.n_studies-1}, p={self._results.q_p_value:.4f})",
            ""
        ]
        
        # Add bias test results
        if self._bias_tests:
            lines.append("Publication Bias Tests:")
            for test_name, result in self._bias_tests.items():
                if isinstance(result, BiasTestResult):
                    lines.append(f"  {result.test_name}: p={result.p_value:.4f}")
                    lines.append(f"    {result.interpretation}")
            lines.append("")
        
        # Add TSA results
        if self._tsa_result:
            lines.append("Trial Sequential Analysis:")
            lines.append(f"  Required Information Size: {self._tsa_result.required_information_size:.1f}")
            lines.append(f"  Monitoring Boundary Reached: {self._tsa_result.monitoring_boundary_reached}")
            if self._tsa_result.superiority_reached:
                lines.append("  Superiority boundary crossed - consider stopping for efficacy")
            if self._tsa_result.futility_reached:
                lines.append("  Futility boundary crossed - consider stopping for futility")
            lines.append("")
        
        # Add automated advice
        if self._advice:
            lines.append("Automated Advice:")
            for advice in self._advice:
                lines.append(f"  • {advice}")
            lines.append("")
        
        # Study-level information
        lines.append("Study Information:")
        for i, point in enumerate(self.points):
            label = point.label or f"Study {i+1}"
            lines.append(f"  {label}: Effect={point.effect:.3f}, SE={np.sqrt(point.variance):.3f}")
        
        return "\n".join(lines)
    
    def _generate_advice(self):
        """Generate automated analysis advice."""
        self._advice = []
        
        if self._results is None:
            return
        
        # Sample size advice
        if len(self.points) < 5:
            self._advice.append("Small number of studies - results should be interpreted cautiously")
        
        # Heterogeneity advice
        if self._results.i_squared > 75:
            self._advice.append("High heterogeneity (I² > 75%) - consider subgroup analysis or meta-regression")
        elif self._results.i_squared > 50:
            self._advice.append("Moderate heterogeneity (I² > 50%) - random effects model recommended")
        
        # Significance advice
        if self._results.p_value < 0.05:
            self._advice.append("Statistically significant result - consider clinical significance")
        else:
            self._advice.append("Non-significant result - consider power and confidence intervals")
        
        # Tau² advice
        if self._results.tau2 > 0.1 and self.model_type == "fixed_effects":
            self._advice.append("Substantial between-study variance - consider random effects model")
        
        # Bias test advice
        if any(test.p_value < 0.10 for test in self._bias_tests.values() 
               if isinstance(test, BiasTestResult)):
            self._advice.append("Potential publication bias detected - interpret results cautiously")
    
    def _get_current_date(self) -> str:
        """Get current date string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_results(self) -> MetaResults:
        """Get analysis results, running analysis if needed."""
        if self._results is None:
            self.analyze()
        return self._results
    
    def available_models(self) -> List[str]:
        """List available model types."""
        return list_models()
    
    def available_estimators(self) -> List[str]:
        """List available tau² estimators."""
        return list_estimators()
    
    def available_bias_tests(self) -> List[str]:
        """List available bias tests."""
        return list_bias_tests()
    
    def model_comparison(self, 
                        models: List[str] = None) -> Dict[str, MetaResults]:
        """Compare results across different models.
        
        Args:
            models: List of model names to compare
            
        Returns:
            Dictionary mapping model names to results
        """
        if models is None:
            models = ["fixed_effects", "random_effects"]
        
        comparison = {}
        original_model_type = self.model_type
        
        for model_name in models:
            try:
                self.model_type = model_name
                self._model = None
                self._results = None
                result = self.analyze()
                comparison[model_name] = result
            except Exception as e:
                comparison[model_name] = f"Error: {e}"
        
        # Restore original model
        self.model_type = original_model_type
        self._model = None
        self._results = None
        
        return comparison