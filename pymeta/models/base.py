"""Base class for meta-analysis models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from scipy import stats

from ..typing import MetaPoint, MetaResults
from ..errors import ModelError, ValidationError
from ..config import config


class MetaModel(ABC):
    """Base class for meta-analysis models."""
    
    def __init__(self, points: List[MetaPoint], alpha: float = None):
        """Initialize meta-analysis model.
        
        Args:
            points: List of MetaPoint objects
            alpha: Significance level (default from config)
        """
        self.points = self._validate_points(points)
        self.alpha = alpha or config.alpha
        self.confidence_level = 1 - self.alpha
        self._results: Optional[MetaResults] = None
    
    def _validate_points(self, points: List[MetaPoint]) -> List[MetaPoint]:
        """Validate input points."""
        if not points:
            raise ValidationError("No data points provided")
        
        if len(points) < 2:
            raise ValidationError("At least 2 studies required for meta-analysis")
        
        # Check for valid effect sizes and variances
        for i, point in enumerate(points):
            if not np.isfinite(point.effect):
                raise ValidationError(f"Invalid effect size in study {i + 1}")
            if not np.isfinite(point.variance) or point.variance <= 0:
                raise ValidationError(f"Invalid variance in study {i + 1}")
        
        return points
    
    @abstractmethod
    def fit(self) -> MetaResults:
        """Fit the meta-analysis model.
        
        Returns:
            MetaResults object with fitted parameters
        """
        pass
    
    def get_results(self) -> MetaResults:
        """Get results, fitting model if necessary.
        
        Returns:
            MetaResults object
        """
        if self._results is None:
            self._results = self.fit()
        return self._results
    
    def _calculate_common_statistics(self, pooled_effect: float, pooled_variance: float,
                                   tau2: float = 0.0, method: str = "unknown") -> MetaResults:
        """Calculate common meta-analysis statistics.
        
        Args:
            pooled_effect: Pooled effect estimate
            pooled_variance: Variance of pooled effect
            tau2: Between-study variance
            method: Method name for results
            
        Returns:
            MetaResults object with calculated statistics
        """
        # Basic statistics
        pooled_se = np.sqrt(pooled_variance)
        z_score = pooled_effect / pooled_se if pooled_se > 0 else 0.0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = pooled_effect - z_critical * pooled_se
        ci_upper = pooled_effect + z_critical * pooled_se
        
        # Heterogeneity statistics
        heterogeneity_stats = self._calculate_heterogeneity_statistics(tau2)
        
        return MetaResults(
            pooled_effect=pooled_effect,
            pooled_variance=pooled_variance,
            confidence_interval=(ci_lower, ci_upper),
            z_score=z_score,
            p_value=p_value,
            tau2=tau2,
            i_squared=heterogeneity_stats['i_squared'],
            q_statistic=heterogeneity_stats['q_statistic'],
            q_p_value=heterogeneity_stats['q_p_value'],
            n_studies=len(self.points),
            method=method,
            heterogeneity_test=heterogeneity_stats
        )
    
    def _calculate_heterogeneity_statistics(self, tau2: float) -> Dict[str, Any]:
        """Calculate heterogeneity statistics.
        
        Args:
            tau2: Between-study variance
            
        Returns:
            Dictionary with heterogeneity statistics
        """
        # Extract data
        effects = np.array([p.effect for p in self.points])
        weights = np.array([p.weight for p in self.points])
        k = len(self.points)
        df = k - 1
        
        # Calculate Q statistic
        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        q_statistic = np.sum(weights * (effects - weighted_mean) ** 2)
        
        # Q test p-value
        q_p_value = 1 - stats.chi2.cdf(q_statistic, df) if df > 0 else 1.0
        
        # I² statistic
        if q_statistic > df and df > 0:
            i_squared = 100 * (q_statistic - df) / q_statistic
        else:
            i_squared = 0.0
        
        # H² statistic
        h_squared = q_statistic / df if df > 0 else 1.0
        
        # Prediction interval (for random effects)
        if tau2 > 0:
            pred_se = np.sqrt(np.sqrt(tau2) ** 2 + np.mean([p.variance for p in self.points]))
            z_critical = stats.norm.ppf(1 - self.alpha / 2)
            pred_lower = weighted_mean - z_critical * pred_se
            pred_upper = weighted_mean + z_critical * pred_se
        else:
            pred_lower = pred_upper = None
        
        return {
            'q_statistic': q_statistic,
            'q_p_value': q_p_value,
            'i_squared': i_squared,
            'h_squared': h_squared,
            'tau2': tau2,
            'tau': np.sqrt(tau2),
            'prediction_interval': (pred_lower, pred_upper) if pred_lower is not None else None,
            'degrees_of_freedom': df
        }
    
    def influence_analysis(self) -> Dict[str, Any]:
        """Perform influence analysis (leave-one-out).
        
        Returns:
            Dictionary with influence statistics
        """
        if len(self.points) < 3:
            raise ModelError("At least 3 studies required for influence analysis")
        
        original_result = self.get_results()
        influence_results = []
        
        for i in range(len(self.points)):
            # Create subset without study i
            subset_points = [p for j, p in enumerate(self.points) if j != i]
            
            # Fit model on subset
            subset_model = self.__class__(subset_points, alpha=self.alpha)
            subset_result = subset_model.fit()
            
            # Calculate influence statistics
            effect_change = subset_result.pooled_effect - original_result.pooled_effect
            tau2_change = subset_result.tau2 - original_result.tau2
            
            influence_results.append({
                'study_index': i,
                'study_label': self.points[i].label,
                'effect_change': effect_change,
                'tau2_change': tau2_change,
                'pooled_effect': subset_result.pooled_effect,
                'pooled_variance': subset_result.pooled_variance,
                'tau2': subset_result.tau2,
                'i_squared': subset_result.i_squared
            })
        
        return {
            'original_result': original_result,
            'influence_results': influence_results,
            'max_effect_change': max(abs(r['effect_change']) for r in influence_results),
            'max_tau2_change': max(abs(r['tau2_change']) for r in influence_results)
        }
    
    def cumulative_analysis(self) -> List[MetaResults]:
        """Perform cumulative meta-analysis.
        
        Returns:
            List of MetaResults for cumulative analyses
        """
        if len(self.points) < 2:
            raise ModelError("At least 2 studies required for cumulative analysis")
        
        cumulative_results = []
        
        for i in range(2, len(self.points) + 1):
            # Fit model on first i studies
            subset_points = self.points[:i]
            subset_model = self.__class__(subset_points, alpha=self.alpha)
            subset_result = subset_model.fit()
            cumulative_results.append(subset_result)
        
        return cumulative_results
    
    def bootstrap_confidence_interval(self, n_bootstrap: int = 1000, 
                                    seed: Optional[int] = None) -> Dict[str, tuple]:
        """Calculate bootstrap confidence intervals.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with bootstrap confidence intervals
        """
        if seed is not None:
            np.random.seed(seed)
        
        original_result = self.get_results()
        bootstrap_effects = []
        bootstrap_tau2s = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(self.points), len(self.points), replace=True)
            bootstrap_points = [self.points[i] for i in indices]
            
            try:
                # Fit model on bootstrap sample
                bootstrap_model = self.__class__(bootstrap_points, alpha=self.alpha)
                bootstrap_result = bootstrap_model.fit()
                
                bootstrap_effects.append(bootstrap_result.pooled_effect)
                bootstrap_tau2s.append(bootstrap_result.tau2)
            except:
                # Skip failed bootstrap samples
                continue
        
        if not bootstrap_effects:
            raise ModelError("All bootstrap samples failed")
        
        # Calculate percentile confidence intervals
        alpha = self.alpha
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        effect_ci = (
            np.percentile(bootstrap_effects, lower_percentile),
            np.percentile(bootstrap_effects, upper_percentile)
        )
        
        tau2_ci = (
            np.percentile(bootstrap_tau2s, lower_percentile),
            np.percentile(bootstrap_tau2s, upper_percentile)
        )
        
        return {
            'pooled_effect': effect_ci,
            'tau2': tau2_ci,
            'n_bootstrap': len(bootstrap_effects),
            'bootstrap_effects': bootstrap_effects,
            'bootstrap_tau2s': bootstrap_tau2s
        }
    
    def summary(self) -> str:
        """Generate a summary string of the meta-analysis results.
        
        Returns:
            Formatted summary string
        """
        result = self.get_results()
        
        summary_lines = [
            f"Meta-Analysis Results ({result.method})",
            "=" * 50,
            f"Number of studies: {result.n_studies}",
            f"Pooled effect: {result.pooled_effect:.4f} (95% CI: {result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f})",
            f"Standard error: {result.pooled_se:.4f}",
            f"Z-score: {result.z_score:.4f}",
            f"P-value: {result.p_value:.4f}",
            "",
            "Heterogeneity:",
            f"  Tau²: {result.tau2:.4f}",
            f"  I²: {result.i_squared:.1f}%",
            f"  Q: {result.q_statistic:.4f} (df={result.n_studies-1}, p={result.q_p_value:.4f})"
        ]
        
        return "\n".join(summary_lines)