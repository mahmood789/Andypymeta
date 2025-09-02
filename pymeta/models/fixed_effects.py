"""Fixed Effects meta-analysis model."""

import numpy as np
from typing import List

from .base import MetaModel
from ..typing import MetaPoint, MetaResults
from ..registries import register_model


@register_model("fixed_effects")
class FixedEffects(MetaModel):
    """Fixed Effects meta-analysis model.
    
    Assumes all studies estimate the same true effect size (no between-study heterogeneity).
    Uses inverse variance weighting to combine studies.
    """
    
    def fit(self) -> MetaResults:
        """Fit the fixed effects model.
        
        Returns:
            MetaResults object with fitted parameters
        """
        # Extract effects and variances
        effects = np.array([p.effect for p in self.points])
        variances = np.array([p.variance for p in self.points])
        
        # Inverse variance weights
        weights = 1.0 / variances
        
        # Update point weights for consistency
        for i, point in enumerate(self.points):
            point.weight = weights[i]
        
        # Pooled effect estimate
        sum_weights = np.sum(weights)
        pooled_effect = np.sum(weights * effects) / sum_weights
        
        # Pooled variance
        pooled_variance = 1.0 / sum_weights
        
        # Calculate statistics (tau2 = 0 for fixed effects)
        results = self._calculate_common_statistics(
            pooled_effect=pooled_effect,
            pooled_variance=pooled_variance,
            tau2=0.0,
            method="Fixed Effects"
        )
        
        self._results = results
        return results
    
    def cochrans_q_test(self) -> dict:
        """Perform Cochran's Q test for heterogeneity.
        
        Returns:
            Dictionary with Q test results
        """
        from scipy import stats
        
        effects = np.array([p.effect for p in self.points])
        weights = np.array([p.weight for p in self.points])
        
        # Weighted mean
        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        
        # Q statistic
        q_statistic = np.sum(weights * (effects - weighted_mean) ** 2)
        
        # Degrees of freedom
        df = len(self.points) - 1
        
        # P-value
        p_value = 1 - stats.chi2.cdf(q_statistic, df) if df > 0 else 1.0
        
        # Critical value
        alpha = self.alpha
        critical_value = stats.chi2.ppf(1 - alpha, df) if df > 0 else 0
        
        return {
            'q_statistic': q_statistic,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'critical_value': critical_value,
            'significant': p_value < alpha,
            'interpretation': self._interpret_q_test(p_value, alpha)
        }
    
    def _interpret_q_test(self, p_value: float, alpha: float) -> str:
        """Interpret Cochran's Q test results."""
        if p_value < alpha:
            return f"Significant heterogeneity detected (p={p_value:.4f} < {alpha}). Consider random effects model."
        else:
            return f"No significant heterogeneity detected (p={p_value:.4f} ≥ {alpha}). Fixed effects model appropriate."
    
    def weights_summary(self) -> dict:
        """Summary of study weights.
        
        Returns:
            Dictionary with weight statistics
        """
        weights = np.array([p.weight for p in self.points])
        
        # Calculate relative weights (as percentages)
        total_weight = np.sum(weights)
        relative_weights = 100 * weights / total_weight
        
        weight_info = []
        for i, point in enumerate(self.points):
            weight_info.append({
                'study': point.label or f"Study {i+1}",
                'effect': point.effect,
                'variance': point.variance,
                'weight': weights[i],
                'relative_weight': relative_weights[i]
            })
        
        return {
            'weight_details': weight_info,
            'total_weight': total_weight,
            'max_weight': np.max(weights),
            'min_weight': np.min(weights),
            'weight_ratio': np.max(weights) / np.min(weights),
            'effective_sample_size': total_weight ** 2 / np.sum(weights ** 2)
        }
    
    def prediction_interval(self) -> tuple:
        """Calculate prediction interval for fixed effects model.
        
        For fixed effects, prediction interval equals confidence interval
        since there's no between-study variance.
        
        Returns:
            Tuple of (lower, upper) bounds
        """
        result = self.get_results()
        return result.confidence_interval
    
    def forest_plot_data(self) -> dict:
        """Prepare data for forest plot visualization.
        
        Returns:
            Dictionary with plot data
        """
        from scipy import stats
        
        result = self.get_results()
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        
        # Study-level data
        study_data = []
        for i, point in enumerate(self.points):
            se = np.sqrt(point.variance)
            ci_lower = point.effect - z_critical * se
            ci_upper = point.effect + z_critical * se
            
            study_data.append({
                'study': point.label or f"Study {i+1}",
                'effect': point.effect,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'weight': point.weight,
                'relative_weight': 100 * point.weight / np.sum([p.weight for p in self.points])
            })
        
        # Overall result
        overall_data = {
            'effect': result.pooled_effect,
            'ci_lower': result.confidence_interval[0],
            'ci_upper': result.confidence_interval[1],
            'method': result.method
        }
        
        return {
            'studies': study_data,
            'overall': overall_data,
            'heterogeneity': {
                'q_statistic': result.q_statistic,
                'q_p_value': result.q_p_value,
                'i_squared': result.i_squared,
                'tau2': result.tau2
            }
        }


@register_model("FE")  # Short alias
class FE(FixedEffects):
    """Short alias for FixedEffects model."""
    pass