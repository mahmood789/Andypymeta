"""Random Effects meta-analysis model with pluggable estimators."""

import numpy as np
from typing import List, Optional

from .base import MetaModel
from ..typing import MetaPoint, MetaResults
from ..registries import register_model, get_estimator
from ..errors import EstimationError
from ..config import config


@register_model("random_effects")
class RandomEffects(MetaModel):
    """Random Effects meta-analysis model.
    
    Assumes studies estimate different but related effect sizes with 
    both within-study and between-study variance components.
    """
    
    def __init__(self, points: List[MetaPoint], 
                 tau2_estimator: str = None,
                 alpha: float = None):
        """Initialize random effects model.
        
        Args:
            points: List of MetaPoint objects
            tau2_estimator: Name of tau² estimator to use
            alpha: Significance level
        """
        super().__init__(points, alpha)
        self.tau2_estimator = tau2_estimator or config.default_tau2_estimator
        self._tau2: Optional[float] = None
    
    def fit(self) -> MetaResults:
        """Fit the random effects model.
        
        Returns:
            MetaResults object with fitted parameters
        """
        # Estimate tau²
        self._tau2 = self._estimate_tau2()
        
        # Extract effects and variances
        effects = np.array([p.effect for p in self.points])
        variances = np.array([p.variance for p in self.points])
        
        # Random effects weights (inverse of total variance)
        total_variances = variances + self._tau2
        weights = 1.0 / total_variances
        
        # Update point weights
        for i, point in enumerate(self.points):
            point.weight = weights[i]
        
        # Pooled effect estimate
        sum_weights = np.sum(weights)
        pooled_effect = np.sum(weights * effects) / sum_weights
        
        # Pooled variance
        pooled_variance = 1.0 / sum_weights
        
        # Calculate statistics
        method_name = f"Random Effects ({self.tau2_estimator})"
        results = self._calculate_common_statistics(
            pooled_effect=pooled_effect,
            pooled_variance=pooled_variance,
            tau2=self._tau2,
            method=method_name
        )
        
        self._results = results
        return results
    
    def _estimate_tau2(self) -> float:
        """Estimate between-study variance using specified estimator.
        
        Returns:
            Estimated tau² value
        """
        try:
            estimator_func = get_estimator(self.tau2_estimator)
            tau2 = estimator_func(self.points)
            return max(0, tau2)  # Ensure non-negative
        except Exception as e:
            raise EstimationError(f"Failed to estimate tau² using {self.tau2_estimator}: {e}")
    
    def get_tau2(self) -> float:
        """Get tau² estimate, calculating if necessary.
        
        Returns:
            Between-study variance estimate
        """
        if self._tau2 is None:
            self._tau2 = self._estimate_tau2()
        return self._tau2
    
    def prediction_interval(self, confidence_level: float = None) -> tuple:
        """Calculate prediction interval for new study.
        
        Args:
            confidence_level: Confidence level (default from instance)
            
        Returns:
            Tuple of (lower, upper) bounds for prediction interval
        """
        from scipy import stats
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        result = self.get_results()
        
        # Prediction standard error includes tau²
        pred_variance = result.pooled_variance + self._tau2
        pred_se = np.sqrt(pred_variance)
        
        # Critical value
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha / 2)
        
        # Prediction interval
        pred_lower = result.pooled_effect - z_critical * pred_se
        pred_upper = result.pooled_effect + z_critical * pred_se
        
        return (pred_lower, pred_upper)
    
    def weights_comparison(self) -> dict:
        """Compare random effects weights to fixed effects weights.
        
        Returns:
            Dictionary comparing weight schemes
        """
        # Current random effects weights
        re_weights = np.array([p.weight for p in self.points])
        
        # Fixed effects weights (inverse variance)
        variances = np.array([p.variance for p in self.points])
        fe_weights = 1.0 / variances
        
        # Calculate relative weights
        re_total = np.sum(re_weights)
        fe_total = np.sum(fe_weights)
        
        re_relative = 100 * re_weights / re_total
        fe_relative = 100 * fe_weights / fe_total
        
        comparison = []
        for i, point in enumerate(self.points):
            comparison.append({
                'study': point.label or f"Study {i+1}",
                'variance': variances[i],
                'fe_weight': fe_weights[i],
                'fe_relative': fe_relative[i],
                're_weight': re_weights[i],
                're_relative': re_relative[i],
                'weight_ratio': re_weights[i] / fe_weights[i],
                'relative_change': re_relative[i] - fe_relative[i]
            })
        
        return {
            'tau2': self._tau2,
            'comparison': comparison,
            'weight_equalization': np.std(re_relative) / np.std(fe_relative),
            'summary': {
                'fe_max_weight': np.max(fe_relative),
                're_max_weight': np.max(re_relative),
                'weight_range_reduction': (np.ptp(fe_relative) - np.ptp(re_relative)) / np.ptp(fe_relative)
            }
        }
    
    def sensitivity_analysis(self, tau2_estimators: List[str] = None) -> dict:
        """Perform sensitivity analysis across different tau² estimators.
        
        Args:
            tau2_estimators: List of estimator names to compare
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        if tau2_estimators is None:
            from ..registries import list_estimators
            tau2_estimators = list_estimators()
        
        original_estimator = self.tau2_estimator
        sensitivity_results = []
        
        for estimator in tau2_estimators:
            try:
                # Temporarily change estimator
                self.tau2_estimator = estimator
                self._tau2 = None  # Reset cached value
                self._results = None  # Reset cached results
                
                # Fit model with new estimator
                result = self.fit()
                
                sensitivity_results.append({
                    'estimator': estimator,
                    'tau2': result.tau2,
                    'pooled_effect': result.pooled_effect,
                    'pooled_se': result.pooled_se,
                    'ci_lower': result.confidence_interval[0],
                    'ci_upper': result.confidence_interval[1],
                    'p_value': result.p_value,
                    'i_squared': result.i_squared
                })
                
            except Exception as e:
                sensitivity_results.append({
                    'estimator': estimator,
                    'error': str(e)
                })
        
        # Restore original estimator
        self.tau2_estimator = original_estimator
        self._tau2 = None
        self._results = None
        
        # Calculate ranges
        valid_results = [r for r in sensitivity_results if 'error' not in r]
        if valid_results:
            tau2_range = (
                min(r['tau2'] for r in valid_results),
                max(r['tau2'] for r in valid_results)
            )
            effect_range = (
                min(r['pooled_effect'] for r in valid_results),
                max(r['pooled_effect'] for r in valid_results)
            )
        else:
            tau2_range = effect_range = (None, None)
        
        return {
            'results': sensitivity_results,
            'tau2_range': tau2_range,
            'effect_range': effect_range,
            'n_estimators': len(tau2_estimators),
            'n_successful': len(valid_results)
        }
    
    def knha_adjustment(self) -> MetaResults:
        """Apply Knapp-Hartung adjustment for small samples.
        
        Uses t-distribution instead of normal distribution for confidence intervals
        and p-values when number of studies is small.
        
        Returns:
            Adjusted MetaResults object
        """
        from scipy import stats
        
        result = self.get_results()
        k = len(self.points)
        df = k - 1
        
        if df <= 0:
            return result
        
        # Use t-distribution instead of normal
        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        
        # Adjusted confidence interval
        pooled_se = result.pooled_se
        ci_lower = result.pooled_effect - t_critical * pooled_se
        ci_upper = result.pooled_effect + t_critical * pooled_se
        
        # Adjusted p-value
        t_statistic = result.pooled_effect / pooled_se if pooled_se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
        
        # Create adjusted results
        adjusted_results = MetaResults(
            pooled_effect=result.pooled_effect,
            pooled_variance=result.pooled_variance,
            confidence_interval=(ci_lower, ci_upper),
            z_score=t_statistic,  # Actually t-statistic now
            p_value=p_value,
            tau2=result.tau2,
            i_squared=result.i_squared,
            q_statistic=result.q_statistic,
            q_p_value=result.q_p_value,
            n_studies=result.n_studies,
            method=f"{result.method} (Knapp-Hartung)",
            heterogeneity_test=result.heterogeneity_test
        )
        
        return adjusted_results


@register_model("RE")  # Short alias
class RE(RandomEffects):
    """Short alias for RandomEffects model."""
    pass


@register_model("DL")  # DerSimonian-Laird specific
class DerSimonianLaird(RandomEffects):
    """Random Effects model using DerSimonian-Laird estimator."""
    
    def __init__(self, points: List[MetaPoint], alpha: float = None):
        super().__init__(points, tau2_estimator="DL", alpha=alpha)


@register_model("REML")  # REML specific
class REMLModel(RandomEffects):
    """Random Effects model using REML estimator."""
    
    def __init__(self, points: List[MetaPoint], alpha: float = None):
        super().__init__(points, tau2_estimator="REML", alpha=alpha)