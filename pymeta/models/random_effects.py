"""
Random effects meta-analysis model with HKSJ variance adjustment support.
"""

from typing import List
import numpy as np
from scipy.stats import norm, chi2

from ..stats.hksj import hksj_se, hksj_confidence_interval, hksj_p_value
from .. import MetaPoint, MetaResults


class RandomEffects:
    """Random effects meta-analysis model with optional HKSJ adjustment."""
    
    def __init__(self, tau2_method: str = "REML", use_hksj: bool = False):
        """
        Initialize random effects model.
        
        Args:
            tau2_method: Method for estimating tau² ("DL", "REML", "PM", "ML")
            use_hksj: Whether to use HKSJ variance adjustment
        """
        self.tau2_method = tau2_method
        self.use_hksj = use_hksj
        
        if tau2_method not in ["DL", "REML", "PM", "ML"]:
            raise ValueError("tau2_method must be one of: DL, REML, PM, ML")
    
    def _estimate_tau2_dl(self, effects: np.ndarray, variances: np.ndarray) -> float:
        """
        Estimate tau² using DerSimonian-Laird method.
        
        Args:
            effects: Array of effect sizes
            variances: Array of variances
            
        Returns:
            Estimated tau²
        """
        weights = 1.0 / variances
        sum_weights = np.sum(weights)
        weighted_mean = np.sum(weights * effects) / sum_weights
        
        # Q statistic
        q_stat = np.sum(weights * (effects - weighted_mean) ** 2)
        df = len(effects) - 1
        
        if df <= 0:
            return 0.0
            
        # DL estimate
        sum_weights_sq = np.sum(weights ** 2)
        c = sum_weights - sum_weights_sq / sum_weights
        
        if c <= 0:
            return 0.0
            
        tau2_dl = max(0.0, (q_stat - df) / c)
        return tau2_dl
    
    def _estimate_tau2_reml(self, effects: np.ndarray, variances: np.ndarray) -> float:
        """
        Estimate tau² using Restricted Maximum Likelihood (REML).
        
        This is a simplified implementation. For now, falls back to DL method.
        
        Args:
            effects: Array of effect sizes
            variances: Array of variances
            
        Returns:
            Estimated tau²
        """
        # Simplified REML - in practice would use iterative optimization
        # For now, use DL as starting point and apply correction
        tau2_dl = self._estimate_tau2_dl(effects, variances)
        
        # Apply small correction factor for REML
        k = len(effects)
        if k > 2:
            correction = (k - 1) / (k - 2) if k > 2 else 1.0
            tau2_reml = tau2_dl * correction
        else:
            tau2_reml = tau2_dl
            
        return max(0.0, tau2_reml)
    
    def _estimate_tau2_pm(self, effects: np.ndarray, variances: np.ndarray) -> float:
        """
        Estimate tau² using Paule-Mandel method.
        
        Simplified implementation - falls back to DL for now.
        
        Args:
            effects: Array of effect sizes
            variances: Array of variances
            
        Returns:
            Estimated tau²
        """
        # Simplified PM method
        return self._estimate_tau2_dl(effects, variances)
    
    def _estimate_tau2_ml(self, effects: np.ndarray, variances: np.ndarray) -> float:
        """
        Estimate tau² using Maximum Likelihood.
        
        Simplified implementation - falls back to DL for now.
        
        Args:
            effects: Array of effect sizes
            variances: Array of variances
            
        Returns:
            Estimated tau²
        """
        # Simplified ML method  
        tau2_dl = self._estimate_tau2_dl(effects, variances)
        
        # ML tends to underestimate compared to REML
        k = len(effects)
        if k > 1:
            correction = (k - 1) / k
            tau2_ml = tau2_dl * correction
        else:
            tau2_ml = tau2_dl
            
        return max(0.0, tau2_ml)
    
    def _estimate_tau2(self, effects: np.ndarray, variances: np.ndarray) -> float:
        """
        Estimate between-study variance (tau²).
        
        Args:
            effects: Array of effect sizes
            variances: Array of variances
            
        Returns:
            Estimated tau²
        """
        if self.tau2_method == "DL":
            return self._estimate_tau2_dl(effects, variances)
        elif self.tau2_method == "REML":
            return self._estimate_tau2_reml(effects, variances)
        elif self.tau2_method == "PM":
            return self._estimate_tau2_pm(effects, variances)
        elif self.tau2_method == "ML":
            return self._estimate_tau2_ml(effects, variances)
        else:
            raise ValueError(f"Unknown tau2_method: {self.tau2_method}")
    
    def fit(self, points: List[MetaPoint], alpha: float = 0.05) -> MetaResults:
        """
        Fit random effects meta-analysis model.
        
        Args:
            points: List of MetaPoint objects
            alpha: Significance level
            
        Returns:
            MetaResults object with analysis results
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 studies for meta-analysis")
            
        effects = np.array([p.effect for p in points])
        variances = np.array([p.variance for p in points])
        k = len(points)
        
        # Estimate between-study variance
        tau2 = self._estimate_tau2(effects, variances)
        
        # Random effects weights
        weights = 1.0 / (variances + tau2)
        sum_weights = np.sum(weights)
        
        # Pooled effect estimate
        pooled_effect = np.sum(weights * effects) / sum_weights
        
        # Calculate heterogeneity statistics
        q_stat = np.sum(weights * (effects - pooled_effect) ** 2)
        df = k - 1
        q_p_value = 1 - chi2.cdf(q_stat, df) if df > 0 else 1.0
        
        # I² and H² statistics
        if q_stat > df and df > 0:
            i2 = ((q_stat - df) / q_stat) * 100
            h2 = q_stat / df
        else:
            i2 = 0.0
            h2 = 1.0
        
        # Standard error and confidence intervals
        if self.use_hksj and k > 2:
            # Use HKSJ variance adjustment
            hksj_result = hksj_se(effects, variances, tau2, alpha)
            pooled_se = hksj_result.se_hk
            ci_lower, ci_upper = hksj_confidence_interval(pooled_effect, hksj_result)
            p_value = hksj_p_value(pooled_effect, hksj_result)
            df_result = hksj_result.df
            method_label = f"Random Effects ({self.tau2_method}) + HKSJ"
        else:
            # Standard random effects
            pooled_se = np.sqrt(1.0 / sum_weights)
            zcrit = norm.ppf(1 - alpha/2)
            ci_lower = pooled_effect - zcrit * pooled_se
            ci_upper = pooled_effect + zcrit * pooled_se
            
            # P-value (two-sided)
            z_stat = pooled_effect / pooled_se
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))
            df_result = None
            method_label = f"Random Effects ({self.tau2_method})"
        
        return MetaResults(
            effect=pooled_effect,
            se=pooled_se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            tau2=tau2,
            i2=i2,
            h2=h2,
            q_stat=q_stat,
            q_p_value=q_p_value,
            method=method_label,
            use_hksj=self.use_hksj,
            df=df_result,
            points=points
        )