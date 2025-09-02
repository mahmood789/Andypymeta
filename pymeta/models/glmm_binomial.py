"""GLMM Binomial model with graceful fallback to log-OR + RandomEffects."""

import numpy as np
from typing import List, Optional, Union
import warnings

from .base import MetaModel
from .random_effects import RandomEffects
from ..typing import MetaPoint, MetaResults
from ..registries import register_model
from ..errors import ModelError, EstimationError
from ..effects.binary import calculate_log_odds_ratio, calculate_from_2x2_tables
from ..config import config


@register_model("glmm_binomial")
class GLMMBinomial(MetaModel):
    """GLMM Binomial meta-analysis model with graceful fallback.
    
    Uses statsmodels BinomialBayesMixedGLM when available, otherwise falls back
    to log odds ratio calculation with RandomEffects model.
    """
    
    def __init__(self, points: List[MetaPoint] = None,
                 tables_2x2: List[np.ndarray] = None,
                 alpha: float = None,
                 fallback_estimator: str = None):
        """Initialize GLMM Binomial model.
        
        Args:
            points: List of MetaPoint objects (for log-OR data)
            tables_2x2: List of 2x2 contingency tables
            alpha: Significance level
            fallback_estimator: Tau² estimator for fallback model
        """
        self.tables_2x2 = tables_2x2
        self.fallback_estimator = fallback_estimator or config.default_tau2_estimator
        self._use_statsmodels = config.check_dependency('statsmodels')
        self._fallback_model: Optional[RandomEffects] = None
        
        # Handle different input formats
        if points is not None:
            super().__init__(points, alpha)
        elif tables_2x2 is not None:
            # Convert 2x2 tables to MetaPoint objects
            points = self._convert_2x2_to_points(tables_2x2)
            super().__init__(points, alpha)
        else:
            raise ModelError("Must provide either points or tables_2x2")
    
    def _convert_2x2_to_points(self, tables: List[np.ndarray]) -> List[MetaPoint]:
        """Convert 2x2 tables to MetaPoint objects using log odds ratios.
        
        Args:
            tables: List of 2x2 numpy arrays
            
        Returns:
            List of MetaPoint objects
        """
        try:
            return calculate_from_2x2_tables(tables, effect_type="log_or")
        except Exception as e:
            raise ModelError(f"Failed to convert 2x2 tables to MetaPoints: {e}")
    
    def fit(self) -> MetaResults:
        """Fit the GLMM Binomial model with graceful fallback.
        
        Returns:
            MetaResults object with fitted parameters
        """
        if self._use_statsmodels and self.tables_2x2 is not None:
            try:
                results = self._fit_statsmodels_glmm()
                self._results = results
                return results
            except Exception as e:
                warnings.warn(f"GLMM fitting failed, falling back to log-OR + RE: {e}")
                return self._fit_fallback_model()
        else:
            # Use fallback model
            if not self._use_statsmodels:
                warnings.warn("statsmodels not available, using log-OR + RandomEffects fallback")
            
            return self._fit_fallback_model()
    
    def _fit_statsmodels_glmm(self) -> MetaResults:
        """Fit GLMM using statsmodels BinomialBayesMixedGLM.
        
        Returns:
            MetaResults object
        """
        try:
            import statsmodels.api as sm
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            import pandas as pd
        except ImportError:
            raise EstimationError("statsmodels not available for GLMM fitting")
        
        if self.tables_2x2 is None:
            raise ModelError("2x2 tables required for GLMM fitting")
        
        # Prepare data for statsmodels
        data_list = []
        for i, table in enumerate(self.tables_2x2):
            study_id = f"study_{i+1}"
            # Treatment group
            data_list.append({
                'study': study_id,
                'group': 'treatment',
                'events': int(table[0, 0]),
                'total': int(table[0, 0] + table[1, 0]),
                'group_numeric': 1
            })
            # Control group  
            data_list.append({
                'study': study_id,
                'group': 'control',
                'events': int(table[0, 1]),
                'total': int(table[0, 1] + table[1, 1]),
                'group_numeric': 0
            })
        
        df = pd.DataFrame(data_list)
        
        # Fit GLMM
        try:
            # Setup design matrices
            endog = df['events']
            trials = df['total']
            exog = sm.add_constant(df['group_numeric'])
            groups = df['study']
            
            # Fit Bayesian Mixed GLM
            model = BinomialBayesMixedGLM(
                endog=endog,
                exog=exog,
                trials=trials,
                groups=groups
            )
            
            result = model.fit()
            
            # Extract results
            pooled_effect = result.params[1]  # Log odds ratio coefficient
            pooled_variance = result.cov_params()[1, 1]
            tau2 = result.random_effects_var if hasattr(result, 'random_effects_var') else 0.0
            
            # Calculate statistics
            glmm_results = self._calculate_common_statistics(
                pooled_effect=pooled_effect,
                pooled_variance=pooled_variance,
                tau2=tau2,
                method="GLMM Binomial"
            )
            
            return glmm_results
            
        except Exception as e:
            raise EstimationError(f"GLMM fitting failed: {e}")
    
    def _fit_fallback_model(self) -> MetaResults:
        """Fit fallback RandomEffects model on log odds ratios.
        
        Returns:
            MetaResults object
        """
        # Create and fit fallback model
        self._fallback_model = RandomEffects(
            self.points, 
            tau2_estimator=self.fallback_estimator,
            alpha=self.alpha
        )
        
        result = self._fallback_model.fit()
        
        # Update method name to indicate fallback
        fallback_results = MetaResults(
            pooled_effect=result.pooled_effect,
            pooled_variance=result.pooled_variance,
            confidence_interval=result.confidence_interval,
            z_score=result.z_score,
            p_value=result.p_value,
            tau2=result.tau2,
            i_squared=result.i_squared,
            q_statistic=result.q_statistic,
            q_p_value=result.q_p_value,
            n_studies=result.n_studies,
            method=f"Log-OR + RE ({self.fallback_estimator}) [GLMM fallback]",
            heterogeneity_test=result.heterogeneity_test
        )
        
        return fallback_results
    
    def fit_from_2x2(self, tables: List[np.ndarray], 
                     continuity_correction: Optional[float] = None) -> MetaResults:
        """Fit model directly from 2x2 tables.
        
        Args:
            tables: List of 2x2 contingency tables
            continuity_correction: Continuity correction for zero cells
            
        Returns:
            MetaResults object
        """
        self.tables_2x2 = tables
        
        # Convert to MetaPoint objects
        self.points = calculate_from_2x2_tables(
            tables, 
            effect_type="log_or",
            continuity_correction=continuity_correction
        )
        
        # Revalidate points
        self.points = self._validate_points(self.points)
        
        return self.fit()
    
    def odds_ratio_results(self) -> dict:
        """Get results on odds ratio scale (exponentiated).
        
        Returns:
            Dictionary with odds ratio results
        """
        result = self.get_results()
        
        # Exponentiate log odds ratios
        or_estimate = np.exp(result.pooled_effect)
        or_ci_lower = np.exp(result.confidence_interval[0])
        or_ci_upper = np.exp(result.confidence_interval[1])
        
        return {
            'odds_ratio': or_estimate,
            'ci_lower': or_ci_lower,
            'ci_upper': or_ci_upper,
            'log_or': result.pooled_effect,
            'log_or_se': result.pooled_se,
            'p_value': result.p_value,
            'tau2': result.tau2,
            'i_squared': result.i_squared
        }
    
    def absolute_risk_measures(self) -> dict:
        """Calculate absolute risk measures if 2x2 data available.
        
        Returns:
            Dictionary with absolute risk measures
        """
        if self.tables_2x2 is None:
            raise ModelError("2x2 tables required for absolute risk calculation")
        
        # Calculate pooled risks
        total_treatment_events = sum(table[0, 0] for table in self.tables_2x2)
        total_treatment_n = sum(table[0, 0] + table[1, 0] for table in self.tables_2x2)
        total_control_events = sum(table[0, 1] for table in self.tables_2x2)
        total_control_n = sum(table[0, 1] + table[1, 1] for table in self.tables_2x2)
        
        # Pooled risks
        risk_treatment = total_treatment_events / total_treatment_n if total_treatment_n > 0 else 0
        risk_control = total_control_events / total_control_n if total_control_n > 0 else 0
        
        # Risk difference
        risk_difference = risk_treatment - risk_control
        
        # Number needed to treat/harm
        if abs(risk_difference) > 0:
            nnt = 1 / abs(risk_difference)
            nnt_type = "NNT" if risk_difference < 0 else "NNH"
        else:
            nnt = float('inf')
            nnt_type = "NNT"
        
        return {
            'risk_treatment': risk_treatment,
            'risk_control': risk_control,
            'risk_difference': risk_difference,
            'relative_risk': risk_treatment / risk_control if risk_control > 0 else float('inf'),
            'nnt': nnt,
            'nnt_type': nnt_type,
            'total_events': total_treatment_events + total_control_events,
            'total_participants': total_treatment_n + total_control_n
        }
    
    def diagnostic_plots_data(self) -> dict:
        """Prepare data for GLMM-specific diagnostic plots.
        
        Returns:
            Dictionary with diagnostic plot data
        """
        if self._fallback_model:
            # Use fallback model diagnostics if GLMM failed
            return self._fallback_model.forest_plot_data()
        
        # Standard forest plot data
        result = self.get_results()
        
        # Study-level odds ratios
        study_data = []
        for i, point in enumerate(self.points):
            se = np.sqrt(point.variance)
            from scipy import stats
            z_critical = stats.norm.ppf(1 - self.alpha / 2)
            
            study_data.append({
                'study': point.label or f"Study {i+1}",
                'log_or': point.effect,
                'odds_ratio': np.exp(point.effect),
                'ci_lower_log': point.effect - z_critical * se,
                'ci_upper_log': point.effect + z_critical * se,
                'ci_lower_or': np.exp(point.effect - z_critical * se),
                'ci_upper_or': np.exp(point.effect + z_critical * se),
                'weight': point.weight
            })
        
        # Overall result
        overall_data = {
            'log_or': result.pooled_effect,
            'odds_ratio': np.exp(result.pooled_effect),
            'ci_lower_log': result.confidence_interval[0],
            'ci_upper_log': result.confidence_interval[1],
            'ci_lower_or': np.exp(result.confidence_interval[0]),
            'ci_upper_or': np.exp(result.confidence_interval[1]),
            'method': result.method
        }
        
        return {
            'studies': study_data,
            'overall': overall_data,
            'scale': 'log_odds_ratio',
            'heterogeneity': {
                'tau2': result.tau2,
                'i_squared': result.i_squared,
                'q_statistic': result.q_statistic,
                'q_p_value': result.q_p_value
            }
        }
    
    def model_comparison(self) -> dict:
        """Compare GLMM results with standard random effects model.
        
        Returns:
            Dictionary comparing models
        """
        # Get GLMM results
        glmm_result = self.get_results()
        
        # Fit standard random effects model for comparison
        re_model = RandomEffects(self.points, alpha=self.alpha)
        re_result = re_model.fit()
        
        comparison = {
            'glmm': {
                'method': glmm_result.method,
                'pooled_effect': glmm_result.pooled_effect,
                'pooled_se': glmm_result.pooled_se,
                'tau2': glmm_result.tau2,
                'i_squared': glmm_result.i_squared,
                'p_value': glmm_result.p_value
            },
            'random_effects': {
                'method': re_result.method,
                'pooled_effect': re_result.pooled_effect,
                'pooled_se': re_result.pooled_se,
                'tau2': re_result.tau2,
                'i_squared': re_result.i_squared,
                'p_value': re_result.p_value
            },
            'differences': {
                'effect_diff': glmm_result.pooled_effect - re_result.pooled_effect,
                'se_diff': glmm_result.pooled_se - re_result.pooled_se,
                'tau2_diff': glmm_result.tau2 - re_result.tau2
            }
        }
        
        return comparison


@register_model("GLMM")  # Short alias
class GLMM(GLMMBinomial):
    """Short alias for GLMMBinomial model."""
    pass