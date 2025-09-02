"""Cumulative Z-score calculations for Trial Sequential Analysis."""

import numpy as np
from typing import List, Optional, Dict, Any
from scipy import stats

from ..typing import MetaPoint, TSAResult
from ..models.fixed_effects import FixedEffects
from ..models.random_effects import RandomEffects
from .boundaries import obrien_fleming_boundary, calculate_information_size
from ..errors import TSAError
from ..config import config


def calculate_cumulative_z(points: List[MetaPoint],
                          model_type: str = "fixed_effects") -> List[float]:
    """Calculate cumulative Z-scores for sequential meta-analysis.
    
    Args:
        points: List of MetaPoint objects in chronological order
        model_type: Type of model to use ("fixed_effects" or "random_effects")
        
    Returns:
        List of cumulative Z-scores
        
    Raises:
        TSAError: If calculation fails
    """
    if len(points) < 2:
        raise TSAError("At least 2 studies required for cumulative analysis")
    
    try:
        cumulative_z = []
        
        for i in range(2, len(points) + 1):
            # Subset for cumulative analysis
            subset_points = points[:i]
            
            # Fit appropriate model
            if model_type == "fixed_effects":
                model = FixedEffects(subset_points)
            elif model_type == "random_effects":
                model = RandomEffects(subset_points)
            else:
                raise TSAError(f"Unknown model type: {model_type}")
            
            result = model.fit()
            
            # Calculate Z-score
            z_score = result.z_score
            cumulative_z.append(z_score)
        
        return cumulative_z
        
    except Exception as e:
        raise TSAError(f"Failed to calculate cumulative Z-scores: {e}")


def calculate_information_fractions(points: List[MetaPoint],
                                  required_information_size: float) -> List[float]:
    """Calculate information fractions for TSA.
    
    Args:
        points: List of MetaPoint objects
        required_information_size: Total required information size
        
    Returns:
        List of information fractions
    """
    if required_information_size <= 0:
        raise TSAError("Required information size must be positive")
    
    try:
        information_fractions = []
        cumulative_information = 0
        
        for i in range(len(points)):
            # Add information from current study
            cumulative_information += points[i].weight
            
            # Calculate fraction
            fraction = min(1.0, cumulative_information / required_information_size)
            information_fractions.append(fraction)
        
        return information_fractions
        
    except Exception as e:
        raise TSAError(f"Failed to calculate information fractions: {e}")


def perform_tsa(points: List[MetaPoint],
               delta: float,
               alpha: float = None,
               beta: float = None,
               model_type: str = "fixed_effects",
               boundary_type: str = "obrien_fleming",
               diversity_adjustment: Optional[float] = None) -> TSAResult:
    """Perform complete Trial Sequential Analysis.
    
    Args:
        points: List of MetaPoint objects in chronological order
        delta: Clinically relevant effect size to detect
        alpha: Type I error rate
        beta: Type II error rate
        model_type: Meta-analysis model type
        boundary_type: Type of monitoring boundary
        diversity_adjustment: Adjustment for heterogeneity
        
    Returns:
        TSAResult object
    """
    if alpha is None:
        alpha = config.tsa_alpha
    if beta is None:
        beta = config.tsa_beta
    
    try:
        # Calculate required information size
        required_info_size = calculate_information_size(delta, alpha=alpha, beta=beta)
        
        # Calculate cumulative Z-scores
        cumulative_z = calculate_cumulative_z(points, model_type)
        
        # Calculate information fractions (starting from 2nd study)
        subset_points = points[1:]  # Skip first study since we start cumulative from 2nd
        information_fractions = calculate_information_fractions(subset_points, required_info_size)
        
        # Calculate monitoring boundaries
        if boundary_type == "obrien_fleming":
            from .boundaries import obrien_fleming_boundary
            superiority_boundaries = obrien_fleming_boundary(information_fractions, alpha)
        elif boundary_type == "pocock":
            from .boundaries import pocock_boundary
            superiority_boundaries = pocock_boundary(information_fractions, alpha)
        else:
            raise TSAError(f"Unknown boundary type: {boundary_type}")
        
        # Calculate futility boundaries
        from .boundaries import futility_boundary
        futility_boundaries = futility_boundary(information_fractions, beta)
        
        # Adjust for diversity if specified
        if diversity_adjustment is not None and diversity_adjustment > 1.0:
            from .boundaries import adjust_boundaries_for_heterogeneity
            superiority_boundaries = adjust_boundaries_for_heterogeneity(
                superiority_boundaries, diversity_adjustment
            )
        
        # Check boundary crossings
        superiority_reached = _check_boundary_crossing(cumulative_z, superiority_boundaries)
        futility_reached = _check_boundary_crossing(cumulative_z, futility_boundaries, direction="below")
        
        # Prepare boundaries dictionary
        boundaries = {
            'superiority_upper': superiority_boundaries.tolist(),
            'superiority_lower': (-superiority_boundaries).tolist(),
            'futility': futility_boundaries.tolist(),
            'boundary_type': boundary_type
        }
        
        return TSAResult(
            cumulative_z=cumulative_z,
            boundaries=boundaries,
            information_fraction=information_fractions,
            required_information_size=required_info_size,
            futility_reached=futility_reached,
            superiority_reached=superiority_reached
        )
        
    except Exception as e:
        raise TSAError(f"TSA analysis failed: {e}")


def _check_boundary_crossing(z_scores: List[float],
                           boundaries: np.ndarray,
                           direction: str = "above") -> bool:
    """Check if any Z-score crosses monitoring boundaries.
    
    Args:
        z_scores: List of Z-scores to check
        boundaries: Boundary values
        direction: Direction to check ("above" or "below")
        
    Returns:
        True if any boundary is crossed
    """
    z_array = np.array(z_scores)
    
    if direction == "above":
        # Check if any |Z| > boundary
        return np.any(np.abs(z_array) > boundaries)
    elif direction == "below":
        # Check if any Z < boundary (for futility)
        return np.any(z_array < boundaries)
    else:
        raise TSAError(f"Unknown direction: {direction}")


def tsa_plot_data(tsa_result: TSAResult,
                 study_labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Prepare data for TSA plot visualization.
    
    Args:
        tsa_result: TSA analysis results
        study_labels: Optional study labels
        
    Returns:
        Dictionary with plot data
    """
    n_points = len(tsa_result.cumulative_z)
    
    if study_labels is None:
        study_labels = [f"Study {i+2}" for i in range(n_points)]  # Start from 2
    
    # Cumulative analysis points
    plot_data = {
        'information_fractions': tsa_result.information_fraction,
        'cumulative_z': tsa_result.cumulative_z,
        'study_labels': study_labels,
        'boundaries': tsa_result.boundaries,
        'required_information_size': tsa_result.required_information_size,
        'monitoring_reached': tsa_result.monitoring_boundary_reached,
        'superiority_reached': tsa_result.superiority_reached,
        'futility_reached': tsa_result.futility_reached
    }
    
    # Add boundary crossing information
    if tsa_result.superiority_reached:
        crossing_indices = _find_boundary_crossings(
            tsa_result.cumulative_z,
            tsa_result.boundaries['superiority_upper']
        )
        if crossing_indices:
            plot_data['first_superiority_crossing'] = {
                'index': crossing_indices[0],
                'study': study_labels[crossing_indices[0]],
                'z_score': tsa_result.cumulative_z[crossing_indices[0]],
                'information_fraction': tsa_result.information_fraction[crossing_indices[0]]
            }
    
    if tsa_result.futility_reached:
        crossing_indices = _find_boundary_crossings(
            tsa_result.cumulative_z,
            tsa_result.boundaries['futility'],
            direction="below"
        )
        if crossing_indices:
            plot_data['first_futility_crossing'] = {
                'index': crossing_indices[0],
                'study': study_labels[crossing_indices[0]],
                'z_score': tsa_result.cumulative_z[crossing_indices[0]],
                'information_fraction': tsa_result.information_fraction[crossing_indices[0]]
            }
    
    return plot_data


def _find_boundary_crossings(z_scores: List[float],
                           boundaries: List[float],
                           direction: str = "above") -> List[int]:
    """Find indices where boundaries are crossed.
    
    Args:
        z_scores: Z-score values
        boundaries: Boundary values
        direction: Direction to check
        
    Returns:
        List of indices where crossings occur
    """
    crossings = []
    z_array = np.array(z_scores)
    boundary_array = np.array(boundaries)
    
    if direction == "above":
        crossing_mask = np.abs(z_array) > boundary_array
    else:  # below
        crossing_mask = z_array < boundary_array
    
    crossings = np.where(crossing_mask)[0].tolist()
    
    return crossings


def conditional_power_analysis(points: List[MetaPoint],
                              delta: float,
                              required_info_size: float,
                              model_type: str = "fixed_effects") -> Dict[str, float]:
    """Calculate conditional power for continuing the meta-analysis.
    
    Args:
        points: Current set of studies
        delta: Target effect size
        required_info_size: Required information size
        model_type: Meta-analysis model type
        
    Returns:
        Dictionary with conditional power statistics
    """
    try:
        # Current meta-analysis
        if model_type == "fixed_effects":
            model = FixedEffects(points)
        else:
            model = RandomEffects(points)
        
        current_result = model.fit()
        
        # Current information
        current_info = sum(p.weight for p in points)
        remaining_info = max(0, required_info_size - current_info)
        
        # Conditional power calculation
        if remaining_info > 0:
            # Expected Z-score at completion
            current_z = current_result.z_score
            expected_final_z = current_z * np.sqrt(required_info_size / current_info)
            
            # Conditional power (probability of detecting effect)
            conditional_power = 1 - stats.norm.cdf(1.96 - expected_final_z)
        else:
            conditional_power = 1.0 if abs(current_result.z_score) > 1.96 else 0.0
        
        return {
            'conditional_power': conditional_power,
            'current_z': current_result.z_score,
            'current_information': current_info,
            'required_information': required_info_size,
            'information_fraction': current_info / required_info_size,
            'remaining_information': remaining_info
        }
        
    except Exception as e:
        raise TSAError(f"Conditional power analysis failed: {e}")