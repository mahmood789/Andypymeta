"""Binary effect size calculations with continuity corrections."""

import numpy as np
from typing import List, Tuple, Union, Optional
from ..typing import MetaPoint
from ..errors import DataError
from ..config import config


def calculate_log_odds_ratio(a: int, b: int, c: int, d: int, 
                           continuity_correction: Optional[float] = None) -> Tuple[float, float]:
    """Calculate log odds ratio and its variance from 2x2 table.
    
    2x2 table format:
        Treatment  Control
    Event    a        b
    No Event c        d
    
    Args:
        a, b, c, d: Cell counts from 2x2 contingency table
        continuity_correction: Continuity correction to add to zero cells
        
    Returns:
        Tuple of (log_odds_ratio, variance)
        
    Raises:
        DataError: If invalid cell counts provided
    """
    if continuity_correction is None:
        continuity_correction = config.continuity_correction
    
    # Validate inputs
    if any(x < 0 for x in [a, b, c, d]):
        raise DataError("All cell counts must be non-negative")
    
    # Apply continuity correction for zero cells
    if any(x == 0 for x in [a, b, c, d]):
        a += continuity_correction
        b += continuity_correction
        c += continuity_correction
        d += continuity_correction
    
    # Calculate log odds ratio
    log_or = np.log((a * d) / (b * c))
    
    # Calculate variance
    variance = (1/a) + (1/b) + (1/c) + (1/d)
    
    return log_or, variance


def calculate_risk_ratio(a: int, b: int, c: int, d: int,
                        continuity_correction: Optional[float] = None) -> Tuple[float, float]:
    """Calculate log risk ratio and its variance from 2x2 table.
    
    Args:
        a, b, c, d: Cell counts from 2x2 contingency table
        continuity_correction: Continuity correction to add to zero cells
        
    Returns:
        Tuple of (log_risk_ratio, variance)
    """
    if continuity_correction is None:
        continuity_correction = config.continuity_correction
    
    # Validate inputs
    if any(x < 0 for x in [a, b, c, d]):
        raise DataError("All cell counts must be non-negative")
    
    # Apply continuity correction for zero cells
    if any(x == 0 for x in [a, b, c, d]):
        a += continuity_correction
        b += continuity_correction
        c += continuity_correction
        d += continuity_correction
    
    # Calculate marginal totals
    n1 = a + c  # Treatment group total
    n2 = b + d  # Control group total
    
    # Calculate risk ratio
    risk_treatment = a / n1
    risk_control = b / n2
    
    if risk_control == 0:
        raise DataError("Cannot calculate risk ratio when control risk is zero")
    
    log_rr = np.log(risk_treatment / risk_control)
    
    # Calculate variance using delta method
    variance = (c / (a * n1)) + (d / (b * n2))
    
    return log_rr, variance


def calculate_risk_difference(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
    """Calculate risk difference and its variance from 2x2 table.
    
    Args:
        a, b, c, d: Cell counts from 2x2 contingency table
        
    Returns:
        Tuple of (risk_difference, variance)
    """
    # Validate inputs
    if any(x < 0 for x in [a, b, c, d]):
        raise DataError("All cell counts must be non-negative")
    
    # Calculate marginal totals
    n1 = a + c  # Treatment group total
    n2 = b + d  # Control group total
    
    if n1 == 0 or n2 == 0:
        raise DataError("Cannot calculate risk difference with zero group totals")
    
    # Calculate risks
    risk_treatment = a / n1
    risk_control = b / n2
    
    # Risk difference
    rd = risk_treatment - risk_control
    
    # Variance
    variance = (risk_treatment * (1 - risk_treatment) / n1) + \
               (risk_control * (1 - risk_control) / n2)
    
    return rd, variance


def calculate_from_2x2_tables(tables: List[np.ndarray], 
                             effect_type: str = "log_or",
                             continuity_correction: Optional[float] = None) -> List[MetaPoint]:
    """Calculate effect sizes from multiple 2x2 tables.
    
    Args:
        tables: List of 2x2 numpy arrays
        effect_type: Type of effect size ("log_or", "log_rr", "rd")
        continuity_correction: Continuity correction for zero cells
        
    Returns:
        List of MetaPoint objects
        
    Raises:
        DataError: If invalid effect type or table format
    """
    if effect_type not in ["log_or", "log_rr", "rd"]:
        raise DataError(f"Unknown effect type: {effect_type}")
    
    points = []
    
    for i, table in enumerate(tables):
        if table.shape != (2, 2):
            raise DataError(f"Table {i} is not 2x2: {table.shape}")
        
        a, b = table[0, 0], table[0, 1]
        c, d = table[1, 0], table[1, 1]
        
        try:
            if effect_type == "log_or":
                effect, variance = calculate_log_odds_ratio(a, b, c, d, continuity_correction)
                label_suffix = "Log OR"
            elif effect_type == "log_rr":
                effect, variance = calculate_risk_ratio(a, b, c, d, continuity_correction)
                label_suffix = "Log RR"
            else:  # rd
                effect, variance = calculate_risk_difference(a, b, c, d)
                label_suffix = "RD"
            
            point = MetaPoint(
                effect=effect,
                variance=variance,
                label=f"Study {i+1} ({label_suffix})",
                study_id=f"study_{i+1:02d}"
            )
            points.append(point)
            
        except DataError as e:
            raise DataError(f"Error calculating effect for table {i}: {e}")
    
    return points


def odds_ratio_to_cohens_d(log_or: float, log_or_variance: float) -> Tuple[float, float]:
    """Convert log odds ratio to Cohen's d using Hasselblad & Hedges (1995) approximation.
    
    Args:
        log_or: Log odds ratio
        log_or_variance: Variance of log odds ratio
        
    Returns:
        Tuple of (cohens_d, variance)
    """
    # Hasselblad & Hedges (1995) conversion
    # d ≈ log(OR) * sqrt(3) / π
    conversion_factor = np.sqrt(3) / np.pi
    
    cohens_d = log_or * conversion_factor
    d_variance = log_or_variance * (conversion_factor ** 2)
    
    return cohens_d, d_variance


def cohens_d_to_odds_ratio(d: float, d_variance: float) -> Tuple[float, float]:
    """Convert Cohen's d to log odds ratio.
    
    Args:
        d: Cohen's d effect size
        d_variance: Variance of Cohen's d
        
    Returns:
        Tuple of (log_odds_ratio, variance)
    """
    # Inverse of Hasselblad & Hedges conversion
    conversion_factor = np.pi / np.sqrt(3)
    
    log_or = d * conversion_factor
    log_or_variance = d_variance * (conversion_factor ** 2)
    
    return log_or, log_or_variance


def effect_size_conversion(points: List[MetaPoint], 
                          from_type: str, 
                          to_type: str) -> List[MetaPoint]:
    """Convert between different effect size types.
    
    Args:
        points: List of MetaPoint objects
        from_type: Source effect size type ("log_or", "cohens_d")
        to_type: Target effect size type ("log_or", "cohens_d")
        
    Returns:
        List of converted MetaPoint objects
        
    Raises:
        DataError: If conversion not supported
    """
    if from_type == to_type:
        return points.copy()
    
    supported_conversions = {
        ("log_or", "cohens_d"): odds_ratio_to_cohens_d,
        ("cohens_d", "log_or"): cohens_d_to_odds_ratio
    }
    
    conversion_key = (from_type, to_type)
    if conversion_key not in supported_conversions:
        raise DataError(f"Conversion from {from_type} to {to_type} not supported")
    
    conversion_func = supported_conversions[conversion_key]
    converted_points = []
    
    for i, point in enumerate(points):
        try:
            new_effect, new_variance = conversion_func(point.effect, point.variance)
            
            new_point = MetaPoint(
                effect=new_effect,
                variance=new_variance,
                weight=None,  # Recalculate weight
                label=point.label,
                study_id=point.study_id
            )
            converted_points.append(new_point)
            
        except Exception as e:
            raise DataError(f"Error converting point {i}: {e}")
    
    return converted_points