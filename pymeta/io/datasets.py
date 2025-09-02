"""Input validation utilities and dataset handling."""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict, Any, Tuple
from ..typing import MetaPoint
from ..errors import ValidationError, DataError


def validate_meta_data(data: Union[List[MetaPoint], pd.DataFrame, np.ndarray]) -> List[MetaPoint]:
    """Validate and convert input data to MetaPoint objects.
    
    Args:
        data: Input data in various formats
        
    Returns:
        List of validated MetaPoint objects
        
    Raises:
        ValidationError: If data validation fails
        DataError: If data conversion fails
    """
    if isinstance(data, list) and all(isinstance(point, MetaPoint) for point in data):
        return _validate_meta_points(data)
    elif isinstance(data, pd.DataFrame):
        return create_meta_points_from_dataframe(data)
    elif isinstance(data, np.ndarray):
        return _create_meta_points_from_array(data)
    else:
        raise ValidationError(f"Unsupported data type: {type(data)}")


def _validate_meta_points(points: List[MetaPoint]) -> List[MetaPoint]:
    """Validate list of MetaPoint objects."""
    if not points:
        raise ValidationError("Empty data provided")
    
    valid_points = []
    for i, point in enumerate(points):
        if not isinstance(point, MetaPoint):
            raise ValidationError(f"Point {i} is not a MetaPoint object")
        
        if not np.isfinite(point.effect):
            raise ValidationError(f"Point {i} has invalid effect size: {point.effect}")
        
        if not np.isfinite(point.variance) or point.variance <= 0:
            raise ValidationError(f"Point {i} has invalid variance: {point.variance}")
        
        valid_points.append(point)
    
    return valid_points


def create_meta_points_from_dataframe(df: pd.DataFrame, 
                                    effect_col: str = 'effect',
                                    variance_col: str = 'variance',
                                    se_col: Optional[str] = None,
                                    weight_col: Optional[str] = None,
                                    label_col: Optional[str] = None,
                                    study_id_col: Optional[str] = None) -> List[MetaPoint]:
    """Create MetaPoint objects from pandas DataFrame.
    
    Args:
        df: Input DataFrame
        effect_col: Column name for effect sizes
        variance_col: Column name for variances
        se_col: Column name for standard errors (alternative to variance)
        weight_col: Column name for weights
        label_col: Column name for study labels
        study_id_col: Column name for study IDs
        
    Returns:
        List of MetaPoint objects
        
    Raises:
        DataError: If required columns are missing or data is invalid
    """
    if df.empty:
        raise DataError("Empty DataFrame provided")
    
    # Check for required columns
    if effect_col not in df.columns:
        raise DataError(f"Effect column '{effect_col}' not found")
    
    # Handle variance vs standard error
    if variance_col in df.columns:
        variances = df[variance_col].values
    elif se_col in df.columns:
        variances = df[se_col].values ** 2
    else:
        raise DataError("Neither variance nor standard error column found")
    
    effects = df[effect_col].values
    
    # Validate numeric data
    if not np.all(np.isfinite(effects)):
        raise DataError("Invalid effect sizes found (NaN or infinite)")
    
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
        raise DataError("Invalid variances found (NaN, infinite, or non-positive)")
    
    # Extract optional columns
    weights = df[weight_col].values if weight_col in df.columns else None
    labels = df[label_col].values if label_col in df.columns else None
    study_ids = df[study_id_col].values if study_id_col in df.columns else None
    
    # Create MetaPoint objects
    points = []
    for i in range(len(df)):
        point = MetaPoint(
            effect=effects[i],
            variance=variances[i],
            weight=weights[i] if weights is not None else None,
            label=str(labels[i]) if labels is not None else f"Study {i+1}",
            study_id=str(study_ids[i]) if study_ids is not None else None
        )
        points.append(point)
    
    return points


def _create_meta_points_from_array(arr: np.ndarray) -> List[MetaPoint]:
    """Create MetaPoint objects from numpy array.
    
    Expected format: [[effect1, variance1], [effect2, variance2], ...]
    """
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise DataError("Array must be 2D with at least 2 columns (effect, variance)")
    
    effects = arr[:, 0]
    variances = arr[:, 1]
    
    if not np.all(np.isfinite(effects)):
        raise DataError("Invalid effect sizes found in array")
    
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
        raise DataError("Invalid variances found in array")
    
    points = []
    for i in range(len(arr)):
        point = MetaPoint(
            effect=effects[i],
            variance=variances[i],
            label=f"Study {i+1}"
        )
        points.append(point)
    
    return points


def validate_2x2_data(data: Union[List[List[int]], np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Validate 2x2 contingency table data.
    
    Args:
        data: 2x2 table data in various formats
        
    Returns:
        Validated numpy array of shape (2, 2)
        
    Raises:
        ValidationError: If data validation fails
    """
    if isinstance(data, list):
        arr = np.array(data)
    elif isinstance(data, pd.DataFrame):
        arr = data.values
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValidationError(f"Unsupported data type for 2x2 table: {type(data)}")
    
    if arr.shape != (2, 2):
        raise ValidationError(f"Expected 2x2 table, got shape {arr.shape}")
    
    if not np.all(arr >= 0):
        raise ValidationError("All values in 2x2 table must be non-negative")
    
    if not np.all(arr == arr.astype(int)):
        raise ValidationError("All values in 2x2 table must be integers")
    
    return arr.astype(int)


def create_example_data(n_studies: int = 10, 
                       true_effect: float = 0.5,
                       tau2: float = 0.1,
                       seed: Optional[int] = None) -> List[MetaPoint]:
    """Create example meta-analysis data for testing.
    
    Args:
        n_studies: Number of studies to simulate
        true_effect: True underlying effect size
        tau2: Between-study variance
        seed: Random seed for reproducibility
        
    Returns:
        List of simulated MetaPoint objects
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Simulate study-specific effects
    study_effects = np.random.normal(true_effect, np.sqrt(tau2), n_studies)
    
    # Simulate within-study variances (inverse gamma distribution)
    study_variances = 1.0 / np.random.gamma(2, 0.1, n_studies)
    
    # Add sampling error
    observed_effects = np.random.normal(study_effects, np.sqrt(study_variances))
    
    points = []
    for i in range(n_studies):
        point = MetaPoint(
            effect=observed_effects[i],
            variance=study_variances[i],
            label=f"Study {i+1}",
            study_id=f"study_{i+1:02d}"
        )
        points.append(point)
    
    return points