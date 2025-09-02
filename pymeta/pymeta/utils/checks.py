"""
Data validation and checking utilities for PyMeta.

This module provides functions for validating data and parameters used in meta-analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Any
from ..errors import (
    DataValidationError, 
    ParameterValidationError,
    InsufficientDataError
)
from ..types import Array, DataFrame


def check_data_frame(data: Any, name: str = "data") -> DataFrame:
    """
    Check that input is a valid DataFrame.
    
    Parameters
    ----------
    data : any
        Data to check
    name : str
        Name of the parameter for error messages
    
    Returns
    -------
    pandas.DataFrame
        Validated DataFrame
    
    Raises
    ------
    DataValidationError
        If data is not a valid DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError(f"{name} must be a pandas DataFrame, got {type(data)}")
    
    if data.empty:
        raise DataValidationError(f"{name} cannot be empty")
    
    return data


def check_required_columns(data: DataFrame, required_columns: List[str], 
                          name: str = "data") -> None:
    """
    Check that DataFrame contains required columns.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to check
    required_columns : list
        List of required column names
    name : str
        Name of the parameter for error messages
    
    Raises
    ------
    DataValidationError
        If required columns are missing
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise DataValidationError(
            f"{name} missing required columns: {missing_columns}"
        )


def check_numeric_column(data: DataFrame, column: str, 
                        allow_missing: bool = False,
                        positive_only: bool = False,
                        finite_only: bool = True) -> None:
    """
    Check that column contains valid numeric data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the column
    column : str
        Name of column to check
    allow_missing : bool, default False
        Whether to allow missing values
    positive_only : bool, default False
        Whether values must be positive
    finite_only : bool, default True
        Whether values must be finite
    
    Raises
    ------
    DataValidationError
        If column fails validation
    """
    if column not in data.columns:
        raise DataValidationError(f"Column '{column}' not found in data")
    
    col_data = data[column]
    
    # Check for missing values
    if not allow_missing and col_data.isnull().any():
        n_missing = col_data.isnull().sum()
        raise DataValidationError(
            f"Column '{column}' contains {n_missing} missing values"
        )
    
    # Check for non-numeric values (excluding NaN if allowed)
    if not allow_missing:
        non_numeric = ~pd.to_numeric(col_data, errors='coerce').notna()
        if non_numeric.any():
            raise DataValidationError(f"Column '{column}' contains non-numeric values")
    
    # Convert to numeric for further checks
    numeric_data = pd.to_numeric(col_data, errors='coerce')
    valid_data = numeric_data.dropna() if allow_missing else numeric_data
    
    # Check for finite values
    if finite_only and not np.isfinite(valid_data).all():
        n_infinite = (~np.isfinite(valid_data)).sum()
        raise DataValidationError(
            f"Column '{column}' contains {n_infinite} infinite values"
        )
    
    # Check for positive values
    if positive_only and (valid_data <= 0).any():
        n_non_positive = (valid_data <= 0).sum()
        raise DataValidationError(
            f"Column '{column}' contains {n_non_positive} non-positive values"
        )


def check_effect_size_data(data: DataFrame, 
                          effect_column: str = 'effect_size',
                          variance_column: Optional[str] = None,
                          se_column: Optional[str] = None) -> None:
    """
    Check effect size data for meta-analysis.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data to check
    effect_column : str, default 'effect_size'
        Name of effect size column
    variance_column : str, optional
        Name of variance column
    se_column : str, optional
        Name of standard error column
    
    Raises
    ------
    DataValidationError
        If data fails validation
    """
    # Check basic requirements
    check_data_frame(data)
    check_required_columns(data, [effect_column])
    
    # Check effect sizes
    check_numeric_column(data, effect_column, finite_only=True)
    
    # Check variance or standard error
    if variance_column and variance_column in data.columns:
        check_numeric_column(data, variance_column, positive_only=True, finite_only=True)
    elif se_column and se_column in data.columns:
        check_numeric_column(data, se_column, positive_only=True, finite_only=True)
    else:
        raise DataValidationError(
            "Data must contain either variance or standard error column"
        )
    
    # Check minimum number of studies
    if len(data) < 2:
        raise InsufficientDataError(
            f"Meta-analysis requires at least 2 studies, got {len(data)}"
        )


def check_binary_outcome_data(data: DataFrame,
                             treatment_events_col: str = 'treatment_events',
                             treatment_total_col: str = 'treatment_total',
                             control_events_col: str = 'control_events',
                             control_total_col: str = 'control_total') -> None:
    """
    Check binary outcome data for meta-analysis.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data to check
    treatment_events_col : str
        Name of treatment events column
    treatment_total_col : str
        Name of treatment total column
    control_events_col : str
        Name of control events column
    control_total_col : str
        Name of control total column
    
    Raises
    ------
    DataValidationError
        If data fails validation
    """
    required_cols = [treatment_events_col, treatment_total_col, 
                    control_events_col, control_total_col]
    
    check_data_frame(data)
    check_required_columns(data, required_cols)
    
    # Check all columns are non-negative integers
    for col in required_cols:
        check_numeric_column(data, col, positive_only=False, finite_only=True)
        
        # Check for non-integer values
        col_data = data[col]
        if not (col_data == col_data.astype(int)).all():
            raise DataValidationError(f"Column '{col}' must contain integer values")
        
        # Check for negative values
        if (col_data < 0).any():
            raise DataValidationError(f"Column '{col}' cannot contain negative values")
    
    # Check that events don't exceed totals
    treatment_valid = data[treatment_events_col] <= data[treatment_total_col]
    control_valid = data[control_events_col] <= data[control_total_col]
    
    if not treatment_valid.all():
        n_invalid = (~treatment_valid).sum()
        raise DataValidationError(
            f"{n_invalid} studies have treatment events > treatment total"
        )
    
    if not control_valid.all():
        n_invalid = (~control_valid).sum()
        raise DataValidationError(
            f"{n_invalid} studies have control events > control total"
        )


def check_continuous_outcome_data(data: DataFrame,
                                 treatment_mean_col: str = 'treatment_mean',
                                 treatment_sd_col: str = 'treatment_sd',
                                 treatment_n_col: str = 'treatment_n',
                                 control_mean_col: str = 'control_mean',
                                 control_sd_col: str = 'control_sd',
                                 control_n_col: str = 'control_n') -> None:
    """
    Check continuous outcome data for meta-analysis.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data to check
    treatment_mean_col : str
        Name of treatment mean column
    treatment_sd_col : str
        Name of treatment SD column
    treatment_n_col : str
        Name of treatment sample size column
    control_mean_col : str
        Name of control mean column
    control_sd_col : str
        Name of control SD column
    control_n_col : str
        Name of control sample size column
    
    Raises
    ------
    DataValidationError
        If data fails validation
    """
    required_cols = [treatment_mean_col, treatment_sd_col, treatment_n_col,
                    control_mean_col, control_sd_col, control_n_col]
    
    check_data_frame(data)
    check_required_columns(data, required_cols)
    
    # Check means (can be any finite value)
    for mean_col in [treatment_mean_col, control_mean_col]:
        check_numeric_column(data, mean_col, finite_only=True)
    
    # Check standard deviations (must be positive)
    for sd_col in [treatment_sd_col, control_sd_col]:
        check_numeric_column(data, sd_col, positive_only=True, finite_only=True)
    
    # Check sample sizes (must be positive integers)
    for n_col in [treatment_n_col, control_n_col]:
        check_numeric_column(data, n_col, positive_only=True, finite_only=True)
        
        # Check for non-integer values
        col_data = data[n_col]
        if not (col_data == col_data.astype(int)).all():
            raise DataValidationError(f"Column '{n_col}' must contain integer values")


def check_parameter(value: Any, name: str, 
                   valid_types: Optional[Union[type, tuple]] = None,
                   valid_values: Optional[List] = None,
                   min_value: Optional[float] = None,
                   max_value: Optional[float] = None,
                   required: bool = True) -> Any:
    """
    Check parameter validity.
    
    Parameters
    ----------
    value : any
        Value to check
    name : str
        Parameter name
    valid_types : type or tuple of types, optional
        Valid types
    valid_values : list, optional
        Valid values
    min_value : float, optional
        Minimum value
    max_value : float, optional
        Maximum value
    required : bool, default True
        Whether parameter is required
    
    Returns
    -------
    any
        Validated value
    
    Raises
    ------
    ParameterValidationError
        If parameter fails validation
    """
    if value is None:
        if required:
            raise ParameterValidationError(f"Parameter '{name}' is required")
        return value
    
    if valid_types and not isinstance(value, valid_types):
        if isinstance(valid_types, tuple):
            type_names = [t.__name__ for t in valid_types]
            expected = " or ".join(type_names)
        else:
            expected = valid_types.__name__
        actual = type(value).__name__
        raise ParameterValidationError(
            f"Parameter '{name}' must be {expected}, got {actual}"
        )
    
    if valid_values is not None and value not in valid_values:
        raise ParameterValidationError(
            f"Parameter '{name}' must be one of {valid_values}, got {value}"
        )
    
    if min_value is not None and value < min_value:
        raise ParameterValidationError(
            f"Parameter '{name}' must be >= {min_value}, got {value}"
        )
    
    if max_value is not None and value > max_value:
        raise ParameterValidationError(
            f"Parameter '{name}' must be <= {max_value}, got {value}"
        )
    
    return value


def check_array_like(value: Any, name: str, 
                    min_length: Optional[int] = None,
                    max_length: Optional[int] = None,
                    numeric: bool = True) -> Array:
    """
    Check that value is array-like.
    
    Parameters
    ----------
    value : any
        Value to check
    name : str
        Parameter name
    min_length : int, optional
        Minimum length
    max_length : int, optional
        Maximum length
    numeric : bool, default True
        Whether array should be numeric
    
    Returns
    -------
    numpy.ndarray
        Validated array
    
    Raises
    ------
    ParameterValidationError
        If parameter fails validation
    """
    try:
        arr = np.asarray(value)
    except (ValueError, TypeError):
        raise ParameterValidationError(f"Parameter '{name}' must be array-like")
    
    if numeric and not np.issubdtype(arr.dtype, np.number):
        raise ParameterValidationError(f"Parameter '{name}' must be numeric")
    
    if min_length is not None and len(arr) < min_length:
        raise ParameterValidationError(
            f"Parameter '{name}' must have length >= {min_length}, got {len(arr)}"
        )
    
    if max_length is not None and len(arr) > max_length:
        raise ParameterValidationError(
            f"Parameter '{name}' must have length <= {max_length}, got {len(arr)}"
        )
    
    return arr


def check_probability(value: float, name: str) -> float:
    """
    Check that value is a valid probability (0 <= p <= 1).
    
    Parameters
    ----------
    value : float
        Value to check
    name : str
        Parameter name
    
    Returns
    -------
    float
        Validated probability
    
    Raises
    ------
    ParameterValidationError
        If value is not a valid probability
    """
    return check_parameter(value, name, valid_types=(int, float), 
                          min_value=0.0, max_value=1.0)


def check_confidence_level(value: float, name: str = "confidence_level") -> float:
    """
    Check that value is a valid confidence level.
    
    Parameters
    ----------
    value : float
        Value to check
    name : str
        Parameter name
    
    Returns
    -------
    float
        Validated confidence level
    
    Raises
    ------
    ParameterValidationError
        If value is not a valid confidence level
    """
    return check_probability(value, name)


def warn_if_few_studies(n_studies: int, method: str, threshold: int = 5) -> None:
    """
    Issue warning if number of studies is small for the method.
    
    Parameters
    ----------
    n_studies : int
        Number of studies
    method : str
        Method name
    threshold : int, default 5
        Minimum recommended number of studies
    """
    if n_studies < threshold:
        import warnings
        warnings.warn(
            f"{method} with only {n_studies} studies may be unreliable. "
            f"Consider using with at least {threshold} studies.",
            UserWarning
        )