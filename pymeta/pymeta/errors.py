"""
Custom errors and exceptions for PyMeta.

This module defines specific exception classes for different error conditions.
"""


class PyMetaError(Exception):
    """Base exception class for PyMeta."""
    pass


class DataError(PyMetaError):
    """Exception raised for data-related errors."""
    pass


class InvalidDataError(DataError):
    """Exception raised when input data is invalid."""
    pass


class MissingDataError(DataError):
    """Exception raised when required data is missing."""
    pass


class IncompatibleDataError(DataError):
    """Exception raised when data formats are incompatible."""
    pass


class ModelError(PyMetaError):
    """Exception raised for model-related errors."""
    pass


class InvalidModelError(ModelError):
    """Exception raised when model specification is invalid."""
    pass


class ConvergenceError(ModelError):
    """Exception raised when model fails to converge."""
    pass


class EstimationError(ModelError):
    """Exception raised when parameter estimation fails."""
    pass


class AnalysisError(PyMetaError):
    """Exception raised for analysis-related errors."""
    pass


class InsufficientDataError(AnalysisError):
    """Exception raised when insufficient data for analysis."""
    pass


class InvalidMethodError(AnalysisError):
    """Exception raised when analysis method is invalid."""
    pass


class NumericalError(AnalysisError):
    """Exception raised for numerical computation errors."""
    pass


class VisualizationError(PyMetaError):
    """Exception raised for plotting and visualization errors."""
    pass


class InvalidPlotTypeError(VisualizationError):
    """Exception raised when plot type is invalid."""
    pass


class PlottingError(VisualizationError):
    """Exception raised when plot generation fails."""
    pass


class IOError(PyMetaError):
    """Exception raised for input/output errors."""
    pass


class FileFormatError(IOError):
    """Exception raised when file format is not supported."""
    pass


class ExportError(IOError):
    """Exception raised when data export fails."""
    pass


class ImportError(IOError):
    """Exception raised when data import fails."""
    pass


class BiasTestError(PyMetaError):
    """Exception raised for publication bias test errors."""
    pass


class InvalidTestError(BiasTestError):
    """Exception raised when bias test is invalid for the data."""
    pass


class TestExecutionError(BiasTestError):
    """Exception raised when bias test execution fails."""
    pass


class LivingReviewError(PyMetaError):
    """Exception raised for living review errors."""
    pass


class SearchError(LivingReviewError):
    """Exception raised when literature search fails."""
    pass


class UpdateError(LivingReviewError):
    """Exception raised when review update fails."""
    pass


class NotificationError(LivingReviewError):
    """Exception raised when notification sending fails."""
    pass


class ConfigurationError(PyMetaError):
    """Exception raised for configuration errors."""
    pass


class InvalidConfigError(ConfigurationError):
    """Exception raised when configuration is invalid."""
    pass


class MissingConfigError(ConfigurationError):
    """Exception raised when required configuration is missing."""
    pass


class ValidationError(PyMetaError):
    """Exception raised for validation errors."""
    pass


class ParameterValidationError(ValidationError):
    """Exception raised when parameter validation fails."""
    pass


class DataValidationError(ValidationError):
    """Exception raised when data validation fails."""
    pass


class ResultValidationError(ValidationError):
    """Exception raised when result validation fails."""
    pass


# Error handling utilities

def handle_error(error: Exception, context: str = None) -> None:
    """
    Handle errors with appropriate logging and re-raising.
    
    Parameters
    ----------
    error : Exception
        The exception to handle
    context : str, optional
        Additional context information
    """
    from .logging import get_logger
    
    logger = get_logger(__name__)
    
    error_message = str(error)
    if context:
        error_message = f"{context}: {error_message}"
    
    if isinstance(error, PyMetaError):
        logger.error(error_message)
    else:
        logger.exception(f"Unexpected error - {error_message}")
    
    # Re-raise the original exception
    raise error


def validate_parameter(value, name: str, valid_types=None, valid_values=None, 
                      min_value=None, max_value=None, required=True):
    """
    Validate a parameter value.
    
    Parameters
    ----------
    value : any
        Value to validate
    name : str
        Parameter name
    valid_types : type or tuple of types, optional
        Valid types for the parameter
    valid_values : list, optional
        Valid values for the parameter
    min_value : float, optional
        Minimum value (for numeric parameters)
    max_value : float, optional
        Maximum value (for numeric parameters)
    required : bool, default True
        Whether the parameter is required
    
    Raises
    ------
    ParameterValidationError
        If validation fails
    """
    if value is None:
        if required:
            raise ParameterValidationError(f"Required parameter '{name}' is None")
        return
    
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
    
    if valid_values and value not in valid_values:
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


def validate_data_columns(data, required_columns, optional_columns=None):
    """
    Validate that data contains required columns.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data to validate
    required_columns : list
        List of required column names
    optional_columns : list, optional
        List of optional column names
    
    Raises
    ------
    DataValidationError
        If validation fails
    """
    import pandas as pd
    
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("Data must be a pandas DataFrame")
    
    if data.empty:
        raise DataValidationError("Data cannot be empty")
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}"
        )
    
    # Check for missing values in required columns
    for col in required_columns:
        if data[col].isnull().any():
            n_missing = data[col].isnull().sum()
            raise DataValidationError(
                f"Column '{col}' contains {n_missing} missing values"
            )


def check_numeric_data(data, column_name):
    """
    Check that column contains valid numeric data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data to check
    column_name : str
        Name of column to check
    
    Raises
    ------
    DataValidationError
        If validation fails
    """
    import numpy as np
    
    if column_name not in data.columns:
        raise DataValidationError(f"Column '{column_name}' not found in data")
    
    column = data[column_name]
    
    # Check for non-numeric values
    if not np.issubdtype(column.dtype, np.number):
        raise DataValidationError(f"Column '{column_name}' must contain numeric data")
    
    # Check for infinite values
    if np.isinf(column).any():
        raise DataValidationError(f"Column '{column_name}' contains infinite values")
    
    # Check for all zero values (for variance/SE columns)
    if column_name.lower() in ['variance', 'standard_error', 'se'] and (column <= 0).any():
        raise DataValidationError(f"Column '{column_name}' must contain positive values")