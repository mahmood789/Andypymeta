"""Custom exception hierarchy for PyMeta."""


class PyMetaError(Exception):
    """Base exception for PyMeta package."""
    pass


class DataError(PyMetaError):
    """Raised when there are issues with input data."""
    pass


class ConvergenceError(PyMetaError):
    """Raised when statistical methods fail to converge."""
    pass


class ModelError(PyMetaError):
    """Raised when there are issues with model specification or fitting."""
    pass


class EstimationError(PyMetaError):
    """Raised when parameter estimation fails."""
    pass


class ValidationError(PyMetaError):
    """Raised when data validation fails."""
    pass


class PlottingError(PyMetaError):
    """Raised when plotting operations fail."""
    pass


class BiasTestError(PyMetaError):
    """Raised when bias tests fail."""
    pass


class TSAError(PyMetaError):
    """Raised when Trial Sequential Analysis fails."""
    pass


class RegistryError(PyMetaError):
    """Raised when registry operations fail."""
    pass