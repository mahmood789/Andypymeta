"""
Logging configuration for PyMeta.

This module sets up consistent logging across the package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import warnings


# Default logging configuration
DEFAULT_LOG_LEVEL = logging.WARNING
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class PyMetaFormatter(logging.Formatter):
    """Custom formatter for PyMeta logging."""
    
    def format(self, record):
        """Format log record with color coding if terminal supports it."""
        # Add color coding for different log levels
        colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m'  # Magenta
        }
        
        reset = '\033[0m'
        
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            record.levelname = f"{colors.get(record.levelname, '')}{record.levelname}{reset}"
        
        return super().format(record)


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    log_file: Optional[Path] = None,
    console: bool = True
) -> None:
    """
    Setup logging configuration for PyMeta.
    
    Parameters
    ----------
    level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format_string : str, optional
        Custom format string for log messages
    log_file : Path, optional
        Path to log file for file logging
    console : bool, default True
        Whether to enable console logging
    """
    # Get log level
    if level is None:
        level = DEFAULT_LOG_LEVEL
    elif isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Get format string
    if format_string is None:
        format_string = DEFAULT_LOG_FORMAT
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_formatter = PyMetaFormatter(format_string, DEFAULT_DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string, DEFAULT_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Setup package logger
    pymeta_logger = logging.getLogger('pymeta')
    pymeta_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(f'pymeta.{name}')


def set_log_level(level: str) -> None:
    """
    Set the log level for all PyMeta loggers.
    
    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper())
    
    # Update all PyMeta loggers
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith('pymeta'):
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_level)
    
    # Update handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setLevel(log_level)


def disable_warnings(category: Optional[type] = None) -> None:
    """
    Disable specific warning categories.
    
    Parameters
    ----------
    category : type, optional
        Warning category to disable. If None, disables all warnings.
    """
    if category is None:
        warnings.filterwarnings('ignore')
    else:
        warnings.filterwarnings('ignore', category=category)


def enable_warnings(category: Optional[type] = None) -> None:
    """
    Enable specific warning categories.
    
    Parameters
    ----------
    category : type, optional
        Warning category to enable. If None, enables all warnings.
    """
    if category is None:
        warnings.filterwarnings('default')
    else:
        warnings.filterwarnings('default', category=category)


# Setup default logging on import
setup_logging()

# Package logger
logger = get_logger(__name__)