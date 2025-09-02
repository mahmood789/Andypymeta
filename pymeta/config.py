"""
Configuration management for PyMeta package.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os


@dataclass
class GlobalConfig:
    """Global configuration for PyMeta."""
    
    # Default analysis settings
    default_model: str = "RE"
    default_tau2_method: str = "REML"
    use_hksj: bool = False
    alpha: float = 0.05
    
    # Plot settings
    figure_dpi: int = 300
    figure_format: str = "png"
    
    # Living meta-analysis settings
    default_update_interval: int = 3600  # seconds
    max_retries: int = 3
    
    # Scheduler settings
    scheduler_timezone: str = "UTC"
    
    # Optional APScheduler settings
    apscheduler_config: Dict[str, Any] = field(default_factory=lambda: {
        'executors': {
            'default': {'type': 'threadpool', 'max_workers': 20}
        },
        'job_defaults': {
            'coalesce': True,
            'max_instances': 1
        }
    })
    
    @classmethod
    def from_env(cls) -> "GlobalConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if "PYMETA_USE_HKSJ" in os.environ:
            config.use_hksj = os.environ["PYMETA_USE_HKSJ"].lower() == "true"
            
        if "PYMETA_ALPHA" in os.environ:
            try:
                config.alpha = float(os.environ["PYMETA_ALPHA"])
            except ValueError:
                pass
                
        if "PYMETA_DEFAULT_MODEL" in os.environ:
            model = os.environ["PYMETA_DEFAULT_MODEL"].upper()
            if model in ["FE", "RE"]:
                config.default_model = model
                
        if "PYMETA_TAU2_METHOD" in os.environ:
            method = os.environ["PYMETA_TAU2_METHOD"].upper()
            if method in ["DL", "REML", "PM", "ML"]:
                config.default_tau2_method = method
                
        if "PYMETA_UPDATE_INTERVAL" in os.environ:
            try:
                config.default_update_interval = int(os.environ["PYMETA_UPDATE_INTERVAL"])
            except ValueError:
                pass
                
        return config


# Global configuration instance
_global_config: Optional[GlobalConfig] = None


def get_config() -> GlobalConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = GlobalConfig.from_env()
    return _global_config


def set_config(config: GlobalConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _global_config
    _global_config = None


# Convenience functions for common configuration access
def get_use_hksj() -> bool:
    """Get the global HKSJ setting."""
    return get_config().use_hksj


def set_use_hksj(use_hksj: bool) -> None:
    """Set the global HKSJ setting."""
    config = get_config()
    config.use_hksj = use_hksj


def get_alpha() -> float:
    """Get the global alpha level."""
    return get_config().alpha


def set_alpha(alpha: float) -> None:
    """Set the global alpha level."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    config = get_config()
    config.alpha = alpha