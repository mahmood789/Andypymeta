"""
Configuration management for PyMeta.

This module handles configuration settings for the package.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


# Default configuration
DEFAULT_CONFIG = {
    'meta_analysis': {
        'default_model': 'random',
        'default_tau2_method': 'dl',
        'alpha': 0.05,
        'digits': 4
    },
    'plotting': {
        'default_style': 'classic',
        'figure_size': (10, 8),
        'dpi': 300,
        'font_size': 12
    },
    'bias_tests': {
        'default_tests': ['egger', 'begg'],
        'trimfill_side': 'auto',
        'alpha': 0.05
    },
    'living_reviews': {
        'update_frequency': 'monthly',
        'auto_search': True,
        'notification_methods': ['email']
    },
    'cache': {
        'enabled': True,
        'max_size': 100,  # MB
        'ttl': 3600  # seconds
    }
}


class Config:
    """Configuration management class."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration."""
        self._config = DEFAULT_CONFIG.copy()
        
        if config_dict:
            self._update_config(config_dict)
        
        # Load from environment variables
        self._load_from_env()
        
        # Load from config file if it exists
        self._load_from_file()
    
    def _update_config(self, new_config: Dict[str, Any], base_config: Optional[Dict[str, Any]] = None) -> None:
        """Recursively update configuration."""
        if base_config is None:
            base_config = self._config
        
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(value, base_config[key])
            else:
                base_config[key] = value
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Meta-analysis settings
        if 'PYMETA_DEFAULT_MODEL' in os.environ:
            self._config['meta_analysis']['default_model'] = os.environ['PYMETA_DEFAULT_MODEL']
        
        if 'PYMETA_DEFAULT_TAU2_METHOD' in os.environ:
            self._config['meta_analysis']['default_tau2_method'] = os.environ['PYMETA_DEFAULT_TAU2_METHOD']
        
        if 'PYMETA_ALPHA' in os.environ:
            self._config['meta_analysis']['alpha'] = float(os.environ['PYMETA_ALPHA'])
        
        # Plotting settings
        if 'PYMETA_PLOT_STYLE' in os.environ:
            self._config['plotting']['default_style'] = os.environ['PYMETA_PLOT_STYLE']
        
        if 'PYMETA_DPI' in os.environ:
            self._config['plotting']['dpi'] = int(os.environ['PYMETA_DPI'])
        
        # Cache settings
        if 'PYMETA_CACHE_ENABLED' in os.environ:
            self._config['cache']['enabled'] = os.environ['PYMETA_CACHE_ENABLED'].lower() == 'true'
        
        if 'PYMETA_CACHE_SIZE' in os.environ:
            self._config['cache']['max_size'] = int(os.environ['PYMETA_CACHE_SIZE'])
    
    def _load_from_file(self) -> None:
        """Load configuration from file."""
        # Look for config file in various locations
        config_locations = [
            Path.cwd() / 'pymeta.yaml',
            Path.cwd() / 'pymeta.yml',
            Path.cwd() / '.pymeta.yaml',
            Path.cwd() / '.pymeta.yml',
            Path.home() / '.pymeta.yaml',
            Path.home() / '.pymeta.yml',
            Path.home() / '.config' / 'pymeta' / 'config.yaml',
            Path.home() / '.config' / 'pymeta' / 'config.yml'
        ]
        
        for config_path in config_locations:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        file_config = yaml.safe_load(f)
                    
                    if file_config:
                        self._update_config(file_config)
                    break
                    
                except Exception as e:
                    # Log warning but continue
                    import warnings
                    warnings.warn(f"Failed to load config from {config_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._update_config(config_dict)
    
    def save(self, file_path: Path) -> None:
        """Save current configuration to file."""
        with open(file_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._config = DEFAULT_CONFIG.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self._config})"


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def set_config(**kwargs) -> None:
    """Set configuration values."""
    for key, value in kwargs.items():
        config.set(key, value)


def reset_config() -> None:
    """Reset configuration to defaults."""
    config.reset()


def load_config_file(file_path: Path) -> None:
    """Load configuration from file."""
    with open(file_path, 'r') as f:
        file_config = yaml.safe_load(f)
    
    if file_config:
        config.update(file_config)


def save_config_file(file_path: Path) -> None:
    """Save current configuration to file."""
    config.save(file_path)