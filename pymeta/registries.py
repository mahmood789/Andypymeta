"""Plugin registries for models, estimators, plots, and bias tests."""

from typing import Dict, Callable, Any, List, Type
from functools import wraps
from .errors import RegistryError


class Registry:
    """Base registry for plugin components."""
    
    def __init__(self):
        self._items: Dict[str, Any] = {}
    
    def register(self, name: str, item: Any) -> Any:
        """Register an item with given name."""
        if name in self._items:
            raise RegistryError(f"Item '{name}' already registered")
        self._items[name] = item
        return item
    
    def get(self, name: str) -> Any:
        """Get registered item by name."""
        if name not in self._items:
            raise RegistryError(f"Item '{name}' not found in registry")
        return self._items[name]
    
    def list_items(self) -> List[str]:
        """List all registered item names."""
        return list(self._items.keys())
    
    def is_registered(self, name: str) -> bool:
        """Check if item is registered."""
        return name in self._items


# Global registries
estimator_registry = Registry()
model_registry = Registry()
plot_registry = Registry()
bias_test_registry = Registry()


def register_estimator(name: str):
    """Decorator to register tau2 estimators."""
    def decorator(func: Callable) -> Callable:
        estimator_registry.register(name, func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def register_model(name: str):
    """Decorator to register meta-analysis models."""
    def decorator(cls: Type) -> Type:
        model_registry.register(name, cls)
        return cls
    return decorator


def register_plot(name: str):
    """Decorator to register plot functions."""
    def decorator(func: Callable) -> Callable:
        plot_registry.register(name, func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def register_bias_test(name: str):
    """Decorator to register bias test functions."""
    def decorator(func: Callable) -> Callable:
        bias_test_registry.register(name, func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_estimator(name: str) -> Callable:
    """Get registered tau2 estimator."""
    return estimator_registry.get(name)


def get_model(name: str) -> Type:
    """Get registered model class."""
    return model_registry.get(name)


def get_plot(name: str) -> Callable:
    """Get registered plot function."""
    return plot_registry.get(name)


def get_bias_test(name: str) -> Callable:
    """Get registered bias test function."""
    return bias_test_registry.get(name)


def list_estimators() -> List[str]:
    """List available tau2 estimators."""
    return estimator_registry.list_items()


def list_models() -> List[str]:
    """List available models."""
    return model_registry.list_items()


def list_plots() -> List[str]:
    """List available plots."""
    return plot_registry.list_items()


def list_bias_tests() -> List[str]:
    """List available bias tests."""
    return bias_test_registry.list_items()