"""Debugging utilities for comprehensive code debugging and profiling."""

import functools
import logging
import pdb
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class DebugManager:
    """Comprehensive debugging manager with multiple debugging tools."""

    def __init__(self, enabled: bool = True, log_level: int = logging.DEBUG) -> None:
        """Initialize the debug manager.

        Args:
            enabled: Whether debugging is enabled
            log_level: Logging level for debug messages
        """
        self.enabled = enabled
        self.log_level = log_level
        self.logger = self._setup_logger()
        self._call_stack: List[str] = []
        self._performance_data: Dict[str, List[float]] = {}

    def _setup_logger(self) -> logging.Logger:
        """Set up debug logger."""
        logger = logging.getLogger("andypymeta.debug")
        logger.setLevel(self.log_level)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def debug_function(
        self,
        trace_args: bool = True,
        trace_return: bool = True,
        measure_time: bool = True,
    ) -> Callable[[F], F]:
        """Decorator to debug function calls.

        Args:
            trace_args: Whether to trace function arguments
            trace_return: Whether to trace return values
            measure_time: Whether to measure execution time

        Returns:
            Decorated function with debugging capabilities
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self.enabled:
                    return func(*args, **kwargs)

                func_name = f"{func.__module__}.{func.__qualname__}"
                self._call_stack.append(func_name)

                # Log function entry
                if trace_args:
                    args_str = ", ".join([repr(arg) for arg in args])
                    kwargs_str = ", ".join(
                        [f"{k}={repr(v)}" for k, v in kwargs.items()]
                    )
                    all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                    self.logger.debug(f"→ {func_name}({all_args})")
                else:
                    self.logger.debug(f"→ {func_name}()")

                start_time = time.time() if measure_time else None

                try:
                    result = func(*args, **kwargs)

                    # Log execution time
                    if measure_time and start_time:
                        execution_time = time.time() - start_time
                        if func_name not in self._performance_data:
                            self._performance_data[func_name] = []
                        self._performance_data[func_name].append(execution_time)
                        self.logger.debug(f"⏱ {func_name} took {execution_time:.4f}s")

                    # Log return value
                    if trace_return:
                        self.logger.debug(f"← {func_name} returned: {repr(result)}")
                    else:
                        self.logger.debug(f"← {func_name} completed")

                    return result

                except Exception as e:
                    self.logger.error(f"✗ {func_name} raised {type(e).__name__}: {e}")
                    self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
                    raise

                finally:
                    self._call_stack.pop()

            return wrapper  # type: ignore

        return decorator

    @contextmanager
    def debug_context(self, name: str) -> Any:
        """Context manager for debugging code blocks.

        Args:
            name: Name of the debug context
        """
        if not self.enabled:
            yield
            return

        self.logger.debug(f"▶ Entering context: {name}")
        start_time = time.time()

        try:
            yield
        except Exception as e:
            self.logger.error(
                f"✗ Exception in context '{name}': {type(e).__name__}: {e}"
            )
            raise
        finally:
            end_time = time.time()
            self.logger.debug(
                f"◀ Exiting context: {name} (took {end_time - start_time:.4f}s)"
            )

    def breakpoint(self, condition: bool = True, message: str = "") -> None:
        """Conditional breakpoint for debugging.

        Args:
            condition: Whether to trigger the breakpoint
            message: Optional message to display
        """
        if not self.enabled or not condition:
            return

        if message:
            self.logger.debug(f"🔴 Breakpoint: {message}")
        else:
            self.logger.debug("🔴 Breakpoint triggered")

        # Get caller information
        frame = sys._getframe(1)
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        self.logger.debug(f"📍 Location: {filename}:{lineno}")
        self.logger.debug(f"📚 Call stack: {' → '.join(self._call_stack)}")

        # Start interactive debugger
        pdb.set_trace()

    def log_variables(self, frame: Optional[Any] = None, **variables: Any) -> None:
        """Log variable values for debugging.

        Args:
            frame: Frame to inspect (defaults to caller frame)
            **variables: Named variables to log
        """
        if not self.enabled:
            return

        if frame is None:
            frame = sys._getframe(1)

        # Log local variables from frame
        if hasattr(frame, "f_locals"):
            locals_vars = frame.f_locals
            self.logger.debug("📋 Local variables:")
            for name, value in locals_vars.items():
                if not name.startswith("_"):
                    self.logger.debug(f"  {name} = {repr(value)}")

        # Log explicitly passed variables
        if variables:
            self.logger.debug("📋 Tracked variables:")
            for name, value in variables.items():
                self.logger.debug(f"  {name} = {repr(value)}")

    def get_performance_report(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for debugged functions.

        Returns:
            Dictionary with performance statistics per function
        """
        report = {}

        for func_name, times in self._performance_data.items():
            if times:
                report[func_name] = {
                    "calls": len(times),
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                }

        return report

    def print_performance_report(self) -> None:
        """Print formatted performance report."""
        report = self.get_performance_report()

        if not report:
            print("No performance data available")
            return

        print("\n📊 Performance Report")
        print("=" * 60)

        for func_name, stats in report.items():
            print(f"\n🔧 {func_name}")
            print(f"   Calls: {stats['calls']}")
            print(f"   Total time: {stats['total_time']:.4f}s")
            print(f"   Average time: {stats['avg_time']:.4f}s")
            print(f"   Min time: {stats['min_time']:.4f}s")
            print(f"   Max time: {stats['max_time']:.4f}s")

    def clear_performance_data(self) -> None:
        """Clear all performance data."""
        self._performance_data.clear()
        self.logger.debug("Performance data cleared")

    def enable(self) -> None:
        """Enable debugging."""
        self.enabled = True
        self.logger.debug("Debugging enabled")

    def disable(self) -> None:
        """Disable debugging."""
        self.enabled = False

    def set_log_level(self, level: int) -> None:
        """Set logging level.

        Args:
            level: New logging level (e.g., logging.DEBUG, logging.INFO)
        """
        self.log_level = level
        self.logger.setLevel(level)


# Global debug manager instance
debug_manager = DebugManager()

# Convenience functions
debug_function = debug_manager.debug_function
debug_context = debug_manager.debug_context
breakpoint_debug = debug_manager.breakpoint
log_variables = debug_manager.log_variables
