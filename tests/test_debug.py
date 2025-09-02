"""Test the debug functionality."""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from andypymeta.debug import DebugManager, debug_context, debug_function


class TestDebugManager:
    """Test DebugManager class."""

    def test_init_default(self):
        """Test default initialization."""
        debug_mgr = DebugManager()
        assert debug_mgr.enabled is True
        assert debug_mgr.log_level == logging.DEBUG
        assert debug_mgr._call_stack == []
        assert debug_mgr._performance_data == {}

    def test_init_disabled(self):
        """Test initialization with debugging disabled."""
        debug_mgr = DebugManager(enabled=False, log_level=logging.INFO)
        assert debug_mgr.enabled is False
        assert debug_mgr.log_level == logging.INFO

    def test_debug_function_decorator_enabled(self, debug_manager):
        """Test function debugging decorator when enabled."""

        @debug_manager.debug_function()
        def test_func(x, y=10):
            return x + y

        with patch.object(debug_manager.logger, "debug") as mock_debug:
            result = test_func(5, y=15)

        assert result == 20

        # Check that debug messages were logged
        debug_calls = [call.args[0] for call in mock_debug.call_args_list]
        assert any("test_func(5, y=15)" in call for call in debug_calls)
        assert any("returned: 20" in call for call in debug_calls)
        assert any("took" in call for call in debug_calls)

    def test_debug_function_decorator_disabled(self):
        """Test function debugging decorator when disabled."""
        debug_mgr = DebugManager(enabled=False)

        @debug_mgr.debug_function()
        def test_func(x):
            return x * 2

        with patch.object(debug_mgr.logger, "debug") as mock_debug:
            result = test_func(5)

        assert result == 10
        assert not mock_debug.called

    def test_debug_function_with_exception(self, debug_manager):
        """Test function debugging with exception."""

        @debug_manager.debug_function()
        def failing_func():
            raise ValueError("Test error")

        with patch.object(debug_manager.logger, "error") as mock_error:
            with pytest.raises(ValueError):
                failing_func()

        # Check that error was logged
        assert mock_error.called
        error_msg = mock_error.call_args[0][0]
        assert "ValueError" in error_msg
        assert "Test error" in error_msg

    def test_debug_context_manager(self, debug_manager):
        """Test debug context manager."""
        with patch.object(debug_manager.logger, "debug") as mock_debug:
            with debug_manager.debug_context("test_context"):
                time.sleep(0.01)  # Small delay to test timing

        debug_calls = [call.args[0] for call in mock_debug.call_args_list]
        assert any("Entering context: test_context" in call for call in debug_calls)
        assert any("Exiting context: test_context" in call for call in debug_calls)

    def test_debug_context_with_exception(self, debug_manager):
        """Test debug context manager with exception."""
        with patch.object(debug_manager.logger, "error") as mock_error:
            with pytest.raises(ValueError):
                with debug_manager.debug_context("failing_context"):
                    raise ValueError("Context error")

        assert mock_error.called
        error_msg = mock_error.call_args[0][0]
        assert "failing_context" in error_msg
        assert "ValueError" in error_msg

    def test_breakpoint_triggered(self, debug_manager):
        """Test breakpoint when condition is true."""
        with patch("pdb.set_trace") as mock_pdb:
            with patch.object(debug_manager.logger, "debug") as mock_debug:
                debug_manager.breakpoint(condition=True, message="Test breakpoint")

        assert mock_pdb.called
        debug_calls = [call.args[0] for call in mock_debug.call_args_list]
        assert any("Breakpoint: Test breakpoint" in call for call in debug_calls)

    def test_breakpoint_not_triggered(self, debug_manager):
        """Test breakpoint when condition is false."""
        with patch("pdb.set_trace") as mock_pdb:
            debug_manager.breakpoint(condition=False)

        assert not mock_pdb.called

    def test_breakpoint_disabled(self):
        """Test breakpoint when debugging is disabled."""
        debug_mgr = DebugManager(enabled=False)

        with patch("pdb.set_trace") as mock_pdb:
            debug_mgr.breakpoint(condition=True)

        assert not mock_pdb.called

    def test_log_variables_explicit(self, debug_manager):
        """Test logging explicitly provided variables."""
        test_var1 = "test_value"
        test_var2 = 42

        with patch.object(debug_manager.logger, "debug") as mock_debug:
            debug_manager.log_variables(var1=test_var1, var2=test_var2)

        debug_calls = [call.args[0] for call in mock_debug.call_args_list]
        assert any("var1 = 'test_value'" in call for call in debug_calls)
        assert any("var2 = 42" in call for call in debug_calls)

    def test_performance_tracking(self, debug_manager):
        """Test performance data collection."""

        @debug_manager.debug_function(measure_time=True)
        def timed_func():
            time.sleep(0.01)
            return "done"

        # Call function multiple times
        for _ in range(3):
            timed_func()

        report = debug_manager.get_performance_report()
        
        # The function name will include the full path
        func_names = list(report.keys())
        assert len(func_names) == 1
        func_name = func_names[0]
        assert "timed_func" in func_name

        stats = report[func_name]
        assert stats["calls"] == 3
        assert stats["total_time"] > 0
        assert stats["avg_time"] > 0
        assert stats["min_time"] > 0
        assert stats["max_time"] > 0

    def test_performance_report_empty(self, debug_manager):
        """Test performance report when no data."""
        report = debug_manager.get_performance_report()
        assert report == {}

    def test_print_performance_report(self, debug_manager):
        """Test printing performance report."""

        @debug_manager.debug_function(measure_time=True)
        def test_func():
            pass

        test_func()

        with patch("builtins.print") as mock_print:
            debug_manager.print_performance_report()

        # Check that print was called with performance data
        assert mock_print.called
        print_calls = [str(call) for call in mock_print.call_args_list]
        output = " ".join(print_calls)
        assert "Performance Report" in output

    def test_print_performance_report_empty(self, debug_manager):
        """Test printing performance report when empty."""
        with patch("builtins.print") as mock_print:
            debug_manager.print_performance_report()

        mock_print.assert_called_with("No performance data available")

    def test_clear_performance_data(self, debug_manager):
        """Test clearing performance data."""

        @debug_manager.debug_function(measure_time=True)
        def test_func():
            pass

        test_func()

        # Verify data exists
        assert debug_manager._performance_data

        # Clear and verify
        debug_manager.clear_performance_data()
        assert not debug_manager._performance_data

    def test_enable_disable(self, debug_manager):
        """Test enabling and disabling debugging."""
        debug_manager.disable()
        assert not debug_manager.enabled

        debug_manager.enable()
        assert debug_manager.enabled

    def test_set_log_level(self, debug_manager):
        """Test setting log level."""
        debug_manager.set_log_level(logging.WARNING)
        assert debug_manager.log_level == logging.WARNING
        assert debug_manager.logger.level == logging.WARNING


class TestDebugDecorators:
    """Test debug decorator functions."""

    def test_debug_function_convenience(self):
        """Test convenience debug_function decorator."""

        @debug_function()
        def test_func(x):
            return x * 2

        # Just verify it doesn't crash
        result = test_func(5)
        assert result == 10

    def test_debug_context_convenience(self):
        """Test convenience debug_context."""
        with debug_context("test"):
            pass  # Just verify it doesn't crash
