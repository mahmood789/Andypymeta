"""Integration tests for the complete package."""

import tempfile
from pathlib import Path

import pytest

from andypymeta import DebugManager, MetadataHandler, get_version, validate_metadata


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_complete_workflow(self):
        """Test complete metadata workflow with debugging."""
        # Create debug manager
        debug_mgr = DebugManager(enabled=True)

        # Create metadata handler with debugging
        @debug_mgr.debug_function(trace_args=True, trace_return=True)
        def process_metadata(metadata_dict):
            handler = MetadataHandler(debug=True)
            handler.load_metadata(metadata_dict)

            # Manipulate metadata
            handler.set("processed", True)
            handler.set("version", get_version())

            return handler.to_dict()

        # Test data
        test_metadata = {
            "name": "integration_test",
            "description": "Testing integration",
            "config": {"enabled": True, "max_items": 100},
        }

        # Process with debugging
        with debug_mgr.debug_context("integration_test"):
            result = process_metadata(test_metadata)

        # Verify results
        assert result["name"] == "integration_test"
        assert result["processed"] is True
        assert result["version"] == get_version()
        assert result["config"]["enabled"] is True

        # Check performance data was collected
        perf_report = debug_mgr.get_performance_report()
        assert len(perf_report) > 0

    def test_metadata_validation_and_debugging(self):
        """Test metadata validation with debugging."""
        debug_mgr = DebugManager(enabled=True)

        @debug_mgr.debug_function()
        def validate_and_fix_metadata(metadata):
            # First validation
            errors = validate_metadata(
                metadata,
                required_keys=["name", "version"],
                type_constraints={"name": str, "version": str},
            )

            debug_mgr.log_variables(
                metadata=metadata, errors=errors, error_count=len(errors)
            )

            if errors:
                # Fix missing required fields
                if "name" not in metadata:
                    metadata["name"] = "default_name"
                if "version" not in metadata:
                    metadata["version"] = "1.0.0"

            return metadata, errors

        # Test with invalid metadata
        invalid_metadata = {"description": "Missing required fields"}

        with debug_mgr.debug_context("validation_test"):
            fixed_metadata, errors = validate_and_fix_metadata(invalid_metadata)

        assert len(errors) > 0  # Original had errors
        assert fixed_metadata["name"] == "default_name"
        assert fixed_metadata["version"] == "1.0.0"

    def test_file_operations_with_debugging(self):
        """Test file operations with comprehensive debugging."""
        debug_mgr = DebugManager(enabled=True)

        @debug_mgr.debug_function(measure_time=True)
        def create_and_save_metadata(file_path):
            handler = MetadataHandler(debug=True)

            # Create sample metadata
            metadata = {
                "project": "file_test",
                "created_by": "integration_test",
                "features": ["testing", "debugging", "metadata"],
                "config": {"auto_save": True, "backup_count": 5},
            }

            handler.load_metadata(metadata)
            handler.save_metadata(file_path)

            return handler.to_dict()

        @debug_mgr.debug_function(measure_time=True)
        def load_and_validate_metadata(file_path):
            handler = MetadataHandler(debug=True)
            loaded_data = handler.load_metadata(file_path)

            # Validate loaded data
            errors = handler.validate(
                {"project": str, "created_by": str, "features": list, "config": dict}
            )

            return loaded_data, errors

        # Test with temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            with debug_mgr.debug_context("file_operations"):
                # Save metadata
                original_data = create_and_save_metadata(temp_path)

                # Load and validate
                loaded_data, validation_errors = load_and_validate_metadata(temp_path)

            # Verify file operations worked
            assert temp_path.exists()
            assert original_data == loaded_data
            assert len(validation_errors) == 0

            # Check performance metrics
            perf_report = debug_mgr.get_performance_report()
            assert len(perf_report) >= 2  # Should have data for both functions

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_error_handling_with_debugging(self):
        """Test error handling with debugging enabled."""
        debug_mgr = DebugManager(enabled=True)

        @debug_mgr.debug_function()
        def function_that_fails():
            raise ValueError("Intentional test error")

        @debug_mgr.debug_function()
        def function_with_conditional_breakpoint(should_break=False):
            debug_mgr.breakpoint(
                condition=should_break, message="Conditional breakpoint triggered"
            )
            return "success"

        # Test exception handling
        with pytest.raises(ValueError):
            function_that_fails()

        # Test conditional breakpoint (should not trigger)
        result = function_with_conditional_breakpoint(should_break=False)
        assert result == "success"

    def test_nested_metadata_operations(self):
        """Test complex nested metadata operations."""
        handler = MetadataHandler(debug=True)
        debug_mgr = DebugManager(enabled=True)

        # Complex nested structure
        complex_metadata = {
            "application": {
                "name": "complex_app",
                "version": "2.1.0",
                "modules": {
                    "core": {"version": "2.1.0", "dependencies": ["utils", "logging"]},
                    "ui": {"version": "2.0.5", "dependencies": ["core", "themes"]},
                },
                "configuration": {
                    "database": {"host": "localhost", "port": 5432, "ssl": True},
                    "logging": {"level": "INFO", "handlers": ["console", "file"]},
                },
            }
        }

        with debug_mgr.debug_context("nested_operations"):
            handler.load_metadata(complex_metadata)

            # Test deeply nested access
            assert handler.get("application.name") == "complex_app"
            assert handler.get("application.modules.core.version") == "2.1.0"
            assert handler.get("application.configuration.database.port") == 5432

            # Test nested modification
            handler.set("application.modules.core.status", "active")
            handler.set("application.configuration.cache.enabled", True)

            # Verify changes
            assert handler.get("application.modules.core.status") == "active"
            assert handler.get("application.configuration.cache.enabled") is True

            # Test merging with nested data
            update_data = {
                "application": {
                    "modules": {
                        "analytics": {"version": "1.0.0", "dependencies": ["core"]}
                    },
                    "configuration": {
                        "logging": {"level": "DEBUG"}  # This should override
                    },
                }
            }

            handler.merge(update_data)

            # Verify merge results
            assert handler.get("application.modules.analytics.version") == "1.0.0"
            assert handler.get("application.configuration.logging.level") == "DEBUG"
            assert handler.get("application.configuration.logging.handlers") == [
                "console",
                "file",
            ]  # Should remain
