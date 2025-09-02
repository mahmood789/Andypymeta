#!/usr/bin/env python3
"""Example script demonstrating andypymeta debugging capabilities."""

import time
from pathlib import Path

from andypymeta import DebugManager, MetadataHandler


def create_sample_data():
    """Create sample metadata for demonstration."""
    return {
        "project": {
            "name": "demo_project",
            "version": "1.2.3",
            "description": "Demonstration of andypymeta debugging",
            "authors": ["Demo Author", "Test User"],
            "license": "Apache-2.0",
        },
        "build": {
            "tools": ["pytest", "black", "mypy"],
            "python_version": ">=3.8",
            "dependencies": {
                "runtime": ["requests", "click"],
                "development": ["pytest", "black", "mypy", "flake8"],
            },
        },
        "configuration": {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "features": {"auto_update": True, "telemetry": False, "debug_mode": True},
        },
    }


def main():
    """Main demonstration function."""
    print("🚀 Andypymeta Debugging Demonstration")
    print("=" * 50)

    # Initialize debug manager
    debug_mgr = DebugManager(enabled=True)

    # Demonstrate function debugging
    @debug_mgr.debug_function(trace_args=True, trace_return=True, measure_time=True)
    def process_metadata_with_validation(data):
        """Process metadata with validation and debugging."""
        handler = MetadataHandler(debug=True)

        # Load and validate data
        handler.load_metadata(data)

        # Log current variables for debugging
        debug_mgr.log_variables(
            project_name=handler.get("project.name"),
            version=handler.get("project.version"),
            tool_count=len(handler.get("build.tools", [])),
        )

        # Perform some operations
        handler.set("processed_at", "2023-12-01T10:00:00Z")
        handler.set("processing.status", "completed")

        # Simulate some processing time
        time.sleep(0.1)

        return handler.to_dict()

    @debug_mgr.debug_function(measure_time=True)
    def analyze_dependencies(metadata):
        """Analyze project dependencies."""
        runtime_deps = (
            metadata.get("build", {}).get("dependencies", {}).get("runtime", [])
        )
        dev_deps = (
            metadata.get("build", {}).get("dependencies", {}).get("development", [])
        )

        analysis = {
            "runtime_count": len(runtime_deps),
            "development_count": len(dev_deps),
            "total_count": len(runtime_deps) + len(dev_deps),
            "has_testing": any("test" in dep for dep in dev_deps),
            "has_linting": any(dep in ["black", "flake8", "mypy"] for dep in dev_deps),
        }

        return analysis

    # Create sample data
    sample_data = create_sample_data()

    # Main processing with debug context
    with debug_mgr.debug_context("main_processing"):
        print("\n📊 Processing metadata with debugging...")

        # Process metadata
        processed_data = process_metadata_with_validation(sample_data)

        # Analyze dependencies
        dependency_analysis = analyze_dependencies(processed_data)

        print("\n✅ Processing completed!")
        print(f"   Project: {processed_data['project']['name']}")
        print(f"   Version: {processed_data['project']['version']}")
        print(f"   Total dependencies: {dependency_analysis['total_count']}")

    # Demonstrate error handling with debugging
    print("\n🔍 Demonstrating error handling...")

    @debug_mgr.debug_function()
    def function_with_potential_error(should_fail=False):
        """Function that might fail for demonstration."""
        if should_fail:
            raise ValueError("Demonstration error for debugging")
        return "Success!"

    try:
        # This will work
        result = function_with_potential_error(should_fail=False)
        print(f"   ✅ Function succeeded: {result}")

        # This will fail and show debug info
        function_with_potential_error(should_fail=True)
    except ValueError as e:
        print(f"   ❌ Caught expected error: {e}")

    # Demonstrate conditional breakpoint (won't actually break in demo)
    print("\n🔧 Demonstrating conditional debugging...")

    @debug_mgr.debug_function()
    def function_with_conditional_debugging(items):
        """Function with conditional debugging logic."""
        for i, item in enumerate(items):
            # This would trigger a breakpoint if enabled interactively
            # debug_mgr.breakpoint(
            #     condition=(i > 5),
            #     message=f"Processing item {i}: {item}"
            # )

            # Log variables every few iterations
            if i % 3 == 0:
                debug_mgr.log_variables(
                    current_index=i, current_item=item, remaining=len(items) - i - 1
                )

        return len(items)

    test_items = [f"item_{i}" for i in range(10)]
    processed_count = function_with_conditional_debugging(test_items)
    print(f"   ✅ Processed {processed_count} items with debugging")

    # Show performance report
    print("\n📈 Performance Report:")
    print("-" * 30)
    debug_mgr.print_performance_report()

    # Demonstrate file operations with debugging
    print("\n💾 Demonstrating file operations...")

    @debug_mgr.debug_function(measure_time=True)
    def save_and_load_demo(data, filename):
        """Save and load data with debugging."""
        handler = MetadataHandler(debug=True)
        handler.load_metadata(data)

        # Save to file
        handler.save_metadata(filename)

        # Load back and verify
        new_handler = MetadataHandler(debug=True)
        loaded_data = new_handler.load_metadata(filename)

        return loaded_data

    demo_file = Path("/tmp/demo_metadata.json")
    try:
        with debug_mgr.debug_context("file_operations"):
            reloaded_data = save_and_load_demo(processed_data, demo_file)

        print("   ✅ Successfully saved and loaded metadata")
        print(f"   📁 File: {demo_file}")
        print(f"   📊 Keys: {len(reloaded_data)}")

    finally:
        # Cleanup
        if demo_file.exists():
            demo_file.unlink()

    print("\n🎉 Demonstration completed!")
    print("💡 In real debugging scenarios, you would:")
    print("   - Enable breakpoints for interactive debugging")
    print("   - Use debug_mgr.log_variables() to inspect state")
    print("   - Monitor performance with the timing decorators")
    print("   - Use debug contexts to track execution flow")


if __name__ == "__main__":
    main()
