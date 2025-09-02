"""Test that examples work correctly."""

import subprocess
import sys
from pathlib import Path
import pytest


def test_debug_demo_runs_without_error():
    """Test that the debug demo example runs without errors."""
    repo_root = Path(__file__).parent.parent
    demo_script = repo_root / "examples" / "debug_demo.py"
    
    # Run the demo script
    result = subprocess.run(
        [sys.executable, str(demo_script)],
        capture_output=True,
        text=True,
        cwd=repo_root
    )
    
    # Check that it ran successfully
    assert result.returncode == 0, f"Demo failed with error: {result.stderr}"
    
    # Check that it produces expected output
    assert "🚀 Andypymeta Debugging Demonstration" in result.stdout
    assert "🎉 Demonstration completed!" in result.stdout
    assert "Performance Report" in result.stdout
    
    # Check that no critical errors were printed to stderr 
    if result.stderr:
        # Filter out expected debug messages and demonstration errors
        error_lines = [
            line for line in result.stderr.split('\n') 
            if line and not any(keyword in line for keyword in [
                'DEBUG', 'INFO', 'WARNING', 'ERROR', 'Traceback',
                'pytest-cov', 'coverage', 'Demonstration error for debugging',
                'ValueError: Demonstration error for debugging',
                'File "', '    ', '             ^'
            ])
        ]
        # Only fail if there are unexpected error lines
        if error_lines:
            pytest.fail(f"Unexpected errors: {error_lines}")