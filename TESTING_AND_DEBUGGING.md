# Testing and Debugging Guide

This document provides comprehensive guidance on testing and debugging code in the Andypymeta repository.

## Overview

The Andypymeta repository includes extensive testing and debugging infrastructure:

- **Testing Framework**: pytest with coverage reporting
- **Debug Management**: Comprehensive debugging utilities with performance tracking
- **Code Quality**: Linting, formatting, and type checking
- **Continuous Integration**: Automated testing pipeline

## Testing Infrastructure

### Test Organization

```
tests/
├── conftest.py              # Test configuration and fixtures
├── test_core.py             # Tests for core metadata functionality
├── test_debug.py            # Tests for debugging utilities
├── test_utils.py            # Tests for utility functions
├── test_integration.py      # Integration tests
└── test_examples.py         # Tests for example scripts
```

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=andypymeta --cov-report=term-missing

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m "not slow"             # Skip slow tests

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_core.py

# Run specific test method
pytest tests/test_core.py::TestMetadataHandler::test_load_metadata_from_dict
```

### Test Coverage

The test suite achieves 98% code coverage across all modules:

- **Core module**: 100% coverage - All metadata handling functionality
- **Debug module**: 96% coverage - Debugging and performance tracking
- **Utils module**: 100% coverage - All utility functions
- **Integration**: 100% coverage - End-to-end workflows

### Test Categories

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Tests combining multiple components
- `@pytest.mark.slow` - Tests that take longer to run

## Debugging Infrastructure

### DebugManager Class

The `DebugManager` provides comprehensive debugging capabilities:

```python
from andypymeta import DebugManager

# Initialize debug manager
debug_mgr = DebugManager(enabled=True)

# Function debugging with performance tracking
@debug_mgr.debug_function(trace_args=True, trace_return=True, measure_time=True)
def my_function(param1, param2):
    return param1 + param2

# Context-based debugging
with debug_mgr.debug_context("processing_data"):
    # Your code here
    pass

# Conditional breakpoints
debug_mgr.breakpoint(condition=some_condition, message="Debug checkpoint")

# Variable logging
debug_mgr.log_variables(var1=value1, var2=value2)

# Performance reporting
debug_mgr.print_performance_report()
```

### Debug Features

#### Function Debugging
- **Argument tracing**: Log function arguments and return values
- **Performance measurement**: Track execution time
- **Exception handling**: Detailed error logging with stack traces
- **Call stack tracking**: Monitor function call hierarchy

#### Context Management
- **Debug contexts**: Group related operations for easier tracking
- **Execution timing**: Measure time spent in specific code blocks
- **Exception handling**: Proper cleanup and error reporting

#### Interactive Debugging
- **Conditional breakpoints**: Trigger debugger based on conditions
- **Variable inspection**: Log and examine variable states
- **Frame analysis**: Access local variables from calling frames

#### Performance Analysis
- **Function timing**: Track execution times across function calls
- **Statistics collection**: Min, max, average, and total execution times
- **Performance reporting**: Formatted reports with detailed metrics

### MetadataHandler Debugging

The `MetadataHandler` includes built-in debugging support:

```python
from andypymeta import MetadataHandler

# Initialize with debugging enabled
handler = MetadataHandler(debug=True)

# All operations are automatically logged
handler.load_metadata(data)
handler.set("key.nested", "value")
result = handler.get("key.nested")
```

Debug output includes:
- Data loading operations
- Key access patterns
- Value modifications
- Validation results
- File I/O operations

## Code Quality Tools

### Linting and Formatting

```bash
# Format code with black
black andypymeta tests examples

# Sort imports with isort
isort andypymeta tests examples

# Lint code with flake8
flake8 andypymeta tests examples

# Type checking with mypy
mypy andypymeta
```

### Pre-commit Hooks

Set up pre-commit hooks for automatic code quality checks:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Example Usage

### Basic Testing Example

```python
import pytest
from andypymeta import MetadataHandler

def test_metadata_operations():
    """Test basic metadata operations."""
    handler = MetadataHandler()
    
    # Test data loading
    test_data = {"name": "test", "version": "1.0.0"}
    handler.load_metadata(test_data)
    
    # Test data retrieval
    assert handler.get("name") == "test"
    assert handler.get("version") == "1.0.0"
    
    # Test data modification
    handler.set("status", "active")
    assert handler.get("status") == "active"
```

### Advanced Debugging Example

```python
from andypymeta import DebugManager, MetadataHandler

def debug_complex_workflow():
    """Example of comprehensive debugging."""
    debug_mgr = DebugManager(enabled=True)
    
    @debug_mgr.debug_function(measure_time=True)
    def process_data(data):
        handler = MetadataHandler(debug=True)
        handler.load_metadata(data)
        
        # Log current state
        debug_mgr.log_variables(
            data_keys=len(data),
            handler_type=type(handler).__name__
        )
        
        # Conditional debugging
        if len(data) > 10:
            debug_mgr.breakpoint(
                condition=True,
                message="Large dataset detected"
            )
        
        return handler.to_dict()
    
    # Process with debug context
    with debug_mgr.debug_context("workflow"):
        result = process_data({"key": "value"})
    
    # Generate performance report
    debug_mgr.print_performance_report()
```

## Integration Testing

### Testing Complete Workflows

Integration tests verify that all components work together correctly:

```python
def test_complete_metadata_workflow():
    """Test complete metadata workflow with debugging."""
    debug_mgr = DebugManager(enabled=True)
    
    with debug_mgr.debug_context("integration_test"):
        # Create and configure handler
        handler = MetadataHandler(debug=True)
        
        # Load complex nested data
        complex_data = {
            "application": {
                "name": "test_app",
                "modules": {"core": {"version": "1.0"}}
            }
        }
        handler.load_metadata(complex_data)
        
        # Perform operations
        handler.set("application.status", "active")
        
        # Validate results
        assert handler.get("application.name") == "test_app"
        assert handler.get("application.status") == "active"
        
        # Check performance metrics
        perf_report = debug_mgr.get_performance_report()
        assert len(perf_report) > 0
```

## Continuous Integration

### GitHub Actions Workflow

The repository includes automated CI/CD that runs:

1. **Code quality checks**: linting, formatting, type checking
2. **Comprehensive testing**: all test categories with coverage
3. **Multi-version testing**: Python 3.8 through 3.12
4. **Coverage reporting**: integrated with codecov

### Local Development Workflow

Recommended development workflow:

```bash
# 1. Set up development environment
pip install -e ".[dev]"
pre-commit install

# 2. Make code changes
# ... edit files ...

# 3. Run quality checks
black .
isort .
flake8
mypy andypymeta

# 4. Run tests
pytest --cov=andypymeta

# 5. Commit changes (pre-commit hooks will run automatically)
git add .
git commit -m "Your commit message"
```

## Best Practices

### Testing Best Practices

1. **Write tests first**: Use TDD when possible
2. **Test behavior, not implementation**: Focus on what the code should do
3. **Use descriptive test names**: Make test purpose clear
4. **Test edge cases**: Include boundary conditions and error cases
5. **Keep tests independent**: Each test should be able to run in isolation
6. **Use fixtures effectively**: Share common setup code

### Debugging Best Practices

1. **Enable debugging early**: Add debug support from the beginning
2. **Use context managers**: Group related operations with debug contexts
3. **Log meaningful information**: Include relevant variable states
4. **Performance awareness**: Monitor execution times for critical paths
5. **Conditional debugging**: Use breakpoints judiciously
6. **Clean up debug code**: Remove debug statements before production

### Code Quality Best Practices

1. **Run linters regularly**: Integrate into development workflow
2. **Maintain type annotations**: Use mypy for type checking
3. **Format consistently**: Use black and isort for code formatting
4. **Write documentation**: Include docstrings and comments
5. **Review coverage reports**: Ensure comprehensive test coverage

## Troubleshooting

### Common Issues

**Tests failing due to imports:**
```bash
# Install package in development mode
pip install -e .
```

**Coverage reports missing lines:**
```bash
# Run with coverage debugging
pytest --cov=andypymeta --cov-report=html
# Open htmlcov/index.html to see detailed coverage
```

**Type checking errors:**
```bash
# Run mypy with verbose output
mypy andypymeta --show-error-codes
```

**Debug output not showing:**
```python
# Ensure debugging is enabled
debug_mgr = DebugManager(enabled=True)
handler = MetadataHandler(debug=True)
```

### Performance Debugging

When debugging performance issues:

1. Use `@debug_function(measure_time=True)` on suspected slow functions
2. Use debug contexts to measure code block execution times
3. Generate performance reports to identify bottlenecks
4. Use conditional breakpoints to examine state at critical points

### Memory Debugging

For memory-related issues:

1. Log object sizes and types with `debug_mgr.log_variables()`
2. Use debug contexts to track memory usage in specific sections
3. Monitor large data structures and file operations
4. Test with large datasets to identify memory leaks

This comprehensive testing and debugging infrastructure ensures code quality and makes development more efficient and reliable.