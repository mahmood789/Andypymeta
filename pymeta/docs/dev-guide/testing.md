# Testing Guidelines

This document describes how tests are organized and how to run them in PyMeta.

## Test Organization

### Directory Structure

```
tests/
├── conftest.py                    # Shared test configuration
├── data/                          # Test data fixtures
│   ├── binary_small.csv
│   └── continuous_small.csv
├── unit/                          # Unit tests
│   ├── test_effects_binary.py
│   ├── test_effects_continuous.py
│   ├── test_estimators_tau2.py
│   ├── test_models_fixed_random.py
│   ├── test_multivariate.py
│   ├── test_glmm_binomial.py
│   ├── test_inference_prediction.py
│   ├── test_bias_suite.py
│   ├── test_viz_forest.py
│   └── test_viz_funnel.py
├── integration/                   # Integration tests
│   ├── test_cli_meta.py
│   ├── test_cli_plots.py
│   └── test_living_scheduler.py
├── property/                      # Property-based tests
│   ├── test_effects_properties.py
│   └── test_weights_stability.py
└── smoke/                        # Smoke tests
    └── test_quickstart_example.py
```

## Test Types

### Unit Tests

Test individual functions and classes in isolation.

```python
import pytest
import numpy as np
from pymeta.effects import binary

def test_risk_ratio_calculation():
    """Test risk ratio calculation with known values."""
    # Arrange
    treatment_events = 10
    treatment_total = 100
    control_events = 5
    control_total = 100
    
    # Act
    result = binary.risk_ratio(
        treatment_events, treatment_total,
        control_events, control_total
    )
    
    # Assert
    expected_rr = 2.0
    assert np.isclose(result.effect_size, expected_rr)
    assert result.lower_ci < result.effect_size < result.upper_ci
```

### Integration Tests

Test complete workflows and interactions between components.

```python
import pytest
from pymeta.api import meta_analysis
from pymeta.io import load_data

def test_complete_meta_analysis_workflow():
    """Test complete meta-analysis from data loading to results."""
    # Arrange
    data = load_data("tests/data/binary_small.csv")
    
    # Act
    result = meta_analysis(data, model="random", tau2_method="dl")
    
    # Assert
    assert result.overall_effect is not None
    assert result.heterogeneity is not None
    assert len(result.studies) == len(data)
```

### Property-Based Tests

Use Hypothesis to test statistical properties and invariants.

```python
import pytest
from hypothesis import given, strategies as st
import numpy as np
from pymeta.effects import continuous

@given(
    mean1=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    sd1=st.floats(0.1, 5, allow_nan=False, allow_infinity=False),
    n1=st.integers(10, 1000),
    mean2=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    sd2=st.floats(0.1, 5, allow_nan=False, allow_infinity=False),
    n2=st.integers(10, 1000)
)
def test_smd_properties(mean1, sd1, n1, mean2, sd2, n2):
    """Test properties of standardized mean difference."""
    result = continuous.standardized_mean_difference(
        mean1, sd1, n1, mean2, sd2, n2
    )
    
    # SMD should be finite
    assert np.isfinite(result.effect_size)
    
    # Confidence interval should contain effect size
    assert result.lower_ci <= result.effect_size <= result.upper_ci
    
    # Variance should be positive
    assert result.variance > 0
```

### Smoke Tests

High-level tests that verify basic functionality works.

```python
def test_quickstart_example():
    """Test that quickstart example runs without errors."""
    import pymeta as pm
    
    # This should run without raising exceptions
    data = pm.load_example_data("binary")
    result = pm.meta_analysis(data)
    plot = pm.forest_plot(result)
    
    assert result is not None
    assert plot is not None
```

## Running Tests

### All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pymeta --cov=cli
```

### Specific Test Types

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Property-based tests only
pytest tests/property/

# Smoke tests only
pytest tests/smoke/
```

### Specific Test Files

```bash
# Single test file
pytest tests/unit/test_effects_binary.py

# Single test function
pytest tests/unit/test_effects_binary.py::test_risk_ratio_calculation
```

### Test Options

```bash
# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run specific markers
pytest -m "not slow"

# Parallel execution
pytest -n auto
```

## Writing Tests

### Test Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Test Structure

Use the Arrange-Act-Assert pattern:

```python
def test_function_name():
    """Test description."""
    # Arrange - set up test data
    input_data = ...
    expected_result = ...
    
    # Act - call the function
    actual_result = function_under_test(input_data)
    
    # Assert - verify results
    assert actual_result == expected_result
```

### Fixtures

Use pytest fixtures for shared test data:

```python
@pytest.fixture
def sample_binary_data():
    """Sample binary outcome data for testing."""
    return pd.DataFrame({
        'study': ['A', 'B', 'C'],
        'treatment_events': [10, 15, 8],
        'treatment_total': [100, 120, 80],
        'control_events': [5, 8, 12],
        'control_total': [100, 110, 85]
    })

def test_with_fixture(sample_binary_data):
    """Test using fixture data."""
    result = some_function(sample_binary_data)
    assert result is not None
```

### Parametrized Tests

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize("method,expected", [
    ("dl", 0.1),
    ("ml", 0.12),
    ("reml", 0.11)
])
def test_tau2_methods(method, expected):
    """Test different tau-squared estimation methods."""
    result = estimate_tau2(data, method=method)
    assert np.isclose(result, expected, rtol=0.1)
```

### Testing Exceptions

```python
def test_invalid_input_raises_error():
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError, match="Sample size must be positive"):
        calculate_effect_size(n=-10)
```

## Test Data

### Synthetic Data

Create minimal synthetic datasets for testing:

```python
def create_test_data(n_studies=5):
    """Create synthetic test data."""
    np.random.seed(42)  # For reproducibility
    return pd.DataFrame({
        'effect_size': np.random.normal(0.5, 0.2, n_studies),
        'variance': np.random.uniform(0.01, 0.1, n_studies),
        'study_id': [f'Study_{i}' for i in range(n_studies)]
    })
```

### Real Data

Use small subsets of real datasets for integration tests.

## Performance Testing

### Benchmarking

```python
import time

def test_large_dataset_performance():
    """Test performance with large dataset."""
    data = create_large_test_data(n_studies=10000)
    
    start_time = time.time()
    result = meta_analysis(data)
    elapsed_time = time.time() - start_time
    
    # Should complete within reasonable time
    assert elapsed_time < 10.0  # seconds
```

### Memory Testing

Monitor memory usage for large datasets.

## Continuous Integration

### GitHub Actions

Tests run automatically on:

- All pull requests
- Pushes to main branch
- Multiple Python versions
- Multiple operating systems

### Test Coverage

- Aim for >90% code coverage
- Generate coverage reports
- Track coverage trends

## Debugging Tests

### Common Issues

1. **Flaky tests**: Use fixed random seeds
2. **Floating-point precision**: Use `np.isclose()`
3. **Platform differences**: Test on multiple platforms

### Debugging Tools

```bash
# Run with pdb debugger
pytest --pdb

# Verbose output
pytest -v -s

# Show local variables on failure
pytest --tb=long
```

## Test Maintenance

### Regular Tasks

- Update test data as needed
- Remove obsolete tests
- Refactor for clarity
- Monitor test execution time

### Best Practices

- Keep tests simple and focused
- Test edge cases and error conditions
- Use descriptive test names
- Document complex test logic

## See Also

- [Architecture](architecture.md) - Package architecture
- [Contributing](contributing.md) - How to contribute