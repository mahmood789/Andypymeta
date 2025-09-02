"""Global test configuration and fixtures for pymeta."""

import os
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np
import pandas as pd


# Set session-wide random seed for reproducibility
np.random.seed(42)


@pytest.fixture(scope="session")
def test_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="pymeta_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def binary_table_2x2():
    """Fixture providing simple 2x2 table data for binary outcomes."""
    return {
        'events_treatment': np.array([10, 15, 8, 12]),
        'n_treatment': np.array([50, 60, 45, 55]),
        'events_control': np.array([5, 8, 6, 7]),
        'n_control': np.array([50, 55, 40, 50]),
        'study_labels': ['Study A', 'Study B', 'Study C', 'Study D']
    }


@pytest.fixture
def binary_table_with_zeros():
    """Fixture with some zero events to test continuity correction."""
    return {
        'events_treatment': np.array([0, 15, 0, 12, 8]),
        'n_treatment': np.array([50, 60, 45, 55, 40]),
        'events_control': np.array([0, 8, 3, 0, 5]),
        'n_control': np.array([50, 55, 40, 50, 35]),
        'study_labels': ['Study 1', 'Study 2', 'Study 3', 'Study 4', 'Study 5']
    }


@pytest.fixture
def continuous_effects_data():
    """Fixture providing continuous effect sizes for meta-analysis."""
    # Simulate some realistic meta-analysis data
    np.random.seed(123)  # Local seed for this fixture
    
    k = 8  # Number of studies
    true_effect = 0.5
    tau = 0.2  # Between-study heterogeneity
    
    # Study sample sizes
    n_per_group = np.random.randint(30, 200, k)
    
    # Within-study variances (function of sample size)
    within_var = 4 / n_per_group  # Assuming pooled SD ~ 2
    
    # Study-specific effects (with heterogeneity)
    study_effects = np.random.normal(true_effect, tau, k)
    
    # Observed effect sizes
    effect_sizes = np.random.normal(study_effects, np.sqrt(within_var))
    
    return {
        'effect_sizes': effect_sizes,
        'variances': within_var,
        'standard_errors': np.sqrt(within_var),
        'sample_sizes': n_per_group,
        'true_effect': true_effect,
        'true_tau': tau,
        'study_labels': [f'Study {i+1}' for i in range(k)]
    }


@pytest.fixture
def fe_re_data():
    """Fixture providing data suitable for fixed/random effects comparison."""
    return {
        'effect_sizes': np.array([0.3, 0.5, 0.4, 0.6, 0.45, 0.55, 0.35, 0.5]),
        'variances': np.array([0.04, 0.06, 0.05, 0.08, 0.07, 0.09, 0.04, 0.06]),
        'study_labels': [f'RCT-{i+1}' for i in range(8)]
    }


@pytest.fixture
def demo_dataframe():
    """Fixture providing a pandas DataFrame with meta-analysis data."""
    np.random.seed(456)
    
    n_studies = 12
    study_ids = [f'Study_{i:03d}' for i in range(1, n_studies + 1)]
    
    # Simulate realistic meta-analysis data
    sample_sizes = np.random.randint(50, 300, n_studies)
    true_effects = np.random.normal(0.4, 0.15, n_studies)
    ses = 2 / np.sqrt(sample_sizes)
    effect_sizes = np.random.normal(true_effects, ses)
    
    # Add some study characteristics
    years = np.random.randint(2010, 2024, n_studies)
    countries = np.random.choice(['USA', 'UK', 'Germany', 'Canada', 'Australia'], n_studies)
    
    return pd.DataFrame({
        'study_id': study_ids,
        'effect_size': effect_sizes,
        'standard_error': ses,
        'variance': ses**2,
        'sample_size': sample_sizes,
        'year': years,
        'country': countries,
        'weight_fe': 1 / ses**2
    })


@pytest.fixture
def heterogeneous_data():
    """Fixture with high heterogeneity for testing I² and tau²."""
    return {
        'effect_sizes': np.array([0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.85]),
        'variances': np.array([0.05, 0.06, 0.04, 0.08, 0.05, 0.07, 0.04, 0.09]),
        'expected_high_heterogeneity': True
    }


@pytest.fixture
def homogeneous_data():
    """Fixture with low heterogeneity for testing."""
    return {
        'effect_sizes': np.array([0.48, 0.52, 0.49, 0.51, 0.50, 0.53, 0.47, 0.50]),
        'variances': np.array([0.04, 0.05, 0.04, 0.06, 0.05, 0.06, 0.04, 0.05]),
        'expected_low_heterogeneity': True
    }


# Reference implementations for validation

def reference_dl_tau2(effect_sizes, variances):
    """
    Reference implementation of DerSimonian-Laird tau² estimation.
    
    This is a simple, clearly correct implementation for validation
    against the main implementation.
    """
    effect_sizes = np.asarray(effect_sizes)
    variances = np.asarray(variances)
    k = len(effect_sizes)
    
    if k <= 1:
        return 0.0
    
    # Fixed effects weights
    weights = 1 / variances
    sum_weights = np.sum(weights)
    
    # Weighted mean
    weighted_mean = np.sum(weights * effect_sizes) / sum_weights
    
    # Cochran's Q
    Q = np.sum(weights * (effect_sizes - weighted_mean)**2)
    
    # C coefficient
    C = sum_weights - np.sum(weights**2) / sum_weights
    
    # DL tau²
    if C <= 0:
        return 0.0
    
    tau2_dl = max(0, (Q - (k - 1)) / C)
    return tau2_dl


def reference_re_summary(effect_sizes, variances, tau2):
    """
    Reference implementation for random effects summary.
    
    Given effect sizes, variances, and tau², calculate the
    random effects pooled estimate and its variance.
    """
    effect_sizes = np.asarray(effect_sizes)
    variances = np.asarray(variances)
    
    # Random effects weights
    weights = 1 / (variances + tau2)
    
    # Pooled effect and variance
    pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
    pooled_variance = 1 / np.sum(weights)
    
    return pooled_effect, pooled_variance


def reference_egger_regression(effect_sizes, standard_errors):
    """
    Reference implementation of Egger's regression test.
    
    Simple implementation for validation purposes.
    """
    effect_sizes = np.asarray(effect_sizes)
    standard_errors = np.asarray(standard_errors)
    
    # Precision
    precision = 1 / standard_errors
    
    # Weighted regression: effect_size = intercept + slope * precision
    weights = precision**2
    
    # Calculate regression coefficients manually
    n = len(effect_sizes)
    sum_w = np.sum(weights)
    sum_wx = np.sum(weights * precision)
    sum_wy = np.sum(weights * effect_sizes)
    sum_wxx = np.sum(weights * precision**2)
    sum_wxy = np.sum(weights * precision * effect_sizes)
    
    # Normal equations
    denominator = sum_w * sum_wxx - sum_wx**2
    
    if abs(denominator) < 1e-12:
        return np.nan, np.nan  # Singular matrix
    
    intercept = (sum_wxx * sum_wy - sum_wx * sum_wxy) / denominator
    slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denominator
    
    return intercept, slope


# Pytest configuration helpers

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "viz: mark test as visualization test"
    )
    config.addinivalue_line(
        "markers", "optional: mark test as requiring optional dependencies"
    )


def pytest_collection_modifyitems(config, items):
    """Add default unit marker to tests without explicit markers."""
    for item in items:
        # Get all markers for this test
        markers = [mark.name for mark in item.iter_markers()]
        
        # If no category marker is present, add 'unit'
        category_markers = {'unit', 'integration', 'slow', 'viz', 'optional'}
        if not any(marker in category_markers for marker in markers):
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def capture_plots():
    """Fixture to capture matplotlib plots for testing."""
    import matplotlib.pyplot as plt
    
    # Store original state
    original_backend = plt.get_backend()
    
    # Set to non-interactive backend for testing
    plt.switch_backend('Agg')
    
    yield
    
    # Clean up any figures
    plt.close('all')
    
    # Restore backend
    plt.switch_backend(original_backend)


# Skip conditions for optional dependencies

def skip_if_no_pymc():
    """Skip test if PyMC is not available."""
    try:
        import pymc
        return False
    except ImportError:
        return True


def skip_if_no_statsmodels():
    """Skip test if statsmodels is not available."""
    try:
        import statsmodels
        return False
    except ImportError:
        return True


# Pytest markers for conditional skipping
requires_pymc = pytest.mark.skipif(
    skip_if_no_pymc(), 
    reason="PyMC not available"
)

requires_statsmodels = pytest.mark.skipif(
    skip_if_no_statsmodels(),
    reason="statsmodels not available"
)