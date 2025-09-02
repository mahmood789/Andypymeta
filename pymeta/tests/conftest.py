"""
Test configuration and fixtures for PyMeta.

This module provides common test fixtures and configuration.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_binary_data():
    """Sample binary outcome data for testing."""
    return pd.DataFrame({
        'study': [f'Study_{i}' for i in range(1, 11)],
        'treatment_events': [10, 15, 8, 12, 6, 18, 9, 11, 14, 7],
        'treatment_total': [100, 120, 80, 110, 90, 150, 85, 105, 130, 75],
        'control_events': [5, 8, 12, 6, 9, 10, 15, 7, 8, 11],
        'control_total': [100, 110, 85, 105, 95, 140, 90, 100, 125, 80],
        'year': [2010, 2012, 2014, 2015, 2016, 2018, 2019, 2020, 2021, 2022]
    })


@pytest.fixture
def sample_continuous_data():
    """Sample continuous outcome data for testing."""
    return pd.DataFrame({
        'study': [f'Study_{i}' for i in range(1, 9)],
        'treatment_mean': [5.2, 4.8, 5.5, 4.9, 5.1, 5.3, 4.7, 5.0],
        'treatment_sd': [1.2, 1.1, 1.3, 1.0, 1.2, 1.4, 1.1, 1.2],
        'treatment_n': [50, 60, 45, 55, 48, 52, 58, 50],
        'control_mean': [4.5, 4.2, 4.8, 4.3, 4.4, 4.6, 4.1, 4.3],
        'control_sd': [1.1, 1.0, 1.2, 0.9, 1.1, 1.3, 1.0, 1.1],
        'control_n': [48, 58, 42, 52, 45, 50, 55, 48],
        'year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    })


@pytest.fixture
def sample_effect_size_data():
    """Sample effect size data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'study': [f'Study_{i}' for i in range(1, 16)],
        'effect_size': np.random.normal(0.3, 0.2, 15),
        'variance': np.random.uniform(0.01, 0.1, 15),
        'sample_size': np.random.randint(50, 200, 15),
        'year': np.random.randint(2010, 2023, 15)
    })


@pytest.fixture
def test_data_dir():
    """Directory containing test data files."""
    return Path(__file__).parent / 'data'


@pytest.fixture
def binary_small_csv(test_data_dir):
    """Path to small binary CSV test file."""
    return test_data_dir / 'binary_small.csv'


@pytest.fixture
def continuous_small_csv(test_data_dir):
    """Path to small continuous CSV test file."""
    return test_data_dir / 'continuous_small.csv'


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test."""
    np.random.seed(42)


@pytest.fixture
def tolerance():
    """Default numerical tolerance for tests."""
    return 1e-10


@pytest.fixture
def alpha():
    """Default significance level for tests."""
    return 0.05


class TestDataGenerator:
    """Helper class for generating test data."""
    
    @staticmethod
    def create_binary_data(n_studies=10, seed=42):
        """Create synthetic binary outcome data."""
        np.random.seed(seed)
        return pd.DataFrame({
            'study': [f'Study_{i}' for i in range(1, n_studies + 1)],
            'treatment_events': np.random.binomial(100, 0.3, n_studies),
            'treatment_total': np.random.randint(80, 120, n_studies),
            'control_events': np.random.binomial(100, 0.2, n_studies),
            'control_total': np.random.randint(80, 120, n_studies)
        })
    
    @staticmethod
    def create_continuous_data(n_studies=8, seed=42):
        """Create synthetic continuous outcome data."""
        np.random.seed(seed)
        return pd.DataFrame({
            'study': [f'Study_{i}' for i in range(1, n_studies + 1)],
            'treatment_mean': np.random.normal(5.0, 0.5, n_studies),
            'treatment_sd': np.random.uniform(1.0, 1.5, n_studies),
            'treatment_n': np.random.randint(40, 60, n_studies),
            'control_mean': np.random.normal(4.5, 0.5, n_studies),
            'control_sd': np.random.uniform(1.0, 1.5, n_studies),
            'control_n': np.random.randint(40, 60, n_studies)
        })
    
    @staticmethod
    def create_effect_size_data(n_studies=15, true_effect=0.3, tau_squared=0.04, seed=42):
        """Create synthetic effect size data."""
        np.random.seed(seed)
        
        # Generate variances
        variances = np.random.uniform(0.01, 0.1, n_studies)
        
        # Generate true study effects
        if tau_squared > 0:
            study_effects = np.random.normal(true_effect, np.sqrt(tau_squared), n_studies)
        else:
            study_effects = np.full(n_studies, true_effect)
        
        # Add sampling error
        observed_effects = study_effects + np.random.normal(0, np.sqrt(variances))
        
        return pd.DataFrame({
            'study': [f'Study_{i}' for i in range(1, n_studies + 1)],
            'effect_size': observed_effects,
            'variance': variances,
            'standard_error': np.sqrt(variances),
            'sample_size': np.random.randint(50, 200, n_studies)
        })


@pytest.fixture
def data_generator():
    """Test data generator instance."""
    return TestDataGenerator()


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "property: marks tests as property-based tests"
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests"
    )