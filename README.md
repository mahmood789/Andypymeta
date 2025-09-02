# pymeta-suite: Comprehensive Meta-Analysis Toolkit

A modern Python package for conducting meta-analyses with comprehensive testing infrastructure.

## Features

### Core Meta-Analysis Functionality
- **Binary Effects**: Odds ratios, risk ratios, risk differences with continuity correction
- **Effect Size Estimation**: DerSimonian-Laird, Hedges, Hunter-Schmidt tau² estimators
- **Fixed & Random Effects**: Standard meta-analysis models with heterogeneity testing
- **Prediction Intervals**: For estimating effects in new studies
- **Publication Bias**: Egger's test, Begg's test, trim-and-fill methods
- **Visualization**: Forest plots and funnel plots

### Optional Advanced Methods
- **GLMM**: Generalized Linear Mixed Models for complex meta-analyses
- **Bayesian**: Bayesian random effects and RCT/RWE synthesis
- **RVE**: Robust variance estimation for dependent effect sizes

### Command Line Interface
- **CLI Tools**: Complete command-line interface for meta-analysis
- **Living Mode**: Conceptual framework for living meta-analyses

## Installation

```bash
# Basic installation
pip install -e .

# With optional dependencies
pip install -e ".[bayes,rve,xgb,viz,dev]"
```

## Quick Start

```python
import numpy as np
from pymeta.effects import binary_effects
from pymeta.models import random_effects

# Binary meta-analysis
events_t = [10, 15, 8, 12]
n_t = [50, 60, 45, 55]
events_c = [5, 8, 6, 7]
n_c = [50, 55, 40, 50]

effects = binary_effects(events_t, n_t, events_c, n_c, effect_measure="OR")
result = random_effects(effects['effect_size'], effects['variance'])
print(f"Pooled OR: {np.exp(result['pooled_effect']):.2f}")
```

## Testing Infrastructure

### Test Categories

The package includes comprehensive testing with different categories:

- **Unit Tests** (`tests/unit/`): Fast, isolated component testing
- **Optional Tests** (`tests/optional/`): Tests requiring extra dependencies
- **Integration Tests** (`tests/integration/`): End-to-end CLI and workflow testing
- **Visualization Tests**: Plot generation and file output validation

### Running Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run specific test categories
pytest -m "unit"                    # Unit tests only
pytest -m "optional"                # Optional dependency tests
pytest -m "slow"                    # Computationally intensive tests
pytest -m "viz"                     # Visualization tests
pytest -m "integration"             # Integration tests

# Run with coverage
pytest --cov=pymeta tests/unit/

# Run excluding certain categories
pytest -m "not slow and not viz"    # Fast tests only
```

### Test Markers

- `unit`: Fast, isolated tests (default)
- `optional`: Requires optional dependencies (bayes, torch, xgb, rve)
- `slow`: Computationally intensive (GLMM, Bayesian methods)
- `viz`: Visualization tests that create plots
- `integration`: End-to-end testing (CLI, living mode)

### Tox Testing

Multi-environment testing with tox:

```bash
# Test across Python versions
tox

# Test specific environments
tox -e py311-unit          # Python 3.11 unit tests
tox -e py312-optional      # Python 3.12 optional tests
tox -e lint                # Code quality checks
tox -e coverage            # Coverage reporting
```

## Development

### Code Quality

```bash
# Linting and formatting
ruff check src/pymeta tests/
black src/pymeta tests/

# Type checking
mypy src/pymeta
```

### Continuous Integration

GitHub Actions workflow includes:
- Multi-Python testing (3.10, 3.11, 3.12)
- Optional dependency testing
- Integration testing
- Code quality checks
- Package building and validation

## Package Structure

```
src/pymeta/
├── effects/           # Effect size calculations
├── estimators/        # Tau² estimation methods
├── models/            # Fixed/random effects models
├── inference/         # Prediction intervals, heterogeneity
├── bias/              # Publication bias detection
├── viz/               # Visualization tools
├── glmm/              # GLMM methods (optional)
├── bayes/             # Bayesian methods (optional)
├── rve/               # Robust variance estimation (optional)
└── cli/               # Command-line interface

tests/
├── conftest.py        # Global fixtures and configuration
├── unit/              # Unit tests
├── optional/          # Optional dependency tests
└── integration/       # Integration tests
```

## Testing Features

### Fixtures and Reference Implementations

- **Global Fixtures**: Binary tables, continuous effects data, demo dataframes
- **Reference Implementations**: Independent implementations for validation
- **Reproducibility**: Fixed random seeds for deterministic testing
- **Error Handling**: Comprehensive edge case testing

### Performance Testing

- **Scalability**: Tests with large datasets
- **Memory Usage**: Memory efficiency validation
- **Computational Time**: Performance benchmarking for slow methods

### Robustness Testing

- **Edge Cases**: Empty inputs, single studies, extreme values
- **Numerical Stability**: Very small/large values, singular matrices
- **Missing Dependencies**: Graceful fallbacks when optional packages unavailable

## License

MIT License - see LICENSE file for details.

## Contributing

1. Run the full test suite: `pytest`
2. Check code quality: `ruff check` and `black --check`
3. Add tests for new functionality
4. Update documentation as needed

## Dependencies

### Core Dependencies
- numpy ≥ 1.21.0
- pandas ≥ 1.3.0
- scipy ≥ 1.7.0
- statsmodels ≥ 0.13.0
- matplotlib ≥ 3.5.0
- plotly ≥ 5.0.0

### Optional Dependencies
- PyMC ≥ 5.0.0 (Bayesian methods)
- ArviZ ≥ 0.12.0 (Bayesian diagnostics)
- XGBoost ≥ 1.5.0 (Machine learning methods)
- scikit-learn ≥ 1.0.0 (RVE methods)

### Development Dependencies
- pytest ≥ 7.0.0
- pytest-cov ≥ 4.0.0
- black ≥ 22.0.0
- ruff ≥ 0.1.0
- mypy ≥ 1.0.0
- tox ≥ 4.0.0