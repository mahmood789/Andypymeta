# Architecture Overview

This document describes the overall architecture and design principles of PyMeta.

## Design Principles

### Modularity

PyMeta is designed with a modular architecture that separates concerns:

- **Core**: Fundamental data structures and algorithms
- **Effects**: Effect size calculations
- **Models**: Meta-analytic models
- **Inference**: Statistical inference methods
- **Visualization**: Plotting and visualization
- **I/O**: Data import/export
- **CLI**: Command-line interface
- **GUI**: Graphical user interface

### Extensibility

The package is designed to be easily extensible:

- Plugin architecture for new effect size measures
- Modular tau-squared estimators
- Customizable plotting themes
- Flexible data import/export

### Performance

Key performance considerations:

- Efficient numerical algorithms using NumPy/SciPy
- Vectorized operations where possible
- Optional parallel processing for computationally intensive tasks
- Memory-efficient data structures

## Package Structure

```
pymeta/
├── core/           # Core functionality
│   ├── data.py     # Data structures
│   ├── weights.py  # Weight calculation
│   ├── estimators/ # Statistical estimators
│   ├── models/     # Meta-analytic models
│   └── inference/  # Inference methods
├── effects/        # Effect size calculations
├── bias/          # Publication bias methods
├── viz/           # Visualization
├── living/        # Living review functionality
├── io/            # Data I/O
├── api/           # Public API
└── gui/           # GUI components
```

## Core Components

### Data Model

Central data structures for representing:

- Study data
- Effect sizes
- Meta-analysis results
- Model parameters

### Estimator Framework

Unified interface for statistical estimators:

- Tau-squared estimators
- Effect size estimators
- Confidence interval methods

### Model Framework

Flexible framework for meta-analytic models:

- Fixed-effect models
- Random-effects models
- Multivariate models
- GLMM models

## API Design

### Public API

The public API (`pymeta.api`) provides high-level functions for common tasks:

```python
import pymeta as pm

# High-level meta-analysis
result = pm.meta_analysis(data, model='random')

# Effect size calculation
effects = pm.effect_sizes(data, measure='SMD')

# Visualization
pm.forest_plot(result)
```

### Internal APIs

Internal modules provide lower-level access for advanced users and extensions.

## Testing Strategy

### Unit Tests

- Individual function testing
- Mock external dependencies
- Edge case coverage

### Integration Tests

- End-to-end workflows
- CLI testing
- GUI testing

### Property-Based Testing

- Hypothesis-based testing
- Statistical property verification
- Invariant checking

### Performance Tests

- Benchmarking critical functions
- Memory usage monitoring
- Scalability testing

## Documentation

### User Documentation

- Tutorial-style guides
- API reference
- Examples and notebooks

### Developer Documentation

- Architecture documentation
- Contribution guidelines
- Testing procedures

## Dependencies

### Core Dependencies

- NumPy: Numerical computing
- SciPy: Statistical functions
- Pandas: Data manipulation
- Matplotlib: Basic plotting

### Optional Dependencies

- Seaborn: Enhanced plotting
- Plotly: Interactive plots
- Streamlit: GUI framework
- Requests: HTTP client for living reviews

## Future Considerations

### Scalability

- Support for very large meta-analyses
- Distributed computing options
- Cloud integration

### Interoperability

- Integration with R meta-analysis packages
- Support for additional data formats
- API endpoints for web integration

## See Also

- [Contributing](contributing.md) - How to contribute
- [Testing](testing.md) - Testing procedures