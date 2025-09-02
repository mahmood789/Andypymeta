# Andypymeta

A Python metadata library with comprehensive testing and debugging capabilities.

## Features

- Robust metadata handling
- Comprehensive test suite with pytest
- Code coverage reporting
- Type checking with mypy
- Code formatting with black and isort
- Linting with flake8
- Debugging utilities and logging

## Installation

```bash
pip install andypymeta
```

For development:

```bash
pip install -e ".[dev]"
```

## Development

### Setting up development environment

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=andypymeta

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

### Code quality

```bash
# Format code
black .
isort .

# Lint code
flake8

# Type check
mypy andypymeta
```

### Debugging

The library includes comprehensive debugging utilities:

- Structured logging with configurable levels
- Debug decorators for function tracing
- Performance profiling tools
- Memory usage monitoring

See the documentation for detailed debugging guides.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.