# Installation

## Requirements

- Python 3.9 or higher
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0

## Install from PyPI

```bash
pip install pymeta
```

## Install from Source

```bash
git clone https://github.com/mahmood789/Andypymeta.git
cd Andypymeta/pymeta
pip install -e .
```

## Development Installation

For development, install with additional dependencies:

```bash
git clone https://github.com/mahmood789/Andypymeta.git
cd Andypymeta/pymeta
pip install -e ".[dev]"
```

This will install all development dependencies including:

- pytest (testing)
- ruff (linting)
- mypy (type checking)
- pre-commit (code quality)
- nox (automation)

## Verify Installation

```python
import pymeta
print(pymeta.__version__)
```

## Optional Dependencies

For additional functionality, you may want to install:

```bash
# For GUI features
pip install streamlit

# For advanced visualizations
pip install plotly

# For living reviews
pip install schedule requests
```