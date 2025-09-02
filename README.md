# PyMeta - Comprehensive Meta-Analysis Package

A Python package for meta-analysis with advanced features including HKSJ variance adjustment, influence diagnostics, and living meta-analysis capabilities.

## Features

- **HKSJ Variance Adjustment**: Hartung-Knapp-Sidik-Jonkman method with t-distribution
- **Influence Diagnostics**: Leave-one-out analysis and influence measures
- **Enhanced Visualizations**: Contour-enhanced funnel plots
- **Living Meta-Analysis**: Automated periodic updates with scheduling
- **CLI Interface**: Comprehensive command-line tools
- **GUI Interface**: Streamlit-based interactive interface

## Installation

```bash
pip install pymeta
```

For scheduler support:
```bash
pip install pymeta[scheduler]
```

## Quick Start

```python
import pymeta as pm

# Load data and run meta-analysis
config = pm.MetaAnalysisConfig(use_hksj=True)
results = pm.analyze_csv("data.csv", config=config)

# Generate plots
pm.plot_funnel_contour(results)
pm.plot_forest(results)
```

## CLI Usage

```bash
# Basic analysis with HKSJ
pymeta analyze --csv data.csv --model RE --tau2 REML --hksj

# Generate contour funnel plot
pymeta plot --csv data.csv --which funnel-contour

# Leave-one-out analysis
pymeta loo --csv data.csv --tau2 PM --no-hksj

# Start living meta-analysis
pymeta live --seconds 1800
```

## License

Apache 2.0 License. See LICENSE file for details.