# PyMeta: Comprehensive Modular Meta-Analysis Package

**Version 4.1-modular**

PyMeta is a comprehensive, modular meta-analysis package for Python that provides a complete toolkit for conducting publication-quality meta-analyses.

## 🚀 Features

### Core Functionality
- **Multiple Models**: Fixed Effects, Random Effects, GLMM Binomial
- **Tau² Estimators**: DerSimonian-Laird, Paule-Mandel, REML, Maximum Likelihood
- **Effect Size Calculations**: Binary outcomes (OR, RR, RD), continuous outcomes
- **Registry System**: Extensible plugin architecture for models, estimators, and plots

### Advanced Analysis
- **Publication Bias Detection**: Egger regression, Begg rank correlation, Trim-and-Fill
- **Trial Sequential Analysis (TSA)**: O'Brien-Fleming and Pocock boundaries
- **Influence Analysis**: Leave-one-out analysis, Baujat plots
- **GOSH Analysis**: Graphical display of study heterogeneity

### Visualization Suite
- **Forest Plots**: Publication-quality with multiple styles
- **Funnel Plots**: Publication bias assessment
- **Baujat Plots**: Heterogeneity and influence detection  
- **Radial Plots**: Galbraith plots for outlier detection
- **GOSH Plots**: Fast subset sampling visualization

### Interfaces
- **Python API**: Comprehensive programmatic interface
- **Command Line**: Full-featured CLI with plot generation
- **Streamlit GUI**: Interactive web application

## 📦 Installation

### From Source
```bash
git clone https://github.com/mahmood789/Andypymeta.git
cd Andypymeta
pip install -e .
```

### Dependencies
**Core:**
- numpy >= 1.19.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.3.0

**Optional:**
- statsmodels >= 0.12.0 (for GLMM models)
- streamlit >= 1.0.0 (for GUI)
- click >= 8.0.0 (for CLI)

## 🚀 Quick Start

### Python API
```python
import pymeta

# Create meta-analysis with example data
meta = pymeta.PyMeta()

# Perform random effects analysis
results = meta.analyze(model_type="random_effects", tau2_estimator="REML")
print(f"Pooled effect: {results.pooled_effect:.4f}")
print(f"I-squared: {results.i_squared:.1f}%")

# Test for publication bias
bias_result = meta.test_bias("egger")
print(f"Egger test p-value: {bias_result.p_value:.4f}")

# Generate plots
meta.plot_forest()
meta.plot_funnel()
meta.plot_baujat()
```

### With Your Own Data
```python
import pandas as pd
import pymeta

# Load data from CSV
df = pd.read_csv("your_data.csv")
points = pymeta.io.create_meta_points_from_dataframe(
    df, effect_col="effect", variance_col="variance"
)

# Analyze
meta = pymeta.PyMeta(points)
results = meta.analyze()

# Get comprehensive report
print(meta.summary_report())
```

### Command Line Interface
```bash
# Analyze data
pymeta analyze -i data.csv -m random_effects --tau2 REML --plots

# Generate specific plots
pymeta plot -i data.csv -t forest -o forest_plot.png

# Simulate test data
pymeta simulate --n-studies 15 --output simulated.csv

# Get help
pymeta --help
```

### Streamlit GUI
```bash
# Launch web interface
streamlit run -m pymeta.gui_app
```

## 📊 Data Format

Your CSV file should contain:
- **Effect sizes**: Study-level effect estimates
- **Variances**: or standard errors
- **Study labels**: (optional) Study identifiers

Example:
```csv
study,effect,variance,sample_size
Study 1,0.25,0.045,120
Study 2,0.31,0.038,140
Study 3,0.18,0.052,98
```

## 🔧 Advanced Usage

### Model Comparison
```python
# Compare different models
comparison = meta.model_comparison([
    "fixed_effects", 
    "random_effects", 
    "glmm_binomial"
])
```

### Trial Sequential Analysis
```python
# Perform TSA
tsa_result = meta.perform_tsa(
    delta=0.2,  # Clinically relevant difference
    alpha=0.05,
    beta=0.20
)

# Plot TSA
meta.plot_tsa(tsa_result)
```

### Sensitivity Analysis
```python
# Test across estimators
re_model = pymeta.RandomEffects(points)
sensitivity = re_model.sensitivity_analysis(['DL', 'PM', 'REML'])
```

### Binary Data (2x2 Tables)
```python
import numpy as np

# Define 2x2 contingency tables
tables = [
    np.array([[10, 5], [20, 15]]),   # [events_treat, events_control]
    np.array([[8, 12], [18, 22]]),   # [non_events_treat, non_events_control]
]

# GLMM analysis with fallback
model = pymeta.GLMMBinomial(tables_2x2=tables)
results = model.fit()

# Get odds ratios
or_results = model.odds_ratio_results()
print(f"Odds Ratio: {or_results['odds_ratio']:.3f}")
```

## 📈 Available Models

### Fixed Effects
- Assumes all studies estimate the same true effect
- Uses inverse variance weighting
- Best when heterogeneity is low

### Random Effects  
- Accounts for between-study heterogeneity
- Multiple tau² estimators available
- Generally preferred when I² > 50%

### GLMM Binomial
- Generalized linear mixed models for binary data
- Uses statsmodels when available
- Graceful fallback to log-OR + Random Effects

## 📊 Available Plots

### Forest Plot
Publication-quality forest plots with confidence intervals and weights.

### Funnel Plot
Detect publication bias through plot asymmetry.

### Baujat Plot
Identify studies contributing to heterogeneity and influence.

### Radial Plot (Galbraith)
Precision-weighted plot for outlier detection.

### GOSH Plot
Graphical display of study heterogeneity across all possible subsets.

## 🔍 Bias Detection

### Egger Regression Test
Tests for small-study effects using regression of standardized effect sizes.

### Begg Rank Correlation
Kendall's tau correlation between effect sizes and variances.

### Trim-and-Fill
Estimates and adjusts for missing studies (placeholder implementation).

## 🎯 Trial Sequential Analysis

- **Monitoring Boundaries**: O'Brien-Fleming, Pocock
- **Futility Analysis**: Early stopping for lack of effect
- **Information Size**: Required sample size calculations
- **Cumulative Analysis**: Sequential evidence accumulation

## 🎨 Plot Styles

### Default
Standard scientific plots with grid and clear typography.

### Publication
High-quality plots optimized for journal submission.

### Presentation
Large fonts and high contrast for conference presentations.

## 📚 Documentation

### API Reference
- Complete function and class documentation
- Parameter descriptions and examples
- Error handling information

### Examples
- Jupyter notebooks with worked examples
- Real-world case studies
- Best practices guide

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Test coverage includes:
- Effect size calculations
- All tau² estimators  
- Model implementations
- Plot generation
- Error handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

### Adding New Estimators
```python
from pymeta.registries import register_estimator

@register_estimator("my_estimator")
def my_tau2_estimator(points):
    # Implementation here
    return tau2_estimate
```

### Adding New Models
```python
from pymeta.registries import register_model
from pymeta.models.base import MetaModel

@register_model("my_model")
class MyModel(MetaModel):
    def fit(self):
        # Implementation here
        return results
```

## 📄 License

Apache License 2.0 - see LICENSE file for details.

## 🙏 Acknowledgments

- DerSimonian & Laird (1986) for the foundational random effects method
- Paule & Mandel (1982) for the iterative tau² estimator
- Egger et al. (1997) for publication bias detection
- Baujat et al. (2002) for heterogeneity visualization
- Olkin et al. (2012) for GOSH methodology

## 📞 Support

- **Issues**: GitHub Issues tracker
- **Documentation**: Built-in help and docstrings
- **Examples**: See `examples/` directory

---

**PyMeta** - *Comprehensive meta-analysis made simple* 📊