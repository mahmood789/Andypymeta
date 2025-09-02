# Heterogeneity Assessment

This guide covers methods for assessing and exploring heterogeneity in meta-analyses.

## Overview

Heterogeneity refers to variability between studies beyond what would be expected by chance alone.

## Statistical Measures

### I-squared (I²)

Proportion of total variation due to heterogeneity rather than chance.

```python
import pymeta

# Example: Calculate I-squared
# Placeholder for code examples
```

### Tau-squared (τ²)

Estimate of between-study variance.

### Q-statistic

Test for heterogeneity (Cochran's Q).

```python
import pymeta

# Example: Q-test for heterogeneity
# Placeholder for code examples
```

## Sources of Heterogeneity

### Subgroup Analysis

Explore heterogeneity by study characteristics.

```python
import pymeta

# Example: Subgroup analysis
# Placeholder for code examples
```

### Meta-Regression

Model effect sizes as a function of study-level covariates.

```python
import pymeta

# Example: Meta-regression
# Placeholder for code examples
```

## Sensitivity Analysis

### Leave-One-Out Analysis

Assess influence of individual studies.

```python
import pymeta

# Example: Leave-one-out analysis
# Placeholder for code examples
```

### Cumulative Meta-Analysis

Examine how results change with addition of studies over time.

## Diagnostic Plots

- **Baujat plots**: Identify influential studies
- **GOSH plots**: Graphical display of heterogeneity
- **Radial plots**: Visualize study precision vs. effect

## Interpretation Guidelines

Guidelines for interpreting heterogeneity measures and deciding on analysis approaches.

## See Also

- [Models](models.md) - Meta-analytic models
- [Plots](plots.md) - Visualization tools
- [Bias](bias.md) - Publication bias assessment