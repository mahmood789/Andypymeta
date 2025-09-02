# Meta-Analytic Models

This guide covers the meta-analytic models available in PyMeta.

## Fixed-Effect Models

Fixed-effect models assume that all studies share a common true effect size.

```python
import pymeta

# Example: Fixed-effect meta-analysis
# Placeholder for code examples
```

## Random-Effects Models

Random-effects models allow for between-study heterogeneity.

### Tau-squared Estimators

PyMeta supports multiple estimators for between-study variance (tau-squared):

- **DerSimonian-Laird (DL)**: Classic method
- **Paule-Mandel (PM)**: Iterative method
- **Maximum Likelihood (ML)**: ML estimation
- **Restricted Maximum Likelihood (REML)**: REML estimation
- **Hunter-Schmidt (HS)**: Psychometric tradition
- **Empirical Bayes (EB)**: Bayesian approach
- **Sidik-Jonkman (SJ)**: Alternative approach

```python
import pymeta

# Example: Random-effects with different tau-squared estimators
# Placeholder for code examples
```

## Multivariate Models

For studies reporting multiple effect sizes or multivariate outcomes.

```python
import pymeta

# Example: Multivariate meta-analysis
# Placeholder for code examples
```

## Generalized Linear Mixed Models (GLMM)

For count and binary outcomes using exact likelihood methods.

### Binomial Models

- Logit link
- Complementary log-log link

### Poisson Models

- Log link for count data

```python
import pymeta

# Example: GLMM meta-analysis
# Placeholder for code examples
```

## Model Selection

Guidelines for choosing appropriate models based on data characteristics and research questions.

## See Also

- [Effects](effects.md) - Effect size calculations
- [Heterogeneity](heterogeneity.md) - Assessing heterogeneity