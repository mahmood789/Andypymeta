# Command-Line Interface

This guide covers the PyMeta command-line interface for conducting meta-analyses from the terminal.

## Overview

The PyMeta CLI provides access to core functionality without requiring Python programming knowledge.

## Installation

The CLI is included with PyMeta installation:

```bash
pip install pymeta
```

## Basic Usage

```bash
pymeta --help
```

## Commands

### Meta-Analysis

Perform basic meta-analysis:

```bash
# Fixed-effect meta-analysis
pymeta meta --data studies.csv --model fixed --output results/

# Random-effects meta-analysis
pymeta meta --data studies.csv --model random --tau2-method dl --output results/

# With subgroup analysis
pymeta meta --data studies.csv --model random --subgroup intervention_type
```

### Plotting

Generate plots:

```bash
# Forest plot
pymeta plots forest --data studies.csv --output forest.png

# Funnel plot
pymeta plots funnel --data studies.csv --output funnel.png --test egger

# Custom plot with options
pymeta plots forest --data studies.csv --output forest.pdf --theme publication --size large
```

### Bias Assessment

Assess publication bias:

```bash
# Multiple bias tests
pymeta bias --data studies.csv --tests egger,begg,pet-peese --output bias_report.html

# Trim-and-fill analysis
pymeta bias trim-fill --data studies.csv --output trimfill_results.csv
```

### Living Reviews

Manage living reviews:

```bash
# Initialize living review
pymeta live init --config living_config.yaml

# Update with new data
pymeta live update --review-id my_review

# Generate report
pymeta live report --review-id my_review --output live_report.html
```

## Data Format

### Input Files

PyMeta accepts CSV files with standard column names:

```csv
study_id,effect_size,variance,sample_size,year,intervention
Study1,0.5,0.1,100,2020,Treatment A
Study2,0.3,0.15,80,2021,Treatment B
```

### Required Columns

- Effect size measure (effect_size, or, rr, etc.)
- Variance or standard error
- Study identifier

### Optional Columns

- Sample sizes
- Moderator variables
- Study characteristics

## Configuration

### Config Files

Use YAML configuration files for complex analyses:

```yaml
# meta_config.yaml
data:
  file: studies.csv
  effect_column: effect_size
  variance_column: variance

model:
  type: random
  tau2_method: reml

output:
  directory: results/
  format: [html, csv, png]

plots:
  forest:
    theme: publication
    size: large
  funnel:
    test: egger
```

Run with config:

```bash
pymeta meta --config meta_config.yaml
```

## Examples

### Quick Analysis

```bash
# Complete analysis pipeline
pymeta meta --data my_studies.csv --model random --plots all --bias all --output complete_analysis/
```

### Batch Processing

```bash
# Process multiple datasets
for file in data/*.csv; do
    pymeta meta --data "$file" --model random --output "results/$(basename "$file" .csv)/"
done
```

## See Also

- [Models](models.md) - Meta-analytic models
- [Plots](plots.md) - Visualization options
- [Living](living.md) - Living review functionality