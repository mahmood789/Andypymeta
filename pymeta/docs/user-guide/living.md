# Living Reviews

This guide covers the living review functionality in PyMeta for automated systematic review updates.

## Overview

Living reviews are systematic reviews that are continually updated as new evidence becomes available. PyMeta provides automation tools to support this process.

## Setup

### Configuration

Configure data sources and search parameters.

```python
import pymeta

# Example: Configure living review
# Placeholder for code examples
```

### Data Sources

PyMeta can monitor multiple sources:

- **PubMed**: Medical literature
- **arXiv**: Preprint server
- **Custom APIs**: User-defined sources

## Automated Monitoring

### Schedulers

Set up automated searches on regular intervals.

```python
import pymeta

# Example: Schedule automated searches
# Placeholder for code examples
```

### Search Strategies

Define and refine search strategies for each data source.

```python
import pymeta

# Example: Define search strategy
# Placeholder for code examples
```

## Data Processing

### Screening

Automated and semi-automated screening of new studies.

```python
import pymeta

# Example: Screen new studies
# Placeholder for code examples
```

### Data Extraction

Extract relevant data from identified studies.

```python
import pymeta

# Example: Extract study data
# Placeholder for code examples
```

## Analysis Updates

### Incremental Analysis

Update meta-analysis results as new studies are added.

```python
import pymeta

# Example: Incremental analysis
# Placeholder for code examples
```

### Trial Sequential Analysis

Monitor for sufficient evidence using TSA methods.

## Notifications

### Alert Systems

Configure notifications for significant changes:

- **Slack**: Team notifications
- **Email**: Automated reports
- **Custom webhooks**: Integration with other systems

```python
import pymeta

# Example: Configure notifications
# Placeholder for code examples
```

## Quality Control

### Change Detection

Monitor for significant changes in results.

### Version Control

Track changes and maintain analysis history.

## Dashboard

Optional web-based dashboard for monitoring living review status.

## See Also

- [CLI](cli.md) - Command-line interface
- [Models](models.md) - Meta-analytic models
- [Plots](plots.md) - Visualization tools