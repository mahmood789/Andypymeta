# Contributing to PyMeta

Thank you for your interest in contributing to PyMeta! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

### Setting Up Your Environment

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/yourusername/Andypymeta.git
cd Andypymeta/pymeta
```

3. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install development dependencies:

```bash
pip install -e ".[dev]"
```

5. Install pre-commit hooks:

```bash
pre-commit install
```

## Development Workflow

### Creating a Branch

```bash
git checkout -b feature/your-feature-name
```

### Making Changes

1. Write your code
2. Add tests for new functionality
3. Update documentation as needed
4. Run tests and linting

### Testing Your Changes

```bash
# Run all tests
make test

# Run linting
make lint

# Run type checking
make type

# Run all checks
make all
```

### Committing Changes

```bash
git add .
git commit -m "Add feature: description of your changes"
```

Pre-commit hooks will automatically run linting and formatting.

### Submitting a Pull Request

1. Push your branch to GitHub:

```bash
git push origin feature/your-feature-name
```

2. Create a pull request on GitHub
3. Fill out the pull request template
4. Wait for review and feedback

## Contribution Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write descriptive docstrings
- Keep functions focused and small

### Testing

- Write tests for all new functionality
- Aim for high test coverage
- Include both unit and integration tests
- Use descriptive test names

### Documentation

- Update documentation for user-facing changes
- Include docstring examples
- Add type information to function signatures

### Commit Messages

Use clear, descriptive commit messages:

```
Add feature: Brief description

Longer description if needed, explaining what the change does
and why it was made.

Fixes #123
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

- Python version
- PyMeta version
- Operating system
- Minimal reproducing example
- Full error traceback

### Feature Requests

For feature requests, please include:

- Use case description
- Proposed API design
- Examples of usage
- Alternative approaches considered

### Code Contributions

We welcome contributions in these areas:

- **New effect size measures**
- **Additional tau-squared estimators**
- **Visualization improvements**
- **Performance optimizations**
- **Documentation improvements**
- **Bug fixes**

### Documentation Contributions

Documentation improvements are always welcome:

- Tutorial examples
- API documentation
- User guides
- Developer documentation

## Code Review Process

### What to Expect

- Initial review within 1-2 weeks
- Constructive feedback and suggestions
- Possible requests for changes
- Final approval before merging

### Review Criteria

- Code quality and style
- Test coverage
- Documentation completeness
- Performance considerations
- API design consistency

## Getting Help

### Questions

- Open a GitHub issue for questions
- Join discussions in existing issues
- Check the documentation first

### Mentorship

New contributors are welcome! Don't hesitate to ask for help or guidance.

## Recognition

Contributors will be:

- Added to the contributors list
- Acknowledged in release notes
- Thanked publicly on social media

## Code of Conduct

Please be respectful and inclusive in all interactions. We follow the Python Community Code of Conduct.

## Development Tips

### IDE Setup

Recommended VS Code extensions:

- Python
- Pylance
- Black Formatter
- GitLens

### Debugging

- Use pytest for debugging tests
- Set breakpoints in your IDE
- Use print statements sparingly

### Performance

- Profile code with cProfile
- Use line_profiler for detailed analysis
- Consider NumPy vectorization

## Release Process

Releases are managed by maintainers:

1. Version bump
2. Changelog update
3. Tag creation
4. PyPI upload
5. GitHub release

## See Also

- [Architecture](architecture.md) - Package architecture
- [Testing](testing.md) - Testing guidelines