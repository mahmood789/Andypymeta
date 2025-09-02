# noxfile.py - optional: nox sessions for test/lint/type/docs
import nox

@nox.session(python=["3.9", "3.10", "3.11", "3.12"])
def tests(session):
    """Run tests."""
    session.install(".")
    session.install("pytest", "pytest-cov", "hypothesis")
    session.run("pytest", *session.posargs)

@nox.session
def lint(session):
    """Run linting."""
    session.install("ruff")
    session.run("ruff", "check", ".")

@nox.session
def type_check(session):
    """Run type checking."""
    session.install(".")
    session.install("mypy")
    session.run("mypy", "pymeta", "cli")

@nox.session
def docs(session):
    """Build documentation."""
    session.install(".")
    session.install("mkdocs", "mkdocs-material")
    session.run("mkdocs", "build")

@nox.session
def coverage(session):
    """Run tests with coverage."""
    session.install(".")
    session.install("pytest", "pytest-cov")
    session.run("pytest", "--cov=pymeta", "--cov=cli", "--cov-report=html")