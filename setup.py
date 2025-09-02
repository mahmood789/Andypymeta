"""Setup configuration for PyMeta package."""

from setuptools import setup, find_packages

# Read long description from README if available
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A comprehensive modular meta-analysis package for Python."

# Core dependencies
REQUIRED = [
    "numpy>=1.19.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "matplotlib>=3.3.0",
]

# Optional dependencies
EXTRAS = {
    "statsmodels": ["statsmodels>=0.12.0"],
    "streamlit": ["streamlit>=1.0.0"],
    "cli": ["click>=8.0.0"],
    "scheduler": ["APScheduler>=3.8.0"],
    "all": [
        "statsmodels>=0.12.0",
        "streamlit>=1.0.0", 
        "click>=8.0.0",
        "APScheduler>=3.8.0"
    ]
}

setup(
    name="pymeta",
    version="4.1-modular",
    author="PyMeta Development Team",
    author_email="pymeta@example.com",
    description="A comprehensive modular meta-analysis package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mahmood789/Andypymeta",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    entry_points={
        "console_scripts": [
            "pymeta=pymeta.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)