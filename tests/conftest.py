"""Test configuration and fixtures."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from andypymeta import DebugManager, MetadataHandler


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Sample metadata for testing."""
    return {
        "name": "test_project",
        "version": "1.0.0",
        "author": "Test Author",
        "config": {"debug": True, "max_retries": 3, "nested": {"deep_value": "deep"}},
        "tags": ["python", "testing"],
        "metadata": {"created": "2023-01-01", "modified": "2023-12-01"},
    }


@pytest.fixture
def temp_json_file(sample_metadata):
    """Temporary JSON file with sample metadata."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_metadata, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def metadata_handler():
    """MetadataHandler instance for testing."""
    return MetadataHandler(debug=True)


@pytest.fixture
def debug_manager():
    """DebugManager instance for testing."""
    return DebugManager(enabled=True)
