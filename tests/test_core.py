"""Test the core MetadataHandler functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from andypymeta.core import MetadataHandler


class TestMetadataHandler:
    """Test MetadataHandler class."""

    def test_init_default(self):
        """Test default initialization."""
        handler = MetadataHandler()
        assert not handler.debug
        assert handler._metadata == {}

    def test_init_debug_enabled(self):
        """Test initialization with debug enabled."""
        handler = MetadataHandler(debug=True)
        assert handler.debug
        assert handler._metadata == {}

    def test_load_metadata_from_dict(self, metadata_handler, sample_metadata):
        """Test loading metadata from dictionary."""
        result = metadata_handler.load_metadata(sample_metadata)
        assert result == sample_metadata
        assert metadata_handler._metadata == sample_metadata

    def test_load_metadata_from_file(
        self, metadata_handler, temp_json_file, sample_metadata
    ):
        """Test loading metadata from file."""
        result = metadata_handler.load_metadata(temp_json_file)
        assert result == sample_metadata

    def test_load_metadata_from_json_string(self, metadata_handler):
        """Test loading metadata from JSON string."""
        json_str = '{"key": "value", "number": 42}'
        expected = {"key": "value", "number": 42}

        result = metadata_handler.load_metadata(json_str)
        assert result == expected

    def test_load_metadata_file_not_found(self, metadata_handler):
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            metadata_handler.load_metadata("/non/existent/file.json")

    def test_load_metadata_invalid_type(self, metadata_handler):
        """Test loading from invalid type raises ValueError."""
        with pytest.raises(ValueError):
            metadata_handler.load_metadata(123)

    def test_save_metadata(self, metadata_handler, sample_metadata):
        """Test saving metadata to file."""
        metadata_handler.load_metadata(sample_metadata)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            metadata_handler.save_metadata(temp_path)

            # Verify file was created and contains correct data
            assert temp_path.exists()
            with open(temp_path, "r") as f:
                saved_data = json.load(f)
            assert saved_data == sample_metadata

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_get_simple_key(self, metadata_handler, sample_metadata):
        """Test getting value by simple key."""
        metadata_handler.load_metadata(sample_metadata)

        assert metadata_handler.get("name") == "test_project"
        assert metadata_handler.get("version") == "1.0.0"

    def test_get_nested_key(self, metadata_handler, sample_metadata):
        """Test getting value by nested key."""
        metadata_handler.load_metadata(sample_metadata)

        assert metadata_handler.get("config.debug") is True
        assert metadata_handler.get("config.max_retries") == 3
        assert metadata_handler.get("config.nested.deep_value") == "deep"

    def test_get_missing_key_default(self, metadata_handler, sample_metadata):
        """Test getting missing key returns default."""
        metadata_handler.load_metadata(sample_metadata)

        assert metadata_handler.get("missing_key") is None
        assert metadata_handler.get("missing_key", "default") == "default"
        assert metadata_handler.get("config.missing", "default") == "default"

    def test_set_simple_key(self, metadata_handler):
        """Test setting simple key."""
        metadata_handler.set("name", "new_name")
        assert metadata_handler.get("name") == "new_name"

    def test_set_nested_key(self, metadata_handler):
        """Test setting nested key."""
        metadata_handler.set("config.debug", True)
        metadata_handler.set("config.nested.value", "test")

        assert metadata_handler.get("config.debug") is True
        assert metadata_handler.get("config.nested.value") == "test"

    def test_validate_no_schema(self, metadata_handler, sample_metadata):
        """Test validation without schema."""
        metadata_handler.load_metadata(sample_metadata)
        errors = metadata_handler.validate()
        assert errors == []

    def test_validate_invalid_metadata(self, metadata_handler):
        """Test validation with invalid metadata."""
        metadata_handler._metadata = "not a dict"
        errors = metadata_handler.validate()
        assert len(errors) == 1
        assert "must be a dictionary" in errors[0]

    def test_validate_with_schema(self, metadata_handler):
        """Test validation with schema."""
        metadata_handler.load_metadata(
            {"name": "test", "version": "1.0.0", "count": 42}
        )

        schema = {"name": str, "version": str, "count": int, "missing": str}

        errors = metadata_handler.validate(schema)
        assert len(errors) == 1
        assert "Required key 'missing' is missing" in errors[0]

    def test_validate_type_mismatch(self, metadata_handler):
        """Test validation with type mismatch."""
        metadata_handler.load_metadata(
            {"name": 123, "count": "not_a_number"}  # Should be string  # Should be int
        )

        schema = {"name": str, "count": int}

        errors = metadata_handler.validate(schema)
        assert len(errors) == 2
        assert any("should be str, got int" in error for error in errors)
        assert any("should be int, got str" in error for error in errors)

    def test_merge_with_dict(self, metadata_handler, sample_metadata):
        """Test merging with dictionary."""
        metadata_handler.load_metadata(sample_metadata)

        other_data = {
            "new_field": "new_value",
            "name": "overridden_name",  # This should override
        }

        metadata_handler.merge(other_data)

        assert metadata_handler.get("new_field") == "new_value"
        assert metadata_handler.get("name") == "overridden_name"
        assert metadata_handler.get("version") == "1.0.0"  # Should remain

    def test_merge_with_metadata_handler(self, metadata_handler, sample_metadata):
        """Test merging with another MetadataHandler."""
        metadata_handler.load_metadata(sample_metadata)

        other_handler = MetadataHandler()
        other_handler.load_metadata(
            {"new_field": "new_value", "name": "overridden_name"}
        )

        metadata_handler.merge(other_handler)

        assert metadata_handler.get("new_field") == "new_value"
        assert metadata_handler.get("name") == "overridden_name"

    def test_keys(self, metadata_handler, sample_metadata):
        """Test getting all keys."""
        metadata_handler.load_metadata(sample_metadata)
        keys = metadata_handler.keys()

        expected_keys = ["name", "version", "author", "config", "tags", "metadata"]
        assert set(keys) == set(expected_keys)

    def test_to_dict(self, metadata_handler, sample_metadata):
        """Test converting to dictionary."""
        metadata_handler.load_metadata(sample_metadata)
        result = metadata_handler.to_dict()

        assert result == sample_metadata
        assert result is not metadata_handler._metadata  # Should be a copy

    def test_clear(self, metadata_handler, sample_metadata):
        """Test clearing metadata."""
        metadata_handler.load_metadata(sample_metadata)
        assert len(metadata_handler._metadata) > 0

        metadata_handler.clear()
        assert len(metadata_handler._metadata) == 0
        assert metadata_handler.keys() == []
