"""Test utility functions."""

import json
import tempfile
from pathlib import Path

import pytest

from andypymeta.utils import (
    flatten_dict,
    get_nested_value,
    get_version,
    load_json_file,
    merge_dicts,
    sanitize_key,
    save_json_file,
    set_nested_value,
    unflatten_dict,
    validate_metadata,
)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_version(self):
        """Test getting version."""
        version = get_version()
        assert isinstance(version, str)
        assert version == "0.1.0"

    def test_validate_metadata_valid(self):
        """Test validating valid metadata."""
        metadata = {"name": "test", "version": "1.0.0", "count": 42}

        errors = validate_metadata(metadata)
        assert errors == []

    def test_validate_metadata_not_dict(self):
        """Test validating non-dict metadata."""
        errors = validate_metadata("not a dict")
        assert len(errors) == 1
        assert "must be a dictionary" in errors[0]

    def test_validate_metadata_missing_required(self):
        """Test validating metadata with missing required keys."""
        metadata = {"name": "test"}
        required_keys = ["name", "version", "author"]

        errors = validate_metadata(metadata, required_keys=required_keys)
        assert len(errors) == 2
        assert any("'version' is missing" in error for error in errors)
        assert any("'author' is missing" in error for error in errors)

    def test_validate_metadata_none_values(self):
        """Test validating metadata with None required values."""
        metadata = {"name": "test", "version": None}
        required_keys = ["name", "version"]

        errors = validate_metadata(metadata, required_keys=required_keys)
        assert len(errors) == 1
        assert "'version' cannot be None" in errors[0]

    def test_validate_metadata_type_constraints(self):
        """Test validating metadata with type constraints."""
        metadata = {
            "name": 123,  # Should be str
            "count": "not_number",  # Should be int
            "valid": "correct",  # Correct type
        }

        type_constraints = {"name": str, "count": int, "valid": str}

        errors = validate_metadata(metadata, type_constraints=type_constraints)
        assert len(errors) == 2
        assert any("should be str, got int" in error for error in errors)
        assert any("should be int, got str" in error for error in errors)

    def test_sanitize_key_normal(self):
        """Test sanitizing normal keys."""
        assert sanitize_key("normal_key") == "normal_key"
        assert sanitize_key("key-with-dash") == "key-with-dash"
        assert sanitize_key("key.with.dots") == "key.with.dots"

    def test_sanitize_key_invalid_chars(self):
        """Test sanitizing keys with invalid characters."""
        assert sanitize_key("key with spaces") == "key_with_spaces"
        assert sanitize_key("key@with#symbols") == "key_with_symbols"
        assert sanitize_key("key/with\\slashes") == "key_with_slashes"

    def test_sanitize_key_starts_with_number(self):
        """Test sanitizing keys that start with numbers."""
        assert sanitize_key("123key") == "_123key"
        assert sanitize_key("9test") == "_9test"

    def test_sanitize_key_empty(self):
        """Test sanitizing empty key."""
        assert sanitize_key("") == "unnamed_key"
        assert sanitize_key("!!!") == "unnamed_key"

    def test_flatten_dict_simple(self):
        """Test flattening simple nested dict."""
        data = {"a": 1, "b": {"c": 2, "d": 3}}

        expected = {"a": 1, "b.c": 2, "b.d": 3}

        result = flatten_dict(data)
        assert result == expected

    def test_flatten_dict_deep_nesting(self):
        """Test flattening deeply nested dict."""
        data = {"level1": {"level2": {"level3": "deep_value"}}}

        expected = {"level1.level2.level3": "deep_value"}

        result = flatten_dict(data)
        assert result == expected

    def test_flatten_dict_custom_separator(self):
        """Test flattening with custom separator."""
        data = {"a": {"b": "value"}}

        result = flatten_dict(data, separator="_")
        assert result == {"a_b": "value"}

    def test_unflatten_dict(self):
        """Test unflattening dictionary."""
        data = {"a": 1, "b.c": 2, "b.d": 3, "e.f.g": "deep"}

        expected = {"a": 1, "b": {"c": 2, "d": 3}, "e": {"f": {"g": "deep"}}}

        result = unflatten_dict(data)
        assert result == expected

    def test_unflatten_dict_custom_separator(self):
        """Test unflattening with custom separator."""
        data = {"a_b_c": "value"}
        expected = {"a": {"b": {"c": "value"}}}

        result = unflatten_dict(data, separator="_")
        assert result == expected

    def test_load_json_file_success(self):
        """Test loading JSON file successfully."""
        data = {"test": "value", "number": 42}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            result = load_json_file(temp_path)
            assert result == data
        finally:
            temp_path.unlink()

    def test_load_json_file_not_found(self):
        """Test loading non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            load_json_file("/non/existent/file.json")

    def test_load_json_file_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                load_json_file(temp_path)
        finally:
            temp_path.unlink()

    def test_save_json_file(self):
        """Test saving JSON file."""
        data = {"test": "value", "nested": {"key": "value"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_json_file(data, temp_path)

            # Verify file was saved correctly
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)
            assert loaded_data == data

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_save_json_file_creates_directory(self):
        """Test that save_json_file creates parent directories."""
        data = {"test": "value"}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "subdir" / "test.json"

            save_json_file(data, temp_path)

            assert temp_path.exists()
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)
            assert loaded_data == data

    def test_merge_dicts_simple(self):
        """Test merging simple dictionaries."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        dict3 = {"e": 5}

        result = merge_dicts(dict1, dict2, dict3)
        expected = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

        assert result == expected

    def test_merge_dicts_overlapping(self):
        """Test merging dictionaries with overlapping keys."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 20, "c": 3}  # 'b' should be overridden

        result = merge_dicts(dict1, dict2)
        expected = {"a": 1, "b": 20, "c": 3}

        assert result == expected

    def test_merge_dicts_nested(self):
        """Test merging nested dictionaries."""
        dict1 = {"a": 1, "nested": {"x": 10, "y": 20}}
        dict2 = {"b": 2, "nested": {"y": 200, "z": 30}}

        result = merge_dicts(dict1, dict2)
        expected = {
            "a": 1,
            "b": 2,
            "nested": {"x": 10, "y": 200, "z": 30},  # Overridden
        }

        assert result == expected

    def test_get_nested_value_success(self):
        """Test getting nested value successfully."""
        data = {"level1": {"level2": {"value": "found"}}}

        result = get_nested_value(data, "level1.level2.value")
        assert result == "found"

    def test_get_nested_value_missing(self):
        """Test getting missing nested value."""
        data = {"a": {"b": "value"}}

        result = get_nested_value(data, "a.missing.key", default="not_found")
        assert result == "not_found"

    def test_get_nested_value_custom_separator(self):
        """Test getting nested value with custom separator."""
        data = {"a": {"b": "value"}}

        result = get_nested_value(data, "a_b", separator="_")
        assert result == "value"

    def test_set_nested_value_new_path(self):
        """Test setting value in new nested path."""
        data = {}

        set_nested_value(data, "level1.level2.value", "new_value")

        expected = {"level1": {"level2": {"value": "new_value"}}}

        assert data == expected

    def test_set_nested_value_existing_path(self):
        """Test setting value in existing nested path."""
        data = {"level1": {"level2": {"existing": "old_value"}}}

        set_nested_value(data, "level1.level2.new_key", "new_value")

        assert data["level1"]["level2"]["new_key"] == "new_value"
        assert data["level1"]["level2"]["existing"] == "old_value"

    def test_set_nested_value_custom_separator(self):
        """Test setting nested value with custom separator."""
        data = {}

        set_nested_value(data, "a_b_c", "value", separator="_")

        expected = {"a": {"b": {"c": "value"}}}
        assert data == expected
