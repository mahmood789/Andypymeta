"""Utility functions for the andypymeta package."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import __version__


def get_version() -> str:
    """Get the current version of andypymeta.

    Returns:
        Version string
    """
    return __version__


def validate_metadata(
    metadata: Dict[str, Any],
    required_keys: Optional[List[str]] = None,
    type_constraints: Optional[Dict[str, type]] = None,
) -> List[str]:
    """Validate metadata dictionary.

    Args:
        metadata: Metadata dictionary to validate
        required_keys: List of required keys
        type_constraints: Dictionary mapping keys to expected types

    Returns:
        List of validation error messages
    """
    errors = []

    if not isinstance(metadata, dict):
        errors.append("Metadata must be a dictionary")
    else:
        # Check required keys
        if required_keys:
            for key in required_keys:
                if key not in metadata:
                    errors.append(f"Required key '{key}' is missing")
                elif metadata[key] is None:
                    errors.append(f"Required key '{key}' cannot be None")

        # Check type constraints
        if type_constraints:
            for key, expected_type in type_constraints.items():
                if key in metadata and metadata[key] is not None:
                    if not isinstance(metadata[key], expected_type):
                        errors.append(
                            f"Key '{key}' should be {expected_type.__name__}, "
                            f"got {type(metadata[key]).__name__}"
                        )

    return errors


def sanitize_key(key: str) -> str:
    """Sanitize a metadata key to be safe for use.

    Args:
        key: Key to sanitize

    Returns:
        Sanitized key string
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", key)

    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"

    # Handle the case where result is only underscores or empty
    if not sanitized or re.match(r"^_+$", sanitized):
        sanitized = "unnamed_key"

    return sanitized


def flatten_dict(
    data: Dict[str, Any], parent_key: str = "", separator: str = "."
) -> Dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        data: Dictionary to flatten
        parent_key: Parent key prefix
        separator: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items: List[tuple[str, Any]] = []

    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))

    return dict(items)


def unflatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """Unflatten a dictionary with dotted keys.

    Args:
        data: Flattened dictionary
        separator: Separator used in keys

    Returns:
        Nested dictionary
    """
    result: Dict[str, Any] = {}

    for key, value in data.items():
        keys = key.split(separator)
        current = result

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    return result


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON data from file with error handling.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in file {path}: {e.msg}", e.doc, e.pos
        )


def save_json_file(
    data: Dict[str, Any],
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Save data to JSON file with error handling.

    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
        ensure_ascii: Whether to escape non-ASCII characters
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries recursively.

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    result: Dict[str, Any] = {}

    for d in dicts:
        for key, value in d.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value

    return result


def get_nested_value(
    data: Dict[str, Any], key_path: str, default: Any = None, separator: str = "."
) -> Any:
    """Get value from nested dictionary using dot notation.

    Args:
        data: Dictionary to search
        key_path: Dot-separated key path
        default: Default value if key not found
        separator: Key separator

    Returns:
        Found value or default
    """
    keys = key_path.split(separator)
    current = data

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_nested_value(
    data: Dict[str, Any], key_path: str, value: Any, separator: str = "."
) -> None:
    """Set value in nested dictionary using dot notation.

    Args:
        data: Dictionary to modify
        key_path: Dot-separated key path
        value: Value to set
        separator: Key separator
    """
    keys = key_path.split(separator)
    current = data

    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value
