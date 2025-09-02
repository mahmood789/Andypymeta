"""Core metadata handling functionality."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MetadataHandler:
    """Handles metadata operations with debugging and validation."""

    def __init__(self, debug: bool = False) -> None:
        """Initialize the metadata handler.

        Args:
            debug: Enable debug logging
        """
        self.debug = debug
        self._metadata: Dict[str, Any] = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def load_metadata(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load metadata from various sources.

        Args:
            source: File path, dict, or JSON string containing metadata

        Returns:
            Loaded metadata dictionary

        Raises:
            ValueError: If source format is invalid
            FileNotFoundError: If file source doesn't exist
        """
        logger.debug(f"Loading metadata from source: {type(source)}")

        if isinstance(source, dict):
            self._metadata = source.copy()
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
            else:
                # Try parsing as JSON string
                try:
                    self._metadata = json.loads(str(source))
                except json.JSONDecodeError:
                    raise FileNotFoundError(f"File not found: {source}")
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        logger.info(f"Loaded metadata with {len(self._metadata)} keys")
        return self._metadata.copy()

    def save_metadata(self, path: Union[str, Path]) -> None:
        """Save metadata to file.

        Args:
            path: File path to save metadata
        """
        path = Path(path)
        logger.debug(f"Saving metadata to: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved successfully to {path}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key.

        Args:
            key: Metadata key (supports dot notation)
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        keys = key.split(".")
        value = self._metadata

        try:
            for k in keys:
                value = value[k]
            logger.debug(f"Retrieved metadata key '{key}': {type(value)}")
            return value
        except (KeyError, TypeError):
            logger.debug(f"Key '{key}' not found, returning default: {default}")
            return default

    def set(self, key: str, value: Any) -> None:
        """Set metadata value by key.

        Args:
            key: Metadata key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        target = self._metadata

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value
        logger.debug(f"Set metadata key '{key}' to: {type(value)}")

    def validate(self, schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate metadata against schema.

        Args:
            schema: Validation schema (simple key-type mapping)

        Returns:
            List of validation errors
        """
        errors = []

        if schema is None:
            # Basic validation
            if not isinstance(self._metadata, dict):
                errors.append("Metadata must be a dictionary")
        else:
            for key, expected_type in schema.items():
                value = self.get(key)
                if value is None:
                    errors.append(f"Required key '{key}' is missing")
                elif not isinstance(value, expected_type):
                    errors.append(
                        f"Key '{key}' should be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

        logger.debug(f"Validation completed with {len(errors)} errors")
        return errors

    def merge(self, other: Union["MetadataHandler", Dict[str, Any]]) -> None:
        """Merge with another metadata source.

        Args:
            other: Another MetadataHandler instance or dict
        """
        if isinstance(other, MetadataHandler):
            other_data = other._metadata
        else:
            other_data = other

        self._metadata = self._merge_dicts_recursive(self._metadata, other_data)
        logger.info(f"Merged metadata, now has {len(self._metadata)} keys")

    def _merge_dicts_recursive(
        self, dict1: Dict[str, Any], dict2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge two dictionaries.

        Args:
            dict1: First dictionary
            dict2: Second dictionary (takes precedence)

        Returns:
            Merged dictionary
        """
        result = dict1.copy()

        for key, value in dict2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_dicts_recursive(result[key], value)
            else:
                result[key] = value

        return result

    def keys(self) -> List[str]:
        """Get all metadata keys."""
        return list(self._metadata.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Get metadata as dictionary."""
        return self._metadata.copy()

    def clear(self) -> None:
        """Clear all metadata."""
        self._metadata.clear()
        logger.info("Metadata cleared")
