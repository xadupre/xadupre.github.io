"""Compatibility helpers for onnx-light backend-test metadata."""

from __future__ import annotations

from typing import Any


def metadata_name(value: Any, metadata_type: str) -> Any:
    """Return the public metadata name, preserving legacy collections."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return value
    try:
        from onnx_light.onnx_py._onnxpybackend import backend_test
    except ImportError:
        backend_test = None
    if backend_test is not None:
        converter = getattr(backend_test, f"test_case_{metadata_type}_name", None)
        if converter is not None:
            try:
                return str(converter(value))
            except TypeError:
                pass
    return str(value)


def kind_name(value: Any) -> str:
    """Return the public string representation of a test-case kind."""
    return metadata_name(value, "kind")


def tag_name(value: Any) -> Any:
    """Return the public string representation of a test-case tag."""
    return metadata_name(value, "tag")
