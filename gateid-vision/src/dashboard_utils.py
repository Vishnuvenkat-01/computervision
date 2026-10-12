"""
Helper utilities for the GateID Vision Streamlit Dashboard.

Includes file content hashing for upload deduplication, environment checks,
and review state tracking.
"""
from typing import Any, Dict, Optional
import hashlib

def get_file_content_hash(file_bytes: bytes) -> str:
    """Generates an MD5 hash of uploaded file content to identify distinct videos."""
    hasher = hashlib.md5()
    hasher.update(file_bytes)
    return hasher.hexdigest()

def check_ultralytics_available() -> bool:
    """Checks if Ultralytics library is importable in the current environment."""
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False
