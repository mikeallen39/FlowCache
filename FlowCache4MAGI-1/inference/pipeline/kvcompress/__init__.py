"""
This package provides efficient decoding-time KV cache compression methods.
"""

__version__ = "0.1.0"

from .monkeypatch import replace_magi

__all__ = ["replace_magi"]
