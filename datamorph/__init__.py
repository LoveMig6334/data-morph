"""datamorph — distill file-format conversion into a small local model.

Public API:

    from datamorph import convert_file, ConversionResult
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from datamorph.convert import ConversionResult, convert_file

try:
    __version__ = version("data-morph-gemma")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["convert_file", "ConversionResult", "__version__"]
