from .parser import parse, parse_file, parse_file_recursive

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = ["parse_file", "parse_file_recursive", "parse"]
