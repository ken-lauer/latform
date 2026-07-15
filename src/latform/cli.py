"""
Shared command-line plumbing for the ``latform`` entry points.
"""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .config import ConfigError, LatformProjectConfig, discover_config

logger = logging.getLogger("latform")

CASE_CHOICES = ("upper", "lower", "same")
LOG_CHOICES = ("DEBUG", "INFO", "WARNING", "CRITICAL")


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``filename`` positional plus ``-r``/``--combine``/``-e``."""
    parser.add_argument(
        "filename",
        nargs="*",
        help=(
            "Filename(s) to process (use '-' for stdin/standard input). "
            "If omitted, the config's top-level lattices are used."
        ),
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively parse lattice files, following call statements.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Process all input files together as a single set, sharing one parse stack.",
    )
    parser.add_argument(
        "-e",
        "--error-if-missing",
        action="store_true",
        help="If a file is missing during parsing, exit with an error.",
    )


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--config`` / ``--no-config``."""
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to a latform config file (latform.toml or pyproject.toml).",
    )
    parser.add_argument(
        "--no-config",
        dest="use_config",
        action="store_false",
        default=True,
        help="Ignore any latform.toml / pyproject.toml configuration.",
    )


def add_lint_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--strict-references`` / ``--ignore``."""
    parser.add_argument(
        "--strict-references",
        dest="assume_defined",
        action="store_false",
        default=True,
        help=(
            "Only recognize element/constant references defined in the loaded files; "
            "report anything else (and unknown element types) as lint warnings."
        ),
    )
    parser.add_argument(
        "--ignore",
        dest="ignore_lints",
        action="append",
        metavar="CODE",
        help=(
            "Lint code(s) to suppress, e.g. --ignore LF002 (repeatable, or "
            "comma-separated: --ignore LF002,LF003)."
        ),
    )


def add_logging_arguments(parser: argparse.ArgumentParser, *, default_level: str = "INFO") -> None:
    """Add ``--log`` and ``--version``."""
    from ._version import __version__ as package_version

    parser.add_argument(
        "--log",
        "-L",
        dest="log_level",
        default=default_level,
        choices=LOG_CHOICES,
        help="Python logging level (e.g. DEBUG, INFO, WARNING).",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=package_version,
        help="Show the latform version number and exit.",
    )


def configure_logging(level: str) -> None:
    logging.getLogger("latform").setLevel(level)
    logging.basicConfig()


def resolve_ignore_codes(ignore_lints: Sequence[str] | None) -> list[str]:
    """Flatten repeated / comma-separated ``--ignore`` values into codes."""
    return [
        code.strip() for entry in (ignore_lints or []) for code in entry.split(",") if code.strip()
    ]


def resolve_config(parsed: argparse.Namespace) -> LatformProjectConfig:
    """
    Discover the config indicated by ``--config`` / ``--no-config``.

    Exits (status 2) with a logged message on a configuration error.
    """
    try:
        return discover_config(
            explicit=getattr(parsed, "config", None),
            enabled=getattr(parsed, "use_config", True),
        )
    except ConfigError as ex:
        logger.error("%s", ex)
        raise SystemExit(2) from None


def resolve_input_files(
    filenames: Sequence[str],
    config: LatformProjectConfig,
) -> tuple[list[str], bool]:
    """
    Resolve the files to operate on, falling back to the config top-level list.

    Returns ``(filenames, from_top_level)``; ``from_top_level`` is True when the
    config's ``top-level`` lattices supplied the files (implying recursion).
    """
    if filenames:
        return list(filenames), False
    if config.top_level:
        return [str(path) for path in config.resolve_top_level()], True
    return [], False


def require_input_files(
    filenames: Sequence[str],
    config: LatformProjectConfig,
) -> tuple[list[str], bool]:
    """Like :func:`resolve_input_files` but exits (status 2) when none are found."""
    resolved, from_top_level = resolve_input_files(filenames, config)
    if not resolved:
        logger.error("No input files given and no top-level lattices configured.")
        raise SystemExit(2)
    return resolved, from_top_level
