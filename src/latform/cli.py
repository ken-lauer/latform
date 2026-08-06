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
        default=None,
        help="Recursively parse lattice files, following call statements.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Do not recursively parse lattice files.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Process all input files together as a single set, sharing one parse stack.",
    )
    parser.add_argument(
        "--format",
        dest="input_format",
        choices=("bmad", "namelist"),
        default=None,
        help=(
            "Force how input file(s) are interpreted: 'bmad' lattice or 'namelist' "
            "(a Tao tao.init). Default: auto-detect from the extension or contents."
        ),
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


NAMELIST_FORMAT_DESTS = (
    "format_namelist",
    "namelist_indent",
    "namelist_field_case",
    "namelist_align_equals",
    "namelist_align_comments",
)

# Config ``[format]`` keys that map directly onto `FormatOptions` fields.
# The flatten toggles change *what* is emitted rather than how it is laid
# out, and the namelist keys live on the nested options (see
# `build_namelist_options`), so neither belongs here.
_FORMAT_OPTION_DESTS = (
    "line_length",
    "max_line_length",
    "compact",
    "name_case",
    "attribute_case",
    "kind_case",
    "builtin_case",
    "controller_variable_case",
    "section_break_character",
    "section_break_width",
    "strip_comments",
)


def config_format_options(config: LatformProjectConfig, base=None):
    """
    Fold the config's ``[format]`` settings into a `FormatOptions`.

    Mirrors the ``latform`` CLI's handling: ``compact`` also flips
    ``newline_before_new_type``, and ``max_line_length`` is derived from
    ``line_length`` when only the latter is configured.

    Parameters
    ----------
    config : LatformProjectConfig
        The resolved project configuration.
    base : FormatOptions, optional
        Options to apply the settings on top of (default:
        `latform.output.default_options`).
    """
    import dataclasses

    from .output import default_options

    base = base if base is not None else default_options
    kwargs = {key: value for key, value in config.format.items() if key in _FORMAT_OPTION_DESTS}
    if "compact" in kwargs:
        kwargs["newline_before_new_type"] = not kwargs["compact"]
    if "line_length" in kwargs and "max_line_length" not in kwargs:
        kwargs["max_line_length"] = int(kwargs["line_length"] * 1.3)
    return dataclasses.replace(base, **kwargs)


def apply_config_argparse_defaults(
    parser: argparse.ArgumentParser,
    args: Sequence[str],
) -> LatformProjectConfig:
    """
    Resolve the config and fold its namelist ``[format]`` settings into
    ``parser`` as argparse defaults, so explicit CLI flags still override.

    Peeks at ``--config``/``--no-config`` via ``parse_known_args`` before the
    real parse; the caller must call ``parser.parse_args`` afterwards.
    """
    prelim, _ = parser.parse_known_args(list(args))
    config = resolve_config(prelim)
    namelist_defaults = {
        key: value for key, value in config.format.items() if key in NAMELIST_FORMAT_DESTS
    }
    if namelist_defaults:
        parser.set_defaults(**namelist_defaults)
    return config


def add_namelist_format_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``tao.init`` / namelist reformatting flags shared by the CLIs."""
    group = parser.add_argument_group("namelist (*.init/*.nml) formatting")
    group.add_argument(
        "--no-format-namelist",
        dest="format_namelist",
        action="store_false",
        default=True,
        help="Preserve namelist layout verbatim (default: reformat it)",
    )
    group.add_argument(
        "--namelist-indent",
        dest="namelist_indent",
        type=int,
        default=None,
        metavar="N",
        help="Field indent width (default: 2)",
    )
    group.add_argument(
        "--namelist-field-case",
        dest="namelist_field_case",
        choices=CASE_CHOICES,
        default=None,
        help="Case for field names, e.g. global%%plot_on (default: lower)",
    )
    group.add_argument(
        "--no-namelist-align-equals",
        dest="namelist_align_equals",
        action="store_false",
        default=True,
        help="Do not re-align '=' (default: column aligned in block)",
    )
    group.add_argument(
        "--no-namelist-align-comments",
        dest="namelist_align_comments",
        action="store_false",
        default=True,
        help="Do not align trailing '!' comments into a column (default: aligned)",
    )


def build_namelist_options(parsed: argparse.Namespace):
    """Fold the :func:`add_namelist_format_arguments` flags into options."""
    from .types import NamelistFormatOptions

    options = NamelistFormatOptions()
    if getattr(parsed, "namelist_indent", None) is not None:
        options.indent_size = parsed.namelist_indent
    if getattr(parsed, "namelist_field_case", None) is not None:
        options.field_case = parsed.namelist_field_case
    options.align_equals = getattr(parsed, "namelist_align_equals", options.align_equals)
    options.align_comments = getattr(parsed, "namelist_align_comments", options.align_comments)
    return options


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
    if config.tao_init:
        # The tao.init paths are expanded to their lattice files downstream by
        # ``build_files`` (any ``*.init`` argument is auto-expanded).
        return [str(path) for path in config.resolve_tao_init()], True
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
