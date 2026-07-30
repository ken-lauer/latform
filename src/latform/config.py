"""
Configuration file support for latform.

Settings may be provided in a standalone ``latform.toml`` or under
``[tool.latform]`` in ``pyproject.toml``.

Example ``latform.toml``::

    top-level = ["lat/main.bmad"]
    # Or derive the lattice list from a Tao init file (filename fixed as tao.init):
    tao-init = "tao.init"

    [format]
    line-length = 100
    name-case = "upper"
    # Tao/namelist formatting (read by ``latform``):
    namelist-field-case = "lower"
    namelist-align-equals = true
    namelist-logicals = ["T", "F"]   # (true, false) tokens, or false to disable

    [lint]
    ignore = ["LF002"]
    min-name-length = 1
    builtin-constant-rtol = 1e-4

    [lint.per-file-ignores]
    "legacy/*.bmad" = ["LF004", "LF006"]

The equivalent in ``pyproject.toml`` nests everything under ``[tool.latform]``
(e.g. ``[tool.latform.format]``, ``[tool.latform.lint.per-file-ignores]``).
"""

from __future__ import annotations

import fnmatch
import logging
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

from .util import load_json_or_similar

logger = logging.getLogger(__name__)

CONFIG_FILENAMES = ("latform.toml", "pyproject.toml")

_CASE_VALUES = frozenset({"upper", "lower", "same"})

# Config ``[format]`` keys (normalized to snake_case) → the argparse dest / main()
# keyword they map to. Values become argparse defaults so that explicit CLI flags
# still override the config.
_FORMAT_KEYS = frozenset(
    {
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
        "flatten",
        "flatten_call",
        "flatten_inline",
        # Namelist (*.init/*.nml) formatting; map to the same argparse dests as
        # the shared --namelist-* / --no-format-namelist CLI flags.
        "format_namelist",
        "namelist_indent",
        "namelist_field_case",
        "namelist_align_equals",
        "namelist_align_comments",
    }
)

_CASE_KEYS = frozenset(
    {
        "name_case",
        "attribute_case",
        "kind_case",
        "builtin_case",
        "controller_variable_case",
        "namelist_field_case",
    }
)

_FORMAT_BOOL_KEYS = frozenset(
    {
        "compact",
        "strip_comments",
        "flatten",
        "flatten_call",
        "flatten_inline",
        "format_namelist",
        "namelist_align_equals",
        "namelist_align_comments",
    }
)

_FORMAT_INT_KEYS = frozenset(
    {
        "line_length",
        "max_line_length",
        "section_break_width",
        "namelist_indent",
    }
)


class ConfigError(Exception):
    """Raised when a configuration file cannot be parsed or is invalid."""


def _normalize_key(key: str) -> str:
    return key.replace("-", "_")


def _get_option(section: dict[str, Any], key: str, default: Any):
    fallback = key.replace("-", "_")
    return section.get(key, section.get(fallback, default))


@dataclass
class LatformProjectConfig:
    """Resolved latform configuration."""

    root: pathlib.Path
    source: pathlib.Path | None = None
    top_level: list[str] = field(default_factory=list)
    tao_init: list[str] = field(default_factory=list)
    format: dict[str, Any] = field(default_factory=dict)
    # Canonical (true, false) tokens for logical namelist values, or None to
    # leave logicals untouched. Consumed by the ``latform`` formatter.
    namelist_logicals: tuple[str, str] | None = ("T", "F")
    lint_ignore: list[str] = field(default_factory=list)
    per_file_ignores: dict[str, list[str]] = field(default_factory=dict)
    min_name_length: int = 1
    builtin_constant_rtol: float = 1e-4

    @classmethod
    def empty(cls, root: pathlib.Path | None = None) -> LatformProjectConfig:
        return cls(root=root or pathlib.Path.cwd())

    @classmethod
    def from_file(cls, path: pathlib.Path) -> LatformProjectConfig:
        """Load and validate a latform config from ``path``."""
        try:
            data = load_json_or_similar(path)
        except Exception as ex:
            raise ConfigError(f"Failed to read config {path}: {ex}") from ex

        section = _extract_section(path, data)
        config = cls(root=path.parent, source=path)

        top_level = _get_option(section, "top-level", [])
        if top_level and not isinstance(top_level, list):
            raise ConfigError(f"{path}: top-level must be a list of paths")
        config.top_level = [str(entry) for entry in top_level]

        tao_init = _get_option(section, "tao-init", [])
        if isinstance(tao_init, (str, pathlib.Path)):
            tao_init = [tao_init]
        if not isinstance(tao_init, list):
            raise ConfigError(f"{path}: tao-init must be a path or list of paths")
        config.tao_init = [str(entry) for entry in tao_init]

        raw_format = section.get("format", {})
        if not isinstance(raw_format, dict):
            raise ConfigError(f"{path}: [format] must be a table")
        fmt: dict[str, Any] = {}
        for key, value in raw_format.items():
            norm = _normalize_key(key)
            # ``namelist-logicals`` is not an argparse dest, so it is stored on
            # the config rather than folded into ``format``.
            if norm == "namelist_logicals":
                config.namelist_logicals = _parse_logicals(path, value)
                continue
            if norm not in _FORMAT_KEYS:
                logger.warning("%s: unknown format setting %r (ignored)", path, key)
                continue
            fmt[norm] = value
        _validate_format_types(path, fmt)
        _parse_case_fields(path, fmt)
        config.format = fmt

        lint = section.get("lint", {})
        if not isinstance(lint, dict):
            raise ConfigError(f"{path}: [lint] must be a table")
        ignore = lint.get("ignore", [])
        if not isinstance(ignore, list):
            raise ConfigError(f"{path}: lint.ignore must be a list of codes")
        config.lint_ignore = [str(code).upper() for code in ignore]

        per_file = _get_option(lint, "per-file-ignores", {})
        if not isinstance(per_file, dict):
            raise ConfigError(f"{path}: lint.per-file-ignores must be a table")
        config.per_file_ignores = {
            str(pattern): [str(code).upper() for code in codes]
            for pattern, codes in per_file.items()
        }

        min_name_length = _get_option(lint, "min-name-length", 1)
        if (
            isinstance(min_name_length, bool)
            or not isinstance(min_name_length, int)
            or min_name_length < 1
        ):
            raise ConfigError(
                f"{path}: lint.min-name-length must be an integer >= 1, got {min_name_length!r}"
            )
        config.min_name_length = min_name_length

        rtol = _get_option(lint, "builtin-constant-rtol", 1e-4)
        if (
            isinstance(rtol, bool)
            or not isinstance(rtol, (int, float))
            or not math.isfinite(rtol)
            or rtol <= 0
        ):
            raise ConfigError(
                f"{path}: lint.builtin-constant-rtol must be a positive number, got {rtol!r}"
            )
        config.builtin_constant_rtol = float(rtol)

        return config

    def format_argparse_defaults(self) -> dict[str, Any]:
        """Return ``[format]`` settings as argparse-style ``dest -> value`` pairs."""
        return dict(self.format)

    def resolve_top_level(self) -> list[pathlib.Path]:
        """Return the ``top-level`` entries resolved against the config directory."""
        return [self.root / entry for entry in self.top_level]

    def resolve_tao_init(self) -> list[pathlib.Path]:
        """Return the ``tao-init`` entries resolved against the config directory."""
        return [self.root / entry for entry in self.tao_init]

    def ignores_for(self, path: pathlib.Path | str) -> set[str]:
        """
        Return the lint codes to ignore for ``path``.

        This is the global ``[lint] ignore`` list plus any
        ``[lint.per-file-ignores]`` entries whose glob matches the file.
        """
        codes = set(self.lint_ignore)
        path = pathlib.Path(path)
        candidates = {path.name, str(path)}
        try:
            candidates.add(str(path.resolve().relative_to(self.root)))
        except (ValueError, OSError):
            pass

        for pattern, pattern_codes in self.per_file_ignores.items():
            if any(fnmatch.fnmatch(candidate, pattern) for candidate in candidates):
                codes.update(pattern_codes)
        return codes


def _has_latform_table(path: pathlib.Path) -> bool:
    try:
        data = load_json_or_similar(path)
    except Exception:
        return False
    return isinstance(data.get("tool"), dict) and "latform" in data["tool"]


def find_config_file(start: pathlib.Path) -> pathlib.Path | None:
    """Search ``start`` and its parents for a latform config file."""
    start = start.resolve()
    for directory in (start, *start.parents):
        for name in CONFIG_FILENAMES:
            candidate = directory / name
            if not candidate.is_file():
                continue
            if name == "pyproject.toml" and not _has_latform_table(candidate):
                continue
            return candidate
    return None


def _extract_section(path: pathlib.Path, data: dict[str, Any]) -> dict[str, Any]:
    if path.name == "pyproject.toml":
        section = data.get("tool", {}).get("latform", {})
    else:
        section = data
    if not isinstance(section, dict):
        raise ConfigError(f"{path}: [tool.latform]/latform config must be a table")
    return section


def _parse_case_fields(source: pathlib.Path, fmt: dict[str, Any]) -> None:
    for key in _CASE_KEYS:
        if key in fmt and fmt[key] not in _CASE_VALUES:
            raise ConfigError(
                f"{source}: format.{key} must be one of {sorted(_CASE_VALUES)}, got {fmt[key]!r}"
            )


def _validate_format_types(source: pathlib.Path, fmt: dict[str, Any]) -> None:
    """Type-check the boolean and integer ``[format]`` settings."""
    for key in _FORMAT_BOOL_KEYS & fmt.keys():
        if not isinstance(fmt[key], bool):
            raise ConfigError(f"{source}: format.{key} must be a boolean, got {fmt[key]!r}")
    for key in _FORMAT_INT_KEYS & fmt.keys():
        value = fmt[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{source}: format.{key} must be an integer, got {value!r}")
        if key == "namelist_indent" and value < 0:
            raise ConfigError(f"{source}: format.namelist-indent must be >= 0, got {value!r}")


def _parse_logicals(source: pathlib.Path, value: Any) -> tuple[str, str] | None:
    """Parse ``namelist-logicals``: a ``[true, false]`` string pair, or ``false``."""
    if value is False:
        return None
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, str) for item in value)
    ):
        return (value[0], value[1])
    raise ConfigError(
        f"{source}: format.namelist-logicals must be a [true, false] pair of strings "
        f"or false to disable, got {value!r}"
    )


def discover_config(
    start: pathlib.Path | None = None,
    *,
    explicit: pathlib.Path | str | None = None,
    enabled: bool = True,
) -> LatformProjectConfig:
    """
    Locate and load the applicable config, returning an empty one if none applies.

    Parameters
    ----------
    start : pathlib.Path, optional
        Directory to begin the upward search. Defaults to the current directory.
    explicit : pathlib.Path or str, optional
        An explicit config file path (from ``--config``); used verbatim.
    enabled : bool, optional
        If False (``--no-config``), configuration is skipped entirely.
    """
    start = start or pathlib.Path.cwd()
    if not enabled:
        return LatformProjectConfig.empty(start)

    if explicit is not None:
        path = pathlib.Path(explicit)
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        return LatformProjectConfig.from_file(path)

    found = find_config_file(start)
    if found is None:
        return LatformProjectConfig.empty(start)
    return LatformProjectConfig.from_file(found)
