"""
Apply values and renames to a single Bmad or namelist template file.

This module is the foundation for :mod:`latform.template`: it holds the value
overrides (`apply_values`, `apply_namelist_values`), the rename
ruleset machinery, and the single-file `interpolate` entry point, all of
which :mod:`latform.template` reuses for multi-instance expansion.
"""

from __future__ import annotations

import logging
import pathlib
import re
import sys
from typing import Literal

from ._namelist import NamelistFile, is_namelist_file
from .comments import Comments
from .parser import MemoryFiles, parse
from .statements import Constant, Element
from .token import Role, Token
from .types import Attribute, FormatOptions, NamelistFormatOptions, Seq
from .util import load_json_or_similar
from .walk import walk

__all__ = [
    "FileFormat",
    "apply_values",
    "apply_namelist_values",
    "interpolate",
    "interpolate_namelist",
    "main_apply",
    "cli_main_apply",
]

FileFormat = Literal["namelist", "bmad"]


def _parse_value(text: str) -> Token | Seq:
    """Parse a scalar override into a value node (a ``Token`` or ``Seq``)."""
    (statement,) = parse(f"__latform_value__ = {text}", "<templating-value>", annotate=False)
    assert isinstance(statement, Constant)
    return statement.value


def _override_element_attribute(element: Element, attr_name: str, value: object) -> None:
    parsed = _parse_value(str(value))
    try:
        attr = element.get_named_attribute(attr_name, partial_match=False)
    except KeyError:
        element.attributes.append(Attribute(name=Token(attr_name), value=parsed))
    else:
        attr.value = parsed


def _remove_element_attribute(element: Element, attr_name: str) -> None:
    """Drop an attribute from an element (no-op if it is not present)."""
    element.attributes = [
        attr
        for attr in element.attributes
        if not (isinstance(attr.name, Token) and str(attr.name).lower() == attr_name.lower())
    ]


def _resolve_targets(named: dict, key: str) -> list:
    """
    Resolve a values key to target statements.

    A ``/regex/`` key (slash-delimited) matches every named item whose name
    matches; any other key is an exact, case-insensitive name.
    """
    if len(key) >= 2 and key.startswith("/") and key.endswith("/"):
        pattern = re.compile(key[1:-1], re.IGNORECASE)
        matches = [statement for name, statement in named.items() if pattern.search(str(name))]
        if not matches:
            raise KeyError(f"no element or constant matches {key!r}")
        return matches
    try:
        return [named[key.upper()]]
    except KeyError:
        raise KeyError(f"template has no element or constant named {key!r}") from None


def apply_values(files: MemoryFiles, values: dict) -> None:
    """
    Apply a values mapping to the parsed template in place.

    Parameters
    ----------
    files : MemoryFiles
        A parsed, annotated file collection.
    values : dict
        Keys are element or constant names (or a ``/regex/`` matching several).
        A ``dict`` value overrides element attributes (``{attr: value}``); an
        attribute value of ``None`` removes that attribute; a scalar value
        overrides a constant's value.
    """
    named = files.get_named_items()
    for key, override in values.items():
        for target in _resolve_targets(named, key):
            if isinstance(override, dict):
                if not isinstance(target, Element):
                    raise TypeError(f"{key!r} is not an element; cannot set attributes on it")
                for attr_name, attr_value in override.items():
                    if attr_value is None:
                        _remove_element_attribute(target, attr_name)
                    else:
                        _override_element_attribute(target, attr_name, attr_value)
            else:
                if not hasattr(target, "value"):
                    raise TypeError(f"{key!r} cannot take a scalar override")
                target.value = _parse_value(str(override))


def split_namelist_key(key: str) -> tuple[str, int]:
    """
    Split a ``namelists`` override key into ``(name, index)``.

    A trailing ``#N`` (1-based) targets the N-th group of a repeated name, e.g.
    ``tao_d1_data#2`` -> ``("tao_d1_data", 1)``. A bare name targets the first.
    """
    if "#" in key:
        name, _, suffix = key.rpartition("#")
        return name, int(suffix) - 1
    return key, 0


def apply_namelist_values(nml_file: NamelistFile, values: dict) -> None:
    """
    Apply a values mapping to a parsed namelist file in place.

    Parameters
    ----------
    nml_file : NamelistFile
        A parsed namelist file.
    values : dict
        Keys are namelist group names, optionally with a ``name#N`` suffix (1-based)
        to target the N-th of a repeated group.
        Each value is a ``{key: value}`` mapping of raw assignment values;
        existing keys are updated in place and
        missing keys are appended.
        A value of ``None`` removes that key.
        A group named only for removals that does not exist is left uncreated.
    """
    for name_key, assignments in values.items():
        name, index = split_namelist_key(name_key)
        removals = [key for key, value in assignments.items() if value is None]
        settings = {key: str(value) for key, value in assignments.items() if value is not None}
        if settings:
            target = nml_file.update_namelist(name, settings, index=index)
        else:
            target = nml_file.get_namelist(name, index)
        if target is not None:
            for key in removals:
                target.remove(key)


def interpolate_namelist(
    contents: str,
    *,
    values: dict | None = None,
    filename: str = "tao.init",
    options: NamelistFormatOptions | None = None,
) -> str:
    """
    Interpolate a single namelist (``*.init``/``*.nml``) template file.

    Parameters
    ----------
    contents : str
        The namelist file contents.
    values : dict, optional
        Namelist overrides. See `apply_namelist_values`.
    filename : str, optional
        Virtual filename used for source locations.
    options : NamelistFormatOptions, optional
        When given, re-format the output (field indentation, case, and
        alignment). When ``None`` (default), the source layout is preserved
        verbatim aside from the applied value edits.

    Returns
    -------
    str
        The interpolated namelist file.
    """
    nml_file = NamelistFile.parse(contents, filename)
    if values:
        apply_namelist_values(nml_file, values)
    return nml_file.render(options)


def interpolate(
    contents: str,
    *,
    values: dict | None = None,
    renames: dict | None = None,
    prefix: dict[str, str] | None = None,
    suffix: dict[str, str] | None = None,
    parts: list[dict] | dict | None = None,
    delimiters: str | list[str] | None = None,
    filename: str = "template.bmad",
    options: FormatOptions | None = None,
    file_format: FileFormat | None = None,
    format_namelist: bool = True,
) -> str:
    """
    Interpolate a single template file and return the formatted result.

    Handles both Bmad lattice files and Fortran-namelist files (``*.init`` /
    ``*.nml``). The format is chosen from ``file_format`` when given, otherwise
    auto-detected from ``filename``'s extension.

    Parameters
    ----------
    contents : str
        The template file contents (valid Bmad, or a namelist file).
    values : dict, optional
        For Bmad, overrides keyed by element/constant name (see
        `apply_values`). For namelist files, overrides keyed by namelist
        group name (see `apply_namelist_values`).
    renames : dict, optional
        Rename rules applied after values: either the flat shortcut form
        (``{pattern: replacement}``, literal unless it contains ``* + ?``) or the
        structured form (``{prefix: ..., suffix: ..., regex: ..., parts: ...}``).
        Bmad only.
    prefix, suffix : dict[str, str], optional
        Convenience ``{from: to}`` maps, equivalent to ``renames.prefix`` /
        ``renames.suffix``. Bmad only.
    parts : list[dict] | dict, optional
        Convenience form equivalent to ``renames.parts`` — a list of
        ``{delimiters, from, to}`` or a ``{from: to}`` map using ``delimiters``.
        Bmad only.
    delimiters : str | list[str], optional
        Default delimiter set for ``prefix``/``suffix``/``parts`` (default ``. _``).
    filename : str, optional
        Virtual filename used for source locations and format auto-detection.
    options : FormatOptions, optional
        Formatting options for the emitted output. For Bmad, always applied.
        For namelist files, applied only when ``format_namelist`` is set.
    file_format : {"bmad", "namelist"}, optional
        Force the input format instead of auto-detecting from ``filename``.
    format_namelist : bool, optional
        Reformat namelist output (field indentation and a blank line after each
        group) using ``options``. On by default; set False to preserve the
        source layout verbatim aside from the applied value edits.

    Returns
    -------
    str
        The interpolated, formatted file.
    """
    if file_format is None:
        file_format = "namelist" if is_namelist_file(filename) else "bmad"
    if file_format == "namelist":
        if renames or prefix or suffix or parts:
            raise ValueError("rename options are not supported for namelist files")
        nml_options = None
        if format_namelist:
            nml_options = (options or FormatOptions()).namelist
        return interpolate_namelist(
            contents,
            values=values,
            filename=filename,
            options=nml_options,
        )
    if file_format != "bmad":
        raise ValueError(f"unknown file format {file_format!r} (expected 'bmad' or 'namelist')")

    files = MemoryFiles.from_contents(contents, filename)
    files.parse(recurse=False)
    files.annotate()

    if values:
        apply_values(files, values)
        # Re-annotate so injected value tokens matching a defined name pick up
        # Role.name_ before renames run.
        files.annotate()

    rules = _build_ruleset(renames, prefix, suffix, parts, delim_set(delimiters))
    if rules["literal"] or rules["regex"] or rules["parts"]:
        apply_renames(files, rules)

    files.reformat(options or FormatOptions())
    return files.formatted_contents


# --------------------------------------------------------------------------- #
# Rename ruleset machinery (shared with latform.template)
# --------------------------------------------------------------------------- #


def _is_regex(pattern: str) -> bool:
    return any(char in pattern for char in "*+?")


def _collect_names(files: MemoryFiles) -> set[str]:
    """All name-role token texts across the loaded files."""
    names: set[str] = set()
    for statements in files.by_filename.values():
        for item in walk(statements):
            node = item.node
            if isinstance(node, Token) and node.role is Role.name_:
                names.add(str(node))
    return names


# A normalized set of rename rules. ``regex`` is ordered (first match wins):
# explicit regex, then prefix-derived, then suffix-derived. ``literal`` (exact
# name) is tried before any regex; ``parts`` (segment rewrite) is tried last.
#   {"literal": {from_lower: to}, "regex": [(pattern, repl)], "parts": [(delims, from, to)]}
_RESERVED_RENAME_KEYS = frozenset({"prefix", "suffix", "regex", "parts"})


def _empty_ruleset() -> dict:
    return {"literal": {}, "regex": [], "parts": []}


def delim_set(delimiters: str | list[str] | None) -> set[str]:
    if delimiters is None:
        return set("._")
    if isinstance(delimiters, (list, tuple)):
        return set("".join(delimiters))
    return set(str(delimiters))


def _prefix_to_regex(frm: str, to: str, delims: str | set[str] = "._") -> tuple[str, str]:
    """Leading-anchored rule: ``FROM`` at the start, bounded by a delimiter/end."""
    chars = "".join(re.escape(c) for c in delims)
    if frm and frm[-1] in delims:  # delimiter already baked into FROM
        pattern = rf"^{re.escape(frm)}(.*)"
    else:
        pattern = rf"^{re.escape(frm)}([{chars}].*|$)"
    return pattern, f"{to}\\1"


def _suffix_to_regex(frm: str, to: str, delims: str | set[str] = "._") -> tuple[str, str]:
    """Trailing-anchored rule (mirror of prefix): ``FROM`` at the end, bounded."""
    chars = "".join(re.escape(c) for c in delims)
    if frm and frm[0] in delims:  # delimiter already baked into FROM
        pattern = rf"(.*){re.escape(frm)}$"
    else:
        pattern = rf"(^|.*[{chars}]){re.escape(frm)}$"
    return pattern, f"\\1{to}"


def normalize_renames(value: dict | None, default_delims: set[str] | None = None) -> dict:
    """
    Turn a ``renames`` value (flat shortcut or structured) into a RuleSet.

    ``default_delims`` is the delimiter set used by ``prefix``/``suffix`` and by
    ``parts`` entries that do not specify their own (settable via a top-level
    ``delimiters`` key). Defaults to ``. _``.
    """
    default_delims = default_delims if default_delims is not None else delim_set(None)
    rules = _empty_ruleset()
    if not value:
        return rules

    if set(value) <= _RESERVED_RENAME_KEYS:  # structured form
        for pattern, repl in (value.get("regex") or {}).items():
            rules["regex"].append((pattern, repl))  # explicit regex: highest precedence
        for frm, to in (value.get("prefix") or {}).items():
            rules["regex"].append(_prefix_to_regex(frm, to, default_delims))
        for frm, to in (value.get("suffix") or {}).items():
            rules["regex"].append(_suffix_to_regex(frm, to, default_delims))
        parts = value.get("parts") or []
        if isinstance(parts, dict):  # {from: to} shorthand using the default delimiters
            for frm, to in parts.items():
                rules["parts"].append((default_delims, frm, to))
        else:  # list of {delimiters, from, to}; per-entry delimiters override the default
            for entry in parts:
                delims = (
                    delim_set(entry["delimiters"])
                    if entry.get("delimiters") is not None
                    else default_delims
                )
                rules["parts"].append((delims, entry["from"], entry["to"]))
    else:  # flat shortcut: literal-vs-regex autodetection
        for pattern, repl in value.items():
            if _is_regex(pattern):
                rules["regex"].append((pattern, repl))
            else:
                # Keyed by lowercased name; the original spelling is kept for
                # case-sensitive contexts (comment text).
                rules["literal"][pattern.lower()] = (pattern, repl)
    return rules


def merge_rulesets(base: dict, overriding: dict) -> dict:
    """Merge two RuleSets; ``overriding`` takes precedence (tried first)."""
    return {
        "literal": {**base["literal"], **overriding["literal"]},
        "regex": overriding["regex"] + base["regex"],
        "parts": overriding["parts"] + base["parts"],
    }


def _build_ruleset(
    renames: dict | None = None,
    prefix: dict[str, str] | None = None,
    suffix: dict[str, str] | None = None,
    parts: list[dict] | dict | None = None,
    default_delims: set[str] | None = None,
) -> dict:
    """Combine a ``renames`` value with convenience prefix/suffix/parts kwargs."""
    base = normalize_renames(renames, default_delims)
    extra = normalize_renames(
        {"prefix": prefix or {}, "suffix": suffix or {}, "parts": parts or []}, default_delims
    )
    return merge_rulesets(extra, base)  # explicit renames precede convenience kwargs


def _apply_part_rule(
    name: str, delims: set[str], frm: str, to: str, *, case_sensitive: bool = False
) -> tuple[str, bool]:
    """Rename whole delimiter-separated segments of ``name`` equal to ``frm``."""
    out: list[str] = []
    segment = ""
    changed = False

    def flush() -> None:
        nonlocal segment, changed
        if segment == frm if case_sensitive else segment.lower() == frm.lower():
            out.append(to)
            changed = True
        else:
            out.append(segment)
        segment = ""

    for char in name:
        if char in delims:
            flush()
            out.append(char)
        else:
            segment += char
    flush()
    return "".join(out), changed


def _expand_ruleset_over_names(
    rules: dict, names: set[str], *, case_sensitive: bool = False
) -> dict[str, str]:
    """
    Resolve a RuleSet against the known name set into literal renames.

    Bmad names are case-insensitive, so code renames match regardless of case;
    ``case_sensitive`` is for prose-adjacent contexts (comment text) where a
    lowercase ordinary word must not match an uppercase name rule.
    """
    literal = rules["literal"]
    flags = 0 if case_sensitive else re.IGNORECASE
    regexes = [(re.compile(pattern, flags), repl) for pattern, repl in rules["regex"]]
    parts = rules["parts"]

    expanded: dict[str, str] = {}
    for name in names:
        entry = literal.get(name.lower())
        if entry is not None and (not case_sensitive or name == entry[0]):
            expanded[name] = entry[1]
            continue
        matched = False
        for pattern, repl in regexes:
            if pattern.search(name):
                expanded[name] = pattern.sub(repl, name)
                matched = True
                break
        if matched:
            continue
        for delims, frm, to in parts:
            new_name, changed = _apply_part_rule(
                name, delims, frm, to, case_sensitive=case_sensitive
            )
            if changed:
                expanded[name] = new_name
                break
    return expanded


def apply_renames(files: MemoryFiles, rules: dict) -> None:
    expanded = _expand_ruleset_over_names(rules, _collect_names(files))
    if expanded:
        files.rename(expanded)
    _rename_in_comments(files, rules)


_COMMENT_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.]*")


def _rename_in_comments(files: MemoryFiles, rules: dict) -> None:
    """
    Apply rename rules to name-like words inside comment text.

    Comments (including commented-out code) reference the same names as the
    code; a name-map rename cannot reach them since commented-out names are
    not part of the parsed name set, so the ruleset itself is resolved over
    the words found in comments.
    """
    all_comments: list[Comments] = []
    for statements in files.by_filename.values():
        for statement in statements:
            if statement.comments:
                all_comments.append(statement.comments)
            for item in walk(statement):
                comments = getattr(item.node, "comments", None)
                if comments:
                    all_comments.append(comments)

    words: set[str] = set()
    for comments in all_comments:
        for token in [*comments.pre, comments.inline]:
            if token is not None:
                words.update(_COMMENT_WORD.findall(str(token)))

    expanded = _expand_ruleset_over_names(rules, words, case_sensitive=True)
    if not expanded:
        return

    def rewrite(token: Token) -> Token:
        new_text = _COMMENT_WORD.sub(lambda m: expanded.get(m.group(0), m.group(0)), str(token))
        if new_text == str(token):
            return token
        return Token(new_text, role=token.role)

    for comments in all_comments:
        comments.pre = [rewrite(token) for token in comments.pre]
        if comments.inline is not None:
            comments.inline = rewrite(comments.inline)


# --------------------------------------------------------------------------- #
# CLI helpers (shared with latform.template) and the latform-apply command
# --------------------------------------------------------------------------- #

_APPLY_DESCRIPTION = """\
Apply values and renames to a single Bmad or namelist template file.

For Bmad, overrides are keyed by element/constant name and renames rewrite element
names. For Fortran-namelist files (*.init/*.nml), overrides are keyed by namelist
group (--values, or --set NAMELIST KEY VALUE); renames do not apply. The format is
auto-detected from the extension unless --format is given. See docs/cli.md.
"""


def add_logging_argument(parser) -> None:
    parser.add_argument(
        "--log",
        "-L",
        dest="log_level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "CRITICAL"),
        help="Python logging level",
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level)
    logging.getLogger("latform").setLevel(level)


def _load_values(source: str) -> dict | None:
    """
    Load a ``--values`` mapping from a file, or from stdin when ``source`` is ``-``.

    A file's format is taken from its extension; stdin is parsed as YAML (a JSON
    superset, so JSON on stdin works too).
    """
    if source == "-":
        import yaml

        return yaml.safe_load(sys.stdin.read())
    return load_json_or_similar(source)


def _merge_set_overrides(values: dict | None, sets: list[tuple[str, str, str]]) -> dict | None:
    """
    Fold ``--set NAMELIST KEY VALUE`` triples into a namelist ``values`` mapping.

    Later triples win, and ``--set`` overrides values loaded from ``--values``.
    """
    if not sets:
        return values
    merged: dict = {k: dict(v) if isinstance(v, dict) else v for k, v in (values or {}).items()}
    for namelist, key, value in sets:
        merged.setdefault(namelist, {})[key] = value
    return merged


def _cmd_apply(parsed) -> None:
    import dataclasses

    from . import cli
    from .output import default_options

    contents = pathlib.Path(parsed.template).read_text()
    file_format = parsed.format or ("namelist" if is_namelist_file(parsed.template) else "bmad")

    options = dataclasses.replace(default_options, namelist=cli.build_namelist_options(parsed))

    values = _load_values(parsed.values) if parsed.values else None
    if parsed.set_ and file_format != "namelist":
        raise SystemExit("--set is only valid for namelist files (*.init, *.nml)")
    values = _merge_set_overrides(values, parsed.set_)

    renames = {old: new for old, new in parsed.rename} or None
    prefix = {frm: to for frm, to in parsed.prefix} or None
    suffix = {frm: to for frm, to in parsed.suffix} or None
    parts = [{"delimiters": d, "from": frm, "to": to} for d, frm, to in parsed.parts] or None
    result = interpolate(
        contents,
        values=values,
        renames=renames,
        prefix=prefix,
        suffix=suffix,
        parts=parts,
        delimiters=parsed.delimiters,
        filename=parsed.template,
        options=options,
        file_format=file_format,
        format_namelist=parsed.format_namelist,
    )
    if parsed.in_place:
        pathlib.Path(parsed.template).write_text(result)
        print(f"wrote: {parsed.template}")
    elif parsed.output:
        pathlib.Path(parsed.output).write_text(result)
        print(f"wrote: {parsed.output}")
    else:
        sys.stdout.write(result)


def main_apply(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="latform-apply",
        description=_APPLY_DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_logging_argument(parser)
    parser.add_argument("template", help="Template file (Bmad, or a namelist *.init/*.nml file)")
    parser.add_argument(
        "--format",
        choices=("bmad", "namelist"),
        default=None,
        help="Force input format (default: auto-detect from extension)",
    )
    parser.add_argument(
        "--values",
        help=(
            "YAML/JSON/TOML overrides (element/constant for Bmad; namelist groups "
            "for namelist). Use '-' to read YAML/JSON from stdin"
        ),
    )
    parser.add_argument(
        "--set",
        nargs=3,
        metavar=("NAMELIST", "KEY", "VALUE"),
        action="append",
        default=[],
        dest="set_",
        help="Set a namelist KEY=VALUE in group NAMELIST (namelist files only); repeatable",
    )
    parser.add_argument(
        "--rename",
        nargs=2,
        metavar=("OLD", "NEW"),
        action="append",
        default=[],
        help="Rename rule (literal or regex); repeatable",
    )
    parser.add_argument(
        "--prefix",
        nargs=2,
        metavar=("FROM", "TO"),
        action="append",
        default=[],
        help="Leading-prefix rename (bounded by . or _); repeatable",
    )
    parser.add_argument(
        "--suffix",
        nargs=2,
        metavar=("FROM", "TO"),
        action="append",
        default=[],
        help="Trailing-suffix rename (bounded by . or _); repeatable",
    )
    parser.add_argument(
        "--parts",
        nargs=3,
        metavar=("DELIMS", "FROM", "TO"),
        action="append",
        default=[],
        help="Segment rename: rename whole DELIMS-separated parts equal to FROM; repeatable",
    )
    parser.add_argument(
        "--delimiters",
        default=None,
        help="Default delimiter set for --prefix/--suffix (default: . and _)",
    )

    from . import cli

    cli.add_namelist_format_arguments(parser)

    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument("-o", "--output", help="Output file (default: stdout)")
    out_group.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help="Rewrite the template file in place",
    )

    parsed = parser.parse_args(argv if argv is not None else sys.argv[1:])
    configure_logging(parsed.log_level)
    _cmd_apply(parsed)


def cli_main_apply(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ``latform-apply`` (interpolate a single file)."""
    main_apply(argv)


if __name__ == "__main__":
    cli_main_apply()
