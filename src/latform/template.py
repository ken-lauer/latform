from __future__ import annotations

import logging
import pathlib
import posixpath
import re
import sys

from .parser import MemoryFiles, parse
from .statements import Constant, Element
from .tao import TaoInit
from .token import Role, Token
from .types import Attribute, FormatOptions, Seq
from .util import load_json_or_similar
from .walk import walk

__all__ = [
    "apply_values",
    "interpolate",
    "instantiate",
    "load_instances",
    "write_instances",
    "main",
    "cli_main",
]


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
    """Resolve a values key to target statements.

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
    """Apply a values mapping to the parsed template in place.

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
) -> str:
    """Interpolate a single template file and return formatted Bmad.

    Parameters
    ----------
    contents : str
        The template file contents (valid Bmad).
    values : dict, optional
        Overrides keyed by element/constant name. See :func:`apply_values`.
    renames : dict, optional
        Rename rules applied after values: either the flat shortcut form
        (``{pattern: replacement}``, literal unless it contains ``* + ?``) or the
        structured form (``{prefix: ..., suffix: ..., regex: ..., parts: ...}``).
    prefix, suffix : dict[str, str], optional
        Convenience ``{from: to}`` maps, equivalent to ``renames.prefix`` /
        ``renames.suffix``.
    parts : list[dict] | dict, optional
        Convenience form equivalent to ``renames.parts`` — a list of
        ``{delimiters, from, to}`` or a ``{from: to}`` map using ``delimiters``.
    delimiters : str | list[str], optional
        Default delimiter set for ``prefix``/``suffix``/``parts`` (default ``. _``).
    filename : str, optional
        Virtual filename used for source locations.
    options : FormatOptions, optional
        Formatting options for the emitted output.

    Returns
    -------
    str
        The interpolated, formatted Bmad file.
    """
    files = MemoryFiles.from_contents(contents, filename)
    files.parse(recurse=False)
    files.annotate()

    if values:
        apply_values(files, values)
        # Re-annotate so injected value tokens matching a defined name pick up
        # Role.name_ before renames run.
        files.annotate()

    rules = _build_ruleset(renames, prefix, suffix, parts, _delim_set(delimiters))
    if rules["literal"] or rules["regex"] or rules["parts"]:
        _apply_renames(files, rules)

    files.reformat(options or FormatOptions())
    return files.formatted_contents


_INTERP = re.compile(r"\{instance(?::(upper|lower))?\}")


def _interpolate(text: str, instance: str) -> str:
    """Resolve ``{instance}``, ``{instance:upper}``, ``{instance:lower}``."""

    def repl(match: re.Match) -> str:
        transform = match.group(1)
        if transform == "upper":
            return instance.upper()
        if transform == "lower":
            return instance.lower()
        return instance

    return _INTERP.sub(repl, text)


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


def _delim_set(delimiters: str | list[str] | None) -> set[str]:
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


def _normalize_renames(value: dict | None, default_delims: set[str] | None = None) -> dict:
    """Turn a ``renames`` value (flat shortcut or structured) into a RuleSet.

    ``default_delims`` is the delimiter set used by ``prefix``/``suffix`` and by
    ``parts`` entries that do not specify their own (settable via a top-level
    ``delimiters`` key). Defaults to ``. _``.
    """
    default_delims = default_delims if default_delims is not None else _delim_set(None)
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
                    _delim_set(entry["delimiters"])
                    if entry.get("delimiters") is not None
                    else default_delims
                )
                rules["parts"].append((delims, entry["from"], entry["to"]))
    else:  # flat shortcut: literal-vs-regex autodetection
        for pattern, repl in value.items():
            if _is_regex(pattern):
                rules["regex"].append((pattern, repl))
            else:
                rules["literal"][pattern.lower()] = repl
    return rules


def _merge_rulesets(base: dict, overriding: dict) -> dict:
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
    base = _normalize_renames(renames, default_delims)
    extra = _normalize_renames(
        {"prefix": prefix or {}, "suffix": suffix or {}, "parts": parts or []}, default_delims
    )
    return _merge_rulesets(extra, base)  # explicit renames precede convenience kwargs


def _interpolate_ruleset(rules: dict, instance: str) -> dict:
    return {
        "literal": {k: _interpolate(v, instance) for k, v in rules["literal"].items()},
        "regex": [(pat, _interpolate(repl, instance)) for pat, repl in rules["regex"]],
        "parts": [(d, frm, _interpolate(to, instance)) for d, frm, to in rules["parts"]],
    }


def _apply_part_rule(name: str, delims: set[str], frm: str, to: str) -> tuple[str, bool]:
    """Rename whole delimiter-separated segments of ``name`` equal to ``frm``."""
    out: list[str] = []
    segment = ""
    changed = False

    def flush() -> None:
        nonlocal segment, changed
        if segment.lower() == frm.lower():
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


def _expand_ruleset_over_names(rules: dict, names: set[str]) -> dict[str, str]:
    """Resolve a RuleSet against the known name set into literal renames."""
    literal = rules["literal"]
    regexes = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in rules["regex"]]
    parts = rules["parts"]

    expanded: dict[str, str] = {}
    for name in names:
        if name.lower() in literal:
            expanded[name] = literal[name.lower()]
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
            new_name, changed = _apply_part_rule(name, delims, frm, to)
            if changed:
                expanded[name] = new_name
                break
    return expanded


def _apply_renames(files: MemoryFiles, rules: dict) -> None:
    expanded = _expand_ruleset_over_names(rules, _collect_names(files))
    if expanded:
        files.rename(expanded)


def _statement_text(statement, options: FormatOptions) -> str:
    from .output import format_statements

    return format_statements(statement, options)


def _apply_inserts(files: MemoryFiles, inserts: dict, options: FormatOptions) -> None:
    """
    Insert extra statements after an anchor in the named files.

    ``inserts`` maps a file basename to a list of ``{after, text}`` directives.
    """
    for basename, directives in inserts.items():
        path = _resolve_by_basename(files, basename)
        statements = list(files.by_filename[path])
        for directive in directives:
            anchor = directive["after"]
            new_statements = list(parse(directive["text"], "<insert>", annotate=False))
            index = _anchor_index(statements, anchor, options)
            statements[index + 1 : index + 1] = new_statements
        files.by_filename[path] = statements


def _anchor_index(statements: list, anchor: str, options: FormatOptions) -> int:
    for idx, statement in enumerate(statements):
        if anchor in _statement_text(statement, options):
            return idx
    raise KeyError(f"insert anchor not found: {anchor!r}")


def _resolve_by_basename(files: MemoryFiles, basename: str) -> pathlib.Path:
    for path in files.by_filename:
        if path.name == basename:
            return path
    raise KeyError(f"template has no file named {basename!r}")


def _rewrite_call_filenames(
    files: MemoryFiles,
    transform_paths: dict,
    in_to_out: dict[str, str],
) -> None:
    """
    Rewrite ``call, file=`` targets that point at other transform-set files.

    Parameters
    ----------
    transform_paths : dict
        ``{resolved_path: (input_rel, output_rel)}`` for the transform set.
    in_to_out : dict[str, str]
        Normalized input relative path -> output relative path.
    """
    for path, (input_rel, output_rel) in transform_paths.items():
        in_dir = posixpath.dirname(input_rel)
        out_dir = posixpath.dirname(output_rel)
        for item in walk(files.by_filename[path]):
            node = item.node
            if not (isinstance(node, Token) and node.role is Role.filename):
                continue
            resolved_in = posixpath.normpath(posixpath.join(in_dir, str(node)))
            new_out = in_to_out.get(resolved_in)
            if new_out is not None:
                new_target = posixpath.relpath(new_out, out_dir or ".")
                item.replace(Token(new_target, role=Role.filename))


def _split_namelist_key(key: str) -> tuple[str, int]:
    """
    Split a ``namelists`` override key into ``(name, index)``.

    A trailing ``#N`` (1-based) targets the N-th group of a repeated name, e.g.
    ``tao_d1_data#2`` -> ``("tao_d1_data", 1)``. A bare name targets the first.
    """
    if "#" in key:
        name, _, suffix = key.rpartition("#")
        return name, int(suffix) - 1
    return key, 0


def _instantiate_tao_init(
    tao_init_spec: dict,
    override: dict | None,
    base_dir: pathlib.Path,
    in_to_out: dict[str, str],
    instance: str,
) -> dict[str, str]:
    """
    Render one instance's ``tao.init`` from the template spec.

    Rewrites ``design_lattice`` files to the instance's output lattice paths
    (via ``in_to_out``, the same map used for ``call`` targets) and applies any
    per-instance namelist add/update overrides.
    """
    input_rel = tao_init_spec["input"]
    output_rel = _interpolate(tao_init_spec["output"], instance)
    tao_init = TaoInit.from_file(base_dir / input_rel)

    in_dir = posixpath.dirname(input_rel)
    out_dir = posixpath.dirname(output_rel)
    remapped: list[str] = []
    changed = False
    for entry in tao_init.lattice_files:
        resolved_in = posixpath.normpath(posixpath.join(in_dir, entry))
        mapped = in_to_out.get(resolved_in)
        if mapped is None:
            remapped.append(entry)
        else:
            remapped.append(posixpath.relpath(mapped, out_dir or "."))
            changed = True
    if changed:
        tao_init.lattice_files = remapped

    for name_key, assignments in ((override or {}).get("namelists") or {}).items():
        name, index = _split_namelist_key(name_key)
        interpolated = {k: _interpolate(str(v), instance) for k, v in assignments.items()}
        tao_init.update_namelist(name, interpolated, index=index)

    return {output_rel: tao_init.render()}


def load_instances(path: pathlib.Path | str) -> dict:
    """Load an instances YAML/etc file into a plain dict."""
    return load_json_or_similar(path)


def instantiate(
    spec: dict,
    *,
    base_dir: pathlib.Path | str,
    options: FormatOptions | None = None,
) -> dict[str, dict[str, str]]:
    """
    Expand a template set across instances.

    Parameters
    ----------
    spec : dict
        Parsed instances file: ``template`` (list of ``{input, output}``),
        optional global ``renames``, optional ``delimiters`` (default delimiter
        set for prefix/suffix/parts), optional ``tao_init`` (``{input, output}``
        for a Tao ``tao.init`` whose ``design_lattice`` files are rewritten to
        the generated lattices), and ``instances`` (name -> overrides). A
        per-instance ``tao_init.namelists`` override (``{namelist: {key: value}}``,
        with a ``name#N`` suffix to target the N-th repeated group) adds or
        updates namelist sections.
    base_dir : pathlib.Path | str
        Directory the ``input`` paths are relative to.
    options : FormatOptions, optional
        Formatting options for emitted files.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{instance_name: {output_basename: formatted_contents}}``.
    """
    from .output import default_options, format_statements

    base_dir = pathlib.Path(base_dir)
    options = options or default_options

    template_files = spec["template"]
    default_delims = _delim_set(spec.get("delimiters"))
    global_rules = _normalize_renames(spec.get("renames"), default_delims)
    instances = spec["instances"]
    context_files = spec.get("context") or []
    tao_init_spec = spec.get("tao_init")

    contents = {
        (base_dir / tf["input"]).resolve(): (base_dir / tf["input"]).read_text()
        for tf in template_files
    }
    input_basenames = {(base_dir / tf["input"]).resolve(): tf["input"] for tf in template_files}
    output_patterns = {(base_dir / tf["input"]).resolve(): tf["output"] for tf in template_files}

    # Resolution-only files: loaded so cross-file references resolve, but never
    # transformed or written, and calls to them are left untouched.
    context_contents = {(base_dir / c).resolve(): (base_dir / c).read_text() for c in context_files}

    results: dict[str, dict[str, str]] = {}
    for name, overrides in instances.items():
        overrides = overrides or {}
        # The whole set (transform + context) is loaded as top-level files, so
        # name resolution is complete across all of them at once.
        files = MemoryFiles.from_mapping({**contents, **context_contents})
        files.parse(recurse=False)
        files.annotate()

        if overrides.get("values"):
            apply_values(files, overrides["values"])
        if overrides.get("insert"):
            _apply_inserts(files, overrides["insert"], options)
        files.annotate()

        rules = _merge_rulesets(
            global_rules, _normalize_renames(overrides.get("renames"), default_delims)
        )
        # Replace {instance} and similar in the rename rules
        rules = _interpolate_ruleset(rules, name)
        if rules["literal"] or rules["regex"] or rules["parts"]:
            _apply_renames(files, rules)

        # Full output path per input (may include directories). Rewrite `call`
        # targets by resolving each against the transform set's input paths.
        output_paths = {p: _interpolate(output_patterns[p], name) for p in contents}
        transform_paths = {p: (input_basenames[p], output_paths[p]) for p in contents}
        in_to_out = {posixpath.normpath(input_basenames[p]): output_paths[p] for p in contents}
        _rewrite_call_filenames(files, transform_paths, in_to_out)

        instance_files = {
            output_paths[p]: format_statements(files.by_filename[p], options) for p in contents
        }
        if tao_init_spec:
            instance_files.update(
                _instantiate_tao_init(
                    tao_init_spec, overrides.get("tao_init"), base_dir, in_to_out, name
                )
            )
        results[name] = instance_files

    return results


def write_instances(
    results: dict[str, dict[str, str]],
    output_dir: pathlib.Path | str,
) -> list[pathlib.Path]:
    """
    Write instantiated files to disk, creating parent directories as needed.

    Parameters
    ----------
    results : dict[str, dict[str, str]]
        The mapping returned by :func:`instantiate`
        (``{instance: {output_path: contents}}``).
    output_dir : pathlib.Path | str
        Base directory; each output path is resolved relative to it.

    Returns
    -------
    list[pathlib.Path]
        The paths written.
    """
    output_dir = pathlib.Path(output_dir)
    written: list[pathlib.Path] = []
    for files in results.values():
        for rel_path, contents in files.items():
            path = output_dir / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
            written.append(path)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_DESCRIPTION = """\
Interpolate and instantiate Bmad lattice templates.

The template is valid Bmad; a YAML sidecar supplies per-element/constant values
and (for instancing) renames and per-file output paths. See docs/templating.md.
"""


def _cmd_interpolate(parsed) -> None:
    from .output import default_options

    contents = pathlib.Path(parsed.template).read_text()
    values = load_json_or_similar(parsed.values) if parsed.values else None
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
        options=default_options,
    )
    if parsed.output:
        pathlib.Path(parsed.output).write_text(result)
        print(f"wrote: {parsed.output}")
    else:
        sys.stdout.write(result)


def _cmd_instantiate(parsed) -> None:
    from .output import default_options

    spec = load_json_or_similar(parsed.instances)
    base_dir = pathlib.Path(parsed.instances).resolve().parent
    results = instantiate(spec, base_dir=base_dir, options=default_options)

    output_dir = pathlib.Path(parsed.output_dir)
    if parsed.dry_run:
        for files in results.values():
            for rel_path in files:
                print(f"would write: {output_dir / rel_path}")
        return

    for path in write_instances(results, output_dir):
        print(f"wrote: {path}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="latform-template",
        description=_DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--log",
        "-L",
        dest="log_level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "CRITICAL"),
        help="Python logging level",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_interp = sub.add_parser("interpolate", help="Apply values/renames to a single template file")
    p_interp.add_argument("template", help="Template file (valid Bmad)")
    p_interp.add_argument("--values", help="YAML/JSON/TOML file of element/constant overrides")
    p_interp.add_argument(
        "--rename",
        nargs=2,
        metavar=("OLD", "NEW"),
        action="append",
        default=[],
        help="Rename rule (literal or regex); repeatable",
    )
    p_interp.add_argument(
        "--prefix",
        nargs=2,
        metavar=("FROM", "TO"),
        action="append",
        default=[],
        help="Leading-prefix rename (bounded by . or _); repeatable",
    )
    p_interp.add_argument(
        "--suffix",
        nargs=2,
        metavar=("FROM", "TO"),
        action="append",
        default=[],
        help="Trailing-suffix rename (bounded by . or _); repeatable",
    )
    p_interp.add_argument(
        "--parts",
        nargs=3,
        metavar=("DELIMS", "FROM", "TO"),
        action="append",
        default=[],
        help="Segment rename: rename whole DELIMS-separated parts equal to FROM; repeatable",
    )
    p_interp.add_argument(
        "--delimiters",
        default=None,
        help="Default delimiter set for --prefix/--suffix (default: . and _)",
    )
    p_interp.add_argument("-o", "--output", help="Output file (default: stdout)")
    p_interp.set_defaults(func=_cmd_interpolate)

    p_inst = sub.add_parser("instantiate", help="Expand a template set across instances")
    p_inst.add_argument("instances", help="Instances YAML file")
    p_inst.add_argument(
        "-d",
        "--output-dir",
        default=".",
        help="Base directory for generated files (default: current directory)",
    )
    p_inst.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; list what would be written",
    )
    p_inst.set_defaults(func=_cmd_instantiate)

    parsed = parser.parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=parsed.log_level)
    logging.getLogger("latform").setLevel(parsed.log_level)
    parsed.func(parsed)


def cli_main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ``latform-template``."""
    main(argv)


if __name__ == "__main__":
    cli_main()
