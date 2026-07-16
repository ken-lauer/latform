"""
Expand a Bmad lattice template set across instances.

Builds on :mod:`latform.apply` (single-file value/rename application) to render a
whole template set once per instance, rewriting ``call`` targets and Tao
``tao.init`` ``design_lattice`` paths to the generated files.
"""

from __future__ import annotations

import pathlib
import posixpath
import re

from .apply import (
    add_logging_argument,
    apply_renames,
    apply_values,
    configure_logging,
    delim_set,
    merge_rulesets,
    normalize_renames,
    split_namelist_key,
)
from .parser import MemoryFiles, parse
from .tao import TaoInit
from .token import Role, Token
from .types import FormatOptions
from .util import load_json_or_similar
from .walk import walk

__all__ = [
    "instantiate",
    "load_instances",
    "write_instances",
    "main",
    "cli_main",
]


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


def _interpolate_ruleset(rules: dict, instance: str) -> dict:
    return {
        "literal": {k: _interpolate(v, instance) for k, v in rules["literal"].items()},
        "regex": [(pat, _interpolate(repl, instance)) for pat, repl in rules["regex"]],
        "parts": [(d, frm, _interpolate(to, instance)) for d, frm, to in rules["parts"]],
    }


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
        name, index = split_namelist_key(name_key)
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
    default_delims = delim_set(spec.get("delimiters"))
    global_rules = normalize_renames(spec.get("renames"), default_delims)
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

        rules = merge_rulesets(
            global_rules, normalize_renames(overrides.get("renames"), default_delims)
        )
        # Replace {instance} and similar in the rename rules
        rules = _interpolate_ruleset(rules, name)
        if rules["literal"] or rules["regex"] or rules["parts"]:
            apply_renames(files, rules)

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

_INSTANTIATE_DESCRIPTION = """\
Expand a Bmad lattice template set across instances.

A YAML sidecar lists the template files and per-instance values, renames, and
output paths. See docs/cli.md.
"""


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
    import sys

    parser = argparse.ArgumentParser(
        prog="latform-template",
        description=_INSTANTIATE_DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_logging_argument(parser)
    parser.add_argument("instances", help="Instances YAML file")
    parser.add_argument(
        "-d",
        "--output-dir",
        default=".",
        help="Base directory for generated files (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; list what would be written",
    )

    parsed = parser.parse_args(argv if argv is not None else sys.argv[1:])
    configure_logging(parsed.log_level)
    _cmd_instantiate(parsed)


def cli_main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ``latform-template`` (instantiate a template set)."""
    main(argv)


if __name__ == "__main__":
    cli_main()
