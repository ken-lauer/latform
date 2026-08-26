"""
`latform-dump` - dump lattice information.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import logging
import pathlib
import re
import sys
from collections.abc import Iterable
from io import StringIO
from typing import Any, Literal

from rich.console import Console
from rich.table import Table

from .location import Location
from .parser import Files, build_files, implicit_location, match_element_selector
from .statements import (
    Constant,
    Element,
    ElementList,
    Line,
    Parameter,
    Simple,
    Statement,
    _attribute_value_text,
    get_deferred_element_attributes,
    normalize_element_selector,
)
from .token import Token
from .types import Attribute, NameCase, Seq

DESCRIPTION = __doc__
logger = logging.getLogger(__name__)


def _fmt(
    obj,
    line_length: int = 100,
    max_line_length: int = 130,
    compact: bool = False,
    indent_size: int = 2,
    indent_char: str = " ",
    comment_col: int = 40,
    newline_before_new_type: bool = False,
    newline_between_lines: bool = True,
    trailing_comma: bool = False,
    statement_comma_threshold_for_multiline: int = 8,
    name_case: NameCase = "upper",
    attribute_case: NameCase = "lower",
    kind_case: NameCase = "lower",
    builtin_case: NameCase = "lower",
    section_break_character: str = "-",
    section_break_width: int | None = None,
    flatten_call: bool = False,
    flatten_inline: bool = False,
    newline_at_eof: bool = True,
    strip_comments: bool = False,
):
    from .output import FormatOptions, format_nodes

    opts = FormatOptions(
        line_length=line_length,
        max_line_length=max_line_length,
        compact=compact,
        indent_size=indent_size,
        indent_char=indent_char,
        comment_col=comment_col,
        newline_before_new_type=newline_before_new_type,
        newline_between_lines=newline_between_lines,
        trailing_comma=trailing_comma,
        statement_comma_threshold_for_multiline=statement_comma_threshold_for_multiline,
        name_case=name_case,
        attribute_case=attribute_case,
        kind_case=kind_case,
        builtin_case=builtin_case,
        section_break_character=section_break_character,
        section_break_width=section_break_width,
        flatten_call=flatten_call,
        flatten_inline=flatten_inline,
        newline_at_eof=newline_at_eof,
        strip_comments=strip_comments,
    )
    if not isinstance(obj, list):
        obj = [obj]
    return "\n".join(line.render(opts) for line in format_nodes(obj, opts))


def _fmt_loc(loc: Location | None, root_path: pathlib.Path | None = None) -> str:
    """Format a location object for tabular output."""
    if not loc:
        return ""

    filename = loc.filename
    if root_path and filename and filename.is_absolute():
        try:
            filename = filename.relative_to(root_path)
        except ValueError:
            pass  # Not relative

    return f"{filename}:{loc.line}"


def _passes_filter(name: str, glob_pat: str | None, re_pat: str | None) -> bool:
    """Determine if a name matches the requested filters."""
    if glob_pat and not fnmatch.fnmatch(name, glob_pat):
        return False
    if re_pat and not re.search(re_pat, name):
        return False
    return True


_FORK_TYPES_UPPER = frozenset({"FORK", "PHOTON_FORK"})
_SLAVE_CONTROLLER_TYPES_UPPER = frozenset({"OVERLAY", "GROUP", "RAMPER", "GIRDER"})


def _iter_seq_element_names(seq: Seq) -> Iterable[Token]:
    """
    Name tokens within a line/list definition sequence.

    Handles nested entries like ``2*qf`` (repetition), ``-sub`` (reflection),
    and ``sub(x)`` (replacement line calls); repeat counts and delimiters are
    skipped.
    """
    for item in seq.items:
        if isinstance(item, Token):
            # Names cannot start with a digit, so this skips repeat counts
            if item and not item[0].isdigit():
                yield item
        elif isinstance(item, Seq):
            yield from _iter_seq_element_names(item)


def _match_selector_names(
    selector: str, named_items: dict[Token, Statement]
) -> frozenset[str] | None:
    """
    Uppercased names of defined elements matched by a Bmad element selector.

    Returns None if the selector syntax is not supported.
    """
    matched = match_element_selector(named_items.values(), normalize_element_selector(selector))
    if matched is None:
        return None
    return frozenset(str(el.name.upper()) for el in matched)


def _controller_slave_selectors(element: Element) -> list[str]:
    """Slave element selectors from a controller's ``{...}`` element list."""
    ele_list = element.ele_list
    base = element.base_element
    while ele_list is None and base is not None:
        # Inherited controllers (``ov2: ov1``) take their slaves from the base
        ele_list = base.ele_list
        base = base.base_element
    if ele_list is None:
        return []
    selectors = []
    for item in ele_list.items:
        if isinstance(item, Token):
            # Girder-style member list
            selectors.append(str(item))
        elif isinstance(item, Seq):
            # Control spec such as ``q*[k1]: 0.1`` -- the slave selector is
            # everything before the ``[attr]`` part (wildcards tokenize as
            # separate delimiters, so the leading pieces are joined back up)
            parts: list[str] = []
            for sub in item.items:
                if isinstance(sub, Seq) or str(sub) == ":":
                    break
                parts.append(str(sub))
            if parts:
                selectors.append("".join(parts))
    return selectors


def _first_used_slave(
    element: Element, named_items: dict[Token, Statement], used: dict[Token, str]
) -> str | None:
    """The name of one used slave of a controller, if any."""
    for selector in _controller_slave_selectors(element):
        names = _match_selector_names(selector, named_items)
        if names is None:
            # Unsupported selector syntax; assume in use rather than
            # incorrectly reporting the controller as unused.
            return selector
        for match in names:
            if match in used:
                return match
    return None


def _fork_targets(
    element: Element, deferred: dict[Token, dict[str, Token | Seq | None]]
) -> list[Token]:
    """Uppercased ``to_line``/``to_element`` targets of a fork element."""
    values: dict[str, Token | Seq | None] = {}
    for attr in element.attributes:
        if isinstance(attr.name, Token) and attr.name._upper in {"TO_LINE", "TO_ELEMENT"}:
            values[attr.name._upper] = attr.value
    for uname, value in deferred.get(element.name.upper(), {}).items():
        if uname in {"TO_LINE", "TO_ELEMENT"}:
            values[uname] = value

    targets = []
    for value in values.values():
        text = _attribute_value_text(value)
        if text:
            targets.append(text.upper())
    return targets


def _resolve_used_elements(
    files: Files,
    named_items: dict[Token, Statement],
) -> dict[str, str]:
    """
    Resolve which named items are active in the expanded lattice.

    Roots are resolved per top-level lattice file: the last USE statement in
    each file's call tree, or the ``@use_line`` names from the corresponding
    ``tao.init`` ``design_lattice(i)%file`` entry, which override that
    lattice's own USE statement.  From those roots, lines are expanded
    recursively (including repetitions, reflections, replacement-line calls,
    list members, and fork targets).  A fixpoint pass then folds in usage that
    depends on other elements being used:

    - superimposed elements, used when superposition is enabled and their
      ``ref`` matches a used element (or defaults to the beginning marker);
    - controllers (overlay/group/ramper/girder), used when at least one of
      their slaves is used;
    - base elements of used elements (``qd: qf`` marks ``qf`` used).

    Returns
    -------
    dict[str, str]
        Mapping of uppercased active names to a human-readable reason.
    """
    all_statements = files.get_statements_in_order(repeat_called_files=False)

    def last_use_roots(statements: list[Statement]) -> list[Token]:
        # Bmad semantics: the last USE statement wins; each of its arguments is
        # the root line of a branch.  Bare argument tokens are parsed as
        # value-less Attributes.
        use_cmds = [
            st
            for st in statements
            if isinstance(st, Simple) and st.statement._upper == "USE" and st.arguments
        ]
        roots: list[Token] = []
        for use_cmd in reversed(use_cmds):
            for arg in use_cmd.arguments:
                if isinstance(arg, Token):
                    roots.append(arg.upper())
                elif (
                    isinstance(arg, Attribute) and isinstance(arg.name, Token) and arg.value is None
                ):
                    roots.append(arg.name.upper())

            # TODO: maybe allow unused lines to be considered used (along with
            # the elements they contain)
            # I think this is a scenario that's common in reused sublattices
            # that can be standalone
            break
        return roots

    # Roots are resolved per top-level lattice file.  A tao.init
    # ``design_lattice(i)%file = 'lat.bmad@line_name'`` suffix overrides that
    # lattice's own USE statement (Tao/bmad_parser semantics); lattices without
    # a suffix fall back to their in-file USE.
    tao_entries: list[tuple[str, list[str]]] = (
        files.tao_init.lattice_file_with_use_line if files.tao_init is not None else []
    )
    root_reasons: dict[Token, str] = {}
    for index, top_file in enumerate(files.top_files):
        use_lines = tao_entries[index][1] if index < len(tao_entries) else []
        if use_lines:
            for name in use_lines:
                root_reasons.setdefault(Token(name.upper()), "tao.init use_line")
        else:
            statements = files.get_statements_in_order(
                repeat_called_files=False, top_files=[top_file]
            )
            for root in last_use_roots(statements):
                root_reasons.setdefault(root, "use statement")

    deferred = get_deferred_element_attributes(all_statements)
    used: dict[Token, str] = {}

    def mark(name: Token, reason: str) -> None:
        if name in used:
            return
        used[name] = reason

        item = named_items.get(name)
        if item is None:
            return  # Referenced but no definition found

        if isinstance(item, Line):
            for token in _iter_seq_element_names(item.elements):
                mark(token.upper(), f"in line {name}")
        elif isinstance(item, ElementList):
            for token in _iter_seq_element_names(item.elements):
                mark(token.upper(), f"in list {name}")
        elif isinstance(item, Element):
            if (item.element_type or "") in _FORK_TYPES_UPPER:
                for target in _fork_targets(item, deferred):
                    mark(target, f"fork target of {name}")

    for root, reason in root_reasons.items():
        mark(root, reason)

    if root_reasons:
        # The implicit lattice endpoints exist in any expanded lattice
        for name in (Token("BEGINNING"), Token("END")):
            if name in named_items:
                mark(name, "lattice endpoint")

    element_defs = {name: item for name, item in named_items.items() if isinstance(item, Element)}

    changed = True
    while changed:
        changed = False
        for name, element in element_defs.items():
            if name in used:
                base_name = element.keyword.upper()
                base = named_items.get(base_name)
                if isinstance(base, Element) and base_name not in used:
                    mark(base_name, f"base of {name}")
                    changed = True
                continue

            resolved_type = element.element_type or element.keyword.upper()
            if resolved_type in _SLAVE_CONTROLLER_TYPES_UPPER:
                slave = _first_used_slave(element, named_items, used)
                if slave is not None:
                    mark(name, f"controls {slave}")
                    changed = True
                continue

            if not root_reasons:
                continue

            superposition = element.get_superposition_settings(deferred)
            if not superposition.enabled:
                continue
            if superposition.ref is None:
                mark(name, "superimposed (default ref)")
                changed = True
            else:
                matches = _match_selector_names(superposition.ref, named_items)
                if matches is None or any(match in used for match in matches):
                    mark(name, f"superimposed on {superposition.ref}")
                    changed = True

    return used


def get_parameters(files: Files) -> Iterable[dict[str, Any]]:
    params = [
        st
        for statements in files.by_filename.values()
        for st in statements
        if isinstance(st, Parameter)
    ]

    for parm in params:
        target = _fmt(parm.target)
        name = _fmt(parm.name)
        value = _fmt(parm.value)

        yield {
            "name": rf"{target}[{name}]",
            "expression": value,
            "filename": parm.target.loc.filename if parm.target.loc else "",
            "line": parm.target.loc.line if parm.target.loc else 0,
            "loc_obj": parm.target.loc,
        }


def get_constants(files: Files) -> Iterable[dict[str, Any]]:
    """
    Generate dictionaries describing constant definitions (``name = value``).
    """
    for statements in files.by_filename.values():
        for st in statements:
            if isinstance(st, Constant):
                yield {
                    "name": st.name,
                    "expression": _fmt(st.value),
                    "loc_obj": st.name.loc,
                }


def get_elements_status(
    files: Files, filter_status: Literal["all", "used", "unused"] = "all"
) -> Iterable[dict[str, Any]]:
    """
    Generate simplified dictionaries for elements based on usage status.
    """

    named_items = files.get_named_items()
    used_reasons = _resolve_used_elements(files, named_items)

    definitions = {
        name: item
        for name, item in named_items.items()
        if isinstance(item, (Line, Element, ElementList)) and item.name.loc != implicit_location
    }

    for name_upper, item in definitions.items():
        is_used = name_upper in used_reasons

        if filter_status == "used" and not is_used:
            continue
        if filter_status == "unused" and is_used:
            continue

        row = {
            "name": name_upper,
            "type": "",
            "parent": "",
            "used": "YES" if is_used else "NO",
            "reason": used_reasons.get(name_upper, ""),
            "loc_obj": item.name.loc,
        }

        if isinstance(item, Line):
            row["type"] = "LINE"
        elif isinstance(item, ElementList):
            row["type"] = "LIST"
        elif isinstance(item, Element):
            row["type"] = item.keyword.upper()
            if row["type"] in named_items:
                row["parent"] = row["type"]

        yield row


def print_data(
    data: list[dict[str, Any]],
    columns: list[str],
    delimiter: str | None = None,
    root_path: pathlib.Path | None = None,
    console: Console | None = None,
):
    display_rows = []
    headers = [c.capitalize() for c in columns if c != "loc_obj"]

    if "loc_obj" in columns:
        headers.append("Location")

    for row in data:
        new_row = []
        for col in columns:
            if col == "loc_obj":
                continue
            new_row.append(str(row.get(col, "")))

        if "loc_obj" in columns:
            new_row.append(_fmt_loc(row.get("loc_obj"), root_path))

        display_rows.append(new_row)

    if not display_rows:
        return

    if delimiter:
        s_io = StringIO()
        writer = csv.writer(s_io, delimiter=delimiter, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(display_rows)
        print(s_io.getvalue(), end="")

    else:
        table = Table(show_header=True, header_style="bold magenta")
        for h in headers:
            table.add_column(h)

        for d_row in display_rows:
            table.add_row(*[item.replace("[", r"\[") for item in d_row])

        console = console or Console()
        console.print(table)

        console.print(f"{len(display_rows)} matches.")


def _load_all_files_and_parse(
    filenames: list[str | pathlib.Path],
    root_path: pathlib.Path,
    verbose: int,
    combine: bool = False,
) -> list[Files]:
    """Build, parse, and annotate one or more :class:`Files` objects."""
    result = build_files(filenames, combine=combine, root_path=root_path)
    try:
        for files in result:
            files.parse(recurse=True)
            files.annotate()
    except Exception as e:
        if verbose > 0:
            logger.exception("Parsing failed")
        else:
            logger.error(f"Parsing failed: {e}")
        sys.exit(1)
    return result


def cmd_parameters(args: argparse.Namespace, files: Files):
    data = []
    headers = ["name", "expression", "loc_obj"]

    for item in get_parameters(files):
        if not _passes_filter(item["name"], args.match, args.match_re):
            continue
        data.append(item)

    return data, headers


def cmd_constants(args: argparse.Namespace, files: Files):
    data = []
    headers = ["name", "expression", "loc_obj"]

    for item in get_constants(files):
        if not _passes_filter(item["name"], args.match, args.match_re):
            continue
        data.append(item)

    return data, headers


def cmd_used_elements(args: argparse.Namespace, files: Files):
    data = []
    headers = ["name", "type", "parent", "reason", "loc_obj"]

    for item in get_elements_status(files, filter_status="used"):
        if not _passes_filter(item["name"], args.match, args.match_re):
            continue
        data.append(item)

    return data, headers


def cmd_unused_elements(args: argparse.Namespace, files: Files):
    data = []
    headers = ["name", "type", "loc_obj"]

    for item in get_elements_status(files, filter_status="unused"):
        if not _passes_filter(item["name"], args.match, args.match_re):
            continue
        data.append(item)

    return data, headers


def cmd_loaded_files(
    args: argparse.Namespace,
    all_files: list[Files],
    normalize_call: bool = True,
    # include_hdf5: bool = False,
):
    # Not using a set here to retain parsing order
    res = []
    for files in all_files:
        for fn in files.get_all_referenced_files():
            if fn not in res:
                res.append(fn)
    return res


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="latform-dump",
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    try:
        from ._version import __version__ as package_version
    except ImportError:
        package_version = "0.0.0"

    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=package_version,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase debug verbosity",
    )
    parser.add_argument(
        "--log",
        "-L",
        dest="log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "CRITICAL"),
        help="Python logging level",
    )
    parser.add_argument(
        "filename",
        help="Filename to parse (use '-' for stdin)",
        nargs="+",
    )
    parser.add_argument(
        "--delimiter",
        "-d",
        help="Use specified delimiter (e.g. ',') instead of formatted table. Useful for machine parsing.",
        default=None,
    )
    parser.add_argument(
        "--match",
        "-m",
        help="Glob pattern to filter names (e.g. 'qf*')",
        default=None,
    )
    parser.add_argument("--match-re", "-r", help="Regex pattern to filter names", default=None)

    # Dump options
    parser.add_argument(
        "-p",
        "--parameters",
        action="store_true",
        help="Dump defined parameters/variables",
        dest="dump_parameters",
    )
    parser.add_argument(
        "-c",
        "--constants",
        action="store_true",
        help="Dump defined constants (name = value)",
        dest="dump_constants",
    )
    parser.add_argument(
        "-U",
        "--used-elements",
        action="store_true",
        help="Dump defined and used elements (in lines, etc.)",
        dest="dump_used_elements",
    )
    parser.add_argument(
        "-u",
        "--unused-elements",
        action="store_true",
        help="Dump defined elements not used",
        dest="dump_unused_elements",
    )
    parser.add_argument(
        "-f",
        "--files",
        action="store_true",
        help="Dump loaded files",
        dest="dump_loaded_files",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help=(
            "Process all input files together as a single set, sharing one parse stack "
            "and one named-item namespace."
        ),
    )

    if args is None:
        raw_args = sys.argv[1:]
    else:
        raw_args = args

    if not raw_args:
        parser.print_help()
        sys.exit(0)

    parsed_args = parser.parse_args(raw_args)

    logging.basicConfig(level=parsed_args.log_level)
    logger_inst = logging.getLogger("latform")
    logger_inst.setLevel(parsed_args.log_level)

    if parsed_args.delimiter:
        parsed_args.delimiter = parsed_args.delimiter.replace("\\t", "\t")

    any_dump_flag = (
        parsed_args.dump_parameters
        or parsed_args.dump_constants
        or parsed_args.dump_used_elements
        or parsed_args.dump_unused_elements
        or parsed_args.dump_loaded_files
    )

    if not any_dump_flag:
        parsed_args.dump_parameters = True
        parsed_args.dump_constants = True
        parsed_args.dump_used_elements = True
        parsed_args.dump_unused_elements = True
        parsed_args.dump_loaded_files = True

    sections = [
        (parsed_args.dump_parameters, "Parameters", cmd_parameters),
        (parsed_args.dump_constants, "Constants", cmd_constants),
        (parsed_args.dump_used_elements, "Used Elements", cmd_used_elements),
        (parsed_args.dump_unused_elements, "Unused Elements", cmd_unused_elements),
    ]
    num_selected = sum(flag for flag, _, _ in sections) + bool(parsed_args.dump_loaded_files)
    show_headers = num_selected > 1 and not parsed_args.delimiter

    all_files = _load_all_files_and_parse(
        parsed_args.filename,
        pathlib.Path.cwd(),
        parsed_args.verbose,
        combine=parsed_args.combine,
    )
    first_section = True
    for files in all_files:
        root_path = files.top_files[0].parent

        for flag, title, cmd in sections:
            if not flag:
                continue
            if show_headers:
                if not first_section:
                    print()
                print(f"--- {title} ---")
            first_section = False
            data, headers = cmd(parsed_args, files)
            print_data(data, headers, delimiter=parsed_args.delimiter, root_path=root_path)

    if parsed_args.dump_loaded_files:
        if show_headers:
            if not first_section:
                print()
            print("--- All loaded files ---")
        for fn in cmd_loaded_files(parsed_args, all_files):
            print(fn)


def cli_main(args: list[str] | None = None) -> None:
    """
    CLI entrypoint for latform-dump.

    Parameters
    ----------
    args : list of str, optional
        Command-line arguments to parse and pass to :func:`main()`.
    """
    main(args)


if __name__ == "__main__":
    cli_main()
