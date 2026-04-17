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
from .parser import Files, MemoryFiles
from .statements import (
    Attribute,
    Element,
    ElementList,
    Line,
    Parameter,
    Simple,
    Statement,
)
from .token import Token
from .types import NameCase

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


def _resolve_used_elements(files: Files, named_items: dict[Token, Statement]) -> set[str]:
    """
    Traverse from USE statements to find all effectively used elements.

    Returns
    -------
    set[str]
        A set of element names (uppercased) that are active in the lattice.
    """

    use_cmds = [
        st
        for statements in files.by_filename.values()
        for st in statements
        if isinstance(st, Simple) and st.statement.lower() == "use"
    ]

    # Roots are the arguments to USE commands
    roots: list[str] = []
    for cmd in use_cmds:
        roots.append(_fmt(cmd.arguments[0]))

    used_names: set[str] = set()
    visited_lines: set[str] = set()

    def visit(name: Token | str):
        if name in used_names:
            return

        # Lines themselves are used
        used_names.add(name)

        item = named_items.get(name)
        if not item:
            return  # Referenced but no definition found

        if isinstance(item, Line):
            if name in visited_lines:
                return
            visited_lines.add(name)

            for ele_token in item.elements.items:
                if isinstance(ele_token, Token):
                    visit(ele_token.upper())

        elif isinstance(item, (Element, ElementList)):
            pass

    for root in roots:
        visit(root)

    for name, item in named_items.items():
        if isinstance(item, Element):
            keyword = item.keyword.upper()
            if keyword in named_items:
                used_names.add(keyword)
                parent = named_items[keyword]
                keyword = parent.keyword.upper()

            if keyword in {"OVERLAY"}:
                used_names.add(name)
            if name in {"BEGINNING", "END"}:
                used_names.add(name)
            for attr in item.attributes:
                if isinstance(attr, Attribute) and _fmt(attr.name).lower() == "superimpose":
                    used_names.add(name)
                    break

    return used_names


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


def get_elements_status(
    files: Files, filter_status: Literal["all", "used", "unused"] = "all"
) -> Iterable[dict[str, Any]]:
    """
    Generate simplified dictionaries for elements based on usage status.
    """

    named_items = files.get_named_items()
    used_names = _resolve_used_elements(files, named_items)

    definitions = {
        name: item
        for name, item in named_items.items()
        if isinstance(item, (Line, Element, ElementList))
    }

    for name_upper, item in definitions.items():
        is_used = name_upper in used_names

        if filter_status == "used" and not is_used:
            continue
        if filter_status == "unused" and is_used:
            continue

        row = {
            "name": name_upper,
            "type": "",
            "parent": "",
            "used": "YES" if is_used else "NO",
            "loc_obj": None,
        }
        row["loc_obj"] = item.name.loc

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
    delimiter = delimiter
    root_path = root_path

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


def _load_files_and_parse(filename: str, root_path: pathlib.Path, verbose: int) -> Files:
    """Helper to handle file loading and parsing errors."""

    is_stdin = filename == "-"

    if is_stdin:
        contents = sys.stdin.read()
        files = MemoryFiles.from_contents(contents, root_path=root_path / "stdin.lat")
        files.local_file_to_source_filename[files.main] = "<stdin>"
    else:
        files = Files(main=pathlib.Path(filename))

    try:
        files.parse(recurse=True)
        files.annotate()
    except Exception as e:
        if verbose > 0:
            logger.exception("Parsing failed")
        else:
            logger.error(f"Parsing failed: {e}")
        sys.exit(1)

    return files


def cmd_parameters(args: argparse.Namespace, files: Files):
    data = []
    headers = ["name", "expression", "loc_obj"]

    for item in get_parameters(files):
        if not _passes_filter(item["name"], args.match, args.match_re):
            continue
        data.append(item)

    return data, headers


def cmd_used_elements(args: argparse.Namespace, files: Files):
    data = []
    headers = ["name", "type", "parent", "loc_obj"]

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


def cmd_all(args: argparse.Namespace, files: Files):
    """Legacy dump behavior."""
    if not args.delimiter:
        print("--- Parameters ---")
    cmd_parameters(args, files)

    if not args.delimiter:
        print("\n--- Used Elements ---")
    cmd_used_elements(args, files)

    if not args.delimiter:
        print("\n--- Unused Elements ---")
    cmd_unused_elements(args, files)


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
        or parsed_args.dump_used_elements
        or parsed_args.dump_unused_elements
        or parsed_args.dump_loaded_files
    )

    if not any_dump_flag:
        parsed_args.dump_parameters = True
        parsed_args.dump_used_elements = True
        parsed_args.dump_unused_elements = True
        parsed_args.dump_loaded_files = True

    all_files: list[Files] = []
    for fn in parsed_args.filename:
        files = _load_files_and_parse(fn, pathlib.Path.cwd(), parsed_args.verbose)
        all_files.append(files)

        if parsed_args.dump_parameters:
            if not any_dump_flag and not parsed_args.delimiter:
                print("--- Parameters ---")

            data, headers = cmd_parameters(parsed_args, files)
            print_data(data, headers, delimiter=parsed_args.delimiter, root_path=files.main.parent)

        if parsed_args.dump_used_elements:
            if not any_dump_flag and not parsed_args.delimiter:
                print("\n--- Used Elements ---")
            data, headers = cmd_used_elements(parsed_args, files)
            print_data(data, headers, delimiter=parsed_args.delimiter, root_path=files.main.parent)

        if parsed_args.dump_unused_elements:
            if not any_dump_flag and not parsed_args.delimiter:
                print("\n--- Unused Elements ---")
            data, headers = cmd_unused_elements(parsed_args, files)
            print_data(data, headers, delimiter=parsed_args.delimiter, root_path=files.main.parent)

    if not any_dump_flag and not parsed_args.delimiter:
        print("\n--- All loaded files ---")
    if parsed_args.dump_loaded_files:
        res = cmd_loaded_files(parsed_args, all_files)
        for fn in res:
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
