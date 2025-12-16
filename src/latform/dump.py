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

from latform.location import Location

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

DESCRIPTION = __doc__
logger = logging.getLogger(__name__)


def _fmt(obj):
    from .output import FormatOptions, format_nodes

    opts = FormatOptions()
    if not isinstance(obj, list):
        obj = [obj]
    return "\n".join(line.render(opts) for line in format_nodes(obj))


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
            "name": f"{target}[{name}]",
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
            table.add_row(*d_row)

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
    headers = ["expression", "loc_obj"]

    for item in get_parameters(files):
        if not _passes_filter(item["name"], args.match, args.match_re):
            continue
        data.append(item)

    print_data(data, headers, delimiter=args.delimiter, root_path=files.main.parent)


def cmd_used_elements(args: argparse.Namespace, files: Files):
    data = []
    headers = ["name", "type", "parent", "loc_obj"]

    for item in get_elements_status(files, filter_status="used"):
        if not _passes_filter(item["name"], args.match, args.match_re):
            continue
        data.append(item)

    print_data(data, headers, delimiter=args.delimiter, root_path=files.main.parent)


def cmd_unused_elements(args: argparse.Namespace, files: Files):
    data = []
    headers = ["name", "type", "loc_obj"]

    for item in get_elements_status(files, filter_status="unused"):
        if not _passes_filter(item["name"], args.match, args.match_re):
            continue
        data.append(item)

    print_data(data, headers, delimiter=args.delimiter, root_path=files.main.parent)


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

    subparsers = parser.add_subparsers(dest="command", help="Information to dump")

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "filename",
        help="Filename to parse (use '-' for stdin)",
        nargs="+",
    )
    parent_parser.add_argument(
        "--delimiter",
        "-d",
        help="Use specified delimiter (e.g. ',') instead of formatted table. Useful for machine parsing.",
        default=None,
    )
    parent_parser.add_argument(
        "--match",
        "-m",
        help="Glob pattern to filter names (e.g. 'qf*')",
        default=None,
    )
    parent_parser.add_argument(
        "--match-re", "-r", help="Regex pattern to filter names", default=None
    )

    sp_params = subparsers.add_parser(
        "parameters", parents=[parent_parser], help="Dump defined parameters/variables"
    )
    sp_params.set_defaults(func=cmd_parameters)

    sp_used = subparsers.add_parser(
        "used-elements",
        parents=[parent_parser],
        help="Dump defined and used elements (in lines, etc.)",
    )
    sp_used.set_defaults(func=cmd_used_elements)

    sp_unused = subparsers.add_parser(
        "unused-elements",
        parents=[parent_parser],
        help="Dump defined elements not used",
    )
    sp_unused.set_defaults(func=cmd_unused_elements)

    sp_all = subparsers.add_parser("all", parents=[parent_parser], help="Dump everything (default)")
    sp_all.set_defaults(func=cmd_all)

    if args is None:
        raw_args = sys.argv[1:]
    else:
        raw_args = args

    if not raw_args:
        parser.print_help()
        sys.exit(0)

    known_commands = {
        "parameters",
        "used-elements",
        "unused-elements",
        "all",
        "-h",
        "--help",
        "--version",
    }
    if raw_args and raw_args[0] not in known_commands and not raw_args[0].startswith("-"):
        parsed_args = parser.parse_args(["all"] + raw_args)
    else:
        parsed_args = parser.parse_args(raw_args)

    logging.basicConfig(level=parsed_args.log_level)
    logger_inst = logging.getLogger("latform")
    logger_inst.setLevel(parsed_args.log_level)

    if not hasattr(parsed_args, "func"):
        parser.print_help()
        sys.exit(1)

    if parsed_args.delimiter:
        parsed_args.delimiter = parsed_args.delimiter.replace("\\t", "\t")

    for fn in parsed_args.filename:
        files = _load_files_and_parse(fn, pathlib.Path.cwd(), parsed_args.verbose)
        parsed_args.func(parsed_args, files)


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
