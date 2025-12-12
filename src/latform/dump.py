"""
`latform-dump` - dump lattice information.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from latform.location import Location

from .parser import Files, MemoryFiles
from .statements import Element, ElementList, Line, Parameter, Simple

DESCRIPTION = __doc__
logger = logging.getLogger(__name__)


def dump_glob_params(files: Files, information: list[str]) -> str:
    from .output import FormatOptions, format_nodes

    params = [
        st
        for statements in files.by_filename.values()
        for st in statements
        if isinstance(st, Parameter)
    ]

    opts = FormatOptions()
    for parm in params:
        name = format_nodes([parm.target])[0].render(opts)
        if "%" in name or "*" in name:
            print(name, parm.target.loc)


def dump(files: Files, information: list[str]) -> str:
    lines = []

    named_items = files.get_named_items()

    line_eles = {item.name.upper(): item for item in named_items.values() if isinstance(item, Line)}

    use_lines = [
        st.arguments[0].name
        for statements in files.by_filename.values()
        for st in statements
        if isinstance(st, Simple) and st.statement == "use"
    ]

    if not use_lines:
        print("No use lines")
        return

    used_eles = []

    def add_used_line(line: Line):
        for ele in line.elements.items:
            if ele in line_eles:
                add_used_line(line_eles[ele])
            else:
                used_eles.append(ele)

    for use in use_lines:
        add_used_line(line_eles[use])

    for name, item in named_items.items():
        if isinstance(item, (Line, Element, ElementList)):
            if isinstance(item, Element):
                key = item.keyword
            else:
                key = type(item).__name__

            assert name.loc is not None
            loc = Location(
                filename=name.loc.filename.relative_to(files.main.parent),
                line=name.loc.line,
                column=name.loc.column,
                end_line=name.loc.end_line,
                end_column=name.loc.end_column,
            )

            if name in used_eles:
                print(name, f"({key} @ {loc})")

    return "\n".join(lines)


def main(
    filename: str | pathlib.Path,
    verbose: int = 0,
    output: pathlib.Path | str | None = None,
    format: str = "text",
) -> None:
    is_stdin = str(filename) == "-"

    files: Files
    if is_stdin:
        contents = sys.stdin.read()
        files = MemoryFiles.from_contents(contents, root_path=pathlib.Path.cwd() / "stdin.lat")
        files.local_file_to_source_filename[files.main] = "<stdin>"
    else:
        files = Files(main=pathlib.Path(filename))

    files.parse(recurse=True)
    files.annotate()

    if output:
        dest_fn = output
    else:
        dest_fn = None

    to_write = dump(files, ["name"])

    if dest_fn:
        pathlib.Path(dest_fn).write_text(to_write)
    elif to_write:
        print(to_write)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latform",
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    from ._version import __version__ as package_version

    parser.add_argument(
        "filename",
        help="Filename to parse (use '-' for stdin/standard input)",
        nargs="+",
    )

    parser.add_argument(
        "--output",
        "-o",
        action="store",
        help="Write to this filename",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["text", "mermaid"],
        default="text",
        help="Output format",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase debug verbosity",
    )

    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=package_version,
        help="Show the latform version number and exit.",
    )

    parser.add_argument(
        "--log",
        "-L",
        dest="log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "CRITICAL"),
        help="Python logging level (e.g. DEBUG, INFO, WARNING)",
    )

    return parser


def cli_main(args: list[str] | None = None) -> None:
    """
    CLI entrypoint for latform-dump.

    Parameters
    ----------
    args : list of str, optional
        Command-line arguments to parse and pass to :func:`main()`.
    """
    parsed = _build_argparser().parse_args(args=args)
    kwargs = vars(parsed)
    log_level = kwargs.pop("log_level")

    # Adjust the package-level logger level as requested:
    logger = logging.getLogger("latform")
    logger.setLevel(log_level)
    logging.basicConfig()

    filenames = kwargs.pop("filename")
    for filename in filenames:
        if len(filename) > 1:
            logger.info("Processing %s", filename)
        main(filename=filename, **kwargs)


if __name__ == "__main__":
    cli_main()
