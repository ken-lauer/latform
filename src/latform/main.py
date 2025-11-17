"""
`latform` - a Bmad lattice parser/formatter tool.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import rich

from .output import format_statements
from .parser import parse, parse_file_recursive
from .tokenizer import Tokenizer
from .types import FormatOptions

DESCRIPTION = __doc__
logger = logging.getLogger(__name__)


def main(
    filename: str | pathlib.Path,
    verbose: int = 0,
    line_length: int = 100,
    compact: bool = False,
    follow_call: bool = False,
    in_place: bool = False,
    output: pathlib.Path | str | None = None,
) -> None:
    if str(filename) == "-":
        contents = sys.stdin.read()
        filename = "<stdin>"
        is_stdin = True
    else:
        filename = pathlib.Path(filename)
        contents = filename.read_text()
        is_stdin = False

    options = FormatOptions(
        line_length=line_length,
        compact=compact,
        indent_size=2,
        indent_char=" ",
        comment_col=40,
        newline_before_new_type=not compact,
    )
    if follow_call:
        if is_stdin:
            raise NotImplementedError(
                "Recursive parsing using a lattice from stdin is not yet supported"
            )

        files = parse_file_recursive(filename)
        if verbose > 1:
            rich.print(files.by_filename)
        if in_place:
            files.reformat(options)
        return

    if verbose > 0:
        tok = Tokenizer(contents, filename=filename)
        blocks = tok.split_blocks()
        stacked = [block.stack() for block in blocks]
        statements = []
        for idx, block in enumerate(stacked):
            if idx > 0:
                rich.print()
            rich.print(f"-- Block {idx} ({block.loc})", file=sys.stderr)
            if verbose > 2:
                rich.print("Original source:", file=sys.stderr)
                rich.print("```", file=sys.stderr)
                rich.print(block.loc.get_string(contents), file=sys.stderr)
                rich.print("```", file=sys.stderr)
            if verbose > 1:
                rich.print(block, file=sys.stderr)
            statement = block.parse()
            statements.append(statement)
            rich.print(statement, file=sys.stderr)

    else:
        statements = parse(contents=contents, filename=filename)

    formatted = format_statements(statements, options=options)
    if output:
        dest_fn = output
    elif in_place and not is_stdin:
        dest_fn = filename
    else:
        dest_fn = None

    if dest_fn:
        pathlib.Path(dest_fn).write_text(formatted)
    else:
        print(formatted)


def _build_argparser() -> argparse.ArgumentParser:
    """
    This function builds an argument parser for your command-line application.

    For help, see the :mod:`argparse` documentation.
    """
    parser = argparse.ArgumentParser(
        prog="latform",
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    from ._version import __version__ as package_version

    parser.add_argument(
        "filename",
        help="Filename to parse (use '-' for stdin/standard input)",
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        default=False,
        help="Compact output mode",
    )

    parser.add_argument(
        "--in-place",
        "-i",
        action="store_true",
        help="Overwrite file(s) with formatted output instead of printing to standard output",
    )
    parser.add_argument(
        "--output",
        "-o",
        action="store_true",
        help="Write to this filename (or directory, if multiple files)",
    )

    parser.add_argument(
        "--line-length",
        "-l",
        type=int,
        default=100,
        help="Approximate max line length",
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
        "--follow-call",
        action="store_true",
        help="Recursively parse lattice files, following call statements",
    )

    parser.add_argument(
        "--log",
        "-L",
        dest="log_level",
        default="INFO",
        type=str,
        help="Python logging level (e.g. DEBUG, INFO, WARNING)",
    )

    return parser


def cli_main(args: list[str] | None = None) -> None:
    """
    CLI entrypoint main.

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
    return main(**kwargs)


if __name__ == "__main__":
    cli_main()
