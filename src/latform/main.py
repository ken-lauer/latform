"""
`latform` - a Bmad lattice parser/formatter tool.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import pathlib
import sys

import rich

from . import output as output_mod
from .lint import lint_statement
from .output import format_statements
from .parser import parse, parse_file_recursive
from .statements import Statement
from .tokenizer import Tokenizer
from .types import FormatOptions, NameCase

DESCRIPTION = __doc__
logger = logging.getLogger(__name__)


def load_renames(
    rename_file: pathlib.Path | str | None,
    raw_renames: list[str] | None,
    renames: dict[str, str] | None,
):
    res = {}

    lines = []
    if rename_file:
        # todo: csv reader, maybe
        lines.extend(
            [line.split(",") for line in pathlib.Path(rename_file).read_text().splitlines()]
        )

    if raw_renames:
        lines.extend([line.split(",") for line in raw_renames])

    for from_, to in lines:
        res[from_.strip()] = to.strip()

    if renames:
        res.update(renames)

    return res


def get_diff(
    original: str, formatted: str, fromfile: pathlib.Path | str, tofile: pathlib.Path | str
):
    udiff = difflib.unified_diff(
        original.splitlines(keepends=True),
        formatted.splitlines(keepends=True),
        fromfile=str(fromfile),
        tofile=str(tofile),
    )
    return "".join(udiff)


def process_file(
    contents: str,
    filename: str | pathlib.Path,
    verbose: int = 0,
) -> list[Statement]:
    if verbose <= 0:
        return list(parse(contents, filename))

    tok = Tokenizer(contents, filename=filename)
    blocks = tok.split_blocks()
    stacked = [block.stack() for block in blocks]
    statements = []
    for idx, block in enumerate(stacked):
        # Level 1: show block header, print statement
        # Level 2: show block header, block, print statement
        # Level 3: show block header, original source, block, print statement
        if idx > 0:
            rich.print()
        rich.print(f"-- Block {idx} ({block.loc})", file=sys.stderr)
        if verbose >= 3:
            rich.print("Original source:", file=sys.stderr)
            rich.print("```", file=sys.stderr)
            rich.print(block.loc.get_string(contents), file=sys.stderr)
            rich.print("```", file=sys.stderr)
        if verbose >= 2:
            rich.print(block, file=sys.stderr)
        statement = block.parse()
        statements.append(statement)
        rich.print(statement, file=sys.stderr)
    return statements


def main(
    filename: str | pathlib.Path,
    verbose: int = 0,
    line_length: int = 100,
    max_line_length: int | None = 0,
    compact: bool = False,
    recursive: bool = False,
    in_place: bool = False,
    name_case: NameCase = "same",
    attribute_case: NameCase = "same",
    kind_case: NameCase = "same",
    builtin_case: NameCase = "same",
    section_break_character: str = "-",
    section_break_width: int = 0,
    output: pathlib.Path | str | None = None,
    diff: bool = False,
    rename_file: pathlib.Path | str | None = None,
    raw_renames: list[str] | None = None,
    renames: dict[str, str] | None = None,
) -> None:
    if str(filename) == "-":
        contents = sys.stdin.read()
        filename = "<stdin>"
        is_stdin = True
    else:
        filename = pathlib.Path(filename)
        contents = filename.read_text()
        is_stdin = False

    if verbose >= 4:
        output_mod.LATFORM_OUTPUT_DEBUG = True
        logger = logging.getLogger("latform")
        logger.setLevel("DEBUG")

    renames = load_renames(rename_file, raw_renames, renames)

    options = FormatOptions(
        line_length=line_length,
        max_line_length=max_line_length or int(line_length * 1.3),
        compact=compact,
        indent_size=2,
        indent_char=" ",
        comment_col=40,
        newline_before_new_type=not compact,
        name_case=name_case,
        attribute_case=attribute_case,
        kind_case=kind_case,
        builtin_case=builtin_case,
        section_break_character=section_break_character,
        section_break_width=section_break_width,
        renames=renames,
    )
    if recursive:
        if is_stdin:
            raise NotImplementedError(
                "Recursive parsing using a lattice from stdin is not yet supported"
            )

        files = parse_file_recursive(filename)

        # TODO this should be separate from format, I think
        # Either that or lint can be treated as an error to not reformat?
        for fn, statements in files.by_filename.items():
            for st in statements:
                for lint in lint_statement(st):
                    logger.warning(lint.to_user_message())

        if verbose > 1:
            # NOTE: double-parsing, erroneously
            for fn in files.by_filename:
                process_file(contents=fn.read_text(), filename=fn, verbose=verbose)

        if in_place:
            if diff:
                raise NotImplementedError("In-place diff is not supported (or sensible?)")
            files.reformat(options)
        else:
            for fn, statements in files.by_filename.items():
                formatted = format_statements(statements, options)
                if diff:
                    original = fn.read_text()
                    to_write = get_diff(original, formatted, fromfile=fn, tofile=fn)
                else:
                    print(f"! {fn}")
                    to_write = formatted

                print(to_write)
        return

    statements = process_file(contents=contents, filename=filename, verbose=verbose)

    for st in statements:
        for lint in lint_statement(st):
            logger.warning(lint.to_user_message())

    formatted = format_statements(statements, options=options)
    if output:
        dest_fn = output
    elif in_place and not is_stdin:
        dest_fn = filename
    else:
        dest_fn = None

    if diff:
        to_write = get_diff(contents, formatted, fromfile=filename, tofile=filename)
    else:
        to_write = formatted

    if dest_fn:
        pathlib.Path(dest_fn).write_text(to_write)
    else:
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
    )

    parser.add_argument(
        "--rename",
        "-R",
        type=str,
        action="append",
        dest="raw_renames",
        help="Rename an element. In the form: 'old,new' (comma-delimited)",
    )

    parser.add_argument(
        "--rename-file",
        type=str,
        help="Load renames from a file. Each line should be comma-delimited in the form of `--rename`.",
    )

    parser.add_argument(
        "--diff",
        action="store_true",
        default=False,
        help="Show diff instead of formatted output",
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
        "--name-case",
        "--name",
        choices=("upper", "lower", "same"),
        default="upper",
        help="Case for element names, kinds, and functions",
    )

    parser.add_argument(
        "--kind-case",
        "--kind",
        choices=("upper", "lower", "same"),
        default="lower",
        help="Case for kinds (keywords)",
    )

    parser.add_argument(
        "--builtin-case",
        choices=("upper", "lower", "same"),
        default="lower",
        help="Case for builtin functions",
    )

    parser.add_argument(
        "--line-length",
        "-l",
        type=int,
        default=100,
        help="Desired line length. Some lines may exceed this (see also --max-line-length).",
    )

    parser.add_argument(
        "--max-line-length",
        "-m",
        type=int,
        default=None,
        help="Force lines over this length to be multilined. Defaults to 130% of line_length.",
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
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively (-r) parse lattice files, following call statements",
    )

    parser.add_argument(
        "--section-break-character",
        type=str,
        default="-",
        help="Section break character.  By default --line-length characters, unless overridden by --section-break-width",
    )

    parser.add_argument(
        "--section-break-width",
        type=int,
        default=None,
        help="Section break line width.  By default --line-length characters",
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
