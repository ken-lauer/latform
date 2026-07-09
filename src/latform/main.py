"""
`latform` - a Bmad lattice parser/formatter tool.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import pathlib

from . import output as output_mod
from .debug import print_blocks
from .lint import lint_statements
from .output import format_statements
from .parser import Files, build_files
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

    for from_, to in list(res.items()):
        if not from_ or not to:
            res.pop(from_)
            logger.error(f"Unable to use empty rename: {from_!r} -> {to!r}")

    return res


def get_diff(
    original: str, formatted: str, fromfile: pathlib.Path | str, tofile: pathlib.Path | str
):
    original_lines = original.splitlines(keepends=True)
    formatted_lines = formatted.splitlines(keepends=True)

    if original_lines and not original_lines[-1].endswith("\n"):
        original_lines[-1] += "\n"
    if formatted_lines and not formatted_lines[-1].endswith("\n"):
        formatted_lines[-1] += "\n"
    udiff = difflib.unified_diff(
        original_lines,
        formatted_lines,
        fromfile=str(fromfile),
        tofile=str(tofile),
    )
    return "".join(udiff)


def process_files(
    files_obj: Files,
    options: FormatOptions,
    *,
    recursive: bool,
    verbose: int,
    in_place: bool,
    diff: bool,
    output: pathlib.Path | str | None,
    error_if_missing: bool,
    assume_defined: bool = True,
) -> None:
    """Parse, annotate, lint, format, and emit one Files set."""
    files_obj.parse(
        recurse=recursive,
        raise_if_missing=error_if_missing,
        keep_blocks=verbose > 0,
    )
    files_obj.annotate()

    if options.renames:
        files_obj.rename(options.renames, assume_defined=assume_defined)

    if verbose > 0:
        print_blocks(files_obj, verbose=verbose)

    named = files_obj.get_named_items()
    for fn, statements in files_obj.by_filename.items():
        logger.info("Processing %s", fn)
        for lint in lint_statements(statements, named=named, assume_defined=assume_defined):
            msg = lint.to_user_message()
            if recursive:
                name = files_obj.local_file_to_source_filename.get(fn, fn.name)
                logger.warning(f"[{name}] {msg}")
            else:
                logger.warning(msg)

    top_set = set(files_obj.top_files)
    results: dict[pathlib.Path, tuple[str, str]] = {}

    if options.flatten_call:
        for top, statements in files_obj.flatten_all(
            call=options.flatten_call, inline=options.flatten_inline
        ).items():
            formatted = format_statements(statements, options)
            results[top] = (files_obj._get_file_contents(top), formatted)
    else:
        for fn, statements in files_obj.by_filename.items():
            formatted_text = format_statements(statements, options)
            original_text = files_obj._get_file_contents(fn)
            results[fn] = (original_text, formatted_text)

    if output and not in_place and len(top_set & set(results)) > 1:
        raise ValueError(
            "--output with multiple top-level files is ambiguous; use --in-place instead."
        )

    for fn, (original, formatted) in results.items():
        is_top_entry = fn in top_set
        is_stdin_entry = files_obj.local_file_to_source_filename.get(fn) == "<stdin>"
        display_name = files_obj.local_file_to_source_filename.get(fn, str(fn))

        if diff:
            if in_place:
                raise NotImplementedError("In-place diff is not supported.")
            diff_output = get_diff(original, formatted, fromfile=display_name, tofile=display_name)
            if diff_output:
                print(diff_output)
            continue

        if output and is_top_entry and not in_place:
            pathlib.Path(output).write_text(formatted)
            continue
        if output and not is_top_entry and not in_place:
            continue

        if in_place:
            if is_stdin_entry:
                print(formatted)
            else:
                fn.write_text(formatted)
            continue

        if recursive and not is_stdin_entry:
            print(f"! {display_name}")
        print(formatted)


def main(
    filename: pathlib.Path | str | list[pathlib.Path | str],
    verbose: int = 0,
    line_length: int = 100,
    max_line_length: int | None = 0,
    compact: bool = False,
    recursive: bool = False,
    in_place: bool = False,
    name_case: NameCase = "upper",
    attribute_case: NameCase = "lower",
    kind_case: NameCase = "lower",
    builtin_case: NameCase = "lower",
    controller_variable_case: NameCase = "same",
    section_break_character: str = "-",
    section_break_width: int = 0,
    output: pathlib.Path | str | None = None,
    diff: bool = False,
    rename_file: pathlib.Path | str | None = None,
    raw_renames: list[str] | None = None,
    renames: dict[str, str] | None = None,
    flatten: bool = False,
    flatten_call: bool = False,
    flatten_inline: bool = False,
    strip_comments: bool = False,
    error_if_missing: bool = False,
    combine: bool = False,
    assume_defined: bool = True,
) -> None:
    if verbose >= 4:
        output_mod.LATFORM_OUTPUT_DEBUG = True
        logger.setLevel("DEBUG")

    if isinstance(filename, (str, pathlib.Path)):
        filenames: list[str | pathlib.Path] = [filename]
    else:
        filenames = list(filename)

    loaded_renames = load_renames(rename_file, raw_renames, renames)

    options = FormatOptions(
        line_length=line_length,
        max_line_length=max_line_length or int(line_length * 1.3),
        compact=compact,
        indent_size=2,  # Default hardcoded in original
        indent_char=" ",
        comment_col=40,
        newline_before_new_type=not compact,
        name_case=name_case,
        attribute_case=attribute_case,
        kind_case=kind_case,
        builtin_case=builtin_case,
        controller_variable_case=controller_variable_case,
        section_break_character=section_break_character,
        section_break_width=section_break_width,
        renames=loaded_renames,
        flatten_call=flatten or flatten_call,
        flatten_inline=flatten or flatten_inline,
        strip_comments=strip_comments,
    )
    recursive = recursive or options.flatten_call  # implied

    for files_obj in build_files(filenames, combine=combine):
        process_files(
            files_obj,
            options,
            recursive=recursive,
            verbose=verbose,
            in_place=in_place,
            diff=diff,
            output=output,
            error_if_missing=error_if_missing,
            assume_defined=assume_defined,
        )


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
        "--controller-variable-case",
        choices=("upper", "lower", "same"),
        default="same",
        help="Case for overlay/group/ramper control variables (from var={...})",
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
        help="Force lines over this length to be multilined. Defaults to 130%% of line_length.",
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
        "--flatten",
        action="store_true",
        help="Inlining all call statements and call:: arguments into a single output lattice (implies --flatten-call, --flatten-inline)",
    )
    parser.add_argument(
        "--flatten-call",
        action="store_true",
        help="Inlining all call statements into a single output lattice",
    )
    parser.add_argument(
        "--flatten-inline",
        action="store_true",
        help="Inline all call:: arguments",
    )
    parser.add_argument(
        "--strip-comments",
        action="store_true",
        help="Remove comments from the output",
    )
    parser.add_argument(
        "-e",
        "--error-if-missing",
        action="store_true",
        help="If a file is missing during parsing, exit with an error.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help=(
            "Process all input files together as a single set, sharing one parse stack. "
            "Without this, each file is parsed independently of the others."
        ),
    )
    parser.add_argument(
        "--strict-references",
        dest="assume_defined",
        action="store_false",
        default=True,
        help=(
            "Only recognize element/constant references defined in the loaded files. "
            "By default, references to names defined elsewhere are assumed to exist; "
            "with this flag they are left unresolved and reported as lint warnings."
        ),
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
    logging.getLogger("latform").setLevel(log_level)
    logging.basicConfig()

    filenames = kwargs.pop("filename")
    try:
        main(filename=filenames, **kwargs)
    except FileNotFoundError as ex:
        logger.error("%s", ex)
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli_main()
