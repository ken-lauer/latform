"""
`latform` - a Bmad lattice parser/formatter tool.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import pathlib
from typing import Collection

from . import output as output_mod
from .config import LatformProjectConfig
from .debug import print_blocks
from .lint import lint_files
from .output import format_statements
from .parser import Files, build_files
from .tao import format_tao_namelist
from .types import FormatOptions, NameCase, NamelistFormatOptions

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
    lint: bool = False,
    ignore_lints: Collection[str] = (),
    format_namelist: bool = True,
    config: LatformProjectConfig | None = None,
) -> None:
    """Parse, annotate, lint, format, and emit one Files set."""

    only_tao_init = files_obj.tao_init is not None and not recursive

    if not only_tao_init:
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

    if lint:
        for fn, lint_item in lint_files(
            files_obj, assume_defined=assume_defined, ignore=ignore_lints, config=config
        ):
            msg = lint_item.to_user_message()
            if recursive:
                name = files_obj.local_file_to_source_filename.get(fn, fn.name)
                logger.warning(f"[{name}] {msg}")
            else:
                logger.warning(msg)

    top_set = set(files_obj.top_files)
    results: dict[pathlib.Path, tuple[str, str]] = {}

    if files_obj.tao_init is not None and files_obj.tao_init.filename is not None:
        init_path = files_obj.tao_init.filename
        logicals = config.namelist_logicals if config is not None else ("T", "F")
        init_original = files_obj.tao_init.render()
        init_formatted = format_tao_namelist(
            files_obj.tao_init,
            options=options.namelist if format_namelist else None,
            fix_types=format_namelist,
            logicals=logicals,
        )
        results[init_path] = (init_original, init_formatted)
        top_set.add(init_path)

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
        logger.info("Processing %s", fn)
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
    recursive: bool | None = None,
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
    lint: bool = False,
    ignore_lints: list[str] | None = None,
    format_namelist: bool = True,
    namelist_options: NamelistFormatOptions | None = None,
    config: LatformProjectConfig | None = None,
) -> None:
    if verbose >= 4:
        output_mod.LATFORM_OUTPUT_DEBUG = True
        logger.setLevel("DEBUG")

    if isinstance(filename, (str, pathlib.Path)):
        filenames: list[str | pathlib.Path] = [filename]
    else:
        filenames = list(filename)

    # --strict-references promises reference issues "reported as lint warnings".
    lint = lint or not assume_defined

    loaded_renames = load_renames(rename_file, raw_renames, renames)
    ignore_codes = [
        code.strip() for entry in (ignore_lints or []) for code in entry.split(",") if code.strip()
    ]

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
    if namelist_options is not None:
        options.namelist = namelist_options

    for files_obj in build_files(filenames, combine=combine):
        this_recursive = recursive
        if recursive is None:
            this_recursive = options.flatten_call or files_obj.tao_init is not None  # implied
        else:
            this_recursive = recursive

        process_files(
            files_obj,
            options,
            recursive=this_recursive,
            verbose=verbose,
            in_place=in_place,
            diff=diff,
            output=output,
            error_if_missing=error_if_missing,
            assume_defined=assume_defined,
            lint=lint,
            ignore_lints=ignore_codes,
            format_namelist=format_namelist,
            config=config,
        )


def _build_argparser() -> argparse.ArgumentParser:
    from . import cli

    parser = argparse.ArgumentParser(
        prog="latform",
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    cli.add_input_arguments(parser)
    cli.add_config_arguments(parser)

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
        choices=cli.CASE_CHOICES,
        default="upper",
        help="Case for element names, kinds, and functions",
    )
    parser.add_argument(
        "--attribute-case",
        choices=cli.CASE_CHOICES,
        default="lower",
        help="Case for element attribute names",
    )
    parser.add_argument(
        "--kind-case",
        "--kind",
        choices=cli.CASE_CHOICES,
        default="lower",
        help="Case for kinds (keywords)",
    )
    parser.add_argument(
        "--builtin-case",
        choices=cli.CASE_CHOICES,
        default="lower",
        help="Case for builtin functions",
    )
    parser.add_argument(
        "--controller-variable-case",
        choices=cli.CASE_CHOICES,
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
    cli.add_namelist_format_arguments(parser)
    parser.add_argument(
        "--lint",
        action="store_true",
        default=False,
        help=(
            "Report lint warnings (unknown attributes, duplicate attributes, etc.) "
            "in addition to formatting. See also the dedicated 'latform-lint' command."
        ),
    )

    cli.add_lint_arguments(parser)
    cli.add_logging_arguments(parser, default_level="INFO")
    return parser


def cli_main(args: list[str] | None = None) -> None:
    """
    CLI entrypoint main.

    Parameters
    ----------
    args : list of str, optional
        Command-line arguments to parse and pass to `main()`.
    """
    from . import cli

    parser = _build_argparser()

    # Peek at config-related flags so the config can supply argparse defaults
    # (which explicit CLI flags then override).
    prelim, _ = parser.parse_known_args(args=args)
    config = cli.resolve_config(prelim)
    if config.format:
        parser.set_defaults(**config.format)

    parsed = parser.parse_args(args=args)
    cli.configure_logging(parsed.log_level)

    kwargs = vars(parsed)
    kwargs.pop("log_level")
    kwargs.pop("config", None)
    kwargs.pop("use_config", None)
    filenames = kwargs.pop("filename")

    # The individual namelist flags are folded into a single options object;
    # `format_namelist` stays as its own toggle passed straight through.
    kwargs["namelist_options"] = cli.build_namelist_options(parsed)
    for dest in cli.NAMELIST_FORMAT_DESTS:
        if dest != "format_namelist":
            kwargs.pop(dest, None)

    filenames, from_top_level = cli.require_input_files(filenames, config)

    if from_top_level:
        kwargs["recursive"] = True

    try:
        main(filename=filenames, config=config, **kwargs)
    except FileNotFoundError as ex:
        logger.error("%s", ex)
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli_main()
