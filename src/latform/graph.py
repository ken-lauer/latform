"""
`latform-graph` - lattice tree graphs.
"""

from __future__ import annotations

import argparse
import logging
import pathlib

from .parser import parse_file_recursive

DESCRIPTION = __doc__
logger = logging.getLogger(__name__)


def main(
    filename: str | pathlib.Path,
    verbose: int = 0,
    output: pathlib.Path | str | None = None,
) -> None:
    if str(filename) == "-":
        # contents = sys.stdin.read()
        # filename = "<stdin>"
        # is_stdin = True
        raise NotImplementedError("stdin support")
    else:
        filename = pathlib.Path(filename)
        # contents = filename.read_text()
        # is_stdin = False

    files = parse_file_recursive(filename)

    if output:
        dest_fn = output
    else:
        dest_fn = None

    graph_lines = ["graph LR"]

    def make_id(fn: str):
        return fn.replace("/", "_").replace(".", "_").replace("-", "_")

    for fn1, fn2 in files.call_graph_edges:
        id1 = make_id(fn1)
        id2 = make_id(fn2)

        # id["label"] --> id2["label2"]
        graph_lines.append(f'    {id1}["{fn1}"] --> {id2}["{fn2}"]')

    to_write = "\n".join(graph_lines)

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
        "--output",
        "-o",
        action="store_true",
        help="Write to this filename (or directory, if multiple files)",
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
