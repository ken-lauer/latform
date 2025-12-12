"""
`latform-graph` - lattice tree graphs.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from .parser import Files, MemoryFiles, is_call_statement

DESCRIPTION = __doc__
logger = logging.getLogger(__name__)


def _generate_tree_text(files: Files) -> str:
    """
    Generates a tree-like string representation of the parsed file hierarchy.
    """
    lines = []

    def _walk(fn: pathlib.Path, prefix: str, visited_stack: set[pathlib.Path]):
        """
        Recursive walker.
        visited_stack prevents infinite recursion in case of circular calls.
        """
        statements = files.by_filename[fn]

        children: list[pathlib.Path] = []
        for st in statements:
            if is_call_statement(st):
                child_path = st.metadata.get("local_path")
                if child_path:
                    children.append(child_path)

        count = len(children)
        for i, child_path in enumerate(children):
            is_last = i == count - 1
            marker = "└── " if is_last else "├── "

            display = files.local_file_to_source_filename.get(child_path, child_path.name)

            lines.append(f"{prefix}{marker}{display}")

            extension = "    " if is_last else "│   "

            if child_path not in visited_stack:
                _walk(child_path, prefix + extension, visited_stack | {child_path})
            else:
                lines.append(f"{prefix}{extension}└── [Recursive: {display}]")

    lines.append(files.main.name)

    _walk(files.main, "", {files.main})

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

    to_write = ""

    if format == "text":
        to_write = _generate_tree_text(files)
    else:
        # Generate Mermaid Graph View (Default)
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
    CLI entrypoint for latform-graph.

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
