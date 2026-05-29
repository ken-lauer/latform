from __future__ import annotations

import sys
from typing import TextIO

import rich

from .parser import Files
from .token import Token
from .tokenizer import tokenize


def print_blocks(files: Files, verbose: int = 1, out: TextIO | None = None) -> None:
    """
    Print pre-parse blocks and parsed statements for each file in ``files``.

    Verbosity levels:

    - 1: block header + parsed statement
    - 2: + the Block AST
    - 3: + original source text per block
    """
    if verbose <= 0:
        return
    if verbose >= 4:
        Token._detailed_repr_ = True

    if out is None:
        # Resolve lazily so e.g. pytest's capsys-patched sys.stderr is honored.
        out = sys.stderr

    multi = len(files.by_filename) > 1

    for file_idx, (fn, statements) in enumerate(files.by_filename.items()):
        contents = files._get_file_contents(fn)
        display = files.local_file_to_source_filename.get(fn, str(fn))

        if multi:
            if file_idx > 0:
                rich.print(file=out)
            rich.print(f"[bold]Debug: {display}[/bold]", file=out)

        blocks = files.blocks_by_filename.get(fn)
        if blocks is None:
            blocks = tokenize(contents, filename=fn)

        for idx, (block, statement) in enumerate(zip(blocks, statements)):
            if idx > 0:
                rich.print(file=out)
            rich.print(f"-- Block {idx} ({block.loc})", file=out)
            if verbose >= 3:
                rich.print("Original source:", file=out)
                rich.print("```", file=out)
                rich.print(block.loc.get_string(contents), file=out)
                rich.print("```", file=out)
            if verbose >= 2:
                rich.print(block, file=out)
            rich.print(statement, file=out)
