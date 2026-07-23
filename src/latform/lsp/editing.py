"""Text-transformation features: formatting, rename, and shared edit builders."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from ..location import Location
from ..output import format_statements
from ..parser import implicit_location
from ..statements import Statement
from ..token import Role
from ..types import FormatOptions
from .document import AnalyzedDocument
from .positions import _statement_file, _statement_line_span, token_at_position
from .resolve import _format_options


@dataclass
class TextEditSpec:
    """A single text edit: replace ``location`` with ``new_text``."""

    location: Location
    new_text: str


def _line_range(filename: pathlib.Path | None, first: int, last: int) -> Location:
    """A range covering whole lines ``[first, last]`` (to the start of last+1)."""
    return Location(filename=filename, line=first, column=0, end_line=last + 1, end_column=0)


def _rewrite(statement: Statement, replacement: Statement | None, options: FormatOptions):
    """
    A ``TextEditSpec`` replacing ``statement``'s lines with a formatted
    ``replacement`` (or deleting them when ``replacement`` is ``None``).
    """
    span = _statement_line_span(statement)
    if span is None:
        return None
    text = (
        "" if replacement is None else format_statements([replacement], options).rstrip("\n") + "\n"
    )
    return TextEditSpec(_line_range(_statement_file(statement), span[0], span[1]), text)


def format_document(analyzed: AnalyzedDocument) -> str | None:
    """The whole document reformatted per project settings, or ``None``."""
    if analyzed.files is None:
        return None
    return format_statements(analyzed.statements, _format_options(analyzed.config))


def format_range(
    analyzed: AnalyzedDocument, start_line: int, end_line: int
) -> tuple[int, int, str] | None:
    """
    Reformat the statements intersecting 0-indexed lines ``[start_line, end_line]``.

    Returns ``(first_line, last_line, formatted_text)`` spanning the full lines
    of the affected statements, or ``None`` if nothing there can be formatted.
    """
    if analyzed.files is None:
        return None
    selected = []
    for statement in analyzed.statements:
        span = _statement_line_span(statement)
        if span is not None and span[0] <= end_line and span[1] >= start_line:
            selected.append((span[0], span[1], statement))
    if not selected:
        return None
    first = min(span_start for span_start, _, _ in selected)
    last = max(span_end for _, span_end, _ in selected)
    text = format_statements([st for _, _, st in selected], _format_options(analyzed.config))
    return first, last, text


def prepare_rename(analyzed: AnalyzedDocument, line: int, char: int) -> Location | None:
    """
    The range of the renameable name under a 0-indexed position, or ``None``.

    Only element/constant/line names (``Role.name_``) with a real source
    location can be renamed — not keywords, attributes, or builtins.
    """
    if analyzed.files is None:
        return None
    tok = token_at_position(analyzed.statements, line, char)
    if tok is None or tok.role != Role.name_ or tok.loc is None:
        return None
    if tok.loc.filename == implicit_location.filename:
        return None
    return tok.loc
