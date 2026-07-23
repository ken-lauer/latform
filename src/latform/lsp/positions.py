"""AST/position navigation shared by the feature providers."""

from __future__ import annotations

from typing import Sequence

from ..location import Location
from ..statements import Line, Statement
from ..token import Token
from ..types import CallName
from ..walk import walk


def loc_contains(loc: Location, line: int, char: int) -> bool:
    """
    Whether a 0-indexed ``(line, char)`` position falls within ``loc``.

    ``Location`` uses an inclusive ``end_column``; a cursor resting one past the
    last character (i.e. immediately after the token) is treated as outside.
    """
    if line < loc.line or line > loc.end_line:
        return False
    if line == loc.line and char < loc.column:
        return False
    if line == loc.end_line and char > loc.end_column:
        return False
    return True


def _statement_tokens(statement: Statement) -> list[Token]:
    """Every `Token` contained in ``statement``, in walk order."""
    return [item.node for item in walk(statement) if isinstance(item.node, Token)]


def _locate(
    statements: Sequence[Statement], line: int, char: int
) -> tuple[Token | None, Statement | None]:
    """
    The innermost `Token` covering a 0-indexed position and its statement.

    Ties (a position covered by nested tokens) resolve to the smallest span.
    """
    best: Token | None = None
    best_statement: Statement | None = None
    best_width: int | None = None
    for statement in statements:
        for tok in _statement_tokens(statement):
            loc = tok.loc
            if loc is None or not loc_contains(loc, line, char):
                continue
            width = (loc.end_line - loc.line, loc.end_column - loc.column)
            flat = width[0] * 1_000_000 + width[1]
            if best_width is None or flat < best_width:
                best, best_statement, best_width = tok, statement, flat
    return best, best_statement


def token_at_position(statements: Sequence[Statement], line: int, char: int) -> Token | None:
    """
    The innermost `Token` covering a 0-indexed ``(line, char)`` position.

    Ties (a position covered by nested tokens) resolve to the smallest span.
    """
    return _locate(statements, line, char)[0]


def definition_name_token(statement: Statement) -> Token | None:
    """The defining name `Token` of a named statement, or ``None``."""
    if isinstance(statement, Line) and isinstance(statement.name, CallName):
        return statement.name.name
    name = getattr(statement, "name", None)
    return name if isinstance(name, Token) else None


def _statement_line_span(statement: Statement) -> tuple[int, int] | None:
    """The ``(start_line, end_line)`` a statement occupies, or ``None``."""
    locs = [tok.loc for tok in _statement_tokens(statement) if tok.loc is not None]
    if not locs:
        return None
    return min(loc.line for loc in locs), max(loc.end_line for loc in locs)


def _statement_file(statement: Statement):
    """The file a statement's tokens come from."""
    for tok in _statement_tokens(statement):
        if tok.loc is not None:
            return tok.loc.filename
    return None
