"""Document symbols, workspace symbols, and semantic tokens (name classification)."""

from __future__ import annotations

from ..location import Location
from ..parser import implicit_location
from ..statements import Constant, Element, ElementList, Line, Statement
from ..token import Role
from .document import AnalyzedDocument
from .positions import _statement_tokens, definition_name_token


def _symbol_kind(statement: Statement) -> str | None:
    """
    The symbol kind of a definition statement, or ``None`` if it isn't one.

    One of ``"element"``, ``"line"``, ``"list"``, ``"constant"``.
    """
    if isinstance(statement, Element):
        return "element"
    if isinstance(statement, Line):
        return "line"
    if isinstance(statement, ElementList):
        return "list"
    if isinstance(statement, Constant):
        return "constant"
    return None


def document_symbols(analyzed: AnalyzedDocument) -> list[tuple[str, str, Location]]:
    """
    Named symbols defined in the document as ``(name, kind, location)`` tuples.
    """
    out: list[tuple[str, str, Location]] = []
    for statement in analyzed.statements:
        kind = _symbol_kind(statement)
        name_token = definition_name_token(statement)
        if kind is None or name_token is None or name_token.loc is None:
            continue
        out.append((str(name_token), kind, name_token.loc))
    return out


def workspace_symbols(analyzed: AnalyzedDocument, query: str) -> list[tuple[str, str, Location]]:
    """
    Project-wide symbols matching ``query`` as ``(name, kind, location)`` tuples.

    Searches every element/line/list/constant definition across the analyzed
    project tree (case-insensitive substring match; an empty query returns all).
    The client typically re-ranks, so ordering here is by name.
    """
    if analyzed.files is None:
        return []
    needle = query.lower()
    out: list[tuple[str, str, Location]] = []
    for statements in analyzed.files.by_filename.values():
        for statement in statements:
            kind = _symbol_kind(statement)
            name_token = definition_name_token(statement)
            if kind is None or name_token is None or name_token.loc is None:
                continue
            if name_token.loc.filename == implicit_location.filename:
                continue  # implicit BEGINNING/END have no real location
            name = str(name_token)
            if needle and needle not in name.lower():
                continue
            out.append((name, kind, name_token.loc))
    out.sort(key=lambda item: item[0].lower())
    return out


# --------------------------------------------------------------------------- #
# Semantic tokens
# --------------------------------------------------------------------------- #

# Semantic-token legend: token types (index = position) and modifiers, kept in
# sync with the legend advertised to the client in `create_server`.
SEMANTIC_TOKEN_TYPES = (
    "variable",
    "type",
    "class",
    "namespace",
    "property",
    "function",
    "keyword",
    "parameter",
    "string",
)
SEMANTIC_TOKEN_MODIFIERS = ("definition",)
_DEFINITION_MODIFIER = 1  # 1 << index("definition")
_TYPE_INDEX = {name: index for index, name in enumerate(SEMANTIC_TOKEN_TYPES)}
# Roles other than `name_` map to a fixed token type; ``name_`` is resolved by
# what it refers to (see `_name_token_type`) so valid element names stand out.
_ROLE_TOKEN_TYPE = {
    Role.kind: _TYPE_INDEX["type"],
    Role.attribute_name: _TYPE_INDEX["property"],
    Role.builtin: _TYPE_INDEX["function"],
    Role.statement_definition: _TYPE_INDEX["keyword"],
    Role.controller_variable: _TYPE_INDEX["parameter"],
    Role.env_var: _TYPE_INDEX["parameter"],
    Role.filename: _TYPE_INDEX["string"],
}


def _name_token_type(target: Statement | None) -> int:
    """
    Token type for a ``name_`` token, by what it resolves to.

    Elements become ``class`` and lines/lists ``namespace`` (both distinctly
    themed), so a *valid* element name is visibly highlighted; a constant or an
    unresolved reference stays a plain ``variable``.
    """
    if isinstance(target, Element):
        return _TYPE_INDEX["class"]
    if isinstance(target, (Line, ElementList)):
        return _TYPE_INDEX["namespace"]
    return _TYPE_INDEX["variable"]


def semantic_tokens(analyzed: AnalyzedDocument) -> list[tuple[int, int, int, int, int]]:
    """
    Role-classified tokens for semantic highlighting.

    Returns ``(line, start_char, length, type_index, modifiers)`` per token,
    sorted by position.  Uses the parser's `Role` annotations (so highlighting
    matches how latform understands each token), and resolves ``name_`` tokens
    against the symbol table so defined element/line names are coloured as
    ``class``/``namespace`` while unresolved references stay plain.  Multi-line
    tokens are skipped (the LSP encoding is single-line).
    """
    named = analyzed.files.get_named_items() if analyzed.files is not None else {}
    out: list[tuple[int, int, int, int, int]] = []
    for statement in analyzed.statements:
        # Only true definition statements carry a "definition" name (a Parameter's
        # ``name`` is an attribute, not a definition).
        is_definition = isinstance(statement, (Element, Constant, Line, ElementList))
        def_token = definition_name_token(statement) if is_definition else None
        for tok in _statement_tokens(statement):
            loc = tok.loc
            if loc is None or loc.line != loc.end_line:
                continue
            if tok.role == Role.name_:
                target = statement if tok is def_token else named.get(str(tok).upper())
                type_index = _name_token_type(target)
            else:
                type_index = _ROLE_TOKEN_TYPE.get(tok.role)
            if type_index is None:
                continue
            length = loc.end_column - loc.column  # end_column is exclusive
            if length <= 0:
                continue
            modifiers = _DEFINITION_MODIFIER if tok is def_token else 0
            out.append((loc.line, loc.column, length, type_index, modifiers))
    out.sort(key=lambda item: (item[0], item[1]))
    return out
