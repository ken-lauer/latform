"""Navigation features: go-to-definition, references, document highlight, file deps."""

from __future__ import annotations

from ..location import Location
from ..parser import implicit_location
from ..statements import Constant, Element, ElementList, Line
from ..token import Role
from .document import AnalyzedDocument
from .positions import _statement_tokens, definition_name_token, token_at_position


def resolve_definition(analyzed: AnalyzedDocument, line: int, char: int) -> Location | None:
    """
    Location of the definition for the symbol under a 0-indexed position.

    Returns ``None`` if there is no token there, it is not a name, or its
    definition has no real source location (e.g. the implicit ``BEGINNING``).
    """
    if analyzed.files is None:
        return None
    tok = token_at_position(analyzed.statements, line, char)
    if tok is None:
        return None

    named = analyzed.files.get_named_items()
    statement = named.get(str(tok).upper())
    if statement is None:
        return None

    name_token = definition_name_token(statement)
    if name_token is None or name_token.loc is None:
        return None
    if name_token.loc.filename == implicit_location.filename:
        return None
    return name_token.loc


def find_references(
    analyzed: AnalyzedDocument,
    line: int,
    char: int,
    *,
    include_declaration: bool = True,
) -> list[Location]:
    """
    All occurrences of the name under a 0-indexed position, across the tree.

    References and definitions of element/constant/line names all carry
    ``Role.name_``; matching by (uppercased) name against those tokens finds
    every usage without colliding with attribute names, keywords, builtins, or
    controller variables (which carry distinct roles).

    Parameters
    ----------
    analyzed : AnalyzedDocument
        The analyzed document (its ``files`` spans the whole project tree).
    line, char : int
        0-indexed cursor position within the current document.
    include_declaration : bool, optional
        Whether to include the defining occurrence.  Defaults to True.

    Returns
    -------
    list of Location
        Occurrences sorted by ``(filename, line, column)``.
    """
    if analyzed.files is None:
        return []
    tok = token_at_position(analyzed.statements, line, char)
    if tok is None or tok.role != Role.name_:
        return []

    target = str(tok).upper()
    results: list[Location] = []
    for statements in analyzed.files.by_filename.values():
        for statement in statements:
            def_token = definition_name_token(statement)
            for candidate in _statement_tokens(statement):
                if candidate.role != Role.name_ or candidate.loc is None:
                    continue
                if str(candidate).upper() != target:
                    continue
                if candidate is def_token and not include_declaration:
                    continue
                results.append(candidate.loc)

    results.sort(key=lambda loc: (str(loc.filename), loc.line, loc.column))
    return results


def document_highlights(
    analyzed: AnalyzedDocument, line: int, char: int
) -> list[tuple[Location, bool]]:
    """
    Occurrences of the name under a 0-indexed position within *this* document.

    Returns ``(location, is_definition)`` per occurrence.  Unlike
    `find_references`, this is scoped to the current file (that is what an editor
    highlights) and distinguishes the defining occurrence.
    """
    if analyzed.files is None:
        return []
    tok = token_at_position(analyzed.statements, line, char)
    if tok is None or tok.role != Role.name_:
        return []
    target = str(tok).upper()
    out: list[tuple[Location, bool]] = []
    for statement in analyzed.statements:
        is_definition = isinstance(statement, (Element, Constant, Line, ElementList))
        def_token = definition_name_token(statement) if is_definition else None
        for candidate in _statement_tokens(statement):
            if candidate.role != Role.name_ or candidate.loc is None:
                continue
            if str(candidate).upper() == target:
                out.append((candidate.loc, candidate is def_token))
    return out


def file_dependencies(analyzed: AnalyzedDocument) -> dict | None:
    """
    The project's ``call`` include graph for the analyzed document.

    Returns a dict with a rendered ``tree`` (text), a ``mermaid`` diagram, and
    the raw ``edges`` (``[caller, callee]`` display-name pairs), or ``None`` if
    the document did not parse.  Built from the same graph machinery as
    ``latform-graph``.
    """
    from ..graph import _generate_mermaid, _generate_tree_text

    files = analyzed.files
    if files is None or not files.top_files:
        return None
    return {
        "tree": _generate_tree_text(files),
        "mermaid": _generate_mermaid(files),
        "edges": [list(edge) for edge in files.call_graph_edges],
    }
