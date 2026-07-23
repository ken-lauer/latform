"""
Language Server Protocol implementation for Bmad lattice files.

This module provides a `pygls`-based language server exposing go-to-definition,
hover, document symbols, and diagnostics (backed by the linter).  ``pygls`` is an
optional dependency; install it with ``pip install latform[lsp]``.

The analysis layer (`analyze`, `token_at_position`, `resolve_definition`,
`iter_diagnostics`, `hover_text`, `document_symbols`) is pure latform and can be
exercised without ``pygls`` installed.  The pygls glue lives in `create_server`
and `main`.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import logging
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Generator, Sequence

from .attrs import element_key_to_attrs
from .config import LatformProjectConfig, discover_config
from .const import named_physical_constants
from .funcs import BUILTIN_CONSTANTS, INTRINSIC_FUNCTIONS
from .lint import (
    Lint,
    _matching_builtin_names,
    get_used_names,
    lint_statements,
)
from .location import Location
from .output import format_statements
from .parser import MemoryFiles, _expand_element_type, _resolve_lattice_paths, implicit_location
from .statements import (
    BUILTIN_TARGETS,
    Constant,
    Element,
    ElementList,
    Line,
    Parameter,
    Statement,
)
from .tao import TaoInit, is_init_file, looks_like_namelist
from .token import Role, Token
from .types import Attribute, CallName, FormatOptions, Seq
from .walk import walk

_ELEMENT_TYPES = frozenset(k for k in element_key_to_attrs if not k.startswith("!"))
_FORMAT_OPTION_FIELDS = frozenset(f.name for f in dataclass_fields(FormatOptions))

logger = logging.getLogger(__name__)

LOG_LEVELS = ("debug", "info", "warning", "error")
_ENV_LOG_LEVEL = "LATFORM_LSP_LOG_LEVEL"
_ENV_LOG_FILE = "LATFORM_LSP_LOG_FILE"


def configure_logging(level: str = "warning", log_file: str | pathlib.Path | None = None) -> int:
    """
    Route latform/pygls logging to stderr (or a file) at ``level``.

    Logging must never go to stdout: that is the JSON-RPC channel for the
    stdio transport.  Both stderr and a log file are surfaced by LSP clients
    (Neovim's ``:LspLog``, VS Code's output channel).

    Parameters
    ----------
    level : str
        One of ``LOG_LEVELS`` (case-insensitive).
    log_file : path-like, optional
        Write logs here instead of stderr.

    Returns
    -------
    int
        The resolved numeric logging level.
    """
    numeric = getattr(logging, str(level).upper(), logging.WARNING)
    if log_file:
        handler: logging.Handler = logging.FileHandler(pathlib.Path(log_file), encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    for name in ("latform", "pygls"):
        package_logger = logging.getLogger(name)
        package_logger.handlers[:] = [handler]  # replace so restarts don't stack handlers
        package_logger.setLevel(numeric)
        package_logger.propagate = False
    return numeric


# --------------------------------------------------------------------------- #
# Analysis layer (pure latform — no pygls dependency)
# --------------------------------------------------------------------------- #


ParseCache = dict[pathlib.Path, "tuple[str, list[Statement]]"]


def _definition_signature(by_filename: dict) -> tuple:
    """
    A signature of everything that affects cross-file annotation.

    Two builds with the same signature annotate identically: the set of defined
    names, their kinds, element inheritance keywords, and file order.  When it is
    unchanged, files whose contents did not change keep their prior annotation.
    """
    sig: list[tuple] = []
    for filename, statements in by_filename.items():
        for st in statements:
            if isinstance(st, Element):
                sig.append((filename, "E", str(st.name).upper(), str(st.keyword).upper()))
            elif isinstance(st, Constant):
                sig.append((filename, "C", str(st.name).upper()))
            elif isinstance(st, (Line, ElementList)):
                tok = definition_name_token(st)
                sig.append((filename, "L", str(tok).upper() if tok is not None else ""))
    return tuple(sig)


class _OverlayFiles(MemoryFiles):
    """
    `MemoryFiles` with overlay-tolerant reads and incremental parse/annotate.

    Overlay lookup falls back to a resolved path so ``call`` targets containing
    ``..`` or symlinks still match open editor buffers.

    With ``_parse_cache`` set, per-file parsing reuses cached statements for
    files whose contents are unchanged, so an edit only re-parses the changed
    file.  With ``_annotate_state`` also set, the cross-file annotation pass
    re-annotates only the re-parsed files when the definition signature is
    unchanged (an edit that touched no definition), reusing the prior annotation
    of every other file.
    """

    _parse_cache: ParseCache | None = None
    _annotate_state: dict | None = None
    _reparsed: set | None = None
    _named_cache: dict | None = None

    def _get_file_contents(self, filepath: pathlib.Path) -> str:
        for candidate in (filepath, filepath.resolve()):
            if candidate in self.initial_contents:
                return self.initial_contents[candidate]
        return filepath.read_text()

    def get_named_items(self) -> dict:
        # Memoize for this build: statements do not change after parsing, and a
        # publish resolves several documents against the same file set.
        if self._named_cache is None:
            self._named_cache = super().get_named_items()
        return self._named_cache

    def _parse_file(self, contents: str, filename: pathlib.Path) -> list[Statement]:
        cache = self._parse_cache
        if cache is None:
            return super()._parse_file(contents, filename)
        cached = cache.get(filename)
        if cached is not None and cached[0] == contents:
            return cached[1]
        statements = super()._parse_file(contents, filename)
        cache[filename] = (contents, statements)
        if self._reparsed is not None:
            self._reparsed.add(filename)
        return statements

    def annotate(self):
        state = self._annotate_state
        if state is None or self._reparsed is None:
            return super().annotate()

        named = self.get_named_items()
        signature = _definition_signature(self.by_filename)
        # Only reuse prior annotation when no definition changed anywhere.
        incremental = state.get("signature") == signature
        state["signature"] = signature

        defined: dict[str, Element] = {}
        for filename, statements in self.by_filename.items():
            if incremental and filename not in self._reparsed:
                # Prior annotation is still valid; just feed the type accumulator
                # so re-parsed files can resolve inheritance from this one.
                for st in statements:
                    if isinstance(st, Element):
                        defined[str(st.name).upper()] = st
                continue
            self._annotate_file(filename, named, defined)


@dataclass
class AnalyzedDocument:
    """
    Result of parsing a document, either standalone or within its project.

    Attributes
    ----------
    path : pathlib.Path
        Key into ``files.by_filename`` for the analyzed document.
    files : MemoryFiles or None
        The parsed file set, or ``None`` if parsing raised.
    error : Exception or None
        The exception raised during parsing, if any.
    config : LatformProjectConfig or None
        The applicable project config, if one was discovered.
    project_root : pathlib.Path or None
        The project root when the document was resolved as part of a project
        tree; ``None`` for standalone (single-file) analysis.
    """

    path: pathlib.Path
    files: MemoryFiles | None = None
    error: Exception | None = None
    config: LatformProjectConfig | None = None
    project_root: pathlib.Path | None = None

    @property
    def statements(self) -> list[Statement]:
        """Statements parsed for this document (empty on error)."""
        if self.files is None:
            return []
        return list(self.files.by_filename.get(self.path, []))


def _document_key(files: MemoryFiles, resolved: pathlib.Path) -> pathlib.Path | None:
    """
    The ``by_filename`` key matching ``resolved``, or ``None`` if absent.

    ``Files`` stores keys as joined-but-not-canonicalized paths, so a resolved
    comparison is needed to match a document against the parsed tree.
    """
    for key in files.by_filename:
        if key == resolved or key.resolve() == resolved:
            return key
    return None


def _parse_files(
    top_files: list[pathlib.Path],
    contents: dict[pathlib.Path, str],
    parse_cache: ParseCache | None = None,
    annotate_state: dict | None = None,
) -> tuple[MemoryFiles | None, Exception | None]:
    """Parse and annotate a file set, returning ``(files, error)``."""
    files = _OverlayFiles(top_files=top_files, initial_contents=dict(contents))
    files._parse_cache = parse_cache
    files._annotate_state = annotate_state
    files._reparsed = set() if annotate_state is not None else None
    try:
        files.parse(raise_if_missing=False)
        files.annotate()
    except Exception as exc:  # parsing is best-effort; report as a diagnostic
        return None, exc
    return files, None


def _expand_top_files(
    config: LatformProjectConfig, contents: dict[pathlib.Path, str]
) -> tuple[list[pathlib.Path], list[TaoInit]]:
    """
    Resolve a config's ``top-level`` entries to Bmad lattice paths.

    A ``tao.init`` (Fortran namelist) entry is not a lattice; it is expanded
    into the lattice files it references via ``&tao_design_lattice`` so those
    are what get parsed.  The parsed `TaoInit` objects are returned alongside so
    callers can attach them (for tao-init lints).
    """
    top_files: list[pathlib.Path] = []
    tao_inits: list[TaoInit] = []
    for entry in config.resolve_top_level():
        entry = entry.resolve()
        text = contents.get(entry)
        if text is None:
            try:
                text = entry.read_text()
            except OSError:
                text = None
        if text is not None and (is_init_file(entry) or looks_like_namelist(text)):
            tao_init = TaoInit.parse(text, filename=entry)
            tao_init.load_sources(base=entry.parent, reader=contents.get)
            top_files.extend(_resolve_lattice_paths(tao_init.lattice_files, entry.parent))
            tao_inits.append(tao_init)
        else:
            top_files.append(entry)
    return top_files, tao_inits


def _build_project(
    config: LatformProjectConfig,
    contents: dict[pathlib.Path, str],
    parse_cache: ParseCache | None = None,
    annotate_state: dict | None = None,
) -> tuple[MemoryFiles | None, Exception | None]:
    """Parse a project's lattice tree, expanding any ``tao.init`` entries."""
    top_files, tao_inits = _expand_top_files(config, contents)
    files, error = _parse_files(top_files, contents, parse_cache, annotate_state)
    if files is not None and tao_inits:
        # TODO: with several tao.init entries, associate each with its own tree.
        files.tao_init = tao_inits[0]
    return files, error


def analyze(
    path: pathlib.Path | str,
    text: str,
    overlay: dict[pathlib.Path, str] | None = None,
    *,
    config: LatformProjectConfig | None = None,
    parse_cache: ParseCache | None = None,
) -> AnalyzedDocument:
    """
    Parse ``text`` as the document at ``path``, following ``call`` includes.

    When ``config`` declares ``top-level`` entries and ``path`` is reachable
    from them, the document is analyzed within the whole project tree so that
    cross-file references resolve.  Otherwise the document is analyzed
    standalone, as its own top-level entry point.

    Parameters
    ----------
    path : pathlib.Path or str
        Path of the document being analyzed; used to resolve relative includes.
    text : str
        The current (possibly unsaved) contents of the document.
    overlay : dict of pathlib.Path to str, optional
        Contents of other open buffers, so cross-file resolution prefers live
        editor state over what is on disk.
    config : LatformProjectConfig, optional
        Project config; enables project-tree resolution and lint settings.

    Returns
    -------
    AnalyzedDocument
    """
    resolved = pathlib.Path(path).resolve()
    contents: dict[pathlib.Path, str] = {
        pathlib.Path(p).resolve(): t for p, t in (overlay or {}).items()
    }
    contents[resolved] = text

    if config is not None and config.top_level:
        files, error = _build_project(config, contents, parse_cache)
        if files is not None:
            key = _document_key(files, resolved)
            if key is not None:
                return AnalyzedDocument(
                    path=key, files=files, config=config, project_root=config.root
                )
            # Not part of the project tree; fall through to standalone.
        else:
            logger.debug(
                "Project parse failed (%s); using standalone for %s", config.source, resolved
            )

    files, error = _parse_files([resolved], contents, parse_cache)
    if files is None:
        logger.debug("Parse failed for %s: %s", resolved, error)
        return AnalyzedDocument(path=resolved, files=None, error=error, config=config)
    key = _document_key(files, resolved) or resolved
    return AnalyzedDocument(path=key, files=files, config=config)


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
    from .graph import _generate_mermaid, _generate_tree_text

    files = analyzed.files
    if files is None or not files.top_files:
        return None
    return {
        "tree": _generate_tree_text(files),
        "mermaid": _generate_mermaid(files),
        "edges": [list(edge) for edge in files.call_graph_edges],
    }


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


def _statement_line_span(statement: Statement) -> tuple[int, int] | None:
    """The ``(start_line, end_line)`` a statement occupies, or ``None``."""
    locs = [tok.loc for tok in _statement_tokens(statement) if tok.loc is not None]
    if not locs:
        return None
    return min(loc.line for loc in locs), max(loc.end_line for loc in locs)


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


def hover_text(
    analyzed: AnalyzedDocument, line: int, char: int, document_text: str = ""
) -> str | None:
    """
    Markdown hover text for the symbol under a 0-indexed position, or ``None``.

    Resolves, in order: attribute names (element-type metadata), user-defined
    element/constant/line names, element-type keywords, and builtin functions
    and constants.

    ``document_text`` enables resolving an attribute inside ``NAME[…]`` from the
    line even when the statement has not parsed as a `Parameter` (mid-edit it
    may be a `Simple`/unknown statement, or the line may not parse at all).
    """
    named = analyzed.files.get_named_items() if analyzed.files is not None else {}
    tok, statement = _locate(analyzed.statements, line, char)

    # 1) An attribute in a fully parsed statement (element body or Parameter).
    if tok is not None and tok.role == Role.attribute_name:
        hover = _attribute_hover(tok, statement, named)
        if hover is not None:
            return hover

    # 2) An attribute inside ``NAME[…]``, resolved from the line text — robust to
    #    the statement not parsing as a Parameter yet.
    lines = document_text.splitlines()
    line_text = lines[line] if 0 <= line < len(lines) else ""
    owner = _bracket_owner(line_text, char)
    if owner is not None:
        attr = str(tok) if tok is not None else _word_at(line_text, char)
        if attr:
            hover = _attribute_of_target(owner, attr, named, document_text)
            if hover is not None:
                return hover

    if tok is None:
        return None

    # User definitions take precedence over builtins of the same name.
    defined = named.get(str(tok).upper())
    if defined is not None:
        return _named_hover(defined)

    if tok.role == Role.kind:
        element_type = _expand_element_type(str(tok))
        if element_type is not None:
            count = len(element_key_to_attrs.get(element_type, {}))
            return f"**{element_type}** — element type · {count} attributes"

    return _builtin_hover(tok)


def _named_hover(statement: Statement) -> str | None:
    """Hover text for a user-defined element/constant/line/list."""
    if isinstance(statement, Element):
        if statement.element_type and not statement.element_type.lower().startswith(
            statement.keyword.lower()
        ):
            kind = f"{statement.element_type} from {statement.keyword}"
        else:
            kind = statement.element_type or str(statement.keyword)
        return f"**{statement.name}** — element (`{kind}`)"
    if isinstance(statement, Constant):
        return f"**{statement.name}** — constant = `{_seq_text(statement.value)}`"
    if isinstance(statement, (Line, ElementList)):
        return f"**{definition_name_token(statement)}** — {type(statement).__name__.lower()}"
    return None


_KNOWN_PARAMETERS = {
    (str(param.target).lower(), str(param.name).lower()): param for param in Parameter.known
}


def _bracket_owner(line_text: str, col: int) -> str | None:
    """The ``NAME`` whose ``[`` encloses column ``col`` on this line, or ``None``."""
    open_idx = line_text.rfind("[", 0, col)
    if open_idx == -1 or "]" in line_text[open_idx:col]:
        return None
    match = re.search(r"([A-Za-z_][\w.%]*)\s*$", line_text[:open_idx])
    return match.group(1) if match else None


def _word_at(line_text: str, col: int) -> str:
    """The identifier spanning column ``col`` on this line."""
    start, end = col, col
    while start > 0 and (line_text[start - 1].isalnum() or line_text[start - 1] in "_.%"):
        start -= 1
    while end < len(line_text) and (line_text[end].isalnum() or line_text[end] in "_.%"):
        end += 1
    return line_text[start:end]


def _attribute_hover(tok: Token, statement: Statement | None, named: dict) -> str | None:
    """Hover for an attribute in a parsed element body or `Parameter`."""
    if isinstance(statement, Element) and statement.element_type:
        return _element_attribute_hover(statement.element_type, tok)
    if isinstance(statement, Parameter):
        return _attribute_of_target(str(statement.target), str(tok), named)
    return None


def _attribute_of_target(
    target: str, attr: str, named: dict, document_text: str = ""
) -> str | None:
    """
    Hover for attribute ``attr`` of ``target``.

    ``target`` may be a defined element (its type's attribute metadata) or a
    builtin target such as ``parameter`` (whose attributes come from a separate
    table).  The element type is resolved from the symbol table or, failing
    that, by scanning ``document_text`` — so this works before the statement
    parses.
    """
    element_type = _element_type_of(target, named, document_text)
    if element_type is not None:
        hover = _element_attribute_hover(element_type, Token(attr))
        if hover is not None:
            return hover
    known = _KNOWN_PARAMETERS.get((target.lower(), attr.lower()))
    if known is not None:
        type_name = known.type.__name__ if isinstance(known.type, type) else str(known.type)
        comment = f" — {known.comment}" if known.comment else ""
        return f"**{known.name}** — `{target}` parameter · {type_name}{comment}"
    return None


def _element_attribute_hover(element_type: str, tok: Token) -> str | None:
    attr = element_key_to_attrs.get(element_type, {}).get(str(tok).upper())
    if attr is None:
        return None
    kind = getattr(getattr(attr, "kind", None), "name", "")
    parts = [part for part in (kind.lower() if kind else "", attr.units or "") if part]
    desc = f": {attr.desc}" if attr.desc else ""
    detail = f" · {' · '.join(parts)}" if parts else ""
    return f"**{attr.name}** — attribute of `{element_type}`{detail}{desc}"


def _builtin_hover(tok: Token) -> str | None:
    """Hover for a builtin function, physical constant, or target."""
    name = str(tok)
    lower = name.lower()

    function = INTRINSIC_FUNCTIONS.get(lower)
    if function is not None:
        signature = f"{function.name}({', '.join(function.arguments)})"
        description = f" — {function.description}" if function.description else ""
        return f"**{signature}** — function{description}"

    if lower in named_physical_constants:
        return f"**{name}** — builtin constant = `{named_physical_constants[lower]:.10g}`"
    if lower in BUILTIN_CONSTANTS:
        return f"**{name}** — builtin constant"
    if lower in BUILTIN_TARGETS:
        return f"**{name}** — builtin target"
    return None


def _seq_text(value: Token | Seq) -> str:
    """Reconstruct a value's source text for display."""
    if isinstance(value, Token):
        return str(value)
    return str(value.to_token())


@dataclass
class Diagnostic:
    """A location-carrying diagnostic (parse error or lint)."""

    location: Location
    message: str
    code: str
    severity: str = "warning"  # "error" | "warning"
    related: list[tuple[Location, str]] = field(default_factory=list)


def _used_names(files: MemoryFiles) -> frozenset[str]:
    """
    Project-wide used names, memoized per build.

    ``get_used_names`` walks every token, so caching it on the (per-build) files
    keeps publishing diagnostics for several open documents from re-scanning the
    whole project each time.
    """
    cached = getattr(files, "_used_names_cache", None)
    if cached is None:
        cached = get_used_names([st for sts in files.by_filename.values() for st in sts])
        files._used_names_cache = cached
    return cached


def _file_lints(analyzed: AnalyzedDocument) -> list[Lint]:
    """Lints for the current document (empty if it did not parse)."""
    if analyzed.files is None:
        return []
    config = analyzed.config
    return lint_statements(
        list(analyzed.statements),
        named=analyzed.files.get_named_items(),
        assume_defined=False,
        ignore=config.ignores_for(analyzed.path) if config is not None else (),
        used_names=_used_names(analyzed.files),
        min_name_length=config.min_name_length if config is not None else 1,
        builtin_constant_rtol=config.builtin_constant_rtol if config is not None else 1e-4,
    )


def iter_diagnostics(analyzed: AnalyzedDocument) -> Generator[Diagnostic, None, None]:
    """
    Yield diagnostics for the analyzed document (parse errors and lints).
    """
    if analyzed.files is None:
        loc = _error_location(analyzed)
        yield Diagnostic(
            location=loc, message=str(analyzed.error), code="parse-error", severity="error"
        )
        return

    for lint in _file_lints(analyzed):
        primary, related = _lint_locations(lint)
        if primary is None:
            continue
        yield Diagnostic(
            location=primary,
            message=lint.message,
            code=lint.code.value,
            related=[(loc, "related occurrence") for loc in related],
        )


def _error_location(analyzed: AnalyzedDocument) -> Location:
    """A best-effort location for a parse error (line/col from the exception)."""
    loc = getattr(analyzed.error, "loc", None)
    if isinstance(loc, Location):
        return loc
    return Location(filename=analyzed.path, line=0, column=0, end_line=0, end_column=1)


def _lint_locations(lint: Lint) -> tuple[Location | None, list[Location]]:
    """
    The primary location and any related locations for a lint.

    Every lint lists its ``relevant_tokens`` with the *offending* occurrence
    last (an override/duplicate lists the original first, then the offending
    set); the diagnostic anchors on that last token so it lands on the edited
    line, and earlier tokens become related information.  Merging them into one
    span (the previous behaviour) produced a diagnostic covering the original
    definition too.
    """
    locs = [tok.loc for tok in (lint.relevant_tokens or []) if tok.loc is not None]
    if locs:
        return locs[-1], locs[:-1]
    return _context_location(lint.context), []


def _context_location(context: object) -> Location | None:
    """A fallback location from a lint's ``context`` when it is a statement."""
    if not isinstance(context, Statement):
        return None  # e.g. a tao_init Namelist, which carries no walkable tokens
    tokens = [tok for tok in _statement_tokens(context) if tok.loc is not None]
    return Location.from_items(tokens) if tokens else None


# --------------------------------------------------------------------------- #
# Code actions (pure latform)
# --------------------------------------------------------------------------- #


@dataclass
class TextEditSpec:
    """A single text edit: replace ``location`` with ``new_text``."""

    location: Location
    new_text: str


@dataclass
class CodeAction:
    """
    A code action (quick fix or refactor).

    ``kind`` uses LSP code-action kinds (e.g. ``"quickfix"``,
    ``"refactor.inline"``); ``diagnostic_code`` links a quick fix to the lint it
    resolves.
    """

    title: str
    kind: str
    edits: list[TextEditSpec]
    diagnostic_code: str | None = None
    preferred: bool = False


def _line_range(filename: pathlib.Path | None, first: int, last: int) -> Location:
    """A range covering whole lines ``[first, last]`` (to the start of last+1)."""
    return Location(filename=filename, line=first, column=0, end_line=last + 1, end_column=0)


def _statement_file(statement: Statement) -> pathlib.Path | None:
    """The file a statement's tokens come from."""
    for tok in _statement_tokens(statement):
        if tok.loc is not None:
            return tok.loc.filename
    return None


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


def _close_matches(word: str, candidates) -> list[str]:
    """Up to 3 case-insensitive close matches, returned in their display form."""
    by_lower: dict[str, str] = {}
    for cand in candidates:
        by_lower.setdefault(cand.lower(), cand)
    hits = difflib.get_close_matches(word.lower(), list(by_lower), n=3, cutoff=0.6)
    return [by_lower[h] for h in hits]


def _element_type_for(statement: Statement, named: dict) -> str | None:
    """The element type an attribute in ``statement`` belongs to."""
    if isinstance(statement, Element):
        return statement.element_type
    if isinstance(statement, Parameter):
        target = named.get(str(statement.target).upper())
        if isinstance(target, Element):
            return target.element_type
    return None


def _intersects(loc: Location, start_line: int, end_line: int) -> bool:
    return loc.line <= end_line and loc.end_line >= start_line


def code_actions(
    analyzed: AnalyzedDocument,
    start_line: int,
    start_char: int,
    end_line: int,
    end_char: int,
) -> list[CodeAction]:
    """
    Quick fixes and refactors for the given range/cursor.

    Quick fixes come from lints intersecting the range; refactors from the token
    at the range start.  Edits are computed here (pure) so the pygls handler is a
    thin adapter.
    """
    if analyzed.files is None:
        return []
    options = _format_options(analyzed.config)
    named = analyzed.files.get_named_items()
    actions: list[CodeAction] = []

    for lint in _file_lints(analyzed):
        primary, _related = _lint_locations(lint)
        if primary is None or not _intersects(primary, start_line, end_line):
            continue
        actions.extend(_quick_fixes(lint, analyzed, named, options))
        suppress = _suppress_action(lint.code.value, analyzed.config)
        if suppress is not None:
            actions.append(suppress)

    tok, statement = _locate(analyzed.statements, start_line, start_char)
    if tok is not None and statement is not None:
        actions.extend(_refactors(tok, statement, analyzed, named, options))

    return actions


def _replace_action(title, kind, code, loc, new_text, preferred=False) -> CodeAction:
    return CodeAction(
        title=title,
        kind=kind,
        edits=[TextEditSpec(loc, new_text)],
        diagnostic_code=code,
        preferred=preferred,
    )


def _quick_fixes(lint: Lint, analyzed, named: dict, options: FormatOptions) -> list[CodeAction]:
    code = lint.code.value
    tokens = [t for t in (lint.relevant_tokens or []) if t.loc is not None]
    offending = tokens[-1] if tokens else None
    out: list[CodeAction] = []

    if code == "LF002" and offending is not None:  # undefined reference
        defined = [
            str(definition_name_token(st))
            for st in named.values()
            if definition_name_token(st) is not None
        ]
        for cand in _close_matches(str(offending), defined):
            out.append(
                _replace_action(f"Change to '{cand}'", "quickfix", code, offending.loc, cand)
            )
        span = _statement_line_span(lint.context) if isinstance(lint.context, Statement) else None
        if span is not None:
            stub = f"{offending}: marker\n"
            fn = _statement_file(lint.context)
            out.append(
                CodeAction(
                    title=f"Create element '{offending}: marker'",
                    kind="quickfix",
                    edits=[TextEditSpec(_line_range(fn, span[0], span[0] - 1), stub)],
                    diagnostic_code=code,
                )
            )

    elif code == "LF003" and offending is not None:  # unknown element type
        for cand in _close_matches(str(offending), (t.lower() for t in _ELEMENT_TYPES)):
            text = _apply_case(cand, options.kind_case)
            out.append(
                _replace_action(f"Change to '{text}'", "quickfix", code, offending.loc, text)
            )

    elif code == "LF004" and offending is not None:  # unknown attribute
        element_type = _element_type_for(lint.context, named)
        candidates = element_key_to_attrs.get(element_type, {}) if element_type else {}
        for cand in _close_matches(str(offending), candidates):
            text = _apply_case(cand, options.attribute_case)
            out.append(
                _replace_action(f"Change to '{text}'", "quickfix", code, offending.loc, text)
            )

    elif code == "LF006" and isinstance(lint.context, Element) and offending is not None:
        element = lint.context
        replacement = copy.copy(element)
        replacement.attributes = [
            a for a in element.attributes if not (isinstance(a, Attribute) and a.name is offending)
        ]
        edit = _rewrite(element, replacement, options)
        if edit is not None:
            out.append(
                CodeAction(
                    title=f"Remove duplicate attribute '{offending}'",
                    kind="quickfix",
                    edits=[edit],
                    diagnostic_code=code,
                    preferred=True,
                )
            )

    elif code == "LF007" and isinstance(lint.context, Statement):  # unused constant
        edit = _rewrite(lint.context, None, options)
        if edit is not None:
            name = definition_name_token(lint.context)
            out.append(
                CodeAction(
                    title=f"Remove unused constant '{name}'",
                    kind="quickfix",
                    edits=[edit],
                    diagnostic_code=code,
                    preferred=True,
                )
            )

    elif code == "LF008" and isinstance(lint.context, Statement):  # attribute override
        edit = _rewrite(lint.context, None, options)
        if edit is not None:
            out.append(
                CodeAction(
                    title="Remove overriding statement",
                    kind="quickfix",
                    edits=[edit],
                    diagnostic_code=code,
                )
            )

    elif code == "LF010" and offending is not None:  # value matches a builtin constant
        rtol = analyzed.config.builtin_constant_rtol if analyzed.config is not None else 1e-4
        try:
            matches = _matching_builtin_names(float(str(offending)), rtol)
        except ValueError:
            matches = []
        for name in matches:
            out.append(
                _replace_action(
                    f"Use built-in constant '{name}'", "quickfix", code, offending.loc, name
                )
            )

    return out


def _refactors(
    tok: Token, statement: Statement, analyzed, named: dict, options: FormatOptions
) -> list[CodeAction]:
    out: list[CodeAction] = []
    loc = tok.loc

    # Expand an attribute abbreviation to its full name.
    if tok.role == Role.attribute_name and loc is not None:
        element_type = _element_type_for(statement, named)
        full_names = element_key_to_attrs.get(element_type, {}) if element_type else {}
        upper = str(tok).upper()
        if upper not in full_names:
            expansions = [n for n in full_names if n.startswith(upper) and n != upper]
            for full in expansions:
                text = _apply_case(full, options.attribute_case)
                out.append(
                    CodeAction(
                        title=f"Expand to '{text}'",
                        kind="refactor.rewrite",
                        edits=[TextEditSpec(loc, text)],
                    )
                )

    # Inline a constant: replace its references with the value and delete it.
    if tok.role == Role.name_ and loc is not None:
        target = named.get(str(tok).upper())
        if isinstance(target, Constant):
            value = _seq_text(target.value)
            edits = [
                TextEditSpec(ref, value)
                for ref in find_references(
                    analyzed, loc.line, loc.column, include_declaration=False
                )
            ]
            delete = _rewrite(target, None, options)
            if edits and delete is not None:
                out.append(
                    CodeAction(
                        title=f"Inline constant '{target.name}'",
                        kind="refactor.inline",
                        edits=[*edits, delete],
                    )
                )

    # Extract a numeric literal into a new constant.
    if tok.role is None and loc is not None and _is_number(str(tok)):
        name = "new_const"
        out.append(
            CodeAction(
                title="Extract to constant",
                kind="refactor.extract",
                edits=[
                    TextEditSpec(_line_range(loc.filename, 0, -1), f"{name} = {tok}\n"),
                    TextEditSpec(loc, name),
                ],
            )
        )

    # Reformat the enclosing statement per project settings.
    reformat = _rewrite(statement, statement, options)
    if reformat is not None:
        out.append(
            CodeAction(title="Reformat statement", kind="refactor.rewrite", edits=[reformat])
        )

    return out


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _suppress_action(code: str, config) -> CodeAction | None:
    """
    Add ``code`` to ``[lint] ignore`` in the project config (best effort).

    Only offered when the config has an ``ignore = [...]`` list to extend.
    """
    if config is None or config.source is None:
        return None
    try:
        text = config.source.read_text()
    except OSError:
        return None
    match = re.search(r"ignore\s*=\s*\[", text)
    if match is None:
        return None
    insert = match.end()  # just after the "["
    close = text.find("]", insert)
    if close == -1:
        return None
    is_empty = not text[insert:close].strip()
    addition = f'"{code}"' if is_empty else f'"{code}", '
    line = text.count("\n", 0, insert)
    col = insert - (text.rfind("\n", 0, insert) + 1)
    edit = TextEditSpec(
        Location(filename=config.source, line=line, column=col, end_line=line, end_column=col),
        addition,
    )
    return CodeAction(
        title=f"Suppress {code} in {config.source.name}",
        kind="quickfix",
        edits=[edit],
        diagnostic_code=code,
    )


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
# Completion (pure latform)
# --------------------------------------------------------------------------- #


@dataclass
class Completion:
    """A single completion candidate."""

    label: str
    kind: str  # "type" | "attribute" | "element" | "line" | "list" | "constant" | "function"
    detail: str | None = None


# ``NAME[`` with the cursor inside the (still-open) brackets.
_ATTR_BRACKET_RE = re.compile(r"([A-Za-z_][\w.%]*)\[[^\]\[]*$")
# ``NAME:`` at the start of a line (an element/line definition).
_ELEMENT_DEF_RE = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*:\s*(.*)$")


def _open_context(prefix: str, opener: str, closer: str) -> bool:
    """Whether the cursor sits inside an unclosed ``opener`` on this line."""
    depth = 0
    for ch in prefix:
        if ch == opener:
            depth += 1
        elif ch == closer and depth > 0:
            depth -= 1
    return depth > 0


def _element_type_of(name: str, named: dict, document_text: str, _depth: int = 0) -> str | None:
    """
    The canonical element type of ``name``.

    Uses the parsed symbol table when available, else falls back to scanning the
    buffer for ``name: <keyword>`` so completion still works while the current
    line (and thus the parse) is incomplete.  Follows one level of inheritance.
    """
    statement = named.get(name.upper())
    if isinstance(statement, Element) and statement.element_type:
        return statement.element_type

    match = re.search(
        rf"^\s*{re.escape(name)}\s*:\s*([A-Za-z_][\w.]*)",
        document_text,
        re.IGNORECASE | re.MULTILINE,
    )
    if match is None:
        return None
    keyword = match.group(1)
    expanded = _expand_element_type(keyword)
    if expanded is not None:
        return expanded
    if _depth < 3 and keyword.upper() != name.upper():
        return _element_type_of(keyword, named, document_text, _depth + 1)  # base element
    return None


def _attribute_completions(element_type: str) -> list[Completion]:
    out = []
    for attr_name, attr in element_key_to_attrs.get(element_type, {}).items():
        kind = getattr(getattr(attr, "kind", None), "name", "")
        units = getattr(attr, "units", "") or ""
        detail = " · ".join(part for part in (kind.lower() if kind else "", units) if part)
        # Raw (uppercase) label; project casing is applied by `complete`.
        out.append(Completion(label=attr_name, kind="attribute", detail=detail or None))
    return out


def _type_completions() -> list[Completion]:
    return [Completion(label=t, kind="type", detail="element type") for t in _ELEMENT_TYPES]


def _symbol_completions(named: dict, kinds: tuple[str, ...] | None = None) -> list[Completion]:
    out = []
    for statement in named.values():
        name_token = definition_name_token(statement)
        if name_token is None:
            continue
        if isinstance(statement, Element):
            kind, detail = "element", statement.element_type or str(statement.keyword)
        elif isinstance(statement, Line):
            kind, detail = "line", "beamline"
        elif isinstance(statement, ElementList):
            kind, detail = "list", "element list"
        elif isinstance(statement, Constant):
            kind, detail = "constant", _seq_text(statement.value)
        else:
            continue
        if kinds and kind not in kinds:
            continue
        out.append(Completion(label=str(name_token), kind=kind, detail=detail))
    return out


def _value_completions(named: dict) -> list[Completion]:
    out = _symbol_completions(named, kinds=("constant",))
    out += [
        Completion(label=name, kind="function", detail="function") for name in INTRINSIC_FUNCTIONS
    ]
    out += [
        Completion(label=name, kind="builtin", detail="builtin constant")
        for name in BUILTIN_CONSTANTS
    ]
    return out


# Completion kind -> the `FormatOptions` case field that governs its label.
_CASE_FIELD_BY_KIND = {
    "type": "kind_case",
    "attribute": "attribute_case",
    "element": "name_case",
    "line": "name_case",
    "list": "name_case",
    "constant": "name_case",
    "function": "builtin_case",
    "builtin": "builtin_case",
}


def _format_options(config: LatformProjectConfig | None) -> FormatOptions:
    """`FormatOptions` from a project config (defaults when none applies)."""
    if config is None:
        return FormatOptions()
    return FormatOptions(**{k: v for k, v in config.format.items() if k in _FORMAT_OPTION_FIELDS})


def _apply_case(text: str, case: str) -> str:
    """
    Case ``text`` per a `NameCase` setting, mirroring `output.py`.

    The length attribute ``l`` is always rendered ``L`` regardless of the
    setting, matching the formatter's special case.
    """
    if case == "upper" or text.lower() == "l":
        return text.upper()
    if case == "lower":
        return text.lower()
    return text


def complete(
    analyzed: AnalyzedDocument, line_prefix: str, document_text: str = ""
) -> list[Completion]:
    """
    Completion candidates for a cursor at the end of ``line_prefix``.

    Context is inferred from the text before the cursor (which stays reliable
    while the document is mid-edit):

    - inside ``NAME[…`` → attribute names of ``NAME``'s element type;
    - inside ``(…`` → beamline/element names (line contents);
    - ``NAME: <first token>`` → element types and element names (base elements);
    - after an attribute comma → attribute names (or values after ``=``);
    - otherwise → all defined names (references).

    Labels are cased to match the project's format settings (e.g. uppercased
    element names, lowercased builtins), so an accepted completion reads the
    same as formatted output.
    """
    if "!" in line_prefix:  # cursor is within a comment
        return []

    candidates = _context_completions(analyzed, line_prefix, document_text)
    options = _format_options(analyzed.config)
    for candidate in candidates:
        case = getattr(options, _CASE_FIELD_BY_KIND.get(candidate.kind, ""), "same")
        candidate.label = _apply_case(candidate.label, case)
    return candidates


def _context_completions(
    analyzed: AnalyzedDocument, line_prefix: str, document_text: str
) -> list[Completion]:
    named = analyzed.files.get_named_items() if analyzed.files is not None else {}

    bracket = _ATTR_BRACKET_RE.search(line_prefix)
    if bracket is not None and _open_context(line_prefix, "[", "]"):
        element_type = _element_type_of(bracket.group(1), named, document_text)
        return _attribute_completions(element_type) if element_type else []

    if _open_context(line_prefix, "(", ")"):
        return _symbol_completions(named, kinds=("element", "line", "list"))

    definition = _ELEMENT_DEF_RE.match(line_prefix)
    if definition is not None:
        rest = definition.group(2)
        if "," not in rest and "=" not in rest:
            # type / base-element position (first token after the colon)
            return _type_completions() + _symbol_completions(named, kinds=("element",))
        segment = rest.rsplit(",", 1)[-1]
        if "=" in segment:
            return _value_completions(named)
        keyword = rest.split(",", 1)[0].strip()
        element_type = _expand_element_type(keyword) or _element_type_of(
            keyword, named, document_text
        )
        return _attribute_completions(element_type) if element_type else []

    if re.match(r"^\s*use\b", line_prefix, re.IGNORECASE):
        return _symbol_completions(named, kinds=("line", "element", "list"))

    return _symbol_completions(named)


# --------------------------------------------------------------------------- #
# Workspace: open-buffer state + project discovery (pure latform)
# --------------------------------------------------------------------------- #


@dataclass
class Workspace:
    """
    Tracks open editor buffers and resolves each document against its project.

    Configuration is discovered by walking up from each document's directory
    (see `discover_config`); a project's parsed file set is cached and shared
    across all its open documents, so analyzing several files of one project
    costs a single parse per edit.
    """

    config_enabled: bool = True
    open_texts: dict[pathlib.Path, str] = field(default_factory=dict)
    _config_by_dir: dict[pathlib.Path, LatformProjectConfig] = field(default_factory=dict)
    # Per-project cache: config source -> (open-buffer signature, parsed files).
    _project_cache: dict[pathlib.Path | None, tuple[tuple, MemoryFiles]] = field(
        default_factory=dict
    )
    # Per-file parse cache: path -> (contents, statements). Survives edits so an
    # edit only re-parses the changed file; unchanged files reuse their
    # statements.
    _parse_cache: ParseCache = field(default_factory=dict)
    # Incremental-annotation state (last definition signature). Lets a rebuild
    # re-annotate only the re-parsed files when no definition changed.
    _annotate_state: dict = field(default_factory=dict)

    def set_text(self, path: pathlib.Path | str, text: str) -> pathlib.Path:
        """Record the current text of an open document; returns its resolved path."""
        resolved = pathlib.Path(path).resolve()
        self.open_texts[resolved] = text
        self._project_cache.clear()
        return resolved

    def close(self, path: pathlib.Path | str) -> None:
        """Forget an open document."""
        self.open_texts.pop(pathlib.Path(path).resolve(), None)
        self._project_cache.clear()

    def invalidate(self) -> None:
        """
        Drop cached configs and parsed projects.

        Called when files change on disk (or a config file is edited) so the
        next analysis re-discovers config and re-reads unopened files.
        """
        self._config_by_dir.clear()
        self._project_cache.clear()
        self._parse_cache.clear()
        self._annotate_state.clear()

    def config_for(self, path: pathlib.Path | str) -> LatformProjectConfig:
        """Discover (and cache) the project config applicable to ``path``."""
        directory = pathlib.Path(path).resolve().parent
        if directory not in self._config_by_dir:
            self._config_by_dir[directory] = discover_config(
                start=directory, enabled=self.config_enabled
            )
        return self._config_by_dir[directory]

    def text_of(self, path: pathlib.Path | str) -> str:
        """Current text of an open document, or its on-disk contents."""
        return self._text_of(pathlib.Path(path).resolve())

    def _text_of(self, resolved: pathlib.Path) -> str:
        text = self.open_texts.get(resolved)
        if text is not None:
            return text
        try:
            return resolved.read_text()
        except OSError:
            return ""

    def _signature(self) -> tuple:
        return tuple(sorted(self.open_texts.items()))

    def _project_files(self, config: LatformProjectConfig) -> MemoryFiles | None:
        """Parse (and cache) the project tree with all open buffers overlaid."""
        signature = self._signature()
        cached = self._project_cache.get(config.source)
        if cached is not None and cached[0] == signature:
            return cached[1]
        files, error = _build_project(
            config, self.open_texts, self._parse_cache, self._annotate_state
        )
        if files is None:
            logger.debug("Project parse failed (%s): %s", config.source, error)
            return None
        self._project_cache[config.source] = (signature, files)
        return files

    def analyze(self, path: pathlib.Path | str) -> AnalyzedDocument:
        """
        Analyze an open (or on-disk) document, within its project when possible.
        """
        resolved = pathlib.Path(path).resolve()
        text = self._text_of(resolved)
        config = self.config_for(resolved)

        if config.top_level:
            files = self._project_files(config)
            if files is not None:
                key = _document_key(files, resolved)
                if key is not None:
                    logger.debug(
                        "Analyze %s: project mode (root=%s, %d files in tree)",
                        resolved,
                        config.root,
                        len(files.by_filename),
                    )
                    return AnalyzedDocument(
                        path=key, files=files, config=config, project_root=config.root
                    )
                logger.debug(
                    "Analyze %s: not in project tree (%s); standalone", resolved, config.source
                )
        else:
            logger.debug("Analyze %s: no project config; standalone", resolved)

        overlay = {p: t for p, t in self.open_texts.items() if p != resolved}
        return analyze(resolved, text, overlay, config=config, parse_cache=self._parse_cache)


# --------------------------------------------------------------------------- #
# pygls glue
# --------------------------------------------------------------------------- #


def create_server(
    name: str = "latform-lsp",
    version: str = "0.1.0",
    *,
    client_log_level: int | None = None,
):
    """
    Build and return a configured `pygls` `LanguageServer`.

    Parameters
    ----------
    name, version : str
        Server identity reported to the client.
    client_log_level : int, optional
        When set, latform log records at or above this level are also forwarded
        to the client via ``window/logMessage`` (visible in the client's LSP
        output), in addition to stderr/file logging.

    Raises
    ------
    ImportError
        If ``pygls`` is not installed.
    """
    try:
        from lsprotocol import types as lsp
        from pygls.lsp.server import LanguageServer
        from pygls.uris import to_fs_path
    except ImportError as exc:  # pragma: no cover - exercised only without pygls
        raise ImportError(
            "The latform language server requires 'pygls'. "
            "Install it with: pip install 'latform[lsp]'"
        ) from exc

    server = LanguageServer(name, version, text_document_sync_kind=lsp.TextDocumentSyncKind.Full)
    workspace = Workspace()

    if client_log_level is not None:
        _attach_client_log_handler(server, lsp, client_log_level)

    def _uri_to_path(uri: str) -> pathlib.Path:
        return pathlib.Path(to_fs_path(uri)).resolve()

    _SEVERITY = {
        "error": lsp.DiagnosticSeverity.Error,
        "warning": lsp.DiagnosticSeverity.Warning,
    }
    _SYMBOL_KIND = {
        "element": lsp.SymbolKind.Class,
        "line": lsp.SymbolKind.Array,
        "list": lsp.SymbolKind.Array,
        "constant": lsp.SymbolKind.Constant,
    }
    _COMPLETION_KIND = {
        "type": lsp.CompletionItemKind.Class,
        "attribute": lsp.CompletionItemKind.Field,
        "element": lsp.CompletionItemKind.Variable,
        "line": lsp.CompletionItemKind.Module,
        "list": lsp.CompletionItemKind.Module,
        "constant": lsp.CompletionItemKind.Constant,
        "function": lsp.CompletionItemKind.Function,
        "builtin": lsp.CompletionItemKind.Constant,
    }

    def _range(loc: Location) -> "lsp.Range":
        """Convert a latform `Location` (inclusive end_column) to an LSP range."""
        return lsp.Range(
            start=lsp.Position(line=loc.line, character=loc.column),
            end=lsp.Position(line=loc.end_line, character=loc.end_column),
        )

    def _related(diag: Diagnostic):
        info = [
            lsp.DiagnosticRelatedInformation(
                location=lsp.Location(uri=loc.filename.resolve().as_uri(), range=_range(loc)),
                message=message,
            )
            for loc, message in diag.related
            if loc.filename is not None
        ]
        return info or None

    open_uris: dict[pathlib.Path, str] = {}

    def _publish(uri: str) -> None:
        analyzed = workspace.analyze(_uri_to_path(uri))
        diagnostics = [
            lsp.Diagnostic(
                range=_range(diag.location),
                message=diag.message,
                code=diag.code,
                severity=_SEVERITY.get(diag.severity, lsp.DiagnosticSeverity.Warning),
                source="latform",
                related_information=_related(diag),
            )
            for diag in iter_diagnostics(analyzed)
        ]
        logger.debug("Publishing %d diagnostic(s) for %s", len(diagnostics), uri)
        server.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
        )

    def _publish_all() -> None:
        # An edit in one project file can change diagnostics in its siblings.
        for uri in list(open_uris.values()):
            _publish(uri)

    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    def did_open(params: lsp.DidOpenTextDocumentParams) -> None:
        uri = params.text_document.uri
        logger.debug("didOpen %s", uri)
        open_uris[_uri_to_path(uri)] = uri
        workspace.set_text(_uri_to_path(uri), params.text_document.text)
        _publish_all()

    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    def did_change(params: lsp.DidChangeTextDocumentParams) -> None:
        # Full-sync (see create_server): the sole content change is the whole
        # document.
        uri = params.text_document.uri
        logger.debug("didChange %s", uri)
        open_uris[_uri_to_path(uri)] = uri
        workspace.set_text(_uri_to_path(uri), params.content_changes[-1].text)
        _publish_all()

    @server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
    def did_close(params: lsp.DidCloseTextDocumentParams) -> None:
        uri = params.text_document.uri
        logger.debug("didClose %s", uri)
        open_uris.pop(_uri_to_path(uri), None)
        workspace.close(_uri_to_path(uri))
        server.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(uri=uri, diagnostics=[])
        )
        _publish_all()

    @server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
    def definition(params: lsp.DefinitionParams):
        pos = params.position
        logger.debug("definition %s @ %d:%d", params.text_document.uri, pos.line, pos.character)
        analyzed = workspace.analyze(_uri_to_path(params.text_document.uri))
        loc = resolve_definition(analyzed, pos.line, pos.character)
        if loc is None or loc.filename is None:
            logger.debug("definition: no result")
            return None
        logger.debug("definition -> %s", loc)
        return lsp.Location(uri=loc.filename.resolve().as_uri(), range=_range(loc))

    @server.feature(lsp.TEXT_DOCUMENT_REFERENCES)
    def references(params: lsp.ReferenceParams):
        pos = params.position
        logger.debug("references %s @ %d:%d", params.text_document.uri, pos.line, pos.character)
        analyzed = workspace.analyze(_uri_to_path(params.text_document.uri))
        locs = find_references(
            analyzed,
            pos.line,
            pos.character,
            include_declaration=params.context.include_declaration,
        )
        found = [
            lsp.Location(uri=loc.filename.resolve().as_uri(), range=_range(loc))
            for loc in locs
            if loc.filename is not None
        ]
        logger.debug("references -> %d occurrence(s)", len(found))
        return found or None

    @server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_HIGHLIGHT)
    def document_highlight(params: lsp.DocumentHighlightParams):
        pos = params.position
        analyzed = workspace.analyze(_uri_to_path(params.text_document.uri))
        return [
            lsp.DocumentHighlight(
                range=_range(loc),
                kind=(
                    lsp.DocumentHighlightKind.Write
                    if is_definition
                    else lsp.DocumentHighlightKind.Read
                ),
            )
            for loc, is_definition in document_highlights(analyzed, pos.line, pos.character)
        ] or None

    @server.feature(
        lsp.TEXT_DOCUMENT_CODE_ACTION,
        lsp.CodeActionOptions(
            code_action_kinds=[
                lsp.CodeActionKind.QuickFix,
                lsp.CodeActionKind.RefactorRewrite,
                lsp.CodeActionKind.RefactorInline,
                lsp.CodeActionKind.RefactorExtract,
            ]
        ),
    )
    def code_action(params: lsp.CodeActionParams):
        analyzed = workspace.analyze(_uri_to_path(params.text_document.uri))
        rng = params.range
        specs = code_actions(
            analyzed, rng.start.line, rng.start.character, rng.end.line, rng.end.character
        )
        context_diagnostics = params.context.diagnostics or []
        result = []
        for spec in specs:
            changes: dict[str, list] = {}
            for edit in spec.edits:
                if edit.location.filename is None:
                    continue
                uri = edit.location.filename.resolve().as_uri()
                changes.setdefault(uri, []).append(
                    lsp.TextEdit(range=_range(edit.location), new_text=edit.new_text)
                )
            diagnostics = [
                d
                for d in context_diagnostics
                if spec.diagnostic_code and d.code == spec.diagnostic_code
            ]
            result.append(
                lsp.CodeAction(
                    title=spec.title,
                    kind=lsp.CodeActionKind(spec.kind),
                    edit=lsp.WorkspaceEdit(changes=changes),
                    diagnostics=diagnostics or None,
                    is_preferred=spec.preferred or None,
                )
            )
        logger.debug("codeAction %s -> %d action(s)", params.text_document.uri, len(result))
        return result or None

    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    def hover(params: lsp.HoverParams):
        pos = params.position
        logger.debug("hover %s @ %d:%d", params.text_document.uri, pos.line, pos.character)
        path = _uri_to_path(params.text_document.uri)
        analyzed = workspace.analyze(path)
        text = hover_text(analyzed, pos.line, pos.character, workspace.text_of(path))
        if text is None:
            return None
        return lsp.Hover(contents=lsp.MarkupContent(kind=lsp.MarkupKind.Markdown, value=text))

    @server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
    def document_symbol(params: lsp.DocumentSymbolParams):
        logger.debug("documentSymbol %s", params.text_document.uri)
        analyzed = workspace.analyze(_uri_to_path(params.text_document.uri))
        symbols = []
        for name, kind, loc in document_symbols(analyzed):
            rng = _range(loc)
            symbols.append(
                lsp.DocumentSymbol(
                    name=name,
                    kind=_SYMBOL_KIND.get(kind, lsp.SymbolKind.Variable),
                    range=rng,
                    selection_range=rng,
                )
            )
        return symbols

    @server.feature(lsp.WORKSPACE_SYMBOL)
    def workspace_symbol(params: lsp.WorkspaceSymbolParams):
        logger.debug("workspaceSymbol query=%r", params.query)
        # One open document per project surfaces that project's whole symbol
        # table (the tree is shared); analyze one per project and dedupe.
        seen_projects: set = set()
        results = []
        for path in list(workspace.open_texts):
            analyzed = workspace.analyze(path)
            if analyzed.files is None:
                continue
            project_key = analyzed.project_root or analyzed.path
            if project_key in seen_projects:
                continue
            seen_projects.add(project_key)
            for name, kind, loc in workspace_symbols(analyzed, params.query):
                if loc.filename is None:
                    continue
                results.append(
                    lsp.WorkspaceSymbol(
                        name=name,
                        kind=_SYMBOL_KIND.get(kind, lsp.SymbolKind.Variable),
                        location=lsp.Location(
                            uri=loc.filename.resolve().as_uri(), range=_range(loc)
                        ),
                    )
                )
        logger.debug("workspaceSymbol -> %d result(s)", len(results))
        return results

    @server.feature(
        lsp.TEXT_DOCUMENT_COMPLETION,
        # A space is a trigger so attribute completion reopens after ``, `` (the
        # canonical spacing), not only on the comma itself.
        lsp.CompletionOptions(trigger_characters=[":", "[", ",", "(", " "]),
    )
    def completion(params: lsp.CompletionParams):
        uri = params.text_document.uri
        pos = params.position
        path = _uri_to_path(uri)
        text = workspace.text_of(path)
        lines = text.splitlines()
        line_prefix = lines[pos.line][: pos.character] if pos.line < len(lines) else ""
        analyzed = workspace.analyze(path)
        items = [
            lsp.CompletionItem(
                label=candidate.label,
                kind=_COMPLETION_KIND.get(candidate.kind, lsp.CompletionItemKind.Text),
                detail=candidate.detail,
            )
            for candidate in complete(analyzed, line_prefix, text)
        ]
        logger.debug(
            "completion %s @ %d:%d -> %d item(s)", uri, pos.line, pos.character, len(items)
        )
        return lsp.CompletionList(is_incomplete=False, items=items)

    @server.feature(lsp.TEXT_DOCUMENT_PREPARE_RENAME)
    def prepare_rename_(params: lsp.PrepareRenameParams):
        analyzed = workspace.analyze(_uri_to_path(params.text_document.uri))
        loc = prepare_rename(analyzed, params.position.line, params.position.character)
        return _range(loc) if loc is not None else None

    @server.feature(lsp.TEXT_DOCUMENT_RENAME, lsp.RenameOptions(prepare_provider=True))
    def rename(params: lsp.RenameParams):
        pos = params.position
        analyzed = workspace.analyze(_uri_to_path(params.text_document.uri))
        locs = find_references(analyzed, pos.line, pos.character, include_declaration=True)
        changes: dict[str, list] = {}
        for loc in locs:
            if loc.filename is None:
                continue
            uri = loc.filename.resolve().as_uri()
            changes.setdefault(uri, []).append(
                lsp.TextEdit(range=_range(loc), new_text=params.new_name)
            )
        logger.debug(
            "rename -> %d edit(s) across %d file(s)", sum(map(len, changes.values())), len(changes)
        )
        return lsp.WorkspaceEdit(changes=changes) if changes else None

    def _full_document_edit(doc: str, formatted: str) -> "lsp.TextEdit":
        lines = doc.split("\n")
        end = lsp.Position(line=len(lines) - 1, character=len(lines[-1]))
        return lsp.TextEdit(range=lsp.Range(lsp.Position(0, 0), end), new_text=formatted)

    @server.feature(lsp.TEXT_DOCUMENT_FORMATTING)
    def formatting(params: lsp.DocumentFormattingParams):
        path = _uri_to_path(params.text_document.uri)
        formatted = format_document(workspace.analyze(path))
        if formatted is None:
            return None
        return [_full_document_edit(workspace.text_of(path), formatted)]

    @server.feature(lsp.TEXT_DOCUMENT_RANGE_FORMATTING)
    def range_formatting(params: lsp.DocumentRangeFormattingParams):
        path = _uri_to_path(params.text_document.uri)
        result = format_range(
            workspace.analyze(path), params.range.start.line, params.range.end.line
        )
        if result is None:
            return None
        first, last, formatted = result
        lines = workspace.text_of(path).split("\n")
        end_char = len(lines[last]) if last < len(lines) else 0
        rng = lsp.Range(lsp.Position(first, 0), lsp.Position(last, end_char))
        return [lsp.TextEdit(range=rng, new_text=formatted)]

    @server.feature(
        lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
        lsp.SemanticTokensLegend(
            token_types=list(SEMANTIC_TOKEN_TYPES),
            token_modifiers=list(SEMANTIC_TOKEN_MODIFIERS),
        ),
    )
    def semantic_tokens_full(params: lsp.SemanticTokensParams):
        analyzed = workspace.analyze(_uri_to_path(params.text_document.uri))
        data: list[int] = []
        prev_line = prev_char = 0
        for line, char, length, ttype, mods in semantic_tokens(analyzed):
            delta_line = line - prev_line
            delta_char = char - prev_char if delta_line == 0 else char
            data.extend((delta_line, delta_char, length, ttype, mods))
            prev_line, prev_char = line, char
        return lsp.SemanticTokens(data=data)

    @server.command("latform.fileDependencies")
    def latform_file_dependencies(uri: str):
        # Invoked via workspace/executeCommand with ``arguments: [uri]``; returns
        # the ``call`` include graph (text tree + mermaid + edges) for the
        # document's project, or null if it did not parse.
        logger.debug("command latform.fileDependencies %s", uri)
        return file_dependencies(workspace.analyze(_uri_to_path(uri)))

    _WATCH_GLOBS = ("**/*.bmad", "**/*.lat", "**/latform.toml", "**/pyproject.toml")

    @server.feature(lsp.INITIALIZED)
    def initialized(params: lsp.InitializedParams) -> None:
        logger.info("latform-lsp %s initialized", version)
        # Ask the client to notify us when lattice or config files change on
        # disk, so cached parses of unopened files stay fresh.
        try:
            server.client_register_capability(
                lsp.RegistrationParams(
                    registrations=[
                        lsp.Registration(
                            id="latform-watched-files",
                            method=lsp.WORKSPACE_DID_CHANGE_WATCHED_FILES,
                            register_options=lsp.DidChangeWatchedFilesRegistrationOptions(
                                watchers=[
                                    lsp.FileSystemWatcher(glob_pattern=glob)
                                    for glob in _WATCH_GLOBS
                                ]
                            ),
                        )
                    ]
                )
            )
            logger.debug("Registered file watchers: %s", ", ".join(_WATCH_GLOBS))
        except Exception as exc:  # client may not support dynamic registration
            logger.debug("Could not register file watchers: %s", exc)

    @server.feature(lsp.WORKSPACE_DID_CHANGE_WATCHED_FILES)
    def did_change_watched_files(params: lsp.DidChangeWatchedFilesParams) -> None:
        # A lattice or config file changed on disk: drop caches (config may have
        # changed; unopened files must be re-read) and refresh every open doc.
        logger.debug(
            "didChangeWatchedFiles: %s",
            ", ".join(change.uri for change in params.changes) or "(none)",
        )
        workspace.invalidate()
        _publish_all()

    return server


def _attach_client_log_handler(server, lsp, level: int) -> None:
    """
    Forward latform log records (>= ``level``) to the client via logMessage.

    Attached only to the ``latform`` logger to avoid feedback with pygls's own
    logging of the outgoing notifications.
    """
    msg_type = {
        logging.DEBUG: lsp.MessageType.Log,
        logging.INFO: lsp.MessageType.Info,
        logging.WARNING: lsp.MessageType.Warning,
        logging.ERROR: lsp.MessageType.Error,
        logging.CRITICAL: lsp.MessageType.Error,
    }

    class _ClientLogHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                kind = msg_type.get(record.levelno, lsp.MessageType.Log)
                server.window_log_message(
                    lsp.LogMessageParams(type=kind, message=self.format(record))
                )
            except Exception:  # never let logging break request handling
                pass

    handler = _ClientLogHandler(level)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logging.getLogger("latform").addHandler(handler)


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the ``latform-lsp`` command-line parser."""
    parser = argparse.ArgumentParser(prog="latform-lsp", description=__doc__)
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=os.environ.get(_ENV_LOG_LEVEL, "warning").lower(),
        help="Logging verbosity (default: warning, or $%s)." % _ENV_LOG_LEVEL,
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get(_ENV_LOG_FILE),
        help="Write logs to this file instead of stderr (or $%s)." % _ENV_LOG_FILE,
    )
    parser.add_argument(
        "--no-client-log",
        action="store_true",
        help="Do not mirror log messages to the client via window/logMessage.",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Communicate over stdio (the default and only transport; accepted "
        "for clients that pass it explicitly).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Console-script entry point: start the language server on stdio.

    The log level and log file default to the ``LATFORM_LSP_LOG_LEVEL`` and
    ``LATFORM_LSP_LOG_FILE`` environment variables, and can be overridden with
    ``--log-level`` / ``--log-file``.  Both mechanisms work from any LSP client
    that can pass process arguments or environment (e.g. Neovim's ``cmd`` or
    VS Code's ``latform.server.args``).
    """
    # Ignore any other flags a client injects (e.g. --clientProcessId=...).
    args, unknown = build_arg_parser().parse_known_args(argv)

    level = configure_logging(args.log_level, args.log_file)
    if unknown:
        logger.debug("Ignoring unrecognized arguments: %s", " ".join(unknown))
    logger.info(
        "Starting latform-lsp (log-level=%s%s)",
        args.log_level,
        f", log-file={args.log_file}" if args.log_file else "",
    )
    server = create_server(client_log_level=None if args.no_client_log else level)
    server.start_io()


if __name__ == "__main__":
    main()
