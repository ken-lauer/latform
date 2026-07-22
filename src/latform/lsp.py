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
import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Generator, Sequence

from .config import LatformProjectConfig, discover_config
from .lint import Lint, get_used_names, lint_statements
from .location import Location
from .parser import MemoryFiles, _resolve_lattice_paths, implicit_location
from .statements import (
    Constant,
    Element,
    ElementList,
    Line,
    Statement,
)
from .tao import TaoInit, is_init_file, looks_like_namelist
from .token import Role, Token
from .types import CallName, Seq
from .walk import walk

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


class _OverlayFiles(MemoryFiles):
    """
    `MemoryFiles` whose overlay lookup tolerates non-canonical include paths.

    ``call`` targets may resolve to paths containing ``..`` or symlinks that do
    not match the canonicalized keys used for open editor buffers; falling back
    to a resolved lookup ensures unsaved edits in included files are still used.
    """

    def _get_file_contents(self, filepath: pathlib.Path) -> str:
        for candidate in (filepath, filepath.resolve()):
            if candidate in self.initial_contents:
                return self.initial_contents[candidate]
        return filepath.read_text()


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
    top_files: list[pathlib.Path], contents: dict[pathlib.Path, str]
) -> tuple[MemoryFiles | None, Exception | None]:
    """Parse and annotate a file set, returning ``(files, error)``."""
    files = _OverlayFiles(top_files=top_files, initial_contents=dict(contents))
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
    config: LatformProjectConfig, contents: dict[pathlib.Path, str]
) -> tuple[MemoryFiles | None, Exception | None]:
    """Parse a project's lattice tree, expanding any ``tao.init`` entries."""
    top_files, tao_inits = _expand_top_files(config, contents)
    files, error = _parse_files(top_files, contents)
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
        files, error = _build_project(config, contents)
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

    files, error = _parse_files([resolved], contents)
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


def token_at_position(statements: Sequence[Statement], line: int, char: int) -> Token | None:
    """
    The innermost `Token` covering a 0-indexed ``(line, char)`` position.

    Ties (a position covered by nested tokens) resolve to the smallest span.
    """
    best: Token | None = None
    best_width: int | None = None
    for statement in statements:
        for tok in _statement_tokens(statement):
            loc = tok.loc
            if loc is None or not loc_contains(loc, line, char):
                continue
            width = (loc.end_line - loc.line, loc.end_column - loc.column)
            flat = width[0] * 1_000_000 + width[1]
            if best_width is None or flat < best_width:
                best, best_width = tok, flat
    return best


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


def hover_text(analyzed: AnalyzedDocument, line: int, char: int) -> str | None:
    """
    Markdown hover text for the symbol under a 0-indexed position, or ``None``.
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

    if isinstance(statement, Element):
        kind = statement.element_type or str(statement.keyword)
        return f"**{statement.name}** — element (`{kind}`)"
    if isinstance(statement, Constant):
        return f"**{statement.name}** — constant = `{_seq_text(statement.value)}`"
    if isinstance(statement, (Line, ElementList)):
        return f"**{definition_name_token(statement)}** — {type(statement).__name__.lower()}"
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

    statements = analyzed.statements
    named = analyzed.files.get_named_items()
    config = analyzed.config
    all_statements = [st for sts in analyzed.files.by_filename.values() for st in sts]
    lints = lint_statements(
        list(statements),
        named=named,
        assume_defined=False,
        ignore=config.ignores_for(analyzed.path) if config is not None else (),
        used_names=get_used_names(all_statements),
        min_name_length=config.min_name_length if config is not None else 1,
        builtin_constant_rtol=config.builtin_constant_rtol if config is not None else 1e-4,
    )
    for lint in lints:
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


def document_symbols(analyzed: AnalyzedDocument) -> list[tuple[str, str, Location]]:
    """
    Named symbols defined in the document as ``(name, kind, location)`` tuples.

    ``kind`` is one of ``"element"``, ``"line"``, ``"list"``, ``"constant"``.
    """
    out: list[tuple[str, str, Location]] = []
    for statement in analyzed.statements:
        name_token = definition_name_token(statement)
        if name_token is None or name_token.loc is None:
            continue
        if isinstance(statement, Element):
            kind = "element"
        elif isinstance(statement, Line):
            kind = "line"
        elif isinstance(statement, ElementList):
            kind = "list"
        elif isinstance(statement, Constant):
            kind = "constant"
        else:
            continue
        out.append((str(name_token), kind, name_token.loc))
    return out


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

    def config_for(self, path: pathlib.Path | str) -> LatformProjectConfig:
        """Discover (and cache) the project config applicable to ``path``."""
        directory = pathlib.Path(path).resolve().parent
        if directory not in self._config_by_dir:
            self._config_by_dir[directory] = discover_config(
                start=directory, enabled=self.config_enabled
            )
        return self._config_by_dir[directory]

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
        files, error = _build_project(config, self.open_texts)
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
        return analyze(resolved, text, overlay, config=config)


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
        from pygls.server import LanguageServer
        from pygls.uris import to_fs_path
    except ImportError as exc:  # pragma: no cover - exercised only without pygls
        raise ImportError(
            "The latform language server requires 'pygls'. "
            "Install it with: pip install 'latform[lsp]'"
        ) from exc

    server = LanguageServer(name, version)
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

    def _range(loc: Location) -> "lsp.Range":
        """Convert a latform `Location` (inclusive end_column) to an LSP range."""
        return lsp.Range(
            start=lsp.Position(line=loc.line, character=loc.column),
            end=lsp.Position(line=loc.end_line, character=loc.end_column + 1),
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
        server.publish_diagnostics(uri, diagnostics)

    def _publish_all() -> None:
        # An edit in one project file can change diagnostics in its siblings.
        for path in list(workspace.open_texts):
            _publish(path.as_uri())

    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    def did_open(params: lsp.DidOpenTextDocumentParams) -> None:
        logger.debug("didOpen %s", params.text_document.uri)
        workspace.set_text(_uri_to_path(params.text_document.uri), params.text_document.text)
        _publish_all()

    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    def did_change(params: lsp.DidChangeTextDocumentParams) -> None:
        # Full-sync: the last content change holds the whole document.
        logger.debug("didChange %s", params.text_document.uri)
        workspace.set_text(_uri_to_path(params.text_document.uri), params.content_changes[-1].text)
        _publish_all()

    @server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
    def did_close(params: lsp.DidCloseTextDocumentParams) -> None:
        logger.debug("didClose %s", params.text_document.uri)
        workspace.close(_uri_to_path(params.text_document.uri))
        server.publish_diagnostics(params.text_document.uri, [])
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

    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    def hover(params: lsp.HoverParams):
        pos = params.position
        logger.debug("hover %s @ %d:%d", params.text_document.uri, pos.line, pos.character)
        analyzed = workspace.analyze(_uri_to_path(params.text_document.uri))
        text = hover_text(analyzed, pos.line, pos.character)
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

    _WATCH_GLOBS = ("**/*.bmad", "**/*.lat", "**/latform.toml", "**/pyproject.toml")

    @server.feature(lsp.INITIALIZED)
    def initialized(params: lsp.InitializedParams) -> None:
        logger.info("latform-lsp %s initialized", version)
        # Ask the client to notify us when lattice or config files change on
        # disk, so cached parses of unopened files stay fresh.
        try:
            server.register_capability(
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
                server.show_message_log(self.format(record), kind)
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
