"""pygls glue: wire the pure feature providers into a `LanguageServer`."""

from __future__ import annotations

import asyncio
import logging
import pathlib

from ..location import Location
from .assist import complete, hover_text
from .diagnostics import Diagnostic, code_actions, iter_diagnostics
from .editing import format_document, format_range, prepare_rename
from .navigation import (
    document_highlights,
    file_dependencies,
    find_references,
    resolve_definition,
)
from .symbols import (
    SEMANTIC_TOKEN_MODIFIERS,
    SEMANTIC_TOKEN_TYPES,
    document_symbols,
    semantic_tokens,
    workspace_symbols,
)
from .workspace import Workspace

logger = logging.getLogger(__name__)


def create_server(
    name: str = "latform-lsp",
    version: str = "0.1.0",
    *,
    client_log_level: int | None = None,
    publish_delay: float = 0.15,
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
    publish_delay : float, optional
        Debounce interval (seconds) for republishing diagnostics after document
        changes.  Keystroke bursts within this window collapse into a single
        rebuild; ``0`` publishes synchronously on every change.

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
            for diag in iter_diagnostics(analyzed, lint_cache=workspace.lint_cache)
        ]
        logger.debug("Publishing %d diagnostic(s) for %s", len(diagnostics), uri)
        server.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
        )

    def _publish_all_now() -> None:
        # An edit in one project file can change diagnostics in its siblings.
        for uri in list(open_uris.values()):
            _publish(uri)

    pending_publish: dict = {"task": None}

    def _publish_all() -> None:
        """
        Republish diagnostics for every open document, debounced.

        Each call resets the timer, so a burst of changes (typing) collapses
        into one rebuild ``publish_delay`` seconds after the last change.
        Runs synchronously when the delay is 0 or no event loop is running
        (direct calls outside the server, e.g. in tests).
        """
        task = pending_publish["task"]
        if task is not None and not task.done():
            task.cancel()
        if publish_delay <= 0:
            _publish_all_now()
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _publish_all_now()
            return

        async def _delayed() -> None:
            await asyncio.sleep(publish_delay)
            try:
                _publish_all_now()
            except Exception:
                logger.exception("Debounced diagnostics publish failed")

        pending_publish["task"] = loop.create_task(_delayed())

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
