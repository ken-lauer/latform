"""
Language Server Protocol implementation for Bmad lattice files.

This package provides a `pygls`-based language server exposing go-to-definition,
references, hover, completion, rename, formatting, document/workspace symbols,
semantic tokens, document highlight, diagnostics, and code actions.  ``pygls``
is an optional dependency; install it with ``pip install latform[lsp]``.

The analysis layer (`analyze`, `Workspace`, and the feature providers) is pure
latform and can be exercised without ``pygls`` installed.  The pygls glue lives
in `server.create_server` and the CLI entry point in `cli.main`.
"""

from __future__ import annotations

from .assist import Completion, complete, hover_text
from .cli import build_arg_parser, main
from .diagnostics import (
    CodeAction,
    Diagnostic,
    code_actions,
    iter_diagnostics,
)
from .diagnostics import _lint_locations as _lint_locations  # re-exported for tests
from .document import AnalyzedDocument, analyze
from .editing import TextEditSpec, format_document, format_range, prepare_rename
from .logconfig import LOG_LEVELS, configure_logging
from .navigation import (
    document_highlights,
    file_dependencies,
    find_references,
    resolve_definition,
)
from .positions import (
    definition_name_token,
    loc_contains,
    token_at_position,
)
from .server import (
    _attach_client_log_handler as _attach_client_log_handler,
)  # re-exported for tests
from .server import create_server
from .symbols import (
    SEMANTIC_TOKEN_MODIFIERS,
    SEMANTIC_TOKEN_TYPES,
    document_symbols,
    semantic_tokens,
    workspace_symbols,
)
from .workspace import Workspace

__all__ = [
    # analysis core
    "analyze",
    "AnalyzedDocument",
    "Workspace",
    # positions
    "loc_contains",
    "token_at_position",
    "definition_name_token",
    # navigation
    "resolve_definition",
    "find_references",
    "document_highlights",
    "file_dependencies",
    # assist
    "hover_text",
    "complete",
    "Completion",
    # symbols
    "document_symbols",
    "workspace_symbols",
    "semantic_tokens",
    "SEMANTIC_TOKEN_TYPES",
    "SEMANTIC_TOKEN_MODIFIERS",
    # diagnostics + code actions
    "iter_diagnostics",
    "Diagnostic",
    "code_actions",
    "CodeAction",
    # editing
    "format_document",
    "format_range",
    "prepare_rename",
    "TextEditSpec",
    # logging + cli + server
    "configure_logging",
    "LOG_LEVELS",
    "build_arg_parser",
    "main",
    "create_server",
]
