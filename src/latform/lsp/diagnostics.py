"""Diagnostics and their code actions (quick fixes + refactors)."""

from __future__ import annotations

import copy
import difflib
import re
from dataclasses import dataclass, field
from typing import Generator

from ..attrs import element_key_to_attrs
from ..lint import Lint, _matching_builtin_names, get_used_names, lint_statements
from ..location import Location
from ..parser import MemoryFiles
from ..statements import Constant, Element, Statement
from ..token import Role, Token
from ..types import Attribute, FormatOptions
from .document import AnalyzedDocument
from .editing import TextEditSpec, _line_range, _rewrite
from .navigation import find_references
from .positions import (
    _locate,
    _statement_file,
    _statement_line_span,
    _statement_tokens,
    definition_name_token,
)
from .resolve import _ELEMENT_TYPES, _apply_case, _element_type_for, _format_options, _seq_text


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


def _file_lints(analyzed: AnalyzedDocument, lint_cache: dict | None = None) -> list[Lint]:
    """
    Lints for the current document (empty if it did not parse).

    With ``lint_cache`` (a workspace-lifetime dict), a document's lints are
    reused across builds when nothing they depend on changed: the document's
    statements (compared by identity — the parse cache reuses the list object
    when contents are unchanged), the project's definition signatures, its
    used-names set, and the config.
    """
    if analyzed.files is None:
        return []
    config = analyzed.config
    used_names = _used_names(analyzed.files)
    def_sigs = getattr(analyzed.files, "_def_sigs", None)
    statements = analyzed.files.by_filename.get(analyzed.path)
    cacheable = lint_cache is not None and def_sigs is not None and statements is not None
    if cacheable:
        entry = lint_cache.get(analyzed.path)
        if (
            entry is not None
            and entry[0] is statements
            and entry[1] is config
            and entry[2] == def_sigs
            and entry[3] == used_names
        ):
            return entry[4]
    lints = lint_statements(
        list(analyzed.statements),
        named=analyzed.files.get_named_items(),
        assume_defined=False,
        ignore=config.ignores_for(analyzed.path) if config is not None else (),
        used_names=used_names,
        min_name_length=config.min_name_length if config is not None else 1,
        builtin_constant_rtol=config.builtin_constant_rtol if config is not None else 1e-4,
    )
    if cacheable:
        lint_cache[analyzed.path] = (statements, config, def_sigs, used_names, lints)
    return lints


def iter_diagnostics(
    analyzed: AnalyzedDocument, lint_cache: dict | None = None
) -> Generator[Diagnostic, None, None]:
    """
    Yield diagnostics for the analyzed document (parse errors and lints).
    """
    if analyzed.files is None:
        loc = _error_location(analyzed)
        yield Diagnostic(
            location=loc, message=str(analyzed.error), code="parse-error", severity="error"
        )
        return

    for lint in _file_lints(analyzed, lint_cache):
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
# Code actions
# --------------------------------------------------------------------------- #


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


def _close_matches(word: str, candidates) -> list[str]:
    """Up to 3 case-insensitive close matches, returned in their display form."""
    by_lower: dict[str, str] = {}
    for cand in candidates:
        by_lower.setdefault(cand.lower(), cand)
    hits = difflib.get_close_matches(word.lower(), list(by_lower), n=3, cutoff=0.6)
    return [by_lower[h] for h in hits]


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
