"""Hover and completion — position → describe / suggest."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..attrs import element_key_to_attrs
from ..const import named_physical_constants
from ..funcs import BUILTIN_CONSTANTS, INTRINSIC_FUNCTIONS
from ..parser import _expand_element_type
from ..statements import BUILTIN_TARGETS, Constant, Element, ElementList, Line, Parameter, Statement
from ..token import Role, Token
from .document import AnalyzedDocument
from .positions import _locate, definition_name_token
from .resolve import (
    _ELEMENT_TYPES,
    _apply_case,
    _bracket_owner,
    _element_type_of,
    _format_options,
    _seq_text,
    _word_at,
)

# --------------------------------------------------------------------------- #
# Hover
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Completion
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
