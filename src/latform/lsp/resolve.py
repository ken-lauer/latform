"""
Shared interpretation helpers: element-type / bracket / value-text resolution
plus project format-option handling (casing).
"""

from __future__ import annotations

import re
from dataclasses import fields as dataclass_fields

from ..attrs import element_key_to_attrs
from ..config import LatformProjectConfig
from ..parser import _expand_element_type
from ..statements import Element, Parameter, Statement
from ..token import Token
from ..types import FormatOptions, Seq

_ELEMENT_TYPES = frozenset(k for k in element_key_to_attrs if not k.startswith("!"))
_FORMAT_OPTION_FIELDS = frozenset(f.name for f in dataclass_fields(FormatOptions))


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


def _element_type_for(statement: Statement, named: dict) -> str | None:
    """The element type an attribute in ``statement`` belongs to."""
    if isinstance(statement, Element):
        return statement.element_type
    if isinstance(statement, Parameter):
        target = named.get(str(statement.target).upper())
        if isinstance(target, Element):
            return target.element_type
    return None


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


def _seq_text(value: Token | Seq) -> str:
    """Reconstruct a value's source text for display."""
    if isinstance(value, Token):
        return str(value)
    return str(value.to_token())


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
