from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Collection, Sequence

from .attrs import element_key_to_attrs
from .statements import Element, Simple, Statement, get_controller_variables
from .token import Token
from .types import Attribute


class LintCode(str, enum.Enum):
    """Stable identifiers for each lint, usable to opt out via the CLI."""

    unknown_statement = "LF001"
    undefined_reference = "LF002"
    unknown_element_type = "LF003"
    unknown_attribute = "LF004"
    controller_default_missing = "LF005"
    duplicate_attribute = "LF006"


@dataclass()
class Lint:
    code: LintCode
    statement: Statement
    message: str
    relevant_tokens: list[Token] | None

    def to_user_message(self):
        clsname = type(self.statement).__name__
        obj_name = str(getattr(self.statement, "name", "unnamed"))
        parts = [f"[{self.code.value}] {obj_name!r} Statement of type {clsname!r}: {self.message}"]

        if self.relevant_tokens:
            parts.append("\n    Found near:")
            for tok in self.relevant_tokens:
                if tok.loc:
                    parts.append(f"{tok.quoted()} at {tok.loc}")
                else:
                    parts.append(f"{tok.quoted()}")
        return " ".join(parts)


def lint_statements(
    statements: list[Statement],
    named: dict[Token, Statement],
    *,
    assume_defined: bool = True,
    ignore: Collection[str] = (),
) -> list[Lint]:
    ignored = {code.upper() for code in ignore}
    lints = [lint for st in statements for lint in lint_statement(st)]
    if not assume_defined:
        lints.extend(lint_undefined_references(statements, named))
        lints.extend(lint_unknown_element_types(statements))
    return [lint for lint in lints if lint.code.value not in ignored]


def lint_statement(st: Statement) -> list[Lint]:
    if isinstance(st, Simple):
        if not Simple.is_known_statement(st.statement):
            return [
                Lint(
                    code=LintCode.unknown_statement,
                    statement=st,
                    message=f"Statement type is unknown; this may indicate an error in parsing: {st.statement}",
                    relevant_tokens=[st.statement],
                )
            ]
    if isinstance(st, Element):
        return lint_duplicate_attributes(st) + lint_element_attributes(st)
    return []


def lint_duplicate_attributes(element: Element) -> list[Lint]:
    """
    Flag attributes assigned more than once within a single element.

    Overriding an inherited value (re-setting an attribute in a child element
    that its base element also sets) is fine; only repeats within the same
    statement are flagged.
    """
    first_seen: dict[Token, Token] = {}
    lints = []
    for attr in element.attributes:
        # Indexed/struct names (``tt(1)``, ``curve(1)%r0``) are CallName/Seq;
        # only plain attribute names are checked for duplicates.
        if not isinstance(attr, Attribute) or not isinstance(attr.name, Token):
            continue
        name = attr.name
        key = name.lower()
        if key in first_seen:
            lints.append(
                Lint(
                    code=LintCode.duplicate_attribute,
                    statement=element,
                    message=f"Attribute '{name}' is set more than once",
                    relevant_tokens=[first_seen[key], name],
                )
            )
        else:
            first_seen[key] = name
    return lints


def lint_element_attributes(element: Element) -> list[Lint]:
    """
    Flag attributes that are not defined for an element's type.
    """
    if element.element_type is None:
        return []

    valid = element_key_to_attrs[element.element_type]
    controller_vars: set[Token] = {var.lower() for var in get_controller_variables(element)}
    controller_defaults_set: set[Token] = set()

    lints = []
    for attr in element.attributes:
        # Indexed/struct names (``tt(1)``, ``curve(1)%r0``) are CallName/Seq and
        # are not represented in the flat attribute schema; only check plain names.
        if not isinstance(attr, Attribute) or not isinstance(attr.name, Token):
            continue
        name = attr.name
        if name.lower() in controller_vars:
            # Default definition
            controller_defaults_set.add(name.lower())
            continue
        if name.upper() not in valid:
            lints.append(
                Lint(
                    code=LintCode.unknown_attribute,
                    statement=element,
                    message=(
                        f"Unknown attribute '{name} for element type "
                        f"'{element.element_type.lower()}'"
                    ),
                    relevant_tokens=[name],
                )
            )

    missing_defaults = controller_vars - controller_defaults_set
    for missing in missing_defaults:
        lints.append(
            Lint(
                code=LintCode.controller_default_missing,
                statement=element,
                message=(f"Controller variable '{missing}' does not have a default set"),
                relevant_tokens=[missing],
            )
        )

    return lints


def lint_undefined_references(
    statements: Sequence[Statement],
    named: dict[Token, Statement],
) -> list[Lint]:
    """
    Flag ``NAME[attr]`` references whose ``NAME`` is not defined in any loaded file.
    """

    from .parser import _iter_element_references

    lints = []
    for statement, name in _iter_element_references(statements):
        if name.upper() not in named:
            lints.append(
                Lint(
                    code=LintCode.undefined_reference,
                    statement=statement,
                    message=f"Reference to undefined element or constant: {name}",
                    relevant_tokens=[name],
                )
            )
    return lints


def lint_unknown_element_types(statements: Sequence[Statement]) -> list[Lint]:
    """
    Flag elements whose type keyword is neither a known Bmad type (or a valid
    abbreviation of one) nor an element defined in a loaded file.
    """
    lints = []
    for statement in statements:
        if (
            isinstance(statement, Element)
            and statement.element_type is None
            and statement.base_element is None
        ):
            lints.append(
                Lint(
                    code=LintCode.unknown_element_type,
                    statement=statement,
                    message=f"Unknown element type or undefined base element: {statement.keyword}",
                    relevant_tokens=[statement.keyword],
                )
            )
    return lints
