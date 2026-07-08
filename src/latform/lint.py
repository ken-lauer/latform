from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .statements import Simple, Statement
from .token import Token


@dataclass()
class Lint:
    statement: Statement
    message: str
    relevant_tokens: list[Token] | None

    def to_user_message(self):
        clsname = type(self.statement).__name__
        obj_name = str(getattr(self.statement, "name", "unnamed"))
        parts = [f"{obj_name!r} Statement of type {clsname!r}: {self.message}"]

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
) -> list[Lint]:
    lints = [lint for st in statements for lint in lint_statement(st)]
    if not assume_defined:
        lints.extend(lint_undefined_references(statements, named))
    return lints


def lint_statement(st: Statement) -> list[Lint]:
    if isinstance(st, Simple):
        if not Simple.is_known_statement(st.statement):
            return [
                Lint(
                    statement=st,
                    message=f"Statement type is unknown; this may indicate an error in parsing: {st.statement}",
                    relevant_tokens=[st.statement],
                )
            ]
    return []


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
                    statement=statement,
                    message=f"Reference to undefined element or constant: {name}",
                    relevant_tokens=[name],
                )
            )
    return lints
