"""
Type validation for Tao ``*.init`` namelist assignments.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

from ..token import Token
from ._schema import FILE_KEY, MISSING_STRUCTS, NAMELISTS, STRUCTS, Field

__all__ = [
    "Field",
    "FILE_KEY",
    "PathComponent",
    "ProblemKind",
    "PathProblem",
    "ResolvedPath",
    "is_known_namelist",
    "resolve_path",
    "check_value",
    "string_length",
    "logical_value",
]

_MISSING = frozenset(MISSING_STRUCTS)


class PathComponent(NamedTuple):
    """One ``name`` (optionally ``name(idx)``) segment of a namelist key."""

    name: str
    idx: int | None = None


class ProblemKind(enum.Enum):
    """Why a key path failed to resolve to a valid intrinsic leaf."""

    unknown_field = "unknown_field"
    not_a_struct = "not_a_struct"
    not_indexable = "not_indexable"
    index_out_of_bounds = "index_out_of_bounds"


@dataclass(frozen=True)
class PathProblem:
    """A single problem found while resolving a key path."""

    kind: ProblemKind
    component: str
    container: str
    bounds: tuple[int | None, int | None] | None = None
    index: int | None = None


@dataclass
class ResolvedPath:
    """
    The outcome of resolving a key path against the schema.

    Attributes
    ----------
    leaf : Field or None
        The resolved intrinsic leaf field, or ``None`` if resolution stopped at
        a problem, ended on a derived (non-leaf) field, or descended into a
        struct absent from the schema.
    problems : list of PathProblem
        Field-name and index problems, in the order encountered.
    unresolved : bool
        True when the path descended into a struct missing from the schema
        (`MISSING_STRUCTS`); deeper field names and the leaf type are then
        unknowable and left unchecked.
    """

    leaf: Field | None = None
    problems: list[PathProblem] = field(default_factory=list)
    unresolved: bool = False


def is_known_namelist(name: str) -> bool:
    """Whether ``name`` is a Tao namelist group described by the schema."""
    return name.lower() in NAMELISTS


def _check_index(
    spec: Field,
    component: PathComponent,
    container: str,
) -> PathProblem | None:
    """Validate a component's subscript against its field's declared shape."""
    if component.idx is None:
        return None
    if spec.array is None:
        return PathProblem(ProblemKind.not_indexable, component.name, container)
    lbound, ubound = spec.array
    below = lbound is not None and component.idx < lbound
    above = ubound is not None and component.idx > ubound
    if below or above:
        return PathProblem(
            ProblemKind.index_out_of_bounds,
            component.name,
            container,
            bounds=spec.array,
            index=component.idx,
        )
    return None


def resolve_path(namelist: str, components: Sequence[PathComponent]) -> ResolvedPath:
    """
    Resolve a namelist key path to its intrinsic leaf field.

    Parameters
    ----------
    namelist : str
        The namelist group name (must be `is_known_namelist`).
    components : sequence of PathComponent
        The ``name``/``name(index)`` segments of the key, outermost first.

    Returns
    -------
    ResolvedPath
    """
    result = ResolvedPath()
    fields: dict[str, Field] | None = NAMELISTS.get(namelist.lower())
    if fields is None:
        return result
    container = namelist.lower()

    for depth, component in enumerate(components):
        if fields is None:
            # Inside a struct absent from the schema: stop validating names.
            result.unresolved = True
            return result

        spec = fields.get(component.name.lower())
        if spec is None:
            result.problems.append(
                PathProblem(ProblemKind.unknown_field, component.name, container)
            )
            return result

        index_problem = _check_index(spec, component, container)
        if index_problem is not None:
            result.problems.append(index_problem)

        is_last = depth == len(components) - 1
        if is_last:
            result.leaf = spec
            return result

        # A non-final segment must be a struct we can descend into.
        if spec.kind != "derived":
            result.problems.append(PathProblem(ProblemKind.not_a_struct, component.name, container))
            return result
        if spec.base in _MISSING:
            result.unresolved = True
            return result
        fields = STRUCTS.get(spec.base)
        container = spec.base

    return result


# -- value literal type checks -------------------------------------------------


def _strip_repeat(literal: str) -> str | None:
    """
    Reduce a Fortran ``n*value`` repeat to its ``value``.

    Returns the value part when ``literal`` is a repeat with an integer count,
    the literal unchanged when there is no repeat, or ``None`` for a bare
    ``n*`` null-value (nothing to type-check).
    """
    count, star, value = literal.partition("*")
    if not star:
        return literal
    if not count.strip().lstrip("+-").isdigit():
        # Not a repeat count (e.g. a stray '*'); check the whole thing.
        return literal
    value = value.strip()
    return value or None


def _is_integer(text: str) -> bool:
    return text.lstrip("+-").isdigit()


def _is_real(text: str) -> bool:
    # Fortran allows d/D and q/Q exponent markers that Python's float rejects.
    normalized = text
    for marker in ("d", "D", "q", "Q"):
        normalized = normalized.replace(marker, "e")
    try:
        float(normalized)
    except ValueError:
        return False
    return True


def logical_value(text: str) -> bool | None:
    """
    The truth of a Fortran logical literal, or ``None`` if it is not one.

    Follows gfortran's list-directed/namelist rule: optional blanks, an optional
    single leading ``.``, then ``T``/``t`` (true) or ``F``/``f`` (false); any
    trailing characters are ignored. So ``.true.``, ``T``, ``.T.``, ``TRUE``,
    and even ``Fnord`` parse, while ``1``/``0``/``yes`` and empty do not.
    """
    body = text.strip()
    if body.startswith("."):
        body = body[1:]
    first = body[:1]
    if first in ("T", "t"):
        return True
    if first in ("F", "f"):
        return False
    return None


def _is_logical(text: str) -> bool:
    return logical_value(text) is not None


def _is_complex(text: str) -> bool:
    inner = text.strip()
    if not (inner.startswith("(") and inner.endswith(")")):
        return False
    parts = inner[1:-1].split(",")
    return len(parts) == 2 and all(_is_real(part.strip()) for part in parts)


def _is_character(text: str) -> bool:
    return Token(text).is_quoted_string


def check_value(base: str, literal: str) -> bool:
    """
    Whether a single value literal is valid for an intrinsic ``base`` type.

    ``base`` is one of ``character``/``integer``/``real``/``logical``/
    ``complex``. A ``character`` value must be a quoted string; an integer is
    accepted where a real is expected. A leading ``n*`` repeat count is stripped
    before checking. An unrecognized ``base`` (should not occur for schema
    leaves) and an empty or repeat-only literal are treated as valid, so the
    caller does not warn on things this module cannot judge.
    """

    value = _strip_repeat(literal)
    if not value:
        return True

    match base:
        case "character":
            return _is_character(value)
        case "integer":
            return _is_integer(value)
        case "real":
            return _is_real(value)
        case "logical":
            return _is_logical(value)
        case "complex":
            return _is_complex(value)
        case _:
            return True


def _unquote(text: str) -> str | None:
    """
    The content of a quoted-string literal, or ``None`` if it is not quoted.

    Fortran escapes an embedded quote by doubling it (``'it''s'`` -> ``it's``);
    that doubling is collapsed so the returned length matches what Fortran would
    store.
    """
    text = text.strip()
    if len(text) >= 2 and text[0] in "'\"" and text[-1] == text[0]:
        quote = text[0]
        return text[1:-1].replace(quote * 2, quote)
    return None


def string_length(literal: str) -> int | None:
    """
    The stored length of a character value literal, or ``None`` if unknowable.

    A leading ``n*`` repeat count is stripped first. Returns ``None`` when the
    literal is not a quoted string (its type is flagged separately) or is
    repeat-only, so the caller only length-checks genuine strings.
    """
    value = _strip_repeat(literal)
    if not value:
        return None
    inner = _unquote(value)
    return None if inner is None else len(inner)
