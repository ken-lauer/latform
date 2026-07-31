from __future__ import annotations

import pathlib
import typing
from typing import Collection, Generator

from nmlform import Namelist

from ..statements import Statement
from ..token import Token
from ..types import Lint, LintCode
from .file import TaoInit
from .schema import (
    PathComponent,
    PathProblem,
    ProblemKind,
    check_value,
    is_known_namelist,
    resolve_path,
    string_length,
)

if typing.TYPE_CHECKING:
    from ..parser import Files


def _split_element_name(name):
    if not name:
        return

    name = name.split("##", 1)[0]

    for part in name.split("\\"):
        try:
            int(part)
        except ValueError:
            pass
        else:
            continue
        yield part


def lint_datums(
    tao_init: TaoInit | None,
    named: dict[Token, Statement],
) -> list[Lint]:
    """
    Lint tao_init datums (&tao_d1_data) for undefined element references.
    """
    if tao_init is None:
        return []

    lints = []
    for d1_data in tao_init.d1_data:
        for datum in d1_data.datums:
            ele_names = {datum.ele_name, datum.ele_ref_name, datum.ele_start_name}
            for name in ele_names:
                if not name:
                    continue
                for ele_name in _split_element_name(name):
                    if ele_name.upper() not in named:
                        lints.append(
                            Lint(
                                code=LintCode.undefined_reference,
                                context=d1_data.namelist,
                                message=f"Reference to undefined element in tao_init d1_data: {name}",
                                relevant_tokens=[],
                            )
                        )
    return lints


def lint_variables(
    tao_init: TaoInit,
    named: dict[Token, Statement],
) -> list[Lint]:
    """
    Lint tao_init variables (&tao_var) for undefined element references.
    """

    lints = []
    for v1_var in tao_init.variables:
        for var in v1_var.variables:
            name = var.ele_name
            if not name:
                continue
            for ele_name in _split_element_name(name):
                if ele_name.upper() not in named:
                    lints.append(
                        Lint(
                            code=LintCode.undefined_reference,
                            context=v1_var.namelist,
                            message=f"Reference to undefined element in tao_init var: {name}",
                            relevant_tokens=[],
                        )
                    )
    return lints


# Path problems that are all reported as "unknown/invalid field" lints.
_PROBLEM_CODES = {
    ProblemKind.unknown_field: LintCode.tao_unknown_field,
    ProblemKind.not_a_struct: LintCode.tao_unknown_field,
    ProblemKind.not_indexable: LintCode.tao_unknown_field,
    ProblemKind.index_out_of_bounds: LintCode.tao_index_out_of_bounds,
}


def _parse_index(index_text: str | None) -> int | None:
    """A component subscript as an int, or ``None`` if absent/non-integer/multi-dim."""
    if not index_text:
        return None
    try:
        return int(index_text)
    except ValueError:
        return None


def _path_components(assignment) -> list[PathComponent]:
    return [
        PathComponent(component.name, _parse_index(component.index_text))
        for component in assignment.path.components
    ]


def _problem_message(problem: PathProblem, key: str) -> str:
    match problem.kind:
        case ProblemKind.unknown_field:
            detail = f"Unknown field '{problem.component}' in '{problem.container}'"
        case ProblemKind.not_a_struct:
            detail = (
                f"Field '{problem.component}' in '{problem.container}' is not a "
                "structure but is used with '%'"
            )
        case ProblemKind.not_indexable:
            detail = (
                f"Field '{problem.component}' in '{problem.container}' is not an "
                "array but is indexed"
            )
        case ProblemKind.index_out_of_bounds:
            lo, hi = problem.bounds or (None, None)
            lo_text = "*" if lo is None else lo
            hi_text = "*" if hi is None else hi
            detail = (
                f"Index {problem.index} of '{problem.component}' is outside declared "
                f"bounds [{lo_text}, {hi_text}]"
            )
        case _:
            detail = "Invalid field path"
    return f"{detail} (from '{key}')"


def _invalid_values(base: str, assignment) -> Generator[str, None, None]:
    """Yield the text of each value literal that is invalid for ``base``."""
    for token in assignment.field_tokens:
        if not check_value(base, token.text):
            yield token.text


def _lint_tao_assignment(namelist: Namelist, assignment) -> list[Lint]:
    key = assignment.key
    result = resolve_path(namelist.name, _path_components(assignment))
    lints = [
        Lint(
            code=_PROBLEM_CODES[problem.kind],
            context=namelist,
            message=_problem_message(problem, key),
            relevant_tokens=[],
        )
        for problem in result.problems
    ]
    leaf = result.leaf
    if leaf is not None and leaf.kind == "intrinsic":
        base = leaf.base
        for text in _invalid_values(base, assignment):
            lints.append(
                Lint(
                    code=LintCode.tao_type_mismatch,
                    context=namelist,
                    message=f"Value '{text}' for '{key}' is not valid for type {base}",
                    relevant_tokens=[],
                )
            )
        if base == "character" and leaf.length is not None:
            for token in assignment.field_tokens:
                length = string_length(token.text)
                if length is not None and length > leaf.length:
                    lints.append(
                        Lint(
                            code=LintCode.tao_string_too_long,
                            context=namelist,
                            message=(
                                f"Value {token.text} for '{key}' is {length} characters, "
                                f"exceeding the declared length {leaf.length} "
                                "(Fortran will truncate it)"
                            ),
                            relevant_tokens=[],
                        )
                    )
    return lints


def lint_tao_schema(tao_init: TaoInit) -> list[Lint]:
    """
    Lint Tao ``*.init`` namelist assignments against the type schema.
    """
    lints: list[Lint] = []
    for source in (tao_init, *tao_init.sources.values()):
        for name, group in source.namelists_by_name.items():
            if not is_known_namelist(name):
                continue
            for namelist in group:
                for assignment in namelist.assignments:
                    lints.extend(_lint_tao_assignment(namelist, assignment))
    return lints


def lint_tao_init(
    tao_init: TaoInit,
    named: dict[Token, Statement],
) -> Generator[Lint, None, None]:
    yield from lint_datums(tao_init, named)
    yield from lint_variables(tao_init, named)
    yield from lint_tao_schema(tao_init)


def lint_tao_init_files(
    files_obj: Files,
    named: dict[Token, Statement],
    *,
    ignore: Collection[str] = (),
) -> Generator[tuple[pathlib.Path, Lint], None, None]:
    """
    Yield ``(filename, lint)`` for every Tao ``*.init`` lint in a parsed file set.

    Covers datum/variable element references and schema type validation across
    ``tao.init`` and its split-out source files. ``ignore`` lists lint codes
    (e.g. ``"LF012"``) to suppress. No-op when the file set has no ``tao.init``.
    """
    if not files_obj.tao_init:
        return
    init_path = files_obj.tao_init.filename or pathlib.Path("<tao.init>")
    ignored = {code.upper() for code in ignore}
    tao_lints = (
        *lint_datums(files_obj.tao_init, named),
        *lint_variables(files_obj.tao_init, named),
        *lint_tao_schema(files_obj.tao_init),
    )
    for lint in tao_lints:
        if lint.code.value in ignored:
            continue
        # Attribute to the namelist's own source (e.g. a split-out
        # data_file/var_file), falling back to the tao.init path.
        yield (getattr(lint.context, "filename", None) or init_path, lint)
