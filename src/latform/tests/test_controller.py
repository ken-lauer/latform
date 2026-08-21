from __future__ import annotations

from ..output import format_statements
from ..parser import MemoryFiles, parse
from ..statements import Element, get_controller_variables
from ..token import Role, Token
from ..types import FormatOptions
from ..walk import walk

OVERLAY = (
    "O_XX: overlay = {O_XX.CAV01[phase_deg]:phase_deg + f1}, \n"
    "      var={phase_deg, f1}, \n"
    "      phase_deg=0, f1=1"
)


def _roles(code: str) -> list[tuple[str, Role | None]]:
    return [(str(it.node), it.node.role) for it in walk(parse(code)) if isinstance(it.node, Token)]


def _format(code: str, **options) -> str:
    return format_statements(parse(code), FormatOptions(newline_at_eof=False, **options))


def _rename(code: str, renames: dict[str, str], **kwargs) -> str:
    files = MemoryFiles.from_contents(code, "test.bmad")
    files.parse()
    files.annotate()
    files.rename(renames, **kwargs)
    (statements,) = files.by_filename.values()
    return format_statements(statements, FormatOptions(newline_at_eof=False))


def test_get_controller_variables():
    (element,) = parse(OVERLAY)
    assert isinstance(element, Element)
    assert get_controller_variables(element) == {"PHASE_DEG", "F1"}


def test_get_controller_variables_empty_for_plain_element():
    (element,) = parse("q1: quad, l = 0.5")
    assert isinstance(element, Element)
    assert get_controller_variables(element) == set()


def test_annotation_roles():
    # `[phase_deg]` selects an attribute of the controlled element (attribute_name);
    # the var={} members, the control-expression usages, and the default-value
    # attributes are all controller_variable.
    assert _roles(OVERLAY) == [
        ("O_XX", Role.name_),
        ("overlay", Role.kind),
        ("O_XX.CAV01", Role.name_),
        ("phase_deg", Role.attribute_name),
        (":", None),
        ("phase_deg", Role.controller_variable),
        ("+", None),
        ("f1", Role.controller_variable),
        ("var", Role.attribute_name),
        ("phase_deg", Role.controller_variable),
        ("f1", Role.controller_variable),
        ("phase_deg", Role.controller_variable),
        ("0", None),
        ("f1", Role.controller_variable),
        ("1", None),
    ]


def test_rename_controller_variable_literal():
    # renamed in var={}, the expression, and the default; NOT the `[phase_deg]`
    # controlled attribute.
    assert _rename(OVERLAY, {"phase_deg": "PHI"}) == (
        "O_XX: overlay = {O_XX.CAV01[phase_deg]:PHI + f1}, var={PHI, f1}, PHI=0, f1=1"
    )


def test_rename_controller_variable_ignores_regex():
    assert _rename(OVERLAY, {r"f.*": "X"}) == (
        "O_XX: overlay = {O_XX.CAV01[phase_deg]:phase_deg + f1}, var={phase_deg, f1}, phase_deg=0, f1=1"
    )


def test_controller_variable_case_formatting():
    # controlled attribute `[phase_deg]` is not a controller variable, so it stays lower.
    assert _format(OVERLAY, controller_variable_case="upper") == (
        "O_XX: overlay = {O_XX.CAV01[phase_deg]:PHASE_DEG + F1}, var={PHASE_DEG, F1}, PHASE_DEG=0, F1=1"
    )
