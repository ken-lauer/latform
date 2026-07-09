from __future__ import annotations

import pytest

from ..lint import (
    LintCode,
    lint_duplicate_attributes,
    lint_element_attributes,
    lint_statements,
    lint_undefined_references,
    lint_unknown_element_types,
)
from ..parser import MemoryFiles


def _files(src: str) -> MemoryFiles:
    files = MemoryFiles.from_contents(src, "test.bmad")
    files.parse()
    files.annotate()
    return files


def _all_lints(files: MemoryFiles, assume_defined: bool) -> list:
    named = files.get_named_items()
    lints = []
    for statements in files.by_filename.values():
        lints.extend(lint_statements(statements, named=named, assume_defined=assume_defined))
    return lints


def test_undefined_reference_reported():
    files = _files("B0[k1] = BX[k1]*3")
    (statements,) = files.by_filename.values()
    lints = lint_undefined_references(statements, files.get_named_items())
    assert any("BX" in lint.message for lint in lints)


def test_defined_reference_not_reported():
    files = _files("BX: quad\nB0[k1] = BX[k1]*3")
    (statements,) = files.by_filename.values()
    lints = lint_undefined_references(statements, files.get_named_items())
    assert lints == []


def test_lint_statements_suppresses_references_when_assuming_defined():
    files = _files("B0[k1] = BX[k1]*3")
    assert _all_lints(files, assume_defined=True) == []
    assert _all_lints(files, assume_defined=False)


def test_unknown_element_type_reported():
    (statements,) = _files("bad: quadrpole").by_filename.values()
    lints = lint_unknown_element_types(statements)
    assert [lint.message for lint in lints] == [
        "Unknown element type or undefined base element: quadrpole"
    ]


def test_known_and_inherited_types_not_reported():
    src = "qa: quad\nqb: qa\nq3: qua"  # direct, inherited, abbreviated
    (statements,) = _files(src).by_filename.values()
    assert lint_unknown_element_types(statements) == []


def test_unknown_element_type_suppressed_when_assuming_defined():
    files = _files("child: from_another_file")
    assert _all_lints(files, assume_defined=True) == []
    messages = [lint.message for lint in _all_lints(files, assume_defined=False)]
    assert any("from_another_file" in message for message in messages)


def test_lints_carry_stable_codes():
    (statements,) = _files("bad: quadrpole").by_filename.values()
    (lint,) = lint_unknown_element_types(statements)
    assert lint.code is LintCode.unknown_element_type
    assert lint.to_user_message().startswith("[LF003]")


def test_unknown_attribute_reported():
    (statements,) = _files("q1: quadrupole, k1 = 0.5, bogus = 3").by_filename.values()
    (element,) = statements
    lints = lint_element_attributes(element)
    assert [lint.code for lint in lints] == [LintCode.unknown_attribute]
    assert "bogus" in lints[0].message


@pytest.mark.parametrize(
    "src",
    [
        "q1: quadrupole, k1 = 0.5, l = 2",  # exact type, known attrs
        "q1: quad, k1 = 0.5",  # abbreviated type keyword
        "qa: quadrupole\nqb: qa, k1 = 0.5",  # inherited type
    ],
)
def test_known_attributes_not_reported(src):
    statements = list(_files(src).by_filename.values())[0]
    lints = [lint for st in statements for lint in lint_element_attributes(st)]
    assert lints == []


def test_unknown_attribute_on_inherited_type_reported():
    (statements,) = _files("qa: quadrupole\nqb: qa, junk = 1").by_filename.values()
    lints = [lint for st in statements for lint in lint_element_attributes(st)]
    assert [lint.code for lint in lints] == [LintCode.unknown_attribute]
    assert "junk" in lints[0].message


def test_controller_variables_not_reported():
    (statements,) = _files("o1: overlay = {q1[k1]}, var = {x}, x = 0").by_filename.values()
    lints = [lint for st in statements for lint in lint_element_attributes(st)]
    assert lints == []


def test_unknown_type_skips_attribute_lint():
    (statements,) = _files("m1: notarealtype, foo = 1").by_filename.values()
    (element,) = statements
    assert lint_element_attributes(element) == []


def test_controller_default_missing_reported():
    (statements,) = _files("o1: overlay = {q1[k1]}, var = {x, y}, x = 0").by_filename.values()
    (element,) = statements
    lints = lint_element_attributes(element)
    assert [lint.code for lint in lints] == [LintCode.controller_default_missing]
    assert "y" in lints[0].message


def test_controller_defaults_all_set_not_reported():
    (statements,) = _files(
        "o1: overlay = {q1[k1]}, var = {x, y}, x = 0, y = 1"
    ).by_filename.values()
    (element,) = statements
    assert lint_element_attributes(element) == []


def test_duplicate_attribute_reported():
    (statements,) = _files("q1: quadrupole, k1 = 0.5, k1 = 0.6").by_filename.values()
    (element,) = statements
    (lint,) = lint_duplicate_attributes(element)
    assert lint.code is LintCode.duplicate_attribute
    assert "k1" in lint.message
    # Both occurrences are reported, at distinct source locations.
    assert [str(tok) for tok in lint.relevant_tokens] == ["k1", "k1"]
    first, second = lint.relevant_tokens
    assert first.loc != second.loc


def test_duplicate_attribute_case_insensitive():
    (statements,) = _files("q1: quadrupole, K1 = 0.5, k1 = 0.6").by_filename.values()
    (element,) = statements
    (lint,) = lint_duplicate_attributes(element)
    assert lint.code is LintCode.duplicate_attribute
    assert len(lint.relevant_tokens) == 2


def test_inherited_override_not_duplicate():
    src = "qa: quadrupole, k1 = 0.5\nqb: qa, k1 = 0.9"
    (statements,) = _files(src).by_filename.values()
    lints = [lint for st in statements for lint in lint_duplicate_attributes(st)]
    assert lints == []


def test_distinct_attributes_not_reported():
    (statements,) = _files("q1: quadrupole, k1 = 0.5, l = 2").by_filename.values()
    (element,) = statements
    assert lint_duplicate_attributes(element) == []


def test_ignore_suppresses_by_code():
    files = _files("bad: quadrpole\nB0[k1] = BX[k1]*3")
    named = files.get_named_items()
    (statements,) = files.by_filename.values()

    codes = {lint.code for lint in lint_statements(statements, named, assume_defined=False)}
    assert codes == {LintCode.undefined_reference, LintCode.unknown_element_type}

    kept = lint_statements(statements, named, assume_defined=False, ignore=["LF003"])
    assert {lint.code for lint in kept} == {LintCode.undefined_reference}
