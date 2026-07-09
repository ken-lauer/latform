from __future__ import annotations

from ..lint import lint_statements, lint_undefined_references, lint_unknown_element_types
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
