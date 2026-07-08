from __future__ import annotations

from ..lint import lint_statements, lint_undefined_references
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
