from __future__ import annotations

from typing import Any

import pytest

from ..dump import get_constants, get_elements_status
from ..parser import MemoryFiles


def _files(src: str) -> MemoryFiles:
    files = MemoryFiles.from_contents(src, "test.bmad")
    files.parse()
    files.annotate()
    return files


def _status(src: str) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in get_elements_status(_files(src), "all")}


def _used(src: str) -> set[str]:
    return {name for name, row in _status(src).items() if row["used"] == "YES"}


SUPERIMPOSE_BASE = """
qf: quad, l = 0.1
unused_q: quad, l = 0.2
top: line = (qf)
use, top
"""


@pytest.mark.parametrize(
    ("attrs", "expect_used"),
    [
        pytest.param("superimpose, ref = qf", True, id="ref-used"),
        pytest.param("superimpose = T, ref = qf", True, id="explicit-true"),
        pytest.param("superimpose = F, ref = qf", False, id="disabled"),
        pytest.param("superimpose, ref = unused_q", False, id="ref-unused"),
        pytest.param("superimpose, ref = nonexistent", False, id="ref-undefined"),
        pytest.param("superimpose, ref = q%", True, id="ref-wildcard"),
        pytest.param("superimpose", True, id="default-ref"),
    ],
)
def test_superimpose_usage(attrs: str, expect_used: bool):
    src = SUPERIMPOSE_BASE + f"s1: sextupole, {attrs}\n"
    assert ("S1" in _used(src)) == expect_used


def test_deferred_superimpose_enable():
    src = SUPERIMPOSE_BASE + "s1: sextupole, ref = qf\ns1[superimpose] = T\n"
    assert "S1" in _used(src)


def test_deferred_superimpose_disable():
    src = SUPERIMPOSE_BASE + "s1: sextupole, superimpose, ref = qf\ns1[superimpose] = F\n"
    assert "S1" not in _used(src)


def test_deferred_ref_overrides_inline():
    src = SUPERIMPOSE_BASE + "s1: sextupole, superimpose, ref = unused_q\ns1[ref] = qf\n"
    assert "S1" in _used(src)


def test_line_expansion_repetition_and_reflection():
    src = """
qf: quad, l = 0.1
qd: qf
b1: sbend, l = 1
sub: line = (b1, qd)
top: line = (2*qf, -sub)
use, top
"""
    assert {"QF", "QD", "B1", "SUB", "TOP"} <= _used(src)


def test_line_expansion_replacement_call():
    src = """
qq: quad, l = 0.1
sub(x): line = (x)
top: line = (sub(qq))
use, top
"""
    assert {"QQ", "SUB", "TOP"} <= _used(src)


def test_inheritance_marks_base_chain():
    src = """
qbase: quad, l = 1
qmid: qbase
qchild: qmid
top: line = (qchild)
use, top
"""
    assert {"QCHILD", "QMID", "QBASE"} <= _used(src)


def test_unused_child_does_not_mark_base():
    src = """
qbase: quad, l = 1
qchild: qbase
qf: quad, l = 1
top: line = (qf)
use, top
"""
    used = _used(src)
    assert "QBASE" not in used
    assert "QCHILD" not in used


@pytest.mark.parametrize(
    ("kind", "spec"),
    [
        pytest.param("overlay", "{qf[k1]: 0.1}, var = {a}", id="overlay"),
        pytest.param("group", "{qf[k1]: 0.1}, var = {a}", id="group"),
        pytest.param("girder", "{qf}", id="girder"),
    ],
)
def test_controller_used_when_slave_used(kind: str, spec: str):
    src = SUPERIMPOSE_BASE + f"c1: {kind} = {spec}\n"
    assert "C1" in _used(src)


def test_controller_unused_when_slaves_unused():
    src = SUPERIMPOSE_BASE + "c1: overlay = {unused_q[k1]: 0.1}, var = {a}\n"
    assert "C1" not in _used(src)


def test_controller_wildcard_slave():
    src = SUPERIMPOSE_BASE + "c1: overlay = {q*[k1]: 0.1}, var = {a}\n"
    assert "C1" in _used(src)


def test_inherited_controller_uses_base_slaves():
    src = SUPERIMPOSE_BASE + "ov1: overlay = {qf[k1]: 0.1}, var = {a}\nov2: ov1\n"
    assert {"OV1", "OV2"} <= _used(src)


def test_use_multiple_branches():
    src = """
qf: quad, l = 1
m1: marker
a: line = (qf)
b: line = (m1)
use, a, b
"""
    assert {"A", "B", "QF", "M1"} <= _used(src)


def test_last_use_wins():
    src = """
qf: quad, l = 1
m1: marker
a: line = (qf)
b: line = (m1)
use, a
use, b
"""
    used = _used(src)
    assert {"B", "M1"} <= used
    assert "A" not in used
    assert "QF" not in used


def test_element_list_members_used():
    src = """
m1: marker
b1: sbend, l = 1
ml: list = (m1, b1)
top: line = (ml)
use, top
"""
    status = _status(src)
    assert status["ML"]["type"] == "LIST"
    assert {"ML", "M1", "B1"} <= _used(src)


def test_unused_element_list_reported():
    src = """
m1: marker
ml: list = (m1)
qf: quad, l = 1
top: line = (qf)
use, top
"""
    assert _status(src)["ML"]["used"] == "NO"


def test_fork_pulls_in_target_line():
    src = """
qf: quad, l = 1
m1: marker
other: line = (m1)
frk: fork, to_line = other
top: line = (qf, frk)
use, top
"""
    assert {"FRK", "OTHER", "M1"} <= _used(src)


def test_no_use_statement_means_nothing_used():
    src = """
qf: quad, l = 1
s1: sextupole, superimpose
top: line = (qf)
"""
    assert _used(src) == set()


def test_synthetic_placeholders_excluded():
    src = """
qf: quad, l = 1
top: line = (qf)
use, top
"""
    names = set(_status(src))
    assert not names & {"PARAMETER", "PARTICLE_START", "PTC_COM", "BEGINNING", "END"}


def test_reasons():
    src = SUPERIMPOSE_BASE + "s1: sextupole, superimpose, ref = qf\n"
    status = _status(src)
    assert status["TOP"]["reason"] == "use statement"
    assert status["QF"]["reason"] == "in line TOP"
    assert status["S1"]["reason"] == "superimposed on qf"
    assert status["UNUSED_Q"]["reason"] == ""


TAO_TWO_LINE_LATTICE = """\
m1: marker
m2: marker
l1: line = (m1)
l2: line = (m2)
use, l1
"""


def _tao_files(entries: list[str], lattice_contents: dict[str, str]) -> MemoryFiles:
    body = "\n".join(
        f"  design_lattice({i})%file = '{entry}'" for i, entry in enumerate(entries, start=1)
    )
    files = MemoryFiles.from_tao_init_contents(
        f"&tao_design_lattice\n{body}\n/\n",
        "/virtual/tao.init",
        lattice_contents=lattice_contents,
    )
    files.parse()
    files.annotate()
    return files


def _tao_used(entries: list[str], lattice_contents: dict[str, str]) -> set[str]:
    files = _tao_files(entries, lattice_contents)
    return {row["name"] for row in get_elements_status(files, "all") if row["used"] == "YES"}


@pytest.mark.parametrize(
    ("entry", "expected_used", "expected_unused"),
    [
        pytest.param("mem.lat.bmad", {"L1", "M1"}, {"L2", "M2"}, id="no-use-line"),
        pytest.param("mem.lat.bmad@l2", {"L2", "M2"}, {"L1", "M1"}, id="use-line-overrides-use"),
        pytest.param(
            "mem.lat.bmad@l1@l2", {"L1", "M1", "L2", "M2"}, set(), id="multiple-use-lines"
        ),
    ],
)
def test_tao_init_use_line_roots(entry: str, expected_used: set[str], expected_unused: set[str]):
    used = _tao_used([entry], {"mem.lat.bmad": TAO_TWO_LINE_LATTICE})
    assert expected_used <= used
    assert not expected_unused & used


def test_tao_init_use_line_is_per_file():
    """A use_line overrides only its own lattice's USE; other lattices keep theirs."""
    first = "m1: marker\nm2: marker\nl1: line = (m1)\nl2: line = (m2)\nuse, l1\n"
    second = "m3: marker\nm4: marker\nl3: line = (m3)\nl4: line = (m4)\nuse, l3\n"
    used = _tao_used(
        ["first.lat.bmad", "second.lat.bmad@l4"],
        {"first.lat.bmad": first, "second.lat.bmad": second},
    )
    assert {"L1", "M1", "L4", "M4"} <= used
    assert not {"L2", "M2", "L3", "M3"} & used


def test_tao_init_use_line_reason():
    files = _tao_files(["mem.lat.bmad@l2"], {"mem.lat.bmad": TAO_TWO_LINE_LATTICE})
    status = {row["name"]: row for row in get_elements_status(files, "all")}
    assert status["L2"]["reason"] == "tao.init use_line"
    assert status["M2"]["reason"] == "in line L2"


def test_get_constants():
    src = """
my_const = 0.5
qf: quad, l = my_const
top: line = (qf)
use, top
"""
    rows = list(get_constants(_files(src)))
    assert [(row["name"], row["expression"]) for row in rows] == [("MY_CONST", "0.5")]
