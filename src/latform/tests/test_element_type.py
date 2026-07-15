from __future__ import annotations

from ..parser import MemoryFiles, parse
from ..statements import Element


def _elements(code: str) -> dict[str, Element]:
    return {str(s.name): s for s in parse(code) if isinstance(s, Element)}


def test_direct_type():
    (qa,) = _elements("qa: quad, l = 0.6").values()
    assert qa.element_type == "QUADRUPOLE"
    assert qa.base_element is None


def test_abbreviated_type():
    (q3,) = _elements("q3: qua").values()
    assert q3.element_type == "QUADRUPOLE"
    assert q3.base_element is None


def test_ambiguous_abbreviation_is_unresolved():
    # `s` prefixes sbend, sextupole, solenoid, ... -> ambiguous.
    (s1,) = _elements("s1: s").values()
    assert s1.element_type is None


def test_inheritance_links_base_and_type():
    eles = _elements("qa: quad, l = 0.6, tilt = pi/4\nqb: qa")
    assert eles["qb"].base_element is eles["qa"]
    assert eles["qb"].element_type == "QUADRUPOLE"


def test_inheritance_is_transitive():
    eles = _elements("qa: quad\nqb: qa\nqc: qb")
    assert eles["qc"].element_type == "QUADRUPOLE"
    assert eles["qc"].base_element is eles["qb"]


def test_name_shadows_type_in_definition_order():
    # `quad` is redefined as a name partway through; resolution is order-sensitive.
    eles = _elements("q1: quad\nquad: sextupole\nq2: quad")
    assert eles["q1"].element_type == "QUADRUPOLE"  # before the redefinition
    assert eles["quad"].element_type == "SEXTUPOLE"
    assert eles["q2"].element_type == "SEXTUPOLE"  # inherits the redefined name
    assert eles["q2"].base_element is eles["quad"]


def test_unresolvable_base_is_none():
    # The base element is not defined in this (single) file.
    (child,) = _elements("child: from_another_file").values()
    assert child.base_element is None
    assert child.element_type is None


def test_inherited_controller_is_recognized():
    eles = _elements("base_ov: overlay = {z[k1]:v}, var={v}, v=0\nchild: base_ov")
    assert eles["child"].element_type == "OVERLAY"
    assert eles["child"].is_controller


def test_element_type_resolves_across_files():
    files = MemoryFiles.from_mapping(
        {
            "base.bmad": "qa: quad",
            "use.bmad": "qb: qa",
        }
    )
    files.parse()
    files.annotate()
    elements = {
        str(s.name): s
        for statements in files.by_filename.values()
        for s in statements
        if isinstance(s, Element)
    }
    assert elements["qb"].element_type == "QUADRUPOLE"
    assert elements["qb"].base_element is elements["qa"]
