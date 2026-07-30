from __future__ import annotations

import logging

import pytest

from ..lint import (
    LintCode,
    cli_main,
    get_used_names,
    lint_ambiguous_names,
    lint_attribute_overrides,
    lint_builtin_constants,
    lint_duplicate_attributes,
    lint_element_attributes,
    lint_files,
    lint_statements,
    lint_undefined_references,
    lint_unknown_element_types,
    lint_unused_constants,
)
from ..parser import MemoryFiles
from ..tao.lint import lint_datums, lint_variables


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


@pytest.mark.parametrize(
    "src",
    [
        "q1: quadrupole, mat6_calc = bmad_standard",  # abbreviation
        "q1: quadrupole, mat6_calc_method = bmad_standard",  # full name
        "q1: quadrupole, tracking = bmad_standard",  # abbreviation of tracking_method
    ],
)
def test_abbreviated_attributes_not_reported(src):
    (statements,) = _files(src).by_filename.values()
    (element,) = statements
    assert lint_element_attributes(element) == []


def test_unknown_attribute_on_inherited_type_reported():
    (statements,) = _files("qa: quadrupole\nqb: qa, junk = 1").by_filename.values()
    lints = [lint for st in statements for lint in lint_element_attributes(st)]
    assert [lint.code for lint in lints] == [LintCode.unknown_attribute]
    assert "junk" in lints[0].message


def test_controller_variables_not_reported():
    (statements,) = _files("o1: overlay = {q1[k1]}, var = {x}, x = 0").by_filename.values()
    lints = [lint for st in statements for lint in lint_element_attributes(st)]
    assert lints == []


@pytest.mark.parametrize(
    "src",
    [
        "q: quad, superimpose, ref = B12, offset = 1.3, ele_origin = beginning, ref_origin = end",
        "q: quad, superimpose = T, ref = B12",
    ],
)
def test_superimpose_enables_superposition_attributes(src):
    (statements,) = _files(src).by_filename.values()
    (element,) = statements
    assert lint_element_attributes(element) == []


def test_ref_alias_accepted():
    # Bmad hardcodes ``ref`` as an alias for ``reference`` (attribute_index2),
    # so it is valid on any element type that has ``reference``.
    (statements,) = _files("q: quad, ref = B12").by_filename.values()
    (element,) = statements
    assert lint_element_attributes(element) == []


def test_superimpose_still_flags_unknown_attributes():
    (statements,) = _files("q: quad, superimpose, bogus = 1").by_filename.values()
    (element,) = statements
    lints = lint_element_attributes(element)
    assert [lint.code for lint in lints] == [LintCode.unknown_attribute]
    assert "bogus" in lints[0].message


@pytest.mark.parametrize(
    "src",
    [
        "q: quadrupole, x_pi = 0",  # ambiguous: x_pitch vs x_pitch_tot
        "q: quadrupole, ab = 0",  # too short (< 3 chars) and not exact
    ],
)
def test_ambiguous_or_short_abbreviation_reported(src):
    (statements,) = _files(src).by_filename.values()
    (element,) = statements
    assert [lint.code for lint in lint_element_attributes(element)] == [LintCode.unknown_attribute]


def test_unknown_type_skips_attribute_lint():
    (statements,) = _files("m1: notarealtype, foo = 1").by_filename.values()
    (element,) = statements
    assert lint_element_attributes(element) == []


def test_controller_all_zero():
    (statements,) = _files(
        "o1: overlay = {q1[k1]}, var = {x, y}, x = 0, y = 0"
    ).by_filename.values()
    (element,) = statements
    lints = lint_element_attributes(element)
    assert [lint.code for lint in lints] == [LintCode.controller_all_zero_defaults]
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


def _override_lints(src: str) -> list:
    files = _files(src)
    (statements,) = files.by_filename.values()
    return lint_attribute_overrides(statements, files.get_named_items())


def test_attribute_override_reported():
    src = "q1: quadrupole, L = 0.4, k1 = 0.\nq1[k1] = 1\nq1[b1_gradient] = 1"
    (lint,) = _override_lints(src)
    assert lint.code is LintCode.attribute_override
    assert "k1" in lint.message
    assert "definition" in lint.message
    # The original setting and the override, at distinct source locations.
    assert [str(tok).lower() for tok in lint.relevant_tokens] == ["k1", "k1"]
    first, second = lint.relevant_tokens
    assert first.loc != second.loc


@pytest.mark.parametrize(
    "src",
    [
        "q1: quadrupole, K1 = 0.\nq1[k1] = 1",  # case-insensitive vs the definition
        "q1: quadrupole\nq1[k1] = 1\nq1[k1] = 2",  # repeated parameter statements
        "parameter[geometry] = open\nparameter[geometry] = closed",  # builtin target twice
    ],
)
def test_attribute_override_variants_reported(src):
    (lint,) = _override_lints(src)
    assert lint.code is LintCode.attribute_override


@pytest.mark.parametrize(
    "src",
    [
        "q1: quadrupole, L = 0.4\nq1[k1] = 1",  # not set in the definition
        "qa: quadrupole, k1 = 0.5\nqb: qa\nqb[k1] = 1",  # only set on the base element
        "parameter[geometry] = open",  # builtin target set once
        "qz[k1] = 1",  # undefined target
    ],
)
def test_attribute_override_not_reported(src):
    assert _override_lints(src) == []


def test_attribute_override_class_selector_reported():
    src = "rf1: rfcavity, voltage = 1e6\nrfcavity::*[voltage] = 3.7"
    (lint,) = _override_lints(src)
    assert lint.code is LintCode.attribute_override
    assert "RFCAVITY::*" in lint.message
    assert [str(tok).lower() for tok in lint.relevant_tokens] == ["voltage", "voltage"]


def test_attribute_override_glob_selector_reports_all_matches():
    src = "q1: quadrupole, k1 = 0\nq2: quadrupole, k1 = 0\nq*[k1] = 1"
    (lint,) = _override_lints(src)
    assert lint.code is LintCode.attribute_override
    # Both matched definitions plus the overriding statement.
    assert [str(tok).lower() for tok in lint.relevant_tokens] == ["k1", "k1", "k1"]


def test_attribute_override_repeated_selector_reported():
    src = "rf1: rfcavity\nrfcavity::*[voltage] = 1\nrfcavity::*[voltage] = 2"
    (lint,) = _override_lints(src)
    assert lint.code is LintCode.attribute_override
    assert "already set" in lint.message


@pytest.mark.parametrize(
    "src",
    [
        "rf1: rfcavity\nrfcavity::*[voltage] = 3.7",  # matched, but not set in the definition
        "q1: quadrupole, k1 = 0.5\nsbend::*[k1] = 1",  # selector matches nothing
        "q1: quadrupole, k1 = 0.5\nq1:q5[k1] = 1",  # TODO: ranges are unsupported
        "lat>>q1[k1] = 1\nlat>>q1[k1] = 2",  # TODO: branch qualifiers are unsupported
        "q1##2[k1] = 1\nq1##2[k1] = 2",  # TODO: instance counts are unsupported
    ],
)
def test_attribute_override_selector_not_reported(src):
    assert _override_lints(src) == []


def test_selector_targets_not_undefined_references():
    src = "rf1: rfcavity\nrfcavity::*[voltage] = 3.7\nq*[k1] = 1\nlat>>q1[k1] = 1"
    files = _files(src)
    (statements,) = files.by_filename.values()
    assert lint_undefined_references(statements, files.get_named_items()) == []


def test_attribute_override_via_lint_statements():
    files = _files("q1: quadrupole, k1 = 0.\nq1[k1] = 1")
    codes = {lint.code for lint in _all_lints(files, assume_defined=True)}
    assert LintCode.attribute_override in codes


def test_unused_constant_reported():
    (statements,) = _files("my_k = 0.5").by_filename.values()
    (lint,) = lint_unused_constants(statements)
    assert lint.code is LintCode.unused_constant
    assert "my_k" in lint.message


@pytest.mark.parametrize(
    "src",
    [
        "my_k = 0.5\nq1: quadrupole, k1 = my_k",  # element attribute value
        "my_k = 0.5\nother = my_k * 2\nq1: quadrupole, k1 = other",  # another constant
        "my_k = 0.5\nq1[k1] = my_k",  # parameter statement value
        "n_cell = 4\nlat: line = (n_cell*q1)",  # line definition
    ],
)
def test_used_constant_not_reported(src):
    (statements,) = _files(src).by_filename.values()
    assert lint_unused_constants(statements) == []


def test_attribute_name_is_not_constant_usage():
    # ``l = 2`` sets the element's length attribute; it does not *use* a
    # constant that happens to be named ``l``.
    (statements,) = _files("l = 2\nq1: quadrupole, l = 2").by_filename.values()
    (lint,) = lint_unused_constants(statements)
    assert lint.code is LintCode.unused_constant


def test_unused_constant_cross_file_usage_not_reported(tmp_path):
    main = tmp_path / "main.bmad"
    sub = tmp_path / "sub.bmad"
    sub.write_text("my_k = 0.5\n")
    main.write_text('call, filename = "sub.bmad"\nq1: quadrupole, k1 = my_k\n')

    from ..parser import Files

    files = Files(top_files=[main])
    files.parse(recurse=True)
    files.annotate()
    codes = {lint.code for _fn, lint in lint_files(files)}
    assert LintCode.unused_constant not in codes


def test_unused_constant_via_lint_files():
    files = _files("dead = 1\nq1: quadrupole, k1 = 0.5")
    codes = {lint.code for _fn, lint in lint_files(files)}
    assert LintCode.unused_constant in codes


def test_get_used_names_uppercase():
    (statements,) = _files("q1: quadrupole, k1 = my_k").by_filename.values()
    assert "MY_K" in get_used_names(statements)


def _ambiguous_lints(src: str, **kwargs) -> list:
    (statements,) = _files(src).by_filename.values()
    return lint_ambiguous_names(statements, **kwargs)


@pytest.mark.parametrize(
    "src",
    [
        "i = 1",  # constant
        "L: quadrupole",  # element, case-insensitive
        "o: line = (q1)",  # line
        "l(a): line = (a, a)",  # line with call-style name
        "i: list = (q1, q2)",  # element list
    ],
)
def test_ambiguous_name_reported(src):
    (lint,) = _ambiguous_lints(src)
    assert lint.code is LintCode.ambiguous_name
    assert "confused" in lint.message


@pytest.mark.parametrize(
    "src",
    [
        "ab = 1",  # two characters: short but not ambiguous
        "x = 1",  # single character but not i/l/o
        "q1: quadrupole",
    ],
)
def test_short_names_not_reported_at_default_minimum(src):
    assert _ambiguous_lints(src) == []


def test_short_name_reported_with_minimum_length():
    (lint,) = _ambiguous_lints("ab = 1", min_name_length=3)
    assert lint.code is LintCode.ambiguous_name
    assert "minimum length (3)" in lint.message


def test_minimum_length_name_not_reported():
    assert _ambiguous_lints("abc = 1", min_name_length=3) == []


def test_ambiguous_name_single_lint_when_also_short():
    (lint,) = _ambiguous_lints("i = 1", min_name_length=3)
    assert lint.code is LintCode.ambiguous_name


def test_min_name_length_from_config_via_lint_files():
    import pathlib

    from ..config import LatformProjectConfig

    files = _files("ab: quadrupole")
    config = LatformProjectConfig(root=pathlib.Path("."), min_name_length=3)
    codes = {lint.code for _fn, lint in lint_files(files, config=config)}
    assert LintCode.ambiguous_name in codes

    codes = {lint.code for _fn, lint in lint_files(files)}
    assert LintCode.ambiguous_name not in codes


def test_ambiguous_name_via_lint_statements():
    files = _files("i: quadrupole")
    codes = {lint.code for lint in _all_lints(files, assume_defined=True)}
    assert LintCode.ambiguous_name in codes


def _builtin_constant_lints(src: str, **kwargs) -> list:
    (statements,) = _files(src).by_filename.values()
    return lint_builtin_constants(statements, **kwargs)


def test_builtin_constant_reported():
    (lint,) = _builtin_constant_lints("my_pi = 3.1415926535897931")
    assert lint.code is LintCode.use_builtin_constant
    assert "'pi'" in lint.message
    assert [str(tok) for tok in lint.relevant_tokens] == ["3.1415926535897931"]


@pytest.mark.parametrize(
    "src, expected",
    [
        ("my_pi = 3.14159", "'pi'"),  # ~8e-7 relative error
        ("my_pi = 3.1416", "'pi'"),  # ~2.3e-6
        ("emass_ev = 0.511e6", "'m_electron'"),  # ~2.1e-6
        ("c = 2.998e8", "'c_light'"),  # ~2.5e-5
    ],
)
def test_builtin_constant_rounded_value_reported(src, expected):
    # Hand-rounded values land within the default 1e-4 relative tolerance.
    (lint,) = _builtin_constant_lints(src)
    assert expected in lint.message


def test_builtin_constant_negative_value_reported():
    (lint,) = _builtin_constant_lints("anom = -1.9130427299999999")
    assert "'anom_moment_neutron'" in lint.message


@pytest.mark.parametrize(
    "src, expected",
    [
        ("neg_pi = -3.1415926535897931", "'-pi'"),
        ("pos_anom = 1.9130427299999999", "'-anom_moment_neutron'"),
    ],
)
def test_negated_builtin_constant_reported(src, expected):
    (lint,) = _builtin_constant_lints(src)
    assert expected in lint.message


def test_builtin_constant_reports_all_matching_names():
    (lint,) = _builtin_constant_lints("my_c = 299792458")
    assert "'c_light'" in lint.message
    assert "'clight'" in lint.message


@pytest.mark.parametrize(
    "src",
    [
        "my_pi = 3.14",  # outside the default relative tolerance
        "my_pi = 2*pi",  # no numeric literal in the expression matches
        'name = "pi"',  # non-numeric value
        "x = 0",
        "n = 1",
        "half = 0.5",
    ],
)
def test_builtin_constant_not_reported(src):
    assert _builtin_constant_lints(src) == []


def test_builtin_constant_literal_inside_expression_reported():
    (lint,) = _builtin_constant_lints("tau = 2 * 3.1415")
    assert lint.code is LintCode.use_builtin_constant
    assert "'pi'" in lint.message
    assert [str(tok) for tok in lint.relevant_tokens] == ["3.1415"]


@pytest.mark.parametrize(
    "src",
    [
        "q: quadrupole, k1 = 3.1415 * 2",  # element attribute expression
        "q: quadrupole, tilt = 3.1416",  # element attribute literal
        "q: quadrupole\nq[tilt] = 3.1416",  # parameter statement value
    ],
)
def test_builtin_constant_in_attribute_value_reported(src):
    (lint,) = _builtin_constant_lints(src)
    assert lint.code is LintCode.use_builtin_constant
    assert "'pi'" in lint.message


def test_builtin_constant_tightened_rtol():
    assert _builtin_constant_lints("my_pi = 3.14159", rtol=1e-12) == []


def test_builtin_constant_rtol_from_config_via_lint_files():
    import pathlib

    from ..config import LatformProjectConfig

    files = _files("my_pi = 3.14159\nq1: quadrupole, k1 = my_pi")
    codes = {lint.code for _fn, lint in lint_files(files)}
    assert LintCode.use_builtin_constant in codes

    config = LatformProjectConfig(root=pathlib.Path("."), builtin_constant_rtol=1e-12)
    codes = {lint.code for _fn, lint in lint_files(files, config=config)}
    assert LintCode.use_builtin_constant not in codes


def test_ignore_suppresses_by_code():
    files = _files("bad: quadrpole\nB0[k1] = BX[k1]*3")
    named = files.get_named_items()
    (statements,) = files.by_filename.values()

    codes = {lint.code for lint in lint_statements(statements, named, assume_defined=False)}
    assert codes == {LintCode.undefined_reference, LintCode.unknown_element_type}

    kept = lint_statements(statements, named, assume_defined=False, ignore=["LF003"])
    assert {lint.code for lint in kept} == {LintCode.undefined_reference}


def test_lint_files():
    files = _files("q1: quadrupole, bogus = 1")
    codes = {lint.code for _fn, lint in lint_files(files)}
    assert LintCode.unknown_attribute in codes


def test_lint_cli_exits_nonzero_on_findings(tmp_path):
    path = tmp_path / "t.bmad"
    path.write_text("q1: quadrupole, bogus = 1\n")
    with pytest.raises(SystemExit) as excinfo:
        cli_main([str(path)])
    assert excinfo.value.code == 1


def test_lint_cli_exits_zero_when_clean(tmp_path):
    path = tmp_path / "t.bmad"
    path.write_text("q1: quadrupole, k1 = 0.5\n")
    with pytest.raises(SystemExit) as excinfo:
        cli_main([str(path)])
    assert excinfo.value.code == 0


def test_lint_cli_ignore_suppresses_findings(tmp_path):
    path = tmp_path / "t.bmad"
    path.write_text("q1: quadrupole, bogus = 1\n")
    with pytest.raises(SystemExit) as excinfo:
        cli_main([str(path), "--ignore", "LF004"])
    assert excinfo.value.code == 0


def test_main_lint_flag_gates_warnings(tmp_path, capsys, caplog):
    from ..main import main

    path = tmp_path / "t.bmad"
    path.write_text("q1: quadrupole, bogus = 1\n")

    with caplog.at_level(logging.WARNING, logger="latform.main"):
        main(filename=str(path))
    assert not any("LF004" in rec.getMessage() for rec in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="latform.main"):
        main(filename=str(path), lint=True)
    assert any("LF004" in rec.getMessage() for rec in caplog.records)


TAO_DEFAULT_LATTICE = "R0_MAR_END: marker\nml: line = (R0_MAR_END)\nuse, ml\n"


def _tao_files(tao_init: str, lattice: str = TAO_DEFAULT_LATTICE) -> MemoryFiles:
    files = MemoryFiles.from_tao_init_contents(
        tao_init,
        "proj/tao.init",
        lattice_contents={"ring.lat.bmad": lattice},
    )
    files.parse()
    files.annotate()
    return files


def _datum_init(ele_name: str) -> str:
    return (
        "&tao_design_lattice\n"
        "  design_lattice(1)%file = 'ring.lat.bmad'\n"
        "/\n\n"
        "&tao_d1_data\n"
        "  ix_d1_data = 1\n"
        "  d1_data%name = 'x'\n"
        f"  datum(1) = 'orbit.x' '' '' '{ele_name}' 'target' 0 1e1\n"
        "/\n"
    )


def test_datum_undefined_element_reported():
    files = _tao_files(_datum_init("MISSING"))
    lints = lint_datums(files.tao_init, files.get_named_items())
    assert [lint.code for lint in lints] == [LintCode.undefined_reference]
    assert "MISSING" in lints[0].message


def test_datum_defined_element_not_reported():
    files = _tao_files(_datum_init("R0_MAR_END"))
    assert lint_datums(files.tao_init, files.get_named_items()) == []


def test_datum_element_index_suffix_stripped():
    """The ``\\N`` slice suffix is not itself treated as an element reference."""
    files = _tao_files(_datum_init("R0_MAR_END\\2"))
    assert lint_datums(files.tao_init, files.get_named_items()) == []


def test_datum_component_form_undefined_reported():
    tao_init = (
        "&tao_design_lattice\n"
        "  design_lattice(1)%file = 'ring.lat.bmad'\n"
        "/\n\n"
        "&tao_d1_data\n"
        "  d1_data%name = 'x'\n"
        "  datum(1)%data_type = 'orbit.x'\n"
        "  datum(1)%ele_name = 'MISSING'\n"
        "/\n"
    )
    files = _tao_files(tao_init)
    lints = lint_datums(files.tao_init, files.get_named_items())
    assert [lint.code for lint in lints] == [LintCode.undefined_reference]


def test_datum_lints_surfaced_through_lint_files():
    files = _tao_files(_datum_init("MISSING"))
    reported = list(lint_files(files, assume_defined=False))
    codes = [lint.code for _fn, lint in reported]
    assert LintCode.undefined_reference in codes
    # The datum lint is attributed to the tao.init file.
    (init_fn,) = {fn for fn, lint in reported if lint.code is LintCode.undefined_reference}
    assert init_fn.name == "tao.init"


def test_no_datum_lints_without_tao_init():
    files = MemoryFiles.from_contents(TAO_DEFAULT_LATTICE, "test.bmad")
    files.parse()
    files.annotate()
    assert lint_datums(files.tao_init, files.get_named_items()) == []


def _var_init(ele_name: str) -> str:
    return (
        "&tao_design_lattice\n"
        "  design_lattice(1)%file = 'ring.lat.bmad'\n"
        "/\n\n"
        "&tao_var\n"
        "  v1_var%name = 'q'\n"
        f"  var(1) = '{ele_name}' 'k1' '' 1e2 1e-4\n"
        "/\n"
    )


def test_variable_undefined_element_reported():
    files = _tao_files(_var_init("MISSING"))
    lints = lint_variables(files.tao_init, files.get_named_items())
    assert [lint.code for lint in lints] == [LintCode.undefined_reference]
    assert "MISSING" in lints[0].message


def test_variable_defined_element_not_reported():
    files = _tao_files(_var_init("R0_MAR_END"))
    assert lint_variables(files.tao_init, files.get_named_items()) == []


def test_variable_component_form_undefined_reported():
    tao_init = (
        "&tao_design_lattice\n"
        "  design_lattice(1)%file = 'ring.lat.bmad'\n"
        "/\n\n"
        "&tao_var\n"
        "  v1_var%name = 'q'\n"
        "  var(1)%ele_name = 'MISSING'\n"
        "  var(1)%attribute = 'k1'\n"
        "/\n"
    )
    files = _tao_files(tao_init)
    lints = lint_variables(files.tao_init, files.get_named_items())
    assert [lint.code for lint in lints] == [LintCode.undefined_reference]


def test_variable_lints_surfaced_through_lint_files():
    files = _tao_files(_var_init("MISSING"))
    reported = list(lint_files(files, assume_defined=False))
    codes = [lint.code for _fn, lint in reported]
    assert LintCode.undefined_reference in codes
    (init_fn,) = {fn for fn, lint in reported if lint.code is LintCode.undefined_reference}
    assert init_fn.name == "tao.init"


def test_datum_lint_attributed_to_split_data_file():
    """A datum from a split-out data_file is attributed to that file, not tao.init."""
    contents = (
        "&tao_design_lattice\n"
        "  design_lattice(1)%file = 'ring.lat.bmad'\n"
        "/\n\n"
        "&tao_start\n"
        "  data_file = 'x.dat.bmad'\n"
        "/\n"
    )
    files = MemoryFiles.from_tao_init_contents(
        contents,
        "proj/tao.init",
        lattice_contents={
            "ring.lat.bmad": TAO_DEFAULT_LATTICE,
            "x.dat.bmad": (
                "&tao_d1_data\n"
                "  d1_data%name = 'x'\n"
                "  datum(1) = 'orbit.x' '' '' 'MISSING' 'target'\n"
                "/\n"
            ),
        },
    )
    files.parse()
    files.annotate()
    reported = list(lint_files(files, assume_defined=False))
    (data_fn,) = {fn for fn, lint in reported if lint.code is LintCode.undefined_reference}
    assert data_fn.name == "x.dat.bmad"
