from __future__ import annotations

import logging

import pytest
from nmlform import NamelistFile

from ..tao import collect_tao_element_names, rename_tao_elements

RENAMES = {"Q1": "C1_Q1", "B2": "C1_B2"}


@pytest.fixture(autouse=True)
def _propagate_latform_logs():
    """
    Re-enable propagation on the ``latform`` logger for caplog.

    The LSP tests call ``lsp.configure_logging``, which sets
    ``propagate = False`` on the package logger and leaves it that way;
    caplog's root handler would then never see this module's log records.
    """
    logger = logging.getLogger("latform")
    old = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = old


def renamed(source: str, renames: dict[str, str] = RENAMES) -> str:
    nml = NamelistFile.parse(source, filename="tao.init")
    rename_tao_elements(nml, renames)
    return nml.render()


@pytest.mark.parametrize("field", ["ele_name", "ele_ref_name", "ele_start_name"])
def test_datum_element_name_fields_renamed(field: str):
    out = renamed(f"&tao_d1_data\n  datum(1)%{field} = 'q1'\n/\n")
    assert f"datum(1)%{field} = 'C1_Q1'" in out


def test_var_ele_name_renamed():
    out = renamed("&tao_var\n  var(1)%ele_name = 'Q1'\n/\n")
    assert "var(1)%ele_name = 'C1_Q1'" in out


def test_curve_ele_ref_name_renamed():
    out = renamed("&tao_template_graph\n  curve(1)%ele_ref_name = 'Q1'\n/\n")
    assert "curve(1)%ele_ref_name = 'C1_Q1'" in out


def test_occurrence_and_multipass_suffixes_preserved():
    out = renamed("&tao_d1_data\n  datum(1)%ele_name = 'Q1##2'\n  datum(2)%ele_name = 'Q1\\2'\n/\n")
    assert "datum(1)%ele_name = 'C1_Q1##2'" in out
    assert "datum(2)%ele_name = 'C1_Q1\\2'" in out


def test_unrelated_names_and_fields_untouched():
    src = "&tao_var\n  var(1)%ele_name = 'Q9'\n  var(1)%attribute = 'q1'\n/\n"
    assert renamed(src) == NamelistFile.parse(src, filename="tao.init").render()


def test_unknown_namelist_untouched():
    src = "&my_custom\n  ele_name = 'Q1'\n/\n"
    assert "ele_name = 'Q1'" in renamed(src)


def test_pattern_exact_name_with_class_prefix_renamed():
    out = renamed("&lat_layout_drawing\n  ele_shape(1)%ele_id = 'quadrupole::Q1'\n/\n")
    assert "ele_shape(1)%ele_id = 'quadrupole::C1_Q1'" in out


def test_pattern_wildcard_left_untouched_with_warning(caplog: pytest.LogCaptureFixture):
    src = "&tao_d1_data\n  search_for_lat_eles = 'quad::Q*'\n/\n"
    with caplog.at_level(logging.WARNING, logger="latform.tao.rename"):
        out = renamed(src)
    assert "search_for_lat_eles = 'quad::Q*'" in out
    assert any("wildcards" in record.message for record in caplog.records)


def test_expression_element_in_lat_slot_renamed():
    out = renamed(
        "&tao_d1_data\n  datum(1)%data_type = 'expression: lat::orbit.x[Q1]|model - 1'\n/\n"
    )
    assert "'expression: lat::orbit.x[C1_Q1]|model - 1'" in out


def test_expression_element_range_in_slot_renamed():
    out = renamed("&tao_d1_data\n  datum(1)%data_type = 'expression: lat::orbit.x[Q1:B2]'\n/\n")
    assert "lat::orbit.x[C1_Q1:C1_B2]" in out


def test_expression_element_in_ele_path_renamed():
    out = renamed("&tao_d1_data\n  datum(1)%data_type = 'expression: 2*ele::q1[k1]'\n/\n")
    assert "2*ele::C1_Q1[k1]" in out


def test_expression_data_and_var_refs_untouched():
    src = "&tao_d1_data\n  datum(1)%data_type = 'expression: data::q1.x[1]|meas + var::q1[2]'\n/\n"
    out = renamed(src)
    assert "data::q1.x[1]|meas + var::q1[2]" in out


def test_expression_unrecognized_scope_logged_at_debug(caplog: pytest.LogCaptureFixture):
    src = "&tao_d1_data\n  datum(1)%data_type = 'expression: mystery::q1[1]'\n/\n"
    with caplog.at_level(logging.DEBUG, logger="latform.tao.rename"):
        out = renamed(src)
    assert "mystery::q1[1]" in out
    assert any("unrecognized scope" in record.message.lower() for record in caplog.records)


def test_expression_unscoped_ref_logged_at_debug(caplog: pytest.LogCaptureFixture):
    src = "&tao_d1_data\n  datum(1)%data_type = 'expression: orbit.x[1] + 1'\n/\n"
    with caplog.at_level(logging.DEBUG, logger="latform.tao.rename"):
        out = renamed(src)
    assert "orbit.x[1] + 1" in out
    assert any("unscoped reference" in record.message.lower() for record in caplog.records)


def test_plain_data_type_without_refs_untouched():
    out = renamed("&tao_d1_data\n  default_data_type = 'orbit.x'\n/\n")
    assert "default_data_type = 'orbit.x'" in out


def test_positional_datum_element_name_renamed():
    """The ele fields of ``datum(i) = ...`` positional assignments are renamed."""
    out = renamed(
        "&tao_d1_data\n"
        "  datum(1) = 'beta.a' '' '' 'Q1' 'target' 1.0 1e1\n"
        "  datum(2) = 'expression: ele::Q1[k1]' 'B2' '' 'Q1' 'target' 0 1e1\n"
        "/\n"
    )
    assert "'beta.a' '' '' 'C1_Q1' 'target' 1.0 1e1" in out
    assert "'expression: ele::C1_Q1[k1]' 'C1_B2' '' 'C1_Q1' 'target' 0 1e1" in out


def test_positional_var_element_name_renamed():
    out = renamed("&tao_var\n  var(1) = 'q1' 'k1'\n/\n")
    assert "var(1) = 'C1_Q1' 'k1'" in out


def test_positional_attribute_slot_untouched():
    """A positional value in a non-element slot is not renamed even if it matches."""
    out = renamed("&tao_var\n  var(1) = 'B2' 'q1'\n/\n")
    assert "var(1) = 'C1_B2' 'q1'" in out


def test_positional_repeat_spanning_element_fields_renamed():
    # 2*'Q1' fills ele_ref_name and ele_start_name — both element slots.
    out = renamed("&tao_d1_data\n  datum(1) = 'beta.a' 2*'Q1' 'B2' 'target' 0 1e1\n/\n")
    assert "'beta.a' 2*'C1_Q1' 'C1_B2' 'target' 0 1e1" in out


def test_collect_tao_element_names():
    src = (
        "&tao_d1_data\n"
        "  datum(1) = 'orbit.x' '' '' 'M_END' 'target' 0 1e1\n"
        "  datum(2)%ele_name = 'Q1##2'\n"
        "  datum(3)%data_type = 'expression: lat::orbit.x[B2] + data::orb.x[1]'\n"
        "  datum(4)%ele_name = 'end'\n"
        "/\n"
        "&tao_var\n  var(1)%ele_name = 'beginning'\n/\n"
        "&lat_layout_drawing\n  ele_shape(1)%ele_id = 'quad::S*'\n/\n"
    )
    nml = NamelistFile.parse(src, filename="tao.init")
    names = collect_tao_element_names(nml)
    # Base names from positional/component/expression slots; reserved
    # pseudo-elements, wildcard patterns, and data:: refs are excluded.
    assert names == {"M_END", "Q1", "B2"}


def test_rename_is_case_insensitive_on_old_name():
    out = renamed("&tao_var\n  var(1)%ele_name = 'q1'\n/\n", {"q1": "ZZ"})
    assert "var(1)%ele_name = 'ZZ'" in out


def test_empty_rename_map_is_noop():
    src = "&tao_var\n  var(1)%ele_name = 'Q1'\n/\n"
    assert "'Q1'" in renamed(src, {})
