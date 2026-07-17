"""Free-form namelist input: continued values, shared lines, inline terminators."""

from __future__ import annotations

import pathlib

import pytest

from .._namelist import NamelistFile, _find_terminator, _scan_namelist, split_values
from ..tao import TaoInit
from ..token import Token

FILES = pathlib.Path(__file__).resolve().parent / "files" / "tao_init"
FREEFORM = FILES / "freeform.init"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("/", 0),
        ("  /", 2),
        ("  x = 1 /", 8),
        ("&g x = 1 /", 9),
        ("'a / b' /", 8),
        ("  file = 'a/b'", None),
        ("  ! a comment with a /", None),
        ("  x = 1 ! /", None),
        ("  x = a(1/2)", None),  # '/' inside a subscript does not terminate
    ],
)
def test_find_terminator(line: str, expected: int | None):
    assert _find_terminator(line) == expected


def test_scan_multiple_pairs_per_line():
    scan = _scan_namelist(["&g", "  y = 2  z = 3", "/"])
    assert [(a.key, str(a.value)) for a in scan.assignments] == [("y", "2"), ("z", "3")]


def test_scan_repeat_count_merges_with_quote():
    scan = _scan_namelist(["&g", "  x = 2*'a' 'b'", "/"])
    (assignment,) = scan.assignments
    assert [str(v) for v in assignment.values] == ["'a'", "'a'", "'b'"]


def test_scan_doubled_quote_escape():
    scan = _scan_namelist(["&g", "  s = 'it''s'", "/"])
    (assignment,) = scan.assignments
    assert str(assignment.value) == "'it''s'"
    assert [str(v) for v in assignment.values] == ["'it''s'"]


def test_split_values_doubled_quote():
    assert [str(v) for v in split_values(Token("'it''s' 'b'"))] == ["'it''s'", "'b'"]


def test_continuation_values_joined_and_located():
    source = "&g\n  x = 4, 5,\n      6\n/\n"
    (group,) = NamelistFile.parse(source).namelists
    (assignment,) = group.assignments
    assert assignment.key == "x"
    assert [str(v) for v in assignment.values] == ["4", "5", "6"]
    assert str(assignment.value) == "4, 5\n6"
    assert assignment.value.loc.line == 1
    assert assignment.value.loc.end_line == 2
    for value in assignment.values:
        assert value.loc.get_string(source) == str(value)


def test_continued_quoted_string_is_not_a_phantom_assignment():
    (group,) = NamelistFile.parse("&g\n  s = 'a',\n      'x = y'\n/\n").namelists
    (assignment,) = group.assignments
    assert assignment.key == "s"
    assert [str(v) for v in assignment.values] == ["'a'", "'x = y'"]


def test_interior_comment_on_continuation_line():
    (group,) = NamelistFile.parse("&g\n  x = 1, ! part one\n      2\n/\n").namelists
    (assignment,) = group.assignments
    assert assignment.comment == "part one"
    assert [str(v) for v in assignment.values] == ["1", "2"]


def test_empty_rhs_keeps_following_assignment():
    (group,) = NamelistFile.parse("&g\n  x =\n  y = 1\n/\n").namelists
    assert [(a.key, str(a.value)) for a in group.assignments] == [("x", ""), ("y", "1")]


def test_group_ends_at_inline_terminator_rest_is_raw():
    source = "&g\n  x = 1 /\nafter_group_text\n"
    nml = NamelistFile.parse(source)
    (group,) = nml.namelists
    assert [(a.key, str(a.value)) for a in group.assignments] == [("x", "1")]
    raw = [item for item in nml.items if isinstance(item, str)]
    assert any("after_group_text" in chunk for chunk in raw)
    assert nml.render() == source


def test_opener_line_assignments_parsed():
    source = "&g x = 1 /\nafter\n"
    nml = NamelistFile.parse(source)
    (group,) = nml.namelists
    assert [(a.key, str(a.value)) for a in group.assignments] == [("x", "1")]
    assert nml.render() == source


def test_set_collapses_multiline_value_without_orphans():
    nml = NamelistFile.parse("&g\n  x = 1, 2, 3,\n      4, 5, 6\n  y = 7\n/\n")
    (group,) = nml.namelists
    group.set("x", "99")
    assert group.render() == "&g\n  x = 99\n  y = 7\n/"
    # A reparse of the rendered text reads back exactly one value.
    (reparsed,) = NamelistFile.parse(nml.render()).namelists
    assert [str(v) for v in reparsed.get("x").values] == ["99"]


def test_set_single_line_value_is_unchanged_behavior():
    nml = NamelistFile.parse("&g\n  x = 1  ! keep me\n/\n")
    (group,) = nml.namelists
    group.set("x", "42")
    assert group.render() == "&g\n  x = 42  ! keep me\n/"


def test_set_appends_into_one_line_group():
    nml = NamelistFile.parse("&g x = 1 /\n")
    (group,) = nml.namelists
    group.set("y", "2")
    assert group.render() == "&g x = 1\n  y = 2\n/"
    assert str(group.get("x").value) == "1"
    assert str(group.get("y").value) == "2"


def test_set_multiline_replacement_value():
    nml = NamelistFile.parse("&g\n  x = 1\n/\n")
    (group,) = nml.namelists
    group.set("x", "'a',\n      'b'")
    assert group.render() == "&g\n  x = 'a',\n      'b'\n/"
    assert [str(v) for v in group.get("x").values] == ["'a'", "'b'"]


def test_remove_multiline_value_removes_all_lines():
    nml = NamelistFile.parse("&g\n  x = 1, 2,\n      3\n  y = 7\n/\n")
    (group,) = nml.namelists
    group.remove("x")
    assert group.render() == "&g\n  y = 7\n/"


def test_remove_one_pair_from_shared_line_keeps_other():
    nml = NamelistFile.parse("&g\n  y = 2  z = 3\n/\n")
    (group,) = nml.namelists
    group.remove("z")
    assert group.render() == "&g\n  y = 2\n/"

    nml = NamelistFile.parse("&g\n  y = 2, z = 3\n/\n")
    (group,) = nml.namelists
    group.remove("y")
    assert group.render() == "&g\n  z = 3\n/"


def test_remove_assignment_from_one_line_group_keeps_group():
    nml = NamelistFile.parse("&g a = 1 /\n")
    (group,) = nml.namelists
    group.remove("a")
    assert group.render() == "&g /"
    assert group.assignments == []


def test_interpolate_namelist_on_wrapped_value():
    from ..apply import interpolate_namelist

    source = "&g\n  x = 1, 2,\n      3\n/\n"
    out = interpolate_namelist(source, values={"g": {"x": "9"}})
    assert out == "&g\n  x = 9\n/\n"


def test_freeform_wrapped_positional_datum():
    tao = TaoInit.from_file(FREEFORM)
    datums = tao.d1_data[0].datums
    assert [d.index for d in datums] == [1, 2]
    first = datums[0]
    assert first.data_type == "orbit.x"
    assert first.ele_ref_name == ""
    assert first.ele_name == "END\\2"
    assert first.merit_type == "target"
    assert first.meas == "0"
    assert first.weight == "1e1"
    assert first.comment == "continues below"
    assert datums[1].data_type == "orbit.y"


def test_freeform_slice_distribution_across_wrapped_values():
    tao = TaoInit.from_file(FREEFORM)
    (v1,) = tao.variables
    variables = v1.variables
    assert [v.index for v in variables] == [1, 2, 3]
    assert [str(v.ele_name) for v in variables] == ["Q1", "Q2", "Q3"]
    assert [str(v.attribute) for v in variables] == ["k1", "k1", "k1"]


def test_freeform_pairs_on_one_line():
    nml = NamelistFile.from_file(FREEFORM)
    group = nml.get_namelist("tao_var")
    assert str(group.get("default_weight").value) == "1e1"
    assert str(group.get("default_step").value) == "1e-4"


def test_freeform_one_liner_and_raw_text():
    nml = NamelistFile.from_file(FREEFORM)
    group = nml.get_namelist("one_liner")
    assert group is not None
    assert [(a.key, str(a.value)) for a in group.assignments] == [
        ("a", "1"),
        ("b", "'has = / chars'"),
    ]
    raw = [item for item in nml.items if isinstance(item, str)]
    assert any("raw text after the one-liner group" in chunk for chunk in raw)


def test_freeform_strings_group():
    nml = NamelistFile.from_file(FREEFORM)
    group = nml.get_namelist("strings")
    values = {a.key: str(a.value) for a in group.assignments}
    assert values == {"s1": "'it''s'", "s2": "'a / b'", "s3": "'x'\n'x = y'"}


def test_format_wrapped_record_keeps_wrapping():
    from ..apply import interpolate_namelist
    from ..types import NamelistFormatOptions

    src = "&g\n  x = 1, 2,\n        3\n  yy = 4\n/\n"
    out = interpolate_namelist(src, options=NamelistFormatOptions())
    # Continuation values re-align under the first value character.
    assert out == "&g\n  x = 1, 2,\n      3\n  yy = 4\n/\n"


def test_format_one_line_group_expands():
    from ..apply import interpolate_namelist
    from ..types import NamelistFormatOptions

    out = interpolate_namelist("&one a = 1  b = 2 /\n", options=NamelistFormatOptions())
    assert out == "&one\n  a = 1\n  b = 2\n/\n"


def test_format_never_rewrites_quoted_equals():
    from ..apply import interpolate_namelist
    from ..types import NamelistFormatOptions

    src = "&g\n  s = 'a',\n      'x=y'\n/\n"
    out = interpolate_namelist(src, options=NamelistFormatOptions())
    assert out == src  # 'x=y' is string content, not an assignment


def test_format_aligns_continuation_comments():
    from ..apply import interpolate_namelist
    from ..types import NamelistFormatOptions

    src = "&g\n  x = 1, ! one\n      2  ! two\n/\n"
    out = interpolate_namelist(src, options=NamelistFormatOptions())
    assert out == "&g\n  x = 1,  ! one\n      2   ! two\n/\n"


def test_format_preserves_text_after_terminator():
    from ..apply import interpolate_namelist
    from ..types import NamelistFormatOptions

    out = interpolate_namelist("&g x = 1 / trailing\n", options=NamelistFormatOptions())
    assert out == "&g\n  x = 1\n/ trailing\n"


def test_format_freeform_is_idempotent():
    from ..apply import interpolate_namelist
    from ..types import NamelistFormatOptions

    options = NamelistFormatOptions(align_equals=True)
    once = interpolate_namelist(FREEFORM.read_text(), options=options)
    twice = interpolate_namelist(once, options=options)
    assert once == twice
    # The formatted output still reads back with identical field values.
    before = TaoInit.parse(FREEFORM.read_text())
    after = TaoInit.parse(once)
    key = lambda tao: [  # noqa: E731
        [str(v) for v in a.values] for group in tao.namelists for a in group.assignments
    ]
    assert key(after) == key(before)
