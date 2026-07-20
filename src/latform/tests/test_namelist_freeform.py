"""Free-form namelist input: continued values, shared lines, inline terminators, quoting."""

from __future__ import annotations

import pathlib

import pytest

from .._namelist import (
    NamelistFile,
    _find_comment_index,
    _find_terminator,
    _scan_namelist,
    quote_value,
    unquote_value,
)
from ..tao import TaoD1Data, TaoInit
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


@pytest.mark.parametrize(
    "code",
    [
        "  datum(1) = 'orbit.x' '' '' 'END\\2' 'target' 0 1e1",
        "  var(1 : 6)%ele_name = 'a', 'b'",
        "  datum (1) = 'x'",
        "  a % b = 1",
        "  a %b = 1  c% d = 2",
        "  x = 6*'beginning' 3*0 2*'a','b'",
        "  x = a(1/2) / ignored",
        "  s = 'it''s' \"d\"\"q\" 'unterminated",
        "  file='sub/c.lat.bmad'",
        "&g x = 1 /",
        "  x = 1, 2,",
        "      3",
        "  a*'q' 12* 'q' -1.5e-3",
        "",
        "   ",
        "  x = (1.0, 2.0)",
    ],
)
def test_lex_line_matches_scan_line(code: str):
    from .._namelist import _lex_line, _scan_line

    assert _lex_line(code) == list(_scan_line(code))


def test_lex_line_matches_scan_line_on_corpus():
    from .._namelist import _lex_line, _scan_line, _split_comment

    for path in FILES.glob("**/*.init"):
        for line in path.read_text().splitlines():
            code, _ = _split_comment(line)
            assert _lex_line(code) == list(_scan_line(code)), line


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("'it''s'", "it's"),
        ('"say ""hi"""', 'say "hi"'),
        ("''''", "'"),
        ("''", ""),
        ('"don\'t"', "don't"),
        ("'a''''b'", "a''b"),
        ("'a\"\"b'", 'a""b'),  # non-delimiter quotes are literal, not escapes
        ("'abc", "'abc"),  # unterminated: returned unchanged
        ("bare", "bare"),
        ("1.5e-3", "1.5e-3"),
    ],
)
def test_unquote_value(text: str, expected: str):
    assert str(unquote_value(Token(text))) == expected


def test_unquote_value_keeps_location_and_comments():
    from ..comments import Comments
    from ..location import Location

    token = Token(
        "'it''s'",
        loc=Location(filename=pathlib.Path("x.init"), line=3, column=7),
        comments=Comments(inline=Token("c")),
    )
    unquoted = unquote_value(token)
    assert unquoted.loc == token.loc
    assert unquoted.comments == token.comments


@pytest.mark.parametrize(
    ("text", "quote", "expected"),
    [
        ("it's", "'", "'it''s'"),
        ('say "hi"', '"', '"say ""hi"""'),
        ("plain", "'", "'plain'"),
        ("", "'", "''"),
        ("'", "'", "''''"),
        ('say "hi"', "'", "'say \"hi\"'"),
    ],
)
def test_quote_value(text: str, quote: str, expected: str):
    assert quote_value(text, quote) == expected


def test_quote_value_rejects_bad_quote():
    with pytest.raises(ValueError):
        quote_value("x", quote="`")


@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize("text", ["", "plain", "it's", 'say "hi"', "both ' and \" here", "''"])
def test_quote_unquote_round_trip(text: str, quote: str):
    assert str(unquote_value(Token(quote_value(text, quote)))) == text


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("x = 1 ! c", 6),  # fast path: no quotes before the '!'
        ("s = 'it''s ! x' ! real", 16),
        ("x = '''' ! c", 9),
        ('t = "say ""hi!"" " ! c', 19),
        ("x = 'abc ! unterminated", None),
        ("no comment here", None),
    ],
)
def test_find_comment_index_with_escaped_quotes(line: str, expected: int | None):
    assert _find_comment_index(line) == expected


ESCAPED_QUOTES_INIT = '''\
&tao_d1_data
  d1_data%name = 'it\'\'s'
  datum(1)%ele_name = "say ""hi"""
  datum(2) = 'orbit.x' \'\' \'\' 'Q\'\'1' 'target' 0 1e1
/
'''


def test_group_values_unescape_quotes():
    (namelist,) = NamelistFile.parse(ESCAPED_QUOTES_INIT).namelists
    group = TaoD1Data(namelist)
    assert str(group.name) == "it's"
    first, second = group.datums
    assert str(first.ele_name) == 'say "hi"'
    assert str(second.ele_name) == "Q'1"
    assert str(second.ele_ref_name) == ""


def test_render_preserves_escaped_quotes():
    nml = NamelistFile.parse(ESCAPED_QUOTES_INIT)
    assert nml.render() == ESCAPED_QUOTES_INIT


def test_lattice_files_setter_escapes_quotes():
    tao = TaoInit.parse("&tao_design_lattice\n/\n")
    tao.lattice_files = ["it's.bmad"]
    assert [str(f) for f in tao.lattice_files] == ["it's.bmad"]
    assert "'it''s.bmad'" in tao.render()


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
