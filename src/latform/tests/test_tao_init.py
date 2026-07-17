from __future__ import annotations

import os
import pathlib

import pytest

from .._namelist import KeyPath, Namelist, NamelistFile, split_values
from ..parser import Files, MemoryFiles, build_files
from ..tao import TaoInit, is_init_file
from ..token import Token

MODULE_PATH = pathlib.Path(__file__).resolve().parent
FILES = MODULE_PATH / "files" / "tao_init"
FEATURES = FILES / "features.init"
PROJ_INIT = FILES / "proj" / "tao.init"
D1_DATA_INIT = FILES / "d1_data.init"

CORPUS = [FEATURES, PROJ_INIT, D1_DATA_INIT]

# Local real-world tao.init files: drop symlinks to lattice repositories or
# files themselves under src/latform/tests/other-repos
#
# These are not version-controlled (.gitignore in 'other'), so the tests
# parametrized over them skip cleanly when none are present.
OTHER_REPOS = MODULE_PATH / "other-repos"


def _discover_other_tao_inits() -> list[pathlib.Path]:
    if not OTHER_REPOS.is_dir():
        return []
    found: list[pathlib.Path] = []
    for root, _dirs, files in os.walk(OTHER_REPOS, followlinks=True):
        if "tao.init" in files:
            found.append(pathlib.Path(root) / "tao.init")
    return sorted(found)


OTHER_TAO_INITS = _discover_other_tao_inits()


def _other_params() -> list:
    if not OTHER_TAO_INITS:
        return [
            pytest.param(
                None,
                id="none",
                marks=pytest.mark.skip(
                    reason="no user-provided tao.init under other-repos/ "
                    "(symlink a lattice repo there to exercise these)"
                ),
            )
        ]
    return [pytest.param(path, id=str(path.relative_to(OTHER_REPOS))) for path in OTHER_TAO_INITS]


_corpus_params = [pytest.param(fn, id=fn.name) for fn in CORPUS]


@pytest.mark.parametrize("path", _corpus_params + _other_params())
def test_roundtrip_byte_exact(path: pathlib.Path):
    text = path.read_text()
    assert NamelistFile.parse(text).render() == text


@pytest.mark.parametrize("path", _other_params())
def test_other_lattice_files_adjustable(path: pathlib.Path):
    """Rewriting the design lattices works on user-provided tao.init files."""
    tao = TaoInit.from_file(path)
    if tao.design_lattice is None:
        pytest.skip("no &tao_design_lattice group")

    new = [f"adjusted_{i}.lat.bmad" for i in range(1, 4)]
    tao.lattice_files = new
    assert tao.lattice_files == new
    # The edited file still parses and reports the adjusted lattices.
    assert TaoInit.parse(tao.render()).lattice_files == new


def test_lattice_files_ordered_and_skips_comments():
    tao = TaoInit.from_file(FEATURES)
    assert tao.lattice_files == [
        "$PROJ/a.lat.bmad",
        "b.lat.bmad",
        "sub/c.lat.bmad",
        "sub/d.lat.bmad",
    ]


def test_keypath_decomposes_nested_key():
    path = KeyPath.parse("foo(3)%bar(2)%val")
    assert path.names == ("foo", "bar", "val")
    assert tuple(c.index_text for c in path.components) == ("3", "2", None)
    assert tuple(c.index for c in path.components) == (3, 2, None)
    assert str(path) == "foo(3)%bar(2)%val"


def test_keypath_equality():
    path = KeyPath.parse("DESIGN_LATTICE(1)%file")
    # name-path checks go through .names
    assert path.names == ("design_lattice", "file")
    # KeyPath == KeyPath is index-sensitive (full components)
    assert path == KeyPath.parse("DESIGN_LATTICE(1)%file")
    assert path != KeyPath.parse("design_lattice(2)%file")


def test_keypath_range_index_is_none():
    path = KeyPath.parse("var(1:8)%ele_name")
    assert tuple(c.index for c in path.components) == (None, None)
    assert path.names == ("var", "ele_name")
    assert path.components[0].index_text == "1:8"


def test_nested_key_edits_and_round_trips():
    group = Namelist(name="t", lines=["&t", "  foo(3)%bar(2)%val = 1.5", "/"])
    assignment = group.get("foo(3)%bar(2)%val")
    assert assignment is not None
    assert assignment.path.names == ("foo", "bar", "val")
    group.set("foo(3)%bar(2)%val", "9.9")
    assert group.render() == "&t\n  foo(3)%bar(2)%val = 9.9\n/"


def test_assignment_location_points_at_value():
    text = PROJ_INIT.read_text()
    tao = TaoInit.parse(text, filename=PROJ_INIT)
    (first,) = [a for a in tao.design_lattice.assignments if a.key == "design_lattice(1)%file"]
    assert first.loc is not None
    assert first.loc.filename == PROJ_INIT
    # loc spans exactly the value text
    assert first.loc.get_string(text) == first.value == "'ring.lat.bmad'"
    # absolute (0-indexed) source line, single line span
    assert (
        first.loc.line
        == first.loc.end_line
        == text.split("\n").index("  design_lattice(1)%file = 'ring.lat.bmad'")
    )


def test_namelist_location_spans_block():
    text = PROJ_INIT.read_text()
    tao = TaoInit.parse(text, filename=PROJ_INIT)
    loc = tao.design_lattice.loc
    lines = text.split("\n")
    assert lines[loc.line].lstrip().startswith("&tao_design_lattice")
    assert lines[loc.end_line].lstrip().startswith("/")


def test_commented_group_opener_does_not_open_a_group():
    tao = TaoInit.from_file(FEATURES)
    by_name = tao.namelists_by_name
    assert "tao_d2_data" not in by_name  # `!&tao_d2_data` is a comment
    assert "tao_var" in by_name


def test_n_universes_read():
    tao = TaoInit.from_file(PROJ_INIT)
    assert tao.get_namelist("tao_start").get("n_universes").value == "2"


def test_set_value_preserves_everything_else():
    text = PROJ_INIT.read_text()
    tao = TaoInit.parse(text)
    tao.get_namelist("tao_params").set("global%n_opti_cycles", "42")
    expected = text.replace("global%n_opti_cycles = 100", "global%n_opti_cycles = 42")
    assert tao.render() == expected


def test_set_lattice_files_shrink():
    tao = TaoInit.from_file(PROJ_INIT)
    tao.lattice_files = ["only.bmad"]
    assert tao.lattice_files == ["only.bmad"]
    rendered = tao.design_lattice.render()
    assert "design_lattice(1)%file = 'only.bmad'" in rendered
    assert "design_lattice(2)" not in rendered
    # unrelated line preserved
    assert '!  unique_name_suffix="*::_##?"' in rendered


def test_set_lattice_files_grow():
    tao = TaoInit.from_file(PROJ_INIT)
    tao.lattice_files = ["a.bmad", "b.bmad", "c.bmad"]
    assert tao.lattice_files == ["a.bmad", "b.bmad", "c.bmad"]


def test_update_updates_existing_and_adds_new():
    tao = TaoInit.from_file(PROJ_INIT)
    tao.update_namelist("tao_params", {"global%track_type": "'beam'"})
    assert tao.get_namelist("tao_params").get("global%track_type").value == "'beam'"

    assert "tao_beam_init" not in tao.namelists_by_name
    tao.update_namelist("tao_beam_init", {"beam_init%n_particle": "5000"})
    added = tao.get_namelist("tao_beam_init")
    assert added is not None
    assert added.get("beam_init%n_particle").value == "5000"
    # the new section survives a render/reparse round-trip
    assert "tao_beam_init" in TaoInit.parse(tao.render()).namelists_by_name


def test_files_from_tao_init_loads_lattices():
    files = Files.from_tao_init(PROJ_INIT)
    assert [p.name for p in files.top_files] == ["ring.lat.bmad", "inj.lat.bmad"]
    assert files.tao_init is not None
    files.parse()
    files.annotate()
    named = files.get_named_items()
    assert "RING_Q1" in named
    assert "INJ_B1" in named


def test_build_files_auto_expands_init(monkeypatch):
    (single,) = build_files([str(PROJ_INIT)])
    assert [p.name for p in single.top_files] == ["ring.lat.bmad", "inj.lat.bmad"]

    (combined,) = build_files([str(PROJ_INIT)], combine=True)
    assert [p.name for p in combined.top_files] == ["ring.lat.bmad", "inj.lat.bmad"]


def test_memory_files_from_tao_init_contents():
    contents = "&tao_design_lattice\n  design_lattice(1)%file = 'mem.lat.bmad'\n/\n"
    root = FILES / "virtual" / "tao.init"
    files = MemoryFiles.from_tao_init_contents(
        contents,
        root,
        lattice_contents={"mem.lat.bmad": "M_Q: quadrupole, l = 1\nml: line = (M_Q)\nuse, ml\n"},
    )
    files.parse()
    files.annotate()
    assert "M_Q" in files.get_named_items()


@pytest.mark.parametrize(
    ("name", "expected"),
    [("tao.init", True), ("tao_plot.init", True), ("foo.INIT", True), ("lat.bmad", False)],
)
def test_is_init_file(name: str, expected: bool):
    assert is_init_file(name) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "'orbit.x' '' '' 'END\\2' 'target' 0 1e1",
            ["'orbit.x'", "''", "''", "'END\\2'", "'target'", "0", "1e1"],
        ),
        ("'a','b' , 'c'", ["'a'", "'b'", "'c'"]),
        ('"quoted value" bare', ['"quoted value"', "bare"]),
        ("", []),
        ("'unterminated", ["'unterminated"]),
    ],
)
def test_split_values(text: str, expected: list[str]):
    assert [str(v) for v in split_values(Token(text))] == expected


def test_split_values_locations_point_at_each_field():
    source = "&t\n  datum(1) = 'orbit.x' 'target'\n/"
    tao = TaoInit.parse(source)
    assignment = tao.get_namelist("t").get("datum(1)")
    (first, second) = split_values(assignment.value)
    # Each field's location slices its own text back out of the full source.
    assert first.loc.get_string(source) == first == "'orbit.x'"
    assert second.loc.get_string(source) == second == "'target'"


def test_d1_data_groups_parsed():
    tao = TaoInit.from_file(D1_DATA_INIT)
    d1s = tao.d1_data
    assert [d.name for d in d1s] == ["12", "y"]
    assert [d.ix_d1_data for d in d1s] == ["1", "2"]
    assert d1s[0].default_weight == "1e1"


def test_d1_data_positional_datums():
    tao = TaoInit.from_file(D1_DATA_INIT)
    datums = tao.d1_data[0].datums
    # The commented-out datum(1) is ignored; two active datums remain.
    assert [d.index for d in datums] == [1, 2]

    first = datums[0]
    assert first.data_type == "orbit.x"
    assert first.ele_ref_name == ""
    assert first.ele_start_name == ""
    assert first.ele_name == "R0_MAR_END\\2"
    assert first.merit_type == "target"
    assert first.meas == "0"
    assert first.weight == "1e1"
    # Nothing beyond the supplied fields.
    assert first.value("good_user") is None
    assert first.value("eval_point") is None

    assert datums[1].data_type == "orbit.px"
    assert datums[1].comment == "horizontal"


def test_d1_data_component_datums():
    tao = TaoInit.from_file(D1_DATA_INIT)
    (datum,) = tao.d1_data[1].datums
    assert datum.index == 1
    assert datum.data_type == "orbit.y"
    assert datum.ele_name == "R0_MAR_END\\2"
    assert datum.merit_type == "target"
    # A field neither positional nor set by component reads as unset.
    assert datum.weight is None


def test_d1_data_component_overrides_positional():
    tao = TaoInit.parse(
        "&tao_d1_data\n"
        "  datum(1) = 'orbit.x' '' '' 'ELE' 'target'\n"
        "  datum(1)%ele_name = 'OVERRIDE'\n"
        "/\n"
    )
    (datum,) = tao.d1_data[0].datums
    assert datum.data_type == "orbit.x"  # from positional
    assert datum.ele_name == "OVERRIDE"  # component wins


def test_no_d1_data():
    assert TaoInit.from_file(PROJ_INIT).d1_data == []


def test_blank_tao_init():
    init = TaoInit()
    assert init.lattice_files == []

    files = ["a.lat", "b.lat", "c.lat"]
    init.lattice_files = files
    assert init.lattice_files == files

    assert (
        init.render().strip()
        == """
&tao_design_lattice
  design_lattice(1)%file = 'a.lat'
  design_lattice(2)%file = 'b.lat'
  design_lattice(3)%file = 'c.lat'
/
""".strip()
    )


def test_blankish_tao_init():
    init = TaoInit.parse(
        """
&foo
  bar = 3
/
"""
    )
    assert init.lattice_files == []

    files = ["a.lat", "b.lat", "c.lat"]
    init.lattice_files = files
    assert init.lattice_files == files

    assert (
        init.render().strip()
        == """
&foo
  bar = 3
/

&tao_design_lattice
  design_lattice(1)%file = 'a.lat'
  design_lattice(2)%file = 'b.lat'
  design_lattice(3)%file = 'c.lat'
/
""".strip()
    )
