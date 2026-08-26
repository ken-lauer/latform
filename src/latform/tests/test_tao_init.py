from __future__ import annotations

import os
import pathlib

import pytest
from nmlform import KeyPath, Namelist, NamelistFile
from nmlform import Token as NmlToken

from ..parser import Files, MemoryFiles, build_files
from ..tao import TaoInit, is_init_file

MODULE_PATH = pathlib.Path(__file__).resolve().parent
FILES = MODULE_PATH / "files" / "tao_init"
FEATURES = FILES / "features.init"
PROJ_INIT = FILES / "proj" / "tao.init"
D1_DATA_INIT = FILES / "d1_data.init"
FREEFORM_INIT = FILES / "freeform.init"

CORPUS = [FEATURES, PROJ_INIT, D1_DATA_INIT, FREEFORM_INIT]

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


USE_LINE_INIT = """\
&tao_design_lattice
  design_lattice(2)%file = 'b.lat.bmad@lineA'
  design_lattice(1)%file = 'a.lat.bmad'
  design_lattice(3)%file = 'sub/c.lat.bmad@lineA@lineB'
/
"""


def test_lattice_file_with_use_line():
    tao = TaoInit.parse(USE_LINE_INIT)
    assert tao.lattice_file_with_use_line == [
        ("a.lat.bmad", []),
        ("b.lat.bmad", ["lineA"]),
        ("sub/c.lat.bmad", ["lineA", "lineB"]),
    ]
    # The '@use_line' suffixes are stripped from the plain filename view
    assert tao.lattice_files == ["a.lat.bmad", "b.lat.bmad", "sub/c.lat.bmad"]
    assert tao.render() == USE_LINE_INIT


def test_set_lattice_file_with_use_line():
    tao = TaoInit.parse(USE_LINE_INIT)
    entries = [("a2.lat.bmad", []), ("b2.lat.bmad", ["lineC"]), ("c2.lat.bmad", ["l1", "l2"])]
    tao.lattice_file_with_use_line = entries
    assert tao.lattice_file_with_use_line == entries
    assert tao.lattice_files == ["a2.lat.bmad", "b2.lat.bmad", "c2.lat.bmad"]
    assert TaoInit.parse(tao.render()).lattice_file_with_use_line == entries


def test_set_lattice_files_passes_at_suffix_through():
    """The plain setter writes '@' suffixes verbatim; the getter strips them (asymmetric)."""
    tao = TaoInit.parse(USE_LINE_INIT)
    tao.lattice_files = ["x.lat.bmad@ln"]
    assert tao.lattice_files == ["x.lat.bmad"]
    assert tao.lattice_file_with_use_line == [("x.lat.bmad", ["ln"])]


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


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("var(3)%ele_name", [3]),
        ("var(1 : 6)%ele_name", [1, 2, 3, 4, 5, 6]),
        ("var(1:6: 2)%ele_name", [1, 3, 5]),
        ("var(-2:1)%ele_name", [-2, -1, 0, 1]),
        ("var(:3)%ele_name", [1, 2, 3]),  # missing start: lower bound of 1
        ("var(:6:2)%ele_name", [1, 3, 5]),
        ("var(5:1:-2)%ele_name", [5, 3, 1]),  # negative stride descends
        ("var(1:5:-1)%ele_name", []),  # wrong-direction range: no elements
        ("var%ele_name", None),  # no subscript
        ("var(N)%ele_name", None),  # named/non-integer bound
        ("var(1:6:0)%ele_name", None),  # zero stride
        ("var(1,2)%ele_name", None),  # multi-dimensional subscript
    ],
)
def test_keycomponent_indices(key: str, expected: list[int] | None):
    assert KeyPath.parse(key).components[0].indices == expected


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


def test_memory_files_from_tao_init_use_line():
    """A '@use_line' suffix does not interfere with resolving the lattice file itself."""
    contents = "&tao_design_lattice\n  design_lattice(1)%file = 'mem.lat.bmad@ml'\n/\n"
    root = FILES / "virtual" / "tao.init"
    files = MemoryFiles.from_tao_init_contents(
        contents,
        root,
        lattice_contents={"mem.lat.bmad": "M_Q: quadrupole, l = 1\nml: line = (M_Q)\nuse, ml\n"},
    )
    assert [p.name for p in files.top_files] == ["mem.lat.bmad"]
    files.parse()
    files.annotate()
    assert "M_Q" in files.get_named_items()


@pytest.mark.parametrize(
    ("name", "expected"),
    [("tao.init", True), ("tao_plot.init", True), ("foo.INIT", True), ("lat.bmad", False)],
)
def test_is_init_file(name: str, expected: bool):
    assert is_init_file(name) is expected


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


def test_datum_value_token_keeps_location():
    source = "&tao_d1_data\n  datum(1) = 'orbit.x' '' '' 'END\\2' 'target'\n/"
    tao = TaoInit.parse(source)
    (datum,) = tao.d1_data[0].datums
    ele = datum.ele_name
    assert isinstance(ele, NmlToken)
    assert ele == "END\\2"
    # The unquoted token still carries a source location pointing at the field.
    assert ele.loc.get_string(source) == "'END\\2'"


def test_variables_parsed():
    tao = TaoInit.parse(
        "&tao_var\n"
        "  v1_var%name = 'quad_k1'\n"
        "  default_attribute = 'k1'\n"
        "  var(1) = 'Q1' 'k1' '' 1e2 1e-4\n"
        "  var(2)%ele_name = 'Q2'\n"
        "/\n"
    )
    (v1,) = tao.variables
    assert v1.name == "quad_k1"
    assert v1.default_attribute == "k1"
    variables = v1.variables
    assert [v.index for v in variables] == [1, 2]
    assert variables[0].ele_name == "Q1"
    assert variables[0].attribute == "k1"
    assert variables[0].weight == "1e2"
    assert variables[0].step == "1e-4"
    assert variables[1].ele_name == "Q2"


def test_variable_slice_assignment_distributes_values():
    tao = TaoInit.parse(
        "&tao_var\n"
        "  v1_var%name = 'twiss'\n"
        "  var(1:6)%ele_name  = 'beginning', 'beginning', 'beginning', "
        "'beginning', 'beginning', 'beginning'\n"
        "  var(1:6)%attribute = 'beta_a', 'alpha_a', 'beta_b', 'alpha_b', "
        "'eta_x', 'etap_x'\n"
        "/\n"
    )
    (v1,) = tao.variables
    variables = v1.variables
    assert [v.index for v in variables] == [1, 2, 3, 4, 5, 6]
    assert [v.ele_name for v in variables] == ["beginning"] * 6
    assert [v.attribute for v in variables] == [
        "beta_a",
        "alpha_a",
        "beta_b",
        "alpha_b",
        "eta_x",
        "etap_x",
    ]


def test_slice_assignment_repeat_count():
    tao = TaoInit.parse(
        "&tao_var\n"
        "  var(1:6)%ele_name  = 6*'beginning'\n"
        "  var(1:6)%attribute = 'beta_a', 'alpha_a', 'beta_b', 'alpha_b', "
        "'eta_x', 'etap_x'\n"
        "/\n"
    )
    (v1,) = tao.variables
    assert [v.ele_name for v in v1.variables] == ["beginning"] * 6
    assert v1.variables[2].attribute == "beta_b"


def test_slice_assignment_shortfall_and_scalar_merge():
    tao = TaoInit.parse(
        "&tao_var\n"
        "  var(1:4)%ele_name  = 'A', 'B'\n"  # fewer values than the 1:4 section
        "  var(2)%attribute   = 'k1'\n"  # scalar merges into the slice-made entry
        "/\n"
    )
    (v1,) = tao.variables
    variables = v1.variables
    # Only indices that received a value exist; 3 and 4 are not conjured.
    assert [v.index for v in variables] == [1, 2]
    assert variables[0].ele_name == "A"
    assert variables[1].ele_name == "B"
    assert variables[1].attribute == "k1"


def test_open_ended_slice_extent_from_values():
    # ``var(1:)%x = a, b`` fills entries 1..2: the extent comes from the values.
    tao = TaoInit.parse(
        "&tao_var\n"
        "    v1_var%name = 'connect'\n"
        "    default_step = 1e-4\n"
        "    default_attribute = 'L'\n"
        "    var(1:)%ele_name = 'FOO1_PIP5', 'FOO2_PIP0'\n"
        '  !  search_for_lat_eles = "quad::*"\n'
        "   default_key_bound = T\n"
        "   default_key_delta = 0.1\n"
        "/\n"
    )
    (v1,) = tao.variables
    variables = v1.variables
    assert [v.index for v in variables] == [1, 2]
    assert [str(v.ele_name) for v in variables] == ["FOO1_PIP5", "FOO2_PIP0"]


def test_open_ended_slice_start_offset():
    tao = TaoInit.parse("&tao_var\n  var(3:)%ele_name = 'A', 'B'\n/\n")
    (v1,) = tao.variables
    assert [v.index for v in v1.variables] == [3, 4]


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("var(1:)%x", 1),
        ("var(3:)%x", 3),
        ("var(-2:)%x", -2),
        ("var(:)%x", 1),  # missing start: lower bound of 1
        ("var(2::3)%x", 2),
        ("var(1:6)%x", None),  # closed sections enumerate via .indices
        ("var(1)%x", None),
        ("var%x", None),
        ("var(N:)%x", None),
    ],
)
def test_keycomponent_slice_start(key: str, expected: int | None):
    assert KeyPath.parse(key).components[0].slice_start == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("var(:)%x", (1, 1)),
        ("var(2:)%x", (2, 1)),
        ("var(2::3)%x", (2, 3)),
        ("var(::2)%x", (1, 2)),
        ("var(2::0)%x", None),  # zero step
        ("var(1:6)%x", None),
        ("var(1)%x", None),
    ],
)
def test_keycomponent_open_slice(key: str, expected: tuple[int, int] | None):
    assert KeyPath.parse(key).components[0].open_slice == expected


def test_slice_whitespace_and_empty_section():
    # Whitespace inside a subscript is fine (keys are whitespace-normalized),
    # and an ascending a:b with a > b is an empty section that assigns nothing.
    tao = TaoInit.parse(
        "&tao_var\n  var(1 : 3)%ele_name = 'A', 'B', 'C'\n  var(5 : 3)%ele_name = 'X', 'Y'\n/\n"
    )
    (v1,) = tao.variables
    assert [v.index for v in v1.variables] == [1, 2, 3]
    assert [v.ele_name for v in v1.variables] == ["A", "B", "C"]


def test_datum_slice_assignment_distributes_values():
    tao = TaoInit.parse(
        "&tao_d1_data\n"
        "  d1_data%name = 'x'\n"
        "  datum(1:3)%data_type = 'orbit.x', 'orbit.y', 'orbit.z'\n"
        "  datum(1:3)%ele_name  = 3*'END'\n"
        "/\n"
    )
    (d1,) = tao.d1_data
    datums = d1.datums
    assert [d.index for d in datums] == [1, 2, 3]
    assert [d.data_type for d in datums] == ["orbit.x", "orbit.y", "orbit.z"]
    assert [d.ele_name for d in datums] == ["END", "END", "END"]


def test_tao_start_file_properties():
    tao = TaoInit.parse(
        "&tao_start\n"
        "  n_universes = 1\n"
        "  plot_file    = '$KYBER/tao_plot.init '\n"
        "  data_file = 'xA.dat.bmad'\n"
        "  var_file = 'xA.var.bmad'\n"
        "/\n"
    )
    assert tao.data_file == "xA.dat.bmad"
    assert tao.var_file == "xA.var.bmad"
    assert tao.n_universes == "1"
    assert tao.beam_file is None
    # Note the intentional space suffix:
    assert tao.plot_file == "$KYBER/tao_plot.init "


def test_source_for_falls_back_to_self():
    tao = TaoInit.parse(
        "&tao_d1_data\n  d1_data%name = 'x'\n  datum(1) = 'orbit.x' '' '' 'END' 'target'\n/\n"
    )
    # No &tao_start data_file → the category is read from the tao.init itself.
    assert tao._source_for("tao_d1_data") is tao
    (d1,) = tao.d1_data
    assert d1.name == "x"


def test_load_sources_reads_split_data_and_vars(tmp_path):
    (tmp_path / "d.dat").write_text(
        "&tao_d1_data\n  d1_data%name = 'x'\n  datum(1) = 'orbit.x' '' '' 'END' 'target' 0 1e1\n/\n"
    )
    (tmp_path / "v.var").write_text(
        "&tao_var\n  v1_var%name = 'q'\n  var(1) = 'Q1' 'k1' '' 1e2 1e-4\n/\n"
    )
    tao = TaoInit.parse(
        "&tao_start\n  data_file = 'd.dat'\n  var_file = 'v.var'\n/\n",
        filename=tmp_path / "tao.init",
    )
    tao.load_sources()

    assert tao._source_for("tao_d1_data") is not tao
    (d1,) = tao.d1_data
    assert d1.name == "x"
    assert d1.datums[0].ele_name == "END"

    (v1,) = tao.variables
    assert v1.name == "q"
    assert v1.variables[0].ele_name == "Q1"
    assert v1.variables[0].attribute == "k1"


def test_load_sources_skips_missing_aux_file(tmp_path):
    tao = TaoInit.parse(
        "&tao_start\n  data_file = 'nope.dat'\n/\n",
        filename=tmp_path / "tao.init",
    )
    tao.load_sources()
    # A named-but-missing file is skipped; the category falls back to self.
    assert "data_file" not in tao.sources
    assert tao._source_for("tao_d1_data") is tao
    assert tao.d1_data == []


def test_memory_files_split_data_file():
    contents = (
        "&tao_design_lattice\n  design_lattice(1)%file = 'mem.lat.bmad'\n/\n"
        "&tao_start\n  data_file = 'd.dat.bmad'\n/\n"
    )
    root = FILES / "virtual" / "tao.init"
    files = MemoryFiles.from_tao_init_contents(
        contents,
        root,
        lattice_contents={
            "mem.lat.bmad": "M_Q: quadrupole, l = 1\nml: line = (M_Q)\nuse, ml\n",
            "d.dat.bmad": (
                "&tao_d1_data\n"
                "  d1_data%name = 'x'\n"
                "  datum(1) = 'orbit.x' '' '' 'M_Q' 'target'\n"
                "/\n"
            ),
        },
    )
    assert files.tao_init is not None
    (d1,) = files.tao_init.d1_data
    assert d1.name == "x"
    assert d1.datums[0].ele_name == "M_Q"


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
