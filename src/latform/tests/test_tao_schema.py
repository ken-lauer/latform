from __future__ import annotations

import pytest

from ..lint import LintCode
from ..parser import MemoryFiles
from ..tao.lint import lint_tao_schema
from ..tao.schema import (
    NAMELISTS,
    STRUCTS,
    PathComponent,
    ProblemKind,
    check_value,
    is_known_namelist,
    resolve_path,
)


def _components(*parts: tuple[str, int | None] | str) -> list[PathComponent]:
    """Build path components from ``"name"`` or ``("name", index)`` parts."""
    return [
        PathComponent(part) if isinstance(part, str) else PathComponent(*part) for part in parts
    ]


# -- resolver: successful resolution to an intrinsic leaf ---------------------


@pytest.mark.parametrize(
    "namelist, parts, expected_base",
    [
        ("tao_start", ["n_universes"], "integer"),
        ("tao_start", ["startup_file"], "character"),
        ("tao_params", ["global", "n_opti_cycles"], "integer"),
        ("tao_params", ["global", "track_type"], "character"),
        ("tao_params", ["bmad_com", "radiation_damping_on"], "logical"),
        ("tao_design_lattice", [("design_lattice", 2), "file"], "character"),
        ("tao_var", ["v1_var", "name"], "character"),
        # Case-insensitive namelist and field names.
        ("TAO_PARAMS", ["GLOBAL", "N_OPTI_CYCLES"], "integer"),
    ],
)
def test_resolve_leaf(namelist, parts, expected_base):
    result = resolve_path(namelist, _components(*parts))
    assert result.problems == []
    assert not result.unresolved
    assert result.leaf is not None
    assert result.leaf.kind == "intrinsic"
    assert result.leaf.base == expected_base


# -- resolver: path problems ---------------------------------------------------


@pytest.mark.parametrize(
    "namelist, parts, kind",
    [
        ("tao_params", ["global", "no_such_field"], ProblemKind.unknown_field),
        ("tao_start", ["bogus_key"], ProblemKind.unknown_field),
        ("tao_design_lattice", [("design_lattice", 0), "file"], ProblemKind.index_out_of_bounds),
        ("tao_design_lattice", [("design_lattice", 300), "file"], ProblemKind.index_out_of_bounds),
        ("tao_params", [("n_data_max", 3)], ProblemKind.not_indexable),
        ("tao_params", [("global", 1), "track_type"], ProblemKind.not_indexable),
        ("tao_params", ["n_data_max", "foo"], ProblemKind.not_a_struct),
    ],
)
def test_resolve_problem(namelist, parts, kind):
    result = resolve_path(namelist, _components(*parts))
    assert [p.kind for p in result.problems] == [kind]


def test_resolve_index_bounds_reported():
    result = resolve_path("tao_design_lattice", _components(("design_lattice", 300), "file"))
    (problem,) = result.problems
    assert problem.bounds == (1, 200)
    assert problem.index == 300


def test_resolve_missing_struct_is_unresolved():
    # element_shapes.shape is old_tao_ele_shape_struct, absent from the schema.
    result = resolve_path("element_shapes", _components(("shape", 5), "shape_name"))
    assert result.unresolved
    assert result.leaf is None
    assert result.problems == []


def test_resolve_missing_struct_still_bound_checks_index():
    result = resolve_path("element_shapes", _components(("shape", 99), "shape_name"))
    assert [p.kind for p in result.problems] == [ProblemKind.index_out_of_bounds]


def test_resolve_unknown_namelist_returns_empty():
    result = resolve_path("not_a_tao_namelist", _components("whatever"))
    assert result.leaf is None
    assert result.problems == []
    assert not result.unresolved


def test_is_known_namelist():
    assert is_known_namelist("tao_params")
    assert is_known_namelist("TAO_PARAMS")
    assert not is_known_namelist("tao_template_curve")


# -- value literal type checks -------------------------------------------------


@pytest.mark.parametrize(
    "base, literal, valid",
    [
        ("character", "'quoted'", True),
        ("character", '"double_quoted"', True),
        ("character", "unquoted", False),
        ("integer", "100", True),
        ("integer", "-5", True),
        ("integer", "1.5", False),
        ("integer", "T", False),
        ("real", "3", True),
        ("real", "1.0", True),
        ("real", "1.0d0", True),
        ("real", "0.511e6", True),
        ("real", "pi", False),
        ("logical", "T", True),
        ("logical", "F", True),
        ("logical", ".true.", True),
        ("logical", ".FALSE.", True),
        ("logical", "5", False),
        ("logical", "yes", False),
        ("complex", "(1.0, 2.0)", True),
        ("complex", "(1.0)", False),
        # Fortran n*value repeat: the repeated value is what gets checked.
        ("real", "3*0.0", True),
        ("integer", "2*x", False),
    ],
)
def test_check_value(base, literal, valid):
    assert check_value(base, literal) is valid


# -- schema integrity ----------------------------------------------------------


def test_every_derived_reference_resolves_or_is_missing():
    from ..tao.schema import MISSING_STRUCTS

    known = set(STRUCTS) | set(MISSING_STRUCTS)
    for fields in (*NAMELISTS.values(), *STRUCTS.values()):
        for spec in fields.values():
            if spec.kind == "derived":
                assert spec.base in known


# -- lint integration ----------------------------------------------------------


def _tao_lints(contents: str, sources: dict[str, str] | None = None):
    files = MemoryFiles.from_tao_init_contents(contents, "/proj/tao.init", lattice_contents=sources)
    return lint_tao_schema(files.tao_init)


def _codes(contents: str, sources: dict[str, str] | None = None) -> set[LintCode]:
    return {lint.code for lint in _tao_lints(contents, sources)}


VALID_INIT = """\
&tao_start
  n_universes = 2
  plot_file = 'tao_plot.init'
/

&tao_params
  global%track_type = 'single'
  global%n_opti_cycles = 100
  bmad_com%radiation_damping_on = F
/
"""


def test_valid_tao_init_has_no_schema_lints():
    assert _tao_lints(VALID_INIT) == []


def test_type_mismatch_reported():
    contents = "&tao_start\n  n_universes = 2.5\n/\n"
    lints = _tao_lints(contents)
    assert [lint.code for lint in lints] == [LintCode.tao_type_mismatch]
    assert "n_universes" in lints[0].message


def test_unknown_field_reported():
    contents = "&tao_params\n  global%no_such_field = 1\n/\n"
    assert _codes(contents) == {LintCode.tao_unknown_field}


def test_index_out_of_bounds_reported():
    contents = "&tao_design_lattice\n  design_lattice(0)%file = 'a.bmad'\n/\n"
    assert _codes(contents) == {LintCode.tao_index_out_of_bounds}


def test_unknown_namelist_not_flagged():
    # A namelist absent from the schema is skipped, not reported.
    contents = "&tao_template_curve\n  anything = whatever\n/\n"
    assert _tao_lints(contents) == []


def test_unquoted_character_value_reported():
    # Character fields must be quoted strings; an unquoted value is flagged.
    contents = "&tao_start\n  plot_file = tao_plot.init\n/\n"
    assert _codes(contents) == {LintCode.tao_type_mismatch}


@pytest.mark.parametrize(
    "literal, expected",
    [
        ("'abc'", 3),
        ('"abcd"', 4),
        ("''", 0),
        # Doubled quote is one stored character.
        ("'it''s'", 4),
        ("3*'ab'", 2),  # repeat: the repeated value's length
        ("unquoted", None),  # not a string; type-checked, not length-checked
        ("42", None),
    ],
)
def test_string_length(literal, expected):
    from ..tao.schema import string_length

    assert string_length(literal) == expected


def test_string_too_long_reported():
    # init_name is character(16); a longer quoted value is truncated by Fortran.
    contents = "&tao_start\n  init_name = 'this_name_is_far_too_long'\n/\n"
    assert _codes(contents) == {LintCode.tao_string_too_long}


def test_string_within_length_not_reported():
    contents = "&tao_start\n  init_name = 'short'\n/\n"
    assert _tao_lints(contents) == []


def test_split_out_source_file_is_validated():
    contents = "&tao_start\n  var_file = 'vars.init'\n/\n"
    vars_init = "&tao_var\n  ix_min_var = 1.5\n/\n"
    assert _codes(contents, {"vars.init": vars_init}) == {LintCode.tao_type_mismatch}


# -- fix / format --------------------------------------------------------------


def _formatted(src: str, **kwargs) -> str:
    from nmlform import NamelistFile

    from ..tao.file import format_tao_namelist

    return format_tao_namelist(NamelistFile.parse(src), **kwargs)


def test_fix_quotes_unquoted_character_values():
    src = "&tao_start\n  plot_file = tao_plot.init\n  n_universes = 2\n/\n"
    out = _formatted(src)
    assert "plot_file = 'tao_plot.init'" in out
    assert "n_universes = 2" in out  # numeric untouched


def test_fix_leaves_quoted_and_typed_values_alone():
    src = "&tao_params\n  global%track_type = 'single'\n  bmad_com%radiation_damping_on = F\n/\n"
    assert _formatted(src) == src


def test_fix_types_false_renders_verbatim():
    src = "&tao_start\n  plot_file = tao_plot.init\n/\n"
    assert _formatted(src, fix_types=False) == src


def test_fix_skips_unknown_namelist():
    src = "&tao_template_curve\n  data_type = orbit.x\n/\n"
    assert _formatted(src) == src


def test_fix_quotes_character_array_values():
    # design_lattice(i)%file entries are character; each unquoted value is quoted.
    src = "&tao_design_lattice\n  design_lattice(1)%file = ring.bmad\n/\n"
    assert "design_lattice(1)%file = 'ring.bmad'" in _formatted(src)


def test_fix_namelist_file_fixes_all_groups():
    from nmlform import NamelistFile

    from ..tao.file import fix_tao_namelist

    nf = NamelistFile.parse(
        "&tao_start\n  plot_file = a.init\n/\n\n&tao_params\n  global%track_type = single\n/\n"
    )
    fix_tao_namelist(nf)
    out = nf.render()
    assert "plot_file = 'a.init'" in out
    assert "global%track_type = 'single'" in out


# -- integer enum (color) fixing ----------------------------------------------


def test_integer_enum_for_field():
    from ..tao.enums import TAO_COLORS, integer_enum_for_field

    assert integer_enum_for_field("prompt_color") is TAO_COLORS
    assert integer_enum_for_field("floor_plan_orbit_color") is TAO_COLORS
    assert integer_enum_for_field("COLOR") is TAO_COLORS
    assert integer_enum_for_field("track_type") is None


def test_fix_color_index_to_name():
    # global%prompt_color is a character color field; 2 is red in TAO_COLORS.
    src = "&tao_params\n  global%prompt_color = 2\n/\n"
    assert "global%prompt_color = 'red'" in _formatted(src)


def test_fix_color_bareword_name_quoted():
    src = "&tao_params\n  global%prompt_color = blue\n/\n"
    assert "global%prompt_color = 'blue'" in _formatted(src)


def test_fix_color_already_named_unchanged():
    src = "&tao_params\n  global%prompt_color = 'red'\n/\n"
    assert _formatted(src) == src


def test_fix_color_unknown_index_left_as_number():
    # An index outside TAO_COLORS is left numeric (Tao accepts the numeric form).
    src = "&tao_params\n  global%prompt_color = 999\n/\n"
    assert _formatted(src) == src


def test_fix_color_in_nested_struct_field():
    src = "&tao_template_graph\n  graph%floor_plan_orbit_color = 8\n/\n"
    assert "graph%floor_plan_orbit_color = 'orange'" in _formatted(src)
