from __future__ import annotations

import pathlib

import pytest

from ..apply import interpolate
from ..output import default_options
from ..template import cli_main, instantiate, load_instances, write_instances

FILES = pathlib.Path(__file__).resolve().parent / "files" / "templating"
CX = FILES / "cx"


def _format(src: str) -> str:
    """Reformat through latform with no transform, for byte-for-byte comparison."""
    return interpolate(src, options=default_options)


_CX_FILES = ["bmad", "cor.bmad", "bpm.bmad", "col.bmad", "lat.bmad"]
_CX_INSTANCES = ["c1", "c2", "c3", "c4"]


@pytest.fixture(scope="module")
def cx_generated() -> dict[str, dict[str, str]]:
    spec = load_instances(CX / "template" / "instances.yaml")
    return instantiate(spec, base_dir=CX / "template", options=default_options)


@pytest.mark.parametrize("instance", _CX_INSTANCES)
@pytest.mark.parametrize("suffix", _CX_FILES)
def test_cx_instantiate_byte_for_byte(cx_generated, instance, suffix):
    """The reworked cx template reproduces the cookiecutter output, file for file."""
    out_name = f"{instance}.{suffix}"
    expected = _format((CX / "expected" / out_name).read_text())
    actual = cx_generated[instance][out_name]
    assert actual == expected


def test_template_set_resolves_cross_file_refs_and_rewrites_calls(tmp_path):
    """Files in the transform set are annotated together; a renamed file in a
    ``call`` is rewritten."""
    (tmp_path / "a.bmad").write_text("TPL_K = 0.5\n")
    (tmp_path / "b.bmad").write_text("call, file=a.bmad\nTPL_Q: quadrupole, k1=TPL_K\n")
    spec = {
        "template": [
            {"input": "a.bmad", "output": "{instance}_a.bmad"},
            {"input": "b.bmad", "output": "{instance}_b.bmad"},
        ],
        "renames": {r"TPL(_.*|$)": r"{instance:upper}\1"},
        "instances": {"m1": {}},
    }
    out = instantiate(spec, base_dir=tmp_path, options=default_options)["m1"]["m1_b.bmad"]
    assert "k1=M1_K" in out.replace(" ", "")  # cross-file value ref resolved + renamed
    assert "call, file=m1_a.bmad" in out  # renamed file rewritten in the call


def test_context_file_resolves_reference_but_is_not_written(tmp_path):
    """A non-templated ``call``ed file, listed under ``context``, is loaded for
    resolution only: references resolve, but it is not output and calls to it
    are left untouched."""
    (tmp_path / "entry.bmad").write_text(
        "call, file=shared.bmad\nTPL_Q: quadrupole, k1=TPL_KSHARED\n"
    )
    (tmp_path / "shared.bmad").write_text("TPL_KSHARED = 0.5\n")
    spec = {
        "template": [{"input": "entry.bmad", "output": "{instance}_entry.bmad"}],
        "context": ["shared.bmad"],
        "renames": {r"TPL(_.*|$)": r"{instance:upper}\1"},
        "instances": {"m1": {}},
    }
    res = instantiate(spec, base_dir=tmp_path, options=default_options)["m1"]
    assert set(res) == {"m1_entry.bmad"}  # shared.bmad not written
    out = res["m1_entry.bmad"]
    assert "k1=M1_KSHARED" in out.replace(" ", "")  # resolved across files -> renamed
    assert "call, file=shared.bmad" in out  # shared call left untouched


def test_call_target_basename_rewritten_dir_preserved(tmp_path):
    (tmp_path / "top.bmad").write_text("call, file=sub/leaf.bmad\nTPL_M: marker\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "leaf.bmad").write_text("TPL_L: marker\n")
    spec = {
        "template": [
            {"input": "top.bmad", "output": "{instance}.bmad"},
            {"input": "sub/leaf.bmad", "output": "sub/{instance}_leaf.bmad"},
        ],
        "renames": {r"TPL(_.*|$)": r"{instance:upper}\1"},
        "instances": {"m1": {}},
    }
    out = instantiate(spec, base_dir=tmp_path, options=default_options)["m1"]["m1.bmad"]
    assert "call, file=sub/m1_leaf.bmad" in out  # basename swapped, dir preserved


def test_call_target_directory_component_rewritten(tmp_path):
    """A prefixed directory in a call path is rewritten: cx/foo/bar.bmad -> c1/foo/bar.bmad."""
    (tmp_path / "cx").mkdir()
    (tmp_path / "cx" / "foo").mkdir()
    # top-level entry calls into the prefixed directory tree
    (tmp_path / "entry.bmad").write_text("call, file=cx/foo/bar.bmad\n")
    (tmp_path / "cx" / "foo" / "bar.bmad").write_text("CX_M: marker\n")
    spec = {
        "template": [
            {"input": "entry.bmad", "output": "{instance}.bmad"},
            {"input": "cx/foo/bar.bmad", "output": "{instance}/foo/bar.bmad"},
        ],
        "renames": {r"CX(_.*|$)": r"{instance:upper}\1"},
        "instances": {"c1": {}},
    }
    res = instantiate(spec, base_dir=tmp_path, options=default_options)["c1"]
    assert "c1/foo/bar.bmad" in res  # output path directory rewritten
    assert "call, file=c1/foo/bar.bmad" in res["c1.bmad"]  # call path directory rewritten


def test_inline_call_target_rewritten(tmp_path):
    """Inline ``call::`` paths (bare and attribute-valued) pointing at
    transform-set files are rewritten; other targets are left untouched."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "top.bmad").write_text(
        "c: crystal, call::sub/surface.bmad, h_misalign = call::sub/surface.bmad\n"
        "d: crystal, call::external.bmad\n"
    )
    (tmp_path / "sub" / "surface.bmad").write_text("qq: quadrupole, l = 0.5\n")
    spec = {
        "template": [
            {"input": "top.bmad", "output": "{instance}.bmad"},
            {"input": "sub/surface.bmad", "output": "{instance}_surface.bmad"},
        ],
        "instances": {"m1": {}},
    }
    out = instantiate(spec, base_dir=tmp_path, options=default_options)["m1"]["m1.bmad"]
    squashed = out.replace(" ", "")
    assert "C:crystal,call::m1_surface.bmad,h_misalign=call::m1_surface.bmad" in squashed
    assert "call::external.bmad" in squashed  # not in the transform set: untouched


def test_template_root_inputs_read_from_root(tmp_path):
    """With ``template_root``, template/context inputs are read relative to it
    while cross-file resolution and call rewriting work as usual."""
    root = tmp_path / "shared" / "templates"
    (root / "sub").mkdir(parents=True)
    (root / "top.bmad").write_text("call, file=sub/leaf.bmad\nTPL_M: marker, x_offset=TPL_X\n")
    (root / "sub" / "leaf.bmad").write_text("TPL_L: marker\n")
    (root / "ctx.bmad").write_text("TPL_X = 1\n")
    proj = tmp_path / "proj"
    proj.mkdir()
    spec = {
        "template_root": "../shared/templates",
        "template": [
            {"input": "top.bmad", "output": "{instance}.bmad"},
            {"input": "sub/leaf.bmad", "output": "{instance}_leaf.bmad"},
        ],
        "context": ["ctx.bmad"],
        "renames": {r"TPL(_.*|$)": r"{instance:upper}\1"},
        "instances": {"m1": {}},
    }
    res = instantiate(spec, base_dir=proj, options=default_options)["m1"]
    assert set(res) == {"m1.bmad", "m1_leaf.bmad"}
    out = res["m1.bmad"]
    assert "call, file=m1_leaf.bmad" in out
    assert "x_offset=M1_X" in out.replace(" ", "")  # context resolved + renamed


def test_template_root_tao_init_and_paths(tmp_path):
    """``tao_init`` inputs are read from ``template_root``; ``paths`` keys stay
    in template-root coordinates."""
    root = tmp_path / "templates"
    root.mkdir()
    (root / "cx.lat.bmad").write_text("CX_Q: quadrupole, k1=0.0\ncall, file=../common.bmad\n")
    (root / "tao.init").write_text(
        "&tao_design_lattice\n  design_lattice(1)%file = 'cx.lat.bmad'\n/\n"
    )
    spec = {
        "template_root": "templates",
        "template": [{"input": "cx.lat.bmad", "output": "{instance}.lat.bmad"}],
        "tao_init": {"input": "tao.init", "output": "{instance}/tao.init"},
        "paths": {"../common.bmad": "common_{instance}.bmad"},
        "instances": {"c1": {}},
    }
    res = instantiate(spec, base_dir=tmp_path, options=default_options)["c1"]
    assert "call, file=common_c1.bmad" in res["c1.lat.bmad"]
    assert "design_lattice(1)%file = '../c1.lat.bmad'" in res["c1/tao.init"]


def test_paths_replacement_outside_transform_set(tmp_path):
    """An explicit ``paths`` entry rewrites references to files that are not
    part of the transform set, for both call forms."""
    (tmp_path / "top.bmad").write_text(
        "call, file=../foo.bmad\nc: crystal, call::../foo.bmad\ncall, file=../keep.bmad\n"
    )
    spec = {
        "template": [{"input": "top.bmad", "output": "{instance}.bmad"}],
        "paths": {"../foo.bmad": "../bar.bmad"},
        "instances": {"m1": {}},
    }
    out = instantiate(spec, base_dir=tmp_path, options=default_options)["m1"]["m1.bmad"]
    assert "call, file=../bar.bmad" in out
    assert "call::../bar.bmad" in out.replace(" ", "")
    assert "call, file=../keep.bmad" in out  # unmapped reference untouched


def test_paths_per_instance_override_and_interpolation(tmp_path):
    """Global ``paths`` interpolate ``{instance}``; a per-instance block wins,
    and replacements are made relative to each output file's directory."""
    (tmp_path / "top.bmad").write_text("call, file=../settings.bmad\n")
    spec = {
        "template": [{"input": "top.bmad", "output": "{instance}/main.bmad"}],
        "paths": {"../settings.bmad": "settings_{instance}.bmad"},
        "instances": {
            "m1": {},
            "m2": {"paths": {"../settings.bmad": "special.bmad"}},
        },
    }
    res = instantiate(spec, base_dir=tmp_path, options=default_options)
    # values are relative to the output base dir; main.bmad lives one level down
    assert "call, file=../settings_m1.bmad" in res["m1"]["m1/main.bmad"]
    assert "call, file=../special.bmad" in res["m2"]["m2/main.bmad"]


def test_paths_as_written_reference_match(tmp_path):
    """A ``paths`` key also matches a reference exactly as written in the
    calling file; the replacement is inserted verbatim and wins over the
    transform set's automatic rewrite."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "top.lat.bmad").write_text("call, file=dfq.bmad\ncall, file=../y1.bmad\n")
    (sub / "dfq.bmad").write_text("Q: marker\n")
    spec = {
        "template": [
            {"input": "sub/top.lat.bmad", "output": "sub_{instance}/top.lat.bmad"},
            # dfq.bmad is in the transform set: without the explicit entry the
            # reference would be auto-rewritten to the (same-named) output.
            {"input": "sub/dfq.bmad", "output": "sub_{instance}/dfq.bmad"},
        ],
        "paths": {
            "dfq.bmad": "dfrepl.bmad",
            "../y1.bmad": "../{instance}.bmad",
        },
        "instances": {"m1": {}},
    }
    out = instantiate(spec, base_dir=tmp_path, options=default_options)["m1"]["sub_m1/top.lat.bmad"]
    assert "call, file=dfrepl.bmad" in out
    assert "call, file=../m1.bmad" in out


def test_paths_env_var_replacement_kept_verbatim(tmp_path):
    """A replacement containing ``$VAR`` (or an absolute path) is not made
    relative to the output file's directory."""
    (tmp_path / "top.bmad").write_text("call, file=old/settings.bmad\n")
    spec = {
        "template": [{"input": "top.bmad", "output": "{instance}/main.bmad"}],
        "paths": {"old/settings.bmad": "$LATTICE_ROOT/settings.bmad"},
        "instances": {"m1": {}},
    }
    out = instantiate(spec, base_dir=tmp_path, options=default_options)["m1"]["m1/main.bmad"]
    assert "call, file=$LATTICE_ROOT/settings.bmad" in out


def test_paths_replacement_in_tao_init(tmp_path):
    """``paths`` entries also apply to tao.init design_lattice files."""
    (tmp_path / "cx.lat.bmad").write_text("CX_Q: quadrupole, k1=0.0\n")
    (tmp_path / "tao.init").write_text(
        "&tao_design_lattice\n"
        "  design_lattice(1)%file = 'cx.lat.bmad'\n"
        "  design_lattice(2)%file = '../common.bmad'\n"
        "/\n"
    )
    spec = {
        "template": [{"input": "cx.lat.bmad", "output": "{instance}.lat.bmad"}],
        "tao_init": {"input": "tao.init", "output": "{instance}/tao.init"},
        "paths": {"../common.bmad": "../common_{instance}.bmad"},
        "instances": {"c1": {}},
    }
    res = instantiate(spec, base_dir=tmp_path, options=default_options)
    c1 = res["c1"]["c1/tao.init"]
    assert "design_lattice(1)%file = '../c1.lat.bmad'" in c1
    assert "design_lattice(2)%file = '../../common_c1.bmad'" in c1


def test_instantiate_rewrites_tao_init(tmp_path):
    """A tao_init spec rewrites design_lattice files and adds/updates namelists."""
    (tmp_path / "cx.lat.bmad").write_text("CX_Q: quadrupole, k1=0.0\ncl: line=(CX_Q)\nuse, cl\n")
    (tmp_path / "tao.init").write_text(
        "&tao_design_lattice\n"
        "  design_lattice(1)%file = 'cx.lat.bmad'\n"
        "/\n\n"
        "&tao_params\n"
        "  global%n_opti_cycles = 100\n"
        "/\n"
    )
    spec = {
        "template": [{"input": "cx.lat.bmad", "output": "{instance}.lat.bmad"}],
        "renames": {r"CX(_.*|$)": r"{instance:upper}\1"},
        "tao_init": {"input": "tao.init", "output": "{instance}/tao.init"},
        "instances": {
            "c1": {},
            "c2": {
                "tao_init": {
                    "namelists": {
                        "tao_params": {"global%n_opti_cycles": "50"},
                        "tao_beam_init": {"beam_init%n_particle": "5000"},
                    }
                }
            },
        },
    }
    res = instantiate(spec, base_dir=tmp_path, options=default_options)

    c1 = res["c1"]["c1/tao.init"]
    # design_lattice rewritten to the generated lattice, relative to the output dir
    assert "design_lattice(1)%file = '../c1.lat.bmad'" in c1
    assert "global%n_opti_cycles = 100" in c1  # untouched for c1

    c2 = res["c2"]["c2/tao.init"]
    assert "design_lattice(1)%file = '../c2.lat.bmad'" in c2
    assert "global%n_opti_cycles = 50" in c2  # updated
    assert "&tao_beam_init" in c2  # section added
    assert "beam_init%n_particle = 5000" in c2


def test_instantiate_renames_elements_in_tao_init(tmp_path):
    """Lattice element renames propagate into tao.init element references."""
    (tmp_path / "cx.lat.bmad").write_text("CX_Q: quadrupole, k1=0.0\ncl: line=(CX_Q)\nuse, cl\n")
    (tmp_path / "tao.init").write_text(
        "&tao_design_lattice\n"
        "  design_lattice(1)%file = 'cx.lat.bmad'\n"
        "/\n\n"
        "&tao_d1_data\n"
        "  datum(1)%ele_name = 'CX_Q'\n"
        "  datum(2)%data_type = 'expression: lat::orbit.x[CX_Q]|model'\n"
        "/\n\n"
        "&tao_var\n"
        "  var(1)%ele_name = 'cx_q'\n"
        "/\n"
    )
    spec = {
        "template": [{"input": "cx.lat.bmad", "output": "{instance}.lat.bmad"}],
        "renames": {r"CX(_.*|$)": r"{instance:upper}\1"},
        "tao_init": {"input": "tao.init", "output": "{instance}/tao.init"},
        "instances": {"c1": {}},
    }
    out = instantiate(spec, base_dir=tmp_path, options=default_options)["c1"]["c1/tao.init"]
    flat = out.replace(" ", "")
    assert "datum(1)%ele_name='C1_Q'" in flat
    assert "lat::orbit.x[C1_Q]|model" in out
    assert "var(1)%ele_name='C1_Q'" in flat


def test_instantiate_renames_tao_init_only_elements(tmp_path):
    """Names only the tao.init references (not defined in the template lattices)
    are still renamed by the instance's rename rules."""
    (tmp_path / "cx.lat.bmad").write_text("CX_Q: quadrupole, k1=0.0\ncl: line=(CX_Q)\nuse, cl\n")
    (tmp_path / "tao.init").write_text(
        "&tao_var\n"
        "  var(1)%ele_name = 'cx.pat'\n"
        "  var(2)%ele_name = 'beginning'\n"
        "/\n\n"
        "&tao_d1_data\n"
        "  datum(1) = 'orbit.x' '' '' 'CX_EXTERNAL' 'target' 0 1e1\n"
        "/\n"
    )
    spec = {
        "template": [{"input": "cx.lat.bmad", "output": "{instance}.lat.bmad"}],
        "renames": {"parts": {"CX": "{instance:upper}"}},
        "tao_init": {"input": "tao.init", "output": "{instance}/tao.init"},
        "instances": {"c1": {}},
    }
    out = instantiate(spec, base_dir=tmp_path, options=default_options)["c1"]["c1/tao.init"]
    assert "'C1.pat'" in out
    assert "'C1_EXTERNAL'" in out
    assert "'beginning'" in out  # pseudo-element never renamed


def _multi_tao_init_spec(tmp_path) -> dict:
    (tmp_path / "cx.lat.bmad").write_text("CX_Q: quadrupole, k1=0.0\ncl: line=(CX_Q)\nuse, cl\n")
    (tmp_path / "tao.init").write_text(
        "&tao_design_lattice\n"
        "  design_lattice(1)%file = 'cx.lat.bmad'\n"
        "/\n\n"
        "&tao_params\n"
        "  global%n_opti_cycles = 100\n"
        "/\n"
    )
    (tmp_path / "tao_smooth.init").write_text(
        "&tao_design_lattice\n  design_lattice(1)%file = 'cx.lat.bmad'\n/\n"
    )
    return {
        "template": [{"input": "cx.lat.bmad", "output": "{instance}.lat.bmad"}],
        "tao_init": [
            {"input": "tao.init", "output": "{instance}/tao.init"},
            {"input": "tao_smooth.init", "output": "{instance}/tao_smooth.init"},
        ],
        "instances": {"c1": {}},
    }


def test_instantiate_multiple_tao_inits(tmp_path):
    """The tao_init list form renders each entry, rewriting lattices in all of them."""
    spec = _multi_tao_init_spec(tmp_path)
    res = instantiate(spec, base_dir=tmp_path, options=default_options)["c1"]
    assert set(res) == {"c1.lat.bmad", "c1/tao.init", "c1/tao_smooth.init"}
    assert "design_lattice(1)%file = '../c1.lat.bmad'" in res["c1/tao.init"]
    assert "design_lattice(1)%file = '../c1.lat.bmad'" in res["c1/tao_smooth.init"]


def test_instantiate_multiple_tao_inits_override_keyed_by_input(tmp_path):
    """List-form per-instance overrides target one entry by its input path."""
    spec = _multi_tao_init_spec(tmp_path)
    spec["instances"]["c1"] = {
        "tao_init": {
            "tao.init": {"namelists": {"tao_params": {"global%n_opti_cycles": "50"}}},
        }
    }
    res = instantiate(spec, base_dir=tmp_path, options=default_options)["c1"]
    assert "global%n_opti_cycles = 50" in res["c1/tao.init"]
    assert "tao_params" not in res["c1/tao_smooth.init"]  # other entry untouched


def test_instantiate_multiple_tao_inits_unknown_override_key_raises(tmp_path):
    spec = _multi_tao_init_spec(tmp_path)
    spec["instances"]["c1"] = {"tao_init": {"tao_typo.init": {"namelists": {}}}}
    with pytest.raises(ValueError, match="tao_typo.init"):
        instantiate(spec, base_dir=tmp_path, options=default_options)


def test_instantiate_tao_init_duplicate_input_raises(tmp_path):
    spec = _multi_tao_init_spec(tmp_path)
    spec["tao_init"][1] = {"input": "tao.init", "output": "{instance}/other.init"}
    with pytest.raises(ValueError, match="duplicate tao_init input"):
        instantiate(spec, base_dir=tmp_path, options=default_options)


def test_instantiate_tao_init_output_collision_raises(tmp_path):
    spec = _multi_tao_init_spec(tmp_path)
    spec["tao_init"][1]["output"] = "{instance}/tao.init"
    with pytest.raises(ValueError, match="collision"):
        instantiate(spec, base_dir=tmp_path, options=default_options)


def _messy_tao_init_spec(tmp_path) -> dict:
    (tmp_path / "cx.lat.bmad").write_text("CX_Q: quadrupole, k1=0.0\ncl: line=(CX_Q)\nuse, cl\n")
    (tmp_path / "tao.init").write_text(
        "&tao_design_lattice\n"
        "      design_lattice(1)%file = 'cx.lat.bmad'\n"
        "/\n\n\n"
        "&tao_params\n"
        "        global%n_opti_cycles = 100\n"
        "/\n"
    )
    return {
        "template": [{"input": "cx.lat.bmad", "output": "{instance}.lat.bmad"}],
        "tao_init": {"input": "tao.init", "output": "{instance}/tao.init"},
        "instances": {"c1": {}},
    }


def test_instantiate_formats_tao_init_by_default(tmp_path):
    spec = _messy_tao_init_spec(tmp_path)
    out = instantiate(spec, base_dir=tmp_path, options=default_options)["c1"]["c1/tao.init"]
    assert "\n  design_lattice(1)%file = '../c1.lat.bmad'\n" in out  # re-indented to 2
    assert "\n  global%n_opti_cycles = 100\n" in out
    assert "/\n\n&tao_params" in out  # collapsed to a single blank line


def test_instantiate_tao_init_format_can_be_disabled(tmp_path):
    spec = _messy_tao_init_spec(tmp_path)
    out = instantiate(spec, base_dir=tmp_path, options=default_options, format_namelist=False)[
        "c1"
    ]["c1/tao.init"]
    assert "\n      design_lattice(1)%file = '../c1.lat.bmad'\n" in out  # 6-space indent preserved
    assert "/\n\n\n&tao_params" in out  # source blank lines preserved


def test_write_instances_creates_parent_directories(tmp_path):
    """Output paths with not-yet-existing directories are created on write."""
    (tmp_path / "cx.bmad").write_text("CX_Q: quadrupole\n")
    spec = {
        "template": [{"input": "cx.bmad", "output": "{instance}/{instance}.bmad"}],
        "renames": {r"CX(_.*|$)": r"{instance:upper}\1"},
        "instances": {"c1": {}, "c2": {}},
    }
    results = instantiate(spec, base_dir=tmp_path, options=default_options)
    assert set(results["c1"]) == {"c1/c1.bmad"}  # subdir preserved in output path

    out_dir = tmp_path / "out"
    written = write_instances(results, out_dir)

    assert (out_dir / "c1" / "c1.bmad").is_file()
    assert (out_dir / "c2" / "c2.bmad").is_file()
    assert "C1_Q" in (out_dir / "c1" / "c1.bmad").read_text()
    assert len(written) == 2


def _header_spec(tmp_path) -> dict:
    (tmp_path / "cx.lat.bmad").write_text("CX_Q: quadrupole, k1=0.0\ncl: line=(CX_Q)\nuse, cl\n")
    (tmp_path / "tao.init").write_text(
        "&tao_design_lattice\n  design_lattice(1)%file = 'cx.lat.bmad'\n/\n"
    )
    return {
        "template": [{"input": "cx.lat.bmad", "output": "{instance}.lat.bmad"}],
        "tao_init": {"input": "tao.init", "output": "{instance}/tao.init"},
        "instances": {"c1": {}},
    }


def test_default_header_on_generated_files(tmp_path):
    """All generated files start with the default header, with the source and
    instances file filled in, followed by a blank line."""
    spec = _header_spec(tmp_path)
    res = instantiate(
        spec, base_dir=tmp_path, options=default_options, instances_path="instances.yaml"
    )["c1"]
    lat_header = (
        "!** Generated by latform-template from cx.lat.bmad (instances.yaml); do not edit. **"
    )
    tao_header = "!** Generated by latform-template from tao.init (instances.yaml); do not edit. **"
    assert res["c1.lat.bmad"].startswith(lat_header + "\n\n")
    assert res["c1/tao.init"].startswith(tao_header + "\n\n")


def test_custom_header_interpolates_placeholders(tmp_path):
    spec = _header_spec(tmp_path)
    spec["header"] = "! {instance:upper} from {source} via {instances}"
    res = instantiate(
        spec, base_dir=tmp_path, options=default_options, instances_path="sub/instances.yaml"
    )["c1"]
    assert res["c1.lat.bmad"].startswith("! C1 from cx.lat.bmad via sub/instances.yaml\n\n")
    assert res["c1/tao.init"].startswith("! C1 from tao.init via sub/instances.yaml\n\n")


@pytest.mark.parametrize("opt_out", [None, ""])
def test_header_opt_out(tmp_path, opt_out):
    spec = _header_spec(tmp_path)
    spec["header"] = opt_out
    res = instantiate(spec, base_dir=tmp_path, options=default_options)["c1"]
    assert res["c1.lat.bmad"].startswith("CX_Q")
    assert res["c1/tao.init"].startswith("&tao_design_lattice")


def test_cli_instantiate_writes_files(tmp_path, capsys):
    (tmp_path / "cx.bmad").write_text("CX_Q: quadrupole\n")
    (tmp_path / "instances.yaml").write_text(
        "template:\n"
        "  - input: cx.bmad\n"
        '    output: "{instance}/{instance}.bmad"\n'
        "renames:\n"
        '  "CX(_.*|$)": "{instance:upper}\\\\1"\n'
        "instances:\n"
        "  c1: {}\n"
    )
    cli_main([str(tmp_path / "instances.yaml"), "-d", str(tmp_path / "out")])
    assert (tmp_path / "out" / "c1" / "c1.bmad").is_file()
    assert "C1_Q" in (tmp_path / "out" / "c1" / "c1.bmad").read_text()


def test_cli_config_format_defaults(tmp_path, capsys):
    """``[format]`` settings from a latform config supply the generation
    defaults; the instances file's own ``format:`` section still wins."""
    (tmp_path / "cx.bmad").write_text("cx_q: quadrupole, k1=0.0\n")
    (tmp_path / "latform.toml").write_text('[format]\nname-case = "same"\n')
    (tmp_path / "instances.yaml").write_text(
        'template:\n  - input: cx.bmad\n    output: "{instance}.bmad"\ninstances:\n  c1: {}\n'
    )
    args = ["-d", str(tmp_path / "out"), "--config", str(tmp_path / "latform.toml")]
    cli_main([str(tmp_path / "instances.yaml"), *args])
    assert "cx_q: quadrupole" in (tmp_path / "out" / "c1.bmad").read_text()

    (tmp_path / "instances.yaml").write_text(
        "template:\n"
        "  - input: cx.bmad\n"
        '    output: "{instance}.bmad"\n'
        "format:\n"
        "  name-case: upper\n"
        "instances:\n"
        "  c1: {}\n"
    )
    cli_main([str(tmp_path / "instances.yaml"), *args])
    assert "CX_Q: quadrupole" in (tmp_path / "out" / "c1.bmad").read_text()


def test_cli_instantiate_dry_run_writes_nothing(tmp_path, capsys):
    (tmp_path / "cx.bmad").write_text("CX_Q: quadrupole\n")
    (tmp_path / "instances.yaml").write_text(
        'template:\n  - input: cx.bmad\n    output: "{instance}.bmad"\ninstances:\n  c1: {}\n'
    )
    cli_main([str(tmp_path / "instances.yaml"), "-d", str(tmp_path / "out"), "--dry-run"])
    out = capsys.readouterr().out
    assert "would write" in out
    assert not (tmp_path / "out").exists()


def test_global_and_instance_prefix_merge(tmp_path):
    (tmp_path / "a.bmad").write_text("CX_Q: quadrupole\n")
    spec = {
        "template": [{"input": "a.bmad", "output": "{instance}.bmad"}],
        "renames": {"prefix": {"CX": "G"}},  # global
        "instances": {
            "m1": {},  # uses global
            "m2": {"renames": {"prefix": {"CX": "X2"}}},  # overrides global
        },
    }
    res = instantiate(spec, base_dir=tmp_path, options=default_options)
    assert "G_Q" in res["m1"]["m1.bmad"]
    assert "X2_Q" in res["m2"]["m2.bmad"]
    assert "G_Q" not in res["m2"]["m2.bmad"]


def test_instantiate_top_level_delimiters_and_parts_dict(tmp_path):
    (tmp_path / "a.bmad").write_text("CX_Q: quadrupole\nO_CX_M: marker\n")
    spec = {
        "template": [{"input": "a.bmad", "output": "{instance}.bmad"}],
        "delimiters": "._",
        "renames": {"parts": {"CX": "{instance:upper}"}},  # dict form, default delimiters
        "instances": {"c1": {}},
    }
    out = instantiate(spec, base_dir=tmp_path, options=default_options)["c1"]["c1.bmad"]
    assert "C1_Q" in out and "O_C1_M" in out
