from __future__ import annotations

import pathlib

import pytest

from ..output import default_options
from ..templating import cli_main, instantiate, interpolate, load_instances, write_instances

FILES = pathlib.Path(__file__).resolve().parent / "files" / "templating"
CX = FILES / "cx"


def _read(name: str) -> str:
    return (FILES / name).read_text()


def _format(src: str) -> str:
    """Reformat through latform with no transform, for byte-for-byte comparison."""
    return interpolate(src, options=default_options)


def test_cor_prefix_rename_matches_cookiecutter():
    """A prefix rename plus dropping ``type`` (via a regex value key) reproduces
    the expected cor output."""
    template = _read("cx.cor.bmad")
    expected = _format(_read("c1.cor.bmad"))
    actual = interpolate(
        template,
        values={"/CX_[XY]CR/": {"type": None}},
        renames={r"CX(_.*|$)": r"C1\1"},
        options=default_options,
    )
    assert actual == expected


def test_apply_values_removes_attribute():
    src = "Q1: quadrupole, L=0.3, k1=1.0"
    out = interpolate(src, values={"Q1": {"k1": None}})
    assert "k1" not in out.lower()
    assert "l=0.3" in out.lower().replace(" ", "")


def test_apply_values_regex_key_matches_multiple():
    src = "\n".join(["A_BPM1: monitor", "A_BPM2: monitor", "A_QUAD: quadrupole"])
    out = interpolate(src, values={"/A_BPM/": {"type": "BPM_TYPE"}})
    assert out.lower().count("type=bpm_type") == 2  # both BPMs, not the quad


def test_apply_values_overrides_attribute():
    src = "Q1: quadrupole, L=0.3, k1=0.0"
    out = interpolate(src, values={"Q1": {"k1": 1.523}})
    assert "k1=1.523" in out.lower().replace(" ", "")


def test_apply_values_appends_missing_attribute():
    src = "Q1: quadrupole, L=0.3"
    out = interpolate(src, values={"Q1": {"k1": 1.523}})
    assert "k1" in out.lower()


def test_apply_values_overrides_constant():
    src = "CX_LINE_ROT = 0"
    out = interpolate(src, values={"CX_LINE_ROT": "pi/2"})
    assert "pi/2" in out


def test_apply_values_expression_value():
    src = "Q1: quadrupole, L=0.0"
    out = interpolate(src, values={"Q1": {"L": "74e-3/2"}})
    assert "74e-3/2" in out.replace(" ", "")


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


def test_cli_interpolate_to_stdout(tmp_path, capsys):
    (tmp_path / "t.bmad").write_text("CX_Q: quadrupole, k1=0.0\n")
    cli_main(["interpolate", str(tmp_path / "t.bmad"), "--rename", r"CX(_.*|$)", r"C1\1"])
    out = capsys.readouterr().out
    assert "C1_Q" in out and "CX_Q" not in out


def test_cli_interpolate_with_values_file(tmp_path, capsys):
    (tmp_path / "t.bmad").write_text("Q1: quadrupole, k1=0.0\n")
    (tmp_path / "v.yaml").write_text("Q1: {k1: 1.5}\n")
    cli_main(["interpolate", str(tmp_path / "t.bmad"), "--values", str(tmp_path / "v.yaml")])
    assert "k1=1.5" in capsys.readouterr().out.replace(" ", "")


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
    cli_main(["instantiate", str(tmp_path / "instances.yaml"), "-d", str(tmp_path / "out")])
    assert (tmp_path / "out" / "c1" / "c1.bmad").is_file()
    assert "C1_Q" in (tmp_path / "out" / "c1" / "c1.bmad").read_text()


def test_cli_instantiate_dry_run_writes_nothing(tmp_path, capsys):
    (tmp_path / "cx.bmad").write_text("CX_Q: quadrupole\n")
    (tmp_path / "instances.yaml").write_text(
        'template:\n  - input: cx.bmad\n    output: "{instance}.bmad"\ninstances:\n  c1: {}\n'
    )
    cli_main(
        ["instantiate", str(tmp_path / "instances.yaml"), "-d", str(tmp_path / "out"), "--dry-run"]
    )
    out = capsys.readouterr().out
    assert "would write" in out
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- #
# Structured renames: prefix / suffix / regex / parts (+ shortcut)
# --------------------------------------------------------------------------- #

_NAMES_SRC = "\n".join(
    [
        "CX_BEN0: sbend",
        "O_CX_BEN: overlay = {CX_BEN0[g]: g}, var={g}",
        "CX.COL00: ecollimator",
        "CXFOO: marker",
        "CX: line=(CX_BEN0)",
    ]
)


def test_prefix_leading_only():
    out = interpolate(_NAMES_SRC, prefix={"CX": "C1"}).lower()
    assert "c1_ben0" in out  # leading segment renamed (def + ref)
    assert "c1.col00" in out  # dotted boundary
    assert "c1:" in out  # bare line name
    assert "o_cx_ben" in out  # embedded: NOT leading -> untouched
    assert "cxfoo" in out  # not a bounded prefix -> untouched
    assert "cx_ben0" not in out


def test_prefix_second_entry_catches_embedded():
    out = interpolate(_NAMES_SRC, prefix={"CX": "C1", "O_CX": "O_C1"}).lower()
    assert "o_c1_ben" in out
    assert "o_cx_ben" not in out


def test_parts_renames_any_segment():
    out = interpolate(_NAMES_SRC, parts=[{"delimiters": "._", "from": "CX", "to": "C1"}]).lower()
    assert "c1_ben0" in out
    assert "o_c1_ben" in out  # embedded segment renamed
    assert "c1.col00" in out
    assert "c1:" in out
    assert "cxfoo" in out  # whole-segment match only
    assert "cx_ben0" not in out


def test_parts_delimiters_as_list():
    out = interpolate(
        _NAMES_SRC, parts=[{"delimiters": [".", "_"], "from": "CX", "to": "C1"}]
    ).lower()
    assert "o_c1_ben" in out and "c1.col00" in out


_SUFFIX_SRC = "\n".join(
    ["A_XCR: marker", "A.XCR: marker", "XCR: marker", "A_XCRB: marker", "FOOXCR: marker"]
)


def test_suffix_bare_bounded():
    out = interpolate(_SUFFIX_SRC, suffix={"XCR": "HCOR"}).lower()
    assert "a_hcor:" in out
    assert "a.hcor:" in out
    assert "hcor:" in out  # bare XCR at start-of-name
    assert "a_xcrb:" in out  # XCR not at end -> untouched
    assert "fooxcr:" in out  # no preceding boundary -> untouched
    assert "a_xcr:" not in out


def test_suffix_with_delimiter_in_from():
    out = interpolate("Q_XCR: marker\nFOO_XCRB: marker", suffix={"_XCR": "_HCOR"}).lower()
    assert "q_hcor:" in out
    assert "foo_xcrb:" in out


def test_structured_renames_via_dict():
    out = interpolate(_NAMES_SRC, renames={"prefix": {"CX": "C1"}}).lower()
    assert "c1_ben0" in out and "o_cx_ben" in out


def test_structured_regex_equals_flat_regex():
    flat = interpolate(_NAMES_SRC, renames={r"CX([_.].*|$)": r"C1\1"})
    structured = interpolate(_NAMES_SRC, renames={"regex": {r"CX([_.].*|$)": r"C1\1"}})
    assert flat == structured


def test_shortcut_still_literal_or_regex():
    # no * + ? -> literal exact-name match
    out = interpolate("CX: marker\nCX_A: marker", renames={"CX": "C1"}).lower()
    assert "c1: marker" in out
    assert "cx_a: marker" in out  # not an exact match


def test_explicit_regex_beats_prefix():
    out = interpolate(
        "CX_A: marker", renames={"regex": {"CX_A": "EXPLICIT"}, "prefix": {"CX": "C1"}}
    ).lower()
    assert "explicit" in out
    assert "c1_a" not in out


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


def test_cli_prefix_and_parts(tmp_path, capsys):
    (tmp_path / "t.bmad").write_text("CX_Q: quadrupole\nO_CX_M: marker\n")
    cli_main(["interpolate", str(tmp_path / "t.bmad"), "--prefix", "CX", "C1"])
    out = capsys.readouterr().out.lower()
    assert "c1_q" in out and "o_cx_m" in out  # prefix leaves embedded

    cli_main(["interpolate", str(tmp_path / "t.bmad"), "--parts", "._", "CX", "C1"])
    out = capsys.readouterr().out.lower()
    assert "c1_q" in out and "o_c1_m" in out  # parts renames embedded


def test_cli_suffix(tmp_path, capsys):
    (tmp_path / "t.bmad").write_text("Q_XCR: marker\n")
    cli_main(["interpolate", str(tmp_path / "t.bmad"), "--suffix", "_XCR", "_HCOR"])
    assert "q_hcor" in capsys.readouterr().out.lower()


def test_parts_as_dict_uses_default_delimiters():
    out = interpolate(_NAMES_SRC, renames={"parts": {"CX": "C1"}}).lower()
    assert "c1_ben0" in out and "o_c1_ben" in out and "c1.col00" in out
    assert "cxfoo" in out  # still whole-segment only


def test_top_level_delimiters_restrict_parts():
    # only "_" is a delimiter -> "." is not, so CX.COL00 is one segment (untouched)
    out = interpolate(_NAMES_SRC, renames={"parts": {"CX": "C1"}}, delimiters="_").lower()
    assert "c1_ben0" in out  # "_" split still works
    assert "cx.col00" in out  # "." not a delimiter -> whole segment "CX.COL00" != "CX"


def test_top_level_delimiters_apply_to_prefix():
    # with only "_" as delimiter, a dotted prefix boundary no longer matches
    out = interpolate(_NAMES_SRC, renames={"prefix": {"CX": "C1"}}, delimiters="_").lower()
    assert "c1_ben0" in out
    assert "cx.col00" in out  # "." no longer a boundary -> untouched


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


def test_cli_delimiters_option(tmp_path, capsys):
    (tmp_path / "t.bmad").write_text("CX_Q: quadrupole\nCX.COL: ecollimator\n")
    cli_main(["interpolate", str(tmp_path / "t.bmad"), "--prefix", "CX", "C1", "--delimiters", "_"])
    out = capsys.readouterr().out.lower()
    assert "c1_q" in out
    assert "cx.col" in out  # "." not a delimiter with --delimiters _
