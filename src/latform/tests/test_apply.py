from __future__ import annotations

import pathlib

import pytest

from ..apply import cli_main_apply, interpolate, interpolate_namelist
from ..output import default_options
from ..types import FormatOptions, NamelistFormatOptions

FILES = pathlib.Path(__file__).resolve().parent / "files" / "templating"


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


def test_cli_interpolate_to_stdout(tmp_path, capsys):
    (tmp_path / "t.bmad").write_text("CX_Q: quadrupole, k1=0.0\n")
    cli_main_apply([str(tmp_path / "t.bmad"), "--rename", r"CX(_.*|$)", r"C1\1"])
    out = capsys.readouterr().out
    assert "C1_Q" in out and "CX_Q" not in out


def test_cli_interpolate_with_values_file(tmp_path, capsys):
    (tmp_path / "t.bmad").write_text("Q1: quadrupole, k1=0.0\n")
    (tmp_path / "v.yaml").write_text("Q1: {k1: 1.5}\n")
    cli_main_apply([str(tmp_path / "t.bmad"), "--values", str(tmp_path / "v.yaml")])
    assert "k1=1.5" in capsys.readouterr().out.replace(" ", "")


def test_cli_interpolate_values_from_stdin(tmp_path, capsys, monkeypatch):
    import io

    (tmp_path / "t.bmad").write_text("Q1: quadrupole, k1=0.0\n")
    monkeypatch.setattr("sys.stdin", io.StringIO("Q1: {k1: 1.5}\n"))  # YAML/JSON on stdin
    cli_main_apply([str(tmp_path / "t.bmad"), "--values", "-"])
    assert "k1=1.5" in capsys.readouterr().out.replace(" ", "")


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


def test_cli_prefix_and_parts(tmp_path, capsys):
    (tmp_path / "t.bmad").write_text("CX_Q: quadrupole\nO_CX_M: marker\n")
    cli_main_apply([str(tmp_path / "t.bmad"), "--prefix", "CX", "C1"])
    out = capsys.readouterr().out.lower()
    assert "c1_q" in out and "o_cx_m" in out  # prefix leaves embedded

    cli_main_apply([str(tmp_path / "t.bmad"), "--parts", "._", "CX", "C1"])
    out = capsys.readouterr().out.lower()
    assert "c1_q" in out and "o_c1_m" in out  # parts renames embedded


def test_cli_suffix(tmp_path, capsys):
    (tmp_path / "t.bmad").write_text("Q_XCR: marker\n")
    cli_main_apply([str(tmp_path / "t.bmad"), "--suffix", "_XCR", "_HCOR"])
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


def test_cli_delimiters_option(tmp_path, capsys):
    (tmp_path / "t.bmad").write_text("CX_Q: quadrupole\nCX.COL: ecollimator\n")
    cli_main_apply([str(tmp_path / "t.bmad"), "--prefix", "CX", "C1", "--delimiters", "_"])
    out = capsys.readouterr().out.lower()
    assert "c1_q" in out
    assert "cx.col" in out  # "." not a delimiter with --delimiters _


# --------------------------------------------------------------------------- #
# Namelist interpolation (*.init / *.nml) and --in-place
# --------------------------------------------------------------------------- #

_NML_SRC = (
    "&tao_design_lattice\n"
    "  design_lattice(1)%file = 'lat.bmad'\n"
    "/\n\n"
    "&tao_params\n"
    "  global%n_opti_cycles = 100\n"
    "  global%plot_on = T\n"
    "/\n"
)


def test_interpolate_namelist_updates_and_adds():
    out = interpolate_namelist(
        _NML_SRC,
        values={
            "tao_params": {"global%n_opti_cycles": 50},
            "tao_beam_init": {"beam_init%n_particle": 5000},
        },
    )
    assert "global%n_opti_cycles = 50" in out  # updated in place
    assert "global%plot_on = T" in out  # untouched
    assert "&tao_beam_init" in out  # new group added
    assert "beam_init%n_particle = 5000" in out


def test_interpolate_namelist_removal_via_null():
    out = interpolate_namelist(_NML_SRC, values={"tao_params": {"global%plot_on": None}})
    assert "global%plot_on" not in out
    assert "global%n_opti_cycles = 100" in out


def test_interpolate_namelist_removal_only_does_not_create_group():
    out = interpolate_namelist(_NML_SRC, values={"nonexistent": {"foo": None}})
    assert "&nonexistent" not in out


def test_interpolate_namelist_repeated_group_index():
    src = "&d\n a = 1\n/\n\n&d\n a = 2\n/\n"
    out = interpolate_namelist(src, values={"d#2": {"a": 9}})
    assert "a = 1" in out  # first group untouched
    assert "a = 9" in out  # second group updated


def test_interpolate_auto_detects_namelist_by_extension():
    out = interpolate(_NML_SRC, values={"tao_params": {"global%plot_on": "F"}}, filename="tao.init")
    assert "global%n_opti_cycles = 100" in out


def test_interpolate_format_override_forces_namelist():
    out = interpolate(
        _NML_SRC,
        values={"tao_params": {"global%plot_on": "F"}},
        filename="weird.txt",
        file_format="namelist",
        options=FormatOptions(namelist=NamelistFormatOptions(align_equals=False)),
    )
    assert "global%plot_on = F" in out


def test_interpolate_namelist_rejects_renames():
    with pytest.raises(ValueError, match="rename options are not supported"):
        interpolate(_NML_SRC, filename="tao.init", renames={"foo": "bar"})


_MESSY_NML = "\n".join(
    (
        "&group1",
        "      X = 1",
        "  y=2   ! comment",
        "/",
        "",
        "",
        "&group2",
        "        Z = 3",
        "/",
        "",
    )
)

# Differing key lengths + comments, for alignment tests.
_ALIGN_NML = "&g\n a = 1 ! one\n bbbb = 2 ! two\n/\n"


def test_interpolate_namelist_verbatim_by_default():
    out = interpolate_namelist(_MESSY_NML)
    assert out == _MESSY_NML


def test_interpolate_namelist_reindents_and_lowercases_fields():
    out = interpolate_namelist(_MESSY_NML, options=NamelistFormatOptions())
    assert "\n  x = 1\n" in out  # re-indented to 2, X -> x
    assert "\n  z = 3\n" in out
    assert "\n  y = 2  ! comment\n" in out  # spacing normalized, comment aligned


def test_interpolate_namelist_configurable_indent():
    out = interpolate_namelist(_MESSY_NML, options=NamelistFormatOptions(indent_size=4))
    assert "\n    x = 1\n" in out


@pytest.mark.parametrize(
    "case, expected",
    [("lower", "\n  x = 1\n"), ("upper", "\n  X = 1\n"), ("same", "\n  X = 1\n")],
)
def test_interpolate_namelist_field_case(case, expected):
    out = interpolate_namelist(_MESSY_NML, options=NamelistFormatOptions(field_case=case))
    assert expected in out


def test_interpolate_namelist_no_align_equals():
    out = interpolate_namelist(_ALIGN_NML, options=NamelistFormatOptions(align_equals=False))
    assert "\n  a = 1     ! one\n" in out
    assert "\n  bbbb = 2  ! two\n" in out


def test_interpolate_namelist_align_equals_by_default():
    out = interpolate_namelist(_ALIGN_NML, options=NamelistFormatOptions())
    assert "\n  a    = 1" in out  # equals not aligned
    assert "\n  bbbb = 2" in out


def test_interpolate_namelist_aligns_comments_by_default():
    out = interpolate_namelist(_ALIGN_NML, options=NamelistFormatOptions(align_equals=False))
    assert "\n  a = 1     ! one\n" in out  # '!' padded to a common column
    assert "\n  bbbb = 2  ! two\n" in out


def test_interpolate_namelist_align_comments_can_be_disabled():
    out = interpolate_namelist(
        _ALIGN_NML, options=NamelistFormatOptions(align_comments=False, align_equals=False)
    )
    assert "\n  a = 1 ! one\n" in out
    assert "\n  bbbb = 2 ! two\n" in out


def test_interpolate_namelist_alignment_resets_across_blank_lines():
    src = "&g\n a = 1\n bbbb = 2\n\n c = 3\n dd = 4\n/\n"
    out = interpolate_namelist(src, options=NamelistFormatOptions(align_equals=True))
    assert "\n  a    = 1\n  bbbb = 2\n" in out  # first run aligned to width 4
    assert "\n  c  = 3\n  dd = 4\n" in out  # second run aligned to width 2


def test_interpolate_namelist_single_blank_line_after_group():
    out = interpolate_namelist(_MESSY_NML, options=NamelistFormatOptions())
    assert "/\n\n&group2" in out  # collapsed the two blank lines to one
    assert "/\n\n\n" not in out


def test_interpolate_namelist_blank_line_can_be_disabled():
    out = interpolate_namelist(
        "&a\n x=1\n/\n&b\n y=2\n/\n",
        options=NamelistFormatOptions(blank_line_after_group=False),
    )
    assert "/\n&b" in out


def test_interpolate_namelist_format_is_idempotent():
    options = NamelistFormatOptions(align_equals=True, field_case="upper")
    once = interpolate_namelist(_MESSY_NML, options=options)
    twice = interpolate_namelist(once, options=options)
    assert once == twice


def test_interpolate_formats_namelist_by_default():
    out = interpolate(_MESSY_NML, filename="tao.init")
    assert "\n  x = 1\n" in out
    assert "/\n\n&group2" in out


def test_interpolate_format_namelist_can_be_disabled():
    out = interpolate(_MESSY_NML, filename="tao.init", format_namelist=False)
    assert out == _MESSY_NML


def test_cli_interpolate_formats_namelist_by_default(tmp_path, capsys):
    (tmp_path / "tao.init").write_text(_MESSY_NML)
    cli_main_apply([str(tmp_path / "tao.init")])
    out = capsys.readouterr().out
    assert "\n  x = 1\n" in out
    assert "/\n\n&group2" in out


def test_cli_interpolate_namelist_indent(tmp_path, capsys):
    (tmp_path / "tao.init").write_text(_MESSY_NML)
    cli_main_apply([str(tmp_path / "tao.init"), "--namelist-indent", "3"])
    assert "\n   x = 1\n" in capsys.readouterr().out


def test_cli_interpolate_namelist_field_case(tmp_path, capsys):
    (tmp_path / "tao.init").write_text(_MESSY_NML)
    cli_main_apply([str(tmp_path / "tao.init"), "--namelist-field-case", "upper"])
    assert "\n  X = 1\n" in capsys.readouterr().out


def test_cli_interpolate_namelist_align_equals(tmp_path, capsys):
    (tmp_path / "a.nml").write_text(_ALIGN_NML)
    cli_main_apply([str(tmp_path / "a.nml"), "--namelist-align-equals"])
    assert "\n  a    = 1  ! one\n" in capsys.readouterr().out


def test_cli_interpolate_no_namelist_align_comments(tmp_path, capsys):
    (tmp_path / "a.nml").write_text(_ALIGN_NML)
    cli_main_apply([str(tmp_path / "a.nml"), "--no-namelist-align-comments"])
    assert "\n  a = 1 ! one\n" in capsys.readouterr().out


def test_cli_interpolate_no_format_namelist(tmp_path, capsys):
    (tmp_path / "tao.init").write_text(_MESSY_NML)
    cli_main_apply([str(tmp_path / "tao.init"), "--no-format-namelist"])
    assert capsys.readouterr().out == _MESSY_NML


def test_cli_interpolate_namelist_set(tmp_path, capsys):
    (tmp_path / "tao.init").write_text(_NML_SRC)
    cli_main_apply(
        [
            str(tmp_path / "tao.init"),
            "--set",
            "tao_params",
            "global%n_opti_cycles",
            "50",
        ]
    )
    assert "global%n_opti_cycles = 50" in capsys.readouterr().out


def test_cli_interpolate_set_overrides_values_file(tmp_path, capsys):
    (tmp_path / "tao.init").write_text(_NML_SRC)
    (tmp_path / "v.yaml").write_text("tao_params:\n  global%n_opti_cycles: 20\n")
    cli_main_apply(
        [
            str(tmp_path / "tao.init"),
            "--values",
            str(tmp_path / "v.yaml"),
            "--set",
            "tao_params",
            "global%n_opti_cycles",
            "77",
        ]
    )
    out = capsys.readouterr().out
    assert "global%n_opti_cycles = 77" in out  # --set wins over --values


def test_cli_interpolate_set_on_bmad_errors(tmp_path):
    (tmp_path / "t.bmad").write_text("Q1: quadrupole\n")
    with pytest.raises(SystemExit):
        cli_main_apply([str(tmp_path / "t.bmad"), "--set", "a", "b", "c"])


def test_cli_interpolate_in_place_namelist(tmp_path, capsys):
    path = tmp_path / "tao.init"
    path.write_text(_NML_SRC)
    cli_main_apply([str(path), "-i", "--set", "tao_params", "global%plot_on", "F"])
    assert "global%plot_on = F" in path.read_text()
    assert "wrote:" in capsys.readouterr().out


def test_cli_interpolate_in_place_bmad(tmp_path):
    path = tmp_path / "t.bmad"
    path.write_text("CX_Q: quadrupole, k1=0.0\n")
    cli_main_apply([str(path), "-i", "--rename", r"CX(_.*|$)", r"C1\1"])
    text = path.read_text()
    assert "C1_Q" in text and "CX_Q" not in text


def test_cli_interpolate_output_and_in_place_mutually_exclusive(tmp_path):
    path = tmp_path / "tao.init"
    path.write_text(_NML_SRC)
    with pytest.raises(SystemExit):
        cli_main_apply([str(path), "-i", "-o", str(tmp_path / "out.init")])
