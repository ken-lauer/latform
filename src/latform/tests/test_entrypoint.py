from __future__ import annotations

import pathlib
import shutil

import pytest
from pytest_mock import MockerFixture

from ..main import cli_main, load_renames, main
from ..parser import Files, MemoryFiles, build_files
from .conftest import LATTICE_FILES

lattice_file = pytest.mark.parametrize(
    "lattice_file", [pytest.param(p, id=p.name) for p in LATTICE_FILES]
)


@pytest.fixture
def input_filename(tmp_path: pathlib.Path, lattice_file: pathlib.Path) -> pathlib.Path:
    dest_file = tmp_path / lattice_file.name
    shutil.copy(lattice_file, dest_file)
    return dest_file


def test_stdin_processing(capsys: pytest.CaptureFixture, mocker: MockerFixture):
    input_content = "d1    : drift   , L   =1.0;"
    mocker.patch("sys.stdin.read", return_value=input_content)

    main(filename="-")

    captured = capsys.readouterr()
    assert "D1: drift, L=1.0" in captured.out.splitlines()


@lattice_file
def test_standard_formatting_stdout(input_filename: pathlib.Path, capsys: pytest.CaptureFixture):
    main(filename=input_filename)

    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert ":" in captured.out


@lattice_file
def test_formatting_output_file(input_filename: pathlib.Path, tmp_path: pathlib.Path):
    output_file = tmp_path / "_formatted_out.bmad"

    main(filename=input_filename, output=output_file)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


@lattice_file
def test_in_place_modification(input_filename: pathlib.Path):
    original_content = input_filename.read_text()
    main(filename=input_filename, in_place=True)

    assert input_filename.exists()
    new_content = input_filename.read_text()
    assert len(new_content) > 0
    assert new_content != original_content


@lattice_file
def test_in_place_modification_recursive(lattice_file: pathlib.Path, tmp_path: pathlib.Path):
    shutil.copytree(lattice_file.parent, tmp_path, dirs_exist_ok=True)

    input_filename = tmp_path / lattice_file.name
    original_content = input_filename.read_text()
    main(filename=input_filename, in_place=True, recursive=True)

    assert input_filename.exists()
    new_content = input_filename.read_text()
    assert len(new_content) > 0
    assert new_content != original_content


@lattice_file
def test_diff_generation(input_filename: pathlib.Path, capsys: pytest.CaptureFixture):
    main(filename=input_filename, diff=True)

    captured = capsys.readouterr()
    assert "---" in captured.out
    assert "+++" in captured.out


def test_rename_logic(tmp_path: pathlib.Path):
    content = "QF: QUAD, L=1, K1=0.5;"
    f = tmp_path / "test.bmad"
    f.write_text(content)

    main(filename=f, raw_renames=["QF, QUAD_ABC"], compact=True)


def test_rename_functionality(capsys: pytest.CaptureFixture, tmp_path):
    f = tmp_path / "rename_test.bmad"
    f.write_text("OLD_NAME: DRIFT, L=1;")

    main(filename=f, raw_renames=["OLD_NAME, NEW_NAME"])

    captured = capsys.readouterr()
    assert "NEW_NAME: drift, L=1" in captured.out


@lattice_file
def test_verbosity_levels(input_filename: pathlib.Path, capsys: pytest.CaptureFixture):
    main(filename=input_filename, verbose=2)
    captured = capsys.readouterr()
    assert "-- Block" in captured.err


@pytest.fixture
def missing_call_file(tmp_path: pathlib.Path) -> pathlib.Path:
    f = tmp_path / "with_missing_call.bmad"
    f.write_text("d1: drift, L=1.0;\ncall, file=does_not_exist.bmad;\n")
    return f


def test_missing_call_ignored_by_default(missing_call_file: pathlib.Path):
    # error_if_missing not set - does not raise
    main(filename=missing_call_file, recursive=True)


def test_missing_call_raises_when_requested(missing_call_file: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        main(filename=missing_call_file, recursive=True, error_if_missing=True)


def test_cli_exits_on_missing_call(missing_call_file: pathlib.Path):
    with pytest.raises(SystemExit) as exc_info:
        cli_main([str(missing_call_file), "--recursive", "--error-if-missing"])
    assert exc_info.value.code == 1


def test_build_files_default_is_per_file(tmp_path: pathlib.Path):
    f1 = tmp_path / "a.bmad"
    f2 = tmp_path / "b.bmad"
    f1.write_text("Q1: quad;\n")
    f2.write_text("Q2: quad;\n")

    result = build_files([f1, f2])
    assert len(result) == 2
    assert all(isinstance(f, Files) for f in result)
    assert [fobj.top_files[0].name for fobj in result] == ["a.bmad", "b.bmad"]


def test_build_files_combine_groups_into_one(tmp_path: pathlib.Path):
    f1 = tmp_path / "a.bmad"
    f2 = tmp_path / "b.bmad"
    f1.write_text("Q1: quad;\n")
    f2.write_text("Q2: quad;\n")

    (combined,) = build_files([f1, f2], combine=True)
    assert isinstance(combined, Files)
    assert [p.name for p in combined.top_files] == ["a.bmad", "b.bmad"]


def test_build_files_combine_with_stdin_uses_memory_files(
    tmp_path: pathlib.Path, mocker: MockerFixture
):
    mocker.patch("sys.stdin.read", return_value="QS: quad;\n")
    f1 = tmp_path / "a.bmad"
    f1.write_text("Q1: quad;\n")

    (combined,) = build_files([f1, "-"], combine=True, root_path=tmp_path)
    assert isinstance(combined, MemoryFiles)
    assert len(combined.top_files) == 2


def test_cli_combine_outputs_both_files(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture):
    f1 = tmp_path / "a.bmad"
    f2 = tmp_path / "b.bmad"
    f1.write_text("QA: drift, L=1;\n")
    f2.write_text("QB: drift, L=2;\n")

    cli_main([str(f1), str(f2), "--combine"])

    captured = capsys.readouterr()
    assert "QA: drift, L=1" in captured.out
    assert "QB: drift, L=2" in captured.out


def _make_tao_init_set(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "cx.lat.bmad").write_text("Q1: quadrupole, L=0.3, k1=1.0\ncl: line=(Q1)\nuse, cl\n")
    init = tmp_path / "tao.init"
    init.write_text(
        "&tao_design_lattice\n"
        "      design_lattice(1)%file = 'cx.lat.bmad'\n"
        "/\n\n\n"
        "&tao_params\n"
        "        global%n_opti_cycles = 100\n"
        "/\n"
    )
    return init


def test_cli_tao_init_formatted_by_default(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture):
    init = _make_tao_init_set(tmp_path)
    cli_main([str(init)])
    out = capsys.readouterr().out
    assert "Q1: quadrupole" in out  # the referenced lattice is formatted too
    assert "\n  design_lattice(1)%file = 'cx.lat.bmad'\n" in out  # init re-indented to 2
    assert "/\n\n&tao_params" in out  # collapsed to a single blank line


def test_cli_tao_init_no_format_namelist(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture):
    init = _make_tao_init_set(tmp_path)
    cli_main([str(init), "--no-format-namelist"])
    out = capsys.readouterr().out
    assert "\n      design_lattice(1)%file = 'cx.lat.bmad'\n" in out  # 6-space indent preserved
    assert "/\n\n\n&tao_params" in out  # source blank lines preserved


def test_cli_tao_init_in_place_rewrites_init(tmp_path: pathlib.Path):
    init = _make_tao_init_set(tmp_path)
    cli_main([str(init), "--in-place"])
    text = init.read_text()
    assert "\n  design_lattice(1)%file = 'cx.lat.bmad'\n" in text
    assert "/\n\n&tao_params" in text


def test_load_renames(tmp_path: pathlib.Path):
    rename_file = tmp_path / "renames.csv"
    rename_file.write_text("A,B\nC,D")

    renames_arg = {"E": "F"}
    raw_renames = ["G,H"]

    result = load_renames(rename_file, raw_renames, renames_arg)

    expected = {"A": "B", "C": "D", "G": "H", "E": "F"}
    assert result == expected
