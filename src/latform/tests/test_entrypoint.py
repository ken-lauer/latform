from __future__ import annotations

import pathlib
import shutil

import pytest
from pytest_mock import MockerFixture

from ..main import load_renames, main
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


def test_load_renames(tmp_path: pathlib.Path):
    rename_file = tmp_path / "renames.csv"
    rename_file.write_text("A,B\nC,D")

    renames_arg = {"E": "F"}
    raw_renames = ["G,H"]

    result = load_renames(rename_file, raw_renames, renames_arg)

    expected = {"A": "B", "C": "D", "G": "H", "E": "F"}
    assert result == expected
