from __future__ import annotations

import pathlib
import subprocess

import pytest

from ..diff import ParameterChange, calculate_diff, gitdiff, main
from ..parser import MemoryFiles


def _files(src: str) -> MemoryFiles:
    files = MemoryFiles.from_contents(src, "test.bmad")
    files.parse()
    files.annotate()
    return files


def _diff(src1: str, src2: str):
    return calculate_diff(_files(src1), _files(src2))


def test_no_differences():
    src = """
    qf: quad, l = 0.1
    top: line = (qf)
    use, top
    """
    diff = _diff(src, src)
    assert not diff.has_param_diffs
    assert not diff.has_ele_diffs


def test_parameter_added_removed_changed():
    src1 = """
    old_const = 1
    same_const = 5
    changed_const = 2
    """
    src2 = """
    same_const = 5
    changed_const = 3
    new_const = 4
    """
    diff = _diff(src1, src2)
    assert diff.params_added == [ParameterChange("", "new_const", None, "4")]
    assert diff.params_removed == [ParameterChange("", "old_const", "1", None)]
    assert diff.params_changed == [ParameterChange("", "changed_const", "2", "3")]


def test_parameter_target():
    diff = _diff("parameter[e_tot] = 1e9", "parameter[e_tot] = 2e9")
    (change,) = diff.params_changed
    assert change.target == "parameter"
    assert change.name == "e_tot"
    assert change.old_value == "1e9"
    assert change.new_value == "2e9"


def test_element_added_and_removed():
    src1 = """
    qf: quad, l = 0.1
    top: line = (qf)
    use, top
    """
    src2 = """
    qf: quad, l = 0.1
    bend1: sbend, l = 0.5
    top: line = (qf, bend1)
    use, top
    """
    diff = _diff(src1, src2)
    assert "BEND1" in diff.eles_added
    assert diff.eles_removed == []

    reverse = _diff(src2, src1)
    assert "BEND1" in reverse.eles_removed
    assert reverse.eles_added == []


def test_element_attribute_changes():
    src1 = """
    qf: quad, l = 0.1, k1 = 0.5
    top: line = (qf)
    use, top
    """
    src2 = """
    qf: quad, l = 0.2, tilt = 0.1
    top: line = (qf)
    use, top
    """
    diff = _diff(src1, src2)
    details = diff.eles_changed["QF"]
    assert details.type_change is None
    assert details.attrs_added == [("tilt", "0.1")]
    assert details.attrs_removed == [("k1", "0.5")]
    assert details.attrs_changed == [("l", "0.1", "0.2")]


def test_element_type_change():
    src1 = """
    m1: quad, l = 0.1
    top: line = (m1)
    use, top
    """
    src2 = """
    m1: sbend, l = 0.1
    top: line = (m1)
    use, top
    """
    diff = _diff(src1, src2)
    assert diff.eles_changed["M1"].type_change == ("QUAD", "SBEND")


def test_element_renamed():
    src1 = """
    qf: quad, l = 0.1
    top: line = (qf)
    use, top
    """
    src2 = """
    qd: quad, l = 0.1
    top: line = (qd)
    use, top
    """
    diff = _diff(src1, src2)
    assert ("QF", "QD") in diff.eles_renamed
    assert "QF" not in diff.eles_removed
    assert "QD" not in diff.eles_added


@pytest.fixture
def lattice_pair(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    file1 = tmp_path / "one.bmad"
    file2 = tmp_path / "two.bmad"
    file1.write_text(
        """
        qf: quad, l = 0.1
        top: line = (qf)
        use, top
        """
    )
    file2.write_text(
        """
        qf: quad, l = 0.2
        top: line = (qf)
        use, top
        """
    )
    return file1, file2


def test_cli_main(lattice_pair: tuple[pathlib.Path, pathlib.Path], capsys: pytest.CaptureFixture):
    file1, file2 = lattice_pair
    main([str(file1), str(file2)])
    out = capsys.readouterr().out
    assert "QF" in out


@pytest.fixture
def git_lattice_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A git repository with two commits of ``lat.bmad``."""

    def git(*args: str) -> None:
        subprocess.check_call(
            ["git", *args],
            cwd=tmp_path,
            env={
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "test@example.com",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "test@example.com",
                "HOME": str(tmp_path),
                "PATH": "/usr/bin:/bin",
            },
        )

    lat = tmp_path / "lat.bmad"
    git("init")
    lat.write_text("qf: quad, l = 0.1\ntop: line = (qf)\nuse, top\n")
    git("add", "lat.bmad")
    git("commit", "-m", "initial")
    lat.write_text("qf: quad, l = 0.2\ntop: line = (qf)\nuse, top\n")
    git("add", "lat.bmad")
    git("commit", "-m", "change l")
    return tmp_path


def test_cli_main_git_spec(
    git_lattice_repo: pathlib.Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(git_lattice_repo)
    main(["HEAD~1:lat.bmad", "lat.bmad"])
    out = capsys.readouterr().out
    assert "QF" in out


def test_cli_gitdiff(
    git_lattice_repo: pathlib.Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(git_lattice_repo)
    gitdiff(["lat.bmad", "HEAD~1", "HEAD"])
    out = capsys.readouterr().out
    assert "QF" in out
