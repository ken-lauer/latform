from __future__ import annotations

import pathlib

import pytest

from ..config import ConfigError, LatformProjectConfig, discover_config, find_config_file

LATFORM_TOML = """\
top-level = ["lat/main.bmad"]

[format]
name-case = "lower"
line-length = 80

[lint]
ignore = ["lf002"]

[lint.per-file-ignores]
"legacy/*.bmad" = ["LF004", "LF006"]
"""

PYPROJECT_TOML = """\
[tool.latform]
top-level = ["a.bmad"]

[tool.latform.format]
compact = true

[tool.latform.lint]
ignore = ["LF002"]
"""


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "latform.toml").write_text(LATFORM_TOML)
    (tmp_path / "lat").mkdir()
    (tmp_path / "legacy").mkdir()
    return tmp_path


def test_loads_latform_toml(project: pathlib.Path):
    config = discover_config(project)
    assert config.source == project / "latform.toml"
    assert config.top_level == ["lat/main.bmad"]
    assert config.format == {"name_case": "lower", "line_length": 80}
    assert config.lint_ignore == ["LF002"]  # normalized to upper


def test_resolve_top_level(project: pathlib.Path):
    config = discover_config(project)
    assert config.resolve_top_level() == [project / "lat/main.bmad"]


def test_ignores_for_merges_global_and_per_file(project: pathlib.Path):
    config = discover_config(project)
    assert config.ignores_for(project / "legacy" / "old.bmad") == {"LF002", "LF004", "LF006"}
    assert config.ignores_for(project / "lat" / "main.bmad") == {"LF002"}


def test_pyproject_fallback(tmp_path: pathlib.Path):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT_TOML)
    config = discover_config(tmp_path)
    assert config.source == tmp_path / "pyproject.toml"
    assert config.format == {"compact": True}
    assert config.top_level == ["a.bmad"]


def test_latform_toml_preferred_over_pyproject(tmp_path: pathlib.Path):
    (tmp_path / "latform.toml").write_text(LATFORM_TOML)
    (tmp_path / "pyproject.toml").write_text(PYPROJECT_TOML)
    assert find_config_file(tmp_path).name == "latform.toml"


def test_pyproject_without_latform_table_is_ignored(tmp_path: pathlib.Path):
    (tmp_path / "pyproject.toml").write_text("[tool.black]\nline-length = 88\n")
    config = discover_config(tmp_path)
    assert config.source is None
    assert config.format == {}


def test_discovery_walks_up_parents(project: pathlib.Path):
    nested = project / "lat"
    config = discover_config(nested)
    assert config.source == project / "latform.toml"


def test_no_config_returns_empty(project: pathlib.Path):
    config = discover_config(project, enabled=False)
    assert config.source is None
    assert config.format == {}
    assert config.top_level == []


def test_explicit_path(project: pathlib.Path):
    config = discover_config(explicit=project / "latform.toml")
    assert config.top_level == ["lat/main.bmad"]


def test_explicit_missing_path_raises(tmp_path: pathlib.Path):
    with pytest.raises(ConfigError):
        discover_config(explicit=tmp_path / "nope.toml")


def test_min_name_length_default_is_one(project: pathlib.Path):
    assert discover_config(project).min_name_length == 1


@pytest.mark.parametrize("key", ["min-name-length", "min_name_length"])
def test_min_name_length_parsed(tmp_path: pathlib.Path, key: str):
    (tmp_path / "latform.toml").write_text(f"[lint]\n{key} = 3\n")
    assert discover_config(tmp_path).min_name_length == 3


@pytest.mark.parametrize("value", ["0", '"three"', "true"])
def test_invalid_min_name_length_raises(tmp_path: pathlib.Path, value: str):
    (tmp_path / "latform.toml").write_text(f"[lint]\nmin-name-length = {value}\n")
    with pytest.raises(ConfigError):
        discover_config(tmp_path)


def test_invalid_case_value_raises(tmp_path: pathlib.Path):
    (tmp_path / "latform.toml").write_text('[format]\nname-case = "Upper"\n')
    with pytest.raises(ConfigError):
        discover_config(tmp_path)


def test_unknown_format_key_warns_and_is_ignored(tmp_path: pathlib.Path, caplog):
    (tmp_path / "latform.toml").write_text("[format]\nbogus-setting = 1\nline-length = 90\n")
    config = discover_config(tmp_path)
    assert config.format == {"line_length": 90}
    assert any("bogus-setting" in rec.getMessage() for rec in caplog.records)


def test_empty_config_when_none_found(tmp_path: pathlib.Path):
    config = discover_config(tmp_path)
    assert isinstance(config, LatformProjectConfig)
    assert config.source is None


# --- CLI integration ---------------------------------------------------------


def test_cli_applies_format_config(tmp_path, monkeypatch, capsys):
    from ..main import cli_main

    (tmp_path / "latform.toml").write_text('[format]\nname-case = "lower"\n')
    (tmp_path / "m.bmad").write_text("Q1: quadrupole, k1 = 0.5\n")
    monkeypatch.chdir(tmp_path)

    cli_main(["m.bmad"])
    assert "q1: quadrupole" in capsys.readouterr().out


def test_cli_flag_overrides_config(tmp_path, monkeypatch, capsys):
    from ..main import cli_main

    (tmp_path / "latform.toml").write_text('[format]\nname-case = "lower"\n')
    (tmp_path / "m.bmad").write_text("Q1: quadrupole, k1 = 0.5\n")
    monkeypatch.chdir(tmp_path)

    cli_main(["--name-case", "upper", "m.bmad"])
    assert "Q1: quadrupole" in capsys.readouterr().out


def test_cli_top_level_fallback(tmp_path, monkeypatch, capsys):
    from ..main import cli_main

    (tmp_path / "latform.toml").write_text('top-level = ["m.bmad"]\n')
    (tmp_path / "m.bmad").write_text("Q1: quadrupole\n")
    monkeypatch.chdir(tmp_path)

    cli_main([])  # no file args -> use top-level
    assert "quadrupole" in capsys.readouterr().out


def test_lint_cli_per_file_ignore(tmp_path, monkeypatch):
    from ..lint import cli_main as lint_cli_main

    (tmp_path / "latform.toml").write_text('[lint.per-file-ignores]\n"*.bmad" = ["LF004"]\n')
    (tmp_path / "m.bmad").write_text("Q1: quadrupole, bogus = 1\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        lint_cli_main(["m.bmad"])
    assert excinfo.value.code == 0  # LF004 suppressed by config


def test_lint_cli_no_config_disables_ignore(tmp_path, monkeypatch):
    from ..lint import cli_main as lint_cli_main

    (tmp_path / "latform.toml").write_text('[lint.per-file-ignores]\n"*.bmad" = ["LF004"]\n')
    (tmp_path / "m.bmad").write_text("Q1: quadrupole, bogus = 1\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        lint_cli_main(["--no-config", "m.bmad"])
    assert excinfo.value.code == 1  # not suppressed
