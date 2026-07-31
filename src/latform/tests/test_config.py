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


@pytest.mark.parametrize("value", ['"tao.init"', '["a/tao.init", "b/tao.init"]'])
def test_tao_init_parsed_and_resolved(tmp_path: pathlib.Path, value: str):
    (tmp_path / "latform.toml").write_text(f"tao-init = {value}\n")
    config = discover_config(tmp_path)
    assert config.tao_init  # non-empty
    assert config.resolve_tao_init() == [tmp_path / entry for entry in config.tao_init]


def test_tao_init_invalid_type_raises(tmp_path: pathlib.Path):
    (tmp_path / "latform.toml").write_text("tao-init = 123\n")
    with pytest.raises(ConfigError):
        discover_config(tmp_path)


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


# --- tao.init as an implicit top-level -------------------------------------


def test_tao_init_used_as_top_level_when_no_config(tmp_path: pathlib.Path):
    # With no latform.toml / pyproject.toml, a nearby tao.init becomes the
    # top-level entry point.
    (tmp_path / "tao.init").write_text("&tao_start\n/\n")
    config = discover_config(tmp_path)
    assert config.source is None  # not loaded from a config file
    assert config.top_level == [str((tmp_path / "tao.init").resolve())]


def test_tao_init_fallback_walks_up_parents(tmp_path: pathlib.Path):
    (tmp_path / "tao.init").write_text("&tao_start\n/\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    config = discover_config(nested)
    assert config.top_level == [str((tmp_path / "tao.init").resolve())]


def test_config_file_preferred_over_tao_init(tmp_path: pathlib.Path):
    (tmp_path / "tao.init").write_text("&tao_start\n/\n")
    (tmp_path / "latform.toml").write_text('top-level = ["m.bmad"]\n')
    config = discover_config(tmp_path)
    assert config.source == tmp_path / "latform.toml"
    assert config.top_level == ["m.bmad"]  # the config file wins; tao.init ignored


def test_no_config_skips_tao_init_fallback(tmp_path: pathlib.Path):
    (tmp_path / "tao.init").write_text("&tao_start\n/\n")
    assert discover_config(tmp_path, enabled=False).top_level == []


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


def test_builtin_constant_rtol_default(project: pathlib.Path):
    assert discover_config(project).builtin_constant_rtol == 1e-4


@pytest.mark.parametrize("key", ["builtin-constant-rtol", "builtin_constant_rtol"])
def test_builtin_constant_rtol_parsed(tmp_path: pathlib.Path, key: str):
    (tmp_path / "latform.toml").write_text(f"[lint]\n{key} = 1e-9\n")
    assert discover_config(tmp_path).builtin_constant_rtol == 1e-9


@pytest.mark.parametrize("value", ["0", "-1e-6", "true", '"x"'])
def test_invalid_builtin_constant_rtol_raises(tmp_path: pathlib.Path, value: str):
    (tmp_path / "latform.toml").write_text(f"[lint]\nbuiltin-constant-rtol = {value}\n")
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


# --- namelist format settings ------------------------------------------------


def test_namelist_format_keys_parsed(tmp_path: pathlib.Path):
    (tmp_path / "latform.toml").write_text(
        "[format]\n"
        "format-namelist = false\n"
        "namelist-indent = 4\n"
        'namelist-field-case = "upper"\n'
        "namelist-align-equals = false\n"
        "namelist-align-comments = false\n"
    )
    fmt = discover_config(tmp_path).format
    assert fmt == {
        "format_namelist": False,
        "namelist_indent": 4,
        "namelist_field_case": "upper",
        "namelist_align_equals": False,
        "namelist_align_comments": False,
    }


def test_namelist_logicals_default(project: pathlib.Path):
    assert discover_config(project).namelist_logicals == ("T", "F")


def test_namelist_logicals_pair_parsed(tmp_path: pathlib.Path):
    (tmp_path / "latform.toml").write_text('[format]\nnamelist-logicals = [".true.", ".false."]\n')
    config = discover_config(tmp_path)
    assert config.namelist_logicals == (".true.", ".false.")
    assert "namelist_logicals" not in config.format  # not folded into argparse dests


def test_namelist_logicals_false_disables(tmp_path: pathlib.Path):
    (tmp_path / "latform.toml").write_text("[format]\nnamelist-logicals = false\n")
    assert discover_config(tmp_path).namelist_logicals is None


@pytest.mark.parametrize("value", ['["T"]', '"T"', "true", "[1, 2]"])
def test_invalid_namelist_logicals_raises(tmp_path: pathlib.Path, value: str):
    (tmp_path / "latform.toml").write_text(f"[format]\nnamelist-logicals = {value}\n")
    with pytest.raises(ConfigError):
        discover_config(tmp_path)


@pytest.mark.parametrize(
    "line",
    [
        'namelist-indent = "x"',
        "namelist-indent = -1",
        "namelist-align-equals = 1",
        'format-namelist = "yes"',
        'namelist-field-case = "Upper"',
    ],
)
def test_invalid_namelist_format_raises(tmp_path: pathlib.Path, line: str):
    (tmp_path / "latform.toml").write_text(f"[format]\n{line}\n")
    with pytest.raises(ConfigError):
        discover_config(tmp_path)


def test_cli_applies_namelist_config(tmp_path, monkeypatch, capsys):
    from ..main import cli_main

    (tmp_path / "latform.toml").write_text(
        '[format]\nnamelist-field-case = "upper"\nnamelist-logicals = [".true.", ".false."]\n'
    )
    (tmp_path / "ring.bmad").write_text("q1: quadrupole, l=1\ncl: line=(q1)\nuse, cl\n")
    (tmp_path / "tao.init").write_text(
        "&tao_design_lattice\n"
        "  design_lattice(1)%file = 'ring.bmad'\n"
        "/\n\n"
        "&tao_params\n"
        "  global%n_opti_cycles = 100\n"
        "  bmad_com%radiation_damping_on = T\n"
        "/\n"
    )
    monkeypatch.chdir(tmp_path)

    cli_main(["tao.init"])
    out = capsys.readouterr().out
    assert "GLOBAL%N_OPTI_CYCLES" in out  # field-case "upper" from config
    assert ".true." in out  # logicals pair from config (T -> .true.)
    assert "= T\n" not in out  # the default pair was overridden


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
