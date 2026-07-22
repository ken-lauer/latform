# Configuration

`latform` and `latform-lint` read project-wide settings from a configuration
file. Settings may live in a standalone `latform.toml` (bare tables) or under
`[tool.latform]` in `pyproject.toml`. Discovery walks upward from the current
directory; the first `latform.toml` wins, otherwise the first `pyproject.toml`
that contains a `[tool.latform]` table.

```toml
# latform.toml

# Top-level lattice entry points. Used when latform / latform-lint are invoked
# with no file arguments (implies recursive parsing). Paths are relative to this
# config file.
top-level = ["lat/main.bmad"]

# Formatting settings (same names as the CLI flags, without the leading --).
[format]
line-length = 100
name-case = "upper"
kind-case = "lower"

# Lint settings.
[lint]
ignore = ["LF002"]              # codes to suppress everywhere
min-name-length = 1             # LF009: minimum constant/element name length;
                                # at the default of 1, only i/l/o are flagged
builtin-constant-rtol = 1e-4    # LF010: relative tolerance when matching constant
                                # values against built-in physical constants

[lint.per-file-ignores]
"legacy/*.bmad" = ["LF004", "LF006"]
```

The equivalent in `pyproject.toml` nests everything under `[tool.latform]`
(`[tool.latform.format]`, `[tool.latform.lint]`,
`[tool.latform.lint.per-file-ignores]`).

Precedence is **command-line flag > config file > built-in default**, so an
explicit flag such as `--name-case upper` always overrides the config. Lint
ignores are cumulative: `--ignore` on the command line, the global `[lint] ignore`
list, and any matching `[lint.per-file-ignores]` entries are all applied.

| Option        | Description                                                             |
| ------------- | ----------------------------------------------------------------------- |
| `--config PATH` | Use a specific config file instead of discovering one.                |
| `--no-config`   | Ignore any `latform.toml` / `pyproject.toml` configuration.           |

## Deriving inputs from a Tao init file

Instead of (or in addition to) `top-level`, point the entry points at a Tao
`tao.init`; its `design_lattice` files become the top-level lattices:

```toml
# latform.toml
tao-init = "tao.init"   # a path, or a list of paths
```

When invoked with no file arguments, `latform` / `latform-lint` then load the
lattices referenced by that `tao.init` (recursively).

## Common setups

A minimal single-project config — set the entry point and a couple of house
style rules:

```toml
# latform.toml
top-level = ["lat/main.bmad"]

[format]
line-length = 100
name-case = "upper"
```

Silence a noisy lint everywhere, but keep it enforced outside a legacy tree:

```toml
[lint]
ignore = ["LF007"]                 # unused_constant: allowed project-wide

[lint.per-file-ignores]
"vendor/**/*.bmad" = ["LF004"]     # tolerate unknown attributes in vendored files
```

## Namelist formatting settings

The `tao.init` / namelist formatting behavior is currently controlled by
**command-line flags only** (see
[Namelist (tao.init) formatting](cli/index.md#namelist-taoinit-formatting)).
The `[format]` table understands the Bmad formatting keys listed above; namelist
keys such as `namelist-field-case` are not recognized there yet and are ignored
with a warning.
