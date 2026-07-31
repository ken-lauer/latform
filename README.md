# latform

A Bmad lattice parser, formatter, and analysis toolkit for particle accelerator
lattice files. It parses, formats, lints, diffs, inspects, and templates Bmad
lattices, and also understands Tao `tao.init` / Fortran-namelist files.

## Installation

```bash
pip install latform
```

<details>
<summary>From source / with a conda environment</summary>

```bash
conda env create -n latform python=3.12 pip
conda activate latform

git clone https://github.com/ken-lauer/latform
cd latform
python -m pip install .
```

Optional extras — `.[test]` (test suite), `.[doc]` (docs), `.[lsp]` (language
server), or combine them: `.[test,doc]`.

</details>

## Quickstart

```bash
# Format a lattice and print to stdout
latform my_lattice.bmad

# Format in place
latform -i my_lattice.bmad

# Preview what the formatter would change (no writes)
latform --diff my_lattice.bmad

# Format from a pipe
cat my_lattice.bmad | latform -

# Format every file in a directory, in place
latform -i *.bmad

# Lint without reformatting (exits non-zero on findings — good for CI)
latform-lint my_lattice.bmad
```

Point `latform` at a Tao `tao.init` and it formats the `tao.init` **and** every
Bmad lattice it references:

```bash
latform -i tao.init        # rewrite the init and its lattices in place
latform --diff tao.init    # preview all of the changes
```

## The tools

latform installs several focused commands:

| Command            | What it does                                                        |
| ------------------ | ------------------------------------------------------------------- |
| `latform`          | Parse and format Bmad lattice files (and `tao.init` namelists)      |
| `latform-lint`     | Lint lattice files without reformatting (non-zero exit on findings) |
| `latform-apply`    | Apply value overrides / renames to a single template file           |
| `latform-template` | Expand a template set across many instances                         |
| `latform-dump`     | Extract parameters and element information                          |
| `latform-diff`     | Compare two lattice files structurally                              |
| `latform-gitdiff`  | Compare a lattice file across git revisions                         |
| `latform-graph`    | Visualize the file dependency (`call`) tree                         |

Run any command with `-h` for its full option list, or see the
[CLI reference](https://ken-lauer.github.io/latform/cli/). A ninth command,
`latform-lsp`, provides editor integration — see
[Editor integration](#editor-integration-language-server).

## Common operations

### Enforce house style on save / in CI

See also: [latform-pre-commit](https://github.com/ken-lauer/latform-pre-commit).

Format in place and lint as two steps:

```bash
latform -i lat/*.bmad         # reformat
latform-lint -r lat/main.bmad # lint recursively; exits 1 on any finding
```

Set your project's rules once in a `latform.toml` (see
[Configuration](#configuration)) so every invocation and every contributor gets
the same result. `latform --lint` reports lints alongside the formatted output;
`latform-lint` is the CI-friendly, exit-code-driven counterpart.

### Rename elements

```bash
# Inline old,new pairs (repeatable)
latform -R 'Q1,QF' -R 'Q2,QD' my_lattice.bmad

# From a CSV of old,new pairs
latform --rename-file renames.csv my_lattice.bmad
```

### Flatten a multi-file lattice into one

```bash
latform -r my_lattice.bmad         # parse recursively, following `call`s
latform --flatten my_lattice.bmad  # inline all called files into one output
```

### Inspect a lattice

```bash
latform-dump my_lattice.bmad            # parameters, used, and unused elements
latform-dump -u my_lattice.bmad         # only unused elements
latform-dump -m 'Q*' my_lattice.bmad    # filter by glob (or -r for regex)
latform-dump -d ',' my_lattice.bmad     # CSV-style output
```

### Compare lattices

```bash
latform-diff old.bmad new.bmad          # structural diff of params & elements
latform-gitdiff my_lattice.bmad HEAD~1  # compare a file against an earlier commit
```

### Visualize the dependency tree

```bash
latform-graph my_lattice.bmad                 # text tree of `call`ed files
latform-graph -f mermaid -o deps.mmd main.bmad  # Mermaid diagram to a file
```

### Fill in a single template file

A Bmad template is itself valid Bmad; a YAML sidecar supplies per-element values.

```bash
# values.yaml:  Q1: { k1: 1.523 }
latform-apply quad.bmad --values values.yaml
latform-apply quad.bmad --rename 'Q(\d+)' 'ARC_Q\1'   # regex rename
```

The same command edits `tao.init` / namelist files, keyed by namelist group:

```bash
latform-apply tao.init --set tao_params global%n_opti_cycles 50 -i
```

### Expand a template set across instances

Describe the template files, per-instance values, and renames in an
`instances.yaml`, then generate every instance at once:

```bash
latform-template instances.yaml -d build/            # write instances to build/
latform-template instances.yaml -d build/ --dry-run  # list files without writing
```

See [Templating](https://ken-lauer.github.io/latform/cli/templating/) for the `instances.yaml` schema
(prefix / suffix / regex / parts renames, shared `context:` files, and per-instance
`tao_init:` generation).

## Configuration

`latform` and `latform-lint` read project-wide settings from a `latform.toml`
(bare tables) or a `[tool.latform]` section in `pyproject.toml`. Discovery walks
upward from the current directory: the first `latform.toml` wins, otherwise the
first `pyproject.toml` that contains a `[tool.latform]` table. `--config PATH`
forces a specific file; `--no-config` skips config entirely.

Precedence is **CLI flag → config → built-in default**. Only `latform` and
`latform-lint` read this file (`latform-apply` / `latform-template` take their
namelist options as flags). The block below lists every recognized key with its
default; unknown keys in `[format]` warn and are ignored.

```toml
# latform.toml — all options shown with their defaults.

# ── Entry points ────────────────────────────────────────────────────────────
# Used when latform / latform-lint run with NO file arguments. Paths are
# relative to this config file. Providing either implies recursive parsing.
top-level = ["lat/main.bmad"]     # list of lattice files
tao-init  = "tao.init"            # a path, or a list of paths; its
                                  # design_lattice files become the top level

# ── Formatting ──────────────────────────────────────────────────────────────
# Same names as the CLI flags, minus the leading `--`. An explicit CLI flag
# always overrides the value here.
[format]
line-length              = 100        # int
max-line-length          = 130        # int; default = 130% of line-length
compact                  = false      # bool; no blank lines between statement types
name-case                = "upper"    # upper | lower | same  (element names)
attribute-case           = "lower"    # upper | lower | same  (attribute names)
kind-case                = "lower"    # upper | lower | same  (element types/keywords)
builtin-case             = "lower"    # upper | lower | same  (builtin functions)
controller-variable-case = "same"     # upper | lower | same  (overlay/group/ramper vars)
section-break-character  = "-"        # str
section-break-width      = 100        # int; default = line-length
strip-comments           = false      # bool

# Namelist (tao.init / *.nml) formatting — also under [format]:
format-namelist          = true       # bool; false = preserve layout verbatim
namelist-indent          = 2          # int >= 0; field indent width
namelist-field-case      = "lower"    # upper | lower | same  (field names)
namelist-align-equals    = true       # bool; align `=` into a column
namelist-align-comments  = true       # bool; align trailing `!` comments
namelist-logicals        = ["T", "F"] # [true, false] tokens, or `false` to
                                      # leave logical values untouched

# ── Linting ─────────────────────────────────────────────────────────────────
[lint]
ignore                = ["LF002"]     # list of codes to suppress everywhere
min-name-length       = 1             # int >= 1; LF009 flags names shorter than this
                                      # (at 1, only the confusable i/l/o are flagged)
builtin-constant-rtol = 1e-4          # positive float; LF010 match tolerance vs.
                                      # built-in physical constants

# Per-file lint suppression. Keys are globs matched against the file's name,
# full path, and path relative to this config; values are codes to ignore there.
[lint.per-file-ignores]
"legacy/*.bmad"    = ["LF004", "LF006"]
"vendor/**/*.bmad" = ["LF004"]
```

Lint ignores are cumulative: `--ignore` on the CLI, the global `[lint] ignore`
list, and any matching `[lint.per-file-ignores]` entries all apply.

The equivalent in `pyproject.toml` nests everything one level deeper under
`[tool.latform]`:

```toml
[tool.latform]
top-level = ["lat/main.bmad"]

[tool.latform.format]
line-length = 100
name-case   = "upper"
kind-case   = "lower"

[tool.latform.lint]
ignore = ["LF002"]

[tool.latform.lint.per-file-ignores]
"legacy/*.bmad" = ["LF004", "LF006"]
```

With no config at all, the tools fall back to a `tao.init` found in the current
directory or a parent, so they work in a bare Tao project directory. See the
[Lint Codes](https://ken-lauer.github.io/latform/cli/lint/#lint-codes) reference
for what each `LF0xx` means.

## Editor integration (language server)

latform ships a Language Server Protocol implementation, `latform-lsp`, for Bmad
lattice files. It provides go-to-definition, find-references, hover, completion,
rename, formatting (whole-file and range), document/workspace symbols, semantic
highlighting, code actions, and live diagnostics from the same lint rules as
`latform-lint`. It reads your project's `latform.toml` / `pyproject.toml`, so
formatting and diagnostics match the CLI.

The server is behind the optional `lsp` extra (it needs [`pygls`](https://github.com/openlawlibrary/pygls)):

```bash
pip install "latform[lsp]"
latform-lsp --help        # runs over stdio; normally launched by your editor
```

### Editor plugins

Ready-made clients that discover `latform-lsp` and wire up the Bmad filetype for
you:

| Editor  | Plugin                                                                      |
| ------- | --------------------------------------------------------------------------- |
| VS Code | [ken-lauer/latform-vscode](https://github.com/ken-lauer/latform-vscode)     |
| Vim     | [ken-lauer/vim-latform](https://github.com/ken-lauer/vim-latform)           |
| Neovim  | [ken-lauer/latform-lsp.nvim](https://github.com/ken-lauer/latform-lsp.nvim) |

Each still needs `latform-lsp` on `PATH` (`pip install "latform[lsp]"`); see the
plugin's README for install and configuration.

<details>
<summary>Manual LSP Configuration</summary>

### Manual setup

Any LSP client works — point it at the `latform-lsp` command over stdio,
associated with your Bmad files (`*.bmad`, `*.lat`, `tao.init`). Without one of
the plugins above:

**Neovim** — built-in [LSP client](https://neovim.io/doc/user/lsp.html) (0.11+):

```lua
vim.filetype.add({
  extension = { bmad = "bmad", lat = "bmad" },
  filename = { ["tao.init"] = "bmad" },
})

vim.lsp.config("latform", {
  cmd = { "latform-lsp" },
  filetypes = { "bmad" },
  root_markers = { "latform.toml", "pyproject.toml", "tao.init", ".git" },
})
vim.lsp.enable("latform")
```

**Vim** — with [`vim-lsp`](https://github.com/prabirshrestha/vim-lsp):

```vim
augroup latform_ft
  autocmd!
  autocmd BufRead,BufNewFile *.bmad,*.lat,tao.init setfiletype bmad
augroup END

if executable('latform-lsp')
  autocmd User lsp_setup call lsp#register_server({
    \ 'name': 'latform',
    \ 'cmd': {server_info->['latform-lsp']},
    \ 'allowlist': ['bmad'],
    \ })
endif
```

The log level and log file can be passed through any client (`--log-level`,
`--log-file`, or the `LATFORM_LSP_LOG_LEVEL` / `LATFORM_LSP_LOG_FILE` environment
variables). For other editors, see the LSP project's
[list of client implementations](https://microsoft.github.io/language-server-protocol/implementors/tools/).

</details>

## Using latform as a library

```python
from latform import parse, parse_file, parse_file_recursive

result = parse_file("my_lattice.bmad")
```

See the [Python API](https://ken-lauer.github.io/latform/api/) docs for details.

## Documentation

- [CLI reference](https://ken-lauer.github.io/latform/cli/) — full usage for every command
- [Configuration](https://ken-lauer.github.io/latform/configuration/) — `latform.toml` / `pyproject.toml`
- [Style guide](https://ken-lauer.github.io/latform/style_guide/) — the formatting rules latform applies
- [Python API](https://ken-lauer.github.io/latform/api/) — using latform programmatically
