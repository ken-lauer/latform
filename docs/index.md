# latform Documentation

latform is a Bmad lattice parser/formatter tool for parsing, formatting,
analyzing, and comparing particle accelerator lattice files.

## Installation

```bash
pip install latform
```

For development:

```bash
pip install -e ".[test]"
```

## Quick Start

Format a lattice file and print to stdout:

```bash
latform my_lattice.bmad
```

Format in-place:

```bash
latform -i my_lattice.bmad
```

Preview what the formatter would change:

```bash
latform --diff my_lattice.bmad
```

## Tools

latform provides these CLI commands:

- [**latform**](cli/latform.md) -- Format Bmad lattice files (and `tao.init` namelists)
- [**latform-lint**](cli/lint.md) -- Lint lattice files without reformatting
- [**latform-apply**](cli/templating.md#latform-apply) -- Apply overrides/renames to a template file
- [**latform-template**](cli/templating.md#latform-template) -- Expand a template set across instances
- [**latform-dump**](cli/inspection.md#latform-dump) -- Extract parameters and element information
- [**latform-diff**](cli/inspection.md#latform-diff) -- Compare two lattice files
- [**latform-gitdiff**](cli/inspection.md#latform-gitdiff) -- Compare a lattice file across git revisions
- [**latform-graph**](cli/inspection.md#latform-graph) -- Visualize file dependency trees
- [**latform-lsp**](cli/lsp.md) -- Language server for editor integration (optional `lsp` extra)

## Documentation

- [CLI Reference](cli/index.md) -- Full command-line usage for all tools
- [Configuration](configuration.md) -- Project settings via latform.toml / pyproject.toml
- [Python API](api.md) -- Using latform as a library
- [Style Guide](style_guide.md) -- The formatting rules latform applies
