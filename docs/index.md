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

- [**latform**](cli.md#latform) -- Format Bmad lattice files
- [**latform-dump**](cli.md#latform-dump) -- Extract parameters and element information
- [**latform-diff**](cli.md#latform-diff) -- Compare two lattice files
- [**latform-gitdiff**](cli.md#latform-gitdiff) -- Compare a lattice file across git revisions
- [**latform-graph**](cli.md#latform-graph) -- Visualize file dependency trees

## Documentation

- [CLI Reference](cli.md) -- Full command-line usage for all tools
- [Python API](api.md) -- Using latform as a library
- [Style Guide](style_guide.md) -- The formatting rules latform applies
