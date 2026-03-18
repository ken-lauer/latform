# Python API

latform can also be used as a Python library for parsing, formatting, and
analyzing Bmad lattice files programmatically.

## Parsing

### parse

Parse a Bmad lattice string into a list of statements.

```python
import latform

statements = latform.parse("""
parameter[particle] = electron
Q1: quadrupole, L=0.5, k1=1.2
D1: drift, L=2.0
FODO: line = (Q1, D1)
use, FODO
""")

for st in statements:
    print(type(st).__name__, st)
```

### parse_file

Parse a single `.bmad` file from disk.

```python
statements = latform.parse_file("my_lattice.bmad")
```

### parse_file_recursive

Parse a lattice file and all files it references via `call` statements.
Returns a `Files` object containing parsed statements organized by filename.

```python
files = latform.parse_file_recursive("my_lattice.bmad")

# Iterate over all files and their statements
for filename, statements in files.by_filename.items():
    print(f"{filename}: {len(statements)} statements")

# Access the call graph
for caller, callees in files.filename_calls.items():
    for callee in callees:
        print(f"{caller} -> {callee}")
```

## Formatting

### format_statements

Format parsed statements back to a Bmad string using `FormatOptions`.

```python
from latform.output import format_statements
from latform.types import FormatOptions

statements = latform.parse_file("my_lattice.bmad")

options = FormatOptions(
    line_length=100,
    name_case="upper",
    kind_case="lower",
)

formatted = format_statements(statements, options)
print(formatted)
```

### format_file

A convenience function that parses and formats a file in one step.

```python
from latform.output import format_file
from latform.types import FormatOptions

formatted = format_file("my_lattice.bmad", FormatOptions())
```

### FormatOptions

All formatting behavior is controlled through the `FormatOptions` dataclass.
The defaults match the `latform` CLI defaults:

```python
from latform.types import FormatOptions

options = FormatOptions(
    line_length=100,             # target line length
    max_line_length=130,         # force multiline above this (default: 130% of line_length)
    compact=False,               # if True, no blank lines between statement types
    indent_size=2,               # spaces per indent level
    indent_char=" ",             # indentation character
    comment_col=40,              # column for inline comment alignment
    name_case="upper",           # element names: "upper", "lower", "same"
    attribute_case="lower",      # attribute names: "upper", "lower", "same"
    kind_case="lower",           # element types/keywords: "upper", "lower", "same"
    builtin_case="lower",        # builtin functions: "upper", "lower", "same"
    section_break_character="-", # character for section break lines
    section_break_width=None,    # width of section breaks (None = line_length)
    trailing_comma=False,        # trailing comma in multiline blocks
    renames={},                  # element rename mapping {"old": "new"}
    flatten_call=False,          # inline call statements
    flatten_inline=False,        # inline call:: arguments
    strip_comments=False,        # remove all comments
    newline_at_eof=True,         # ensure trailing newline
)
```

## Statement Types

Parsed statements are instances of these classes from `latform.statements`:

| Class        | Description           | Example                              |
| ------------ | --------------------- | ------------------------------------ |
| `Element`    | Element definition    | `Q1: quadrupole, L=0.5`              |
| `Line`       | Beamline definition   | `FODO: line = (Q1, D1)`              |
| `Constant`   | Constant assignment   | `K1_VAL = 1.5`                       |
| `Parameter`  | Bracketed parameter   | `parameter[particle] = electron`     |
| `Simple`     | Keyword statement     | `use, FODO` or `call, file=sub.bmad` |
| `Assignment` | General assignment    | `Q1[k1] = 0.5`                       |
| `Empty`      | Empty/whitespace-only |                                      |

## Working with Files

The `Files` class manages recursive parsing. `MemoryFiles` is a subclass
that starts from a string rather than a file on disk.

```python
from latform.parser import Files, MemoryFiles

# From disk
files = Files(main=pathlib.Path("my_lattice.bmad"))
files.parse(recurse=True)
files.annotate()

# From a string
files = MemoryFiles.from_contents(
    contents="Q1: quadrupole, L=0.5\n",
    root_path="/path/to/lattice_dir/virtual.bmad",
)
files.parse()
files.annotate()
```

## Diffing

Compare two parsed lattice files programmatically.

```python
from latform.parser import Files
from latform.diff import calculate_diff

files1 = Files(main=pathlib.Path("old_lattice.bmad"))
files1.parse()
files1.annotate()

files2 = Files(main=pathlib.Path("new_lattice.bmad"))
files2.parse()
files2.annotate()

diff = calculate_diff(files1, files2)

for p in diff.params_added:
    print(f"Added parameter: {p.name} = {p.new_value}")
for name, details in diff.eles_changed.items():
    print(f"Changed element: {name}")
    for attr, old, new in details.attrs_changed:
        print(f"  {attr}: {old} -> {new}")
```

## API Reference

::: latform
::: latform.output
::: latform.types
::: latform.parser
::: latform.diff
::: latform.statements
