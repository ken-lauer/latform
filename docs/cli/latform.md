# latform

`latform` is the main formatter command-line tool. It parses Bmad lattice files
and outputs consistently formatted code. Given a Tao `tao.init` file, it also
formats the lattices the file references and the `tao.init` itself (see
[Tao init files](#tao-init-files)).

```
latform [-h] [-i] [-o] [-r] [-R old,new] [--diff] [--compact]
        [--name-case {upper,lower,same}] [--kind-case {upper,lower,same}]
        [--builtin-case {upper,lower,same}]
        [-l LINE_LENGTH] [-m MAX_LINE_LENGTH]
        [--section-break-character CHAR] [--section-break-width WIDTH]
        [--flatten] [--flatten-call] [--flatten-inline]
        [--strip-comments] [--rename-file FILE]
        [--format {bmad,namelist}]
        [--lint] [--strict-references] [--ignore CODE]
        [--no-format-namelist] [--namelist-indent N]
        [--namelist-field-case {upper,lower,same}]
        [--no-namelist-align-equals] [--no-namelist-align-comments]
        [-v] [-V] [-L {DEBUG,INFO,WARNING,CRITICAL}]
        filename [filename ...]
```

## Basic Usage

Format a file and print to stdout:

```bash
latform my_lattice.bmad
```

Format in-place:

```bash
latform -i my_lattice.bmad
```

Format from stdin:

```bash
cat my_lattice.bmad | latform -
```

Format multiple files in-place:

```bash
latform -i *.bmad
```

## Formatting Options

| Option                      | Default                 | Description                                           |
| --------------------------- | ----------------------- | ----------------------------------------------------- |
| `--name-case`               | `upper`                 | Case for element names                                |
| `--kind-case`               | `lower`                 | Case for element types (keywords)                     |
| `--builtin-case`            | `lower`                 | Case for builtin functions                            |
| `--line-length`, `-l`       | `100`                   | Target line length                                    |
| `--max-line-length`, `-m`   | 130% of `--line-length` | Force multiline above this length                     |
| `--compact`                 | off                     | Compact mode (no blank lines between statement types) |
| `--section-break-character` | `-`                     | Character used in section break lines                 |
| `--section-break-width`     | same as `--line-length` | Width of section break lines                          |
| `--strip-comments`          | off                     | Remove all comments from output                       |

## Diff Mode

Show what the formatter would change without modifying the file:

```bash
latform --diff example_fodo.bmad
```

```diff
--- example_fodo.bmad
+++ example_fodo.bmad
@@ -1,25 +1,23 @@
-
 ! Simple FODO cell example
 ! This demonstrates basic Bmad syntax
-
 ! Define constants
 LQUAD = 0.6
 LDRIFT = 2.0
 K1_VAL = 1.5

 ! Define elements
-Q1: QUADRUPOLE, L=LQUAD, K1=K1_VAL
-Q2: QUADRUPOLE, L=LQUAD, K1=-K1_VAL
-D1: DRIFT, L=LDRIFT
+Q1: quadrupole, L=LQUAD, k1=K1_VAL
+Q2: quadrupole, L=LQUAD, k1=-K1_VAL
+D1: drift, L=LDRIFT

 ! Define a FODO cell
-CELL: LINE = (Q1, D1, Q2, D1)
+CELL: line = (Q1, D1, Q2, D1)

 ! Build a ring from 8 cells
-RING: LINE = (8*CELL)
+RING: line = (8*CELL)

 ! Optional: modify all quads
-Q*[TILT] = 0.0
+Q*[tilt] = 0.0

 ! Use the ring
-USE, RING
+use, RING
```

## Renaming Elements

Rename elements in the output:

```bash
latform -R 'Q1,QF' -R 'Q2,QD' example_fodo.bmad
```

```
QF: quadrupole, L=LQUAD, k1=K1_VAL
QD: quadrupole, L=LQUAD, k1=-K1_VAL
...
CELL: line = (QF, D1, QD, D1)
```

Renames can also be loaded from a CSV file:

```bash
latform --rename-file renames.csv my_lattice.bmad
```

Where `renames.csv` contains one `old,new` pair per line.

## Recursive Parsing and Flattening

Parse lattice files recursively, following `call` statements:

```bash
latform -r parse_test.bmad
```

Flatten all called files into a single output:

```bash
latform --flatten parse_test.bmad
```

`--flatten` implies both `--flatten-call` (inline call statements) and
`--flatten-inline` (inline `call::` arguments). These can also be used
independently.

## Linting

By default `latform` only reformats. Pass `--lint` to also report lint warnings
(unknown attributes, duplicate attributes, and so on) alongside the formatted
output:

```bash
latform --lint my_lattice.bmad
```

For linting without reformatting — for example in CI — use the dedicated
[`latform-lint`](lint.md) command instead, which exits non-zero when any
findings are reported.

`--strict-references` additionally treats element/constant references that are
not defined in the loaded files as lint warnings (it implies `--lint`). By
default such references are assumed to be defined elsewhere. Suppress individual
lints by code with `--ignore`, e.g. `--ignore LF004` (repeatable, or
comma-separated: `--ignore LF004,LF006`).

When the input is a `tao.init`, `--lint` also validates its namelist assignments
against the bundled Tao schema — unknown fields, type mismatches, out-of-bounds
indices, and over-length strings (`LF011`–`LF014`). See
[Lint Codes](lint.md#lint-codes) for the full list.

## Tao init files

When the input is a Tao `tao.init` file, `latform` expands it: it formats each
Bmad lattice file the init references (via `design_lattice`) **and** the
`tao.init` namelist file itself. The input is recognized as a namelist when it is
named `*.init` or its contents look like one; pass `--format namelist` to force
it for a differently-named file (or `--format bmad` to opt out). See
[Detecting namelist inputs](index.md#detecting-namelist-inputs).

```bash
latform tao.init            # print the formatted lattices and the tao.init
latform -i tao.init         # rewrite the lattices and the tao.init in place
latform --diff tao.init     # preview the changes to all of them
```

Reformatting the namelist also **normalizes its values** against the bundled Tao
schema — quoting bare strings, mapping enum indices to names, and canonicalizing
logicals to `T`/`F` (see
[Value normalization](index.md#value-normalization-tao-schema)). Layout and
alignment are controlled by the shared
[namelist formatting flags](index.md#namelist-taoinit-formatting); pass
`--no-format-namelist` to leave the `tao.init` untouched, values and all (the
referenced Bmad lattices are still formatted):

```bash
latform -i tao.init --no-namelist-align-equals  # don't align '=' in the tao.init
latform -i tao.init --no-format-namelist        # lattices only; init left verbatim
```

Because a `tao.init` expands to several top-level files, writing them all to a
single `-o`/`--output` target is ambiguous — use `-i`/`--in-place` instead.
