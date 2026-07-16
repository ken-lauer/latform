# CLI Reference

latform provides several command-line tools for working with Bmad lattice files.

See [Configuration](#configuration) for project-wide settings via
`latform.toml` / `pyproject.toml`.

## latform

`latform` is the main formatter command-line tool. It parses Bmad lattice files
and outputs consistently formatted code.

```
latform [-h] [-i] [-o] [-r] [-R old,new] [--diff] [--compact]
        [--name-case {upper,lower,same}] [--kind-case {upper,lower,same}]
        [--builtin-case {upper,lower,same}]
        [-l LINE_LENGTH] [-m MAX_LINE_LENGTH]
        [--section-break-character CHAR] [--section-break-width WIDTH]
        [--flatten] [--flatten-call] [--flatten-inline]
        [--strip-comments] [--rename-file FILE]
        [--lint] [--strict-references] [--ignore CODE]
        [-v] [-V] [-L {DEBUG,INFO,WARNING,CRITICAL}]
        filename [filename ...]
```

### Basic Usage

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

### Formatting Options

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

### Diff Mode

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

### Renaming Elements

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

### Recursive Parsing and Flattening

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

### Linting

By default `latform` only reformats. Pass `--lint` to also report lint warnings
(unknown attributes, duplicate attributes, and so on) alongside the formatted
output:

```bash
latform --lint my_lattice.bmad
```

For linting without reformatting — for example in CI — use the dedicated
[`latform-lint`](#latform-lint) command instead, which exits non-zero when any
findings are reported.

`--strict-references` additionally treats element/constant references that are
not defined in the loaded files as lint warnings (it implies `--lint`). By
default such references are assumed to be defined elsewhere. Suppress individual
lints by code with `--ignore`, e.g. `--ignore LF004` (repeatable, or
comma-separated: `--ignore LF004,LF006`).

See [Lint Codes](#lint-codes) for the full list.

---

## latform-lint

Lint Bmad lattice files without reformatting them. This is the linting-focused
counterpart to `latform --lint`: it prints any findings and **exits non-zero
when lints are reported**, which makes it suitable for use in CI or pre-commit
checks.

```
latform-lint [-h] [-r] [--combine] [-e]
             [--strict-references] [--ignore CODE]
             [-V] [-L {DEBUG,INFO,WARNING,CRITICAL}]
             filename [filename ...]
```

### Basic Usage

Lint one or more files:

```bash
latform-lint my_lattice.bmad
```

Lint from stdin:

```bash
cat my_lattice.bmad | latform-lint -
```

The command exits with status `1` if any lints are found and `0` otherwise:

```bash
latform-lint my_lattice.bmad && echo "clean"
```

### Options

| Option                     | Default | Description                                                                                    |
| -------------------------- | ------- | ---------------------------------------------------------------------------------------------- |
| `--recursive`, `-r`        | off     | Recursively parse lattice files, following `call` statements                                   |
| `--combine`                | off     | Process all input files together as a single set, sharing one parse stack                      |
| `--error-if-missing`, `-e` | off     | Exit with an error if a file is missing during parsing                                         |
| `--strict-references`      | off     | Report references not defined in the loaded files (and unknown element types) as lint warnings |
| `--ignore CODE`            | none    | Suppress the given lint code(s); repeatable or comma-separated (e.g. `--ignore LF004,LF006`)   |

### Lint Codes

Each lint carries a stable code so it can be suppressed with `--ignore` (in
either `latform --lint` or `latform-lint`).

| Code    | Name                         | Description                                                                                        |
| ------- | ---------------------------- | -------------------------------------------------------------------------------------------------- |
| `LF001` | `unknown_statement`          | Statement type is unrecognized (may indicate a parsing error)                                      |
| `LF002` | `undefined_reference`        | `NAME[attr]` reference whose `NAME` is not defined (needs `--strict-references`)                   |
| `LF003` | `unknown_element_type`       | Element type is neither a known Bmad type nor a defined base element (needs `--strict-references`) |
| `LF004` | `unknown_attribute`          | Attribute is not valid for the element's (resolved) type                                           |
| `LF005` | `controller_default_missing` | An overlay/group/ramper `var={...}` variable has no default value set                              |
| `LF006` | `duplicate_attribute`        | The same attribute is set more than once on a single element                                       |
| `LF007` | `unused_constant`            | A constant is defined but never referenced in any loaded file                                      |
| `LF008` | `attribute_override`         | A `name[attr] = value` statement overrides a value set in the element's definition (or repeats an earlier `name[attr]` setting) |
| `LF009` | `ambiguous_name`             | A defined name is a single easily-confused character (`i`/`l`/`o`) or shorter than the configured `min-name-length`             |
| `LF010` | `use_builtin_constant`       | A numeric literal in a constant or attribute value matches a built-in physical constant (or its negation) within `builtin-constant-rtol` |

Overriding an inherited attribute value (re-setting in a child element an
attribute its base element also sets) is allowed and is not flagged as a
duplicate.

`LF008` also understands element-set targets: `rfcavity::*[voltage] = 3.7` and
`q*[k1] = 1` lint against every matched element definition (`*` matches any run
of characters, `%` a single character, and `class::pattern` matches against the
element's resolved type). Ranges (`q1:q5`), branch qualifiers (`lat>>q1`),
instance counts (`q1##2`), and s-position selectors are not supported yet and
are skipped, as is overlap between different selectors (`q*` vs `q1`).

---

## latform-dump

Extract and report parameters, used elements, and unused elements from lattice files.

```
latform-dump [-h] [-p] [-U] [-u]
             [-m MATCH] [-r MATCH_RE] [-d DELIMITER]
             [-v] [-V] [-L {DEBUG,INFO,WARNING,CRITICAL}]
             filename [filename ...]
```

### Basic Usage

With no flags, all three categories are shown:

```bash
latform-dump example_fodo.bmad
```

```
--- Parameters ---
┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name     ┃ Expression ┃ Location             ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Q*[tilt] │ 0.0        │ example_fodo.bmad:21 │
└──────────┴────────────┴──────────────────────┘

--- Used Elements ---
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name      ┃ Type          ┃ Parent ┃ Location             ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ RING      │ LINE          │        │ example_fodo.bmad:18 │
│ BEGINNING │ BEGINNING_ELE │        │ <implicit>:0         │
│ END       │ MARKER        │        │ <implicit>:0         │
└───────────┴───────────────┴────────┴──────────────────────┘

--- Unused Elements ---
┏━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name ┃ Type       ┃ Location             ┃
┡━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Q1   │ QUADRUPOLE │ example_fodo.bmad:10 │
│ Q2   │ QUADRUPOLE │ example_fodo.bmad:11 │
│ D1   │ DRIFT      │ example_fodo.bmad:12 │
│ CELL │ LINE       │ example_fodo.bmad:15 │
└──────┴────────────┴──────────────────────┘
```

### Selective Output

Show only specific categories:

```bash
latform-dump -p example_fodo.bmad   # parameters only
latform-dump -U example_fodo.bmad   # used elements only
latform-dump -u example_fodo.bmad   # unused elements only
```

### Filtering

Filter results by glob or regex pattern:

```bash
latform-dump -m 'Q*' example_fodo.bmad
latform-dump -r 'Q[0-9]+' example_fodo.bmad
```

### CSV / Machine-Readable Output

Use a delimiter for CSV-style output:

```bash
latform-dump -d ',' example_fodo.bmad
```

```
Name,Expression,Location
Q*[tilt],0.0,example_fodo.bmad:21
Name,Type,Parent,Location
RING,LINE,,example_fodo.bmad:18
...
```

---

## latform-diff

Compare two lattice files structurally. Reports differences in parameters and
elements (added, removed, changed, renamed).

```
latform-diff [-h] [-v] file1 file2
```

### Example

```bash
latform-diff fodo.bmad example_fodo.bmad
```

```
────────────────────────────────── Parameters ──────────────────────────────────
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ State   ┃ Target    ┃ Name       ┃ Value (Left)           ┃ Value (Right) ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Added   │           │ k1_val     │                        │ 1.5           │
│ Added   │           │ ldrift     │                        │ 2.0           │
│ Removed │           │ k1_optimal │ (1/LQ)*2*sqrt(2)/L_TOT │               │
│ Removed │           │ l_tot      │ 2                      │               │
│ ...     │           │            │                        │               │
└─────────┴───────────┴────────────┴────────────────────────┴───────────────┘

─────────────────────────────────── Elements ───────────────────────────────────
┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ State   ┃ Element ┃ Property/Attribute ┃ Value (Left) ┃ Value (Right) ┃
┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Added   │ CELL    │ Element            │              │ Exist         │
│ Added   │ D1      │ Element            │              │ Exist         │
│ Changed │ Q1      │ Type               │ Q0           │ QUADRUPOLE    │
│ Changed │ Q1      │ Attr: k1           │              │ K1_VAL        │
│ ...     │         │                    │              │               │
└─────────┴─────────┴────────────────────┴──────────────┴───────────────┘
```

---

## latform-gitdiff

Compare a lattice file across two git revisions. Defaults to comparing
against `HEAD`.

```
latform-gitdiff [-h] [-v] lattice_file rev1 [rev2]
```

### Examples

Compare a file between two commits:

```bash
latform-gitdiff my_lattice.bmad abc123 def456
```

Compare a file at a specific commit against HEAD:

```bash
latform-gitdiff my_lattice.bmad abc123
```

The output format is identical to `latform-diff`.

---

## latform-graph

Visualize the file dependency tree of a lattice (following `call` statements).

```
latform-graph [-h] [-o OUTPUT] [-f {text,mermaid}]
              [-v] [-V] [-L {DEBUG,INFO,WARNING,CRITICAL}]
              filename [filename ...]
```

### Text Output (default)

```bash
latform-graph parse_test.bmad
```

```
parse_test.bmad
└── sub_dir/sub.bmad
    └── sub2_dir/sub2.bmad
```

### Mermaid Output

```bash
latform-graph -f mermaid parse_test.bmad
```

```
graph LR
    parse_test_bmad["parse_test.bmad"] --> sub_dir_sub_bmad["sub_dir/sub.bmad"]
    sub_dir_sub_bmad["sub_dir/sub.bmad"] --> sub2_dir_sub2_bmad["sub2_dir/sub2.bmad"]
```

### Write to File

```bash
latform-graph -o deps.txt parse_test.bmad
latform-graph -f mermaid -o deps.mmd parse_test.bmad
```

## latform-apply

Apply value overrides and/or renames to a **single** template file and write the
result to stdout (`-o FILE`, or `-i`/`--in-place` to rewrite the input file). The
file may be Bmad or a Fortran-namelist file (`*.init` / `*.nml`); the format is
auto-detected from the extension and can be forced with `--format {bmad,namelist}`.
Unlike jinja/cookiecutter, a Bmad template is itself valid Bmad — it parses, lints,
and formats standalone — while a YAML sidecar supplies per-element values.

```
latform-apply TEMPLATE [--format {bmad,namelist}] [--values VALUES.yaml]
              [--set NAMELIST KEY VALUE]
              [--rename OLD NEW] [--prefix FROM TO] [--suffix FROM TO]
              [--parts DELIMS FROM TO] [--delimiters CHARS] [-o OUT | -i]
```

Given `quad.bmad`:

```
Q1: quadrupole, L=0.3, k1=0.0
```

and `values.yaml`:

```yaml
Q1:
  k1: 1.523 # override an attribute
# "/.*_BPM/": { type: BPM_TYPE }   # regex key: every matching element
# BEN0: { type: null }             # null removes an attribute
```

```bash
latform-apply quad.bmad --values values.yaml
# -> Q1: quadrupole, L=0.3, k1=1.523

# --values - reads the overrides (YAML or JSON) from stdin, for programmatic use
echo '{"Q1": {"k1": 1.523}}' | latform-apply quad.bmad --values -
```

`--rename` is repeatable and accepts literal or regex rules:

```bash
latform-apply quad.bmad --rename 'Q(\d+)' 'ARC_Q\1'
# -> ARC_Q1: quadrupole, L=0.3, k1=0.0
```

`--prefix`, `--suffix`, and `--parts` are the named rename forms (all repeatable):

```bash
# prefix: leading FROM only, bounded by . or _  (CX_A -> C1_A; O_CX_A untouched)
latform-apply lat.bmad --prefix CX C1

# suffix: trailing FROM, bounded by . or _  (Q_XCR -> Q_HCOR)
latform-apply lat.bmad --suffix _XCR _HCOR

# parts: rename whole segments anywhere, split on the given delimiters
#        (CX_A -> C1_A and O_CX_A -> O_C1_A)
latform-apply lat.bmad --parts ._ CX C1

# --delimiters sets the boundary chars for --prefix/--suffix (default . and _)
latform-apply lat.bmad --prefix CX C1 --delimiters _
```

### Namelist files

For a namelist file (`*.init` / `*.nml`, e.g. a Tao `tao.init`), overrides are
keyed by **namelist group** instead of by element. Renames do not apply. Each
group maps to a `{key: value}` block; existing keys are updated in place, missing
keys are appended, and a `null` value removes a key. A `name#N` suffix (1-based)
targets the N-th of a repeated group.

`values.yaml`:

```yaml
tao_params:
  global%n_opti_cycles: 50 # update in place
  global%plot_on: null # remove the key
tao_beam_init:
  beam_init%n_particle: 5000 # group appended if absent
```

```bash
latform-apply tao.init --values values.yaml

# --set NAMELIST KEY VALUE is the inline, repeatable equivalent (and wins over
# --values); -i rewrites tao.init in place instead of printing to stdout.
latform-apply tao.init --set tao_params global%n_opti_cycles 50 -i
```

## latform-template

Expand a Bmad lattice template **set** across several instances, writing files
under `--output-dir` (default: current directory). A YAML sidecar lists the
template files and per-instance values, renames, and output paths. Use
`--dry-run` to list the files that would be written without writing them.

```
latform-template INSTANCES.yaml [-d OUTPUT_DIR] [--dry-run]
```

`instances.yaml` (paths are relative to the file's own directory):

```yaml
template:
  - input: cx.bmad
    output: "{instance}/{instance}.bmad" # {instance} -> c1, c2, ...
  - input: cx.cor.bmad
    output: "{instance}/{instance}.cor.bmad"

renames:
  "CX([_.].*|$)": "{instance:upper}\\1" # flat shortcut: literal unless it has * + ?

instances:
  c1: {}
  c2:
    values: { CX_LINE_ROT: pi/2 } # per-instance overrides
```

`renames` also accepts a **structured** form with any of `prefix`, `suffix`,
`regex`, and `parts`, each usable globally (in place of the flat block above) or
per instance. They are typically used one at a time — pick the form that fits.
`{instance}`, `{instance:upper}`, and `{instance:lower}` interpolate in every
replacement.

#### Prefix / suffix renames

`prefix` renames a leading `FROM` bounded by a delimiter (default `.` or `_`);
`suffix` is the mirror for a trailing `FROM`. Both are `{from: to}` maps.

```yaml
renames:
  prefix:
    CX: "{instance:upper}" # CX_BEN0 -> C1_BEN0, CX.COL00 -> C1.COL00, bare CX -> C1
    O_CX: "O_{instance:upper}" # embedded needs its own entry (prefix is leading-only)
  suffix:
    _XCR: _HCOR # A_XCR -> A_HCOR (trailing, bounded)
```

#### Regex renames

`regex` is the escape hatch: raw `re.sub` per name, so you write your own
backreferences. (Equivalent to the flat shortcut when the key contains `* + ?`.)

```yaml
renames:
  regex:
    "CX([_.].*|$)": "{instance:upper}\\1"
```

#### Parts renames

`parts` renames whole delimiter-separated segments *anywhere* in a name — the one
form that also rewrites embedded segments (e.g. `O_CX_BEN`), so a single rule
covers leading, embedded, dotted, and bare occurrences.

```yaml
renames:
  parts:
    - delimiters: "._" # a string "._" or a list [".", "_"]
      from: CX
      to: "{instance:upper}"
```

Set a top-level `delimiters` to change the default delimiter set for `prefix`,
`suffix`, and `parts`. With a default in place, `parts` may be written as a plain
`{from: to}` map, matching the shape of `prefix`/`suffix`:

```yaml
delimiters: "._" # applies to prefix / suffix / parts (default: . and _)

renames:
  parts:
    CX: "{instance:upper}" # uses the top-level delimiters
```

Per name, the first matching rule wins in the order
`literal → regex → prefix → suffix → parts`. A plain `{from: to}` map with no
`prefix`/`suffix`/`regex`/`parts` key is the flat shortcut shown earlier (today's
literal-or-regex behavior).

```bash
latform-template instances.yaml -d build/
# wrote: build/c1/c1.bmad
# wrote: build/c1/c1.cor.bmad
# wrote: build/c2/c2.bmad
# ...

latform-template instances.yaml -d build/ --dry-run
# would write: build/c1/c1.bmad
# ...
```

Files that the template `call`s but are not in `template` (e.g. shared
`settings/`) can be listed under a top-level `context:` key to be loaded for
name resolution only — they are never written, and `call`s to them are left
untouched. `call`s between transform-set files are rewritten to the instance
outputs automatically.

---

## Configuration

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
