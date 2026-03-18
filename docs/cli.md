# CLI Reference

latform provides several command-line tools for working with Bmad lattice files.

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
