# Inspection & comparison

Read-only tools for extracting information from lattices, comparing them, and
visualizing their structure.

## latform-dump

Extract and report parameters, constants, used elements, and unused elements
from lattice files.

```
latform-dump [-h] [-p] [-c] [-U] [-u] [-f]
             [-m MATCH] [-r MATCH_RE] [-d DELIMITER] [--combine]
             [-v] [-V] [-L {DEBUG,INFO,WARNING,CRITICAL}]
             filename [filename ...]
```

### Basic Usage

With no flags, all categories are shown:

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

--- Constants ---
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Name   ┃ Expression ┃ Location            ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ LQUAD  │ 0.6        │ example_fodo.bmad:5 │
│ LDRIFT │ 2.0        │ example_fodo.bmad:6 │
│ K1_VAL │ 1.5        │ example_fodo.bmad:7 │
└────────┴────────────┴─────────────────────┘

--- Used Elements ---
┏━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name ┃ Type       ┃ Parent ┃ Reason        ┃ Location             ┃
┡━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Q1   │ QUADRUPOLE │        │ in line CELL  │ example_fodo.bmad:10 │
│ Q2   │ QUADRUPOLE │        │ in line CELL  │ example_fodo.bmad:11 │
│ D1   │ DRIFT      │        │ in line CELL  │ example_fodo.bmad:12 │
│ CELL │ LINE       │        │ in line RING  │ example_fodo.bmad:15 │
│ RING │ LINE       │        │ use statement │ example_fodo.bmad:18 │
└──────┴────────────┴────────┴───────────────┴──────────────────────┘

--- Unused Elements ---
```

### Usage resolution

Whether an element is "used" is determined by expanding the lattice from the
last `use` statement (each of its arguments is a branch root):

- Lines are expanded recursively, including repetitions (`8*CELL`),
  reflections (`-SUB`), replacement-line calls (`SUB(X)`), and `list` members.
- Superimposed elements are used when superposition is enabled
  (`superimpose` / `superimpose = T`, also via a later `name[superimpose] = T`)
  and their `ref` matches a used element; with no `ref`, the superposition is
  relative to the beginning of the lattice and always counts as used.
- Controllers (overlay/group/ramper/girder) are used when at least one of
  their slaves is used.
- Base elements of used elements are used (`QD: QF` marks `QF` used).
- `fork` / `photon_fork` elements pull in their `to_line` / `to_element`
  targets.

The Reason column shows which of these applied.

### Selective Output

Show only specific categories:

```bash
latform-dump -p example_fodo.bmad   # parameters only
latform-dump -c example_fodo.bmad   # constants only
latform-dump -U example_fodo.bmad   # used elements only
latform-dump -u example_fodo.bmad   # unused elements only
latform-dump -f example_fodo.bmad   # loaded files only
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
Name,Type,Parent,Reason,Location
Q1,QUADRUPOLE,,in line CELL,example_fodo.bmad:10
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
