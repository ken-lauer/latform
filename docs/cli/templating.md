# Templating

Tools for turning template lattice/`tao.init` files into concrete outputs:
`latform-apply` for a single file, `latform-template` for a whole set across
instances.

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
              [--parts DELIMS FROM TO] [--delimiters CHARS]
              [--no-format-namelist] [--namelist-indent N]
              [--namelist-field-case {upper,lower,same}]
              [--no-namelist-align-equals] [--no-namelist-align-comments]
              [-o OUT | -i]
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

The output namelist is **reformatted by default** (field indentation, lowercase
field names, aligned `=` and comments, and a blank line after each group).
`latform-apply` changes layout only — unlike `latform`/`latform-template` it does
not normalize values. Pass `--no-format-namelist` to preserve the source layout,
or tune the result with the
[namelist formatting flags](index.md#namelist-taoinit-formatting):

```bash
latform-apply tao.init --no-format-namelist -i          # edit values, keep layout
latform-apply tao.init --no-namelist-align-equals -i    # reformat but don't align '='
```

## latform-template

Expand a Bmad lattice template **set** across several instances, writing files
under `--output-dir` (default: current directory). A YAML sidecar lists the
template files and per-instance values, renames, and output paths. Use
`--dry-run` to list the files that would be written without writing them.

```
latform-template INSTANCES.yaml [-d OUTPUT_DIR] [--dry-run]
                 [--no-format-namelist] [--namelist-indent N]
                 [--namelist-field-case {upper,lower,same}]
                 [--no-namelist-align-equals] [--no-namelist-align-comments]
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

### Tao init (`tao_init:`)

An optional top-level `tao_init:` key renders a Tao `tao.init` per instance. It
takes an `input` (the template `tao.init`) and an `output` path (which may use
`{instance}`). The `design_lattice` file entries are rewritten to the instance's
generated lattice paths — the same rewriting applied to `call` targets.

```yaml
tao_init:
  input: tao.init
  output: "{instance}/tao.init"

instances:
  c1: {}
  c2:
    tao_init:
      namelists: # add/update namelist sections for this instance
        tao_params:
          global%n_opti_cycles: 50 # updated in place
        tao_beam_init:
          beam_init%n_particle: 5000 # group appended if absent
```

A per-instance `tao_init.namelists` block adds or updates namelist sections:
each entry is a `{key: value}` map, values interpolate `{instance}`, and a
`name#N` suffix (1-based) targets the N-th of a repeated group.

The emitted `tao.init` is **reformatted by default**, and its values are
normalized against the bundled Tao schema — strings quoted, enum indices mapped
to names, and logicals canonicalized to `T`/`F` (see
[Value normalization](index.md#value-normalization-tao-schema)). This is
controlled by the shared
[namelist formatting flags](index.md#namelist-taoinit-formatting) (e.g.
`--no-format-namelist` to keep the template's layout and values). The generated
Bmad lattices are always reformatted regardless.
