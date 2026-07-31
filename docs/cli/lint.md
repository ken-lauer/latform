# latform-lint

Lint Bmad lattice files without reformatting them. This is the linting-focused
counterpart to [`latform --lint`](latform.md#linting): it prints any findings and
**exits non-zero when lints are reported**, which makes it suitable for use in CI
or pre-commit checks.

```
latform-lint [-h] [-r] [--combine] [-e]
             [--strict-references] [--ignore CODE]
             [-V] [-L {DEBUG,INFO,WARNING,CRITICAL}]
             filename [filename ...]
```

## Basic Usage

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

## Options

| Option                     | Default | Description                                                                                    |
| -------------------------- | ------- | ---------------------------------------------------------------------------------------------- |
| `--recursive`, `-r`        | off     | Recursively parse lattice files, following `call` statements                                   |
| `--combine`                | off     | Process all input files together as a single set, sharing one parse stack                      |
| `--error-if-missing`, `-e` | off     | Exit with an error if a file is missing during parsing                                         |
| `--strict-references`      | off     | Report references not defined in the loaded files (and unknown element types) as lint warnings |
| `--ignore CODE`            | none    | Suppress the given lint code(s); repeatable or comma-separated (e.g. `--ignore LF004,LF006`)   |

## Lint Codes

Each lint carries a stable code so it can be suppressed with `--ignore` (in
either `latform --lint` or `latform-lint`).

| Code    | Name                         | Description                                                                                        |
| ------- | ---------------------------- | -------------------------------------------------------------------------------------------------- |
| `LF001` | `unknown_statement`          | Statement type is unrecognized (may indicate a parsing error)                                      |
| `LF002` | `undefined_reference`        | `NAME[attr]` reference whose `NAME` is not defined (needs `--strict-references`)                   |
| `LF003` | `unknown_element_type`       | Element type is neither a known Bmad type nor a defined base element (needs `--strict-references`) |
| `LF004` | `unknown_attribute`          | Attribute is not valid for the element's (resolved) type                                           |
| `LF005` | `controller_all_zero_defaults` | An overlay/group/ramper's `var={...}` variables are all set to `0` (the Bmad implicit default)   |
| `LF006` | `duplicate_attribute`        | The same attribute is set more than once on a single element                                       |
| `LF007` | `unused_constant`            | A constant is defined but never referenced in any loaded file                                      |
| `LF008` | `attribute_override`         | A `name[attr] = value` statement overrides a value set in the element's definition (or repeats an earlier `name[attr]` setting) |
| `LF009` | `ambiguous_name`             | A defined name is a single easily-confused character (`i`/`l`/`o`) or shorter than the configured `min-name-length`             |
| `LF010` | `use_builtin_constant`       | A numeric literal in a constant or attribute value matches a built-in physical constant (or its negation) within `builtin-constant-rtol` |
| `LF011` | `tao_unknown_field`          | A Tao namelist assignment names a field not in the group's schema (or indexes a scalar, or uses `%` on a non-structure) |
| `LF012` | `tao_type_mismatch`          | A Tao namelist value is not valid for the field's declared type (e.g. an unquoted string, or a non-integer for an integer field) |
| `LF013` | `tao_index_out_of_bounds`    | A Tao namelist array subscript is outside the field's declared bounds |
| `LF014` | `tao_string_too_long`        | A Tao namelist string value is longer than the field's declared `character(N)` length (Fortran would silently truncate it) |

### Tao namelist checks (`LF011`–`LF014`)

These apply to Tao `*.init` namelist files. Assignments are validated against a
schema of the standard Tao namelist groups bundled with latform, across
`tao.init` and any split-out `data_file` / `var_file` / `plot_file` /
`beam_file` / `building_wall_file` sources named in `&tao_start`. Namelist groups
not in the schema (and files that are not a `tao.init`) are skipped.

Many of the conditions these detect are auto-fixable and are rewritten by the
formatter instead of merely reported — see
[Value normalization](index.md#value-normalization-tao-schema). Logical values
follow gfortran's permissive rule (optional leading `.`, then `T`/`F`, trailing
characters ignored), so `T`, `.true.`, `TRUE`, and `.T.` are all accepted.

Overriding an inherited attribute value (re-setting in a child element an
attribute its base element also sets) is allowed and is not flagged as a
duplicate.

`LF008` also understands element-set targets: `rfcavity::*[voltage] = 3.7` and
`q*[k1] = 1` lint against every matched element definition (`*` matches any run
of characters, `%` a single character, and `class::pattern` matches against the
element's resolved type). Ranges (`q1:q5`), branch qualifiers (`lat>>q1`),
instance counts (`q1##2`), and s-position selectors are not supported yet and
are skipped, as is overlap between different selectors (`q*` vs `q1`).
