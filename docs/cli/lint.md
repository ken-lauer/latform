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
