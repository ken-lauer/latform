# CLI Overview

latform provides several command-line tools for working with Bmad lattice files
and Tao `tao.init` namelist files.

See [Configuration](../configuration.md) for project-wide settings via
`latform.toml` / `pyproject.toml`.

## Tools at a glance

| Command                                              | Purpose                                                             |
| ---------------------------------------------------- | ------------------------------------------------------------------- |
| [`latform`](latform.md)                              | Parse and format Bmad lattice files (and `tao.init` namelists)      |
| [`latform-lint`](lint.md)                            | Lint lattice files without reformatting (non-zero exit on findings) |
| [`latform-apply`](templating.md#latform-apply)       | Apply value overrides / renames to a single template file           |
| [`latform-template`](templating.md#latform-template) | Expand a template set across instances                              |
| [`latform-dump`](inspection.md#latform-dump)         | Extract parameters and element information                          |
| [`latform-diff`](inspection.md#latform-diff)         | Compare two lattice files                                           |
| [`latform-gitdiff`](inspection.md#latform-gitdiff)   | Compare a lattice file across git revisions                         |
| [`latform-graph`](inspection.md#latform-graph)       | Visualize file dependency trees                                     |
| [`latform-lsp`](lsp.md)                              | Language server for editor integration (optional `lsp` extra)       |

## Namelist (tao.init) formatting

`latform`, `latform-apply`, and `latform-template` can reformat Fortran-namelist
files (`*.init` / `*.nml`, e.g. a Tao `tao.init`). Reformatting is **on by
default**; the same flags control it on every command:

| Behavior                    | Flag                                       | Default |
| --------------------------- | ------------------------------------------ | ------- |
| Reformat the namelist       | `--no-format-namelist` disables            | on      |
| Field indent width          | `--namelist-indent N`                      | `2`     |
| Field-name case             | `--namelist-field-case {upper,lower,same}` | `lower` |
| Align `=` into a column     | `--no-namelist-align-equals` disables      | on      |
| Align trailing `!` comments | `--no-namelist-align-comments` disables    | on      |

!!! note "Default change"

    `=` alignment is now **on by default** (previously opt-in via a
    `--namelist-align-equals` flag). Use `--no-namelist-align-equals` to restore
    the unaligned layout.

Only the field section between a `&name` opener and its `/` terminator is
affected:

- the opener and terminator stay at column zero; field-name case and the
  whitespace around `=` are normalized;
- for Tao namelist groups in the bundled schema, `latform` and
  `latform-template` also normalize **values** (see
  [Value normalization](#value-normalization-tao-schema) below);
  `latform-apply` changes layout only;
- a single blank line is enforced after each group's `/`;
- alignment is scoped to **contiguous runs** — it resets at each blank line, so
  one long field never pushes an unrelated block.

Given a messy `tao.init`:

```
&Tao_Params
   Global%N_Opti_Cycles = 100   ! cycles
  global%plot_on=T ! plotting
     x = 1
/
&tao_beam_init
  beam_init%n_particle = 5000
/
```

the default reformat (lowercase field names, aligned `=` and comments, 2-space
indent, blank line after each group) produces:

```
&Tao_Params
  global%n_opti_cycles = 100  ! cycles
  global%plot_on       = T    ! plotting
  x                    = 1
/

&tao_beam_init
  beam_init%n_particle = 5000
/
```

Pass `--no-namelist-align-equals` to leave the `=` unaligned (a single space on
each side):

```
&Tao_Params
  global%n_opti_cycles = 100  ! cycles
  global%plot_on = T          ! plotting
  x = 1
/
```

**Note:** these options can also be set in a `latform.toml` `[format]` table
(read by `latform`), using the same names without the leading `--` — e.g.
`namelist-field-case`, `namelist-align-equals`, `namelist-logicals`. See
[Configuration](../configuration.md#namelist-formatting-settings).

## Value normalization (Tao schema)

`latform` and `latform-template` know the types of the fields in the standard
Tao namelist groups (from a schema bundled with latform) and normalize field
**values** as part of reformatting a `tao.init`:

- **String quoting** — a character value written unquoted is quoted:
  `plot_file = tao_plot.init` → `plot_file = 'tao_plot.init'`.
- **Enum index → name** — a field whose value is a named enum accepts either the
  name or its integer index; the index (bare or quoted) is rewritten to the
  name: `curve(1)%line%color = 2` → `'red'`,
  `curve(1)%symbol%type = '1'` → `'dot'`. Covers colors, line patterns, symbol
  types, and fill patterns.
- **Logicals** — canonicalized to `T` / `F`: `.true.`/`TRUE`/`t` → `T` and
  `.false.`/`F`/`f` → `F`. (The true/false tokens are configurable through the
  Python API; the CLI uses `T`/`F`.)

Only groups and fields present in the schema are touched; unknown namelists,
unknown fields, and already-canonical values are left as-is. Values written in
the positional/anonymous field form (`datum(4) = 'a' '' '' 'LA.Mar.MID' ...`)
are not normalized yet. Normalization turns off together with layout
reformatting via `--no-format-namelist`, and `latform-apply` reformats layout
only (it does not normalize values).

The same rules power the Tao namelist lints (`LF011`–`LF014`); see
[Lint Codes](lint.md#lint-codes). Anything the formatter cannot safely rewrite
(a bad field name, an out-of-range index, an over-length string) is reported
there instead.
