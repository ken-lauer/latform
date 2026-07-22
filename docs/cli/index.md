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

## Namelist (tao.init) formatting

`latform`, `latform-apply`, and `latform-template` can reformat Fortran-namelist
files (`*.init` / `*.nml`, e.g. a Tao `tao.init`). Reformatting is **on by
default**; the same flags control it on every command:

| Behavior                    | Flag                                       | Default |
| --------------------------- | ------------------------------------------ | ------- |
| Reformat the namelist       | `--no-format-namelist` disables            | on      |
| Field indent width          | `--namelist-indent N`                      | `2`     |
| Field-name case             | `--namelist-field-case {upper,lower,same}` | `lower` |
| Align `=` into a column     | `--namelist-align-equals`                  | off     |
| Align trailing `!` comments | `--no-namelist-align-comments` disables    | on      |

Only the field section between a `&name` opener and its `/` terminator is
affected:

- the opener and terminator stay at column zero, and **values are never
  modified** (only field-name case and whitespace around `=` change);
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

the default reformat (lowercase field names, aligned comments, 2-space indent,
blank line after each group) produces:

```
&Tao_Params
  global%n_opti_cycles = 100  ! cycles
  global%plot_on = T          ! plotting
  x = 1
/

&tao_beam_init
  beam_init%n_particle = 5000
/
```

Add `--namelist-align-equals` to also line up the `=` within each run:

```
&Tao_Params
  global%n_opti_cycles = 100  ! cycles
  global%plot_on       = T    ! plotting
  x                    = 1
/
```

**Note:** these options are command-line flags only; they cannot yet be set in a
`latform.toml` `[format]` table. See [Configuration](../configuration.md).
