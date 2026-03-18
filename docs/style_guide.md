# Style Guide

This documents the default formatting style applied by `latform`. All examples
show real output from the formatter with default settings:

- Line length: 100
- Names: `UPPER`, kinds/keywords: `lower`, attributes: `lower`, Bmad builtins: `lower`
- Indent: 2 spaces
- Section break character: `-`

## Case Normalization

latform normalizes the case of identifiers. By default, element **names** are
uppercased, while element **types** (kinds), **attributes**, and **builtins**
are lowercased. The `L` attribute is always uppercased as a special case to
avoid ambiguity with the number `1`.

Given the input (from `$ACC_ROOT_DIR/bmad/example_fodo.bmad`):

```
Q1: QUADRUPOLE, L=LQUAD, K1=K1_VAL
Q2: QUADRUPOLE, L=LQUAD, K1=-K1_VAL
D1: DRIFT, L=LDRIFT
...
CELL: LINE = (Q1, D1, Q2, D1)
...
USE, RING
```

latform produces:

```
Q1: quadrupole, L=LQUAD, k1=K1_VAL
Q2: quadrupole, L=LQUAD, k1=-K1_VAL
D1: drift, L=LDRIFT
...
CELL: line = (Q1, D1, Q2, D1)
...
use, RING
```

Note how `QUADRUPOLE` becomes `quadrupole` (kind), `K1` becomes `k1`
(attribute), while `Q1`, `LQUAD`, and `K1_VAL` stay uppercase (names).
`LINE` becomes `line` (keyword), and `USE` becomes `use` (keyword).

## Spacing

latform enforces consistent spacing around operators and delimiters:

- No spaces around `=` in attribute assignments: `L=0.5`
- A space around `=` in statement definitions: `CELL: line = (Q1, D1)`
- No spaces inside brackets or parentheses: `parameter[particle]`, `(Q1, D1)`
- A space after commas: `L=0.5, k1=1.2`
- Spaces around `+` and `-` in expressions (except unary minus): `797e6 + m_proton`, `k1=-K1_VAL`
- No spaces around `*`, `/`, `^`: `2*sqrt(2)/L_TOT`

For example, from `$ACC_ROOT_DIR/bmad-doc/lattices/Dragt_PSR_small_ring/Dragt_PSR_small_ring.bmad`:

```
parameter[E_tot] = 797e6 + m_proton
```

becomes:

```
parameter[e_tot] = 797e6 + m_proton
```

And from the input:

```
k1_optimal =  (1/Lq) * 2*sqrt(2) / L_tot
```

becomes:

```
K1_OPTIMAL = (1/LQ)*2*sqrt(2)/L_TOT
```

## Line Breaking

Lines that exceed the target line length (100 characters) are broken and
indented by 2 spaces. Breaks occur at commas and opening delimiters.

From `$ACC_ROOT_DIR/bmad-doc/lattices/cbeta/model/sub/erl.param.bmad`, the input:

```
srqua : quadrupole, aperture = 0.0254, field_master=t, fringe_type=full,
 l=+1.5123131574978999e-01, fq1=-8.4454203521079928e-05, fq2=+4.5142804358600607e-06
```

becomes:

```
SRQUA: quadrupole,
  aperture=0.0254,
  field_master=t,
  fringe_type=full,
  L=+1.5123131574978999e-01,
  fq1=-8.4454203521079928e-05,
  fq2=+4.5142804358600607e-06
```

Short lines are kept on a single line:

```
VC1: kicker, L=140e-3, field_master=t
```

## Nested Structures

Nested blocks (like `grid_field`, `wall`, `displacement`) are broken and
indented when they would exceed the line length. From
`src/latform/tests/files/parse_test.bmad`:

```
SBEND0: sbend, r_custom(3)=4, lr_self_wake_on=F, superimpose=F, offset=3.2, field_calc=fieldmap,
  grid_field={
    geometry=xyz,
    curved_coords=T,
    r0=(0, 0, 0),
    dr=(0.001, 0.001, 0.002),
    pt(1, 2, 3)=(1, 2, 3, 4, 5, 6)
  }
```

When a nested structure fits on one line, it stays compact:

```
CAP: capillary, wall={section={s=0, v(1)={1, 1}}, section={s=1, v(1)={1, 1}}}
```

## Blank Lines

By default and preferred by the style guide is non-compact mode, latform
inserts blank lines between different statement types and between line
definitions. This groups related statements visually.

From `$ACC_ROOT_DIR/bmad-doc/lattices/Dragt_PSR_small_ring/Dragt_PSR_small_ring.bmad`:

```
B36: sbend, L=2.54948, angle=36*raddeg
QD: quadrupole, L=0.5, b1_gradient=-2.68
QF: quadrupole, L=0.5, b1_gradient=1.95
SH: sextupole, L=0.5
SV: sextupole, L=0.5
D228: drift, L=2.28646
D148: drift, L=1.48646
D45: drift, L=0.45
D30: drift, L=0.30

P_NO: line = (D228, QD, D45, B36, D45, QF, D228)

P_TS: line = (D228, QD, D45, B36, D45, QF, D30, SH, D148)

P_LS: line = (D148, SV, D30, QD, D45, B36, D45, QF, D228)

PSR: line = (P_NO, P_TS, P_LS, 3*P_NO, P_TS, P_LS, 2*P_NO)

use, PSR
```

Element definitions are grouped together. Each `line` definition gets its own
block with blank lines between them. The `use` statement is separated from the
line definitions.

In `--compact` mode, these blank lines are removed:

```
parameter[e_tot] = 797e6 + m_proton
parameter[particle] = proton
parameter[geometry] = closed
B36: sbend, L=2.54948, angle=36*raddeg
QD: quadrupole, L=0.5, b1_gradient=-2.68
QF: quadrupole, L=0.5, b1_gradient=1.95
SH: sextupole, L=0.5
SV: sextupole, L=0.5
D228: drift, L=2.28646
D148: drift, L=1.48646
D45: drift, L=0.45
D30: drift, L=0.30
P_NO: line = (D228, QD, D45, B36, D45, QF, D228)
P_TS: line = (D228, QD, D45, B36, D45, QF, D30, SH, D148)
P_LS: line = (D148, SV, D30, QD, D45, B36, D45, QF, D228)
PSR: line = (P_NO, P_TS, P_LS, 3*P_NO, P_TS, P_LS, 2*P_NO)
use, PSR
```

## Section Breaks

Comment lines consisting of three or more repeated characters (like `!---`,
`!***`, `!===`, `!###`) are recognized as section breaks and normalized to a
full-width line of the configured character (default: `-`).

Input:

```
!***
```

Output (at default line length 100):

```
!----------------------------------------------------------------------------------------------------
```

## Comments

### Inline Comments

Inline comments are preserved and aligned:

```
B02: sbend, L=3.237903, angle=0.102289270  ! RHO =  31.65434
B03: rbend, L=2.945314, angle=0.020944245  ! RHO = 140.6264
B04: rbend, L=1.643524, angle=0.018699330  ! RHO =  87.8915
```

### Block Comments

Block comments (lines beginning with `!`) are preserved and associated with the
statement that follows them:

```
! Injector dipole, used with one face perpendicular to beam
INDPA.ASYM: sbend, fringe_type=full, L=0.255856230795702133
! Injector dipole, used symmetrically as rbend
INDPA.SYM: sbend, fringe_type=full, L=0.31932/sinc(0.5*INDPA.ANGLE)
```

### Commented-Out Code

Commented-out code is preserved as-is:

```
! qx: overlay = {q1[k1]:k1, q2[k1]:-k1}, var = {k1}, k1= k1_optimal
```

## Statement Definition Syntax

Element definitions use the `NAME: type` syntax with no space before the colon:

```
Q1: quadrupole, L=0.5, k1=1.2
FODO: line = (Q1, D1, Q2, D1)
OV1: overlay = {AA}, hkick
GANG0: group = {AA[tilt]:1, BB[tilt]:1}, var={t}, t=3
```

## Parameter Assignments

Standalone parameter assignments separate the target from the value with spaces
around `=`:

```
parameter[particle] = electron
beginning[e_tot] = 150e6
Q1[k1] = K1_OPTIMAL
*[tracking_method] = runge_kutta
SEXTUPOLE::CC[k2] = 3
```
