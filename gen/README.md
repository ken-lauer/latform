## Code generation tools

Generates `../src/latform/_attrs.py`: a dictionary of element name to a
dictionary of attributes`). All element kinds and attribute names are
upper-cased.

Run `build.sh` to regenerate in one step.

This assumes `ACC_ROOT_DIR` points to a valid Bmad installation; the build type
may need adjustment to link against a Debug/Production build.

### Pipeline

1. **`attrs.f90`** : links against Bmad and dumps every element attribute as a
   pipe-delimited table (`ELEMENT|ATTR|STATE|KIND|UNITS`).

2. **`descriptions.py`** : parses per-attribute descriptions out of the Bmad
   reference manual (`$ACC_ROOT_DIR/bmad/doc/elements.tex`).
   Coverage isn't great: descriptions are terse and many attributes are
   undocumented.

3. **`common_attrs.py`** : a hand-maintained table of descriptions for
   attributes common to many elements, , which the manual rarely re-documents
   per element. Edit this to fill in blanks.

4. **`gen_attrs.py`** : reads the Fortran dump on stdin, merges descriptions
   (an element's own manual description wins; otherwise the `common_attrs`
   table), and writes the final Python module to stdout.

To add or improve descriptions, edit `common_attrs.py` and re-run `build.sh`.

### Tao namelist schema

`gen_tao_schema.py` generates `../src/latform/tao/_schema.py`: the Tao `*.init`
namelist groups, their variables, and the reachable closure of derived-type
structs, used by `latform.tao.schema` for namelist type validation.

It depends on two external files (not in this repo or CI):

- `tao_namelists_schema.json` (from the `bmad` repo)
- `structs.json` (from the `cppbmad` repo; `codegen` must be importable)

```
PYTHONPATH=~/Repos/cppbmad python gen_tao_schema.py \
    --schema ~/Repos/bmad/tao_namelists_schema.json \
    --structs ~/Repos/cppbmad/structs.json
```

Regenerate and commit `tao/_schema.py` whenever the upstream schema changes.
