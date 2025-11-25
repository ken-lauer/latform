## Code generation tools

### `attrs.f90`

This generates `../src/latform/_attrs.py`, with a dictionary of element name to
dictionary of attributes.

All element kinds and attribute names are upper-cased.

See `build.sh` for a simple way to regenerate this in a single step.
Assumes `ACC_ROOT_DIR` points to a valid Bmad installation.
Build type may need adjustment to properly link to Debug/Production builds.
