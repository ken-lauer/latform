from __future__ import annotations

import os
import pathlib

TEST_ROOT = pathlib.Path(__file__).resolve().parent
LATTICE_ROOT = TEST_ROOT / "files"
LATTICE_FILES = [
    LATTICE_ROOT / "fodo.bmad",
]

ACC_ROOT_DIR = (
    pathlib.Path(os.environ.get("ACC_ROOT_DIR", "")) if "ACC_ROOT_DIR" in os.environ else None
)

if ACC_ROOT_DIR is not None:
    LATTICE_FILES.extend([ACC_ROOT_DIR / "bmad-doc/lattices/figure_8/figure_8.bmad"])
