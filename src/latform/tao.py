"""
Tao ``*.init`` file support.

Tao (the Bmad simulation front-end) is configured through Fortran-namelist
``*.init`` files. The one latform cares about is ``tao.init``, whose
``&tao_design_lattice`` group lists the lattice files to load via
``design_lattice(i)%file='...'`` derived-type array entries. Treating a
``tao.init`` as a source of lattice filenames lets every latform tool operate on
a Tao project without the user re-listing each lattice.

This module layers the small conveniences on top of the generic
:mod:`latform.namelist` model.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from ._namelist import Assignment, Namelist, NamelistFile

__all__ = [
    "TaoInit",
    "TaoPlot",
    "is_init_file",
]


def is_init_file(path: pathlib.Path | str) -> bool:
    """Whether ``path`` names a Tao namelist init file (``*.init``)."""
    return pathlib.Path(path).suffix.lower() == ".init"


@dataclass
class TaoInit(NamelistFile):
    """A ``tao.init`` namelist file, with ``design_lattice`` conveniences."""

    @property
    def design_lattice(self) -> Namelist | None:
        """The ``&tao_design_lattice`` group (the first, if repeated)."""
        return self.get_namelist("tao_design_lattice")

    def _lattice_file_assignments(self) -> list[Assignment]:
        namelist = self.design_lattice
        if namelist is None:
            return []
        return [
            assignment
            for assignment in namelist.assignments
            if assignment.path.names == ("design_lattice", "file")
            and assignment.path.components[0].index is not None
        ]

    @property
    def lattice_files(self) -> list[str]:
        """
        Ordered ``design_lattice(i)%file`` values (unquoted), by index.

        When set, rewritse the ``design_lattice(i)%file`` entries to ``files`` (1-based).
        Existing entries are updated in place. Non-matching additional entries
        are removed, and new entries are appended.
        """
        by_index = {
            assignment.path.components[0].index: assignment.value.strip().remove_quotes()
            for assignment in self._lattice_file_assignments()
            if assignment.path.components[0].index is not None
        }
        return [by_index[i] for i in sorted(by_index)]

    @lattice_files.setter
    def lattice_files(self, files: list[str]) -> None:
        namelist = self.design_lattice
        if namelist is None:
            namelist = self.update_namelist("tao_design_lattice", {})

        existing_indices = {
            assn.path.components[0].index for assn in self._lattice_file_assignments()
        }
        for position, filename in enumerate(files, start=1):
            namelist.set(f"design_lattice({position})%file", f"'{filename}'")
        for surplus in existing_indices - set(range(1, len(files) + 1)):
            namelist.remove(f"design_lattice({surplus})%file")


@dataclass
class TaoPlot(NamelistFile):
    """A ``tao_plot.init`` file. No lattice-specific conveniences (yet)."""
