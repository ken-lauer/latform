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
from dataclasses import dataclass, field

from ._namelist import Assignment, Namelist, NamelistFile, split_values
from .token import Token

__all__ = [
    "DATUM_FIELDS",
    "TaoDatum",
    "TaoD1Data",
    "TaoInit",
    "TaoPlot",
    "is_init_file",
]

DATUM_FIELDS: tuple[str, ...] = (
    "data_type",
    "ele_ref_name",
    "ele_start_name",
    "ele_name",
    "merit_type",
    "meas",
    "weight",
    "good_user",
    "good_opt",
    "data_source",
    "eval_point",
    "s_offset",
    "ref_s_offset",
    "ix_bunch",
)


def is_init_file(path: pathlib.Path | str) -> bool:
    """Whether ``path`` names a Tao namelist init file (``*.init``)."""
    return pathlib.Path(path).suffix.lower() == ".init"


@dataclass
class TaoDatum:
    """
    A single ``datum(i) = ...`` entry within a ``&tao_d1_data`` group.

    Attributes
    ----------
    index : int or None
        The ``i`` in ``datum(i)``. ``0`` is the conventional slot Tao reads for
        ``SEARCH:``/``SAME:`` element specifications; ``None`` for a
        non-integer subscript (e.g. a range).
    positional : list[Token]
        Values from an anonymous ``datum(i) = ...`` assignment, in field order.
    components : dict[str, Token]
        Values from ``datum(i)%field = ...`` assignments, keyed by field name.
    comment : str
        Trailing comment on the anonymous assignment, if any.
    """

    index: int | None
    positional: list[Token] = field(default_factory=list)
    components: dict[str, Token] = field(default_factory=dict)
    comment: str = ""

    def get(self, name: str) -> Token | None:
        """The raw (quoted, if a string) value token for ``name``, or ``None``."""
        name = name.lower()
        if name in self.components:
            return self.components[name]
        if name in DATUM_FIELDS:
            position = DATUM_FIELDS.index(name)
            if position < len(self.positional):
                return self.positional[position]
        return None

    def value(self, name: str) -> Token | None:
        """The value for ``name``, unquoted; ``None`` if unset, ``''`` if blank."""
        token = self.get(name)
        if token is None:
            return None
        return token.strip().remove_quotes()

    @property
    def data_type(self) -> Token | None:
        return self.value("data_type")

    @property
    def ele_ref_name(self) -> Token | None:
        return self.value("ele_ref_name")

    @property
    def ele_start_name(self) -> Token | None:
        return self.value("ele_start_name")

    @property
    def ele_name(self) -> Token | None:
        return self.value("ele_name")

    @property
    def merit_type(self) -> Token | None:
        return self.value("merit_type")

    @property
    def meas(self) -> Token | None:
        return self.value("meas")

    @property
    def weight(self) -> Token | None:
        return self.value("weight")


@dataclass
class TaoD1Data:
    """A ``&tao_d1_data`` group: one d1 array of `TaoDatum` entries."""

    namelist: Namelist

    def _scalar(self, key: str) -> Token | None:
        assignment = self.namelist.get(key)
        if assignment is None:
            return None
        return assignment.value.strip().remove_quotes()

    @property
    def name(self) -> Token | None:
        """The ``d1_data%name`` value (the d1 name, e.g. ``'12'``)."""
        return self._scalar("d1_data%name")

    @property
    def ix_d1_data(self) -> Token | None:
        """The ``ix_d1_data`` index within the parent d2 group."""
        return self._scalar("ix_d1_data")

    @property
    def ix_min_data(self) -> Token | None:
        return self._scalar("ix_min_data")

    @property
    def ix_max_data(self) -> Token | None:
        return self._scalar("ix_max_data")

    @property
    def search_for_lat_eles(self) -> Token | None:
        return self._scalar("search_for_lat_eles")

    @property
    def use_same_lat_eles_as(self) -> Token | None:
        return self._scalar("use_same_lat_eles_as")

    @property
    def default_data_type(self) -> Token | None:
        return self._scalar("default_data_type")

    @property
    def default_merit_type(self) -> Token | None:
        return self._scalar("default_merit_type")

    @property
    def default_weight(self) -> Token | None:
        return self._scalar("default_weight")

    @property
    def default_data_source(self) -> Token | None:
        return self._scalar("default_data_source")

    @property
    def datums(self) -> list[TaoDatum]:
        """The ``datum(i)`` entries, ordered by index (non-integer subscripts last)."""
        by_index: dict[object, TaoDatum] = {}

        def datum_for(component) -> TaoDatum:
            key = component.index if component.index is not None else component.index_text
            existing = by_index.get(key)
            if existing is None:
                existing = TaoDatum(index=component.index)
                by_index[key] = existing
            return existing

        for assignment in self.namelist.assignments:
            path = assignment.path
            if path.names[0] != "datum":
                continue
            datum = datum_for(path.components[0])
            if len(path.components) == 1:
                datum.positional = split_values(assignment.value)
                datum.comment = assignment.comment
            else:
                datum.components[path.names[1]] = assignment.value

        def sort_key(key: object) -> tuple[int, object]:
            return (0, key) if isinstance(key, int) else (1, str(key))

        return [by_index[key] for key in sorted(by_index, key=sort_key)]


@dataclass
class TaoInit(NamelistFile):
    """A ``tao.init`` namelist file, with ``design_lattice`` conveniences."""

    @property
    def design_lattice(self) -> Namelist | None:
        """The ``&tao_design_lattice`` group (the first, if repeated)."""
        return self.get_namelist("tao_design_lattice")

    @property
    def d1_data(self) -> list[TaoD1Data]:
        """All ``&tao_d1_data`` groups, in file order."""
        return [TaoD1Data(namelist) for namelist in self.namelists_by_name.get("tao_d1_data", [])]

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
