"""
Tao ``*.init`` file support.

Tao (the Bmad simulation front-end) is configured through Fortran-namelist
``*.init`` files. The primary one latform cares about is ``tao.init``, whose
``&tao_design_lattice`` group lists the lattice files to load via
``design_lattice(i)%file='...'`` derived-type array entries. Treating a
``tao.init`` as a source of lattice filenames lets every latform tool operate on
a Tao project without the user re-listing each lattice.

This module layers the small conveniences on top of the generic
:mod:`latform._namelist` model.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

from ._namelist import Assignment, Namelist, NamelistArrayEntry, NamelistArrayGroup, NamelistFile
from .token import Token

__all__ = [
    "DATUM_FIELDS",
    "VAR_FIELDS",
    "TaoDatum",
    "TaoVariable",
    "TaoD1Data",
    "TaoV1Var",
    "TaoInit",
    "is_init_file",
]

# `tao_datum_input` fields in declaration order
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

# `tao_var_input` fields in declaration order
# (ref $ACC_ROOT_DIR/tao/code/tao_input_struct.f90)
VAR_FIELDS: tuple[str, ...] = (
    "ele_name",
    "attribute",
    "universe",
    "weight",
    "step",
    "low_lim",
    "high_lim",
    "merit_type",
    "good_user",
    "key_bound",
    "key_delta",
    "meas",
)

# ``&tao_start`` keys that name an auxiliary file whose namelists we can parse.
SOURCE_FILE_KEYS: tuple[str, ...] = (
    "data_file",
    "var_file",
    "plot_file",
    "beam_file",
    "building_wall_file",
)

# The `&tao_start` file each namelist group is read from.
# Other groups are from `tao.init` only.
NAMELIST_SOURCE: dict[str, str] = {
    #
    "tao_d2_data": "data_file",
    "tao_d1_data": "data_file",
    #
    "tao_var": "var_file",
    #
    "tao_beam_init": "beam_file",
    #
    "tao_plot_page": "plot_file",
    "tao_template_plot": "plot_file",
    "tao_template_graph": "plot_file",
    "floor_plan_drawing": "plot_file",
    "lat_layout_drawing": "plot_file",
    "shape_pattern": "plot_file",
    #
    "building_wall_orientation": "building_wall_file",
    "building_wall_section": "building_wall_file",
}


def is_init_file(path: pathlib.Path | str) -> bool:
    """Whether ``path`` names a Tao namelist init file (``*.init``)."""
    return pathlib.Path(path).suffix.lower() == ".init"


def _read_if_exists(path: pathlib.Path) -> str | None:
    """Read ``path``'s text, or return ``None`` if it cannot be read."""
    try:
        return path.read_text()
    except OSError:
        return None


@dataclass
class TaoDatum(NamelistArrayEntry):
    """A single ``datum(i)`` entry within a ``&tao_d1_data`` group."""

    FIELDS: ClassVar[tuple[str, ...]] = DATUM_FIELDS

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
class TaoVariable(NamelistArrayEntry):
    """A single ``var(i)`` entry within a ``&tao_var`` group."""

    FIELDS: ClassVar[tuple[str, ...]] = VAR_FIELDS

    @property
    def ele_name(self) -> Token | None:
        return self.value("ele_name")

    @property
    def attribute(self) -> Token | None:
        return self.value("attribute")

    @property
    def universe(self) -> Token | None:
        return self.value("universe")

    @property
    def weight(self) -> Token | None:
        return self.value("weight")

    @property
    def step(self) -> Token | None:
        return self.value("step")

    @property
    def low_lim(self) -> Token | None:
        return self.value("low_lim")

    @property
    def high_lim(self) -> Token | None:
        return self.value("high_lim")

    @property
    def merit_type(self) -> Token | None:
        return self.value("merit_type")


@dataclass
class TaoD1Data(NamelistArrayGroup):
    """A ``&tao_d1_data`` group: one d1 array of `TaoDatum` entries."""

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
        return self._entries("datum", TaoDatum)


@dataclass
class TaoV1Var(NamelistArrayGroup):
    """A ``&tao_var`` group: one v1 array of `TaoVariable` entries."""

    @property
    def name(self) -> Token | None:
        """The ``v1_var%name`` value (the v1 name)."""
        return self._scalar("v1_var%name")

    @property
    def ix_min_var(self) -> Token | None:
        return self._scalar("ix_min_var")

    @property
    def ix_max_var(self) -> Token | None:
        return self._scalar("ix_max_var")

    @property
    def search_for_lat_eles(self) -> Token | None:
        return self._scalar("search_for_lat_eles")

    @property
    def use_same_lat_eles_as(self) -> Token | None:
        return self._scalar("use_same_lat_eles_as")

    @property
    def default_attribute(self) -> Token | None:
        return self._scalar("default_attribute")

    @property
    def default_merit_type(self) -> Token | None:
        return self._scalar("default_merit_type")

    @property
    def default_weight(self) -> Token | None:
        return self._scalar("default_weight")

    @property
    def default_universe(self) -> Token | None:
        return self._scalar("default_universe")

    @property
    def variables(self) -> list[TaoVariable]:
        """The ``var(i)`` entries, ordered by index (non-integer subscripts last)."""
        return self._entries("var", TaoVariable)


@dataclass
class TaoInit(NamelistFile):
    """
    A ``tao.init`` namelist file, with lattice/data/variable conveniences.

    Attributes
    ----------
    sources : dict[str, NamelistFile]
        Auxiliary namelist files from ``&tao_start`` (e.g. ``"data_file"``).
    """

    sources: dict[str, NamelistFile] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: pathlib.Path | str) -> TaoInit:
        tao = super().from_file(path)
        tao.load_sources()
        return tao

    # -- &tao_start and auxiliary file resolution ------------------------------

    @property
    def tao_start(self) -> Namelist | None:
        """The ``&tao_start`` group (the first, if repeated)."""
        return self.get_namelist("tao_start")

    def _start_value(self, key: str) -> Token | None:
        start = self.tao_start
        if start is None:
            return None
        assignment = start.get(key)
        return assignment.value.strip().remove_quotes() if assignment is not None else None

    @property
    def data_file(self) -> Token | None:
        """``&tao_start`` ``data_file`` (where ``&tao_d1_data`` lives), if named."""
        return self._start_value("data_file")

    @property
    def var_file(self) -> Token | None:
        """``&tao_start`` ``var_file`` (where ``&tao_var`` lives), if named."""
        return self._start_value("var_file")

    @property
    def plot_file(self) -> Token | None:
        return self._start_value("plot_file")

    @property
    def beam_file(self) -> Token | None:
        return self._start_value("beam_file")

    @property
    def building_wall_file(self) -> Token | None:
        return self._start_value("building_wall_file")

    @property
    def startup_file(self) -> Token | None:
        return self._start_value("startup_file")

    @property
    def hook_init_file(self) -> Token | None:
        return self._start_value("hook_init_file")

    @property
    def init_name(self) -> Token | None:
        return self._start_value("init_name")

    @property
    def n_universes(self) -> Token | None:
        return self._start_value("n_universes")

    def load_sources(
        self,
        base: pathlib.Path | None = None,
        reader: Callable[[pathlib.Path], str | None] | None = None,
    ) -> None:
        """
        Resolve and load the auxiliary files named in ``&tao_start``.

        Parameters
        ----------
        base : pathlib.Path, optional
            Directory relative names resolve against. Defaults to the tao.init
            directory.
        reader : callable, optional
            ``path -> text | None`` hook used to read each resolved file
            (``None`` = "missing"). Defaults to reading from disk.
        """
        if base is None:
            base = self.filename.parent if self.filename is not None else pathlib.Path()
        if reader is None:
            reader = _read_if_exists

        for key in SOURCE_FILE_KEYS:
            name = self._start_value(key)
            if not name:
                continue
            path = pathlib.Path(os.path.expandvars(name.strip()))
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
            text = reader(path)
            if text is not None:
                self.sources[key] = NamelistFile.parse(text, filename=path)

    def _source_for(self, namelist_name: str) -> NamelistFile:
        """
        The namelist file that provides ``namelist_name``.

        Returns the loaded auxiliary file when ``&tao_start`` split that
        category out (and it was loaded); otherwise this ``tao.init`` itself.
        """
        key = NAMELIST_SOURCE.get(namelist_name.lower())
        if key is not None:
            aux = self.sources.get(key)
            if aux is not None:
                return aux
        return self

    def namelists_for(self, namelist_name: str) -> list[Namelist]:
        """All ``namelist_name`` groups from their resolved source, in file order."""
        source = self._source_for(namelist_name)
        return source.namelists_by_name.get(namelist_name.lower(), [])

    # -- lattice --------------------------------------------------------------

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

    # -- data / variables (resolved across auxiliary sources) -----------------

    @property
    def d1_data(self) -> list[TaoD1Data]:
        """All ``&tao_d1_data`` groups, from the ``data_file`` source, in order."""
        return [TaoD1Data(namelist) for namelist in self.namelists_for("tao_d1_data")]

    @property
    def variables(self) -> list[TaoV1Var]:
        """All ``&tao_var`` groups, from the ``var_file`` source, in order."""
        return [TaoV1Var(namelist) for namelist in self.namelists_for("tao_var")]

    # -- placeholders for future work (raw groups from their sources) ---------

    @property
    def beam_init(self) -> list[Namelist]:
        """The ``&tao_beam_init`` groups, from the ``beam_file`` source."""
        return self.namelists_for("tao_beam_init")

    @property
    def plot_page(self) -> list[Namelist]:
        """The ``&tao_plot_page`` groups, from the ``plot_file`` source."""
        return self.namelists_for("tao_plot_page")

    @property
    def building_wall_sections(self) -> list[Namelist]:
        """The ``&building_wall_section`` groups, from the ``building_wall_file`` source."""
        return self.namelists_for("building_wall_section")
