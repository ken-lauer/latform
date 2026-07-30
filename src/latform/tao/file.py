"""
Tao ``*.init`` file support.

Tao (the Bmad simulation front-end) is configured through Fortran-namelist
``*.init`` files. The primary one latform cares about is ``tao.init``, whose
``&tao_design_lattice`` group lists the lattice files to load via
``design_lattice(i)%file='...'`` derived-type array entries. Treating a
``tao.init`` as a source of lattice filenames lets every latform tool operate on
a Tao project without the user re-listing each lattice.

This module layers the small conveniences on top of the nmlform package's Namelist.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

from nmlform import (
    Assignment,
    Namelist,
    NamelistArrayEntry,
    NamelistArrayGroup,
    NamelistFile,
    NamelistFormatOptions,
    quote_value,
    unquote_value,
)
from nmlform import Token as NmlToken

from ._schema import STRUCTS
from .enums import integer_enum_for_field
from .schema import PathComponent, check_value, is_known_namelist, resolve_path

__all__ = [
    "TaoDatum",
    "TaoVariable",
    "TaoD1Data",
    "TaoV1Var",
    "TaoInit",
    "is_init_file",
    "looks_like_namelist",
    "fix_tao_namelist",
    "format_tao_namelist",
]

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


def looks_like_namelist(contents: str) -> bool:
    """Whether ``contents`` looks like it could be a namelist."""
    # Not a very accurate mechanism, but we at least know that lattice files
    # won't start out with ampersands
    return any(line.lstrip().startswith("&") for line in contents.splitlines())


def _read_if_exists(path: pathlib.Path) -> str | None:
    """Read ``path``'s text, or return ``None`` if it cannot be read."""
    try:
        return path.read_text()
    except OSError:
        return None


@dataclass
class TaoDatum(NamelistArrayEntry):
    """A single ``datum(i)`` entry within a ``&tao_d1_data`` group."""

    FIELDS: ClassVar[tuple[str, ...]] = tuple(STRUCTS["tao_datum_input"])

    @property
    def data_type(self) -> NmlToken | None:
        return self.value("data_type")

    @property
    def ele_ref_name(self) -> NmlToken | None:
        return self.value("ele_ref_name")

    @property
    def ele_start_name(self) -> NmlToken | None:
        return self.value("ele_start_name")

    @property
    def ele_name(self) -> NmlToken | None:
        return self.value("ele_name")

    @property
    def merit_type(self) -> NmlToken | None:
        return self.value("merit_type")

    @property
    def meas(self) -> NmlToken | None:
        return self.value("meas")

    @property
    def weight(self) -> NmlToken | None:
        return self.value("weight")


@dataclass
class TaoVariable(NamelistArrayEntry):
    """A single ``var(i)`` entry within a ``&tao_var`` group."""

    FIELDS: ClassVar[tuple[str, ...]] = tuple(STRUCTS["tao_var_input"])

    @property
    def ele_name(self) -> NmlToken | None:
        return self.value("ele_name")

    @property
    def attribute(self) -> NmlToken | None:
        return self.value("attribute")

    @property
    def universe(self) -> NmlToken | None:
        return self.value("universe")

    @property
    def weight(self) -> NmlToken | None:
        return self.value("weight")

    @property
    def step(self) -> NmlToken | None:
        return self.value("step")

    @property
    def low_lim(self) -> NmlToken | None:
        return self.value("low_lim")

    @property
    def high_lim(self) -> NmlToken | None:
        return self.value("high_lim")

    @property
    def merit_type(self) -> NmlToken | None:
        return self.value("merit_type")


@dataclass
class TaoD1Data(NamelistArrayGroup):
    """A ``&tao_d1_data`` group: one d1 array of `TaoDatum` entries."""

    @property
    def name(self) -> NmlToken | None:
        """The ``d1_data%name`` value (the d1 name, e.g. ``'12'``)."""
        return self._scalar("d1_data%name")

    @property
    def ix_d1_data(self) -> NmlToken | None:
        """The ``ix_d1_data`` index within the parent d2 group."""
        return self._scalar("ix_d1_data")

    @property
    def ix_min_data(self) -> NmlToken | None:
        return self._scalar("ix_min_data")

    @property
    def ix_max_data(self) -> NmlToken | None:
        return self._scalar("ix_max_data")

    @property
    def search_for_lat_eles(self) -> NmlToken | None:
        return self._scalar("search_for_lat_eles")

    @property
    def use_same_lat_eles_as(self) -> NmlToken | None:
        return self._scalar("use_same_lat_eles_as")

    @property
    def default_data_type(self) -> NmlToken | None:
        return self._scalar("default_data_type")

    @property
    def default_merit_type(self) -> NmlToken | None:
        return self._scalar("default_merit_type")

    @property
    def default_weight(self) -> NmlToken | None:
        return self._scalar("default_weight")

    @property
    def default_data_source(self) -> NmlToken | None:
        return self._scalar("default_data_source")

    @property
    def datums(self) -> list[TaoDatum]:
        """The ``datum(i)`` entries, ordered by index (non-integer subscripts last)."""
        return self._entries("datum", TaoDatum)


@dataclass
class TaoV1Var(NamelistArrayGroup):
    """A ``&tao_var`` group: one v1 array of `TaoVariable` entries."""

    @property
    def name(self) -> NmlToken | None:
        """The ``v1_var%name`` value (the v1 name)."""
        return self._scalar("v1_var%name")

    @property
    def ix_min_var(self) -> NmlToken | None:
        return self._scalar("ix_min_var")

    @property
    def ix_max_var(self) -> NmlToken | None:
        return self._scalar("ix_max_var")

    @property
    def search_for_lat_eles(self) -> NmlToken | None:
        return self._scalar("search_for_lat_eles")

    @property
    def use_same_lat_eles_as(self) -> NmlToken | None:
        return self._scalar("use_same_lat_eles_as")

    @property
    def default_attribute(self) -> NmlToken | None:
        return self._scalar("default_attribute")

    @property
    def default_merit_type(self) -> NmlToken | None:
        return self._scalar("default_merit_type")

    @property
    def default_weight(self) -> NmlToken | None:
        return self._scalar("default_weight")

    @property
    def default_universe(self) -> NmlToken | None:
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

    def _start_value(self, key: str) -> NmlToken | None:
        start = self.tao_start
        if start is None:
            return None
        assignment = start.get(key)
        return unquote_value(assignment.value.strip()) if assignment is not None else None

    @property
    def data_file(self) -> NmlToken | None:
        """``&tao_start`` ``data_file`` (where ``&tao_d1_data`` lives), if named."""
        return self._start_value("data_file")

    @property
    def var_file(self) -> NmlToken | None:
        """``&tao_start`` ``var_file`` (where ``&tao_var`` lives), if named."""
        return self._start_value("var_file")

    @property
    def plot_file(self) -> NmlToken | None:
        return self._start_value("plot_file")

    @property
    def beam_file(self) -> NmlToken | None:
        return self._start_value("beam_file")

    @property
    def building_wall_file(self) -> NmlToken | None:
        return self._start_value("building_wall_file")

    @property
    def startup_file(self) -> NmlToken | None:
        return self._start_value("startup_file")

    @property
    def hook_init_file(self) -> NmlToken | None:
        return self._start_value("hook_init_file")

    @property
    def init_name(self) -> NmlToken | None:
        return self._start_value("init_name")

    @property
    def n_universes(self) -> NmlToken | None:
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
            assignment.path.components[0].index: unquote_value(assignment.value.strip())
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
            namelist.set(f"design_lattice({position})%file", quote_value(filename))
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


def _parse_index(index_text: str | None) -> int | None:
    """A component subscript as an int, or ``None`` if absent/non-integer/multi-dim."""
    if not index_text:
        return None
    try:
        return int(index_text)
    except ValueError:
        return None


def path_components(assignment: Assignment) -> list[PathComponent]:
    """The schema `PathComponent` sequence for a namelist assignment's key."""
    return [
        PathComponent(component.name, _parse_index(component.index_text))
        for component in assignment.path.components
    ]


def _fix_character_value(value: str, enum: dict[int, str] | None) -> str:
    """
    Normalize one character value: map an enum index to its name, else quote.

    When ``enum`` governs the field (e.g. `TAO_COLORS`) and ``value`` is an
    integer index in it, the value becomes the quoted enum name (``2`` ->
    ``'red'``); an integer not in the map is left as is, since Tao also accepts
    the numeric form. Otherwise an unquoted value is quoted and an
    already-quoted value is returned unchanged.
    """
    if enum is not None:
        try:
            index = int(value)
        except ValueError:
            pass
        else:
            name = enum.get(index)
            return quote_value(name) if name is not None else value
    if check_value("character", value):
        return value
    return quote_value(value)


def _fix_character_token(text: str, enum: dict[int, str] | None) -> str:
    """
    Normalize a character value literal, preserving any Fortran ``n*`` repeat.

    Only the value part of an ``n*value`` repeat is normalized (via
    `_fix_character_value`); everything else is left intact.
    """
    count, star, value = text.partition("*")
    if star and count.strip().lstrip("+-").isdigit():
        return f"{count}*{_fix_character_value(value, enum)}"
    return _fix_character_value(text, enum)


def _fixed_character_value(assignment: Assignment, enum: dict[int, str] | None) -> str | None:
    """
    The assignment's right-hand side with each character value normalized.

    Returns ``None`` when nothing changed, so the caller can skip the edit.
    """
    changed = False
    parts = []
    for token in assignment.field_tokens:
        fixed = _fix_character_token(token.text, enum)
        parts.append(fixed)
        changed = changed or fixed != token.text
    return " ".join(parts) if changed else None


def _fix_namelist_group(group: Namelist) -> None:
    """Normalize character assignments (quoting, enum index -> name) in a group."""
    if not is_known_namelist(group.name):
        return
    # `Namelist.set` reparses (invalidating `group.assignments`), so collect the
    # edits first, then apply them.
    edits: list[tuple[str, str]] = []
    for assignment in group.assignments:
        components = path_components(assignment)
        leaf = resolve_path(group.name, components).leaf
        if leaf is None or leaf.kind != "intrinsic" or leaf.base != "character":
            continue
        enum = integer_enum_for_field(components[-1].name) if components else None
        fixed = _fixed_character_value(assignment, enum)
        if fixed is not None:
            edits.append((assignment.key, fixed))
    for key, value in edits:
        group.set(key, value)


def fix_tao_namelist(namelist: Namelist | NamelistFile) -> None:
    """
    Normalize namelist assignments in place against the Tao type schema.

    This quotes character-valued assignments written unquoted (e.g.
    ``plot_file = tao.init`` -> ``plot_file = 'tao.init'``), which Tao reads the
    same but which are otherwise a type error, and rewrites integer indices in
    enum-valued character fields to their names (e.g. a color ``prompt_color = 2``
    -> ``prompt_color = 'red'``). Given a `NamelistFile`, every group in it is
    fixed; groups (and files) whose namelist name is not in the schema are left
    untouched.
    """
    if isinstance(namelist, NamelistFile):
        for item in namelist.items:
            if isinstance(item, Namelist):
                _fix_namelist_group(item)
    else:
        _fix_namelist_group(namelist)


def format_tao_namelist(
    namelist: Namelist | NamelistFile,
    *,
    options: NamelistFormatOptions | None = None,
    fix_types: bool = True,
) -> str:
    """
    Render a Tao namelist (or file), first fixing argument types when asked.

    With ``fix_types`` (the default), `fix_tao_namelist` runs first, so the
    rendered text has its character values quoted. This mutates ``namelist`` in
    place; render the original first if you need the unmodified text.
    """
    if fix_types:
        fix_tao_namelist(namelist)
    return namelist.render(options)
