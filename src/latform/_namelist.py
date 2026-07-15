"""
Minimal Fortran-namelist reader/writer.

This will **eventually** go away when f90nml supports everything needed from
our init.tao files, so it's considered mostly internal API.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import cast

from typing_extensions import Self

from .location import Location
from .token import Token

__all__ = [
    "KeyComponent",
    "KeyPath",
    "Assignment",
    "Namelist",
    "NamelistFile",
]

_RE_GROUP_OPEN = re.compile(r"\s*&(\w+)")
_RE_COMPONENT = re.compile(r"([^()%]+)(?:\((.*)\))?")
_RE_NORMALIZE_KEY = re.compile(r"\s+")


def _split_comment(line: str, comment_char: str = "!", escape_char: str = "\\") -> tuple[str, str]:
    """
    Splits a line into code and comment parts based on the specified comment character,
    while respecting quoted strings and escape characters.

    Parameters
    ----------
    line : str
        The input string to split.
    comment_char : str, optional
        The character indicating the start of a comment (default is '!').
    escape_char : str, optional
        The character used to escape other characters (default is '\\').

    Returns
    -------
    tuple[str, str]
        A tuple containing two strings:
        - The first element is the code part of the line, with leading and trailing whitespace removed.
        - The second element is the comment part of the line, with leading and trailing whitespace removed.
          If no comment is found, the second element is an empty string.
    """
    # (ref cppbmad codegen.struct_parser.util.split_comment)
    in_single_quote = False
    in_double_quote = False
    escape_next = False

    for i, char in enumerate(line):
        if escape_next:
            escape_next = False
        elif char == escape_char:
            escape_next = True
        elif char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == comment_char and not in_single_quote and not in_double_quote:
            # Found comment char outside of quotes
            return (line[:i], line[i + 1 :])

    return line, ""


def _normalize_key(key: str) -> str:
    """Whitespace-insensitive, case-insensitive key form used for lookups."""
    return _RE_NORMALIZE_KEY.sub("", key).lower()


@dataclass(frozen=True)
class KeyComponent:
    """One ``name`` or ``name(index)`` segment of a derived-type key path."""

    name: str
    index_text: str | None = None

    @property
    def index(self) -> int | None:
        """The integer index, or ``None`` for no index or a non-integer (e.g. a ``1:8`` range)."""
        if self.index_text is not None and self.index_text.isdigit():
            return int(self.index_text)
        return None

    def __str__(self) -> str:
        return self.name if self.index_text is None else f"{self.name}({self.index_text})"


@dataclass(frozen=True, eq=False)
class KeyPath:
    """A decomposed derived-type assignment key, e.g. ``foo(3)%bar(2)%val``."""

    components: tuple[KeyComponent, ...]

    @classmethod
    def parse(cls, key: str) -> KeyPath:
        normalized = re.sub(r"\s+", "", key)
        components: list[KeyComponent] = []
        for part in normalized.split("%"):
            match = _RE_COMPONENT.fullmatch(part)
            if match is None:
                components.append(KeyComponent(part))
            else:
                components.append(KeyComponent(match.group(1), match.group(2)))
        return cls(tuple(components))

    @property
    def names(self) -> tuple[str, ...]:
        """The component names in order, without indices."""
        return tuple(component.name.lower() for component in self.components)

    def __str__(self) -> str:
        return "%".join(str(component) for component in self.components)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, KeyPath):
            return self.components == other.components
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.components)


@dataclass
class Assignment:
    """A single ``key = value`` assignment within a namelist group."""

    key: str
    value: Token
    comment: str = ""

    @property
    def loc(self) -> Location:
        """Source location of the value (``value.loc``)."""
        return self.value.loc

    @property
    def path(self) -> KeyPath:
        """The key decomposed into :class:`KeyComponent` parts (any nesting depth)."""
        return KeyPath.parse(self.key)

    @classmethod
    def from_line(
        cls,
        line: str,
        idx: int,
        filename: pathlib.Path | None = None,
        base_line: int = 0,
    ) -> Assignment | None:
        """Parse a namelist file line into an `Assignment`."""
        stripped = line.lstrip()
        if not stripped or stripped[0] in "!&/":
            return None
        code, comment = _split_comment(line)

        try:
            eq = code.index("=")
        except ValueError:
            return None

        key = code[:eq].strip()
        if not key:
            return None
        rhs = code[eq + 1 :]
        leading = len(rhs) - len(rhs.lstrip())
        value = rhs.strip()
        value_start = eq + 1 + leading
        value_end = value_start + len(value)
        absolute_line = base_line + idx
        value_token = Token(
            value,
            loc=Location(
                filename=filename,
                line=absolute_line,
                column=value_start,
                end_line=absolute_line,
                end_column=value_end,
            ),
        )
        return cls(key=re.sub(r"\s+", "", key), value=value_token, comment=comment.strip())


@dataclass
class Namelist:
    """
    One ``&name ... /`` namelist group from a .nml file like `"tao.init"`.

    Attributes
    ----------
    lines : list[str]
        The verbatim source lines of the block (including the ``&name``
        opener and ``/`` terminator) and are the source of truth for rendering.
    assignments : list[Assignment]
        Derived from ``lines`` and refreshed after every edit.
    filename : pathlib.Path or None
        The source filename, if applicable.
    start_line : int
        The absolute, 0-indexed line of the ``&name`` opener in the source
        file) anchor.
    """

    name: str
    # Warning: you must call `_reparse()` if this is edited in-place.
    lines: list[str] = field(default_factory=list)
    assignments: list[Assignment] = field(default_factory=list)
    filename: pathlib.Path | None = None
    start_line: int = 0

    def __post_init__(self) -> None:
        self._reparse()

    def _reparse(self) -> None:
        self.assignments = []
        for idx, line in enumerate(self.lines):
            assignment = Assignment.from_line(line, idx, self.filename, self.start_line)
            if assignment is not None:
                self.assignments.append(assignment)

    @property
    def loc(self) -> Location:
        """Source location spanning the whole ``&name ... /`` block."""
        end_line = self.start_line + max(len(self.lines) - 1, 0)
        end_column = len(self.lines[-1]) if self.lines else 0
        return Location(
            filename=self.filename,
            line=self.start_line,
            column=0,
            end_line=end_line,
            end_column=end_column,
        )

    def get(self, key: str) -> Assignment | None:
        """Return the assignment matching ``key`` (case/space-insensitive)."""
        target = _normalize_key(key)
        for assignment in self.assignments:
            if _normalize_key(assignment.key) == target:
                return assignment
        return None

    def _line_of(self, assignment: Assignment) -> int:
        """Index into ``self.lines`` of ``assignment`` (from its value's loc)."""
        return assignment.value.loc.line - self.start_line

    def _indent(self) -> str:
        """Indentation to use for inserted lines, mirroring existing entries."""
        for assignment in self.assignments:
            line = self.lines[self._line_of(assignment)]
            return line[: len(line) - len(line.lstrip())]
        return "  "

    def _terminator_index(self) -> int:
        for idx in range(len(self.lines) - 1, -1, -1):
            if self.lines[idx].lstrip().startswith("/"):
                return idx
        return len(self.lines)

    def set(self, key: str, value: str, *, comment: str = "") -> Assignment:
        """Update ``key``'s value in place, or append it before the terminator."""
        existing = self.get(key)
        if existing is not None:
            idx = self._line_of(existing)
            line = self.lines[idx]
            value_loc = existing.value.loc
            self.lines[idx] = line[: value_loc.column] + value + line[value_loc.end_column :]
        else:
            new_line = f"{self._indent()}{key} = {value}"
            if comment:
                if not comment.lstrip().startswith("!"):
                    comment = f"!{comment}"
                new_line = f"{new_line} {comment}"
            self.lines.insert(self._terminator_index(), new_line)

        self._reparse()
        return cast(Assignment, self.get(key))

    def remove(self, key: str) -> None:
        """Remove the line defining ``key`` (no-op if it is not present)."""
        existing = self.get(key)
        if existing is None:
            return
        del self.lines[self._line_of(existing)]
        self._reparse()

    def render(self) -> str:
        return "\n".join(self.lines)


@dataclass
class NamelistFile:
    """A parsed namelist file: an ordered list of groups and raw text chunks."""

    # Namelists and strings between those namelists:
    items: list[Namelist | str] = field(default_factory=list)

    filename: pathlib.Path | None = None

    @classmethod
    def parse(cls, text: str, filename: pathlib.Path | str | None = None) -> Self:
        path = pathlib.Path(filename) if filename is not None else None
        items: list[Namelist | str] = []
        raw: list[str] = []
        current: Namelist | None = None

        def flush_raw() -> None:
            if raw:
                items.append("\n".join(raw))
                raw.clear()

        for idx, line in enumerate(text.split("\n")):
            stripped = line.lstrip()
            if current is None:
                if stripped.startswith("&"):
                    flush_raw()
                    name = _RE_GROUP_OPEN.match(line).group(1)  # type: ignore[union-attr]
                    current = Namelist(name=name, lines=[line], filename=path, start_line=idx)
                else:
                    raw.append(line)
            else:
                current.lines.append(line)
                if stripped.startswith("/"):
                    current._reparse()
                    items.append(current)
                    current = None
        if current is not None:
            current._reparse()
            items.append(current)
        flush_raw()
        return cls(items=items, filename=path)

    @classmethod
    def from_file(cls, path: pathlib.Path | str) -> Self:
        path = pathlib.Path(path)
        return cls.parse(path.read_text(), filename=path)

    def render(self) -> str:
        out: list[str] = []
        for item in self.items:
            if isinstance(item, Namelist):
                out.extend(item.lines)
            else:
                out.extend(item.split("\n"))
        return "\n".join(out)

    @property
    def namelists(self) -> list[Namelist]:
        return [item for item in self.items if isinstance(item, Namelist)]

    @property
    def namelists_by_name(self) -> dict[str, list[Namelist]]:
        """Namelist groups keyed by (lowercased) name; a name may repeat."""
        result: dict[str, list[Namelist]] = {}
        for namelist in self.namelists:
            result.setdefault(namelist.name.lower(), []).append(namelist)
        return result

    def get_namelist(self, name: str, index: int = 0) -> Namelist | None:
        """The ``index``-th namelist named ``name`` (case-insensitive), or ``None``."""
        matches = self.namelists_by_name.get(name.lower(), [])
        if -len(matches) <= index < len(matches):
            return matches[index]
        return None

    def update_namelist(
        self,
        name: str,
        assignments: dict[str, str],
        *,
        index: int = 0,
    ) -> Namelist:
        """
        Add/update a namelist section.

        Parameters
        ----------
        name : str
            The namelist group name.
        assignments : dict[str, str]
            ``key -> raw value`` entries to set. Existing keys are updated in
            place; missing keys are appended before the terminator.
        index : int, optional
            When several groups share ``name``, which one to update. Ignored
            when creating a new group. Defaults to the first.

        Returns
        -------
        Namelist
            The updated or newly created group.
        """
        name = name.removeprefix("&")
        target = self.get_namelist(name, index)
        if target is None:
            target = Namelist(name=name, lines=[f"&{name}", "/"])

            if self.items:
                last_item = self.items[-1]
                if isinstance(last_item, Namelist) or last_item.strip():
                    self.items.append("")

            self.items.append(target)
        for key, value in assignments.items():
            target.set(key, value)
        return target
