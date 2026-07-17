"""
Minimal Fortran-namelist reader/writer.

This will **eventually** go away when f90nml supports everything needed from
our init.tao files, so it's considered mostly internal API.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, cast

from .location import Location
from .token import Token
from .types import NamelistFormatOptions

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Literal

    try:
        from typing import Self
    except ImportError:
        from typing_extensions import Self


__all__ = [
    "Assignment",
    "KeyComponent",
    "KeyPath",
    "Namelist",
    "NamelistArrayEntry",
    "NamelistArrayGroup",
    "NamelistFile",
    "is_namelist_file",
    "split_values",
]

_NAMELIST_SUFFIXES = frozenset({".init", ".nml"})


def is_namelist_file(path: pathlib.Path | str) -> bool:
    """Whether ``path`` names a Fortran-namelist file (``*.init`` or ``*.nml``)."""
    return pathlib.Path(path).suffix.lower() in _NAMELIST_SUFFIXES


_RE_GROUP_OPEN = re.compile(r"\s*&(\w+)")
_RE_COMPONENT = re.compile(r"([^()%]+)(?:\((.*)\))?")
_RE_NORMALIZE_KEY = re.compile(r"\s+")
_RE_INT = re.compile(r"-?\d+")
_RE_SLICE = re.compile(r"(-?\d+):(-?\d+)(?::(-?\d+))?")
_RE_OPEN_SLICE = re.compile(r"(-?\d+):")
# A namelist value field is a quoted string or a bare token; fields are
# separated by whitespace/commas, which fall between the matches. Fortran
# escapes a quote inside a string by doubling it ('it''s').
_SINGLE_QUOTED = r"'(?:''|[^'])*'?"  # trailing '? tolerates a missing closing quote
_DOUBLE_QUOTED = r'"(?:""|[^"])*"?'
_BARE_TOKEN = r"""[^ \t\r\n,'"]+"""  # runs until a separator or quote
_REPEAT_COUNT = r"(?:(?P<count>\d+)\*)?"  # optional Fortran repeat: 3*0 -> 0 0 0
_RE_VALUE_FIELD = re.compile(
    rf"{_REPEAT_COUNT}(?P<value>{_SINGLE_QUOTED}|{_DOUBLE_QUOTED}|{_BARE_TOKEN})"
)

# Spaces between the longest field in a run and its aligned inline comment.
_INLINE_COMMENT_GAP = 2


def _find_comment_index(line: str, comment_char: str = "!", escape_char: str = "\\") -> int | None:
    """
    Index of the comment character that starts a trailing comment, or ``None``.

    Quoted strings and escaped characters are respected, so a ``comment_char``
    inside quotes does not start a comment.
    """
    first = line.find(comment_char)
    if first < 0:
        return None
    prefix = line[:first]
    if "'" not in prefix and '"' not in prefix and escape_char not in prefix:
        return first

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
            return i
    return None


def _split_comment(line: str, comment_char: str = "!", escape_char: str = "\\") -> tuple[str, str]:
    """
    Split a line into its code part and its comment part (comment ``!`` dropped).

    Quoted strings and escape characters are respected. Neither part is
    stripped; the comment part is empty when the line has no comment.
    """
    idx = _find_comment_index(line, comment_char, escape_char)
    if idx is None:
        return line, ""
    return line[:idx], line[idx + 1 :]


def _split_field_comment(line: str) -> tuple[str, str]:
    """
    Split a line into its code part and its whole trailing comment.

    The returned comment keeps its leading ``!`` and is right-stripped; it is
    empty when the line has no comment.
    """
    idx = _find_comment_index(line)
    if idx is None:
        return line, ""
    return line[:idx], line[idx:].rstrip()


def _apply_field_case(key: str, case: str) -> str:
    """Case a namelist field name per ``case`` (``"lower"``/``"upper"``/other)."""
    if case == "lower":
        return key.lower()
    if case == "upper":
        return key.upper()
    return key


def _normalize_key(key: str) -> str:
    """Whitespace-insensitive, case-insensitive key form used for lookups."""
    return _RE_NORMALIZE_KEY.sub("", key).lower()


@dataclass(slots=True)
class _FieldRecord:
    """A ``key = value`` record being formatted, possibly spanning lines."""

    key: str
    chunks: list[str]  # value text per rendered row; chunks[0] follows "key = "
    comments: list[str]  # trailing comment per rendered row ("" when none)
    key_width: int = 0
    comment_column: int = 0

    def code_lines(self, indent: str) -> list[str]:
        """The rendered rows, without trailing comments."""
        first = f"{indent}{self.key.ljust(self.key_width)} = {self.chunks[0]}"
        # Continuation values align under the first value character.
        continuation = " " * (len(indent) + max(len(self.key), self.key_width) + 3)
        return [first, *(f"{continuation}{chunk}" for chunk in self.chunks[1:])]

    def render_row(self, index: int, indent: str) -> str:
        code = self.code_lines(indent)[index]
        comment = self.comments[index]
        if not comment:
            return code.rstrip()
        padded = code.ljust(self.comment_column) if self.comment_column else f"{code} "
        return f"{padded}{comment}"


def _format_group_lines(lines: list[str], options: NamelistFormatOptions) -> list[str]:
    """
    Format the source lines of one ``&name ... /`` group.

    ``key = value`` records are re-indented, their names cased per
    ``options.field_case``, and the spacing around ``=`` normalized. A record
    whose value continues across lines keeps its wrapping, continuation values
    aligned under the first value character; several records sharing one source
    line are split onto separate lines. Within each blank-line-delimited run of
    records, the ``=`` (when ``align_equals``) and the trailing ``!`` comments
    (when ``align_comments``) are padded into columns. The opener and
    terminator stay at column zero, blank lines stay empty, and comment-only
    lines are indented but do not participate in a run's alignment.
    """
    indent = options.indent_char * options.indent_size
    scan = _scan_namelist(lines)

    # anchors: physical line -> rows to emit there; line_rows: physical line ->
    # the (record, row) owning that line's trailing comment (last record wins);
    # consumed: lines whose code is re-rendered through a record.
    anchors: dict[int, list[tuple[_FieldRecord, int]]] = {}
    line_rows: dict[int, tuple[_FieldRecord, int]] = {}
    consumed: set[int] = set()
    records: list[_FieldRecord] = []

    for assignment in scan.assignments:
        start = assignment.span.line  # base_line=0: locations are line indices
        # ``str(value)`` is the raw field-to-field slice per physical line
        # (including any repeat-count prefix), joined with newlines.
        chunk_lines = sorted({token.loc.line for token in assignment.values})
        chunks = []
        for part, ln in zip(str(assignment.value).split("\n"), chunk_lines):
            if ln != chunk_lines[-1]:
                # Keep a trailing comma marking the continuation.
                end = max(t.loc.end_column for t in assignment.values if t.loc.line == ln)
                rest = lines[ln][end:].lstrip(" \t")
                if rest.startswith(","):
                    part += ","
            chunks.append(part)
        record = _FieldRecord(
            key=_apply_field_case(assignment.key, options.field_case),
            chunks=chunks or [""],
            comments=[""] * max(len(chunks), 1),
        )
        records.append(record)
        anchors.setdefault(start, []).append((record, 0))
        line_rows[start] = (record, 0)
        for row, ln in enumerate(chunk_lines):
            if row > 0:
                anchors.setdefault(ln, []).append((record, row))
            line_rows[ln] = (record, row)
        consumed.update(range(assignment.span.line, assignment.span.end_line + 1))

    for index, line in enumerate(lines):
        _, comment = _split_field_comment(line)
        comment = comment.strip()
        if not comment:
            continue
        owner = line_rows.get(index)
        if owner is not None:
            record, row = owner
            existing = record.comments[row]
            record.comments[row] = f"{existing} {comment}".strip() if existing else comment

    # A run is a maximal group of records uninterrupted by a blank line;
    # comment-only and unparsable lines neither join nor break it.
    runs: list[list[_FieldRecord]] = [[]]
    for index, line in enumerate(lines):
        if index in anchors:
            runs[-1].extend(record for record, row in anchors[index] if row == 0)
        elif not line.strip() and index not in consumed:
            runs.append([])

    for run in runs:
        if not run:
            continue
        if options.align_equals:
            width = max(len(r.key) for r in run)
            for r in run:
                r.key_width = width
        if options.align_comments:
            column = (
                max(len(code) for r in run for code in r.code_lines(indent)) + _INLINE_COMMENT_GAP
            )
            for r in run:
                r.comment_column = column

    terminator = scan.terminator
    opener = bool(lines) and lines[0].lstrip().startswith("&")
    opener_text = lines[0].strip() if lines else ""
    if opener and (0 in anchors or (terminator is not None and terminator[0] == 0)):
        # Records and/or the terminator share the opener line: keep only &name.
        match = _RE_GROUP_OPEN.match(lines[0])
        if match is not None:
            opener_text = f"&{match.group(1)}"

    out: list[str] = []
    for index, line in enumerate(lines):
        if index == 0 and opener:
            out.append(opener_text)
        for record, row in anchors.get(index, ()):
            out.append(record.render_row(row, indent))
        if terminator is not None and terminator[0] == index:
            out.append(line[terminator[1] :].rstrip())
            continue
        if (index == 0 and opener) or index in anchors:
            continue
        stripped = line.strip()
        if index in consumed:
            # Comment-only lines inside a record's span keep their place;
            # blank lines inside one are dropped.
            if stripped.startswith("!"):
                out.append(f"{indent}{stripped}")
            continue
        out.append("" if not stripped else f"{indent}{stripped}")
    return out


def split_values(value: Token) -> list[Token]:
    """
    Split a namelist value into its whitespace/comma-separated fields.

    An anonymous derived-type assignment such as
    ``datum(1) = 'orbit.x' '' '' 'R0\\2' 'target' 0 1e1`` packs several field
    values onto the right-hand side. Split them while keeping quoted strings
    (including empty ``''``) intact, carrying a source `Location` for each field
    derived from ``value``'s own location. A Fortran repeat count ``r*c`` (e.g.
    ``6*'beginning'`` or ``3*0``) expands to ``r`` copies of the constant.

    ``value`` is assumed to lie on a single source line; for parsed
    assignments (whose values may continue across lines), `Assignment.values`
    is authoritative and carries correct per-line locations.
    """
    loc = value.loc
    tokens: list[Token] = []
    for match in _RE_VALUE_FIELD.finditer(str(value)):
        start, end = match.span("value")
        count = int(match["count"]) if match["count"] else 1
        tokens.extend(
            Token(
                match["value"],
                loc=Location(
                    filename=loc.filename,
                    line=loc.line,
                    column=loc.column + start,
                    end_line=loc.line,
                    end_column=loc.column + end,
                ),
            )
            for _ in range(count)
        )
    return tokens


_RE_REPEAT_PREFIX = re.compile(r"\d+\*")
# Fast path for `_read_field`: a bare run with none of the characters that
# need the stateful scan (separators, '=', '/', parens, quotes).
_RE_PLAIN_FIELD = re.compile(r"""[^ \t,=/()'"]+""")


@dataclass(frozen=True, slots=True)
class _ScanToken:
    """One lexical token of a group's comment-stripped source lines."""

    kind: Literal["open", "eq", "slash", "field"]
    text: str
    line: int  # index into the scanned lines, not the source file
    column: int
    end_column: int


def _read_quoted(code: str, start: int) -> int:
    """Index one past the quoted string starting at ``code[start]`` (a quote)."""
    quote = code[start]
    i = start + 1
    while i < len(code):
        if code[i] == quote:
            if code[i + 1 : i + 2] == quote:  # doubled-quote escape, e.g. 'it''s'
                i += 2
                continue
            return i + 1
        i += 1
    return len(code)  # unterminated: runs to the end of the line


def _read_field(code: str, start: int) -> int:
    """
    Index one past the field token starting at ``start``.

    A quoted string is a single token. A bare token runs until a separator,
    ``=``, ``/``, or quote at parenthesis depth zero; within ``(...)`` nothing
    separates, so subscripts like ``datum( 1)`` and ``var(1 : 6)%x`` stay
    whole. Blanks inside a designator (``datum (1)``, ``a % b``) are stepped
    over, and a Fortran repeat count directly followed by a quote (``6*'x'``)
    continues into the string.
    """
    if code[start] in "'\"":
        return _read_quoted(code, start)

    match = _RE_PLAIN_FIELD.match(code, start)
    if match is not None:
        end = match.end()
        follow = code[end : end + 1]
        if follow in ",=/" or not follow:
            return end
        if follow in " \t" and code[end - 1] != "%":
            # Only a designator continuation ("name (1)", "a % b") reads past
            # the blank; otherwise the token ends here.
            j = end + 1
            while j < len(code) and code[j] in " \t":
                j += 1
            if j >= len(code) or code[j] not in "(%":
                return end
        # Parens, quotes, or a designator continuation: take the stateful scan.

    depth = 0
    i = start
    while i < len(code):
        char = code[i]
        if depth == 0:
            if char in " \t":
                j = i
                while j < len(code) and code[j] in " \t":
                    j += 1
                if j < len(code) and (code[j] in "(%" or code[i - 1] == "%"):
                    i = j
                    continue
                break
            if char in ",=/":
                break
            if char in "'\"":
                if _RE_REPEAT_PREFIX.fullmatch(code, start, i):
                    return _read_quoted(code, i)
                break
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        i += 1
    return i


def _scan_line(code: str) -> Iterator[tuple[str, int, int]]:
    """Yield ``(kind, start, end)`` lexical tokens of one comment-stripped line."""
    i = 0
    while i < len(code):
        char = code[i]
        if char in " \t,":
            i += 1
        elif char == "=":
            yield ("eq", i, i + 1)
            i += 1
        elif char == "/":
            yield ("slash", i, i + 1)
            return  # the group ends here; the rest of the line is not code
        else:
            end = _read_field(code, i)
            yield ("field", i, end)
            i = end


def _find_terminator(line: str) -> int | None:
    """Column of a group-terminating ``/`` on this line, or ``None``.

    Comment- and quote-aware: a ``/`` inside a string or after a ``!`` does
    not terminate.
    """
    if "/" not in line:
        return None
    code, _ = _split_comment(line)
    for kind, start, _end in _scan_line(code):
        if kind == "slash":
            return start
    return None


def _scan_group(lines: list[str]) -> Iterator[_ScanToken]:
    """
    Lex a group's source lines into `_ScanToken`s.

    Comments are stripped per line. The first token, when it starts with
    ``&``, is the group opener. Scanning stops at the terminating ``/``.
    """
    first = True
    for index, line in enumerate(lines):
        code, _ = _split_comment(line)
        for kind, start, end in _scan_line(code):
            if kind == "field" and first and code[start] == "&":
                kind = "open"
            first = False
            yield _ScanToken(kind, code[start:end], index, start, end)
            if kind == "slash":
                return


def _build_assignment(
    key: _ScanToken,
    eq: _ScanToken,
    fields: list[_ScanToken],
    lines: list[str],
    filename: pathlib.Path | None,
    base_line: int,
) -> Assignment:
    """Assemble an `Assignment` from its key/``=``/value tokens."""
    values: list[Token] = []
    for token in fields:
        text = token.text
        if "*" not in text:  # no repeat-count prefix to strip (the common case)
            start, end, count = 0, len(text), 1
        else:
            match = _RE_VALUE_FIELD.fullmatch(text)
            if match is None:
                start, end, count = 0, len(text), 1
            else:
                start, end = match.span("value")
                count = int(match["count"]) if match["count"] else 1
        loc = Location(
            filename=filename,
            line=base_line + token.line,
            column=token.column + start,
            end_line=base_line + token.line,
            end_column=token.column + end,
        )
        if count == 1:
            values.append(Token(text[start:end], loc=loc))
        else:
            values.extend(Token(text[start:end], loc=loc) for _ in range(count))

    if fields:
        first, last = fields[0], fields[-1]
        if first.line == last.line:
            text = lines[first.line][first.column : last.end_column]
        else:
            parts: list[str] = []
            for index in dict.fromkeys(token.line for token in fields):
                row = [token for token in fields if token.line == index]
                parts.append(lines[index][row[0].column : row[-1].end_column])
            text = "\n".join(parts)
        value = Token(
            text,
            loc=Location(
                filename=filename,
                line=base_line + first.line,
                column=first.column,
                end_line=base_line + last.line,
                end_column=last.end_column,
            ),
        )
    else:
        # Empty right-hand side (``x =``): a zero-width token just after the '='.
        line = base_line + eq.line
        loc = Location(
            filename=filename,
            line=line,
            column=eq.end_column,
            end_line=line,
            end_column=eq.end_column,
        )
        value = Token("", loc=loc)

    return Assignment(
        key=_RE_NORMALIZE_KEY.sub("", key.text),
        value=value,
        values=values,
        key_loc=Location(
            filename=filename,
            line=base_line + key.line,
            column=key.column,
            end_line=base_line + key.line,
            end_column=key.end_column,
        ),
    )


@dataclass
class _GroupScan:
    """Parsed content of one ``&name ... /`` group's source lines."""

    assignments: list[Assignment]
    terminator: tuple[int, int] | None  # (line index, column) of the '/'


def _scan_namelist(
    lines: list[str],
    *,
    filename: pathlib.Path | None = None,
    base_line: int = 0,
) -> _GroupScan:
    """
    Parse a group's source lines into `Assignment`s with true source spans.

    Namelist input is free-form: values may continue across lines, several
    ``key = value`` pairs may share a line, assignments may sit on the
    ``&name`` opener line, and ``/`` terminates the group anywhere outside a
    quoted string. Trailing comments attach to the last assignment whose span
    covers their line.
    """
    assignments: list[Assignment] = []
    pending: list[_ScanToken] = []
    key: _ScanToken | None = None
    eq: _ScanToken | None = None
    terminator: tuple[int, int] | None = None

    def finalize() -> None:
        if key is not None and eq is not None:
            assignments.append(_build_assignment(key, eq, pending, lines, filename, base_line))
        pending.clear()

    for token in _scan_group(lines):
        if token.kind == "field":
            pending.append(token)
        elif token.kind == "eq":
            # The token just before '=' keys the next assignment; the rest of
            # the pending fields close out the previous one. A stray '=' (no
            # preceding token, or a quoted string) is ignored.
            if pending and pending[-1].text[0] not in "'\"":
                next_key = pending.pop()
                finalize()
                key, eq = next_key, token
        elif token.kind == "slash":
            terminator = (token.line, token.column)
            break
    finalize()

    if assignments:
        # (start line, end line) span bounds, avoiding Location construction.
        bounds = [
            (a.key_loc.line if a.key_loc is not None else a.value.loc.line, a.value.loc.end_line)
            for a in assignments
        ]
        for index, line in enumerate(lines):
            if "!" not in line:
                continue
            _, comment = _split_comment(line)
            comment = comment.strip()
            if not comment:
                continue
            absolute = base_line + index
            for position in range(len(assignments) - 1, -1, -1):
                start, end = bounds[position]
                if start <= absolute <= end:
                    assignment = assignments[position]
                    assignment.comment = f"{assignment.comment} {comment}".strip()
                    break

    return _GroupScan(assignments=assignments, terminator=terminator)


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

    @property
    def indices(self) -> list[int] | None:
        """
        The explicit integer indices this component designates.

        ``[i]`` for a single subscript ``name(i)``; the expanded, ``stop``-
        inclusive range for a Fortran array section ``name(a:b)`` or
        ``name(a:b:step)``. ``None`` when there is no subscript or it cannot be
        enumerated (open-ended range, non-integer bound, a named parameter,
        non-positive stride, ...).
        """
        text = self.index_text
        if text is None:
            return None
        if _RE_INT.fullmatch(text):
            return [int(text)]
        match = _RE_SLICE.fullmatch(text)
        if match is None:
            return None
        start, stop = int(match[1]), int(match[2])
        step = int(match[3]) if match[3] else 1
        if step <= 0:
            return None
        return list(range(start, stop + 1, step))

    @property
    def slice_start(self) -> int | None:
        """The ``a`` of an open-ended array section ``name(a:)``, or ``None``."""
        text = self.index_text
        if text is None:
            return None
        match = _RE_OPEN_SLICE.fullmatch(text)
        return int(match[1]) if match is not None else None

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
    """
    A single ``key = value`` assignment within a namelist group.

    Values are free-form: they may continue across lines and several
    assignments may share one line.

    Attributes
    ----------
    key : str
        The whitespace-normalized key (e.g. ``datum(1)%ele_name``).
    value : Token
        The value text. Its ``loc`` spans from the first character of the
        first value field through the last character of the last one
        (multi-line when the value continues across lines). Per physical
        line, ``str(value)`` is the raw source slice from that line's first
        field to its last, joined with newlines — so for single-line values
        ``str(value) == value.loc.get_string(source)`` exactly.
    values : list[Token]
        The individual value fields with per-field source locations; Fortran
        repeat counts (``3*0``) are expanded.
    comment : str
        Trailing ``!`` comments on the lines the assignment spans.
    key_loc : Location or None
        Source location of the key.
    """

    key: str
    value: Token
    values: list[Token] = field(default_factory=list)
    comment: str = ""
    key_loc: Location | None = None

    @property
    def loc(self) -> Location:
        """Source location of the value (``value.loc``)."""
        return self.value.loc

    @property
    def span(self) -> Location:
        """Source span from the start of the key through the last value character."""
        if self.key_loc is None:
            return self.value.loc
        return self.value.loc + self.key_loc

    @property
    def path(self) -> KeyPath:
        """The key decomposed into `KeyComponent` parts (any nesting depth)."""
        return KeyPath.parse(self.key)


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
    # (line index into ``lines``, column) of the terminating '/', or None.
    _terminator: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._reparse()

    def _reparse(self) -> None:
        scan = _scan_namelist(self.lines, filename=self.filename, base_line=self.start_line)
        self.assignments = scan.assignments
        self._terminator = scan.terminator

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

    def _indent(self) -> str:
        """Indentation to use for inserted lines, mirroring existing entries."""
        for assignment in self.assignments:
            line = self.lines[assignment.span.line - self.start_line]
            if line.lstrip().startswith("&"):
                continue  # assignments on the opener line carry no useful indent
            return line[: len(line) - len(line.lstrip())]
        return "  "

    def _insert_before_terminator(self, new_line: str) -> None:
        if self._terminator is None:
            self.lines.append(new_line)
            return
        index, column = self._terminator
        line = self.lines[index]
        if line[:column].strip():
            # One-line group: split the code before '/' onto its own line.
            self.lines[index : index + 1] = [line[:column].rstrip(), new_line, line[column:]]
        else:
            self.lines.insert(index, new_line)

    def set(self, key: str, value: str, *, comment: str = "") -> Assignment:
        """
        Update ``key``'s value in place, or append it before the terminator.

        A value that continues across several lines is collapsed onto its
        first line (interior comments on those lines are dropped with them).
        """
        existing = self.get(key)
        if existing is not None:
            loc = existing.value.loc
            first = loc.line - self.start_line
            last = loc.end_line - self.start_line
            spliced = self.lines[first][: loc.column] + value + self.lines[last][loc.end_column :]
            self.lines[first : last + 1] = spliced.split("\n")
        else:
            new_line = f"{self._indent()}{key} = {value}"
            if comment:
                if not comment.lstrip().startswith("!"):
                    comment = f"!{comment}"
                new_line = f"{new_line} {comment}"
            self._insert_before_terminator(new_line)

        self._reparse()
        return cast(Assignment, self.get(key))

    def remove(self, key: str) -> None:
        """
        Remove ``key``'s assignment, including continuation lines (no-op if absent).

        Lines the assignment shares with the opener, the terminator, or another
        assignment are spliced rather than deleted.
        """
        existing = self.get(key)
        if existing is None:
            return
        span = existing.span
        first = span.line - self.start_line
        last = span.end_line - self.start_line
        others = [a for a in self.assignments if a is not existing]

        def shared(index: int) -> bool:
            absolute = index + self.start_line
            return any(a.span.line <= absolute <= a.span.end_line for a in others)

        for index in range(last, first - 1, -1):
            line = self.lines[index]
            protected = (
                index == 0
                or (self._terminator is not None and self._terminator[0] == index)
                or shared(index)
            )
            if not protected:
                del self.lines[index]
                continue
            start = span.column if index == first else 0
            end = span.end_column if index == last else len(line)
            while end < len(line) and line[end] in " \t,":
                end += 1
            remainder = (line[:start] + line[end:]).rstrip()
            if remainder:
                self.lines[index] = remainder
            else:
                del self.lines[index]
        self._reparse()

    def render(self, options: NamelistFormatOptions | None = None) -> str:
        """
        Render the group's source lines.

        With ``options`` given, fields are re-indented, cased, and aligned per
        those options; otherwise the verbatim source lines are returned.
        """
        if options is None:
            return "\n".join(self.lines)
        return "\n".join(_format_group_lines(self.lines, options))


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

        def close() -> None:
            nonlocal current
            if current is not None:
                current._reparse()
                items.append(current)
                current = None

        for idx, line in enumerate(text.split("\n")):
            stripped = line.lstrip()
            if current is None:
                if stripped.startswith("&"):
                    flush_raw()
                    name = _RE_GROUP_OPEN.match(line).group(1)  # type: ignore[union-attr]
                    current = Namelist(name=name, lines=[line], filename=path, start_line=idx)
                else:
                    raw.append(line)
                    continue
            else:
                current.lines.append(line)
            # '/' terminates the group anywhere outside a quoted string, even
            # on the opener line; anything after it on the line stays verbatim.
            if _find_terminator(line) is not None:
                close()
        if current is not None:
            current._reparse()
            items.append(current)
        flush_raw()
        return cls(items=items, filename=path)

    @classmethod
    def from_file(cls, path: pathlib.Path | str) -> Self:
        path = pathlib.Path(path)
        return cls.parse(path.read_text(), filename=path)

    def _render_with_options(self, options: NamelistFormatOptions) -> str:
        out = []
        for index, item in enumerate(self.items):
            is_namelist = isinstance(item, Namelist)
            if is_namelist:
                lines = item.render(options).split("\n")
            else:
                lines = item.split("\n")
                after_group = index > 0 and isinstance(self.items[index - 1], Namelist)
                if options.blank_line_after_group and after_group:
                    # The single separating blank is re-added below; drop the
                    # source's own leading blanks so runs collapse to one.
                    while lines and not lines[0].strip():
                        lines.pop(0)
            out.extend(lines)
            last = index == len(self.items) - 1
            if options.blank_line_after_group and is_namelist and not last:
                out.append("")
        return "\n".join(out)

    def render(self, options: NamelistFormatOptions | None = None) -> str:
        """
        Render the whole Namelist file.

        Without ``options`` the source is reproduced verbatim. When specified,
        the output Namelist file will be formatted according to the provided
        options.
        """
        if options is None:
            out: list[str] = []
            for item in self.items:
                if isinstance(item, Namelist):
                    out.extend(item.lines)
                else:
                    out.extend(item.split("\n"))
            return "\n".join(out)

        return self._render_with_options(options)

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


@dataclass
class NamelistArrayEntry:
    """
    One indexed entry of a namelist array-of-derived-type.

    Tao packs an array such as ``datum`` or ``var`` two ways, and both are
    merged here:

    * anonymously/positionally --- ``datum(1) = 'orbit.x' '' '' 'END\\2' ...``,
      whose values fill :data:`FIELDS` in declaration order;
    * by component --- ``datum(1)%ele_name = 'END\\2'``.

    An explicit component takes precedence over the same positional slot.

    Attributes
    ----------
    index : int or None
        The ``i`` in ``name(i)``. ``0`` is the conventional slot Tao reads for
        ``SEARCH:``/``SAME:`` element specifications; ``None`` for a
        non-integer subscript (e.g. a range).
    positional : list[Token]
        Values from an anonymous ``name(i) = ...`` assignment, in field order.
    components : dict[str, Token]
        Values from ``name(i)%field = ...`` assignments, keyed by field name.
    comment : str
        Trailing comment on the anonymous assignment, if any.
    """

    #: Component names, in declaration order, that a positional assignment fills.
    FIELDS: ClassVar[tuple[str, ...]] = ()

    index: int | None
    positional: list[Token] = field(default_factory=list)
    components: dict[str, Token] = field(default_factory=dict)
    comment: str = ""

    def get(self, name: str) -> Token | None:
        """The raw (quoted, if a string) value token for ``name``, or ``None``."""
        name = name.lower()
        if name in self.components:
            return self.components[name]
        if name in self.FIELDS:
            position = self.FIELDS.index(name)
            if position < len(self.positional):
                return self.positional[position]
        return None

    def value(self, name: str) -> Token | None:
        """
        The value token for ``name``, unquoted.

        ``None`` if unset; an empty (``''``) token if explicitly blank. The
        returned token keeps its source `Location`.
        """
        token = self.get(name)
        return token.strip().remove_quotes() if token is not None else None


@dataclass
class NamelistArrayGroup:
    """
    Base for a namelist group that carries an indexed array of derived types.

    Subclasses wrap a single `latform._namelist.Namelist` (e.g. a
    ``&tao_d1_data`` or ``&tao_var`` block), exposing its scalar settings and
    the parsed array entries.
    """

    namelist: Namelist

    def _scalar(self, key: str) -> Token | None:
        assignment = self.namelist.get(key)
        return assignment.value.strip().remove_quotes() if assignment is not None else None

    def _entries(self, array_name: str, entry_cls: type[NamelistArrayEntry]) -> list:
        """
        Parse ``array_name(i)`` positional/component assignments into entries.

        A Fortran array-section component assignment,
        ``array_name(a:b)%field = v_a, v_b, ...``, is expanded: the values are
        distributed across ``field`` of entries ``a`` through ``b`` in order.
        """
        array_name = array_name.lower()
        by_index: dict[object, NamelistArrayEntry] = {}

        def entry_for(key: object, index: int | None) -> NamelistArrayEntry:
            existing = by_index.get(key)
            if existing is None:
                existing = entry_cls(index=index)
                by_index[key] = existing
            return existing

        for assignment in self.namelist.assignments:
            path = assignment.path
            if path.names[0] != array_name:
                continue
            component = path.components[0]
            indices = component.indices

            # An open-ended section ``name(a:)`` takes its extent from the
            # number of supplied values.
            if indices is None and len(path.components) > 1:
                start = component.slice_start
                if start is not None:
                    indices = list(range(start, start + len(assignment.values)))

            # An empty array section (e.g. an ascending ``5:3`` slice) designates
            # no elements, so the assignment is a no-op.
            if indices is not None and not indices:
                continue

            # Array-section component assignment: distribute the values across
            # the designated indices (e.g. ``var(1:6)%ele_name = a, b, ...``).
            if len(path.components) > 1 and indices is not None and len(indices) > 1:
                field_name = path.names[1]
                for value, index in zip(assignment.values, indices):
                    entry_for(index, index).components[field_name] = value
                continue

            # Single element (``name(i)``), a section that names one index, or a
            # non-enumerable/positional subscript.
            if indices is not None and len(indices) == 1:
                key = index = indices[0]
            else:
                index = component.index
                key = component.index if component.index is not None else component.index_text
            entry = entry_for(key, index)
            if len(path.components) == 1:
                entry.positional = list(assignment.values)
                entry.comment = assignment.comment
            else:
                entry.components[path.names[1]] = assignment.value

        def sort_key(key: object) -> tuple[int, object]:
            return (0, key) if isinstance(key, int) else (1, str(key))

        return [by_index[key] for key in sorted(by_index, key=sort_key)]
