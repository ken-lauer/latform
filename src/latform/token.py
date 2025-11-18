from __future__ import annotations

import enum
import typing
from typing import Any, Sequence

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .comments import Comments
from .location import Location

if typing.TYPE_CHECKING:
    from .types import Block


class Role(str, enum.Enum):
    statement_definition = "statement_definition"


class Token(str):
    """
    String with source loc information.

    Comparisons are case-insensitive.
    """

    loc: Location
    comments: Comments
    role: Role | None = None

    def __new__(
        cls,
        content: str,
        loc: Location | None = None,
        comments: Comments | None = None,
        role: Role | None = None,
    ):
        return super().__new__(cls, content)

    def __init__(
        self,
        content: str,
        loc: Location | None = None,
        comments: Comments | None = None,
        role: Role | None = None,
    ):
        self.loc = loc or Location(end_column=len(content))
        self.comments = comments or Comments()
        self.role = role

        # internal error
        if not isinstance(self.loc, Location):
            raise ValueError(type(self.loc))
        if not isinstance(self.comments, Comments):
            raise ValueError(type(self.comments))
        if role and not isinstance(self.role, Role):
            raise ValueError(type(self.role))

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other) -> bool:
        case_insensitive_equality = str(other).upper() == str(self).upper()
        if hasattr(other, "comments"):
            return case_insensitive_equality and self.comments == other.comments

        return case_insensitive_equality

    @property
    def is_quoted_string(self) -> bool:
        return (self.startswith(_SQUOTE) and self.endswith(_SQUOTE)) or (
            self.startswith(_DQUOTE) and self.endswith(_DQUOTE)
        )

    def remove_quotes(self) -> Token:
        if self.is_quoted_string:
            # TODO fix location
            return Token(self[1:-1], loc=self.loc, comments=self.comments)
        return self

    def __repr__(self) -> str:
        # return super().__repr__()
        # parts = [super().__repr__(), self.loc]
        parts: list[Any] = [super().__repr__()]
        if bool(self.comments):
            parts.append(self.comments)
        desc = ", ".join(str(part) for part in parts if part)
        return f"{type(self).__name__}({desc})"

    @classmethod
    def join(cls: type[Token], args: Sequence[Token | Block], delim: Delimiter | None = None):
        if not args:
            return cls("")

        from .types import _flatten_blocks

        strs = _flatten_blocks(list(args))
        if not strs:
            return cls("")

        str0, *_ = strs
        if not str0.loc:
            return cls("".join(str(s) for s in strs))

        result_parts = [str(str0)]

        if delim and len(strs) > 1:
            result_parts.append(str(delim))
            result_parts.append(" ")

        prev_end_line = str0.loc.end_line or str0.loc.line
        prev_end_col = str0.loc.end_column or (str0.loc.column + len(str(str0)))

        for idx, arg in enumerate(strs[1:], start=2):
            if not arg.loc:
                result_parts.append(str(arg))
                if delim and idx < len(strs):
                    result_parts.append(str(delim))
                    result_parts.append(" ")
                continue

            # Different line: add a space
            if arg.loc.line != prev_end_line:
                result_parts.append(" ")
            # Same line but column is beyond where previous token ended: add a space
            elif arg.loc.column > prev_end_col:
                result_parts.append(" ")

            result_parts.append(str(arg))
            if delim and idx < len(strs):
                result_parts.append(str(delim))
                result_parts.append(" ")

            prev_end_line = arg.loc.end_line or arg.loc.line
            prev_end_col = arg.loc.end_column or (arg.loc.column + len(str(arg)))

        return cls(
            "".join(result_parts),
            loc=Location.from_items(strs),
            comments=str0.comments,
        )

    def replace(self, old, new, count=-1) -> Self:
        return type(self)(
            str(self).replace(old, new, count),
            loc=self.loc,
            comments=self.comments,
            role=self.role,
        )


class Delimiter(Token):
    pass


_DQUOTE = Delimiter('"')
_SQUOTE = Delimiter("'")
