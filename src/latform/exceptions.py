from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .types import Delimiter


class EndOfLine(Exception):
    pass


class ParsingError(Exception):
    pass


class UnexpectedCallName(ParsingError):
    pass


class UnexpectedAssignment(ParsingError):
    pass


class MismatchedDelimiter(ParsingError):
    open: Delimiter
    close: Delimiter

    def __init__(self, msg: str, open: Delimiter, close: Delimiter) -> None:
        super().__init__(msg)
        self.open = open
        self.close = close


class UnterminatedString(ParsingError):
    delim: Delimiter

    def __init__(self, msg: str, delim: Delimiter) -> None:
        super().__init__(msg)
        self.delim = delim


class ExtraCloseDelimiter(ParsingError):
    delim: Delimiter

    def __init__(self, msg: str, delim: Delimiter) -> None:
        super().__init__(msg)
        self.delim = delim


class MissingCloseDelimiter(ParsingError):
    delims: list[Delimiter]

    def __init__(self, msg: str, delims: list[Delimiter]) -> None:
        super().__init__(msg)
        self.delims = list(delims)


class ParticleSpeciesError(ParsingError):
    pass


class InvalidChargeError(ParticleSpeciesError):
    pass
