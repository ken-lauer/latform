from __future__ import annotations

import pathlib
import typing
from typing import Literal, TypeVar

from .const import CLOSE_TO_OPEN, COMMA
from .exceptions import ExtraCloseDelimiter, MismatchedDelimiter
from .token import Delimiter, Token

if typing.TYPE_CHECKING:
    from .types import Attribute, CallName, Seq


class DelimiterState:
    def __init__(self):
        self._stack: list[Delimiter] = []

    @property
    def depth(self) -> int:
        return len(self._stack)

    def update(self, delim: Delimiter) -> tuple[Delimiter | None, int]:
        assert isinstance(delim, Delimiter)
        if delim in "({[":
            self._stack.append(delim)
            return delim, 1

        if delim in CLOSE_TO_OPEN:
            try:
                opening_delim = self._stack.pop(-1)
            except IndexError:
                raise ExtraCloseDelimiter(
                    f"Unmatched closing delimiter: {delim!r} at {delim.loc}", delim=delim
                )

            expected_open = CLOSE_TO_OPEN[delim]
            if opening_delim != expected_open:
                raise MismatchedDelimiter("Mismatched delimiter", open=opening_delim, close=delim)
            return opening_delim, -1

        return None, 0

    @property
    def current_delimiters(self) -> list[Delimiter]:
        return list(self._stack)


def delimit(items, delimiter: Delimiter | None, trailing: bool = False):
    if not items:
        return

    for item in items[:-1]:
        yield item
        if delimiter is not None:
            yield delimiter
    yield items[-1]
    if trailing:
        yield delimiter


def comma_delimit(items, trailing: bool = False):
    yield from delimit(items, COMMA, trailing=trailing)


T = TypeVar("T")


def split_items(items: list[T], delimiter: Delimiter = COMMA) -> list[list[T]]:
    res = []
    part = []
    for item in items:
        if item == delimiter:
            res.append(part)
            part = []
        else:
            part.append(item)

    if part:
        res.append(part)

    return res


def partition_items(
    items: list[T],
    delimiter: Delimiter | str = COMMA,
) -> tuple[list[T], Delimiter, list[T]]:
    delimiter = Delimiter(delimiter)
    try:
        pos = items.index(delimiter)
    except ValueError as ex:
        raise ValueError(f"Delimiter {delimiter} not found") from ex

    delim = items[pos]
    assert isinstance(delim, Delimiter)
    return items[:pos], delim, items[pos + 1 :]


def flatten(item: Seq | Attribute | Token | CallName | None) -> list[Token]:
    """Flatten an Array to a list of Tokens."""
    from .types import Attribute, CallName, Seq

    if isinstance(item, Seq):
        return item.flatten()

    res = []
    if isinstance(item, (CallName, Attribute)):
        for node in item.to_output_nodes():
            res.extend(flatten(node))
    elif isinstance(item, Token):
        res.append(item)
    elif item is None:
        pass
    else:
        raise NotImplementedError(type(item))

    return res


LoadFormat = Literal["json", "toml", "yaml"]


def load_json_or_similar(fn: pathlib.Path | str, *, format: LoadFormat | None = None):
    path = pathlib.Path(fn)
    suffix = path.suffix.lower()

    if not format:
        if suffix in {".yaml", ".yml"}:
            format = "yaml"
        elif suffix in {".json"}:
            format = "json"
        elif suffix in {".toml"}:
            format = "toml"
        else:
            raise ValueError(
                f"File format not specified and file extension not recognized: {suffix} ({fn}"
            )

    if format == "yaml":
        import yaml

        loader = yaml.safe_load

    elif format == "toml":
        try:
            import tomllib
        except ImportError:  # python 3.10 (tomllib was based on tomli)
            import tomli as tomllib

        loader = tomllib.load

    elif format == "json":
        import json

        loader = json.load
    else:
        raise NotImplementedError(f"Format {format}")

    with open(path, "rb") as fp:
        return loader(fp)
