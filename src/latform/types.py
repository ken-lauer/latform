from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from .const import COMMA, EQUALS, SPACE
from .exceptions import UnexpectedCallName
from .token import Comments, Delimiter, Location, Token
from .util import comma_delimit, delimit, flatten, partition_items, split_items


def _flatten_blocks(
    items: list[Block | Token],
) -> list[Token]:
    res = []
    for item in items:
        if isinstance(item, Block):
            if item.opener:
                res.append(item.opener)
            res.extend(_flatten_blocks(item.items))
            if item.closer:
                res.append(item.closer)
        else:
            res.append(item)

    return res


@dataclass
class Seq:
    """
    Ordered sequence of mixed items:
    * Attribute (a named value, i.e., a name=value pair)
    * Expression (may be a single token)
    * Array (a nested array)
    """

    opener: Delimiter | None = None
    closer: Delimiter | None = None
    items: list[SequencePart] = field(default_factory=list)

    delimiter: Delimiter | None = SPACE

    def maybe_unwrap(self) -> tuple[list[SequencePart], Delimiter | None] | tuple[Self, None]:
        if not self.opener and not self.closer:
            return self.items, self.delimiter
        return self, None

    def _in_expected_delimiter(
        self, opener: Delimiter, closer: Delimiter, expected_delimiter: Delimiter
    ) -> Self:
        unwrapped, inner_delimiter = self.maybe_unwrap()
        assert unwrapped is not None
        assert not isinstance(unwrapped, Seq)
        if inner_delimiter == expected_delimiter or not self.items:
            # If the sequence is already delimited as expected, give it back directly
            items = unwrapped
        else:
            # Otherwise, we need to wrap that sequence in the implicit delimiter
            # That is, even if there is no comma found in {a}, that block is still technically comma-delimited
            items = [self]

        return type(self)(
            opener=opener,
            closer=closer,
            items=list(items),
            delimiter=expected_delimiter,
        )

    @classmethod
    def from_item(cls: type[Self], item: TokenizerItem) -> Self | Seq | Token:
        if isinstance(item, Block):
            inner = cls.from_items(item.items)

            if item.opener is None or item.closer is None:
                raise ValueError("Internal error - item opener/closer delimiter unset")

            implicit_delimiter = {"{": COMMA, "(": COMMA, "[": SPACE}[item.opener]
            if isinstance(inner, Seq):
                return inner._in_expected_delimiter(
                    opener=item.opener,
                    closer=item.closer,
                    expected_delimiter=implicit_delimiter,
                )

            if isinstance(inner, (Attribute, Token)):
                inner = [inner]
                return cls(
                    opener=item.opener,
                    closer=item.closer,
                    items=list(inner),
                    delimiter=implicit_delimiter,
                )

            raise NotImplementedError(type(inner))

        if isinstance(item, Token):
            return item
        raise NotImplementedError(type(item))

    @classmethod
    def from_delimited_block(cls: type[Self], item: Block, delimiter: Delimiter) -> Self | Seq:
        inner = cls.from_items(item.items)
        if isinstance(inner, Seq):
            if item.opener is None or item.closer is None:
                raise ValueError("Internal error - item opener/closer delimiter unset")

            return inner._in_expected_delimiter(
                opener=item.opener,
                closer=item.closer,
                expected_delimiter=delimiter,
            )

        elif isinstance(inner, (Attribute, Token)):
            inner = [inner]
        else:
            raise NotImplementedError(type(inner))

        return cls(opener=item.opener, closer=item.closer, items=list(inner), delimiter=delimiter)

    @classmethod
    def from_items(cls: type[Self], items: list[TokenizerItem]) -> Self | Attribute | Token:
        if len(items) == 1:
            return cls.from_item(items[0])

        if COMMA in items:
            parts = split_items(items, delimiter=COMMA)
            return cls(items=[Seq.from_items(part) for part in parts], delimiter=COMMA)

        if EQUALS in items:
            left, _, right = partition_items(items, EQUALS)
            right_seq = Seq.from_items(right)
            name = Seq.from_items(left)
            if not isinstance(name, (Token, Seq)):
                raise ValueError(f"Unexpected attribute name in sequence from tokens: {name}")
            if isinstance(name, Seq):
                try:
                    name = name.to_call_name()
                except UnexpectedCallName:
                    pass

            if isinstance(right_seq, Attribute):
                raise ValueError(f"Unexpected nested attribute found: {right_seq}")
            return Attribute(name=name, value=right_seq)

        res = cls(items=[Seq.from_items([item]) for item in items], delimiter=SPACE)
        _filter_species_calls(res.items)
        return res

    def to_output_nodes(self):
        # return [self]
        if self.opener and self.closer:
            return [self.opener, *comma_delimit(self.items), self.closer]
        return list(delimit(self.items, self.delimiter))

    def to_call_name(self) -> CallName:
        """Convert Array to a single Token."""
        match self.items:
            case Token() as name, Seq(opener="(") as args:
                return CallName(name=name, args=args)
        raise UnexpectedCallName(
            f"Expected function call pattern not matched: {self} at {self.loc}"
        )

    @property
    def loc(self) -> Location:
        if self.opener and self.closer:
            return Location(
                filename=self.opener.loc.filename,
                line=self.opener.loc.line,
                column=self.opener.loc.column,
                end_line=self.closer.loc.end_line,
                end_column=self.closer.loc.end_column,
            )
        assert len(self.items)
        return Location(
            filename=self.items[0].loc.filename,
            line=self.items[0].loc.line,
            column=self.items[0].loc.column,
            end_line=self.items[-1].loc.end_line,
            end_column=self.items[-1].loc.end_column,
        )

    def to_token(self) -> Token:
        """Convert Array to a single Token."""
        return Token.join(self.flatten())

    def flatten(self) -> list[Token]:
        res = []
        if self.opener:
            res.append(self.opener)

        for idx, item in enumerate(self.items):
            res.extend(flatten(item))
            if self.delimiter is not None and idx < len(self.items) - 1:
                if self.delimiter == SPACE:
                    # TODO
                    continue
                res.append(self.delimiter)

        if self.opener:
            res.append(self.closer)
        return res


def _filter_species_calls(items):
    # A hack: species names have a bunch of delimiters in them.
    # They appear to only be used in certain scenarios. Let's see if this holds.
    # It's not entirely different from what Bmad does under the hood (see
    # bmad_expression_mod.f90 reverse_polish_pass)
    for idx, item in enumerate(items[:-1]):
        if isinstance(item, Seq):
            _filter_species_calls(item.items)
        elif isinstance(item, Token) and item.lower() in {
            "mass_of",
            "charge_of",
            "anomalous_moment_of",
            "species",
        }:
            nxt = items[idx + 1]
            if (
                isinstance(nxt, Seq)
                and nxt.opener == "("
                and all(isinstance(inner, Token) for inner in nxt.items)
            ):
                new_inner = [Token.join(list(flatten(nxt))[1:-1])]
                nxt.items = new_inner
            elif (
                isinstance(nxt, Seq)
                and nxt.opener == "("
                and len(nxt.items) == 1
                and isinstance(nxt.items[0], Seq)
                and all(isinstance(inner, Token) for inner in nxt.items[0].items)
                and not nxt.items[0].opener
            ):
                new_inner = [Token.join(list(flatten(nxt.items[0])))]
                nxt.items = new_inner


@dataclass
class CallName:
    name: Token | Seq
    args: Seq

    @property
    def loc(self) -> Location:
        return Location.from_items((self.name, self.args))

    def to_output_nodes(self):
        return [self.name, self.args]


@dataclass
class Attribute:
    name: Token | CallName | Seq
    value: Seq | Token | None = None

    @property
    def loc(self) -> Location:
        return Location.from_items((self.name, self.value))

    def to_output_nodes(self):
        if self.value is not None:
            return [self.name, EQUALS, self.value]
        return [self.name]


@dataclass
class Block:
    opener: Delimiter | None = None
    closer: Delimiter | None = None
    items: list[Token | Block | Delimiter] = field(default_factory=list)

    @property
    def loc(self) -> Location:
        if self.opener and self.closer:
            return Location(
                filename=self.opener.loc.filename,
                line=self.opener.loc.line,
                column=self.opener.loc.column,
                end_line=self.closer.loc.end_line,
                end_column=self.closer.loc.end_column,
            )
        assert len(self.items)
        return Location(
            filename=self.items[0].loc.filename,
            line=self.items[0].loc.line,
            column=self.items[0].loc.column,
            end_line=self.items[-1].loc.end_line,
            end_column=self.items[-1].loc.end_column,
        )

    def to_output_nodes(self):
        if self.opener and self.closer:
            return [self.opener, *self.items, self.closer]
        return list(self.items)

    def to_token(self) -> Token:
        """
        Convert this Block to a single Token with merged location information.
        """
        return Token.join(self.flatten())

    @property
    def comments(self) -> Comments:
        items = self.flatten()
        if not items:
            return Comments()
        return items[0].comments

    def flatten(self) -> list[Token]:
        if self.opener:
            return _flatten_blocks([self.opener, *self.items, self.closer])
        return _flatten_blocks(self.items)

    def squeeze_single_token(self) -> Token:
        if len(self.items) > 1:
            raise ValueError(f"Unexpected multiple tokens: {self.items}")

        if not isinstance(self.items[0], Token):
            raise ValueError(f"Unexpected structured data: {self.items[0]}")

        return self.items[0]

    def parse(self):
        from .parser import parse_items

        return parse_items(self.items)


@dataclass()
class FormatOptions:
    line_length: int = 100
    compact: bool = False
    indent_size: int = 2
    indent_char: str = " "
    comment_col: int = 40
    newline_before_new_type: bool = False


@dataclass
class OutputLine:
    """A single line of output with indentation and an optional comment."""

    indent: int = 0
    parts: list[str] = field(default_factory=list)
    comment: str | None = None

    def __len__(self) -> int:
        return sum(len(part) for part in self.parts)

    def render(self, options: FormatOptions) -> str:
        indent_str = options.indent_char * (self.indent * options.indent_size)
        line = indent_str + "".join(self.parts)
        if self.comment is None:
            return line
        comment_spaces = options.comment_col - len(line)
        if comment_spaces < 0:
            comment_spaces = 2

        return (" " * comment_spaces).join((line, self.comment))


SequencePart = Attribute | Seq | Token
TokenizerItem = Block | Token | Delimiter
