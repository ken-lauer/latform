from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from .const import COMMA, STATEMENT_NAME_COLON, STATEMENT_NAME_EQUALS
from .types import (
    Attribute,
    CallName,
    Comments,
    Delimiter,
    Seq,
    Token,
)
from .util import comma_delimit


@dataclass(kw_only=True)
class Statement:
    comments: Comments = field(default_factory=Comments)

    def to_output_nodes(self):
        raise NotImplementedError()


@dataclass
class Empty(Statement):
    def to_output_nodes(self):
        return []


@dataclass
class Simple(Statement):
    known_statements: ClassVar[frozenset] = frozenset(
        {
            "beam",  # from source; what else is in there? (TODO)
            "call",
            "calc_reference_orbit",
            "combine_consecutive_elements",
            "debug_marker",
            "end_file",
            "merge_elements",
            "no_digested",
            "no_superimpose",
            "parser_debug",
            "print",
            "remove_elements",
            "return",
            "slice_lattice",
            "start_branch_at",
            "title",
            "use",
            "use_local_lat_file",
            "write_digested",
        }
    )
    statement: Token
    arguments: list[Attribute | Token]

    def get_named_attribute(self, name: Token | str, *, partial_match: bool = True) -> Attribute:
        for arg in self.arguments:
            if isinstance(arg, Attribute) and isinstance(arg.name, Token):
                if arg.name == name:
                    return arg
                if partial_match and name.lower().startswith(arg.name.lower()):
                    return arg
        raise KeyError(str(name))

    @staticmethod
    def is_known_statement(name: Token) -> bool:
        # TODO: can these be shortened?
        return name.lower() in Simple.known_statements

    def to_output_nodes(self):
        nodes = [self.statement, *self.arguments]
        if self.statement.lower() in {"print", "parser_debug"}:
            return nodes
        for idx in range(len(nodes) - 1, 0, -1):
            nodes.insert(idx, COMMA)
        return nodes


@dataclass
class Constant(Statement):
    """There are five types of parameters in Bmad: reals, integers, switches, logicals (booleans), and strings."""

    name: Token
    value: Seq | Token
    redef: bool = False

    def to_output_nodes(self):
        nodes = [self.name, STATEMENT_NAME_EQUALS, self.value]
        if self.redef:
            return [Token("redef:"), *nodes]
        return nodes


@dataclass
class Assignment(Statement):
    name: Seq | Token
    value: Seq | Token

    def to_output_nodes(self):
        return [self.name, STATEMENT_NAME_EQUALS, self.value]


@dataclass
class Parameter(Statement):
    target: Seq | Token
    name: Token
    value: Seq | Token

    def to_output_nodes(self):
        return [
            self.target,
            Delimiter("["),
            self.name,
            Delimiter("]"),
            STATEMENT_NAME_EQUALS,
            self.value,
        ]


@dataclass
class NonstandardParameter(Parameter):
    pass


@dataclass
class Line(Statement):
    name: Token | CallName
    elements: Seq
    multipass: bool = False

    def to_output_nodes(self):
        if self.multipass:
            key = Token("line[multipass]")
        else:
            key = Token("line")
        return [self.name, STATEMENT_NAME_COLON, key, STATEMENT_NAME_EQUALS, self.elements]


@dataclass
class ElementList(Statement):
    name: Token
    elements: Seq

    def to_output_nodes(self):
        return [
            self.name,
            STATEMENT_NAME_COLON,
            Token("list"),
            STATEMENT_NAME_EQUALS,
            self.elements,
        ]


@dataclass
class Element(Statement):
    name: Token
    keyword: Token
    ele_list: Seq | None = None  # ele_name: keyword = { ele_list }
    attributes: list[Attribute] = field(default_factory=list)

    def to_output_nodes(self):
        if self.ele_list is not None:
            return [
                self.name,
                STATEMENT_NAME_COLON,
                self.keyword,
                STATEMENT_NAME_EQUALS,
                self.ele_list,
                COMMA,
                *comma_delimit(self.attributes),
            ]
        return [self.name, STATEMENT_NAME_COLON, *comma_delimit([self.keyword, *self.attributes])]
