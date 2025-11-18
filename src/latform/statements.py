from __future__ import annotations

import os.path
import pathlib
from dataclasses import dataclass, field
from typing import ClassVar

from .comments import Comments
from .const import COMMA, STATEMENT_NAME_COLON, STATEMENT_NAME_EQUALS
from .token import Role, Token
from .types import (
    Attribute,
    CallName,
    Delimiter,
    Seq,
)
from .util import comma_delimit


@dataclass(kw_only=True)
class Statement:
    comments: Comments = field(default_factory=Comments)

    def to_output_nodes(self):
        raise NotImplementedError()

    # @classmethod
    # def from_text(cls,
    #
    #               *, strict_class: bool  = False):


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
        nodes = [self.name.with_(role=Role.name_), STATEMENT_NAME_EQUALS, self.value]
        if self.redef:
            return [Token("redef:"), *nodes]
        return nodes


@dataclass
class Assignment(Statement):
    name: Seq | Token
    value: Seq | Token

    def to_output_nodes(self):
        return [self.name.with_(role=Role.name_), STATEMENT_NAME_EQUALS, self.value]


@dataclass
class BmadParameter:
    target: str
    name: str
    type: type | str
    comment: str


@dataclass
class Parameter(Statement):
    target: Seq | Token
    name: Token
    value: Seq | Token

    known: ClassVar[list[BmadParameter]] = [
        # BmadParameter(
        #     "parameter",
        #     "custom_attributeN",
        #     type=str,
        #     comment="Defining custom attributes (\\sref{s:cust.att}).",
        # ),
        BmadParameter(
            "parameter",
            "default_tracking_species",
            type="species",
            comment="Default type of tracked particle. Default is ref_particle.",
        ),
        BmadParameter(
            "parameter",
            "e_tot",
            type=float,
            comment="Reference total Energy. Default: 1000 * rest_energy.",
        ),
        BmadParameter(
            "parameter",
            "electric_dipole_moment",
            type=float,
            comment="Particle electric dipole moment.",
        ),
        BmadParameter("parameter", "live_branch", type=bool, comment="Is branch fit for tracking?"),
        BmadParameter("parameter", "geometry", type="geometry", comment="Open or closed"),
        BmadParameter("parameter", "lattice", type=str, comment="Lattice name."),
        BmadParameter("parameter", "machine", type=str, comment="Machine name."),
        BmadParameter("parameter", "n_part", type=float, comment="Number of particles in a bunch."),
        BmadParameter("parameter", "no_end_marker", type=bool, comment="Default: False."),
        BmadParameter("parameter", "p0c", type=float, comment="Reference momentum."),
        BmadParameter(
            "parameter",
            "particle",
            type="species",
            comment="Reference species: positron, proton, etc.",
        ),
        BmadParameter(
            "parameter", "photon_type", type=str, comment="Incoherent or coherent photons?"
        ),
        BmadParameter("parameter", "ran_seed", type=int, comment="Random number generator init."),
        BmadParameter("parameter", "taylor_order", type=int, comment="Default: 3"),
    ]

    def to_output_nodes(self):
        return [
            self.target.with_(role=Role.name_),
            Delimiter("["),
            self.name.with_(role=Role.name_),
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
            key = Token("line[multipass]", role=Role.kind)
        else:
            key = Token("line", role=Role.name_)

        if isinstance(self.name, Token):
            name = self.name.with_(role=Role.name_)
        else:
            name = CallName(name=self.name.name.with_(role=Role.name_), args=self.name.args)
        return [name, STATEMENT_NAME_COLON, key, STATEMENT_NAME_EQUALS, self.elements]


@dataclass
class ElementList(Statement):
    name: Token
    elements: Seq

    def to_output_nodes(self):
        return [
            self.name.with_(role=Role.name_),
            STATEMENT_NAME_COLON,
            Token("list", role=Role.kind),
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
                self.name.with_(role=Role.name_),
                STATEMENT_NAME_COLON,
                self.keyword.with_(role=Role.kind),
                STATEMENT_NAME_EQUALS,
                self.ele_list,
                COMMA,
                *comma_delimit(self.attributes),
            ]
        return [
            self.name.with_(role=Role.name_),
            STATEMENT_NAME_COLON,
            *comma_delimit([self.keyword.with_(role=Role.kind), *self.attributes]),
        ]


def get_call_filename(
    st: Simple,
    *,
    caller_filename: pathlib.Path | None + None,
    expand_vars: bool = True,
) -> tuple[str, pathlib.Path]:
    if not isinstance(st, Simple):
        raise ValueError(f"Unexpected type: {type(st).__name__}")

    if st.statement != "call":
        raise ValueError(f"Statement is not 'call': {st.statement}")

    attr = st.get_named_attribute("filename", partial_match=True)

    sub_filename = (
        attr.value if isinstance(attr.value, Token) else attr.value.to_token()
    ).remove_quotes()
    expanded = os.path.expandvars(sub_filename) if expand_vars else sub_filename

    if not caller_filename:
        return sub_filename, pathlib.Path(expanded)
    return sub_filename, caller_filename.parent / expanded
