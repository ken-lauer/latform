"""Walk through latform statement trees, yielding nodes with context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from .statements import (
    Assignment,
    Constant,
    Element,
    ElementList,
    Empty,
    Line,
    Parameter,
    Simple,
    Statement,
)
from .token import Token
from .types import Attribute, CallName, Seq

WalkNode = Token | Seq | CallName | Attribute


@dataclass
class WalkItem:
    """A node yielded during statement traversal, carrying its context."""

    node: Token | Seq | CallName | Attribute  # WalkNode
    statement: Statement
    attributes: tuple[Attribute, ...] = ()
    depth: int = 0

    @property
    def attribute(self) -> Attribute | None:
        """The innermost containing Attribute, or None."""
        return self.attributes[-1] if self.attributes else None


def walk(
    statements: Statement | list[Statement],
) -> Generator[WalkItem, None, None]:
    """
    Walk through statement trees, yielding each node with its context.

    Parameters
    ----------
    statements : Statement or list of Statement
        One or more parsed statements to walk.

    Yields
    ------
    WalkItem
    """
    if isinstance(statements, Statement):
        statements = [statements]
    for stmt in statements:
        yield from _walk_statement(stmt)


def _walk_statement(stmt: Statement) -> Generator[WalkItem, None, None]:
    match stmt:
        case Empty():
            return

        case Simple(statement=kw, arguments=args):
            yield WalkItem(node=kw, statement=stmt)
            for arg in args:
                yield from _walk_part(arg, stmt, attributes=(), depth=0)

        case Constant(name=name, value=value):
            yield WalkItem(node=name, statement=stmt)
            yield from _walk_part(value, stmt, attributes=(), depth=0)

        case Assignment(name=name, value=value):
            yield from _walk_part(name, stmt, attributes=(), depth=0)
            yield from _walk_part(value, stmt, attributes=(), depth=0)

        case Parameter(target=target, name=name, value=value):
            yield from _walk_part(target, stmt, attributes=(), depth=0)
            yield WalkItem(node=name, statement=stmt)
            yield from _walk_part(value, stmt, attributes=(), depth=0)

        case Line(name=name, elements=elements):
            yield from _walk_part(name, stmt, attributes=(), depth=0)
            yield from _walk_part(elements, stmt, attributes=(), depth=0)

        case ElementList(name=name, elements=elements):
            yield WalkItem(node=name, statement=stmt)
            yield from _walk_part(elements, stmt, attributes=(), depth=0)

        case Element(name=name, keyword=keyword, ele_list=ele_list, attributes=attrs):
            yield WalkItem(node=name, statement=stmt)
            yield WalkItem(node=keyword, statement=stmt)
            if ele_list is not None:
                yield from _walk_part(ele_list, stmt, attributes=(), depth=0)
            for attr in attrs:
                yield from _walk_part(attr, stmt, attributes=(), depth=0)


def _walk_part(
    node: WalkNode,
    stmt: Statement,
    attributes: tuple[Attribute, ...],
    depth: int,
) -> Generator[WalkItem, None, None]:
    match node:
        case Token():
            yield WalkItem(node=node, statement=stmt, attributes=attributes, depth=depth)

        case Attribute(name=name, value=value):
            nested = (*attributes, node)
            yield WalkItem(node=node, statement=stmt, attributes=nested, depth=depth)
            yield from _walk_part(name, stmt, attributes=nested, depth=depth + 1)
            if value is not None:
                yield from _walk_part(value, stmt, attributes=nested, depth=depth + 1)

        case CallName(name=name, args=args):
            yield WalkItem(node=node, statement=stmt, attributes=attributes, depth=depth)
            yield WalkItem(node=name, statement=stmt, attributes=attributes, depth=depth + 1)
            yield from _walk_part(args, stmt, attributes=attributes, depth=depth + 1)

        case Seq(items=items):
            yield WalkItem(node=node, statement=stmt, attributes=attributes, depth=depth)
            for child in items:
                yield from _walk_part(child, stmt, attributes=attributes, depth=depth + 1)
