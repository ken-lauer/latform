"""Walk through latform statement trees, yielding nodes with context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Sequence

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
    node: Token | Seq | CallName | Attribute  # WalkNode
    statement: Statement
    attributes: tuple[Attribute, ...] = ()
    depth: int = 0

    @property
    def attribute(self) -> Attribute | None:
        """The innermost containing Attribute, or None."""
        return self.attributes[-1] if self.attributes else None

    def replace(self, new: WalkNode) -> None:
        """Write ``new`` back into the container that holds ``node``."""
        raise NotImplementedError(f"{type(self).__name__} does not support replacement")


@dataclass
class AttrItem(WalkItem):
    """A node held as a named attribute of a container (dataclass field)."""

    obj: object | None = None
    attr: str = ""

    def replace(self, new: WalkNode) -> None:
        setattr(self.obj, self.attr, new)


@dataclass
class ListItem(WalkItem):
    """A node held at an index within a list."""

    container: list | None = None
    index: int = 0

    def replace(self, new: WalkNode) -> None:
        self.container[self.index] = new


def iter_tokens(node: WalkNode | None) -> Generator[Token, None, None]:
    """Yield every `Token` nested within ``node`` (depth-first)."""
    if node is None:
        return

    match node:
        case Token():
            yield node
        case Seq():
            for item in node.items:
                yield from iter_tokens(item)
        case Attribute():
            yield from iter_tokens(node.name)
            if node.value is not None:
                yield from iter_tokens(node.value)
        case CallName():
            yield node.name
            yield from iter_tokens(node.args)


def walk(
    statements: Statement | Sequence[Statement],
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
            yield AttrItem(kw, stmt, obj=stmt, attr="statement")
            for i, arg in enumerate(args):
                yield from _walk_part(ListItem(arg, stmt, container=args, index=i))

        case Constant(name=name, value=value):
            yield AttrItem(name, stmt, obj=stmt, attr="name")
            yield from _walk_part(AttrItem(value, stmt, obj=stmt, attr="value"))

        case Assignment(name=name, value=value):
            yield from _walk_part(AttrItem(name, stmt, obj=stmt, attr="name"))
            yield from _walk_part(AttrItem(value, stmt, obj=stmt, attr="value"))

        case Parameter(target=target, name=name, value=value):
            yield from _walk_part(AttrItem(target, stmt, obj=stmt, attr="target"))
            yield AttrItem(name, stmt, obj=stmt, attr="name")
            yield from _walk_part(AttrItem(value, stmt, obj=stmt, attr="value"))

        case Line(name=name, elements=elements):
            yield from _walk_part(AttrItem(name, stmt, obj=stmt, attr="name"))
            yield from _walk_part(AttrItem(elements, stmt, obj=stmt, attr="elements"))

        case ElementList(name=name, elements=elements):
            yield AttrItem(name, stmt, obj=stmt, attr="name")
            yield from _walk_part(AttrItem(elements, stmt, obj=stmt, attr="elements"))

        case Element(name=name, keyword=keyword, ele_list=ele_list, attributes=attrs):
            yield AttrItem(name, stmt, obj=stmt, attr="name")
            yield AttrItem(keyword, stmt, obj=stmt, attr="keyword")
            if ele_list is not None:
                yield from _walk_part(AttrItem(ele_list, stmt, obj=stmt, attr="ele_list"))
            for i, attr in enumerate(attrs):
                yield from _walk_part(ListItem(attr, stmt, container=attrs, index=i))


def _walk_part(item: WalkItem) -> Generator[WalkItem, None, None]:
    """Yield ``item`` and recurse into its children."""
    node = item.node
    stmt = item.statement
    depth = item.depth

    match node:
        case Token():
            yield item

        case Attribute(name=name, value=value):
            yield item
            nested = (*item.attributes, node)
            yield from _walk_part(AttrItem(name, stmt, nested, depth + 1, obj=node, attr="name"))
            if value is not None:
                yield from _walk_part(
                    AttrItem(value, stmt, nested, depth + 1, obj=node, attr="value")
                )

        case CallName(name=name, args=args):
            yield item
            attrs = item.attributes
            yield AttrItem(name, stmt, attrs, depth + 1, obj=node, attr="name")
            yield from _walk_part(AttrItem(args, stmt, attrs, depth + 1, obj=node, attr="args"))

        case Seq(items=items):
            yield item
            attrs = item.attributes
            for i, child in enumerate(items):
                yield from _walk_part(
                    ListItem(child, stmt, attrs, depth + 1, container=items, index=i)
                )
