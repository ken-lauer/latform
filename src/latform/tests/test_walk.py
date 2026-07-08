from __future__ import annotations

import pytest

from ..output import default_options, format_statements
from ..parser import parse
from ..statements import Element, Empty, Line
from ..token import Role, Token
from ..walk import AttrItem, ListItem, WalkItem, walk


def token_items(code: str) -> list[WalkItem]:
    """Walk ``code`` and return only the items whose node is a Token."""
    return [item for item in walk(parse(code)) if isinstance(item.node, Token)]


def find_token(code: str, text: str) -> WalkItem:
    """Return the first walked item whose node is the Token ``text``."""
    for item in token_items(code):
        if item.node == text:
            return item
    raise RuntimeError()


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        pytest.param(
            "q1: quad, l = 0.5",
            ["q1", "quad", "l", "0.5"],
            id="element",
        ),
        pytest.param(
            "myline: line = (a, b, c)",
            ["myline", "a", "b", "c"],
            id="line",
        ),
        pytest.param(
            "x = 3.0",
            ["x", "3.0"],
            id="constant",
        ),
        pytest.param(
            "q1[k1] = 2.0",
            ["q1", "k1", "2.0"],
            id="parameter",
        ),
        pytest.param(
            "call, filename = sub.bmad",
            ["call", "filename", "sub.bmad"],
            id="simple",
        ),
    ],
)
def test_walk_yields_tokens_in_order(code: str, expected: list[str]):
    assert [str(item.node) for item in token_items(code)] == expected


def test_empty_statement_yields_nothing():
    assert list(walk([Empty()])) == []


def test_walk_accepts_single_statement():
    (stmt,) = parse("q1: quad")
    assert [str(item.node) for item in walk(stmt)] == ["q1", "quad"]


def test_attr_item_replace_mutates_scalar_field():
    (elem,) = parse("q1: quad, l = 0.5")
    (item,) = list(it for it in walk([elem]) if it.node == "q1")
    assert isinstance(item, AttrItem)

    item.replace(Token("q2"))
    assert isinstance(elem, Element)
    assert str(elem.name) == "q2"


def test_list_item_replace_mutates_line_element():
    (line,) = parse("myline: line = (a, b, c)")
    (item,) = list(it for it in walk([line]) if it.node == "b")
    assert isinstance(item, ListItem)

    item.replace(Token("z"))
    assert isinstance(line, Line)
    assert [str(e) for e in line.elements.items] == ["a", "z", "c"]


def test_replace_attribute_value():
    (elem,) = parse("q1: quad, l = 0.5")
    (item,) = list(it for it in walk([elem]) if it.node == "0.5")
    assert item.attribute is not None

    item.replace(Token("0.75"))
    assert isinstance(elem, Element)
    assert str(elem.attributes[0].value) == "0.75"


def test_attribute_context_is_innermost_attribute():
    (elem,) = parse("q1: quad, l = 0.5")
    assert isinstance(elem, Element)
    (item,) = list(it for it in walk([elem]) if it.node == "l")
    assert item.attribute is elem.attributes[0]


def test_top_level_nodes_have_no_attribute_context():
    item = find_token("q1: quad, l = 0.5", "q1")
    assert item.attribute is None


def test_depth_increments_inside_containers():
    depths = {str(item.node): item.depth for item in token_items("myline: line = (a, b, c)")}
    assert depths["myline"] == 0
    assert depths["a"] == depths["b"] == depths["c"] == 1


def test_base_walkitem_replace_raises():
    item = WalkItem(node=Token("x"), statement=Empty())
    with pytest.raises(NotImplementedError):
        item.replace(Token("y"))


def test_replace_reflected_in_output():
    statements = parse("q1: quad, l = 0.5\nmyline: line = (q1, q1)")
    for item in walk(statements):
        if isinstance(item.node, Token) and item.node.role is Role.name_ and item.node == "q1":
            item.replace(Token("q2", role=item.node.role))

    text = format_statements(statements, default_options)
    assert "q1" not in text.lower()
    assert text.lower().count("q2") == 3
