import textwrap

import pytest
import rich
from typing_extensions import Literal

from ..const import COMMA, OPEN_TO_CLOSE, SPACE
from ..output import _flatten_output_nodes, format_nodes
from ..parser import (
    Attribute,
    CallName,
    Comments,
    Delimiter,
    Element,
    Seq,
    Token,
    parse,
)
from ..types import FormatOptions, OutputLine

T = Token
D = Delimiter


def seq(opener: str, items, delimiter: Delimiter = COMMA) -> Seq:
    return Seq(
        opener=Delimiter(opener),
        closer=Delimiter(OPEN_TO_CLOSE[opener]),
        items=list(items),
        delimiter=delimiter,
    )


def render(lines: list[OutputLine], options: FormatOptions) -> str:
    return "\n".join(line.render(options) for line in lines)


def check_format(item, expected: str, options: FormatOptions | None = None) -> None:
    if options is None:
        options = FormatOptions()
    options.comment_col = 0
    rich.print(item)

    print("Nodes:")
    rich.print(_flatten_output_nodes(item))

    expected = textwrap.dedent(expected).strip()
    print("Expected:")
    print(f"```\n{expected}\n```")

    formatted = format_nodes(item, options)
    res = render(formatted, options)
    print("Actual:")
    print(f"```\n{res}\n```")

    rich.print(formatted)
    assert res.strip() == expected


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        pytest.param(seq(opener=D("{"), items=[]), "{}", id="empty-brace"),
        pytest.param(seq(opener=D("{"), items=[T("1")]), "{1}", id="1-ele-brace"),
        pytest.param(
            seq(opener=D("{"), items=[T("1"), T("2"), T("3")]), "{1, 2, 3}", id="3-ele-brace"
        ),
        pytest.param(
            seq(opener=D("("), items=[T("1"), T("2"), T("3")]), "(1, 2, 3)", id="3-ele-paren"
        ),
        pytest.param(
            seq(opener=D("["), items=[T("1"), T("2"), T("3")]), "[1, 2, 3]", id="3-ele-bracket"
        ),
        pytest.param(
            seq(
                opener=D("("),
                items=[
                    T("1"),
                    seq(opener=D("("), items=[T("4"), T("5"), T("6")]),
                    T("3"),
                ],
            ),
            "(1, (4, 5, 6), 3)",
            id="nested-parens",
        ),
        pytest.param(
            Attribute(
                name=T("foo"),
                value=seq(opener=D("{"), items=[T("4"), T("5"), T("6")]),
            ),
            "foo={4, 5, 6}",
            id="named-attribute",
        ),
        pytest.param(
            seq(
                opener=D("("),
                items=[
                    T("1"),
                    Attribute(
                        name=T("foo"),
                        value=seq(opener=D("{"), items=[T("4"), T("5"), T("6")]),
                    ),
                    T("3"),
                ],
            ),
            "(1, foo={4, 5, 6}, 3)",
            id="nested-named-attribute",
        ),
    ],
)
def test_format_array_single_line(item, expected):
    check_format(item, expected=expected)


cmt = Comments(pre=[T("pre")], inline=T("inline"))


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        pytest.param(Token("foo", comments=cmt), "!pre\nfoo  !inline", id="comment-level0"),
        pytest.param(
            seq(opener=D("{"), items=[T("1", comments=cmt)]),
            "{\n  !pre\n  1  !inline\n}",
            id="comment-level1",
        ),
        pytest.param(
            seq(opener=D("{"), items=[seq(opener=D("("), items=[T("1", comments=cmt)])]),
            """\
            {
              (
                !pre
                1  !inline
              )
            }
            """,
            id="comment-level2",
        ),
        pytest.param(
            seq(
                opener=D("{"),
                items=[
                    seq(
                        opener=D("("),
                        items=[
                            T("1", comments=cmt),
                            seq(
                                opener=D("{"),
                                items=[T("2", comments=cmt)],
                            ),
                        ],
                    )
                ],
            ),
            """\
            {
              (
                !pre
                1,  !inline
                {
                  !pre
                  2  !inline
                }
              )
            }
            """,
            id="comment-level3",
        ),
        pytest.param(
            seq(
                opener=D("{"),
                items=[
                    seq(
                        opener=D("{"),
                        items=[
                            T("1", comments=cmt),
                            seq(
                                opener=D("{"),
                                items=[T("2", comments=cmt)],
                            ),
                            T("3"),
                        ],
                    )
                ],
            ),
            """\
            {
              {
                !pre
                1,  !inline
                {
                  !pre
                  2  !inline
                },
                3
              }
            }
            """,
            id="comment-level3-surrounded",
        ),
        pytest.param(
            seq(
                opener=D("{"),
                items=[
                    T("0"),
                    seq(
                        opener=D("{"),
                        items=[
                            T("1", comments=cmt),
                            T("2"),
                            seq(
                                opener=D("{"),
                                items=[
                                    T("3", comments=cmt),
                                    T("4", comments=cmt),
                                ],
                            ),
                            T("5"),
                        ],
                    ),
                    T("6"),
                ],
            ),
            """\
            {
              0,
              {
                !pre
                1,  !inline
                2,
                {
                  !pre
                  3,  !inline
                  !pre
                  4  !inline
                },
                5
              },
              6
            }
            """,
            id="comment-level3-surrounded-1",
        ),
    ],
)
def test_format_token_single_line(item, expected):
    check_format(item, expected=expected)


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        pytest.param(
            Element(
                name=T("qq"),
                keyword=T("quadrupole"),
                ele_list=None,
                attributes=[
                    Attribute(
                        name=CallName(
                            name=T("r_custom"),
                            args=seq(opener="(", items=[T("-2"), T("1"), T("5")]),
                        ),
                        value=T("34.5"),
                    ),
                    Attribute(
                        name=CallName(
                            name=T("r_custom"),
                            args=seq(opener="(", items=[T("-3")], delimiter=SPACE),
                        ),
                        value=T("77.9"),
                    ),
                ],
            ),
            "qq: quadrupole, r_custom(-2, 1, 5) = 34.5, r_custom(-3) = 77.9",
            id="qq-no-comment",
        ),
        pytest.param(
            Element(
                comments=Comments(pre=[T("pre")], inline=T("inline")),
                name=T("qq"),
                keyword=T("quadrupole"),
                ele_list=None,
                attributes=[
                    Attribute(
                        name=CallName(
                            name=T("r_custom"),
                            args=seq(opener="(", items=[T("-2"), T("1"), T("5")]),
                        ),
                        value=T("34.5"),
                    ),
                    Attribute(
                        name=CallName(
                            name=T("r_custom"),
                            args=seq(opener="(", items=[T("-3")], delimiter=SPACE),
                        ),
                        value=T("77.9"),
                    ),
                ],
            ),
            "!pre\nqq: quadrupole, r_custom(-2, 1, 5) = 34.5, r_custom(-3) = 77.9  !inline",
            id="qq-comment",
        ),
    ],
)
def test_format_element(item, expected):
    check_format(item, expected=expected)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        pytest.param(
            "O_L: overlay = {p1[L]:(Lcell - 2*Lq)/2}, var = {Lcell}",
            "O_L: overlay = {p1[L]:(Lcell - 2*Lq)/2}, var={Lcell}",
            id="overlay_with_expr",
        ),
        pytest.param(
            "ele: key, foo = call::/path/to/file",
            "ele: key, foo = call::/path/to/file",
            id="call_expr",
        ),
        pytest.param(
            "ele: line = (a, b, c)",
            "ele: line = (a, b, c)",
            id="simple_line",
        ),
        pytest.param(
            "ele: line = (a, b, c, --d, e, f)",
            "ele: line = (a, b, c, --d, e, f)",
            id="line_with_reverse",
        ),
    ],
)
def test_format_element_from_source(code: str, expected: str) -> None:
    (stmt,) = parse(code)
    check_format(stmt, expected=expected)


@pytest.mark.parametrize(
    ("code", "expected", "case"),
    [
        pytest.param(
            "O_L: overlay = {p1[L]:(Lcell - 2*Lq)/2}, var = {Lcell}",
            "o_l: overlay = {p1[L]:(Lcell - 2*Lq)/2}, var={Lcell}",
            "lower",
            id="lowercase_name",
        ),
        pytest.param(
            "ele: key, foo = call::/path/to/file",
            "ELE: KEY, foo = call::/path/to/file",
            "upper",
            id="uppercase_name",
        ),
        pytest.param(
            "ElE: line = (a, b, c)",
            "ElE: line = (a, b, c)",
            "same",
            id="same_case_name",
        ),
    ],
)
def test_format_case_opts(
    code: str, expected: str, case: Literal["lower", "upper", "same"]
) -> None:
    (stmt,) = parse(code)
    check_format(stmt, expected=expected, options=FormatOptions(name_case=case))


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        pytest.param(
            "a+ b    - 3",
            "a + b - 3",
            id="addition_subtraction_spacing",
        ),
        pytest.param(
            "a/ b    - 3",
            "a/b - 3",
            id="division_no_space",
        ),
        pytest.param(
            "a(c) / b    - 3",
            "a(c)/b - 3",
            id="function_call_division",
        ),
        pytest.param(
            "a(c) * b    - 3",
            "a(c)*b - 3",
            id="function_call_multiplication",
        ),
        pytest.param(
            "- a(c) * b    - 3",
            "-a(c)*b - 3",
            id="unary_minus_function_call",
        ),
        pytest.param(
            "5 - -a(c) * b    - 3",
            "5 - -a(c)*b - 3",
            id="binary_minus_unary_minus",
        ),
        pytest.param(
            "b01w[   rho  ]   /b01w[L]",
            "b01w[rho]/b01w[L]",
            id="array_indexing_division",
        ),
        pytest.param(
            "b01w[   rho  ]   ^3",
            "b01w[rho]^3",
            id="exp_after_bracket",
        ),
        pytest.param(
            "sqrt(foo)   ^3",
            "sqrt(foo)^3",
            id="exp_after_func",
        ),
        pytest.param(
            "1.3e4 ^ 4",
            "1.3e4^4",
            id="exp_after_sci_not",
        ),
        pytest.param(
            "ename[frequencies(1)%amp]",
            "ename[frequencies(1)%amp]",
            id="nested_indexing_with_percent",
        ),
        pytest.param(
            "species(He++)",
            "species(He++)",
            id="double_plus_species",
        ),
        pytest.param(
            "species(He--)",
            "species(He--)",
            id="double_minus_species",
        ),
        pytest.param(
            "species(He+)",
            "species(He+)",
            id="single_plus_species",
        ),
        pytest.param(
            "species(He-)",
            "species(He-)",
            id="single_minus_species",
        ),
    ],
)
def test_format_expression(expression: str, expected: str) -> None:
    prefix = "ele: name, foo = "
    (stmt,) = parse(f"{prefix}{expression}")
    check_format(stmt, expected=f"{prefix}{expected}")


@pytest.mark.parametrize(
    ("expression", "expected", "trailing_comma"),
    [
        pytest.param(
            "{1, 2, 3,}",
            "{1, 2, 3}",
            False,
            id="single_line_no_trailing_comma",
        ),
        pytest.param(
            "{1, 2, 3,}",
            "{1, 2, 3}",
            True,  # -> we never put a trailing comma in a sequence on a single line
            id="single_line_trailing_comma",
        ),
        pytest.param(
            """
            {
            1, 2,  ! foo
            3,}
            """,
            "{\n  1,\n  2,  ! foo\n  3\n}",
            False,
            id="multi_line_no_trailing_comma",
        ),
        pytest.param(
            """
            {
            1, 2,  ! foo
            3,}
            """,
            "{\n  1,\n  2,  ! foo\n  3,\n}",
            True,
            id="multi_line_trailing_comma",
        ),
    ],
)
def test_format_trailing_comma(expression: str, expected: str, trailing_comma: bool) -> None:
    prefix = "ele: name, foo="
    (stmt,) = parse(f"{prefix}{expression}")
    check_format(
        stmt, expected=f"{prefix}{expected}", options=FormatOptions(trailing_comma=trailing_comma)
    )
