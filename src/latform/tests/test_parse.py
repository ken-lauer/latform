import inspect

import pytest
import rich

from ..const import (
    COMMA,
    EQUALS,
    LBRACE,
    LPAREN,
    OPEN_TO_CLOSE,
    RBRACE,
    RPAREN,
    SLASH,
    SPACE,
    STAR,
)
from ..exceptions import ExtraCloseDelimiter, MissingCloseDelimiter
from ..output import format_statements
from ..parser import (
    Assignment,
    Attribute,
    Block,
    CallName,
    Comments,
    Constant,
    Delimiter,
    Element,
    Empty,
    Line,
    NonstandardParameter,
    Parameter,
    Seq,
    Simple,
    Token,
    UnexpectedAssignment,
    parse,
)
from ..tokenizer import tokenize
from ..types import FormatOptions

T = Token


def parse_verbose(code: str):
    caller = inspect.stack()[1].function
    prefix = f"{caller}:"

    print(prefix)
    print(f"```\n{code}\n```")

    rich.print("\nTokenized:")
    tokens = tokenize(code)
    rich.print("-" * 30)
    rich.print(tokens)
    rich.print("-" * 30)

    res = parse(code)
    rich.print("\nResult:")
    rich.print("-" * 30)
    rich.print(res)
    rich.print("-" * 30)

    return res


def test_from_items_tokens():
    items = [Token("a"), Token("b"), Token("c")]
    res = Seq.from_items(list(items))
    assert res == Seq(
        opener=None, closer=None, items=[Token("a"), Token("b"), Token("c")], delimiter=SPACE
    )


def test_from_items_block():
    opener = LBRACE
    closer = RBRACE
    items = [Block(opener=opener, closer=closer, items=[Token("a"), Token("b"), Token("c")])]
    res = Seq.from_items(list(items))
    assert res == Seq(
        opener=opener,
        closer=closer,
        items=[Seq(items=[Token("a"), Token("b"), Token("c")], delimiter=SPACE)],
        delimiter=COMMA,
    )


def test_from_items_delimited_block():
    opener = LBRACE
    closer = RBRACE
    items = [
        Block(
            opener=opener, closer=closer, items=[Token("a"), COMMA, Token("b"), COMMA, Token("c")]
        )
    ]
    res = Seq.from_items(list(items))
    assert res == Seq(
        opener=opener, closer=closer, items=[Token("a"), Token("b"), Token("c")], delimiter=COMMA
    )


def test_from_items_delimited_nested_block():
    opener = LBRACE
    closer = RBRACE

    inner_tokens = [Token("a"), Token("b"), Token("c")]
    blk = Block(opener=opener, closer=closer, items=list(inner_tokens))
    items = [Block(opener=opener, closer=closer, items=[Token("a"), COMMA, Token("b"), COMMA, blk])]
    res = Seq.from_items(list(items))
    assert res == Seq(
        opener=opener,
        closer=closer,
        items=list(
            [
                Token("a"),
                Token("b"),
                Seq(
                    opener=opener,
                    closer=closer,
                    items=[Seq(items=list(inner_tokens), delimiter=SPACE)],
                    delimiter=COMMA,
                ),
            ]
        ),
        delimiter=COMMA,
    )


def test_from_items_named_token():
    items = [Token("a"), EQUALS, Token("c")]
    res = Seq.from_items(items)
    assert res == Attribute(name=Token("a"), value=Token("c"))


def test_from_items_named_sequence_curly():
    opener = LBRACE
    closer = RBRACE

    inner_tokens = [Token("a"), Token("b"), Token("c")]
    blk = Block(opener=opener, closer=closer, items=list(inner_tokens))
    items = [Token("a"), EQUALS, blk]
    res = Seq.from_items(items)
    assert res == Attribute(
        name=Token("a"),
        value=Seq(
            opener=opener,
            closer=closer,
            items=[Seq(items=list(inner_tokens), delimiter=SPACE)],
            delimiter=COMMA,
        ),
    )


def test_from_items_named_sequence_curly_comma_delimited():
    opener = LBRACE
    closer = RBRACE
    blk = Block(
        opener=opener, closer=closer, items=[Token("a"), COMMA, Token("b"), COMMA, Token("c")]
    )
    items = [Token("a"), EQUALS, blk]
    res = Seq.from_items(items)
    assert res == Attribute(
        name=Token("a"),
        value=Seq(
            opener=opener,
            closer=closer,
            items=[Token("a"), Token("b"), Token("c")],
            delimiter=COMMA,
        ),
    )


def make_seq(opener: str, items, delimiter: str | None = COMMA) -> Seq:
    return Seq(
        opener=Delimiter(opener),
        closer=Delimiter(OPEN_TO_CLOSE[opener]),
        items=items,
        delimiter=Delimiter(delimiter) if delimiter else None,
    )


def make_block(opener: str, items) -> Seq:
    if not opener:
        return Seq(items=items)

    return Seq(opener=Delimiter(opener), closer=Delimiter(OPEN_TO_CLOSE[opener]), items=items)


def test_extra_close_delimiter():
    line = "parameter[foo]        ]"
    with pytest.raises(ExtraCloseDelimiter) as raised:
        assert parse_verbose(line)
    print(raised)
    delim = raised.value.delim
    assert delim == "]"
    assert delim.loc.column == len(line) - 1
    assert delim.loc.end_column == len(line)
    assert line[delim.loc.column : delim.loc.end_column] == delim
    assert delim.loc.line == 0
    assert delim.loc.end_line == 0


def test_missing_close_delimiter():
    line = "parameter[foo"
    with pytest.raises(MissingCloseDelimiter) as raised:
        assert parse_verbose(line)
    print(raised)

    (delim,) = raised.value.delims
    assert delim == "["
    assert delim.loc.column == line.index("[")
    assert delim.loc.end_column == line.index("[") + 1
    assert line[delim.loc.column : delim.loc.end_column] == delim
    assert delim.loc.line == 0
    assert delim.loc.end_line == 0


def roundtrip_code(code: str) -> None:
    options = FormatOptions()
    # code1 -> tokens1 -> roundtrip_code -> roundtrip_tokens
    #  check tokens1 == tokens2
    tokens1 = parse_verbose(code)

    roundtrip_code = format_statements(tokens1, options)

    rich.print("Back to string:")
    rich.print("=" * 30)
    print(roundtrip_code)
    rich.print("=" * 30)

    roundtrip_tokens = parse(roundtrip_code)
    assert tokens1 == roundtrip_tokens


code_tests = pytest.mark.parametrize(
    ("code", "expected"),
    [
        pytest.param(
            """
    global_rf_frequency = 1.3e9
    """,
            [
                Constant(
                    name=T("global_rf_frequency"),
                    value=T("1.3e9"),
                )
            ],
            id="constant_without_comments",
        ),
        pytest.param(
            """
    global_rf_frequency &
    = &
    1.3e9
    """,
            [
                Constant(
                    name=T("global_rf_frequency"),
                    value=T("1.3e9"),
                )
            ],
            id="excessive_continuation",
        ),
        pytest.param(
            """
    global_rf_frequency := 1.3e9
    """,
            [
                Constant(
                    name=T("global_rf_frequency"),
                    value=T("1.3e9"),
                )
            ],
            id="constant_using_walrus_operator",
        ),
        pytest.param(
            "my_const = sqrt(10.3 + 1) * pi^3",
            [
                Constant(
                    name=T("my_const"),
                    value=Seq(
                        items=[
                            T("sqrt"),
                            make_seq(
                                "(",
                                items=[
                                    Seq(
                                        items=[
                                            T("10.3"),
                                            T("+"),
                                            T("1"),
                                        ],
                                        delimiter=SPACE,
                                    ),
                                ],
                                delimiter=COMMA,
                            ),
                            T("*"),
                            T("pi"),
                            T("^"),
                            T("3"),
                        ],
                        delimiter=SPACE,
                    ),
                )
            ],
            id="constant_with_function_and_power",
        ),
        pytest.param(
            "abc := my_const * 23",
            [
                Constant(
                    name=T("abc"),
                    value=Seq(items=[T("my_const"), T("*"), T("23")], delimiter=SPACE),
                )
            ],
            id="constant_with_colon_equals",
        ),
        pytest.param(
            """
    !pre
    global_rf_frequency = 1.3e9  !inline
    """,
            [
                Constant(
                    comments=Comments(pre=[T("pre")], inline=T("inline")),
                    name=T("global_rf_frequency"),
                    value=T("1.3e9"),
                )
            ],
            id="constant_with_comments",
        ),
        pytest.param(
            """
    !pre
    global_rf_frequency = 1.3e9  !inline
    !post1
    !post2
    !post3
    """,
            [
                Constant(
                    comments=Comments(pre=[T("pre")], inline=T("inline")),
                    name=T("global_rf_frequency"),
                    value=T("1.3e9"),
                ),
                Empty(
                    comments=Comments(
                        pre=[
                            T("post1"),
                            T("post2"),
                            T("post3"),
                        ]
                    ),
                ),
            ],
            id="post_comments",
        ),
        pytest.param(
            """
    !pre
    redef: global_rf_frequency = 1.3e9  !inline
    """,
            [
                Constant(
                    comments=Comments(pre=[T("pre")], inline=T("inline")),
                    name=T("global_rf_frequency"),
                    value=T("1.3e9"),
                    redef=True,
                )
            ],
            id="constant_redef",
        ),
        pytest.param(
            """
    !pre
    my_const = b01w[rho] / b01w[L] ! Use of attribute values in an expression.
    """,
            [
                Constant(
                    comments=Comments(
                        pre=[T("pre")],
                        inline=T(T(" Use of attribute values in an expression.")),
                    ),
                    name=T("my_const"),
                    value=Seq(
                        items=[
                            T("b01w"),
                            make_block(opener="[", items=[T("rho")]),
                            Token("/"),
                            T("b01w"),
                            make_block(opener="[", items=[T("L")]),
                        ],
                    ),
                )
            ],
            id="constant_with_attribute_value",
        ),
        pytest.param(
            "x_ray: line[multipass] = (el1, el2, el3)",
            [
                Line(
                    name=T("x_ray"),
                    elements=make_seq(
                        opener=LPAREN,
                        items=[
                            T("el1"),
                            T("el2"),
                            T("el3"),
                        ],
                        delimiter=",",
                    ),
                    multipass=True,
                )
            ],
            id="line_element_multipass",
        ),
        pytest.param(
            "x_ray: line = (el1, el2, el3)",
            [
                Line(
                    name=T("x_ray"),
                    elements=make_seq(
                        opener=LPAREN,
                        items=[
                            T("el1"),
                            T("el2"),
                            T("el3"),
                        ],
                        delimiter=",",
                    ),
                )
            ],
            id="line_element",
        ),
        pytest.param(
            "arg_ln(a, b): line = (ele1, a, ele2, b)",
            [
                Line(
                    name=CallName(
                        name=T("arg_ln"),
                        args=make_seq(opener="(", items=[T("a"), T("b")], delimiter=","),
                    ),
                    elements=make_seq(
                        opener="(",
                        items=[
                            T("ele1"),
                            T("a"),
                            T("ele2"),
                            T("b"),
                        ],
                        delimiter=",",
                    ),
                )
            ],
            id="line_element_with_arguments",
        ),
        pytest.param(
            """
    betweenhalf00(sfw,sdw,ofw,odw) : line=(  dbt1, qbt1,qbt1, dbt2,  qbt2,qbt2, dbt3,  &
            ofw, sfw, dbt4, qbt3,qbt3, dbt5, qbt4,qbt4, dbt6, qbt5,qbt5, dbt7, odw, sdw ) ;
    """,
            [
                Line(
                    name=CallName(
                        name=T("betweenhalf00"),
                        args=make_seq(
                            opener="(",
                            items=[
                                T("sfw"),
                                T("sdw"),
                                T("ofw"),
                                T("odw"),
                            ],
                            delimiter=",",
                        ),
                    ),
                    elements=make_seq(
                        opener=LPAREN,
                        items=[
                            T(ele)
                            for ele in "dbt1,qbt1,qbt1,dbt2,qbt2,qbt2,dbt3,ofw,sfw,dbt4,qbt3,qbt3,dbt5,qbt4,qbt4,dbt6,qbt5,qbt5,dbt7,odw,sdw".split(
                                ","
                            )
                        ],
                        delimiter=",",
                    ),
                )
            ],
            id="line_element_with_arguments_wild",
        ),
        pytest.param(
            """
    ! pre
    cst1: crystal, b_param = -0.6, crystal_type = "Si(620)", use_reflectivity_table = T,
            reflectivity_table = {polarization = sigma, call::reflect_pi.bmad},
            reflectivity_table = {polarization = pi, call::reflect_pi.bmad}
    """,
            [
                Element(
                    comments=Comments(pre=[T(" pre")]),
                    name=T("cst1"),
                    keyword=T("crystal"),
                    attributes=[
                        Attribute(name=T("b_param"), value=T("-0.6")),
                        Attribute(name=T("crystal_type"), value=T('"Si(620)"')),
                        Attribute(
                            name=T("use_reflectivity_table"),
                            value=T("T"),
                        ),
                        Attribute(
                            name=T("reflectivity_table"),
                            value=make_seq(
                                opener=LBRACE,
                                items=[
                                    Attribute(
                                        name=T("polarization"),
                                        value=T("sigma"),
                                    ),
                                    Seq(
                                        delimiter=SPACE,
                                        items=[T("call"), Delimiter("::"), T("reflect_pi.bmad")],
                                    ),
                                ],
                            ),
                        ),
                        Attribute(
                            name=T("reflectivity_table"),
                            value=make_seq(
                                opener=LBRACE,
                                items=[
                                    Attribute(
                                        name=T("polarization"),
                                        value=T("pi"),
                                    ),
                                    Seq(
                                        delimiter=SPACE,
                                        items=[T("call"), Delimiter("::"), T("reflect_pi.bmad")],
                                    ),
                                ],
                            ),
                        ),
                    ],
                )
            ],
            id="crystal_element_with_reflectivity_tables",
        ),
        pytest.param(
            """
cst3: crystal, b_param =  0.6, crystal_type = "Ge(111)",
  h_misalign = {
    ix_bounds = (-1, 1),
    iy_bounds = (-2, 0),
    r0 = (0.01, 0.02),
    dr = (0.02, 0.03),
    pt(-1, -2) = (0.0010, 0.0020, 0.0030, 0.0040),
    pt(-1, -1) = (0.0011, 0.0021, 0.0031, 0.0041),
  }
        """,
            [
                Element(
                    name=T("cst3"),
                    keyword=T("crystal"),
                    attributes=[
                        Attribute(name=T("b_param"), value=T("0.6")),
                        Attribute(name=T("crystal_type"), value=T('"Ge(111)"')),
                        Attribute(
                            name=T("h_misalign"),
                            value=make_seq(
                                "{",
                                items=[
                                    Attribute(
                                        name=T("ix_bounds"),
                                        value=make_seq(
                                            opener="(",
                                            items=[T("-1"), T("1")],
                                        ),
                                    ),
                                    Attribute(
                                        name=T("iy_bounds"),
                                        value=make_seq(
                                            opener="(",
                                            items=[T("-2"), T("0")],
                                        ),
                                    ),
                                    Attribute(
                                        name=T("r0"),
                                        value=make_seq(
                                            opener="(",
                                            items=[T("0.01"), T("0.02")],
                                        ),
                                    ),
                                    Attribute(
                                        name=T("dr"),
                                        value=make_seq(
                                            opener="(",
                                            items=[T("0.02"), T("0.03")],
                                        ),
                                    ),
                                    Attribute(
                                        name=CallName(
                                            name=T("pt"),
                                            args=make_seq(opener="(", items=[T("-1"), T("-2")]),
                                        ),
                                        value=make_seq(
                                            opener="(",
                                            items=[
                                                T("0.0010"),
                                                T("0.0020"),
                                                T("0.0030"),
                                                T("0.0040"),
                                            ],
                                        ),
                                    ),
                                    Attribute(
                                        name=CallName(
                                            name=T("pt"),
                                            args=make_seq(opener="(", items=[T("-1"), T("-1")]),
                                        ),
                                        value=make_seq(
                                            opener="(",
                                            items=[
                                                T("0.0011"),
                                                T("0.0021"),
                                                T("0.0031"),
                                                T("0.0041"),
                                            ],
                                        ),
                                    ),
                                ],
                            ),
                        ),
                    ],
                )
            ],
            id="attribute_with_call_args",
        ),
        pytest.param(
            """
        parameter[lattice] = "CBETA"
        """,
            [
                Parameter(
                    target=T("parameter"),
                    name=T("lattice"),
                    value=T('"CBETA"'),
                ),
            ],
            id="parameter",
        ),
        pytest.param(
            """
        beginning[e_tot] = 6e6
        """,
            [
                Parameter(
                    target=T("beginning"),
                    name=T("e_tot"),
                    value=T("6e6"),
                ),
            ],
            id="beginning",
        ),
        pytest.param(
            """
        q2: quadrupole, tracking_method = symp_lie_ptc
        q2[tracking_method] = symp_lie_ptc
        quadrupole::*[tracking_method] = symp_lie_ptc
        """,
            [
                Element(
                    name=T("q2"),
                    keyword=T("quadrupole"),
                    attributes=[Attribute(name=T("tracking_method"), value=T("symp_lie_ptc"))],
                ),
                Parameter(
                    target=T("q2"),
                    name=T("tracking_method"),
                    value=T("symp_lie_ptc"),
                ),
                Parameter(
                    target=Seq(delimiter=SPACE, items=[T("quadrupole"), Delimiter("::"), T("*")]),
                    name=T("tracking_method"),
                    value=T("symp_lie_ptc"),
                ),
            ],
            id="manual_chap6_intro",
        ),
        pytest.param(
            "b01w[roll] = 6.5 ! Set an attribute value.",
            [
                Parameter(
                    comments=Comments(inline=T(T(" Set an attribute value."))),
                    target=T("b01w"),
                    name=T("roll"),
                    value=T("6.5"),
                )
            ],
            id="element_parameter_assignment_roll",
        ),
        pytest.param(
            "b01w[L] = 6.5 ! Change an attribute value.",
            [
                Parameter(
                    comments=Comments(inline=T(T(" Change an attribute value."))),
                    target=T("b01w"),
                    name=T("L"),
                    value=T("6.5"),
                )
            ],
            id="element_parameter_assignment_L",
        ),
        pytest.param(
            "b01w[L] = b01w[rho] / 12 ! OK to reset an attribute value.",
            [
                Parameter(
                    comments=Comments(inline=T(T(" OK to reset an attribute value."))),
                    target=T("b01w"),
                    name=T("L"),
                    value=Seq(
                        items=[
                            T("b01w"),
                            make_seq(
                                opener=Delimiter("["),
                                items=[T("rho")],
                                delimiter=SPACE,
                            ),
                            T("/"),
                            T("12"),
                        ],
                    ),
                )
            ],
            id="element_parameter_assignment_expression",
        ),
        pytest.param(
            "97[x_offset] = 0.0023 ! Set x_offset attribute of 97th element",
            [
                Parameter(
                    comments=Comments(inline=T(T(" Set x_offset attribute of 97th element"))),
                    target=T("97"),
                    name=T("x_offset"),
                    value=T("0.0023"),
                )
            ],
            id="element_parameter_assignment_numeric_target",
        ),
        pytest.param(
            'b2>>si_cryst##2[tilt] = 0.1 ! Tilt the 2nd instance of "si_cryst" in branch "b2"',
            [
                Parameter(
                    comments=Comments(
                        inline=T(T(' Tilt the 2nd instance of "si_cryst" in branch "b2"'))
                    ),
                    target=T("b2>>si_cryst##2"),
                    name=T("tilt"),
                    value=T("0.1"),
                )
            ],
            id="element_parameter_assignment_branch_instance",
        ),
        pytest.param(
            "5:32[x_limit] = 0.3 ! Sets elements with indexes 5 through 32 in branch 0.",
            [
                Parameter(
                    comments=Comments(
                        inline=T(T(" Sets elements with indexes 5 through 32 in branch 0."))
                    ),
                    target=Seq(items=[T("5"), T(":"), T("32")], delimiter=SPACE),
                    name=T("x_limit"),
                    value=T("0.3"),
                )
            ],
            id="element_parameter_assignment_range",
        ),
        pytest.param(
            "parameter[custom_attribute1] = quadrupole::error_k1",
            [
                Parameter(
                    target=T("parameter"),
                    name=T("custom_attribute1"),
                    value=Seq(
                        delimiter=SPACE, items=[T("quadrupole"), Delimiter("::"), T("error_k1")]
                    ),
                )
            ],
            id="custom_element_attributes_quadrupole_error",
        ),
        pytest.param(
            "parameter[custom_attribute1] = mag_id",
            [
                Parameter(
                    target=T("parameter"),
                    name=T("custom_attribute1"),
                    value=T("mag_id"),
                )
            ],
            id="custom_element_attributes_mag_id",
        ),
        pytest.param(
            "parameter[custom_attribute1] = sextupole::error_k2",
            [
                Parameter(
                    target=T("parameter"),
                    name=T("custom_attribute1"),
                    value=Seq(
                        delimiter=SPACE, items=[T("sextupole"), Delimiter("::"), T("error_k2")]
                    ),
                )
            ],
            id="custom_element_attributes_sextupole_error",
        ),
        pytest.param(
            "parameter[custom_attribute2] = color",
            [
                Parameter(
                    target=T("parameter"),
                    name=T("custom_attribute2"),
                    value=T("color"),
                )
            ],
            id="custom_element_attributes_color",
        ),
        pytest.param(
            "parameter[custom_attribute2] = parameter::quad_mag_moment",
            [
                Parameter(
                    target=T("parameter"),
                    name=T("custom_attribute2"),
                    value=Seq(
                        delimiter=SPACE,
                        items=[T("parameter"), Delimiter("::"), T("quad_mag_moment")],
                    ),
                )
            ],
            id="custom_element_attributes_quad_mag_moment",
        ),
        pytest.param(
            "qq: quadrupole, r_custom(-2,1,5) = 34.5, r_custom(-3) = 77.9",
            [
                Element(
                    name=T("qq"),
                    keyword=T("quadrupole"),
                    attributes=[
                        Attribute(
                            name=CallName(
                                name=T("r_custom"),
                                args=make_seq(opener="(", items=[T("-2"), T("1"), T("5")]),
                            ),
                            value=T("34.5"),
                        ),
                        Attribute(
                            name=CallName(
                                name=T("r_custom"),
                                args=make_seq(opener="(", items=[T("-3")], delimiter=COMMA),
                            ),
                            value=T("77.9"),
                        ),
                    ],
                )
            ],
            id="negative_indices",
        ),
        pytest.param(
            "parameter[particle] = #12C+3",
            [
                Parameter(
                    target=T("parameter"),
                    name=T("particle"),
                    value=T("#12C+3"),
                )
            ],
            id="species_names_triply_charged_carbon",
        ),
        pytest.param(
            "parameter[p0c] = 12 * 500e6",
            [
                Parameter(
                    target=T("parameter"),
                    name=T("p0c"),
                    value=Seq(
                        items=[
                            T("12"),
                            T("*"),
                            T("500e6"),
                        ],
                    ),
                )
            ],
            id="species_names_reference_momentum",
        ),
        pytest.param(
            "parameter[particle] = He--",
            [
                Parameter(
                    target=T("parameter"),
                    name=T("particle"),
                    value=T("He--"),
                )
            ],
            id="species_names_doubly_charged_he",
        ),
        pytest.param(
            "parameter[particle] = C2H3@M28.4+",
            [
                Parameter(
                    target=T("parameter"),
                    name=T("particle"),
                    value=T("C2H3@M28.4+"),
                )
            ],
            id="species_names_singly_charged_c2h3",
        ),
        pytest.param(
            "parameter[particle] = CH2",
            [
                Parameter(
                    target=T("parameter"),
                    name=T("particle"),
                    value=T("CH2"),
                )
            ],
            id="species_names_natural_ch2",
        ),
        pytest.param(
            "parameter[particle] = @M37.54++",
            [
                Parameter(
                    target=T("parameter"),
                    name=T("particle"),
                    value=T("@M37.54++"),
                )
            ],
            id="species_names_doubly_charged_molecule",
        ),
        pytest.param(
            """
! Overlay element definition with internal comments
ov1: overlay = {  ! ele_comment
    q1,  ! list_comment
    q2
}, master_attr = 3.4, flag_attr  ! flag_attr comment
    """,
            [
                Element(
                    comments=Comments(
                        pre=[T(" Overlay element definition with internal comments")],
                        inline=T(" ele_comment"),
                    ),
                    name=T("ov1"),
                    keyword=T("overlay"),
                    ele_list=Seq(
                        opener=LBRACE,
                        closer=RBRACE,
                        items=[
                            T("q1", comments=Comments(inline=T(" list_comment"))),
                            T("q2"),
                        ],
                        delimiter=COMMA,
                    ),
                    attributes=[
                        Attribute(
                            name=T(
                                "master_attr",
                            ),
                            value=T("3.4"),
                        ),
                        Attribute(
                            name=T(
                                "flag_attr",
                                comments=Comments(inline=T(" flag_attr comment")),
                            ),
                            value=None,
                        ),
                    ],
                )
            ],
            id="overlay_parsing",
        ),
        pytest.param(
            """
A.B.Gat02: IN.CM1.Gat00,  !inline
    wall = {
section = { s = 0,  !0-comment
  v(1) = {0, 0, A.B.Gat02[aperture]}},  !aperture-comment
section = { s = A.B.Gat02[L],
  v(1) = {0, 0, A.B.Gat02[aperture]}}}
""",
            [
                Element(
                    comments=Comments(inline=Token("inline")),
                    name=Token("A.B.Gat02"),
                    keyword=Token("IN.CM1.Gat00"),
                    ele_list=None,
                    attributes=[
                        Attribute(
                            name=Token("wall"),
                            value=Seq(
                                opener=LBRACE,
                                closer=RBRACE,
                                items=[
                                    Attribute(
                                        name=Token("section"),
                                        value=Seq(
                                            opener=LBRACE,
                                            closer=RBRACE,
                                            items=[
                                                Attribute(
                                                    name=Token("s"),
                                                    value=Token(
                                                        "0",
                                                        comments=Comments(
                                                            inline=Token("0-comment")
                                                        ),
                                                    ),
                                                ),
                                                Attribute(
                                                    name=CallName(
                                                        name=Token("v"),
                                                        args=Seq(
                                                            opener=LPAREN,
                                                            closer=RPAREN,
                                                            items=[Token("1")],
                                                            delimiter=COMMA,
                                                        ),
                                                    ),
                                                    value=Seq(
                                                        opener=LBRACE,
                                                        closer=RBRACE,
                                                        items=[
                                                            Token("0"),
                                                            Token("0"),
                                                            Seq(
                                                                opener=None,
                                                                closer=None,
                                                                items=[
                                                                    Token("A.B.Gat02"),
                                                                    Seq(
                                                                        opener=Delimiter("["),
                                                                        closer=Delimiter("]"),
                                                                        items=[
                                                                            Token(
                                                                                "aperture",
                                                                                comments=Comments(
                                                                                    inline=Token(
                                                                                        "aperture-comment"
                                                                                    )
                                                                                ),
                                                                            )
                                                                        ],
                                                                        delimiter=SPACE,
                                                                    ),
                                                                ],
                                                                delimiter=SPACE,
                                                            ),
                                                        ],
                                                        delimiter=COMMA,
                                                    ),
                                                ),
                                            ],
                                            delimiter=COMMA,
                                        ),
                                    ),
                                    Attribute(
                                        name=Token("section"),
                                        value=Seq(
                                            opener=LBRACE,
                                            closer=RBRACE,
                                            items=[
                                                Attribute(
                                                    name=Token("s"),
                                                    value=Seq(
                                                        opener=None,
                                                        closer=None,
                                                        items=[
                                                            Token("A.B.Gat02"),
                                                            Seq(
                                                                opener=Delimiter("["),
                                                                closer=Delimiter("]"),
                                                                items=[Token("L")],
                                                                delimiter=SPACE,
                                                            ),
                                                        ],
                                                        delimiter=SPACE,
                                                    ),
                                                ),
                                                Attribute(
                                                    name=CallName(
                                                        name=Token("v"),
                                                        args=Seq(
                                                            opener=LPAREN,
                                                            closer=RPAREN,
                                                            items=[Token("1")],
                                                            delimiter=COMMA,
                                                        ),
                                                    ),
                                                    value=Seq(
                                                        opener=LBRACE,
                                                        closer=RBRACE,
                                                        items=[
                                                            Token("0"),
                                                            Token("0"),
                                                            Seq(
                                                                opener=None,
                                                                closer=None,
                                                                items=[
                                                                    Token("A.B.Gat02"),
                                                                    Seq(
                                                                        opener=Delimiter("["),
                                                                        closer=Delimiter("]"),
                                                                        items=[Token("aperture")],
                                                                        delimiter=SPACE,
                                                                    ),
                                                                ],
                                                                delimiter=SPACE,
                                                            ),
                                                        ],
                                                        delimiter=Delimiter(","),
                                                    ),
                                                ),
                                            ],
                                            delimiter=Delimiter(","),
                                        ),
                                    ),
                                ],
                                delimiter=Delimiter(","),
                            ),
                        )
                    ],
                )
            ],
            id="wall_parsing",
        ),
        pytest.param(
            """
QTov1 : overlay = {QT1[k1]:k1*dk , QT4[k1]:k1*dk , QT13[k1]:k1*dk, QT16[k1]:k1*dk}, var = {k1,dk}, k1= -0.553299, dk=1.0
""",
            [
                Element(
                    name=T("QTov1"),
                    keyword=T("overlay"),
                    ele_list=Seq(
                        opener=LBRACE,
                        closer=RBRACE,
                        items=[
                            Seq(
                                items=[
                                    T("QT1"),
                                    Seq(
                                        opener=Delimiter("["),
                                        closer=Delimiter("]"),
                                        items=[T("k1")],
                                    ),
                                    Delimiter(":"),
                                    T("k1"),
                                    Delimiter("*"),
                                    T("dk"),
                                ],
                            ),
                            Seq(
                                items=[
                                    T("QT4"),
                                    Seq(
                                        opener=Delimiter("["),
                                        closer=Delimiter("]"),
                                        items=[T("k1")],
                                    ),
                                    Delimiter(":"),
                                    T("k1"),
                                    Delimiter("*"),
                                    T("dk"),
                                ],
                            ),
                            Seq(
                                items=[
                                    T("QT13"),
                                    Seq(
                                        opener=Delimiter("["),
                                        closer=Delimiter("]"),
                                        items=[T("k1")],
                                    ),
                                    Delimiter(":"),
                                    T("k1"),
                                    Delimiter("*"),
                                    T("dk"),
                                ],
                            ),
                            Seq(
                                items=[
                                    T("QT16"),
                                    Seq(
                                        opener=Delimiter("["),
                                        closer=Delimiter("]"),
                                        items=[T("k1")],
                                    ),
                                    Delimiter(":"),
                                    T("k1"),
                                    Delimiter("*"),
                                    T("dk"),
                                ],
                            ),
                        ],
                        delimiter=COMMA,
                    ),
                    attributes=[
                        Attribute(
                            name=T("var"),
                            value=Seq(
                                opener=LBRACE,
                                closer=RBRACE,
                                items=[T("k1"), T("dk")],
                                delimiter=COMMA,
                            ),
                        ),
                        Attribute(name=T("k1"), value=T("-0.553299")),
                        Attribute(name=T("dk"), value=T("1.0")),
                    ],
                )
            ],
            id="overlay_with_expressions",
        ),
        pytest.param(
            """
HB01: overlay = {B04W /0.5, B05W /0.5}, HKICK, Alias = "HB1", &
        Type = "CSR HBND CUR   1"
""",
            [
                Element(
                    name=T("HB01"),
                    keyword=T("overlay"),
                    ele_list=Seq(
                        opener=LBRACE,
                        closer=RBRACE,
                        items=[
                            Seq(items=[T("B04W"), SLASH, T("0.5")]),
                            Seq(items=[T("B05W"), SLASH, T("0.5")]),
                        ],
                        delimiter=COMMA,
                    ),
                    attributes=[
                        Attribute(name=T("HKICK"), value=None),
                        Attribute(name=T("Alias"), value=T('"HB1"')),
                        Attribute(name=T("Type"), value=T('"CSR HBND CUR   1"')),
                    ],
                )
            ],
            id="overlay_with_division_weight",
        ),
        pytest.param(
            """
O_B7_b12: overlay = {B7, B12, B4:-g*b7[l]/b4[l]}, var = {g}
""",
            [
                Element(
                    name=T("O_B7_b12"),
                    keyword=T("overlay"),
                    ele_list=Seq(
                        opener=LBRACE,
                        closer=RBRACE,
                        delimiter=COMMA,
                        items=[
                            T("B7"),
                            T("B12"),
                            Seq(
                                items=[
                                    T("B4"),
                                    Delimiter(":"),
                                    Delimiter("-"),
                                    T("g"),
                                    Delimiter("*"),
                                    T("b7"),
                                    make_seq(
                                        opener=Delimiter("["), items=[T("l")], delimiter=SPACE
                                    ),
                                    Delimiter("/"),
                                    T("b4"),
                                    make_seq(
                                        opener=Delimiter("["), items=[T("l")], delimiter=SPACE
                                    ),
                                ],
                            ),
                        ],
                    ),
                    attributes=[
                        Attribute(
                            name=T("var"),
                            value=Seq(
                                opener=LBRACE,
                                closer=RBRACE,
                                items=[T("g")],
                                delimiter=COMMA,
                            ),
                        )
                    ],
                )
            ],
            id="overlay_negative_expression",
        ),
        pytest.param(
            """
delta1: group = {len1[L]:0.23061/360*deg}, var={deg}
""",
            [
                Element(
                    name=T("delta1"),
                    keyword=T("group"),
                    ele_list=Seq(
                        opener=LBRACE,
                        closer=RBRACE,
                        delimiter=COMMA,
                        items=[
                            Seq(
                                delimiter=SPACE,
                                items=[
                                    T("len1"),
                                    make_seq(
                                        opener=Delimiter("["), items=[T("L")], delimiter=SPACE
                                    ),
                                    Delimiter(":"),
                                    T("0.23061"),
                                    SLASH,
                                    T("360"),
                                    STAR,
                                    T("deg"),
                                ],
                            )
                        ],
                    ),
                    attributes=[
                        Attribute(
                            name=T("var"),
                            value=Seq(
                                opener=LBRACE,
                                closer=RBRACE,
                                items=[T("deg")],
                                delimiter=COMMA,
                            ),
                        )
                    ],
                )
            ],
            id="group_with_arithmetic",
        ),
        pytest.param(
            """
fig8 : line  = (ir1, L4, ir2, ir1, p1, --L3, p2, ir2);
""",
            [
                Line(
                    name=T("fig8"),
                    elements=Seq(
                        opener=LPAREN,
                        closer=RPAREN,
                        delimiter=COMMA,
                        items=[
                            T("ir1"),
                            T("L4"),
                            T("ir2"),
                            T("ir1"),
                            T("p1"),
                            Seq(
                                items=[
                                    Delimiter("--"),
                                    T("L3"),
                                ],
                                delimiter=SPACE,
                            ),
                            T("p2"),
                            T("ir2"),
                        ],
                    ),
                    multipass=False,
                )
            ],
            id="reversed_line",
        ),
        pytest.param(
            """
cell3 : LINE=  (d1, qd, d2, bb, d2, qf, d1)
L3 : LINE = (bb, d2, qf, d1, 8*cell3, d1, qd, d2, bb)
""",
            [
                Line(
                    name=T("cell3"),
                    elements=Seq(
                        opener=LPAREN,
                        closer=RPAREN,
                        items=[
                            T("d1"),
                            T("qd"),
                            T("d2"),
                            T("bb"),
                            T("d2"),
                            T("qf"),
                            T("d1"),
                        ],
                        delimiter=COMMA,
                    ),
                    multipass=False,
                ),
                Line(
                    name=T("L3"),
                    elements=Seq(
                        opener=LPAREN,
                        closer=RPAREN,
                        items=[
                            T("bb"),
                            T("d2"),
                            T("qf"),
                            T("d1"),
                            Seq(
                                items=[
                                    T("8"),
                                    STAR,
                                    T("cell3"),
                                ],
                                delimiter=SPACE,
                            ),
                            T("d1"),
                            T("qd"),
                            T("d2"),
                            T("bb"),
                        ],
                        delimiter=COMMA,
                    ),
                    multipass=False,
                ),
            ],
            id="line_with_repetition",
        ),
        pytest.param(
            """
b  : rbend,l_arc = larc, angle=ang
bb: b, g_err = -2*ang/larc
""",
            [
                Element(
                    name=T("b"),
                    keyword=T("rbend"),
                    ele_list=None,
                    attributes=[
                        Attribute(name=T("l_arc"), value=T("larc")),
                        Attribute(name=T("angle"), value=T("ang")),
                    ],
                ),
                Element(
                    name=T("bb"),
                    keyword=T("b"),
                    ele_list=None,
                    attributes=[
                        Attribute(
                            name=T("g_err"),
                            value=Seq(
                                items=[
                                    T("-2"),
                                    STAR,
                                    T("ang"),
                                    SLASH,
                                    T("larc"),
                                ],
                                delimiter=SPACE,
                            ),
                        )
                    ],
                ),
            ],
            id="element_with_error",
        ),
        pytest.param(
            """
D_PATCH: patch, x_offset = 0.034, x_pitch = asin(0.32)
""",
            [
                Element(
                    name=T("D_PATCH"),
                    keyword=T("patch"),
                    ele_list=None,
                    attributes=[
                        Attribute(name=T("x_offset"), value=T("0.034")),
                        Attribute(
                            name=T("x_pitch"),
                            value=Seq(
                                items=[
                                    T("asin"),
                                    Seq(
                                        opener=LPAREN,
                                        closer=RPAREN,
                                        items=[T("0.32")],
                                        delimiter=COMMA,
                                    ),
                                ],
                                delimiter=SPACE,
                            ),
                        ),
                    ],
                )
            ],
            id="patch_with_expression",
        ),
        pytest.param(
            """
RETURN1.TIME_PATCH[T_OFFSET] = +1.22459198215588780E-10
RETURN7.TIME_PATCH[T_OFFSET] = RETURN1.TIME_PATCH[T_OFFSET]
""",
            [
                Parameter(
                    target=T("RETURN1.TIME_PATCH"),
                    name=T("T_OFFSET"),
                    value=T("+1.22459198215588780E-10"),
                ),
                Parameter(
                    target=T("RETURN7.TIME_PATCH"),
                    name=T("T_OFFSET"),
                    value=Seq(
                        items=[
                            T("RETURN1.TIME_PATCH"),
                            Seq(
                                opener=Delimiter("["),
                                closer=Delimiter("]"),
                                items=[T("T_OFFSET")],
                            ),
                        ],
                    ),
                ),
            ],
            id="element_attribute_reference",
        ),
        pytest.param(
            """
RETURN1.TIME_MATCH[DELTA_TIME] =  -8.273e-12
RETURN5.TIME_MATCH[DELTA_TIME] =  return3.time_match[delta_time]
""",
            [
                Parameter(
                    target=T("RETURN1.TIME_MATCH"),
                    name=T("DELTA_TIME"),
                    value=T("-8.273e-12"),
                ),
                Parameter(
                    target=T("RETURN5.TIME_MATCH"),
                    name=T("DELTA_TIME"),
                    value=Seq(
                        items=[
                            T("return3.time_match"),
                            Seq(
                                opener=Delimiter("["),
                                closer=Delimiter("]"),
                                items=[T("delta_time")],
                            ),
                        ],
                    ),
                ),
            ],
            id="lowercase_attribute_reference",
        ),
        pytest.param(
            """
cavity7[wall] = {
section = { s = 0., v(1) = {0.0, 0.0, 0.05519118826235362}},
}
""",
            [
                Parameter(
                    target=T("cavity7"),
                    name=T("wall"),
                    value=Seq(
                        opener=LBRACE,
                        closer=RBRACE,
                        delimiter=COMMA,
                        items=[
                            Attribute(
                                name=T("section"),
                                value=Seq(
                                    opener=LBRACE,
                                    closer=RBRACE,
                                    delimiter=COMMA,
                                    items=[
                                        Attribute(name=T("s"), value=T("0.")),
                                        Attribute(
                                            name=CallName(
                                                name=T("v"),
                                                args=make_seq(
                                                    opener="(", items=[T("1")], delimiter=COMMA
                                                ),
                                            ),
                                            value=Seq(
                                                opener=LBRACE,
                                                closer=RBRACE,
                                                items=[
                                                    T("0.0"),
                                                    T("0.0"),
                                                    T("0.05519118826235362"),
                                                ],
                                                delimiter=COMMA,
                                            ),
                                        ),
                                    ],
                                ),
                            )
                        ],
                    ),
                )
            ],
            id="wall_multiline_definition",
        ),
        pytest.param(
            """
IN.CRMOD.Pip01: PIPE, L = 5.318*0.0254, aperture = 0.060198/2,
    wall = {
	section = { s = 0,
	  v(1) = {0, 0, IN.CRMOD.Pip01[aperture]}},
	section = { s = IN.CRMOD.Pip01[L],
	  v(1) = {0, 0, IN.CRMOD.Pip01[aperture]}}}
""",
            [
                Element(
                    name=T("IN.CRMOD.Pip01"),
                    keyword=T("PIPE"),
                    ele_list=None,
                    attributes=[
                        Attribute(
                            name=T("L"),
                            value=Seq(
                                items=[
                                    T("5.318"),
                                    STAR,
                                    T("0.0254"),
                                ]
                            ),
                        ),
                        Attribute(
                            name=T("aperture"),
                            value=Seq(
                                items=[
                                    T("0.060198"),
                                    SLASH,
                                    T("2"),
                                ]
                            ),
                        ),
                        Attribute(
                            name=T("wall"),
                            value=Seq(
                                opener=LBRACE,
                                closer=RBRACE,
                                delimiter=COMMA,
                                items=[
                                    Attribute(
                                        name=T("section"),
                                        value=Seq(
                                            opener=LBRACE,
                                            closer=RBRACE,
                                            delimiter=COMMA,
                                            items=[
                                                Attribute(name=T("s"), value=T("0")),
                                                Attribute(
                                                    name=CallName(
                                                        name=T("v"),
                                                        args=make_seq(
                                                            opener="(",
                                                            items=[T("1")],
                                                            delimiter=COMMA,
                                                        ),
                                                    ),
                                                    value=Seq(
                                                        opener=LBRACE,
                                                        closer=RBRACE,
                                                        delimiter=COMMA,
                                                        items=[
                                                            T("0"),
                                                            T("0"),
                                                            Seq(
                                                                items=[
                                                                    T("IN.CRMOD.Pip01"),
                                                                    Seq(
                                                                        opener=Delimiter("["),
                                                                        closer=Delimiter("]"),
                                                                        items=[T("aperture")],
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                ),
                                            ],
                                        ),
                                    ),
                                    Attribute(
                                        name=T("section"),
                                        value=Seq(
                                            opener=LBRACE,
                                            closer=RBRACE,
                                            delimiter=COMMA,
                                            items=[
                                                Attribute(
                                                    name=T("s"),
                                                    value=Seq(
                                                        items=[
                                                            T("IN.CRMOD.Pip01"),
                                                            Seq(
                                                                opener=Delimiter("["),
                                                                closer=Delimiter("]"),
                                                                items=[T("L")],
                                                            ),
                                                        ],
                                                    ),
                                                ),
                                                Attribute(
                                                    name=CallName(
                                                        name=T("v"),
                                                        args=make_seq(
                                                            opener="(",
                                                            items=[T("1")],
                                                            delimiter=COMMA,
                                                        ),
                                                    ),
                                                    value=Seq(
                                                        opener=LBRACE,
                                                        closer=RBRACE,
                                                        delimiter=COMMA,
                                                        items=[
                                                            T("0"),
                                                            T("0"),
                                                            Seq(
                                                                items=[
                                                                    T("IN.CRMOD.Pip01"),
                                                                    Seq(
                                                                        opener=Delimiter("["),
                                                                        closer=Delimiter("]"),
                                                                        items=[T("aperture")],
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                ),
                                            ],
                                        ),
                                    ),
                                ],
                            ),
                        ),
                    ],
                )
            ],
            id="complex_wall_definition",
        ),
        pytest.param(
            """
facT : overlay = {QTov1, QTov2, QTov3, QTov4, QTov5, QTov6, QTov7, QTov8}, var={dk}, dk=1.0193
""",
            [
                Element(
                    name=T("facT"),
                    keyword=T("overlay"),
                    ele_list=Seq(
                        opener=LBRACE,
                        closer=RBRACE,
                        delimiter=COMMA,
                        items=[
                            T("QTov1"),
                            T("QTov2"),
                            T("QTov3"),
                            T("QTov4"),
                            T("QTov5"),
                            T("QTov6"),
                            T("QTov7"),
                            T("QTov8"),
                        ],
                    ),
                    attributes=[
                        Attribute(
                            name=T("var"),
                            value=Seq(
                                opener=LBRACE,
                                closer=RBRACE,
                                items=[T("dk")],
                                delimiter=COMMA,
                            ),
                        ),
                        Attribute(name=T("dk"), value=T("1.0193")),
                    ],
                )
            ],
            id="overlay_multiple_masters",
        ),
        pytest.param(
            """
tmatch : match, beta_a0=1, beta_a1=1, beta_b0=1, beta_b1=1
return1.time_match: tmatch
""",
            [
                Element(
                    name=T("tmatch"),
                    keyword=T("match"),
                    ele_list=None,
                    attributes=[
                        Attribute(name=T("beta_a0"), value=T("1")),
                        Attribute(name=T("beta_a1"), value=T("1")),
                        Attribute(name=T("beta_b0"), value=T("1")),
                        Attribute(name=T("beta_b1"), value=T("1")),
                    ],
                ),
                Element(name=T("return1.time_match"), keyword=T("tmatch")),
            ],
            id="match_element_parameters",
        ),
        pytest.param(
            """
A_PATCH: patch, flexible = T
""",
            Element(
                name=T("A_PATCH"),
                keyword=T("patch"),
                ele_list=None,
                attributes=[Attribute(name=T("flexible"), value=T("T"))],
            ),
            id="flexible_patch",
        ),
        pytest.param(
            """
*[tracking_method] = bmad_standard  ! Matches all elements.
""",
            [
                Parameter(
                    comments=Comments(inline=T(" Matches all elements.")),
                    target=STAR,
                    name=T("tracking_method"),
                    value=T("bmad_standard"),
                )
            ],
            id="wildcard_all",
        ),
        pytest.param(
            """
Q%1[k1] = 0.234                     ! Matches to "Q01" but not "Q001".
""",
            [
                Parameter(
                    comments=Comments(inline=T(' Matches to "Q01" but not "Q001".')),
                    target=T("Q%1"),
                    name=T("k1"),
                    value=T("0.234"),
                )
            ],
            id="single_character_match",
        ),
        pytest.param(
            """
quadrupole::Q*[k1] = 0.234    ! Matches all quadrupoles with names beginning with Q.
""",
            [
                Parameter(
                    comments=Comments(
                        inline=T(" Matches all quadrupoles with names beginning with Q.")
                    ),
                    target=Seq(
                        delimiter=SPACE,
                        items=[T("quadrupole"), Delimiter("::"), T("Q"), Delimiter("*")],
                    ),
                    name=T("k1"),
                    value=T("0.234"),
                )
            ],
            id="class_qualified_wildcard",
        ),
        pytest.param(
            """
expand_lattice              ! Expand the lattice.
97[x_offset] = 0.0023       ! Set x_offset attribute of 97th element
b2>>si_cryst##2[tilt] = 0.1 ! Tilt the 2nd instance of "si_cryst" in branch "b2"
5:32[x_limit] = 0.3         ! Sets elements with indexes 5 through 32 in branch 0.
""",
            [
                Simple(
                    comments=Comments(inline=T(" Expand the lattice.")),
                    statement=T("expand_lattice"),
                    arguments=[],
                ),
                Parameter(
                    comments=Comments(inline=T(" Set x_offset attribute of 97th element")),
                    target=T("97"),
                    name=T("x_offset"),
                    value=T("0.0023"),
                ),
                Parameter(
                    comments=Comments(
                        inline=T(' Tilt the 2nd instance of "si_cryst" in branch "b2"')
                    ),
                    target=T("b2>>si_cryst##2"),
                    name=T("tilt"),
                    value=T("0.1"),
                ),
                Parameter(
                    comments=Comments(
                        inline=T(" Sets elements with indexes 5 through 32 in branch 0.")
                    ),
                    target=Seq(items=[T("5"), Delimiter(":"), T("32")]),
                    name=T("x_limit"),
                    value=T("0.3"),
                ),
            ],
            id="element_index_syntax",
        ),
        pytest.param(
            """
2>>45[x_pitch] = 0.1
x_br>>quad::q*[k1] = 0.5
""",
            [
                Parameter(target=T("2>>45"), name=T("x_pitch"), value=T("0.1")),
                Parameter(
                    target=Seq(
                        delimiter=SPACE,
                        items=[T("x_br>>quad"), Delimiter("::"), T("q"), Delimiter("*")],
                    ),
                    name=T("k1"),
                    value=T("0.5"),
                ),
            ],
            id="branch_qualified_name",
        ),
        pytest.param(
            """
sbend::q1:q5[field_master] = T
""",
            [
                Parameter(
                    target=Seq(
                        delimiter=SPACE,
                        items=[
                            T("sbend"),
                            Delimiter("::"),
                            T("q1"),
                            Delimiter(":"),
                            T("q5"),
                        ],
                    ),
                    name=T("field_master"),
                    value=T("T"),
                )
            ],
            id="element_range_syntax_between_named_eles",
        ),
        pytest.param(
            """
3,15:17[x_offset] = 0.001
""",
            [
                Parameter(
                    target=Seq(
                        delimiter=COMMA,
                        items=[T("3"), Seq(items=[T("15"), Delimiter(":"), T("17")])],
                    ),
                    name=T("x_offset"),
                    value=T("0.001"),
                )
            ],
            id="element_range_syntax_with_commas",
        ),
        pytest.param(
            """
3 15:17[x_offset] = 0.001
""",
            [
                Parameter(
                    target=Seq(
                        items=[T("3"), T("15"), Delimiter(":"), T("17")],
                    ),
                    name=T("x_offset"),
                    value=T("0.001"),
                )
            ],
            id="element_range_syntax_without_commas",
        ),
        pytest.param(
            """
100:200 & sbend::*[tracking_method] = runge_kutta
""",
            [
                Parameter(
                    target=Seq(
                        items=[
                            T("100"),
                            Delimiter(":"),
                            T("200"),
                            Delimiter("&"),
                            T("sbend"),
                            Delimiter("::"),
                            T("*"),
                        ],
                    ),
                    name=T("tracking_method"),
                    value=T("runge_kutta"),
                )
            ],
            id="list_intersection",
        ),
        pytest.param(
            """
q* & quad::* b1[k1] = 0.5
""",
            [
                Parameter(
                    target=Seq(
                        items=[
                            T("q"),
                            Delimiter("*"),
                            Delimiter("&"),
                            T("quad"),
                            Delimiter("::"),
                            T("*"),
                            T("b1"),
                        ],
                    ),
                    name=T("k1"),
                    value=T("0.5"),
                )
            ],
            id="list_intersection_three",
        ),
        pytest.param(
            """
*::*, ~quadrupole::*[x_offset] = 0.001  ! All elements except quadrupoles
""",
            [
                Parameter(
                    comments=Comments(inline=T(" All elements except quadrupoles")),
                    target=Seq(
                        delimiter=COMMA,
                        items=[
                            Seq(delimiter=SPACE, items=[T("*"), Delimiter("::"), T("*")]),
                            Seq(delimiter=SPACE, items=[T("~quadrupole"), Delimiter("::"), T("*")]),
                        ],
                    ),
                    name=T("x_offset"),
                    value=T("0.001"),
                )
            ],
            id="list_exclusion",
        ),
        pytest.param(
            """
! All elements with names beginning with B except elements with names beginning
! with B1. However, elements named B13 are retained.
b*, ~b1*, b13[hkick] = 0.01
""",
            [
                Parameter(
                    comments=Comments(
                        pre=[
                            T(
                                " All elements with names beginning with B except elements with names beginning"
                            ),
                            T(" with B1. However, elements named B13 are retained."),
                        ]
                    ),
                    target=Seq(
                        delimiter=COMMA,
                        items=[
                            Seq(delimiter=SPACE, items=[T("b"), Delimiter("*")]),
                            Seq(delimiter=SPACE, items=[T("~b1"), Delimiter("*")]),
                            T("b13"),
                        ],
                    ),
                    name=T("hkick"),
                    value=T("0.01"),
                )
            ],
            id="list_exclusion1",
        ),
        pytest.param(
            """
type::"det bpm*"[x_pitch] = 0.1  ! All elements with a type starting with "det bpm"
alias::*[is_on] = F  ! all elements with a non-blank alias
""",
            [
                Parameter(
                    comments=Comments(
                        inline=T(' All elements with a type starting with "det bpm"')
                    ),
                    target=Seq(
                        delimiter=SPACE, items=[T("type"), Delimiter("::"), T('"det bpm*"')]
                    ),
                    name=T("x_pitch"),
                    value=T("0.1"),
                ),
                Parameter(
                    comments=Comments(inline=T(" all elements with a non-blank alias")),
                    target=Seq(delimiter=SPACE, items=[T("alias"), Delimiter("::"), T("*")]),
                    name=T("is_on"),
                    value=T("F"),
                ),
            ],
            id="attribute_matching",
        ),
        pytest.param(
            """
s%z[k2] = %[k2] + 0.03 * ran_gauss()  ! %[k2] refers to the element being set (like s0z)
""",
            [
                Parameter(
                    comments=Comments(
                        inline=T(" %[k2] refers to the element being set (like s0z)")
                    ),
                    target=T("s%z"),
                    name=T("k2"),
                    value=Seq(
                        items=[
                            T("%"),
                            Seq(
                                opener=Delimiter("["),
                                closer=Delimiter("]"),
                                items=[T("k2")],
                                delimiter=SPACE,
                            ),
                            T("+"),
                            T("0.03"),
                            T("*"),
                            T("ran_gauss"),
                            Seq(opener=LPAREN, closer=RPAREN, items=[], delimiter=COMMA),
                        ],
                    ),
                )
            ],
            id="percent_in_expression",
        ),
        pytest.param(
            """
c: crystal, call::$AB/my_curvature.bmad, h_misalign = call::my_surface.bmad
""",
            [
                Element(
                    name=T("c"),
                    keyword=T("crystal"),
                    ele_list=None,
                    attributes=[
                        Attribute(
                            name=Seq(
                                delimiter=SPACE,
                                items=[
                                    T("call"),
                                    Delimiter("::"),
                                    T("$AB"),
                                    Delimiter("/"),
                                    T("my_curvature.bmad"),
                                ],
                            ),
                            value=None,
                        ),
                        Attribute(
                            name=T("h_misalign"),
                            value=Seq(
                                delimiter=SPACE,
                                items=[T("call"), Delimiter("::"), T("my_surface.bmad")],
                            ),
                        ),
                    ],
                )
            ],
            id="inline_call",
        ),
        pytest.param(
            "parameter[particle] = deuteron",
            [Parameter(target=T("parameter"), name=T("particle"), value=T("deuteron"))],
            id="species_function_0",
        ),
        pytest.param(
            "am = anomalous_moment_of(parameter[particle])^2",
            [
                Constant(
                    name=T("am"),
                    value=Seq(
                        delimiter=SPACE,
                        items=[
                            T("anomalous_moment_of"),
                            Seq(
                                opener=LPAREN,
                                closer=RPAREN,
                                delimiter=COMMA,
                                items=[
                                    Seq(
                                        delimiter=SPACE,
                                        items=[
                                            T("parameter"),
                                            Seq(
                                                opener=Delimiter("["),
                                                closer=Delimiter("]"),
                                                items=[T("particle")],
                                                delimiter=SPACE,
                                            ),
                                        ],
                                    )
                                ],
                                # items=[T("parameter[particle]")],
                            ),
                            T("^"),
                            T("2"),
                        ],
                    ),
                    redef=False,
                )
            ],
            id="species_function_1",
        ),
        pytest.param(
            "my_particle = species(He++)      ! my_particle now represents He++",
            [
                Constant(
                    comments=Comments(inline=T(" my_particle now represents He++")),
                    name=T("my_particle"),
                    value=Seq(
                        items=[
                            T("species"),
                            Seq(
                                opener=LPAREN,
                                closer=RPAREN,
                                items=[T("He++")],
                                delimiter=COMMA,
                            ),
                        ],
                    ),
                    redef=False,
                )
            ],
            id="species_function_2",
        ),
        pytest.param(
            "chg1 = charge_of(my_particle)    ! chg = charge of He++",
            [
                Constant(
                    comments=Comments(inline=T(" chg = charge of He++")),
                    name=T("chg1"),
                    value=Seq(
                        items=[
                            T("charge_of"),
                            Seq(
                                opener=LPAREN,
                                closer=RPAREN,
                                items=[T("my_particle")],
                                delimiter=COMMA,
                            ),
                        ],
                    ),
                    redef=False,
                )
            ],
            id="species_function_3",
        ),
        pytest.param(
            "chg2 = charge_of(He++)           ! Same as previous line",
            [
                Constant(
                    comments=Comments(inline=T(" Same as previous line")),
                    name=T("chg2"),
                    value=Seq(
                        items=[
                            T("charge_of"),
                            Seq(
                                opener=LPAREN,
                                closer=RPAREN,
                                items=[T("He++")],
                                delimiter=COMMA,
                            ),
                        ],
                    ),
                    redef=False,
                )
            ],
            id="species_function_4",
        ),
        pytest.param(
            "chg3 = charge_of(species(He++))  ! Same as previous line",
            [
                Constant(
                    comments=Comments(inline=T(" Same as previous line")),
                    name=T("chg3"),
                    value=Seq(
                        items=[
                            T("charge_of"),
                            Seq(
                                opener=LPAREN,
                                closer=RPAREN,
                                delimiter=COMMA,
                                items=[
                                    Seq(
                                        delimiter=SPACE,
                                        items=[
                                            T("species"),
                                            Seq(
                                                opener=LPAREN,
                                                closer=RPAREN,
                                                items=[T("He++")],
                                                delimiter=COMMA,
                                            ),
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                    redef=False,
                )
            ],
            id="species_function_5",
        ),
        pytest.param(
            """
a[x_offset] = 0.001*ran_gauss(3.0)
b[y_offset] = 0.002*ran_gauss()
""",
            [
                Parameter(
                    target=T("a"),
                    name=T("x_offset"),
                    value=Seq(
                        items=[
                            T("0.001"),
                            Delimiter("*"),
                            T("ran_gauss"),
                            Seq(
                                opener=LPAREN,
                                closer=RPAREN,
                                items=[T("3.0")],
                                delimiter=COMMA,
                            ),
                        ],
                    ),
                ),
                Parameter(
                    target=T("b"),
                    name=T("y_offset"),
                    value=Seq(
                        items=[
                            T("0.002"),
                            Delimiter("*"),
                            T("ran_gauss"),
                            Seq(opener=LPAREN, closer=RPAREN, items=[], delimiter=COMMA),
                        ],
                    ),
                ),
            ],
            id="ran_gauss_with_cut",
        ),
        pytest.param(
            """
parameter[E_TOT] = 5e9
parameter[geometry] = open
parameter[particle] = POSITRON
parameter[ran_seed] = 12345
""",
            [
                Parameter(target=T("parameter"), name=T("E_TOT"), value=T("5e9")),
                Parameter(target=T("parameter"), name=T("geometry"), value=T("open")),
                Parameter(target=T("parameter"), name=T("particle"), value=T("POSITRON")),
                Parameter(target=T("parameter"), name=T("ran_seed"), value=T("12345")),
            ],
            id="docs_parameter_statement",
        ),
        pytest.param(
            """
beginning[beta_a] = 14.5011548
beginning[alpha_a] = -0.53828197
beginning[theta_position] = theta_in + phi/2
""",
            [
                Parameter(target=T("beginning"), name=T("beta_a"), value=T("14.5011548")),
                Parameter(target=T("beginning"), name=T("alpha_a"), value=T("-0.53828197")),
                Parameter(
                    target=T("beginning"),
                    name=T("theta_position"),
                    value=Seq(
                        items=[
                            T("theta_in"),
                            Delimiter("+"),
                            T("phi"),
                            Delimiter("/"),
                            T("2"),
                        ]
                    ),
                ),
            ],
            id="docs_beginning_statement",
        ),
        pytest.param(
            """
qq: quadrupole, r_custom(-2,1,5) = 34.5, r_custom(-3) = 77.9
""",
            [
                Element(
                    name=T("qq"),
                    keyword=T("quadrupole"),
                    ele_list=None,
                    attributes=[
                        Attribute(
                            name=CallName(
                                name=T("r_custom"),
                                args=make_seq(
                                    opener="(", items=[T("-2"), T("1"), T("5")], delimiter=COMMA
                                ),
                            ),
                            value=T("34.5"),
                        ),
                        Attribute(
                            name=CallName(
                                name=T("r_custom"),
                                args=make_seq(opener="(", items=[T("-3")], delimiter=COMMA),
                            ),
                            value=T("77.9"),
                        ),
                    ],
                )
            ],
            id="r_custom_array",
        ),
        pytest.param(
            """
qa, quad, l = 0.6, tilt = pi/4  ! Define QA.
qb: qa                          ! QB Inherits from QA.
qa[k1] = 0.12                   ! QB unaffected by modifications of QA.
""",
            [
                Assignment(
                    comments=Comments(inline=T(" Define QA.")),
                    name=Seq(
                        delimiter=COMMA,
                        items=[T("qa"), T("quad"), T("l")],
                    ),
                    value=Seq(
                        delimiter=COMMA,
                        items=[
                            T("0.6"),
                            Attribute(
                                name=T("tilt"),
                                value=Seq(delimiter=SPACE, items=[T("pi"), T("/"), T("4")]),
                            ),
                        ],
                    ),
                ),
                Element(
                    comments=Comments(inline=T(" QB Inherits from QA.")),
                    name=T("qb"),
                    keyword=T("qa"),
                ),
                Parameter(
                    comments=Comments(inline=T(" QB unaffected by modifications of QA.")),
                    target=T("qa"),
                    name=T("k1"),
                    value=T("0.12"),
                ),
            ],
            id="element_inheritance",
        ),
        pytest.param(
            """
BR: fork, to_line = RING_L, to_element = M
""",
            [
                Element(
                    name=T("BR"),
                    keyword=T("fork"),
                    ele_list=None,
                    attributes=[
                        Attribute(name=T("to_line"), value=T("RING_L")),
                        Attribute(name=T("to_element"), value=T("M")),
                    ],
                )
            ],
            id="fork_element",
        ),
        pytest.param(
            """
parser_debug var lat ele 34 78
""",
            [
                Simple(
                    statement=T("parser_debug"),
                    arguments=[
                        T("var"),
                        T("lat"),
                        T("ele"),
                        T("34"),
                        T("78"),
                    ],
                )
            ],
            id="parser_debug",
        ),
        pytest.param(
            """
print Remember Q01 quad strength of `q01[k1]` not yet optimized
""",
            [
                Simple(
                    statement=T("print"),
                    arguments=[
                        T(word)
                        for word in "Remember Q01 quad strength of `q01[k1]` not yet optimized".split()
                    ],
                )
            ],
            marks=pytest.mark.xfail,
            id="print_unquoted",
        ),
        pytest.param(
            """
print "Remember Q01 quad strength of `q01[k1]` not yet optimized"
""",
            [
                Simple(
                    statement=T("print"),
                    arguments=[T('"Remember Q01 quad strength of `q01[k1]` not yet optimized"')],
                )
            ],
            id="print_with_quotes",
        ),
        pytest.param(
            """
call, no_abort_on_open_error, file = "file a b c"
""",
            [
                Simple(
                    statement=T("call"),
                    arguments=[
                        Attribute(name=T("no_abort_on_open_error"), value=None),
                        Attribute(name=T("file"), value=T('"file a b c"')),
                    ],
                )
            ],
            id="call_no_abort_on_error",
        ),
        pytest.param(
            """
call , f = "file a b c"
""",
            [
                Simple(
                    statement=T("call"),
                    arguments=[
                        Attribute(name=T("f"), value=T('"file a b c"')),
                    ],
                )
            ],
            id="call_regular",
        ),
        pytest.param(
            """
beam, particle = electron, n_part = 1.6e10
""",
            [
                Simple(
                    statement=T("beam"),
                    arguments=[
                        Attribute(name=T("particle"), value=T("electron")),
                        Attribute(name=T("n_part"), value=T("1.6e10")),
                    ],
                )
            ],
            id="beam_statement",
        ),
        pytest.param(
            """
particle_start[x]    = 0.2
""",
            [Parameter(target=T("particle_start"), name=T("x"), value=T("0.2"))],
            id="particle_start",
        ),
        pytest.param(
            """
title, "foo"
""",
            [Simple(statement=T("title"), arguments=[Attribute(name=T('"foo"'))])],
            id="title_oneline",
        ),
        pytest.param(
            """
title
"foo"
""",
            [Simple(statement=T("title"), arguments=[Attribute(name=T('"foo"'))])],
            id="title_twoline",
        ),
    ],
)


@code_tests
def test_parsing(code: str, expected) -> None:
    res = parse_verbose(code)

    if not isinstance(expected, list):
        (res,) = res

    assert res == expected


@code_tests
def test_roundtrip(code: str, expected) -> None:
    roundtrip_code(code)


@pytest.mark.parametrize(
    ("code",),
    [
        pytest.param("ename[amp_vs_time(1)%time]"),
        pytest.param("ename[amp_vs_time(N)%amp]"),  #
        pytest.param("ename[frequencies(N)%freq]"),
        pytest.param("ename[frequencies(N)%amp]"),  #
        pytest.param("ename[frequencies(N)%phi]"),  #
        pytest.param("ename[cartesian_map(N)%field_scale]"),  #
        pytest.param("ename[cartesian_map(N)%r0(1)]"),
        pytest.param("ename[cartesian_map(N)%r0(2)]"),
        pytest.param("ename[cartesian_map(N)%r0(3)]"),
        pytest.param("ename[cartesian_map(N)%master_parameter]"),
        pytest.param("ename[cartesian_map(N)%t(M)%A]"),  # -- M^th term in N^th map.
        pytest.param("ename[cartesian_map(N)%t(M)%kx]"),
        pytest.param("ename[cartesian_map(N)%t(M)%ky]"),
        pytest.param("ename[cartesian_map(N)%t(M)%kz]"),
        pytest.param("ename[cartesian_map(N)%t(M)%x0]"),
        pytest.param("ename[cartesian_map(N)%t(M)%y0]"),
        pytest.param("ename[cartesian_map(N)%t(M)%phi_z]"),
        pytest.param("ename[x_knot(N)]"),  # -- N^th x_knot point.
        pytest.param("ename[slave(M)%y_knot(N)]"),  # -- N^th y_knot point for M^th slave.
        pytest.param("ename[r_custom(N1, N2, N3)]"),
        pytest.param(
            "ename[r_custom(N1,N2)]"
        ),  # N2)]                 -- Equivalent to ename[r_custom(N1, N2,  0)]
        pytest.param("ename[r_custom(N1)]"),  # -- Equivalent to ename[r_custom(N1,  0,  0)]
        pytest.param("ename[cylindrical_map(N)%phi0_fieldmap]"),  #
        pytest.param("ename[cylindrical_map(N)%theta0_azimuth]"),
        pytest.param("ename[cylindrical_map(N)%field_scale]"),
        pytest.param("ename[cylindrical_map(N)%dz]"),
        pytest.param("ename[cylindrical_map(N)%r0(1)]"),
        pytest.param("ename[cylindrical_map(N)%r0(2)]"),
        pytest.param("ename[cylindrical_map(N)%r0(3)]"),
        pytest.param("ename[cylindrical_map(N)%master_parameter]"),
        pytest.param("ename[gen_grad_map(N)%field_scale]"),  #
        pytest.param("ename[gen_grad_map(N)%r0(1)]"),
        pytest.param("ename[gen_grad_map(N)%r0(2)]"),
        pytest.param("ename[gen_grad_map(N)%r0(3)]"),
        pytest.param("ename[gen_grad_map(N)%master_parameter]"),
        pytest.param("ename[grid_field(N)%phi0_fieldmap]"),  #
        pytest.param("ename[grid_field(N)%interpolation_order]"),
        pytest.param("ename[grid_field(N)%harmonic]"),
        pytest.param("ename[grid_field(N)%geometry]"),
        pytest.param("ename[grid_field(N)%ename_anchor_pt]"),
        pytest.param("ename[grid_field(N)%phi0_fieldmap]"),
        pytest.param("ename[grid_field(N)%field_scale]"),
        pytest.param("ename[grid_field(N)%dr(1)]"),
        pytest.param("ename[grid_field(N)%dr(2)]"),
        pytest.param("ename[grid_field(N)%dr(3)]"),
        pytest.param("ename[grid_field(N)%r0(1)]"),
        pytest.param("ename[grid_field(N)%r0(2)]"),
        pytest.param("ename[grid_field(N)%r0(3)]"),
        pytest.param("ename[grid_field(N)%master_parameter]"),
        pytest.param("ename[lr_wake%amp_scale]"),
        pytest.param("ename[lr_wake%time_scale]"),
        pytest.param("ename[lr_wake%freq_spread]"),
        pytest.param("ename[lr_wake%mode(N)%freq_in]"),  #
        pytest.param("ename[lr_wake%mode(N)%freq]"),  #
        pytest.param("ename[lr_wake%mode(N)%r_over_q]"),  #
        pytest.param("ename[lr_wake%mode(N)%damp]"),  #
        pytest.param("ename[lr_wake%mode(N)%phi]"),  #
        pytest.param("ename[lr_wake%mode(N)%polar_angle]"),
        pytest.param("ename[lr_wake%mode(N)%polarized]"),  #
        pytest.param("ename[curvature%spherical]"),
        pytest.param("ename[curvature%elliptical_x]"),
        pytest.param("ename[curvature%elliptical_y]"),
        pytest.param("ename[curvature%elliptical_z]"),
        pytest.param("ename[curvature%x(N1)y(N2)]"),
        pytest.param("ename[tt012]"),  # ! Orbital terms. EG: rot[tt13] -> M13 matrix term
        pytest.param("ename[ttS0123]"),  # ! S0 spin quaternion terms.
        pytest.param("ename[ttSx123]"),  # ! Sx spin quaternion terms.
        pytest.param("ename[ttSy123]"),  # ! Sy spin quaternion terms.
        pytest.param("ename[ttSz123]"),  # ! Sz spin quaternion terms.
        pytest.param("ename[wall%section(N)%s]"),
        pytest.param("ename[wall%section(N)%wall%dr_ds]"),
        pytest.param("ename[wall%section(N)%v(M)%x]"),
        pytest.param("ename[wall%section(N)%v(M)%y]"),
        pytest.param("ename[wall%section(N)%v(M)%radius_x]"),
        pytest.param("ename[wall%section(N)%v(M)%radius_y]"),
        pytest.param("ename[wall%section(N)%v(M)%tilt]"),
    ],
)
def test_nonstandard_parameter_syntax(code: str) -> None:
    target, rest = code.split("[", 1)
    name = rest.rstrip("]")

    expected = NonstandardParameter(
        target=T(target),
        name=T(name),
        value=T("0"),
    )

    code = f"{code} = 0"
    (res,) = parse_verbose(code)

    if not isinstance(res, NonstandardParameter):
        expected = Parameter(
            target=T(target),
            name=T(name),
            value=T("0"),
        )
        # TODO this is aliased to regular Parameter; is it an issue?

    if expected.name == "r_custom(N1, N2, N3)":
        # TODO note spacing
        expected.name = T("r_custom(N1,N2,N3)")

    assert res == expected

    # These aren't in the parametrized list, don't overlook them
    roundtrip_code(code)


def test_attribute_in_parameter_assignment() -> None:
    code = """
foo[bar] = bar = baz
"""
    with pytest.raises(UnexpectedAssignment):
        parse_verbose(code)


def test_attribute_in_constant_target() -> None:
    code = """
foo = bar = baz
"""
    with pytest.raises(UnexpectedAssignment):
        parse_verbose(code)
