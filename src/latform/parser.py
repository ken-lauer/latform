from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Sequence

from .const import EQUALS
from .exceptions import UnexpectedAssignment
from .statements import (
    Assignment,
    Constant,
    Element,
    ElementList,
    Empty,
    Line,
    NonstandardParameter,
    Parameter,
    Simple,
    Statement,
)
from .token import Comments
from .tokenizer import tokenize
from .types import (
    COMMA,
    Attribute,
    Block,
    CallName,
    Delimiter,
    FormatOptions,
    Seq,
    Token,
    TokenizerItem,
)
from .util import partition_items


def _make_attribute(item: Attribute | Token | Seq) -> Attribute:
    if isinstance(item, Attribute):
        return item
    if isinstance(item, Delimiter):
        raise ValueError(f"Unexpected delimiter found in place of attribute: {item} at {item.loc}")
    if isinstance(item, Token):
        return Attribute(name=item)
    if isinstance(item, Seq):
        return Attribute(name=item)
    raise ValueError(f"Unexpected item found in place of attribute: {item} at {item.loc}")


def _make_attribute_list(items: list[TokenizerItem]) -> list[Attribute]:
    item = Seq.from_items(items)
    if not isinstance(item, Seq):
        return [_make_attribute(item)]

    return [_make_attribute(item) for item in item.items]


def _is_multipass_marker(blk: Block) -> bool:
    """Check if Array represents a multipass marker."""
    return (
        blk.opener == "["
        and len(blk.items) == 1
        and isinstance(blk.items[0], Token)
        and str(blk.items[0]).lower() == "multipass"
    )


def _extract_leading_comment(first: TokenizerItem) -> Comments:
    comments = first.comments.clone()
    first.comments.clear()
    return comments


def _nab_comments(items) -> Comments:
    res = Comments()
    inline = []
    for item in items:
        if isinstance(item, Seq):
            comment = _nab_comments(item.items)
        else:
            comment = item.comments

        res.pre.extend(comment.pre)
        comment.pre.clear()
        if comment.inline:
            inline.append(comment.inline)
            comment.inline = None

    if not inline:
        pass
    elif len(inline) == 1:
        res.inline = inline[0]
    else:
        res.pre.extend(inline[:-1])
        res.inline = inline[-1]
    return res


def _line_elements_from_block(block: Block) -> Seq:
    if block.opener != "(":
        raise ValueError(f"Unexpected block opener: {block.opener}")

    eles = Seq.from_delimited_block(block, delimiter=COMMA)
    assert isinstance(eles, Seq)
    for ele in eles.items:
        match ele:
            case Seq(items=["-", "-", name]):
                ele.items = [Delimiter("--"), name]

    return eles


def parse_items(items: list[TokenizerItem]):
    if not items:
        raise ValueError("No items provided")

    first = items[0]
    comments = _extract_leading_comment(first)
    first.comments.clear()

    match items:
        case [Token("beginning") as target, Block() as name, "=", _ as value]:
            return Parameter(
                comments=comments,
                target=target,
                name=name.squeeze_single_token(),
                value=Seq.from_item(value),
            )

        case [Token("parameter") as target, Block() as name, "=", Token() as value]:
            return Parameter(
                comments=comments,
                target=target,
                name=name.squeeze_single_token(),
                value=value,
            )

        case [Token("redef"), ":", Token() as name, "=", *rest]:
            value = Seq.from_items(rest)
            if isinstance(value, Attribute):
                raise UnexpectedAssignment(
                    f"Unexpected named attribute assignment: {value} at {value.loc}"
                )
            return Constant(comments=comments, name=name, value=value, redef=True)

        case [Token() as name, "=", *rest]:
            value = Seq.from_items(rest)
            if isinstance(value, Attribute):
                raise UnexpectedAssignment(
                    f"Unexpected named attribute assignment: {value} at {value.loc}"
                )
            return Constant(comments=comments, name=name, value=value)

        case [Token() as name, ":", Token("list"), "=", Block(opener="(") as elements_block]:
            return ElementList(
                comments=comments,
                name=name,
                elements=_line_elements_from_block(elements_block),
            )

        case [Token() as name, ":", Token("line"), "=", Block(opener="(") as elements_block]:
            return Line(
                comments=comments,
                name=name,
                elements=_line_elements_from_block(elements_block),
            )

        case [
            Token() as name,
            ":",
            Token("line"),
            Block(opener="[") as multipass,
            "=",
            Block(opener="(") as elements_block,
        ] if _is_multipass_marker(multipass):
            return Line(
                comments=comments,
                name=name,
                elements=_line_elements_from_block(elements_block),
                multipass=True,
            )

        case [
            Token() as name,
            Block(opener="(") as line_args,
            ":",
            Token("line"),
            "=",
            Block(opener="(") as elements_block,
        ]:
            return Line(
                comments=comments,
                name=CallName(
                    name=Seq.from_item(name),
                    args=Seq.from_item(line_args),
                ),
                elements=_line_elements_from_block(elements_block),
            )

        case [Token() as name, ":", Token() as element_type, *rest]:
            match rest:
                case ["=", Block(opener="{") as ele_list, *after]:
                    if after and after[0] == COMMA:
                        after = after[1:]
                    return Element(
                        comments=comments,
                        name=name,
                        keyword=element_type,
                        ele_list=Seq.from_delimited_block(ele_list, delimiter=COMMA),
                        attributes=_make_attribute_list(after),
                    )

                case [",", *after]:
                    return Element(
                        comments=comments,
                        name=name,
                        keyword=element_type,
                        attributes=_make_attribute_list(after),
                    )

                case []:
                    return Element(
                        comments=comments,
                        name=name,
                        keyword=element_type,
                        attributes=[],
                    )

        case [Token() as stmt]:
            if stmt == "":
                return Empty(comments=comments)

            return Simple(comments=comments, statement=stmt, arguments=[])

    if isinstance(first, Token):
        if first.lower() in {"print", "parser_debug"}:
            args = items[1:]
            if args[0] == COMMA:
                args = args[1:]
            return Simple(
                comments=comments,
                statement=first,
                arguments=[item.to_token() if isinstance(item, Block) else item for item in args],
            )

        if Simple.is_known_statement(first):
            args = items[1:]

            if args[0] == COMMA:
                args = args[1:]

            attrs = _make_attribute_list(args)
            assert isinstance(first, Token)

            return Simple(
                comments=comments,
                statement=first,
                arguments=attrs,
            )

    # Match assignment patterns
    if EQUALS in items:
        before_equals, _, after_equals = partition_items(items, EQUALS)
        if not before_equals or not after_equals:
            raise ValueError("Unhandled assignment: missing name or value")

        value = Seq.from_items(after_equals)

        if isinstance(value, Attribute):
            raise UnexpectedAssignment(
                f"Unexpected named attribute assignment: {value} at {value.loc}"
            )

        match before_equals:
            # Parameter with [attribute] syntax: target[name] = value
            case [*target, Block(opener="[") as name_block]:
                target = Seq.from_items(target)

                # This couldn't be an attribute as there's no '=' in there
                assert not isinstance(target, Attribute)

                try:
                    name = name_block.squeeze_single_token()
                    if "%" in name:
                        raise ValueError("Nonstandard parameter name")
                except ValueError:
                    name = name_block.to_token(include_opener=False)
                    name = Token(name.replace(" ", ""), comments=name.comments, loc=name.loc)
                    cls = NonstandardParameter
                else:
                    cls = Parameter

                if name == "particle" and isinstance(value, Seq):
                    value = value.to_token(include_opener=False).replace(" ", "")

                return cls(
                    comments=comments,
                    target=target,
                    name=name,
                    value=value,
                )
            # Generic assignment: name = value
            case _:
                name = Seq.from_items(before_equals)
                # This couldn't be an attribute as there's no '=' in there
                assert not isinstance(name, Attribute)
                return Assignment(
                    name=name,
                    value=value,
                    comments=comments,
                )

    raise ValueError("Unhandled - unknown")


def parse(contents: str, filename: pathlib.Path | str = "unset") -> Sequence[Statement]:
    blocks = tokenize(contents, filename)
    return [block.parse() for block in blocks]


def parse_file(filename: pathlib.Path | str) -> Sequence[Statement]:
    contents = pathlib.Path(filename).read_text()
    return parse(contents=contents, filename=filename)


def parse_file_recursive(filename: pathlib.Path | str) -> Files:
    files = Files(main=filename)
    files.parse()
    return files


@dataclass
class Files:
    main: pathlib.Path
    stack: list[pathlib.Path] = field(default_factory=list)
    by_filename: dict[pathlib.Path, list[Statement]] = field(default_factory=dict)
    local_file_to_source_filename: dict[pathlib.Path, str] = field(default_factory=dict)

    def parse(self):
        if not self.stack:
            self.stack = [self.main]

        parent_fn = self.main

        total_lines = 0
        while self.stack:
            filename = self.stack.pop()
            num_lines = len(filename.read_text().splitlines())
            total_lines += num_lines
            print(f"Parsing {filename} ({num_lines} / {total_lines} total lines)", file=sys.stderr)

            statements = list(parse_file(parent_fn))
            self.by_filename[parent_fn] = statements
            for st in statements:
                if isinstance(st, Simple) and st.statement == "call":
                    attr = st.get_named_attribute("filename", partial_match=True)

                    sub_filename = (
                        attr.value if isinstance(attr.value, Token) else attr.value.to_token()
                    ).remove_quotes()
                    fn = parent_fn.parent / os.path.expandvars(sub_filename)
                    self.local_file_to_source_filename[sub_filename] = fn
                    self.stack.append(fn)
            parent_fn = pathlib.Path(filename)
        return self.by_filename

    def reformat(self, options: FormatOptions) -> None:
        from .output import format_statements

        for fn, statements in self.by_filename.items():
            formatted = format_statements(statements, options)
            fn.write_text(formatted)
