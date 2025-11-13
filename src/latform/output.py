from __future__ import annotations

import pathlib
from typing import Sequence

from .const import (
    CLOSE_TO_OPEN,
    COMMA,
    EQUALS,
    LBRACE,
    LBRACK,
    LPAREN,
    OPEN_TO_CLOSE,
    SPACE,
    STATEMENT_NAME_COLON,
    STATEMENT_NAME_EQUALS,
)
from .statements import Statement
from .token import Comments, Token
from .types import (
    Attribute,
    CallName,
    Delimiter,
    FormatOptions,
    OutputLine,
    Seq,
)
from .util import DelimiterState, flatten

OutputNodeType = Delimiter | Attribute | Token | Seq | CallName


def _has_comments(node: OutputNodeType) -> bool:
    comments: Comments | None = getattr(node, "comments", None)
    if comments:
        return bool(comments.pre) or bool(comments.inline)
    return False


def _requires_multiline(node: OutputNodeType) -> bool:
    if _has_comments(node):
        return True
    if isinstance(node, Seq):
        return any(_requires_multiline(child) for child in node.items)
    return False


NO_SPACE_AFTER = frozenset("({[:")
NO_SPACE_BEFORE = frozenset(")}],:;")
SPACE_AFTER = frozenset(",:;")
CLOSING = frozenset(")}]")


def _needs_space_before(prev: Token | None, cur: Token, next_: Token | None) -> bool:
    if not prev:
        return False

    # TODO: think of a better way for denoting context-specific information
    # when creating the output nodes? `Token.purpose` or similar, perhaps...
    #
    # ":" serves two purposes:
    # 1. Delimiter between the element name and its definition;
    # 2. Delimiter between ranges in expressions.
    # We use a sentinel value of sorts here to differentiate.
    # This sentinel also passes COLON == STATEMENT_NAME_COLON.
    if cur is STATEMENT_NAME_COLON:
        return False
    elif prev is STATEMENT_NAME_COLON:
        return True

    # "=" is similar. It can be either:
    # 1. Delimiter between the element key and its attributes;
    # 2. Delimiter between an Attribute name and its value.
    # We use a sentinel value of sorts here to differentiate.
    if cur is STATEMENT_NAME_EQUALS or prev is STATEMENT_NAME_EQUALS:
        return True

    if cur.startswith("%"):
        # Found in foo()%bar parameter names
        return False

    match (prev, cur, next_):
        # No space around = with opening brackets
        case (_, "=", "(") | (_, "=", "{") | (_, "=", "["):
            return False
        case ("=", "(", _) | ("=", "{", _) | ("=", "[", _):
            return False

        # No space after opening brackets (except before =)
        case (p, c, _) if p in NO_SPACE_AFTER and c != "=":
            return False

        # No space before closing brackets, commas, colons, semicolons
        case (_, c, _) if c in NO_SPACE_BEFORE:
            return False

        # Space after commas, colons, semicolons
        case (p, _, _) if p in SPACE_AFTER:
            return True

        # Space before = when next is opening bracket
        case (_, "=", n) if n in set("({["):
            return True

        # Space around = in other cases
        case (p, _, _) if p == "=":
            return True
        case (_, "=", _):
            return True

        # Space after closing brackets
        case (p, _, _) if p in CLOSING:
            return True

        # Space between alphanumeric tokens
        case (p, c, _) if p and c and p[-1].isalnum() and c[0].isalnum():
            return True

    if cur and cur.is_quoted_string:
        return prev not in NO_SPACE_AFTER

    return False


def _output_node_block_contains_comments(parts: list[Token], start_idx: int) -> bool:
    """Scan ahead to check if block contains any comments (including nested blocks)."""

    depth = 0
    for token in parts[start_idx:]:
        if token.comments:
            return True

        if token in OPEN_TO_CLOSE:
            depth += 1
        elif token in CLOSE_TO_OPEN:
            depth -= 1
            if depth == 0:
                return False

    return False


def _flatten_output_nodes(nodes: list[OutputNodeType] | Statement | OutputNodeType) -> list[Token]:
    if isinstance(nodes, Statement):
        nodelist = nodes.to_output_nodes()
    elif not isinstance(nodes, list):
        nodelist = [nodes]
    else:
        nodelist = nodes

    parts = []
    for node in nodelist:
        if isinstance(node, (Delimiter, Token)):
            parts.append(node)
        else:
            parts.extend(flatten(node))

    assert all(isinstance(part, Token) for part in parts)
    return parts


def _should_break_for_length(
    line: OutputLine, parts: list[Token], start_idx: int, max_length: int
) -> bool:
    """
    Check if continuing would exceed max_length.
    Only applies outside blocks. Looks ahead until next breakpoint: ,({[=
    """
    test_length = len(line)
    breakpoints = {COMMA, LPAREN, LBRACE, LBRACK, EQUALS}
    test_prev = None

    for token in parts[start_idx:]:
        if test_prev and _needs_space_before(test_prev, token, None):
            test_length += 1

        test_length += len(token)

        if token in breakpoints:
            if test_length > max_length:
                return True
            return False

        if test_length > max_length:
            return True

        test_prev = token

    return False


def _format(
    parts: list[Token],
    options: FormatOptions,
    *,
    indent_level: int = 0,
    outer_comments: Comments | None = None,
) -> list[OutputLine]:
    lines: list[OutputLine] = []
    prev: Token | None = None
    idx = 0
    delim_state = DelimiterState()
    block_has_newlines_stack: list[bool] = []
    next_ = None

    line = OutputLine(indent=indent_level, parts=[])

    def newline(lookahead: bool = True):
        nonlocal idx
        nonlocal line

        if lookahead and next_ in {COMMA}:
            assert isinstance(next_, Delimiter)
            line.parts.append(next_)
            idx += 1

        if line is not None and (line.parts or line.comment):
            lines.append(line)

        return OutputLine(indent=indent_level, parts=[])

    while idx < len(parts):
        cur = parts[idx]
        prev = parts[idx - 1] if idx > 0 else None
        next_ = parts[idx + 1] if idx < len(parts) - 1 else None

        if cur.comments.pre:
            for pre_comment in cur.comments.pre:
                lines.append(
                    OutputLine(indent=indent_level, parts=[f"!{pre_comment}"], comment=None)
                )

        is_opening = False
        is_closing = False
        if isinstance(cur, Delimiter):
            _, level_change = delim_state.update(cur)

            if level_change < 0:
                is_closing = True
            elif level_change > 0:
                is_opening = True

        has_comments = is_opening and _output_node_block_contains_comments(parts, idx)

        if line.parts and not cur.comments.pre:
            if _needs_space_before(prev, cur, next_):
                line.parts.append(" ")

        if is_closing:
            had_newlines = block_has_newlines_stack.pop() if block_has_newlines_stack else False
            if had_newlines:
                # don't use the lookahead logic since we haven't yet added the closing delimiter
                indent_level -= 1
                # if multiline_list_final_comma:
                #  todo this doesn't quite work; and can we assume comma-delimited anyway?
                #     line.parts.append(COMMA)
                line = newline(lookahead=False)
                line.parts.append(cur)
                if next_ in {COMMA}:
                    assert isinstance(next_, Delimiter)
                    line.parts.append(next_)
                    idx += 1
                if block_has_newlines_stack and block_has_newlines_stack[-1]:
                    next_ = None
                    line = newline()
            else:
                line.parts.append(cur)
            idx += 1
            continue

        line.parts.append(cur)

        if is_opening:
            block_has_newlines_stack.append(has_comments)
            if has_comments:
                indent_level += 1
                line = newline()

        if cur.comments.inline:
            line.comment = f"!{cur.comments.inline}"

            if delim_state.depth == 0 and next_ not in {EQUALS, COMMA, None}:
                # No implicit continuation char?
                line.parts.append(SPACE)
                line.parts.append(Delimiter("&"))
                line = newline()
            else:
                line = newline()

            idx += 1
            continue

        # if delim_state.depth == 0 and not is_opening and not is_closing:
        if cur in {COMMA, LPAREN, LBRACE, EQUALS}:
            if _should_break_for_length(
                parts=parts, line=line, max_length=options.line_length, start_idx=idx + 1
            ):
                if indent_level == 0:
                    indent_level += 1
                line = newline()

        if any(block_has_newlines_stack) and cur in {COMMA}:
            line = newline()

        idx += 1

    line = newline()

    if outer_comments:
        if not lines:
            return [
                OutputLine(indent=indent_level, parts=[f"!{comment}"])
                for comment in outer_comments.pre
            ]

        if outer_comments.inline:
            if lines[0].comment:
                existing = lines[0].comment[1:]  # strip off the !
                lines[0].comment = f"!{outer_comments.inline} / {existing}"
            else:
                lines[0].comment = f"!{outer_comments.inline}"

        if outer_comments.pre:
            for comment in reversed(outer_comments.pre):
                lines.insert(0, OutputLine(indent=indent_level, parts=[f"!{comment}"]))

    return lines


default_options = FormatOptions()


def format_nodes(
    nodes: list[OutputNodeType] | Statement,
    options: FormatOptions = default_options,
) -> list[OutputLine]:
    parts = _flatten_output_nodes(nodes)
    if isinstance(nodes, Statement):
        outer_comments = nodes.comments
    else:
        outer_comments = None
    return _format(parts, options, outer_comments=outer_comments)


def format_statement(statement: Statement, options: FormatOptions) -> str:
    """Format a statement and return the code string"""
    lines = format_nodes(statement, options=options)
    return "\n".join(line.render(options) for line in lines)


def format_statements(statements: Sequence[Statement], options: FormatOptions) -> str:
    """Format a statement and return the code string"""
    res = []

    last_statement = None
    for statement in statements:
        if options.newline_before_new_type:
            if last_statement is not None and not isinstance(statement, type(last_statement)):
                res.append(OutputLine())
        res.extend(format_nodes(statement, options=options))

        last_statement = statement

    return "\n".join(line.render(options) for line in res)


def format_file(filename: pathlib.Path | str, options: FormatOptions) -> str:
    from .parser import parse_file

    statements = parse_file(filename)
    return format_statements(statements, options=options)
