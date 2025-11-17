from __future__ import annotations

import logging
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
)
from .statements import Statement
from .token import Comments, Role, Token
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

logger = logging.getLogger(__name__)


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


open_brackets = frozenset("({[")
close_brackets = frozenset(")]}")
no_space_after = open_brackets | frozenset(":")


def _needs_space_before(prev: Token | None, cur: Token, next_: Token | None) -> bool:
    if not prev:
        return False

    if cur == ":" and cur.role == Role.statement_definition:
        return False
    if prev == ":" and prev.role == Role.statement_definition:
        return True

    if cur == "=" and cur.role == Role.statement_definition:
        return True
    if prev == "=" and prev.role == Role.statement_definition:
        return True

    if cur.startswith("%"):
        # Found in foo()%bar parameter names
        return False

    # No space around = with opening brackets
    if (prev == "=" and cur in open_brackets) or (cur == "=" and next_ in open_brackets):
        return False

    # No space after opening brackets (except before =)
    if prev in no_space_after and cur != "=":
        return False

    # No space before closing brackets, commas, colons, semicolons
    if cur in frozenset(")}],:;"):
        return False

    # Space after commas, colons, semicolons
    if prev in frozenset(",:;"):
        return True

    # Space before = when next is opening bracket
    if cur == "=" and next_ in open_brackets:
        return True

    # Space around = in other cases
    if prev == "=":
        return True
    if cur == "=":
        return True

    # Space after closing brackets
    if prev in close_brackets:
        return True

    if prev == "/" or cur == "/":
        return True

    # Space between alphanumeric tokens
    if prev and cur and prev[-1].isalnum() and cur[0].isalnum():
        return True
    if cur and cur.is_quoted_string:
        return prev not in no_space_after

    return False


def _get_output_block(parts: list[Token], start_idx: int) -> list[Token]:
    """Scan ahead to check if block contains any comments (including nested blocks)."""

    block = []
    depth = 0
    for token in parts[start_idx:]:
        block.append(token)
        if token in OPEN_TO_CLOSE:
            depth += 1
        elif token in CLOSE_TO_OPEN:
            depth -= 1
            if depth == 0:
                break

    return block


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


def _output_range_would_break(
    start_length: int,
    parts: list[Token],
    start_idx: int,
    end_idx: int,
    max_length: int,
) -> bool:
    test_length = start_length
    prev = parts[start_idx - 1] if start_idx > 0 else None

    for token in parts[start_idx:end_idx]:
        if prev and _needs_space_before(prev, token, None):
            test_length += 1

        test_length += len(token)
        if test_length > max_length:
            return True

        prev = token

    return False


def _output_block_would_break(
    start_length: int,
    parts: list[Token],
    start_idx: int,
    max_length: int,
) -> bool:
    block = _get_output_block(parts, start_idx)
    return _output_range_would_break(
        start_length=start_length,
        parts=parts,
        start_idx=start_idx,
        end_idx=start_idx + len(block),
        max_length=max_length,
    )


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
    start_length: int, parts: list[Token], start_idx: int, max_length: int
) -> bool:
    """
    Check if continuing would exceed max_length.
    Only applies outside blocks. Looks ahead until next breakpoint: ,({[=
    """
    breakpoints = {COMMA, LPAREN, LBRACE, LBRACK, EQUALS}
    end_idx = len(parts)
    for idx, ch in enumerate(parts[start_idx:], start=start_idx):
        if ch in breakpoints:
            end_idx = idx
            break

    return _output_range_would_break(
        start_length=start_length,
        parts=parts,
        start_idx=start_idx,
        end_idx=end_idx,
        max_length=max_length,
    )


def _format(
    parts: list[Token],
    options: FormatOptions,
    *,
    indent_level: int = 0,
    outer_comments: Comments | None = None,
) -> list[OutputLine]:
    top_level_indent = indent_level
    lines: list[OutputLine] = []
    prev: Token | None = None
    idx = 0
    delim_state = DelimiterState()
    block_has_newlines_stack: list[bool] = []
    next_ = None

    line = OutputLine(indent=indent_level, parts=[])

    def newline(lookahead: bool = True, reason: str = ""):
        nonlocal idx
        nonlocal line

        if lookahead and next_ in {COMMA}:
            assert isinstance(next_, Delimiter)
            line.parts.append(next_)
            idx += 1

        if line is not None and (line.parts or line.comment):
            lines.append(line)

        if reason:
            logger.debug(f"{idx}: {prev}, {cur}, {next_}: break {reason}")

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
        would_break_inside = has_comments or (
            is_opening
            and _output_block_would_break(
                parts=parts, start_idx=idx, start_length=len(line), max_length=options.line_length
            )
        )

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
                line = newline(lookahead=False, reason="closing multiline block")
                line.parts.append(cur)
                if next_ in {COMMA}:
                    assert isinstance(next_, Delimiter)
                    line.parts.append(next_)
                    idx += 1
                if block_has_newlines_stack and block_has_newlines_stack[-1]:
                    next_ = None
                    line = newline(reason="newline stack post multiline close")
            else:
                line.parts.append(cur)
            idx += 1
            continue

        line.parts.append(cur)

        if is_opening:
            block_has_newlines_stack.append(would_break_inside)
            if would_break_inside:
                indent_level += 1
                line = newline(reason="opening + would break inside")

        if cur.comments.inline:
            line.comment = f"!{cur.comments.inline}"

            if delim_state.depth == 0 and next_ not in {EQUALS, COMMA, None}:
                # No implicit continuation char?
                line.parts.append(SPACE)
                line.parts.append(Delimiter("&"))
                line = newline(reason="inline comment without implicit continuation")
            else:
                line = newline(reason="inline comment")

            idx += 1
            continue

        # if delim_state.depth == 0 and not is_opening and not is_closing:
        if cur in {COMMA, LPAREN, LBRACE}:  # , EQUALS}:
            if _should_break_for_length(
                parts=parts,
                start_length=len(line),
                max_length=options.line_length,
                start_idx=idx + 1,
            ):
                if indent_level == 0:
                    indent_level += 1
                line = newline(reason="break for length ',({'")

        if block_has_newlines_stack and block_has_newlines_stack[-1] and cur in {COMMA}:
            line = newline(reason="newline stack comma")

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
                lines.insert(0, OutputLine(indent=top_level_indent, parts=[f"!{comment}"]))

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
