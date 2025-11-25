from __future__ import annotations

import logging
import os
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
    NameCase,
    OutputLine,
    Seq,
)
from .util import DelimiterState, flatten

OutputNodeType = Delimiter | Attribute | Token | Seq | CallName

logger = logging.getLogger(__name__)


open_brackets = frozenset("({[")
close_brackets = frozenset(")]}")
no_space_after = open_brackets | frozenset(":")

LATFORM_OUTPUT_DEBUG = os.environ.get("LATFORM_OUTPUT_DEBUG", "") == "1"


def _needs_space_before(parts: list[Token], idx: int) -> tuple[bool, str]:
    cur = parts[idx]
    prev = parts[idx - 1] if idx > 0 else None
    nxt = parts[idx + 1] if idx < len(parts) - 1 else None

    if not prev:
        return False, "no previous token"

    if cur == ":" and cur.role == Role.statement_definition:
        return False, "colon in statement definition"
    if prev == ":" and prev.role == Role.statement_definition:
        return True, "after colon in statement definition"

    if cur == "=" and cur.role == Role.statement_definition:
        return True, "before equals in statement definition"
    if prev == "=" and prev.role == Role.statement_definition:
        return True, "after equals in statement definition"

    if cur.startswith("%"):
        # Found in foo()%bar parameter names
        return False, "token starts with % (parameter name)"

    # No space around = with opening brackets
    if (prev == "=" and cur in open_brackets) or (cur == "=" and nxt in open_brackets):
        return False, "no space around = with opening brackets"

    # No space after opening brackets (except before =)
    if prev in no_space_after and cur != "=":
        return False, f"no space after opening bracket '{prev}'"

    # No space before closing brackets, commas, colons, semicolons
    if cur in frozenset(")}],:;"):
        return False, f"no space before '{cur}'"

    # Space after commas, colons, semicolons
    if prev in frozenset(",:;"):
        return True, f"space after '{prev}'"

    # Space before = when next is opening bracket
    if cur == "=" and nxt in open_brackets:
        return True, "space before = when next is opening bracket"

    # Space around = in other cases
    if prev == "=":
        return True, "space after ="
    if cur == "=":
        return True, "space before ="

    # Separate addition/subtraction from rest of expressions
    if prev == "+" or cur == "+":
        return True, "space around +"
    if prev == "-":
        prev_prev = parts[idx - 2] if idx >= 2 else None
        if prev_prev in open_brackets or prev_prev in {"=", ":"}:
            return False, "no space when minus looks like unary negation"
        if prev_prev in {"-"}:
            return False, "double minus sign means second is negation"
        return True, "space after minus"
    if prev == "/" or cur == "/":
        return False, "no space around /"
    if prev == "*" or cur == "*":
        return False, "no space around *"
    if cur == "^":
        return False, "no space around caret"
    if cur == "-":
        if prev in close_brackets:
            return True, "space before minus:- after closing bracket"
        if prev in open_brackets or prev in {"=", ":"}:
            return False, "no space before minus after opening bracket or =/:"
        return True, "space before minus (default case)"

    # Space after closing brackets
    if prev in close_brackets:
        return True, f"space after closing bracket '{prev}'"

    # Space between alphanumeric tokens
    if prev and cur and prev[-1].isalnum() and cur[0].isalnum():
        return True, "space between alphanumeric tokens"
    if cur and cur.is_quoted_string:
        if prev in no_space_after:
            return False, f"quoted string, prev '{prev}' in {no_space_after}"
        return True, f"quoted string, prev '{prev}' not in {no_space_after}"

    return False, "default: no space"


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


def _tokens_at_depth(parts: list[Token], target_depth: int):
    depth = 0

    for token in parts:
        if depth == target_depth:
            yield token

        if token in OPEN_TO_CLOSE:
            depth += 1
        elif token in CLOSE_TO_OPEN:
            depth -= 1


def _count_top_level(parts: list[Token], to_count: Token) -> int:
    count = 0

    for tok in _tokens_at_depth(parts, target_depth=0):
        if tok == to_count:
            count += 1
    return count


def _length_top_level(parts: list[Token]) -> int:
    length = 0

    for tok in _tokens_at_depth(parts, target_depth=0):
        length += len(tok)
    return length


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

    for idx, token in enumerate(parts[start_idx:end_idx], start=start_idx):
        spc, _ = _needs_space_before(parts, idx)
        if prev and spc:
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
    nxt = None

    commas = _count_top_level(parts, COMMA)
    top_level_length = _length_top_level(parts)

    if commas > options.statement_comma_threshold_for_multiline or top_level_length > (
        options.line_length * options.always_multiline_factor
    ):
        block_has_newlines_stack.append(True)
    else:
        block_has_newlines_stack.append(False)
    top_level_multiline = block_has_newlines_stack[0]

    if LATFORM_OUTPUT_DEBUG:
        logger.debug(
            f"{top_level_multiline=}:"
            f"\n* {commas=} vs {options.statement_comma_threshold_for_multiline=}, "
            f"\n* {top_level_length=} vs {options.line_length=} * {options.always_multiline_factor=}"
        )

    line = OutputLine(indent=indent_level, parts=[])

    def add_part_to_line(part: Token):
        def apply_case(case: NameCase):
            if case == "upper" or part == "l":  # special case
                val = part.upper()
            elif case == "lower":
                val = part.lower()
            else:
                val = part

            line.parts.append(val)

        if part.role in {Role.attribute_name}:
            return apply_case(options.attribute_case)
        if part.role in {Role.name_, Role.kind, Role.builtin, Role.attribute_name}:
            return apply_case(options.name_case)

        line.parts.append(part)

    def newline(lookahead: bool = True, reason: str = ""):
        nonlocal idx
        nonlocal indent_level
        nonlocal line

        if lookahead and nxt is not None and nxt in {COMMA}:
            idx += 1

            if should_include_comma():
                add_part_to_line(nxt)

        if line is not None and (line.parts or line.comment):
            lines.append(line)

        if indent_level == top_level_indent and top_level_multiline and len(lines) == 1:
            indent_level += 1

        if reason and LATFORM_OUTPUT_DEBUG:
            logger.debug(f"{idx}: {prev}, {cur}, {nxt}: break {reason}")

        return OutputLine(indent=indent_level, parts=[])

    def in_newline_block() -> bool:
        return bool(block_has_newlines_stack and block_has_newlines_stack[-1])

    def should_include_comma():
        cur = parts[idx]

        assert cur == COMMA

        nxt = parts[idx + 1] if idx < len(parts) - 1 else None

        if nxt in close_brackets:
            if not options.trailing_comma:
                return False
            if not in_newline_block():
                return False
        if nxt is None:
            return False
        return True

    while idx < len(parts):
        cur = parts[idx]
        prev = parts[idx - 1] if idx > 0 else None
        nxt = parts[idx + 1] if idx < len(parts) - 1 else None

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
            spc, reason = _needs_space_before(parts, idx)
            if spc:
                add_part_to_line(SPACE)

            if LATFORM_OUTPUT_DEBUG:
                if spc:
                    logger.debug("Adding space before %r: %s", cur, reason)
                else:
                    logger.debug("No space before %r: %s", cur, reason)

        if is_closing:
            had_newlines = block_has_newlines_stack.pop() if block_has_newlines_stack else False
            if had_newlines:
                # don't use the lookahead logic since we haven't yet added the closing delimiter
                indent_level -= 1

                line = newline(lookahead=False, reason="closing multiline block")
                add_part_to_line(cur)
                if nxt in {COMMA}:
                    assert isinstance(nxt, Delimiter)
                    idx += 1
                    if should_include_comma():
                        add_part_to_line(nxt)
                if block_has_newlines_stack and block_has_newlines_stack[-1]:
                    nxt = None
                    line = newline(reason="newline stack post multiline close")
            else:
                # single line block
                if line.parts and line.parts[-1] == COMMA:
                    # Never a trailing comma when full sequence is on a single line
                    line.parts.pop()
                add_part_to_line(cur)
            idx += 1
            continue

        if cur == COMMA:
            if nxt in close_brackets:
                if not options.trailing_comma:
                    idx += 1
                    continue
                elif not in_newline_block():
                    idx += 1
                    continue
            if nxt is None:
                idx += 1
                continue

        add_part_to_line(cur)

        if is_opening:
            block_has_newlines_stack.append(would_break_inside)
            if would_break_inside:
                indent_level += 1
                line = newline(reason="opening + would break inside")

        if cur.comments.inline:
            line.comment = f"!{cur.comments.inline}"

            if delim_state.depth == 0 and nxt not in {EQUALS, COMMA, None}:
                # No implicit continuation char?
                add_part_to_line(SPACE)
                add_part_to_line(Delimiter("&"))
                line = newline(reason="inline comment without implicit continuation")
            else:
                line = newline(reason="inline comment")

            idx += 1
            continue

        # if delim_state.depth == 0 and not is_opening and not is_closing:
        if cur in {COMMA, LPAREN, LBRACE}:
            if _should_break_for_length(
                parts=parts,
                start_length=len(line),
                max_length=options.line_length,
                start_idx=idx + 1,
            ):
                if indent_level == 0:
                    indent_level += 1
                line = newline(reason="break for length ',({'")

        if in_newline_block() and cur in {COMMA}:
            line = newline(reason="multiline block post comma")

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


def format_statements(statements: Sequence[Statement] | Statement, options: FormatOptions) -> str:
    """Format a statement and return the code string"""
    if isinstance(statements, Statement):
        statements = [statements]

    res: list[OutputLine] = []

    last_statement = None
    for statement in statements:
        if options.newline_before_new_type:
            if last_statement is not None and not isinstance(statement, type(last_statement)):
                res.append(OutputLine())
        res.extend(format_nodes(statement, options=options))

        last_statement = statement

    if options.renames:
        lower_renames = {from_.lower(): to for from_, to in options.renames.items()}

        def apply_rename(item: Token | str):
            if not isinstance(item, Token):
                return item

            if item.lower() in lower_renames:
                return lower_renames[item.lower()]

            # if item.role == Role.name_:
            # % single char, * 0+
            if "%" in item or "*" in item:
                print("Saw match", item, item.role)
            else:
                print("saw", repr(item))

            return item

        for line in res:
            line.parts = [apply_rename(part) for part in line.parts]

    return "\n".join(line.render(options) for line in res)


def format_file(filename: pathlib.Path | str, options: FormatOptions) -> str:
    from .parser import parse_file

    statements = parse_file(filename)
    return format_statements(statements, options=options)
