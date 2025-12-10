from __future__ import annotations

import logging
import os
import pathlib
import re
from typing import Sequence

from .attrs import element_key_to_attrs
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
from .statements import Element, Simple, Statement
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


RE_ES25_17E3 = re.compile(r"^[+-]?\d\.\d{17}E[+-]\d{3}$")


def _is_es25_17e3(tok: Token) -> bool:
    """Check if a token matches the Fortran format ES25.17E3."""

    # Example: 4.24223267191081721E+000
    # *Does not* include leading +-, as that's a separate token.
    #
    # Regex breakdown:
    # ^            Start of string
    # \d           Exactly one digit before the dot
    # \.           Decimal point
    # \d{17}       Exactly 17 digits after the dot
    # E            Exponent indicator (capital E)
    # [+-]         Sign of the exponent (required by format)
    # \d{3}        Exactly 3 digits for the exponent
    # $            End of string
    return bool(RE_ES25_17E3.match(tok))


def _num_spaces_before(parts: list[Token], idx: int) -> tuple[int, str]:
    cur = parts[idx]
    prev = parts[idx - 1] if idx > 0 else None
    nxt = parts[idx + 1] if idx < len(parts) - 1 else None

    if not prev:
        return 0, "no previous token"

    if cur == ":" and cur.role == Role.statement_definition:
        return 0, "colon in statement definition"
    if prev == ":" and prev.role == Role.statement_definition:
        return 1, "after colon in statement definition"

    if cur == "=" and cur.role == Role.statement_definition:
        return 1, "before equals in statement definition"
    if prev == "=" and prev.role == Role.statement_definition:
        if not cur.startswith(("+", "-")) and _is_es25_17e3(cur):
            return 2, "ES25.17E3 alignment after ="
        return 1, "after equals in statement definition"

    if cur.startswith("%"):
        # Found in foo()%bar parameter names
        return 0, "token starts with % (parameter name)"

    if prev == "=" or cur == "=":
        return 0, "no space around ="

    # No space after opening brackets (except before =)
    if prev in no_space_after:
        return 0, f"no space after opening bracket '{prev}'"

    # No space before closing brackets, commas, colons, semicolons
    if cur in frozenset(")}],:;"):
        return 0, f"no space before '{cur}'"

    # Space after commas, colons, semicolons
    if prev in frozenset(",:;"):
        return 1, f"space after '{prev}'"

    # Space before = when next is opening bracket
    if cur == "=" and nxt in open_brackets:
        return 1, "space before = when next is opening bracket"

    # Separate addition/subtraction from rest of expressions
    if prev in {"-", "+"}:
        prev_prev = parts[idx - 2] if idx >= 2 else None
        if prev_prev in open_brackets or prev_prev in {"=", ":"}:
            return 0, "no space when minus looks like unary negation"
        if prev_prev in {"-"}:
            return 0, "double minus sign means second is negation"
        return 1, "space after minus"
    if prev == "/" or cur == "/":
        return 0, "no space around /"
    if prev == "*" or cur == "*":
        return 0, "no space around *"
    if cur == "^":
        return 0, "no space around caret"
    if cur in {"-", "+"}:
        if prev in close_brackets:
            return 1, "space before minus:- after closing bracket"
        if prev in open_brackets or prev in {"=", ":"}:
            return 0, "no space before minus after opening bracket or =/:"
        return 1, "space before minus (default case)"

    # Space after closing brackets
    if prev in close_brackets:
        return 1, f"space after closing bracket '{prev}'"

    # Space between alphanumeric tokens
    if prev and cur and prev[-1].isalnum() and cur[0].isalnum():
        return 1, "space between alphanumeric tokens"
    if cur and cur.is_quoted_string:
        if prev in no_space_after:
            return 0, f"quoted string, prev '{prev}' in {no_space_after}"
        return 1, f"quoted string, prev '{prev}' not in {no_space_after}"

    return 0, "default: no space"


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
        spc, _ = _num_spaces_before(parts, idx)
        if prev and spc:
            test_length += spc

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


def _is_user_defined_name(names: dict[Token, Element], name: str) -> bool:
    # In its current form, this could be reduced to `name.upper() in names`
    # but leaving the logic as a placeholder for futher customization for now
    #
    # `names` is assumed to be upper-cased already
    name = name.upper()

    # Inheritance requires *full* names for a match
    if name in names:
        if name in element_key_to_attrs:
            logger.warning("Element inheritance using full element name? %r", name)
            return False

        return True

    if name in element_key_to_attrs:
        # It's the full name of an element kind
        return False
    # for key in element_key_to_attrs:
    #     if key.startswith(name):
    #         return False

    return False


def looks_like_section_break(comment: Token, empty_line_is_break: bool = False):
    contents = comment.removeprefix("!").strip()
    # !
    if not contents:
        return empty_line_is_break

    # 3 or more characters in a row all of the same type make for a section break:
    # !******
    # !------
    # !______
    # !======
    # !******
    # !######
    # !!!!!!!

    chars = set(contents)
    if len(chars) > 1 or len(contents) < 3:
        return False

    char = list(chars)[0]
    return char in {"*", "-", "_", "=", "#", "!"}


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

        if part.role in {Role.filename}:
            # Always keep the same case
            line.parts.append(part)
            return

        same: NameCase = "same"
        case = {
            Role.attribute_name: options.attribute_case,
            Role.name_: options.name_case,
            Role.kind: options.kind_case,
            Role.builtin: options.builtin_case,
            None: same,
        }.get(part.role, same)

        apply_case(case)

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

    section_break_width = options.section_break_width or options.line_length
    section_break = "!" + options.section_break_character * section_break_width

    def pre_comment_rewrite_section_break(pre: list[Token], indent_level: int) -> list[OutputLine]:
        if not pre:
            return []

        res = []
        if lines and lines[-1].parts:
            if looks_like_section_break(pre[0]):
                res.append(OutputLine(parts=[], comment=None))

        for idx, comment in enumerate(pre):
            if looks_like_section_break(comment, empty_line_is_break=idx == 0):
                res.append(OutputLine(indent=indent_level, parts=[section_break], comment=None))
            else:
                res.append(OutputLine(indent=indent_level, parts=[f"!{comment}"], comment=None))
        return res

    while idx < len(parts):
        cur = parts[idx]
        prev = parts[idx - 1] if idx > 0 else None
        nxt = parts[idx + 1] if idx < len(parts) - 1 else None

        if cur.comments.pre:
            pre_comments = list(cur.comments.pre)
            lines.extend(pre_comment_rewrite_section_break(pre_comments, indent_level=indent_level))

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
            spc, reason = _num_spaces_before(parts, idx)
            for _ in range(spc):
                add_part_to_line(SPACE)

            if LATFORM_OUTPUT_DEBUG:
                if spc:
                    logger.debug("Adding %d spaces before %r: %s", spc, cur, reason)
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
            for line in reversed(
                pre_comment_rewrite_section_break(outer_comments.pre, indent_level=top_level_indent)
            ):
                lines.insert(0, line)

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


def format_statements(
    statements: Sequence[Statement] | Statement,
    options: FormatOptions,
) -> str:
    """Format a statement and return the code string"""
    if isinstance(statements, Statement):
        statements = [statements]

    res: list[OutputLine] = []

    def maybe_add_blank_line():
        if res and not res[-1].parts:
            return
        res.append(OutputLine())

    last_statement = None
    for statement in statements:
        if options.newline_before_new_type:
            if last_statement is not None:
                if not isinstance(statement, type(last_statement)):
                    maybe_add_blank_line()
                elif (
                    isinstance(statement, Simple)
                    and statement.statement != last_statement.statement
                ):
                    maybe_add_blank_line()

        for line in format_nodes(statement, options=options):
            if not line.parts and not line.comment:
                maybe_add_blank_line()
            else:
                res.append(line)

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
            # if "%" in item or "*" in item:
            #     print("Saw match", item, item.role)
            # else:
            #     print("saw", repr(item))

            return item

        for line in res:
            line.parts = [apply_rename(part) for part in line.parts]

    while res and not res[0].parts and not res[0].comment:
        res = res[1:]

    return "\n".join(line.render(options) for line in res)


def format_file(filename: pathlib.Path | str, options: FormatOptions) -> str:
    from .parser import parse_file

    statements = parse_file(filename)
    return format_statements(statements, options=options)
