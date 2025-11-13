from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from .const import DELIMITERS, LBRACE, LBRACK, LPAREN
from .exceptions import EndOfLine, MissingCloseDelimiter
from .token import Comments, Delimiter, Location, Token
from .types import Block
from .util import DelimiterState


@dataclass
class _StatementTokenBlock:
    """
    A raw block from the file that, when parsed, should correspond to a single statement.
    """

    lineno: int
    items: list[Token | Delimiter] = field(default_factory=list)

    def stack(self):
        root = Block()
        stack = [root]
        delim_state = DelimiterState()
        for tok in self.items:
            if isinstance(tok, Delimiter):
                _, level_change = delim_state.update(tok)
                if level_change > 0:
                    new_block = Block(opener=tok)
                    stack[-1].items.append(new_block)
                    stack.append(new_block)
                elif level_change < 0:
                    last_block = stack.pop(-1)
                    last_block.closer = tok
                else:
                    stack[-1].items.append(tok)

            else:
                stack[-1].items.append(tok)

        return root


def split_comment(
    line: str, filename: pathlib.Path | str, lineno: int = 0
) -> tuple[str, Token | None]:
    """
    Split code and comment at first '!' that is not inside quotes.
    """
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "!" and not in_single and not in_double:
            comment = line[idx + 1 :].rstrip()
            loc = Location(
                filename=pathlib.Path(filename),
                line=lineno,
                column=idx,
                end_line=lineno,
                end_column=idx + 1 + len(comment),
            )
            return line[:idx].rstrip(), Token(comment, loc=loc)
    return line.rstrip(), None


def _attach_inline_comment(
    first: Token | Block | None,
    last: Token | Block | None,
    inline: Token | None,
    pending: list[Token],
    is_multiline: bool,
) -> None:
    if first is None:
        if inline:
            pending.append(inline)
        return

    first.comments.pre.extend(pending)
    pending.clear()
    if inline is None:
        return
    if last is None:
        first.comments.inline = inline
        return

    if is_multiline or first.loc.line != last.loc.end_line:
        # On different lines; associate with the last token
        last.comments.inline = inline
    else:
        # On same line; associate with the first token
        first.comments.inline = inline


def _get_first_token_or_block(items) -> Token | Block | None:
    for item in items:
        if isinstance(item, (Token, Block)) and not isinstance(item, Delimiter):
            return item
    return None


_numeric_chars = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "."})


def _is_scientific_notation_start(tok: str | None) -> bool:
    if not tok:
        return False
    if not tok.lower().endswith("e"):
        return False
    val = tok[:-1]

    if _numeric_chars.issuperset(set(tok)):
        return False

    try:
        float(val)
    except ValueError:
        return False
    return True


class Tokenizer:
    filename: pathlib.Path
    line: str
    lineno: int
    pos: int

    def __init__(self, source: str, filename: pathlib.Path | str = "unknown") -> None:
        self.source = source
        self.filename = pathlib.Path(filename)

    def split_blocks(self):
        blocks: list[_StatementTokenBlock] = []
        last_delim = None
        delim_state = DelimiterState()
        pending_comments: list[Token] = []

        for self.lineno, line in enumerate(self.source.splitlines()):
            self.line, inline_comment = split_comment(
                line, lineno=self.lineno, filename=self.filename
            )
            self.pos = 0

            if not self.line.strip():
                if inline_comment is not None:
                    pending_comments.append(inline_comment)
                continue

            if last_delim == "&":
                block = blocks[-1]
                assert block.items[-1] == "&"
                block.items.pop(-1)
            elif last_delim in set("{[(,=") or delim_state.current_delimiters:
                block = blocks[-1]
            elif (
                # Dumb special case for multiline 'title' (see manual)
                blocks
                and blocks[-1].items
                and len(blocks[-1].items) == 1
                and isinstance(blocks[-1].items[0], Token)
                and blocks[-1].items[0].lower() == "title"
            ):
                block = blocks[-1]
            else:
                block = _StatementTokenBlock(lineno=self.lineno)
                blocks.append(block)

            line_start_pos = len(block.items)
            while True:
                try:
                    token, delim = self.get_next_word()
                except EndOfLine:
                    break

                if delim == ";":
                    # statement1; statement2
                    if token:
                        block.items.append(token)
                    last_delim = None
                    block = _StatementTokenBlock(lineno=self.lineno)
                    blocks.append(block)
                    continue

                # TODO : do these in a second pass
                elif last_delim in frozenset({"+", "-"}) and token and token[0].isnumeric():
                    prev_item = block.items[-2] if len(block.items) > 1 else None
                    if (
                        prev_item in {"+", "-", "*", "/", "=", ",", LBRACE, LBRACK, LPAREN}
                        or prev_item is None
                    ):
                        token = Token.join([block.items.pop(), token])
                        last_delim = prev_item
                    block.items.append(token)
                elif token:
                    block.items.append(token)

                if delim:
                    delim_state.update(delim)
                    block.items.append(delim)

                last_delim = delim

            if not block.items:
                continue

            is_multiline = block.items[0].loc.line != block.items[-1].loc.line
            first_item = block.items[line_start_pos]
            last_item = (
                _get_first_token_or_block(
                    reversed(block.items[block.items.index(first_item) + 1 :])
                )
                if first_item
                else None
            )
            _attach_inline_comment(
                first_item, last_item, inline_comment, pending_comments, is_multiline
            )

            if block.items[0].lower() in ("return", "end_file"):
                # TODO: need to include whatever comes after these lines verbatim
                break

        if pending_comments:
            loc = sum((comment.loc for comment in pending_comments), pending_comments[0].loc)
            blocks.append(
                _StatementTokenBlock(
                    lineno=pending_comments[0].loc.line,
                    items=[
                        Token(
                            "",
                            loc=loc,
                            comments=Comments(pre=pending_comments),
                        ),
                    ],
                )
            )

        in_delims = delim_state.current_delimiters
        if in_delims:
            unmatched = ", ".join(f"{delim!r} at {delim.loc}" for delim in in_delims)
            raise MissingCloseDelimiter(f"Unmatched delimiters: {unmatched}", delims=in_delims)

        # Exclude any empty statements (semicolons at EOL, for example)
        return [block for block in blocks if block.items]

    def _scan_word(self, delimiters: str) -> tuple[str, int]:
        start = self.pos
        in_quotes = False
        quote_char = None
        while self.pos < len(self.line):
            ch = self.line[self.pos]
            if ch in ("'", '"'):
                if not in_quotes:
                    in_quotes = True
                    quote_char = ch
                elif ch == quote_char:
                    in_quotes = False
                    quote_char = None
                self.pos += 1
                continue
            if not in_quotes and (ch in delimiters or ch.isspace()):
                break
            self.pos += 1

        word = self.line[start : self.pos]
        if (
            _is_scientific_notation_start(word)
            and self.pos < len(self.line)
            and self.line[self.pos] in {"-", "+"}
        ):
            plus_minus = self.line[self.pos]
            self.pos += 1
            end, end_pos = self._scan_word(delimiters)
            word = "".join((word, plus_minus, end))
            return word, end_pos

        return word, self.pos

    def _scan_delimiter(self, prev_token: Token | None, delimiters: str) -> Delimiter | None:
        while self.pos < len(self.line) and self.line[self.pos].isspace():
            self.pos += 1
        if self.pos >= len(self.line) or self.line[self.pos] not in delimiters:
            return None

        start = self.pos
        ch = self.line[self.pos]
        self.pos += 1

        # Handle compound delimiters and fake delimiters
        if self.pos < len(self.line):
            nxt = self.line[self.pos]

            match (ch, nxt):
                # Compound delimiters: :: and :=
                case (":", "=") | (":", ":"):
                    self.pos += 1
                    compound = ch + nxt
                    loc = Location(self.filename, self.lineno, start, self.lineno, self.pos)
                    if compound == ":=":
                        return Delimiter("=", loc)  # keep existing squashing
                    return Delimiter(compound, loc)

                    # Fake delimiters - part of species names
                    # Skip these entirely.
                    # case ("+", "+") | ("-", "-"):
                    #     while self.pos < len(self.line) and self.line[self.pos] == ch:
                    #         self.pos += 1
                    #     res = self._scan_delimiter(prev_token, delimiters)
                    #     return res

        loc = Location(self.filename, self.lineno, start, self.lineno, start + 1)
        return Delimiter(ch, loc)

    def get_next_word(
        self, delimiters: str = DELIMITERS, upper: bool = False
    ) -> tuple[Token | None, Delimiter | None]:
        """
        Get next word from the parse line.

        Parameters
        ----------
        delimiters : str, optional
            Valid delimiters for this context. If None, uses default.
        upper : bool, optional
            Convert word to uppercase.

        Returns
        -------
        word : str
            The extracted word
        delimiter : str
            The delimiter following the word
        """

        while self.pos < len(self.line) and self.line[self.pos].isspace():
            self.pos += 1

        if self.pos >= len(self.line):
            raise EndOfLine()
        # leading delimiter
        if self.line[self.pos] in delimiters:
            delim = self._scan_delimiter(None, delimiters)
            return None, delim

        word, end_pos = self._scan_word(delimiters)
        loc = Location(self.filename, self.lineno, end_pos - len(word), self.lineno, end_pos)
        token = Token(word, loc=loc)
        delim = self._scan_delimiter(word, delimiters)
        if upper and not (word.startswith('"') or word.startswith("'")):
            token = Token(word.upper(), loc=loc)
        return token, delim


def tokenize(contents: str, filename: pathlib.Path | str = "unset") -> list[Block]:
    tok = Tokenizer(contents, filename=filename)
    return [blk.stack() for blk in tok.split_blocks()]
