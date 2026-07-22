from __future__ import annotations

import fnmatch
import functools
import logging
import os.path
import pathlib
import re
from dataclasses import dataclass, field
from typing import Generator, Iterable, Sequence

from .attrs import element_key_to_attrs
from .const import EQUALS
from .exceptions import UnexpectedAssignment
from .location import Location
from .statements import (
    BUILTIN_TARGETS,
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
    annotate_controller_variables,
    get_call_filename,
)
from .tao import TaoInit, is_init_file
from .token import Comments, Role, Token
from .tokenizer import tokenize
from .types import (
    COMMA,
    Attribute,
    Block,
    CallName,
    Delimiter,
    FormatOptions,
    Seq,
    TokenizerItem,
)
from .util import partition_items
from .walk import walk

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class _RenameContext:
    lower_renames: dict[str, str]
    regex_renames: dict[re.Pattern, str]
    case_sensitive: bool
    roles: frozenset[Role] = frozenset({Role.name_})
    assume_defined: bool = True

    @classmethod
    def from_renames(
        cls,
        renames: dict[str, str],
        case_sensitive: bool = False,
        roles: set[Role] | None = None,
        assume_defined: bool = True,
    ):
        if not roles:
            roles = set({Role.name_})

        flags = 0
        if not case_sensitive:
            flags |= re.IGNORECASE
        return cls(
            lower_renames={from_.lower(): to for from_, to in renames.items()},
            regex_renames={
                re.compile(from_, flags=flags): to
                for from_, to in renames.items()
                if "*" in from_ or "+" in from_ or "?" in from_
            },
            case_sensitive=case_sensitive,
            roles=frozenset(roles),
            assume_defined=assume_defined,
        )

    @functools.lru_cache(maxsize=None)
    def apply_rename(self, tok: Token, allow_regex: bool = True):
        lower = tok.lower()
        renamed = None
        try:
            renamed = self.lower_renames[lower]
        except KeyError:
            if allow_regex:
                for pat, to in self.regex_renames.items():
                    if pat.match(lower):
                        renamed = pat.sub(to, tok)
                        break

        if renamed:
            return Token(renamed, role=tok.role)

        return tok

    def apply(self, statements: list[Statement]):
        for item in walk(statements):
            node = item.node
            if not isinstance(node, Token):
                continue

            if node.role in self.roles:
                renamed = self.apply_rename(node)
            elif node.role is Role.controller_variable:
                # Controller variables are element-scoped names; rename them on an
                # exact match, but never via regex (a broad pattern shouldn't sweep
                # up locally-scoped variables).
                renamed = self.apply_rename(node, allow_regex=False)
            elif self.assume_defined and node.role is None:
                # A bare, unannotated token may be a reference to a name defined
                # in a file that was not loaded; only literal matches are applied
                # so broad regex patterns can't rewrite numbers/operators.
                renamed = self.apply_rename(node, allow_regex=False)
            else:
                continue

            if renamed is not node:
                item.replace(renamed)


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
                # Element reversal and reflection
                ele.items = [Delimiter("--"), name.with_(role=Role.name_)]
            case Token():
                ele.role = Role.name_
            case Seq():
                ele.items = ele.with_(role=Role.name_).items
            case _:
                raise ValueError(f"Unexpected type found in element list: {type(ele)} {ele=}")

    return eles


known_parameters_keyed = {(param.target, param.name): param for param in Parameter.known}


def fix_parameter_value(
    target: Token, name: Token, value: Token | Seq, raw_value: list[Token | Block]
):
    key = (str(target).lower(), str(name).lower())
    try:
        param = known_parameters_keyed[key]
    except KeyError:
        return value

    if param.type == "species":
        if isinstance(value, Seq):
            return value.to_token(include_opener=False).replace(" ", "")
        return value

    if param.type is str:
        value = Token.join(raw_value)
        if not value.is_quoted_string:
            return Token(f'"{value}"', loc=value.loc, comments=value.comments)
        return value

    if param.type == "geometry":
        if isinstance(value, Token):
            if value.lower().startswith("o"):
                return Token("open", loc=value.loc, comments=value.comments)
            if value.lower().startswith("c"):
                return Token("closed", loc=value.loc, comments=value.comments)
        return value

    if param.type is bool:
        if isinstance(value, Token):
            if value.lower().startswith("t"):
                return Token("True", loc=value.loc, comments=value.comments)
            if value.lower().startswith("f"):
                return Token("False", loc=value.loc, comments=value.comments)
        return value

    return value


def parse_items(items: list[TokenizerItem]):
    if not items:
        raise ValueError("No items provided")

    first = items[0]
    comments = _extract_leading_comment(first)
    first.comments.clear()

    match items:
        # These two cases are handled at the end, along with general 'parameters'.
        # case [Token("beginning") as target, Block() as name, "=", _ as value]:
        # case [Token("parameter") as target, Block() as name, "=", Token() as value]:

        case [Token("redef"), ":", Token() as name, "=", *rest]:
            value = Seq.from_items(rest)
            if isinstance(value, Attribute):
                raise UnexpectedAssignment(
                    f"Unexpected named attribute assignment: {value} at {value.loc}"
                )
            return Constant(
                comments=comments,
                name=name.with_(role=Role.name_),
                value=value,
                redef=True,
            )

        case [Token() as name, "=", *rest]:
            value = Seq.from_items(rest)
            if isinstance(value, Attribute):
                raise UnexpectedAssignment(
                    f"Unexpected named attribute assignment: {value} at {value.loc}"
                )
            return Constant(comments=comments, name=name.with_(role=Role.name_), value=value)

        case [Token() as name, ":", Token("list"), "=", Block(opener="(") as elements_block]:
            return ElementList(
                comments=comments,
                name=name.with_(role=Role.name_),
                elements=_line_elements_from_block(elements_block),
            )

        case [Token() as name, ":", Token("line"), "=", Block(opener="(") as elements_block]:
            return Line(
                comments=comments,
                name=name.with_(role=Role.name_),
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
                name=name.with_(role=Role.name_),
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
            assert isinstance(name, Token)
            return Line(
                comments=comments,
                name=CallName(
                    name=name.with_(role=Role.name_),
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
                        name=name.with_(role=Role.name_),
                        keyword=element_type,
                        ele_list=Seq.from_delimited_block(ele_list, delimiter=COMMA),
                        attributes=_make_attribute_list(after),
                    )

                case [",", *after]:
                    return Element(
                        comments=comments,
                        name=name.with_(role=Role.name_),
                        keyword=element_type,
                        attributes=_make_attribute_list(after),
                    )

                case []:
                    return Element(
                        comments=comments,
                        name=name.with_(role=Role.name_),
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

                if isinstance(target, Token):
                    value = fix_parameter_value(target, name, value, raw_value=after_equals)
                    target = target.with_(role=Role.name_)

                return cls(
                    comments=comments,
                    target=target,
                    name=name.with_(role=Role.attribute_name),
                    value=value,
                )
            # Generic assignment: name = value
            case _:
                name = Seq.from_items(before_equals)
                # This couldn't be an attribute as there's no '=' in there
                assert not isinstance(name, Attribute)
                return Assignment(
                    name=name.with_(role=Role.name_),
                    value=value,
                    comments=comments,
                )

    raise ValueError("Unhandled - unknown")


def get_named_items(statements: Sequence[Statement]) -> dict[Token, Statement]:
    named_items = {}
    for statement in statements:
        if isinstance(statement, (Element, Constant)):
            named_items[statement.name.upper()] = statement
        elif isinstance(statement, Line):
            if isinstance(statement.name, CallName):
                named_items[statement.name.name.upper()] = statement
            else:
                named_items[statement.name.upper()] = statement
    return named_items


def target_selector_text(target: Token | Seq) -> str:
    """Reconstruct the text of an attribute-set statement's target (case-normalized)."""
    if isinstance(target, Token):
        return str(target)
    return str(target.to_token())


def _selector_part_matches(pattern: str, name: str) -> bool:
    """
    Case-insensitive match of one selector part against a name.

    ``*`` matches any run of characters and ``%`` a single character (Bmad's
    convention, translated to fnmatch's ``?``).  Square brackets cannot occur
    in a selector (they delimit the attribute part), so fnmatch character
    classes cannot be formed.
    """
    return fnmatch.fnmatchcase(name.upper(), pattern.upper().replace("%", "?"))


def match_element_selector(statements: Iterable[Statement], selector: str) -> list[Element] | None:
    """
    Elements matched by a Bmad element selector, in definition order.

    Supported: ``*`` and ``%`` wildcards in the element name, and an optional
    ``class::pattern`` prefix whose class part (which may also use wildcards)
    is matched against each element's resolved type.  A plain name matches
    exactly.  All matching is case-insensitive.

    Returns None for selector syntax that is not supported yet:
    ranges (``q1:q5``), branch qualifiers (``lat>>q1``), instance counts
    (``q1##2``), and s-position selectors.
    """
    # TODO: branch qualifiers ("lat>>q1") and instance counts ("q1##2")
    if ">>" in selector or "##" in selector:
        return None

    class_pattern: str | None
    match selector.split("::"):
        case [name_pattern]:
            class_pattern = None
        case [class_pattern, name_pattern]:
            pass
        case _:
            return None
    # TODO: ranges ("q1:q5") and s-position selectors
    if ":" in (class_pattern or "") or ":" in name_pattern:
        return None

    matched = []
    for st in statements:
        if not isinstance(st, Element):
            continue
        if class_pattern is not None and (
            st.element_type is None or not _selector_part_matches(class_pattern, st.element_type)
        ):
            continue
        if _selector_part_matches(name_pattern, str(st.name)):
            matched.append(st)
    return matched


def _iter_element_references(
    statements: Sequence[Statement],
) -> Generator[tuple[Statement, Token], None, None]:
    """Yield ``(statement, name_token)`` for every ``NAME[attr]`` reference."""
    for item in walk(statements):
        node = item.node
        if not isinstance(node, Seq):
            continue
        for idx, sub in enumerate(node.items[:-1]):
            nxt = node.items[idx + 1]
            # Token[token...]
            if (
                isinstance(sub, Token)
                and isinstance(nxt, Seq)
                and nxt.opener == "["
                and all(isinstance(inner, Token) for inner in nxt.items)
            ):
                yield item.statement, sub


_ELEMENT_TYPE_NAMES = frozenset(k for k in element_key_to_attrs if not k.startswith("!"))


def _expand_element_type(keyword: str) -> str | None:
    """Resolve a (possibly abbreviated) type keyword to its canonical name."""
    kw = keyword.upper()
    if kw in _ELEMENT_TYPE_NAMES:
        return kw
    matches = [name for name in _ELEMENT_TYPE_NAMES if name.startswith(kw)]
    # More than one match -> ambiguous
    return matches[0] if len(matches) == 1 else None


def _resolve_element_types(
    statements: Sequence[Statement], defined: dict[str, Element] | None = None
) -> dict[str, Element]:
    """Resolve each element's concrete type, following inheritance."""
    if defined is None:
        defined = {}

    for statement in statements:
        if not isinstance(statement, Element):
            continue

        base = defined.get(statement.keyword.upper())
        if base is not None:
            statement.base_element = base
            statement.element_type = base.element_type
        else:
            statement.base_element = None
            statement.element_type = _expand_element_type(statement.keyword)

        defined[statement.name.upper()] = statement

    return defined


def _resolve_references(statements: Sequence[Statement]) -> None:
    """Annotate ``NAME[attr]`` references as names (or builtins)."""
    for _statement, name in _iter_element_references(statements):
        name.role = Role.builtin if name.lower() in BUILTIN_TARGETS else Role.name_

    for statement in statements:
        if isinstance(statement, Element) and statement.is_controller:
            annotate_controller_variables(statement)


def parse(
    contents: str,
    filename: pathlib.Path | str = "unset",
    annotate: bool = True,
) -> Sequence[Statement]:
    blocks = tokenize(contents, filename)
    res = [block.parse() for block in blocks]
    if annotate:
        named = get_named_items(res)
        for st in res:
            st.annotate(named=named)
        _resolve_element_types(res)
        _resolve_references(res)

    return res


def parse_file(filename: pathlib.Path | str, annotate: bool = True) -> Sequence[Statement]:
    contents = pathlib.Path(filename).read_text()
    return parse(contents=contents, filename=filename, annotate=annotate)


def parse_file_recursive(filename: pathlib.Path | str, annotate: bool = True) -> Files:
    files = Files(top_files=[pathlib.Path(filename)])
    files.parse()
    if annotate:
        files.annotate()
    return files


def is_call_statement(st: Statement) -> bool:
    return isinstance(st, Simple) and st.statement == "call"


implicit_location = Location(filename=pathlib.Path("<implicit>"))


def _resolve_lattice_paths(filenames: Iterable[str], base: pathlib.Path) -> list[pathlib.Path]:
    """Resolve ``tao.init`` lattice entries relative to ``base``.

    Environment variables are expanded (matching ``call`` handling); relative
    entries are joined to ``base`` and all results are resolved to absolute.
    """
    resolved: list[pathlib.Path] = []
    for filename in filenames:
        path = pathlib.Path(os.path.expandvars(str(filename)))
        if not path.is_absolute():
            path = base / path
        resolved.append(path.resolve())
    return resolved


@dataclass
class Files:
    """
    Represents a collection of parsed files starting from one or more
    top-level entry points.
    """

    top_files: list[pathlib.Path] = field(default_factory=list)
    # Stack stores: (relative_filename_to_parse, parent_directory_of_caller)
    stack: list[tuple[pathlib.Path, pathlib.Path]] = field(default_factory=list)
    by_filename: dict[pathlib.Path, list[Statement]] = field(default_factory=dict)
    blocks_by_filename: dict[pathlib.Path, list[Block]] = field(default_factory=dict)
    local_file_to_source_filename: dict[pathlib.Path, str] = field(default_factory=dict)
    filename_calls: dict[pathlib.Path, list[pathlib.Path]] = field(default_factory=dict)
    # When the top files were derived from a Tao ``tao.init`` file, the parsed
    # namelist model and its path are retained here for callers (e.g. templating).
    tao_init: TaoInit | None = None

    @classmethod
    def from_tao_init(cls, path: pathlib.Path | str) -> "Files":
        """Build a :class:`Files` over the lattices listed in a ``tao.init``.

        The ``design_lattice(i)%file`` entries of the ``&tao_design_lattice``
        group become the top-level files, resolved (with environment-variable
        expansion) relative to the ``tao.init`` file's directory.
        """
        path = pathlib.Path(path)
        tao_init = TaoInit.from_file(path)
        top_files = _resolve_lattice_paths(tao_init.lattice_files, path.parent)
        return cls(top_files=top_files, tao_init=tao_init)

    @property
    def main(self) -> pathlib.Path:
        """The first top-level file; convenient for single-entry cases."""
        return self.top_files[0]

    def _add_file_by_statement(self, statement_filename: pathlib.Path, st: Simple) -> pathlib.Path:
        """
        Identify a 'call' statement and add the target file to the stack to be parsed.
        """
        assert isinstance(st, Simple) and st.statement == "call"
        sub_filename, fn = get_call_filename(
            st, caller_directory=statement_filename.parent, expand_vars=True
        )
        self.local_file_to_source_filename[fn] = sub_filename
        logger.debug(f"Adding {sub_filename} relative to {statement_filename.parent} which is {fn}")
        self.stack.append((fn, statement_filename.parent))
        self.filename_calls.setdefault(statement_filename, [])
        self.filename_calls[statement_filename].append(fn)
        return fn

    @property
    def call_graph_edges(self) -> list[tuple[str, str]]:
        """
        Return a list of (caller, callee) string edges for visualization.
        """
        graph = []
        for path_fn, calls in self.filename_calls.items():
            # If path_fn is not in local_file_to_source_filename, it is likely the root
            # loaded differently (or absolute), fall back to name or string rep.
            fn = self.local_file_to_source_filename.get(path_fn, str(path_fn))
            for call_path in calls:
                call_fn = self.local_file_to_source_filename.get(call_path, str(call_path))
                graph.append((fn, call_fn))
        return graph

    def _get_file_contents(self, filepath: pathlib.Path) -> str:
        """Hook to read file contents. default: read from disk."""
        return filepath.read_text()

    def parse(
        self,
        recurse: bool = True,
        raise_if_missing: bool = False,
        keep_blocks: bool = False,
    ):
        """
        Parse the top-level file(s) and optionally their dependencies recursively.

        Parameters
        ----------
        recurse : bool, optional
            Recurse into called lattice files.  Defaults to True.
        raise_if_missing : bool, optional
            For lattice files included by way of ``call`` statements,
            this flag will control whether `FileNotFoundError` is raised.
            If a top-level file is missing, `FileNotFoundError` is always raised.
        keep_blocks : bool, optional
            Store the intermediate `Block` objects in
            ``self.blocks_by_filename`` so callers (e.g. verbose debug output)
            don't have to re-tokenize.
        """
        if not self.top_files:
            raise ValueError("Files requires at least one top-level file in top_files")

        self.top_files = [p.resolve() for p in self.top_files]
        top_set = set(self.top_files)

        if not self.stack:
            # Seed the stack so the first top file is popped first.
            for top in reversed(self.top_files):
                self.stack.append((pathlib.Path(top.name), top.parent))
                self.local_file_to_source_filename.setdefault(top, top.name)

        # We need to track processed files to avoid infinite loops in circular refs
        processed = set(self.by_filename.keys())

        while self.stack:
            filename_part, parent_dir = self.stack.pop()

            # Resolve the full path based on the parent context
            # (Note: filename_part might already be absolute if it's the main entry from disk)
            if filename_part.is_absolute():
                full_path = filename_part
            else:
                full_path = parent_dir / filename_part

            # Optimization: skip if already parsed
            if full_path in processed:
                continue

            logger.debug("Processing %s", full_path)
            processed.add(full_path)

            try:
                contents = self._get_file_contents(full_path)
            except FileNotFoundError:
                logger.error(
                    f"Could not find file: {full_path} (parent={parent_dir} file={filename_part})"
                )
                # Top-level files must exist. Otherwise, missing included files
                # are optionally an error.
                if full_path in top_set or raise_if_missing:
                    raise FileNotFoundError(
                        f"Could not find file: {full_path} (parent={parent_dir} file={filename_part})"
                    ) from None
                continue

            try:
                if keep_blocks:
                    blocks = tokenize(contents=contents, filename=full_path)
                    self.blocks_by_filename[full_path] = blocks
                    statements: list[Statement] = [b.parse() for b in blocks]
                else:
                    # We don't annotate individually here, we do it in bulk later
                    statements = list(parse(contents=contents, filename=full_path, annotate=False))
            except Exception as ex:
                if hasattr(ex, "add_note"):  # py 3.11+
                    ex.add_note(f"Exception ocurred while parsing {full_path}")
                raise

            self.by_filename[full_path] = statements

            for st in statements:
                if is_call_statement(st):
                    # assert isinstance(st, Simple)
                    st.metadata["local_path"] = self._add_file_by_statement(
                        statement_filename=full_path, st=st
                    )

            if not recurse:
                # Without recursion, still process remaining top-level files,
                # but drop anything pulled in via `call` from this file.
                self.stack = [item for item in self.stack if (item[1] / item[0]) in top_set]
                if not self.stack:
                    break

        return self.by_filename

    def annotate(self):
        """
        Resolve named items across all parsed files.
        """
        named = self.get_named_items()
        defined: dict[str, Element] = {}
        for statements in self.by_filename.values():
            for st in statements:
                st.annotate(named=named)
            _resolve_element_types(statements, defined)
            _resolve_references(statements)

    def match_elements(self, pattern: str) -> list[Element] | None:
        """
        Element definitions across all loaded files matching an element selector.

        See `match_element_selector` for the supported syntax; returns None for
        selector syntax that is not supported yet.
        """
        return match_element_selector(
            (st for statements in self.by_filename.values() for st in statements),
            pattern,
        )

    def get_named_items(self) -> dict[Token, Statement]:
        """
        Aggregate named items from all files.
        """
        named_items = {}
        for statements in self.by_filename.values():
            new_items = get_named_items(statements)
            # TODO: potential for linting with redef
            named_items.update(new_items)

        if "BEGINNING" not in named_items:
            named_items["BEGINNING"] = Element(
                name=Token("BEGINNING", loc=implicit_location, role=Role.name_),
                keyword=Token(
                    "BEGINNING_ELE",
                    loc=implicit_location,
                    role=Role.kind,
                ),
            )
        if "END" not in named_items:
            named_items["END"] = Element(
                name=Token("END", loc=implicit_location, role=Role.name_),
                keyword=Token("MARKER", loc=implicit_location, role=Role.kind),
            )

        return named_items

    def _write_reformatted(self, path: pathlib.Path, formatted: str) -> None:
        path.write_text(formatted)

    def flatten(self, call: bool, inline: bool, top: pathlib.Path | None = None) -> list[Statement]:
        # TODO inline handling
        def _flatten(fn):
            res = []
            for st in self.by_filename[fn]:
                if is_call_statement(st):
                    res.extend(_flatten(st.metadata["local_path"]))
                else:
                    res.append(st)
            return res

        return _flatten(top if top is not None else self.main)

    def flatten_all(self, call: bool, inline: bool) -> dict[pathlib.Path, list[Statement]]:
        """Flatten each top-level file independently, keyed by its path."""
        return {top: self.flatten(call=call, inline=inline, top=top) for top in self.top_files}

    def rename(
        self,
        renames: dict[str, str],
        case_sensitive: bool = False,
        only_name_role: bool = True,
        assume_defined: bool = True,
    ):
        roles = (
            {Role.name_}
            if only_name_role
            else {
                Role.attribute_name,
                Role.env_var,
                Role.kind,
            }
        )
        ctx = _RenameContext.from_renames(
            renames, case_sensitive=case_sensitive, roles=roles, assume_defined=assume_defined
        )

        for statements in self.by_filename.values():
            ctx.apply(statements)

    def reformat(self, options: FormatOptions) -> None:
        """
        Reformat all files in the collection.
        """
        from .output import format_statements

        if options.flatten_call:
            for top, statements in self.flatten_all(
                call=options.flatten_call, inline=options.flatten_inline
            ).items():
                formatted = format_statements(statements, options)
                self._write_reformatted(top, formatted)
            return

        for fn, statements in self.by_filename.items():
            formatted = format_statements(statements, options)
            self._write_reformatted(fn, formatted)

    def get_all_referenced_files(
        self,
        normalize_call: bool = True,
        # include_hdf5: bool = False,
    ) -> list[pathlib.Path]:
        loaded_files = list(self.by_filename)

        for statements in self.by_filename.values():
            for item in walk(statements):
                match item.node:
                    case Seq(items=["call", *_rest]):
                        fn = pathlib.Path(item.node.to_token().split("::", 1)[1])
                        if normalize_call:
                            fn = pathlib.Path(os.path.expandvars(fn))

                        if fn not in loaded_files:
                            loaded_files.append(pathlib.Path(fn))

        return loaded_files


@dataclass
class MemoryFiles(Files):
    """
    Files alternative that starts parsing from a string in memory rather than a
    file on disk. Recursion will look to the filesystem relative to
    `root_path`.
    """

    initial_contents: dict[pathlib.Path, str] = field(default_factory=dict)
    _formatted_contents: dict[pathlib.Path, str] = field(default_factory=dict)

    @classmethod
    def from_contents(cls, contents: str, root_path: pathlib.Path | str) -> MemoryFiles:
        """
        Create a MemoryFiles instance from a single string.

        Parameters
        ----------
        contents : str
            The source code content.
        root_path : pathlib.Path | str
            A "virtual" path where this file supposedly lives, used for resolving
            relative calls to other files.

        Returns
        -------
        MemoryFiles
            The initialized object (call .parse() on it next).
        """
        path = pathlib.Path(root_path).resolve()
        return cls(top_files=[path], initial_contents={path: contents})

    @classmethod
    def from_tao_init_contents(
        cls,
        contents: str,
        root_path: pathlib.Path | str,
        lattice_contents: dict[pathlib.Path | str, str] | None = None,
    ) -> MemoryFiles:
        """
        Create a MemoryFiles from in-memory ``tao.init`` contents.

        Parameters
        ----------
        contents : str
            The ``tao.init`` file contents.
        root_path : pathlib.Path | str
            The "virtual" path of the ``tao.init`` file; lattice entries resolve
            relative to its parent directory.
        lattice_contents : dict, optional
            In-memory contents for referenced lattice files, keyed by path or
            name (resolved the same way as the ``design_lattice`` entries). Any
            lattice not provided here is read from disk when parsed.

        Returns
        -------
        MemoryFiles
            The initialized object (call .parse() on it next).
        """
        tao_path = pathlib.Path(root_path).resolve()
        base = tao_path.parent
        tao_init = TaoInit.parse(contents, filename=tao_path)
        top_files = _resolve_lattice_paths(tao_init.lattice_files, base)
        initial = {
            _resolve_lattice_paths([key], base)[0]: text
            for key, text in (lattice_contents or {}).items()
        }
        tao_init.load_sources(base=base, reader=initial.get)
        return cls(
            top_files=top_files,
            initial_contents=initial,
            tao_init=tao_init,
        )

    @classmethod
    def from_mapping(cls, contents: dict[pathlib.Path | str, str]) -> MemoryFiles:
        """
        Create a MemoryFiles instance from multiple in-memory files.

        Keys are treated as top-level files in iteration order.
        """
        resolved = {pathlib.Path(path).resolve(): cts for path, cts in contents.items()}
        return cls(top_files=list(resolved.keys()), initial_contents=resolved)

    def _get_file_contents(self, filepath: pathlib.Path) -> str:
        if filepath in self.initial_contents:
            return self.initial_contents[filepath]
        return super()._get_file_contents(filepath)

    def _write_reformatted(self, path: pathlib.Path, formatted: str) -> None:
        if path in self.initial_contents:
            self._formatted_contents[path] = formatted
        else:
            path.write_text(formatted)

    @property
    def formatted_contents(self) -> str:
        """Get the formatted result for a single in-memory top file."""
        if len(self.initial_contents) != 1:
            raise RuntimeError(
                "formatted_contents is only meaningful for single-entry MemoryFiles; "
                "use formatted_contents_by_path instead."
            )
        top = list(self.initial_contents)[0]
        if top not in self._formatted_contents:
            raise RuntimeError("Contents have not been reformatted yet. Call .reformat() first.")
        return self._formatted_contents[top]

    @property
    def formatted_contents_by_path(self) -> dict[pathlib.Path, str]:
        """All formatted in-memory entries."""
        return dict(self._formatted_contents)


STDIN_TOKEN = "-"
STDIN_LABEL = "<stdin>"
STDIN_FAKE_NAME = "stdin.lat"


def build_files(
    filenames: list[str | pathlib.Path],
    *,
    combine: bool = False,
    root_path: pathlib.Path | None = None,
) -> list[Files]:
    """
    Construct one or more `Files` objects from CLI-style filename arguments.

    Parameters
    ----------
    filenames : list of str or Path
        Filenames to load. ``"-"`` reads from stdin.
    combine : bool, optional
        If True, all filenames are combined into a single `Files`
        (or `MemoryFiles` if any entry is stdin). If False (default),
        each filename becomes its own `Files`, preserving the per-file
        semantics of the legacy CLI loop.
    root_path : pathlib.Path, optional
        Directory used to resolve the synthetic stdin path. Defaults to ``Path.cwd()``.

    Returns
    -------
    list of Files
        One element if ``combine`` is True, otherwise one per input filename.
    """
    import sys

    if not filenames:
        return []
    if root_path is None:
        root_path = pathlib.Path.cwd()

    def _is_stdin(fn) -> bool:
        return str(fn) == STDIN_TOKEN

    def _make_one(fn: str | pathlib.Path) -> Files:
        if _is_stdin(fn):
            fake_name = (root_path / STDIN_FAKE_NAME).resolve()
            files = MemoryFiles(
                top_files=[fake_name], initial_contents={fake_name: sys.stdin.read()}
            )
            files.local_file_to_source_filename[fake_name] = STDIN_LABEL
            return files
        if is_init_file(fn):
            return Files.from_tao_init(fn)
        return Files(top_files=[pathlib.Path(fn)])

    if not combine:
        return [_make_one(fn) for fn in filenames]

    # Combined mode: a single Files (or MemoryFiles if any stdin entry).
    stdin_path: pathlib.Path | None = None
    top_files: list[pathlib.Path] = []
    initial_contents: dict[pathlib.Path, str] = {}

    for fn in filenames:
        if _is_stdin(fn):
            if stdin_path is not None:
                raise ValueError("stdin ('-') can only be used once when combining inputs")
            stdin_path = (root_path / STDIN_FAKE_NAME).resolve()
            top_files.append(stdin_path)
            initial_contents[stdin_path] = sys.stdin.read()
        elif is_init_file(fn):
            top_files.extend(Files.from_tao_init(fn).top_files)
        else:
            top_files.append(pathlib.Path(fn))

    if initial_contents:
        files = MemoryFiles(top_files=top_files, initial_contents=initial_contents)
        if stdin_path is not None:
            files.local_file_to_source_filename[stdin_path] = STDIN_LABEL
        return [files]

    return [Files(top_files=top_files)]
