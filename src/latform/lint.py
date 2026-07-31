from __future__ import annotations

import logging
import math
import typing
from functools import lru_cache
from typing import Collection, Generator, Iterable, Sequence

from .attrs import element_key_to_attrs
from .const import named_physical_constants
from .statements import (
    Assignment,
    Constant,
    Element,
    ElementList,
    Line,
    Parameter,
    Simple,
    Statement,
    get_controller_variables,
)
from .token import Token
from .types import Attribute, CallName, Lint, LintCode, Seq
from .walk import iter_tokens

if typing.TYPE_CHECKING:
    import pathlib

    from .config import LatformProjectConfig
    from .parser import Files

logger = logging.getLogger(__name__)


def lint_statements(
    statements: list[Statement],
    named: dict[Token, Statement],
    *,
    assume_defined: bool = True,
    ignore: Collection[str] = (),
    used_names: frozenset[str] | None = None,
    min_name_length: int = 1,
    builtin_constant_rtol: float = 1e-4,
) -> list[Lint]:
    ignored = {code.upper() for code in ignore}
    lints = [lint for st in statements for lint in lint_statement(st)]
    lints.extend(lint_attribute_overrides(statements, named))
    lints.extend(lint_unused_constants(statements, used_names=used_names))
    lints.extend(lint_ambiguous_names(statements, min_name_length=min_name_length))
    lints.extend(lint_builtin_constants(statements, rtol=builtin_constant_rtol))
    if not assume_defined:
        lints.extend(lint_undefined_references(statements, named))
        lints.extend(lint_unknown_element_types(statements))
    return [lint for lint in lints if lint.code.value not in ignored]


def lint_statement(st: Statement) -> list[Lint]:
    if isinstance(st, Simple):
        if not Simple.is_known_statement(st.statement):
            return [
                Lint(
                    code=LintCode.unknown_statement,
                    context=st,
                    message=f"Statement type unknown or parsing issue: {st.statement}",
                    relevant_tokens=[st.statement],
                )
            ]
    if isinstance(st, Element):
        return lint_duplicate_attributes(st) + lint_element_attributes(st)
    return []


def lint_duplicate_attributes(element: Element) -> list[Lint]:
    """
    Flag attributes assigned more than once within a single element.

    Overriding an inherited value (re-setting an attribute in a child element
    that its base element also sets) is fine; only repeats within the same
    statement are flagged.
    """
    first_seen: dict[Token, Token] = {}
    lints = []
    for attr in element.attributes:
        # Indexed/struct names (``tt(1)``, ``curve(1)%r0``) are CallName/Seq;
        # only plain attribute names are checked for duplicates.
        if not isinstance(attr, Attribute) or not isinstance(attr.name, Token):
            continue
        name = attr.name
        key = name.lower()
        if key in first_seen:
            lints.append(
                Lint(
                    code=LintCode.duplicate_attribute,
                    context=element,
                    message=f"Attribute '{name}' is set more than once",
                    relevant_tokens=[first_seen[key], name],
                )
            )
        else:
            first_seen[key] = name
    return lints


def lint_attribute_overrides(
    statements: Sequence[Statement],
    named: dict[Token, Statement],
) -> list[Lint]:
    """
    Flag ``name[attr] = value`` statements that override an earlier setting.

    A parameter statement lints when the attribute was already set in the
    element's own definition, or by an earlier parameter statement for the same
    target and attribute (including builtin targets such as ``parameter``).
    Element-set targets (``rfcavity::*[voltage]``, ``q*[k1]``) lint against
    every matched element definition; two selectors are only considered
    duplicates when their text is identical (overlap between different
    selectors, e.g. ``q*`` vs ``q1``, is not detected).  Overriding a value
    only set on a base element is fine, and names are compared exactly (an
    abbreviation and its full attribute name are not matched), consistent with
    `lint_duplicate_attributes`.
    """
    from .parser import match_element_selector, target_selector_text

    seen: dict[tuple[str, str], Token] = {}
    lints = []
    for st in statements:
        if not isinstance(st, Parameter):
            continue
        if not isinstance(st.target, (Token, Seq)) or not isinstance(st.name, Token):
            continue

        selector = target_selector_text(st.target)
        if isinstance(st.target, Token) and not any(c in selector for c in "*%:>#"):
            original = _defined_attribute_name(named.get(st.target.upper()), st.name)
            originals = [original] if original is not None else []
            message = (
                f"Attribute '{st.name}' of '{st.target}' overrides the value set in its definition"
            )
        else:
            matched = match_element_selector(named.values(), selector)
            if matched is None:
                # TODO: unsupported selector syntax (ranges, branch qualifiers, ...)
                continue
            originals = [
                name
                for name in (_defined_attribute_name(element, st.name) for element in matched)
                if name is not None
            ]
            message = (
                f"Attribute '{st.name}' of elements matching '{selector}' overrides "
                f"values set in their definitions"
            )

        if not originals:
            key = (selector.upper(), str(st.name.upper()))
            original = seen.setdefault(key, st.name)
            if original is st.name:
                continue
            originals = [original]
            message = (
                f"Attribute '{st.name}' of '{selector}' was already set by an earlier statement"
            )

        lints.append(
            Lint(
                code=LintCode.attribute_override,
                context=st,
                message=message,
                relevant_tokens=[*originals, st.name],
            )
        )
    return lints


def _defined_attribute_name(statement: Statement | None, name: Token) -> Token | None:
    """The name token of ``name`` in an element definition's own attributes, if set."""
    if not isinstance(statement, Element):
        return None
    try:
        attr = statement.get_named_attribute(name, partial_match=False)
    except KeyError:
        return None
    return attr.name if isinstance(attr.name, Token) else None


# Single-character names easily mistaken for one another or for the digits
# ``1`` and ``0`` (compare flake8's E741).
_AMBIGUOUS_NAMES = frozenset({"I", "L", "O"})


def lint_ambiguous_names(
    statements: Sequence[Statement],
    *,
    min_name_length: int = 1,
) -> list[Lint]:
    """
    Flag defined names that are too short or easily confused.

    The names of constants, elements, lines, and element lists are checked.
    Names shorter than ``min_name_length`` are flagged; with the default
    minimum of 1, only the single-character names ``i``, ``l``, and ``o`` are
    flagged, as they are easily confused with other names or digits.
    """
    lints = []
    for st in statements:
        if not isinstance(st, (Constant, Element, Line, ElementList)):
            continue
        name = st.name
        if isinstance(name, CallName):
            name = name.name
        if not isinstance(name, Token):
            continue
        if len(name) < min_name_length:
            message = f"Name '{name}' is shorter than the minimum length ({min_name_length})"
        elif str(name.upper()) in _AMBIGUOUS_NAMES:
            message = (
                f"Single-character name '{name}' is easily confused with other names or digits"
            )
        else:
            continue
        lints.append(
            Lint(
                code=LintCode.ambiguous_name,
                context=st,
                message=message,
                relevant_tokens=[name],
            )
        )
    return lints


def _iter_value_tokens(st: Statement) -> Generator[Token, None, None]:
    """
    Yield tokens from a statement's value/expression positions.

    Definition-side tokens (statement names, element attribute names) are
    excluded so that, e.g., an element attribute ``l = 2`` does not count as a
    use of a constant named ``l``.
    """
    match st:
        case Constant(value=value) | Assignment(value=value) | Parameter(value=value):
            yield from iter_tokens(value)
        case Simple(arguments=arguments):
            for arg in arguments:
                if isinstance(arg, Attribute):
                    yield from iter_tokens(arg.value)
                else:
                    yield from iter_tokens(arg)
        case Line(elements=elements) | ElementList(elements=elements):
            yield from iter_tokens(elements)
        case Element(ele_list=ele_list, attributes=attributes):
            yield from iter_tokens(ele_list)
            for attr in attributes:
                if isinstance(attr, Attribute):
                    yield from iter_tokens(attr.value)
                else:
                    yield from iter_tokens(attr)


def _iter_usage_tokens(statements: Sequence[Statement]) -> Generator[Token, None, None]:
    """Yield tokens from positions where a constant could be *referenced*."""
    for st in statements:
        yield from _iter_value_tokens(st)


def get_used_names(statements: Sequence[Statement]) -> frozenset[str]:
    """Uppercase names referenced anywhere in value/expression positions."""
    return frozenset(str(tok).upper() for tok in _iter_usage_tokens(statements))


def lint_unused_constants(
    statements: Sequence[Statement],
    *,
    used_names: frozenset[str] | None = None,
) -> list[Lint]:
    """
    Flag constants that are defined but never referenced.

    Parameters
    ----------
    statements : sequence of Statement
        The statements whose `Constant` definitions are checked.
    used_names : frozenset of str, optional
        Uppercase names considered used.  When linting a multi-file set, pass
        `get_used_names` over *all* files so cross-file usage is recognized;
        defaults to the usages within ``statements`` itself.
    """
    if used_names is None:
        used_names = get_used_names(statements)

    lints = []
    for st in statements:
        if not isinstance(st, Constant) or st.redef:
            continue
        if str(st.name.upper()) not in used_names:
            lints.append(
                Lint(
                    code=LintCode.unused_constant,
                    context=st,
                    message=f"Constant '{st.name}' is defined but never used",
                    relevant_tokens=[st.name],
                )
            )
    return lints


def _iter_numeric_literals(tokens: Iterable[Token]) -> Generator[tuple[Token, float], None, None]:
    """Yield ``(token, float)`` for every finite numeric literal among ``tokens``."""
    for tok in tokens:
        try:
            number = float(str(tok))
        except ValueError:
            continue
        if math.isfinite(number):
            yield tok, number


def _matching_builtin_names(value: float, rtol: float) -> list[str]:
    """Built-in constant names matching ``value`` (or its negation, as ``-name``)."""
    matches = []
    for name, ref in named_physical_constants.items():
        tol = rtol * abs(ref)
        if abs(value - ref) <= tol:
            matches.append(name)
        elif abs(value + ref) <= tol:
            matches.append(f"-{name}")
    return matches


def lint_builtin_constants(
    statements: Sequence[Statement],
    *,
    rtol: float = 1e-4,
) -> list[Lint]:
    """
    Flag numeric literals that duplicate a built-in physical constant.

    Every value/expression position is checked: constant definitions
    (``my_pi = 3.1416``), element attributes (``q: quadrupole, k1 = 3.1415 * 2``),
    and ``name[attr] = value`` statements — including literals nested inside
    expressions.  A literal matches a constant from
    `latform.const.named_physical_constants` when it is within ``rtol``
    relative tolerance of it or its negation; all matching built-in names are
    suggested, negated matches as ``-name``.  Expressions are not numerically
    evaluated (TODO: ``2 * 3.1415`` is caught via the ``3.1415`` literal, but
    ``6.2832`` only matches ``twopi`` directly).

    The default tolerance of 1e-4 catches values hand-rounded to roughly four
    or more significant digits (``3.1416``, ``0.511e6``, ``2.998e8``) while
    staying below the ~1e-3 level where unrelated values start to coincide.
    """
    lints = []
    for st in statements:
        for tok, value in _iter_numeric_literals(_iter_value_tokens(st)):
            matches = _matching_builtin_names(value, rtol)
            if not matches:
                continue
            names = ", ".join(f"'{name}'" for name in matches)
            lints.append(
                Lint(
                    code=LintCode.use_builtin_constant,
                    context=st,
                    message=(
                        f"Value {tok} matches built-in constant(s) {names}; "
                        "use the built-in instead"
                    ),
                    relevant_tokens=[tok],
                )
            )
    return lints


# Hardcoded attribute name aliases, matching Bmad's ``attribute_index2``
# (bmad/modules/attribute_mod.f90). These are resolved before name matching.
_ATTRIBUTE_ALIASES = {
    "REF": "REFERENCE",
    "G_ERR": "DG",
    "B_FIELD_ERR": "DB_FIELD",
}

# Minimum length of an attribute abbreviation (Bmad requires >= 3 characters).
_MIN_ABBREV_LEN = 3


@lru_cache(maxsize=None)
def _acceptable_attribute_names(element_type: str) -> frozenset[str]:
    """
    Every attribute name (uppercase) accepted for an element type.

    Following Bmad's ``attribute_index2`` (``attribute_mod.f90``), this is every
    full attribute name plus every *unambiguous* abbreviation — a prefix of at
    least three characters that is unique to a single attribute. Exact names are
    always included, so a name that is both a full attribute and a (necessarily
    ambiguous) prefix of longer ones still matches, mirroring Bmad's rule that an
    exact match wins over abbreviation.
    """
    names = element_key_to_attrs[element_type]

    prefix_counts: dict[str, int] = {}
    for full in names:
        for length in range(_MIN_ABBREV_LEN, len(full)):
            prefix = full[:length]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    acceptable = set(names)
    acceptable.update(prefix for prefix, count in prefix_counts.items() if count == 1)
    return frozenset(acceptable)


def _is_known_attribute(name: str, element_type: str) -> bool:
    """Whether ``name`` is a valid (possibly abbreviated) attribute of the type."""
    upper = _ATTRIBUTE_ALIASES.get(name.upper(), name.upper())
    return upper in _acceptable_attribute_names(element_type)


def _is_zero_like_token(item: Token | Seq | None):
    if item is None:
        return True
    if isinstance(item, Seq):
        return False
    try:
        return int(item) == 0
    except ValueError:
        return False


def lint_element_attributes(element: Element) -> list[Lint]:
    """
    Flag attributes that are not defined for an element's type.
    """
    if element.element_type is None:
        return []

    element_type = str(element.element_type)
    controller_vars: set[Token] = {var.lower() for var in get_controller_variables(element)}
    controller_values: dict[Token, Token | Seq | None] = {}

    lints = []
    for attr in element.attributes:
        # Indexed/struct names (``tt(1)``, ``curve(1)%r0``) are CallName/Seq and
        # are not represented in the flat attribute schema; only check plain names.
        if not isinstance(attr, Attribute) or not isinstance(attr.name, Token):
            continue
        name = attr.name
        if name.lower() in controller_vars:
            controller_values[name] = attr.value
            continue
        if not _is_known_attribute(str(name), element_type):
            lints.append(
                Lint(
                    code=LintCode.unknown_attribute,
                    context=element,
                    message=(
                        f"Unknown attribute '{name}' for element type "
                        f"'{element.element_type.lower()}'"
                    ),
                    relevant_tokens=[name],
                )
            )

    if len(controller_values) > 1 and all(
        _is_zero_like_token(val) for val in controller_values.values()
    ):
        toks = typing.cast(list[Token], list(controller_values.values()))
        vars = ", ".join(controller_values)
        lints.append(
            Lint(
                code=LintCode.controller_all_zero_defaults,
                context=element,
                message=f"Controller variable values are all set to 0, which is the Bmad implicit default. Variables: {vars}",
                relevant_tokens=toks,
            )
        )

    return lints


def lint_undefined_references(
    statements: Sequence[Statement],
    named: dict[Token, Statement],
) -> list[Lint]:
    """
    Flag ``NAME[attr]`` references whose ``NAME`` is not defined in any loaded file.
    """

    from .parser import _iter_element_references

    lints = []
    for statement, name in _iter_element_references(statements):
        if name.upper() not in named:
            lints.append(
                Lint(
                    code=LintCode.undefined_reference,
                    context=statement,
                    message=f"Reference to undefined element or constant: {name}",
                    relevant_tokens=[name],
                )
            )
    return lints


def lint_unknown_element_types(statements: Sequence[Statement]) -> list[Lint]:
    """
    Flag elements whose type keyword is neither a known Bmad type (or a valid
    abbreviation of one) nor an element defined in a loaded file.
    """
    lints = []
    for statement in statements:
        if (
            isinstance(statement, Element)
            and statement.element_type is None
            and statement.base_element is None
        ):
            lints.append(
                Lint(
                    code=LintCode.unknown_element_type,
                    context=statement,
                    message=f"Unknown element type or undefined base element: {statement.keyword}",
                    relevant_tokens=[statement.keyword],
                )
            )
    return lints


def lint_files(
    files_obj: Files,
    *,
    assume_defined: bool = True,
    ignore: Collection[str] = (),
    config: LatformProjectConfig | None = None,
    check_references: bool = True,
) -> Generator[tuple[pathlib.Path, Lint], None, None]:
    """
    Yield ``(filename, lint)`` for every lint across a parsed `Files` set.

    Parameters
    ----------
    files_obj : Files
        An already-parsed and annotated file set.
    assume_defined : bool, optional
        If False, also report undefined references and unknown element types.
    ignore : collection of str, optional
        Lint codes (e.g. ``"LF004"``) to suppress everywhere.
    config : LatformProjectConfig, optional
        When given, its global and per-file lint ignores are merged in per
        file, and its ``min-name-length`` / ``builtin-constant-rtol`` settings
        apply.
    check_references : bool, optional
        Whether to validate ``tao.init`` datum/variable element references
        against the loaded lattice. Pass False when the set was parsed
        non-recursively, since the element namespace is then incomplete and the
        references cannot be resolved reliably.
    """
    named = files_obj.get_named_items()
    used_names = get_used_names(
        [st for statements in files_obj.by_filename.values() for st in statements]
    )
    min_name_length = config.min_name_length if config is not None else 1
    builtin_constant_rtol = config.builtin_constant_rtol if config is not None else 1e-4
    for fn, statements in files_obj.by_filename.items():
        file_ignore = set(ignore)
        if config is not None:
            file_ignore |= config.ignores_for(fn)
        for lint in lint_statements(
            statements,
            named=named,
            assume_defined=assume_defined,
            ignore=file_ignore,
            used_names=used_names,
            min_name_length=min_name_length,
            builtin_constant_rtol=builtin_constant_rtol,
        ):
            yield fn, lint

    from .tao.lint import lint_tao_init_files

    yield from lint_tao_init_files(
        files_obj, named, ignore=ignore, check_references=check_references
    )


def _build_argparser():
    import argparse

    from . import cli

    parser = argparse.ArgumentParser(
        prog="latform-lint",
        description="Lint Bmad lattice files without reformatting them.",
    )
    cli.add_input_arguments(parser)
    cli.add_config_arguments(parser)
    cli.add_lint_arguments(parser)
    cli.add_logging_arguments(parser, default_level="WARNING")
    return parser


def cli_main(args: list[str] | None = None) -> None:
    """
    CLI entrypoint for ``latform-lint``.

    Parses and lints the given lattice files, printing any findings.  Exits with
    a non-zero status when lints are reported, so it can be used in CI.
    """
    from . import cli
    from .parser import build_files

    parsed = _build_argparser().parse_args(args=args)
    cli.configure_logging(parsed.log_level)

    ignore_codes = cli.resolve_ignore_codes(parsed.ignore_lints)
    config = cli.resolve_config(parsed)
    filenames, from_top_level = cli.require_input_files(parsed.filename, config)

    found = False
    try:
        files_sets = build_files(
            filenames, combine=parsed.combine, input_format=parsed.input_format
        )
    except FileNotFoundError as ex:
        logger.error("%s", ex)
        raise SystemExit(1) from None

    for files_obj in files_sets:
        this_recursive = parsed.recursive
        if this_recursive is None:
            this_recursive = files_obj.tao_init is not None

        # A tao.init is linted on its own (only the namelist, no lattices) when
        # recursion is off, or when it references no design lattices to parse.
        only_tao_init = files_obj.tao_init is not None and (
            not this_recursive or not files_obj.top_files
        )
        if not only_tao_init:
            files_obj.parse(
                recurse=this_recursive,
                raise_if_missing=parsed.error_if_missing,
            )
        files_obj.annotate()
        for fn, lint in lint_files(
            files_obj,
            assume_defined=parsed.assume_defined,
            ignore=ignore_codes,
            config=config,
            check_references=not only_tao_init,
        ):
            found = True
            name = files_obj.local_file_to_source_filename.get(fn, str(fn))
            logger.warning("[%s] %s", name, lint.to_user_message())

    raise SystemExit(1 if found else 0)


if __name__ == "__main__":
    cli_main()
