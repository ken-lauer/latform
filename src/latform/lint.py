from __future__ import annotations

import enum
import logging
import typing
from dataclasses import dataclass
from functools import lru_cache
from typing import Collection, Generator, Sequence

from .attrs import element_key_to_attrs
from .statements import Element, Simple, Statement, get_controller_variables
from .token import Token
from .types import Attribute

if typing.TYPE_CHECKING:
    import pathlib

    from .parser import Files

logger = logging.getLogger(__name__)


class LintCode(str, enum.Enum):
    """Stable identifiers for each lint, usable to opt out via the CLI."""

    unknown_statement = "LF001"
    undefined_reference = "LF002"
    unknown_element_type = "LF003"
    unknown_attribute = "LF004"
    controller_default_missing = "LF005"
    duplicate_attribute = "LF006"


@dataclass()
class Lint:
    code: LintCode
    statement: Statement
    message: str
    relevant_tokens: list[Token] | None

    def to_user_message(self):
        clsname = type(self.statement).__name__
        obj_name = str(getattr(self.statement, "name", "unnamed"))
        parts = [f"[{self.code.value}] {obj_name!r} Statement of type {clsname!r}: {self.message}"]

        if self.relevant_tokens:
            parts.append("\n    Found near:")
            for tok in self.relevant_tokens:
                if tok.loc:
                    parts.append(f"{tok.quoted()} at {tok.loc}")
                else:
                    parts.append(f"{tok.quoted()}")
        return " ".join(parts)


def lint_statements(
    statements: list[Statement],
    named: dict[Token, Statement],
    *,
    assume_defined: bool = True,
    ignore: Collection[str] = (),
) -> list[Lint]:
    ignored = {code.upper() for code in ignore}
    lints = [lint for st in statements for lint in lint_statement(st)]
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
                    statement=st,
                    message=f"Statement type is unknown; this may indicate an error in parsing: {st.statement}",
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
                    statement=element,
                    message=f"Attribute '{name}' is set more than once",
                    relevant_tokens=[first_seen[key], name],
                )
            )
        else:
            first_seen[key] = name
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


def lint_element_attributes(element: Element) -> list[Lint]:
    """
    Flag attributes that are not defined for an element's type.
    """
    if element.element_type is None:
        return []

    element_type = str(element.element_type)
    controller_vars: set[Token] = {var.lower() for var in get_controller_variables(element)}
    controller_defaults_set: set[Token] = set()

    lints = []
    for attr in element.attributes:
        # Indexed/struct names (``tt(1)``, ``curve(1)%r0``) are CallName/Seq and
        # are not represented in the flat attribute schema; only check plain names.
        if not isinstance(attr, Attribute) or not isinstance(attr.name, Token):
            continue
        name = attr.name
        if name.lower() in controller_vars:
            # Default definition
            controller_defaults_set.add(name.lower())
            continue
        if not _is_known_attribute(str(name), element_type):
            lints.append(
                Lint(
                    code=LintCode.unknown_attribute,
                    statement=element,
                    message=(
                        f"Unknown attribute '{name}' for element type "
                        f"'{element.element_type.lower()}'"
                    ),
                    relevant_tokens=[name],
                )
            )

    missing_defaults = controller_vars - controller_defaults_set
    for missing in missing_defaults:
        lints.append(
            Lint(
                code=LintCode.controller_default_missing,
                statement=element,
                message=(f"Controller variable '{missing}' does not have a default set"),
                relevant_tokens=[missing],
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
                    statement=statement,
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
                    statement=statement,
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
        Lint codes (e.g. ``"LF004"``) to suppress.
    """
    named = files_obj.get_named_items()
    for fn, statements in files_obj.by_filename.items():
        for lint in lint_statements(
            statements, named=named, assume_defined=assume_defined, ignore=ignore
        ):
            yield fn, lint


def _build_argparser():
    import argparse

    from ._version import __version__ as package_version

    parser = argparse.ArgumentParser(
        prog="latform-lint",
        description="Lint Bmad lattice files without reformatting them.",
    )
    parser.add_argument(
        "filename",
        nargs="+",
        help="Filename(s) to lint (use '-' for stdin/standard input)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively parse lattice files, following call statements",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Process all input files together as a single set, sharing one parse stack.",
    )
    parser.add_argument(
        "-e",
        "--error-if-missing",
        action="store_true",
        help="If a file is missing during parsing, exit with an error.",
    )
    parser.add_argument(
        "--strict-references",
        dest="assume_defined",
        action="store_false",
        default=True,
        help=(
            "Only recognize element/constant references defined in the loaded files, "
            "reporting anything else (and unknown element types) as lint warnings."
        ),
    )
    parser.add_argument(
        "--ignore",
        dest="ignore_lints",
        action="append",
        metavar="CODE",
        help=(
            "Lint code(s) to suppress, e.g. --ignore LF002 (repeatable, or "
            "comma-separated: --ignore LF002,LF003)."
        ),
    )
    parser.add_argument(
        "--log",
        "-L",
        dest="log_level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "CRITICAL"),
        help="Python logging level (e.g. DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=package_version,
        help="Show the latform version number and exit.",
    )
    return parser


def cli_main(args: list[str] | None = None) -> None:
    """
    CLI entrypoint for ``latform-lint``.

    Parses and lints the given lattice files, printing any findings.  Exits with
    a non-zero status when lints are reported, so it can be used in CI.
    """
    from .parser import build_files

    parsed = _build_argparser().parse_args(args=args)

    logging.getLogger("latform").setLevel(parsed.log_level)
    logging.basicConfig()

    ignore_codes = [
        code.strip()
        for entry in (parsed.ignore_lints or [])
        for code in entry.split(",")
        if code.strip()
    ]

    found = False
    try:
        files_sets = build_files(parsed.filename, combine=parsed.combine)
    except FileNotFoundError as ex:
        logger.error("%s", ex)
        raise SystemExit(1) from None

    for files_obj in files_sets:
        files_obj.parse(recurse=parsed.recursive, raise_if_missing=parsed.error_if_missing)
        files_obj.annotate()
        for fn, lint in lint_files(
            files_obj, assume_defined=parsed.assume_defined, ignore=ignore_codes
        ):
            found = True
            name = files_obj.local_file_to_source_filename.get(fn, str(fn))
            logger.warning("[%s] %s", name, lint.to_user_message())

    raise SystemExit(1 if found else 0)


if __name__ == "__main__":
    cli_main()
