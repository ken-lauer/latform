"""
Element renaming inside Tao namelist files.

Lattice element renames (e.g. from ``latform-template``) must also reach the
element references in a ``tao.init``: datum/variable/curve element-name fields
(both ``datum(1)%ele_name = ...`` and the positional
``datum(1) = 'orbit.x' '' '' 'Q1' ...`` form), element shape IDs and
``search_for_lat_eles`` patterns, and element references embedded in Tao
expressions (``lat::orbit.x[Q1]``, ``ele::Q1[k1]``).

The rename map here is a literal ``old -> new`` mapping (case-insensitive on
the old name), typically the ruleset already expanded over the lattice's real
element names — so a broad rename rule can never rewrite an attribute, datum
name, or expression keyword unless that exact word is a lattice element.
`collect_tao_element_names` harvests every name sitting in an element slot, so
rename rules can also be expanded over names the lattice does not define.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from nmlform import Namelist, NamelistFile, quote_value

from ._schema import STRUCTS
from .file import path_components
from .schema import is_known_namelist, resolve_path

logger = logging.getLogger(__name__)

__all__ = ["rename_tao_elements", "collect_tao_element_names"]

# ``name -> replacement or None`` (None = leave unchanged); the hook shared by
# renaming (map lookup) and harvesting (record the name, change nothing).
Lookup = Callable[[str], "str | None"]

# Fields whose whole value is an element name (modulo ``##N`` / ``\N`` suffixes).
_ELEMENT_NAME_FIELDS = frozenset({"ele_name", "ele_ref_name", "ele_start_name"})
# Fields holding element match patterns: an optional ``class::`` prefix and a
# name that may contain wildcards.
_ELEMENT_PATTERN_FIELDS = frozenset({"ele_id", "search_for_lat_eles"})
# Fields that may hold Tao expressions with embedded element references.
_EXPRESSION_FIELDS = frozenset({"data_type", "data_type_x", "data_type_z", "default_data_type"})

_GLOB_CHARS = frozenset("*%")

# A scoped Tao expression reference: ``scope::path`` with an optional
# ``[slot]`` subscript (e.g. ``lat::orbit.x[Q1]|model``, ``ele::Q1[k1]``).
_EXPR_REF = re.compile(
    r"""
    (?P<scope>[A-Za-z_][A-Za-z0-9_]*)::   # qualifier, e.g. lat::
    (?P<path>[\w.\\\#%*]*)                # parameter or element path
    (?:\[(?P<slot>[^\]]*)\])?             # optional bracketed slot
    """,
    re.VERBOSE,
)

# An unscoped ``name[...]`` reference (for the fallback debug log only).
_BARE_REF = re.compile(r"[A-Za-z_][\w.]*\[")

# Where the element name sits in a scoped reference.
_ELE_PATH_SCOPES = frozenset({"ele", "ele_mid"})  # ele::Q1[k1]
_ELE_SLOT_SCOPES = frozenset({"lat", "beam"})  # lat::orbit.x[Q1]
_NON_ELEMENT_SCOPES = frozenset({"data", "var", "wall"})

# Pseudo-element location names that are not lattice elements; excluded from
# harvesting so broad rename rules cannot sweep them up.
_RESERVED_ELEMENT_NAMES = frozenset({"beginning", "end"})


def _field_category(field: str) -> str | None:
    if field in _ELEMENT_NAME_FIELDS:
        return "name"
    if field in _ELEMENT_PATTERN_FIELDS:
        return "pattern"
    if field in _EXPRESSION_FIELDS:
        return "expression"
    return None


def _rename_element_name(text: str, lookup: Lookup) -> str:
    """
    Rename the element-name parts of ``text``.

    A ``##N`` occurrence suffix and ``\\N`` multipass-slave suffixes are kept;
    only the name parts are looked up (integer parts never match a rename map).
    """
    head, sep, occurrence = text.partition("##")
    parts = [lookup(part) or part for part in head.split("\\")]
    return "\\".join(parts) + sep + occurrence


def _rename_slot(slot: str, lookup: Lookup) -> str:
    """Rename element names in a ``[...]`` slot, honoring ``:``/``,`` lists."""
    out = []
    for part in re.split(r"([:,])", slot):
        stripped = part.strip()
        new = _rename_element_name(stripped, lookup)
        out.append(part.replace(stripped, new, 1) if new != stripped else part)
    return "".join(out)


def _rename_expression(value: str, lookup: Lookup) -> str:
    """Rename element names in the element-bearing slots of scoped references."""
    out: list[str] = []
    last = 0
    covered: list[tuple[int, int]] = []
    for match in _EXPR_REF.finditer(value):
        covered.append(match.span())
        out.append(value[last : match.start()])
        out.append(_rename_ref(match, lookup))
        last = match.end()
    out.append(value[last:])

    for match in _BARE_REF.finditer(value):
        if any(start <= match.start() < end for start, end in covered):
            continue
        logger.debug(
            "Leaving unscoped reference %r in %r untouched during element rename",
            match.group(0),
            value,
        )
    return "".join(out)


def _rename_ref(match: re.Match, lookup: Lookup) -> str:
    scope = match.group("scope")
    path = match.group("path")
    slot = match.group("slot")
    kind = scope.lower()
    if kind in _ELE_PATH_SCOPES:
        path = _rename_element_name(path, lookup)
    elif kind in _ELE_SLOT_SCOPES:
        if slot is not None:
            slot = _rename_slot(slot, lookup)
    elif kind not in _NON_ELEMENT_SCOPES:
        logger.debug(
            "Leaving unrecognized scope %r in %r untouched during element rename",
            f"{scope}::",
            str(match.string),
        )
    rebuilt = f"{scope}::{path}"
    if slot is not None:
        rebuilt += f"[{slot}]"
    return rebuilt


def _rename_pattern(value: str, lookup: Lookup, *, key: str, warn: bool = True) -> str:
    """
    Rename exact element names in a match-pattern value.

    Each whitespace-separated word may carry a ``class::`` prefix; flag words
    (``-...``) are skipped. A word whose name part contains wildcards cannot be
    statically rewritten and is left untouched with a warning.
    """
    out = []
    for word in re.split(r"(\s+)", value):
        if not word or word.isspace() or word.startswith("-"):
            out.append(word)
            continue
        prefix, sep, name = word.rpartition("::")
        if _GLOB_CHARS & set(name):
            if warn:
                logger.warning(
                    "Element match pattern %r for '%s' contains wildcards; element renames "
                    "are not applied to it",
                    word,
                    key,
                )
            out.append(word)
            continue
        out.append(prefix + sep + _rename_element_name(name, lookup))
    return "".join(out)


def _apply_category(category: str, text: str, lookup: Lookup, *, key: str, warn: bool) -> str:
    if category == "name":
        return _rename_element_name(text, lookup)
    if category == "pattern":
        return _rename_pattern(text, lookup, key=key, warn=warn)
    return _rename_expression(text, lookup)


def _transform_quoted(text: str, transform: Callable[[str], str]) -> str:
    """Apply ``transform`` to the string content, preserving the quote style."""
    if len(text) >= 2 and text[0] in "'\"" and text[-1] == text[0]:
        quote = text[0]
        inner = text[1:-1].replace(quote * 2, quote)
        new = transform(inner)
        return text if new == inner else quote_value(new, quote)
    return transform(text)


def _repeat_prefix(text: str) -> tuple[int, str, str]:
    """Split a ``n*value`` Fortran repeat into ``(n, "n*", value)`` (n=1 if none)."""
    count, star, value = text.partition("*")
    if star and count.strip().lstrip("+-").isdigit():
        return int(count), f"{count}*", value
    return 1, "", text


def _transform_token(text: str, transform: Callable[[str], str]) -> str:
    """Apply ``transform`` to a value literal, preserving any ``n*`` repeat."""
    _, prefix, value = _repeat_prefix(text)
    return prefix + _transform_quoted(value, transform)


def _transformed_value(assignment, transform: Callable[[str], str]) -> str | None:
    """The assignment's right-hand side transformed, or ``None`` if unchanged."""
    changed = False
    parts = []
    for token in assignment.field_tokens:
        fixed = _transform_token(token.text, transform)
        parts.append(fixed)
        changed = changed or fixed != token.text
    return " ".join(parts) if changed else None


def _positional_struct_fields(group_name: str, assignment) -> tuple[str, ...] | None:
    """
    The positional field order for a whole-struct assignment, if renameable.

    ``datum(1) = 'orbit.x' '' '' 'Q1' ...`` assigns a derived-type entry whose
    values fill the struct's fields in declaration order. Returns ``None`` when
    the assignment does not target a derived struct or the struct has no
    element-bearing fields.
    """
    leaf = resolve_path(group_name, path_components(assignment)).leaf
    if leaf is None or leaf.kind != "derived":
        return None
    fields = tuple(STRUCTS.get(leaf.base) or ())
    if not any(_field_category(field) for field in fields):
        return None
    return fields


def _positional_transformed(
    assignment,
    fields: tuple[str, ...],
    lookup: Lookup,
    *,
    warn: bool,
) -> str | None:
    """
    Transform a positional whole-struct assignment's element-bearing slots.

    Each value token fills the next field (a ``n*`` repeat fills ``n``); tokens
    whose field has no element category pass through untouched.
    """
    changed = False
    parts = []
    position = 0
    for token in assignment.field_tokens:
        text = token.text
        if getattr(token, "kind", "") == "strcont":
            logger.debug(
                "Positional assignment '%s' continues a string across lines; "
                "element renames are not applied to it",
                assignment.key,
            )
            return None
        repeat, _, _ = _repeat_prefix(text)
        covered = fields[position : position + repeat]
        position += repeat
        categories = {c for c in (_field_category(field) for field in covered) if c}
        if len(categories) != 1:
            if categories:
                logger.debug(
                    "Repeat %r in '%s' spans differently-typed fields; "
                    "element renames are not applied to it",
                    text,
                    assignment.key,
                )
            parts.append(text)
            continue
        (category,) = categories
        fixed = _transform_token(
            text,
            lambda t: _apply_category(category, t, lookup, key=assignment.key, warn=warn),
        )
        parts.append(fixed)
        changed = changed or fixed != text
    return " ".join(parts) if changed else None


def _rename_group(group: Namelist, lookup: Lookup, *, warn: bool) -> None:
    if not is_known_namelist(group.name):
        return
    # `Namelist.set` reparses (invalidating `group.assignments`), so collect the
    # edits first, then apply them.
    edits: list[tuple[str, str]] = []
    for assignment in group.assignments:
        category = _field_category(assignment.path.names[-1].lower())
        if category is not None:
            fixed = _transformed_value(
                assignment,
                lambda t: _apply_category(category, t, lookup, key=assignment.key, warn=warn),
            )
        else:
            fields = _positional_struct_fields(group.name, assignment)
            if fields is None:
                continue
            fixed = _positional_transformed(assignment, fields, lookup, warn=warn)
        if fixed is not None:
            edits.append((assignment.key, fixed))
    for key, value in edits:
        group.set(key, value)


def _each_group(namelist: Namelist | NamelistFile):
    if isinstance(namelist, NamelistFile):
        for item in namelist.items:
            if isinstance(item, Namelist):
                yield item
    else:
        yield namelist


def rename_tao_elements(
    namelist: Namelist | NamelistFile,
    renames: dict[str, str],
) -> None:
    """
    Rename lattice element references in a Tao namelist (or file) in place.

    Parameters
    ----------
    namelist : Namelist | NamelistFile
        The namelist group or file to rewrite. Given a file, every known
        namelist group in it is processed (auxiliary ``&tao_start`` sources are
        separate files and are not touched — process them separately).
    renames : dict[str, str]
        Literal ``old -> new`` element renames, matched case-insensitively.
        This should be the concrete name map (e.g. a ruleset expanded over the
        lattice's element names), not patterns.

    Notes
    -----
    Covers element-name fields (``ele_name``, ``ele_ref_name``,
    ``ele_start_name``) written by component or positionally
    (``datum(1) = 'orbit.x' '' '' 'Q1' ...``), match-pattern fields (``ele_id``,
    ``search_for_lat_eles``; exact names only — wildcard patterns are left
    untouched with a warning), and element references inside expression-bearing
    fields (``data_type`` and friends): ``ele::NAME[...]`` and
    ``lat::param[NAME]`` style references. ``data::``/``var::`` references and
    unrecognized constructs are left untouched.
    """
    if not renames:
        return
    upper = {old.upper(): new for old, new in renames.items()}

    def lookup(name: str) -> str | None:
        return upper.get(name.upper())

    for group in _each_group(namelist):
        _rename_group(group, lookup, warn=True)


def collect_tao_element_names(namelist: Namelist | NamelistFile) -> set[str]:
    """
    Every element name referenced in a Tao namelist's element-bearing slots.

    Walks the same fields as `rename_tao_elements` (element-name fields,
    non-wildcard pattern names, expression element slots) and returns the base
    names found (``##N``/``\\N`` suffixes stripped; integer parts and the
    ``beginning``/``end`` pseudo-elements skipped).
    Useful for expanding rename rules over names a lattice does not define.
    """
    seen: set[str] = set()

    def lookup(name: str) -> str | None:
        if name and not name.isdigit() and name.lower() not in _RESERVED_ELEMENT_NAMES:
            seen.add(name)
        return None

    for group in _each_group(namelist):
        _rename_group(group, lookup, warn=False)
    return seen
