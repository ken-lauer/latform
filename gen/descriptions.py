"""
Extract per-attribute descriptions from the Bmad reference manual.

The Bmad code exposes an attribute's name, state, kind, and units, but not a
human-readable description. Those live only as prose in the reference manual.
The closest thing to a machine-readable source is ``bmad/doc/elements.tex``,
where each element section documents its attributes in ``\\begin{example}``
blocks of the form::

    b1_gradient  = <Real>    ! Field strength.
    k1           = <Real>    ! Quadrupole strength.
    e1, e2       = <Real>    ! Face angles.
    b_field_tot              ! Net field = b_field + db_field. Dependent param.

This module parses those blocks into a per-element mapping plus a global
fallback. Coverage is only as complete as the manual: not every attribute is
documented, and descriptions are terse.
"""

from __future__ import annotations

import os
import re

# Line inside an example block that documents an attribute. The code portion
# (before "!") is either a bare name / comma list of names, or "names = <TYPE>"
# where <TYPE> is a placeholder in angle brackets. Real usage examples (which
# contain a ":" or an "= value" with a concrete value) are excluded.
_ATTR_LINE = re.compile(
    r"""
    ^\s*
    (?P<names>[a-z][a-z0-9_,()\s]*?)        # one or more lowercase names
    \s*
    (?:=\s*<[^>]*>)?                        # optional "= <Real>" placeholder
    \s*
    !\s*(?P<desc>\S.*?)\s*$                 # "! description"
    """,
    re.VERBOSE,
)

_SECTION = re.compile(r"\\section\{(?P<title>[^}]*)\}")
_EXAMPLE_BEGIN = re.compile(r"\\begin\{example\}")
_EXAMPLE_END = re.compile(r"\\end\{example\}")

# LaTeX cross-reference macros carry no useful prose; drop them entirely.
_REF_MACRO = re.compile(r"\\(?:S?ref|sref|Eqs?|eqs?|Fig|fig)\{[^}]*\}")
# Remaining "\cmd{content}" macros (e.g. \vn{k1}) keep only their content.
_CONTENT_MACRO = re.compile(r"\\[a-zA-Z]+\{([^}]*)\}")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

# Words appearing between element names in grouped section titles.
_TITLE_STOPWORDS = {"and", "or"}


def _clean_tex(text: str) -> str:
    """
    Strip LaTeX markup from a description fragment, leaving readable prose.
    """
    text = _REF_MACRO.sub("", text)
    text = _CONTENT_MACRO.sub(r"\1", text)
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"\(\s*\)", "", text)  # parens emptied by ref removal
    text = re.sub(r"\\([%&_#])", r"\1", text)  # unescape \% \& etc.
    # "See below." / "See above." point at manual prose we don't carry; drop
    # the phrase wherever it appears, keeping any surrounding content.
    text = re.sub(r"\bSee\s+(?:below|above)\b\.?", "", text, flags=re.I)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"\.(\s*\.)+", ".", text)  # collapse ". ." left by ref removal
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    if re.fullmatch(r"(?:[Ss]ee)?\.?", text):  # nothing left but a bare "See."
        return ""
    if text and text == text.lower():  # e.g. "x offset from origin point."
        text = text[0].upper() + text[1:]
    return text


def _names_in_title(title: str, known: set[str]) -> list[str]:
    """
    Return the element keys named in a section title.

    Titles may be a single element (``Quadrupole``) or a group
    (``Bends: Rbend and Sbend``, ``Instrument, Monitor, and Pipe``).
    """
    tokens = re.split(r"[\s,:]+", title)
    out = []
    for tok in tokens:
        if tok.lower() in _TITLE_STOPWORDS:
            continue
        key = tok.upper()
        if key in known and key not in out:
            out.append(key)
    return out


def _parse_attr_line(line: str) -> tuple[list[str], str] | None:
    """
    Parse one example-block line into (attr names, description), or None.
    """
    if ":" in line.split("!", 1)[0]:  # usage example, not an attribute doc
        return None
    m = _ATTR_LINE.match(line)
    if not m:
        return None
    names = [n.strip().upper() for n in m.group("names").split(",")]
    names = [n for n in names if _NAME.match(n.lower())]
    if not names:
        return None
    desc = _clean_tex(m.group("desc"))
    if not desc:
        return None
    return names, desc


def parse_descriptions(
    tex_path: str | os.PathLike[str],
    known_elements: set[str],
) -> dict[str, dict[str, str]]:
    """
    Parse attribute descriptions from ``elements.tex``.

    Only descriptions written in an element's own section are returned; the
    same attribute name may be documented differently for different elements,
    so descriptions are not shared across sections. Generic attributes that no
    section documents are filled in from a curated table (see
    ``common_attrs``), not from here.

    Parameters
    ----------
    tex_path : path-like
        Path to the manual's ``elements.tex``.
    known_elements : set of str
        Upper-cased element keys, used to resolve section titles to elements.

    Returns
    -------
    per_element : dict[str, dict[str, str]]
        ``{ELEMENT: {ATTR: description}}`` from each element's own section.
    """
    with open(tex_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    per_element: dict[str, dict[str, str]] = {}

    current: list[str] = []  # element keys for the section being read
    in_example = False

    for line in lines:
        section = _SECTION.search(line)
        if section:
            current = _names_in_title(section.group("title"), known_elements)
            in_example = False
            continue
        if _EXAMPLE_BEGIN.search(line):
            in_example = True
            continue
        if _EXAMPLE_END.search(line):
            in_example = False
            continue
        if not in_example:
            continue

        parsed = _parse_attr_line(line)
        if parsed is None:
            continue
        names, desc = parsed
        for name in names:
            for elem in current:
                per_element.setdefault(elem, {}).setdefault(name, desc)

    return per_element
