"""
Generate ``latform/_attrs.py`` from the Bmad attribute dump.

* ``gen_attrs`` Fortran program dumps ``ELEMENT|ATTR|STATE|KIND|UNITS``, one row per line
* We read it in from `stdin` as `build.sh` manages that for us
* Then we merge in attribute descriptions parsed from the reference manual's
  ``elements.tex``, and writes the Python module to stdout.

Usage
-----
    ./build/gen_attrs | python gen_attrs.py [elements.tex] > _attrs.py

If the tex path is omitted it defaults to ``$ACC_ROOT_DIR/bmad/doc/elements.tex``.
"""

from __future__ import annotations

import json
import os
import sys

from common_attrs import COMMON
from descriptions import parse_descriptions

HEADER = """\
from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    Does_Not_Exist = "Does_Not_Exist"
    Free = "Free"
    Quasi_Free = "Quasi_Free"
    Dependent = "Dependent"
    Private = "Private"
    Overlay_Slave = "Overlay_Slave"
    Field_Master_Dependent = "Field_Master_Dependent"
    Super_Lord_Align = "Super_Lord_Align"
    Unknown = "Unknown"


class Kind(Enum):
    Real = "Real"
    Integer = "Integer"
    Logical = "Logical"
    Switch = "Switch"
    String = "String"
    Struct = "Struct"
    Unknown = "Unknown"


@dataclass(slots=True, frozen=True)
class Attr:
    name: str
    state: State
    kind: Kind
    units: str
    desc: str = ""


by_element: dict[str, dict[str, Attr]] = {}
"""


def read_rows(stream) -> list[tuple[str, str, str, str, str]]:
    """
    Parse the pipe-delimited attribute dump into rows.
    """
    rows = []
    for line in stream:
        line = line.rstrip("\n")
        if not line:
            continue
        element, attr, state, kind, units = line.split("|")
        rows.append((element, attr, state, kind, units))
    return rows


def main() -> None:
    tex_path = sys.argv[1] if len(sys.argv) > 1 else None
    if tex_path is None:
        root = os.environ.get("ACC_ROOT_DIR", "")
        tex_path = os.path.join(root, "bmad", "doc", "elements.tex")

    rows = read_rows(sys.stdin)
    elements = {element for element, *_ in rows}

    per_element = parse_descriptions(tex_path, elements)

    out = [HEADER]
    for element in dict.fromkeys(element for element, *_ in rows):
        out.append("")
        out.append(f'by_element["{element}"] = {{')
        local = per_element.get(element, {})
        for elem, attr, state, kind, units in rows:
            if elem != element:
                continue
            desc = local.get(attr) or COMMON.get(attr, "")
            out.append(
                f'    "{attr}": Attr('
                f'"{attr}", State.{state}, Kind.{kind}, '
                f"{json.dumps(units)}, {json.dumps(desc)}),"
            )
        out.append("}")

    out.append("")
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
