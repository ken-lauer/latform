"""
Generate ``src/latform/tao/_schema.py`` from external Tao/Bmad type data.

* ``tao_namelists_schema.json`` (from the ``bmad`` repo) — the Tao ``*.init``
  namelist groups and their variables.
* ``structs.json`` (from the ``cppbmad`` repo) — every Fortran derived type,
  used to resolve the derived-type members reachable from those namelists.

It resolves the transitive closure of derived types the namelists reference and
writes a self-contained Python module of plain-data literals that latform's
`latform.tao.schema` reads. Regenerate and commit the output whenever the
upstream schema changes::

    python gen/gen_tao_schema.py \\
        --schema tao_namelists_schema.json \\
        --structs bmad-structs.json

Both inputs are read as raw JSON; no ``cppbmad`` packages need to be importable.
"""

from __future__ import annotations

import argparse
import json
import pathlib

INTRINSIC_TYPES = frozenset({"character", "integer", "real", "logical", "complex"})

# Fortran base types that collapse onto one of `INTRINSIC_TYPES`.
_TYPE_ALIASES = {
    "double": "real",
    "doubleprecision": "real",
    "doublecomplex": "complex",
}

# Schema came from LLM analysis of Tao source.
DEFAULT_SCHEMA = pathlib.Path("tao_namelists_schema.json")
# This comes from cppbmad:
# python -m codegen --write-structs-to=structs.json
DEFAULT_STRUCTS = pathlib.Path("bmad-structs.json")
DEFAULT_OUTPUT = pathlib.Path(__file__).resolve().parent.parent / "src/latform/tao/_schema.py"


def _base_intrinsic(type_name: str) -> str | None:
    """
    Reduce a Fortran type spelling to its base intrinsic name, or ``None``.

    ``character(200)`` -> ``"character"``, ``real(rp)`` -> ``"real"``,
    ``double precision`` -> ``"real"``.  Returns ``None`` for derived types.
    """
    base = type_name.split("(", 1)[0].strip().lower().replace(" ", "")
    base = _TYPE_ALIASES.get(base, base)
    return base if base in INTRINSIC_TYPES else None


def _parse_dimension(dimension: str | None) -> tuple[int | None, int | None] | None:
    """
    Turn a struct member ``dimension`` string into ``(lbound, ubound)``.

    Only rank-1 dimensions yield bounds; a numeric extent ``N`` means
    ``(1, N)`` and ``lo:hi`` means ``(lo, hi)``.  Assumed/deferred (``:``),
    symbolic (``NMAX``), and multi-dimensional (``6,6``) extents still denote an
    array but with unknown bounds, reported as ``(None, None)``.  Scalars return
    ``None``.
    """
    if dimension is None:
        return None
    text = dimension.strip()
    if "," in text:
        # Rank > 1: an array, but we don't bound-check multi-dim indices.
        return (None, None)
    if ":" in text:
        lo, _, hi = text.partition(":")
        return (_maybe_int(lo), _maybe_int(hi))
    extent = _maybe_int(text)
    return (1, extent)


def _maybe_int(text: str) -> int | None:
    try:
        return int(text.strip())
    except ValueError:
        return None


def _char_length_from_paren(type_name: str) -> int | None:
    """
    The declared length of a ``character(N)`` type spelling, or ``None``.

    ``character(200)`` -> ``200``; ``character``, ``character(*)``, and symbolic
    lengths yield ``None`` (no length to check).
    """
    _, sep, rest = type_name.partition("(")
    if not sep:
        return None
    return _maybe_int(rest.rstrip(")"))


def _members(struct: dict) -> dict[str, dict]:
    """The struct's members as a ``{name: member}`` dict (accepts list or dict)."""
    members = struct["members"]
    if isinstance(members, dict):
        return members
    return {member["name"]: member for member in members}


def _field_from_type_info(type_info: dict) -> tuple[str, str, tuple | None, int | None]:
    """
    Build a ``(kind, base, array, length)`` field tuple from a member's type info.
    """
    member_type = type_info["type"]
    member_kind = type_info.get("kind") or ""
    array = _parse_dimension(type_info.get("dimension"))
    base = _base_intrinsic(member_type)
    if base is not None:
        length = _maybe_int(member_kind) if base == "character" else None
        return ("intrinsic", base, array, length)
    struct = member_kind.lower() if member_type.lower() == "type" else member_type.lower()
    return ("derived", struct, array, None)


def _field_from_namelist_var(var: dict) -> tuple[str, str, tuple | None, int | None]:
    """Build a ``(kind, base, array, length)`` field tuple from a namelist variable."""
    array = tuple(var["array"]) if var["array"] is not None else None
    if var["kind"] == "derived":
        return ("derived", var["type_name"].lower(), array, None)
    base = _base_intrinsic(var["type_name"])
    length = _char_length_from_paren(var["type_name"]) if base == "character" else None
    # Fall back to the raw spelling so a surprise type is visible, not silently
    # dropped; the validator treats unknown bases as "don't check".
    return ("intrinsic", base or var["type_name"].lower(), array, length)


def build_schema(schema_path: pathlib.Path, structs_path: pathlib.Path) -> dict:
    """
    Resolve namelists and their reachable derived types into plain-data dicts.
    """
    schema = json.loads(schema_path.read_text())
    struct_data = json.loads(structs_path.read_text())
    structs_by_name = {st["name"].lower(): st for st in struct_data}

    file_key: dict[str, str | None] = {}
    namelists: dict[str, dict[str, tuple]] = {}
    for nl in schema["namelists"]:
        name = nl["name"].lower()
        file_key[name] = nl["file_key"]
        namelists[name] = {
            var["var"].lower(): _field_from_namelist_var(var) for var in nl["variables"]
        }

    # Transitive closure of derived types reachable from the namelist variables.
    resolved: dict[str, dict[str, tuple]] = {}
    missing: set[str] = set()
    pending = [
        base
        for fields in namelists.values()
        for kind, base, *_ in fields.values()
        if kind == "derived"
    ]
    while pending:
        struct_name = pending.pop()
        if struct_name in resolved or struct_name in missing:
            continue
        parsed = structs_by_name.get(struct_name)
        if parsed is None:
            missing.add(struct_name)
            continue
        fields = {
            member_name.lower(): _field_from_type_info(member["type_info"])
            for member_name, member in _members(parsed).items()
        }
        resolved[struct_name] = fields
        pending.extend(base for kind, base, *_ in fields.values() if kind == "derived")

    return {
        "file_key": file_key,
        "namelists": namelists,
        "structs": resolved,
        "missing": sorted(missing),
    }


def _format_field(field: tuple) -> str:
    kind, base, array, length = field
    # Length is only meaningful for character fields; omit it (defaulting to
    # None) everywhere else to keep the generated literals compact.
    if length is None:
        return f"Field({kind!r}, {base!r}, {array!r})"
    return f"Field({kind!r}, {base!r}, {array!r}, {length!r})"


def _format_fields(fields: dict[str, tuple], indent: str) -> str:
    lines = [f"{indent}{name!r}: {_format_field(field)}," for name, field in fields.items()]
    return "\n".join(lines)


def _format_mapping(mapping: dict[str, dict[str, tuple]]) -> str:
    blocks = []
    for name, fields in mapping.items():
        if not fields:
            blocks.append(f"    {name!r}: {{}},")
            continue
        blocks.append(f"    {name!r}: {{\n{_format_fields(fields, ' ' * 8)}\n    }},")
    return "\n".join(blocks)


def render_module(schema: dict) -> str:
    """Render the generated ``tao/_schema.py`` source text."""
    file_key_lines = "\n".join(
        f"    {name!r}: {key!r}," for name, key in schema["file_key"].items()
    )
    return f'''\
"""
**Auto-generated; do not edit by hand.**

Tao ``*.init`` namelist type schema.
"""

from __future__ import annotations

from typing import NamedTuple


class Field(NamedTuple):
    """
    A namelist variable or struct member: kind, base type, shape, char length.

    * ``kind``   — ``"intrinsic"`` or ``"derived"``
    * ``base``   — the intrinsic type name (``character``/``integer``/``real``/
    ``logical``/``complex``) or, for derived fields, the lowercased struct name
    * ``array``  — ``None`` for a scalar, else ``(lbound, ubound)``; a ``None`` bound
    means unbounded/unknown (allocatable, symbolic, or multi-dimensional extent)
    * ``length`` — the declared ``character(N)`` length, or ``None`` for
    non-character fields and unknown/assumed lengths
    """

    kind: str
    base: str
    array: tuple[int | None, int | None] | None
    length: int | None = None


# Namelist name -> ``&tao_start`` key naming its optional separate file
# (``None`` = always read from tao.init).
FILE_KEY: dict[str, str | None] = {{
{file_key_lines}
}}

# Namelist name -> {{variable name -> Field}}.
NAMELISTS: dict[str, dict[str, Field]] = {{
{_format_mapping(schema["namelists"])}
}}

# Derived struct name -> {{member name -> Field}} (reachable closure only).
STRUCTS: dict[str, dict[str, Field]] = {{
{_format_mapping(schema["structs"])}
}}

# Derived types referenced by the schema but absent from the struct data; paths
# descending into these cannot have their member names or leaf types validated.
MISSING_STRUCTS: tuple[str, ...] = {tuple(schema["missing"])!r}
'''


def _ruff_format(path: pathlib.Path) -> None:
    """Format the generated module with ruff, matching the rest of the tree."""
    import subprocess

    try:
        subprocess.run(["ruff", "format", str(path)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"warning: could not run 'ruff format' on {path}: {exc}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=pathlib.Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--structs", type=pathlib.Path, default=DEFAULT_STRUCTS)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    schema = build_schema(args.schema, args.structs)
    args.output.write_text(render_module(schema))
    _ruff_format(args.output)
    print(
        f"Wrote {args.output} "
        f"({len(schema['namelists'])} namelists, {len(schema['structs'])} structs, "
        f"{len(schema['missing'])} missing: {schema['missing']})"
    )


if __name__ == "__main__":
    main()
