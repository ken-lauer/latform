"""
`latform-diff` - compare two lattice files.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from .dump import _fmt, _load_files_and_parse
from .parser import Files
from .statements import Assignment, Constant, Element, Line, Parameter

DESCRIPTION = __doc__
logger = logging.getLogger(__name__)


@dataclass
class GitFiles(Files):
    revision: str = "HEAD"
    repo_root: pathlib.Path = pathlib.Path()

    def _get_file_contents(self, filepath: pathlib.Path) -> str:
        try:
            rel_path = filepath.resolve().relative_to(self.repo_root.resolve())
        except (ValueError, FileNotFoundError) as ex:
            raise FileNotFoundError(
                f"{filepath} in {self.revision} is not in {self.repo_root}"
            ) from ex

        cmd = ["git", "show", f"{self.revision}:{rel_path}"]
        logger.debug(f"Loading from git: {' '.join(cmd)}")
        try:
            return subprocess.check_output(
                cmd, cwd=self.repo_root, text=True, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to load {rel_path} from {self.revision}: {e.stderr}")
            raise FileNotFoundError(f"{rel_path} in {self.revision}") from e


def _split_rev_path(rev_path: str) -> tuple[str, str]:
    if ":" not in rev_path:
        raise ValueError(f"Invalid git specifier: {rev_path}. Must be 'revision:path'")

    revision, path_str = rev_path.split(":", 1)
    if not revision:
        revision = "HEAD"
    return revision, path_str


def _load_git_files_and_parse(
    revision: str, path: str | pathlib.Path, workdir: pathlib.Path
) -> Files:
    main_path = workdir / path

    files = GitFiles(main=main_path, revision=revision, repo_root=find_gitroot(main_path))
    files.parse()
    files.annotate()
    return files


@dataclass(frozen=True)
class ParameterChange:
    """Represents a change in a single parameter (target, name)."""

    target: str
    name: str
    old_value: str | None
    new_value: str | None

    @property
    def value_old_str(self) -> str:
        return self.old_value if self.old_value is not None else ""

    @property
    def value_new_str(self) -> str:
        return self.new_value if self.new_value is not None else ""


@dataclass(frozen=True)
class ElementDiffDetails:
    """
    Holds specific differences for a single element.

    Attributes
    ----------
    type_change : tuple[str, str] | None
        (old_type, new_type) if changed, else None.
    attrs_added : list[tuple[str, str]]
        List of (attr_name, value).
    attrs_removed : list[tuple[str, str]]
        List of (attr_name, value).
    attrs_changed : list[tuple[str, str, str]]
        List of (attr_name, old_value, new_value).
    """

    type_change: tuple[str, str] | None = None
    attrs_added: list[tuple[str, str]] = field(default_factory=list)
    attrs_removed: list[tuple[str, str]] = field(default_factory=list)
    attrs_changed: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(
            self.type_change or self.attrs_added or self.attrs_removed or self.attrs_changed
        )


@dataclass
class LatticeDiff:
    """
    Aggregates all differences between two lattice definitions.

    Attributes
    ----------
    params_added : list[ParameterChange]
    params_removed : list[ParameterChange]
    params_changed : list[ParameterChange]
    eles_added : list[str]
        Names of added elements.
    eles_removed : list[str]
        Names of removed elements.
    eles_changed : dict[str, ElementDiffDetails]
        Map of Element Name -> Diff Details.
    """

    params_added: list[ParameterChange] = field(default_factory=list)
    params_removed: list[ParameterChange] = field(default_factory=list)
    params_changed: list[ParameterChange] = field(default_factory=list)

    eles_added: list[str] = field(default_factory=list)
    eles_removed: list[str] = field(default_factory=list)
    eles_changed: dict[str, ElementDiffDetails] = field(default_factory=dict)
    eles_renamed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_param_diffs(self) -> bool:
        return bool(self.params_added or self.params_removed or self.params_changed)

    @property
    def has_ele_diffs(self) -> bool:
        return bool(self.eles_added or self.eles_removed or self.eles_changed or self.eles_renamed)


def _collect_parameters(files: Files) -> dict[tuple[str, str], str]:
    """Collect all parameters from files."""
    params = {}
    for statements in files.by_filename.values():
        for st in statements:
            if isinstance(st, Parameter):
                target = _fmt(st.target).lower()
                name = _fmt(st.name).lower()
                value = _fmt(st.value)
                params[(target, name)] = value
            elif isinstance(st, (Assignment, Constant)):
                target = ""
                name = _fmt(st.name).lower()
                value = _fmt(st.value)
                params[(target, name)] = value

    return params


def _collect_elements(files: Files) -> dict[str, dict[str, Any]]:
    """
    Collect all elements and their attributes, handling naive inheritance.

    Returns
    -------
    dict
        element_name -> {'type': str, 'attributes': dict, 'loc': Any}
    """
    elements = {}
    named_items = files.get_named_items()

    for name, item in named_items.items():
        ele_name = str(name).upper()

        if isinstance(item, Element):
            attrs = {}
            for attr in item.attributes:
                key = _fmt(attr.name).lower()
                val = _fmt(attr.value)
                attrs[key] = val

            elements[ele_name] = {
                "type": _fmt(item.keyword).upper(),
                "attributes": attrs,
                "loc": item.name.loc,
            }
        elif isinstance(item, Line):
            elements[ele_name] = {
                "type": "LINE",
                "attributes": {"elements": _fmt(item.elements)},
                "loc": item.name.loc if hasattr(item.name, "loc") else None,
            }
            if item.multipass:
                elements[ele_name]["attributes"]["multipass"] = "True"

    # Naive inheritance resolution: copy properties from base Type if defined
    for ele in elements.values():
        if ele["type"] in elements:
            base_ele = elements[ele["type"]]
            for attr, value in base_ele["attributes"].items():
                ele["attributes"].setdefault(attr, value)

    return elements


def calculate_diff(files1: Files, files2: Files) -> LatticeDiff:
    """
    Compute differences between two file sets and return a dataclass.

    Parameters
    ----------
    files1 : Files
        Left-hand file set.
    files2 : Files
        Right-hand file set.

    Returns
    -------
    LatticeDiff
        The structured differences.
    """
    diff = LatticeDiff()

    params1 = _collect_parameters(files1)
    params2 = _collect_parameters(files2)

    p_keys1 = set(params1.keys())
    p_keys2 = set(params2.keys())

    # Added
    for key in sorted(p_keys2 - p_keys1):
        diff.params_added.append(
            ParameterChange(key[0], key[1], old_value=None, new_value=params2[key])
        )

    # Removed
    for key in sorted(p_keys1 - p_keys2):
        diff.params_removed.append(
            ParameterChange(key[0], key[1], old_value=params1[key], new_value=None)
        )

    # Changed
    for key in sorted(p_keys1 & p_keys2):
        if params1[key] != params2[key]:
            diff.params_changed.append(
                ParameterChange(key[0], key[1], old_value=params1[key], new_value=params2[key])
            )

    elements1 = _collect_elements(files1)
    elements2 = _collect_elements(files2)

    e_keys1 = set(elements1.keys())
    e_keys2 = set(elements2.keys())

    diff.eles_added = sorted(e_keys2 - e_keys1)
    diff.eles_removed = sorted(e_keys1 - e_keys2)
    common_eles = e_keys1 & e_keys2

    def is_same_ele(ele1, ele2):
        e1 = elements1[ele1]
        e2 = elements2[ele2]
        return e1["type"] == e2["type"] and e1["attributes"] == e2["attributes"]

    diff.eles_renamed = [
        (ele1, ele2)
        for ele1 in diff.eles_removed
        for ele2 in diff.eles_added
        if is_same_ele(ele1, ele2)
    ]
    # Could detect multiple renames; A -> B1, B2
    # Not technically valid as far as a rename goes;
    # Make this instead a remove/add? Hmm
    for ele1, ele2 in diff.eles_renamed:
        try:
            diff.eles_removed.remove(ele1)
        except ValueError:
            pass
        try:
            diff.eles_added.remove(ele2)
        except ValueError:
            pass

    for name in common_eles:
        e1 = elements1[name]
        e2 = elements2[name]

        details = ElementDiffDetails()

        if e1["type"] != e2["type"]:
            details = dataclasses.replace(details, type_change=(e1["type"], e2["type"]))

        attrs1 = e1["attributes"]
        attrs2 = e2["attributes"]

        a_keys1 = set(attrs1.keys())
        a_keys2 = set(attrs2.keys())

        for a in sorted(a_keys2 - a_keys1):
            details.attrs_added.append((a, attrs2[a]))

        for a in sorted(a_keys1 - a_keys2):
            details.attrs_removed.append((a, attrs1[a]))

        for a in sorted(a_keys1 & a_keys2):
            if attrs1[a] != attrs2[a]:
                details.attrs_changed.append((a, attrs1[a], attrs2[a]))

        if details.has_changes:
            diff.eles_changed[name] = details

    return diff


def print_diff(diff: LatticeDiff, console: Console) -> None:
    """
    Render the diff using Rich tables.
    """
    if diff.has_param_diffs:
        console.rule("[bold]Parameters[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("State")
        table.add_column("Target")
        table.add_column("Name")
        table.add_column("Value (Left)", style="red")
        table.add_column("Value (Right)", style="green")

        for p in diff.params_added:
            table.add_row("Added", p.target, p.name, "", p.value_new_str, style="green")

        for p in diff.params_removed:
            table.add_row("Removed", p.target, p.name, p.value_old_str, "", style="red")

        for p in diff.params_changed:
            table.add_row(
                "Changed", p.target, p.name, p.value_old_str, p.value_new_str, style="yellow"
            )

        console.print(table)
        console.print()

    if diff.has_ele_diffs:
        console.rule("[bold]Elements[/bold]")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("State")
        table.add_column("Element")
        table.add_column("Property/Attribute")
        table.add_column("Value (Left)", style="red")
        table.add_column("Value (Right)", style="green")

        for name in diff.eles_added:
            table.add_row("Added", name, "Element", "", "Exist", style="green")

        for name in diff.eles_removed:
            table.add_row("Removed", name, "Element", "Exist", "", style="red")

        for from_, to in diff.eles_renamed:
            table.add_row("Renamed", from_, "Element", from_, to, style="red")

        for name in sorted(diff.eles_changed.keys()):
            details = diff.eles_changed[name]

            if details.type_change:
                old_t, new_t = details.type_change
                table.add_row("Changed", name, "Type", old_t, new_t, style="magenta")

            for attr, val in details.attrs_added:
                table.add_row("Changed", name, f"Attr: {attr}", "", val, style="green")

            for attr, val in details.attrs_removed:
                table.add_row("Changed", name, f"Attr: {attr}", val, "", style="red")

            for attr, old_v, new_v in details.attrs_changed:
                table.add_row("Changed", name, f"Attr: {attr}", old_v, new_v, style="yellow")

        console.print(table)


def diff_lattices(files1: Files, files2: Files, console: Console):
    diff_data = calculate_diff(files1, files2)
    print_diff(diff_data, console)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="latform-diff",
        description=DESCRIPTION,
    )

    parser.add_argument(
        "file1",
        help="First lattice file",
    )
    parser.add_argument(
        "file2",
        help="Second lattice file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase debug verbosity",
    )

    raw_args = args if args else sys.argv[1:]

    if not raw_args:
        parser.print_help()
        sys.exit(0)

    parsed_args = parser.parse_args(raw_args)
    if parsed_args.verbose > 0:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)

    console = Console()

    cwd = pathlib.Path.cwd()

    def load(fn: str):
        if ":" in fn and not pathlib.Path(fn).exists():
            is_git = ":" in fn and not pathlib.Path(fn).exists()
            if is_git:
                return _load_git_files_and_parse(
                    *_split_rev_path(fn), cwd, verbose=parsed_args.verbose
                )
            return _load_files_and_parse(fn, cwd, verbose=parsed_args.verbose)
        return _load_files_and_parse(fn, cwd, verbose=parsed_args.verbose)

    with console.status("Loading file 1..."):
        f1 = load(parsed_args.file1)

    with console.status("Loading file 2..."):
        f2 = load(parsed_args.file2)

    diff_lattices(f1, f2, console)


def find_gitroot(fn: pathlib.Path) -> pathlib.Path:
    try:
        # git rev-parse --show-toplevel
        res = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=fn.parent if fn.exists() else pathlib.Path.cwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return pathlib.Path(res)
    except subprocess.CalledProcessError:
        pass

    start = fn
    root = pathlib.Path("/")
    while fn != root:
        git = fn / ".git"
        if git.is_dir():
            return fn
        fn = fn.parent
    raise ValueError(f"Unable to find .git directory starting from {start}")


def gitdiff(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="latform-diff",
        description=DESCRIPTION,
    )

    parser.add_argument(
        "lattice_file",
        help="Lattice file",
    )
    parser.add_argument("rev1", help="Git revision 1")
    parser.add_argument(
        "rev2",
        nargs="?",
        default="HEAD",
        help="Git revision 2 (defaults to HEAD)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase debug verbosity",
    )

    raw_args = args if args else sys.argv[1:]

    if not raw_args:
        parser.print_help()
        sys.exit(0)

    parsed_args = parser.parse_args(raw_args)

    if parsed_args.verbose > 0:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
    lattice_file = pathlib.Path(parsed_args.lattice_file).expanduser().resolve()

    git_root = find_gitroot(lattice_file)

    lattice_file = lattice_file.relative_to(git_root)
    console = Console()

    with console.status(f"Loading revision {parsed_args.rev1}: {lattice_file}"):
        f1 = _load_git_files_and_parse(parsed_args.rev1, lattice_file, git_root)

    with console.status(f"Loading revision {parsed_args.rev2}: {lattice_file}"):
        f2 = _load_git_files_and_parse(parsed_args.rev2, lattice_file, git_root)

    diff_lattices(f1, f2, console)


def cli_main(args: list[str] | None = None) -> None:
    main(args)


def cli_main_gitdiff(args: list[str] | None = None) -> None:
    gitdiff(args)


if __name__ == "__main__":
    cli_main()
