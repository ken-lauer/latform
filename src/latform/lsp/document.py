"""
Analysis core: parse a document (with its recursive ``call`` includes) into an
`AnalyzedDocument`, with an incremental parse/annotate cache.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass

from ..config import LatformProjectConfig
from ..parser import MemoryFiles, _resolve_lattice_paths, is_call_statement
from ..statements import Constant, Element, ElementList, Line, Statement
from ..tao import TaoInit, is_init_file, looks_like_namelist
from ..token import Token
from ..walk import walk
from .positions import definition_name_token

logger = logging.getLogger(__name__)

ParseCache = dict[pathlib.Path, "tuple[str, list[Statement]]"]

# On-disk contents keyed by (st_mtime_ns, st_size), so unopened files are only
# re-read when they actually change.
DiskCache = dict[pathlib.Path, tuple[tuple[int, int], str]]


def _file_definition_signature(statements: list[Statement]) -> tuple:
    """
    A signature of everything in one file that affects cross-file annotation.

    Two builds in which every file has the same signature (in the same file
    order) annotate identically: the set of defined names, their kinds, element
    inheritance keywords, and ``call`` sites (whose position determines
    evaluation order, and with it inheritance resolution).
    """
    sig: list[tuple] = []
    for st in statements:
        if isinstance(st, Element):
            sig.append(("E", str(st.name).upper(), str(st.keyword).upper()))
        elif isinstance(st, Constant):
            sig.append(("C", str(st.name).upper()))
        elif isinstance(st, (Line, ElementList)):
            tok = definition_name_token(st)
            sig.append(("L", str(tok).upper() if tok is not None else ""))
        elif is_call_statement(st):
            sig.append(("call", str(st.metadata.get("local_path", ""))))
    return tuple(sig)


def _referenced_names(statements: list[Statement]) -> frozenset[str]:
    """
    The upper-cased text of every token in ``statements``.

    Every name the annotation pass can look up for a file is the ``_upper`` of
    some token within it, so this is a (cheap, conservative) superset of the
    file's cross-file name dependencies.
    """
    return frozenset(item.node._upper for item in walk(statements) if isinstance(item.node, Token))


class _OverlayFiles(MemoryFiles):
    """
    `MemoryFiles` with overlay-tolerant reads and incremental parse/annotate.

    Overlay lookup falls back to a resolved path so ``call`` targets containing
    ``..`` or symlinks still match open editor buffers.

    With ``_parse_cache`` set, per-file parsing reuses cached statements for
    files whose contents are unchanged, so an edit only re-parses the changed
    file.  With ``_annotate_state`` also set, the cross-file annotation pass
    re-annotates only the re-parsed files plus the files that may reference a
    changed definition, reusing the prior annotation of every other file.  With
    ``_disk_cache`` set, unopened files are re-read from disk only when their
    (mtime, size) changes.
    """

    _parse_cache: ParseCache | None = None
    _annotate_state: dict | None = None
    _reparsed: set | None = None
    _named_cache: dict | None = None
    _disk_cache: DiskCache | None = None
    # Per-file definition signatures of this build (set by `annotate`); used by
    # the diagnostics layer to validate cached lints across builds.
    _def_sigs: dict | None = None

    def _get_file_contents(self, filepath: pathlib.Path) -> str:
        for candidate in (filepath, filepath.resolve()):
            if candidate in self.initial_contents:
                return self.initial_contents[candidate]
        cache = self._disk_cache
        if cache is None:
            return filepath.read_text()
        stat = filepath.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        entry = cache.get(filepath)
        if entry is not None and entry[0] == key:
            return entry[1]
        text = filepath.read_text()
        cache[filepath] = (key, text)
        return text

    def get_named_items(self) -> dict:
        # Memoize for this build: statements do not change after parsing, and a
        # publish resolves several documents against the same file set.
        if self._named_cache is None:
            self._named_cache = super().get_named_items()
        return self._named_cache

    def _parse_file(self, contents: str, filename: pathlib.Path) -> list[Statement]:
        cache = self._parse_cache
        if cache is None:
            return super()._parse_file(contents, filename)
        cached = cache.get(filename)
        if cached is not None and cached[0] == contents:
            return cached[1]
        statements = super()._parse_file(contents, filename)
        cache[filename] = (contents, statements)
        if self._reparsed is not None:
            self._reparsed.add(filename)
        return statements

    def annotate(self):
        state = self._annotate_state
        if state is None or self._reparsed is None:
            return super().annotate()

        named = self.get_named_items()
        for filename in self._dirty_files(state):
            self._annotate_file(filename, named)
        # Inheritance resolution is cheap relative to token annotation, and it
        # depends on evaluation order rather than any one file: redo it in full
        # every build.
        self._resolve_element_types_in_order()

    def _dirty_files(self, state: dict) -> set[pathlib.Path]:
        """
        The files whose token annotation cannot be reused from the previous
        build.

        Compares per-file definition signatures against the previous build's:
        the re-parsed files are always dirty, plus every file containing a
        token matching a changed definition name (a superset of the files whose
        annotation could differ).  Signatures and per-file token sets are
        cached by statements-list identity, so unchanged files cost a dict
        lookup.
        """
        file_cache: dict = state.setdefault("files", {})
        sigs: dict[pathlib.Path, tuple] = {}
        deps: dict[pathlib.Path, frozenset[str]] = {}
        for filename, statements in self.by_filename.items():
            cached = file_cache.get(filename)
            if cached is None or cached[0] is not statements:
                cached = (
                    statements,
                    _file_definition_signature(statements),
                    _referenced_names(statements),
                )
                file_cache[filename] = cached
            sigs[filename] = cached[1]
            deps[filename] = cached[2]
        for stale in set(file_cache) - set(self.by_filename):
            del file_cache[stale]

        order = tuple(self.by_filename)
        old_sigs = state.get("sigs")
        old_order = state.get("order")
        state["sigs"] = sigs
        state["order"] = order
        self._def_sigs = sigs

        if old_sigs is None or old_order != order:
            # First build, or the file set changed: re-annotate everything.
            return set(self.by_filename)

        # Definition names whose presence changed; ``call`` entries only affect
        # evaluation order (handled by the full inheritance pass), not roles.
        changed: set[str] = set()
        for filename in old_sigs.keys() | sigs.keys():
            a = sigs.get(filename, ())
            b = old_sigs.get(filename, ())
            if a != b:
                changed.update(entry[1] for entry in set(a) ^ set(b) if entry[0] != "call")

        dirty = self._reparsed & set(self.by_filename) if self._reparsed else set()
        if changed:
            for filename in self.by_filename:
                if filename not in dirty and deps[filename] & changed:
                    dirty.add(filename)
        return dirty


@dataclass
class AnalyzedDocument:
    """
    Result of parsing a document, either standalone or within its project.

    Attributes
    ----------
    path : pathlib.Path
        Key into ``files.by_filename`` for the analyzed document.
    files : MemoryFiles or None
        The parsed file set, or ``None`` if parsing raised.
    error : Exception or None
        The exception raised during parsing, if any.
    config : LatformProjectConfig or None
        The applicable project config, if one was discovered.
    project_root : pathlib.Path or None
        The project root when the document was resolved as part of a project
        tree; ``None`` for standalone (single-file) analysis.
    """

    path: pathlib.Path
    files: MemoryFiles | None = None
    error: Exception | None = None
    config: LatformProjectConfig | None = None
    project_root: pathlib.Path | None = None

    @property
    def statements(self) -> list[Statement]:
        """Statements parsed for this document (empty on error)."""
        if self.files is None:
            return []
        return list(self.files.by_filename.get(self.path, []))


def _document_key(files: MemoryFiles, resolved: pathlib.Path) -> pathlib.Path | None:
    """
    The ``by_filename`` key matching ``resolved``, or ``None`` if absent.

    ``Files`` stores keys as joined-but-not-canonicalized paths, so a resolved
    comparison is needed to match a document against the parsed tree.  The
    resolved-key map is built once per build (resolving every key hits the
    filesystem) and memoized on the files object.
    """
    key_map = getattr(files, "_resolved_key_map", None)
    if key_map is None:
        key_map = {}
        for key in files.by_filename:
            key_map.setdefault(key, key)
            key_map.setdefault(key.resolve(), key)
        files._resolved_key_map = key_map
    return key_map.get(resolved)


def _parse_files(
    top_files: list[pathlib.Path],
    contents: dict[pathlib.Path, str],
    parse_cache: ParseCache | None = None,
    annotate_state: dict | None = None,
    disk_cache: DiskCache | None = None,
) -> tuple[MemoryFiles | None, Exception | None]:
    """Parse and annotate a file set, returning ``(files, error)``."""
    files = _OverlayFiles(top_files=top_files, initial_contents=dict(contents))
    files._parse_cache = parse_cache
    files._annotate_state = annotate_state
    files._disk_cache = disk_cache
    files._reparsed = set() if annotate_state is not None else None
    try:
        files.parse(raise_if_missing=False)
        files.annotate()
    except Exception as exc:  # parsing is best-effort; report as a diagnostic
        return None, exc
    return files, None


def _expand_top_files(
    config: LatformProjectConfig, contents: dict[pathlib.Path, str]
) -> tuple[list[pathlib.Path], list[TaoInit]]:
    """
    Resolve a config's ``top-level`` entries to Bmad lattice paths.

    A ``tao.init`` (Fortran namelist) entry is not a lattice; it is expanded
    into the lattice files it references via ``&tao_design_lattice`` so those
    are what get parsed.  The parsed `TaoInit` objects are returned alongside so
    callers can attach them (for tao-init lints).
    """
    top_files: list[pathlib.Path] = []
    tao_inits: list[TaoInit] = []
    for entry in config.resolve_top_level():
        entry = entry.resolve()
        text = contents.get(entry)
        if text is None:
            try:
                text = entry.read_text()
            except OSError:
                text = None
        if text is not None and (is_init_file(entry) or looks_like_namelist(text)):
            tao_init = TaoInit.parse(text, filename=entry)
            tao_init.load_sources(base=entry.parent, reader=contents.get)
            top_files.extend(_resolve_lattice_paths(tao_init.lattice_files, entry.parent))
            tao_inits.append(tao_init)
        else:
            top_files.append(entry)
    return top_files, tao_inits


def _build_project(
    config: LatformProjectConfig,
    contents: dict[pathlib.Path, str],
    parse_cache: ParseCache | None = None,
    annotate_state: dict | None = None,
    disk_cache: DiskCache | None = None,
) -> tuple[MemoryFiles | None, Exception | None]:
    """Parse a project's lattice tree, expanding any ``tao.init`` entries."""
    top_files, tao_inits = _expand_top_files(config, contents)
    files, error = _parse_files(top_files, contents, parse_cache, annotate_state, disk_cache)
    if files is not None and tao_inits:
        # TODO: with several tao.init entries, associate each with its own tree.
        files.tao_init = tao_inits[0]
    return files, error


def analyze(
    path: pathlib.Path | str,
    text: str,
    overlay: dict[pathlib.Path, str] | None = None,
    *,
    config: LatformProjectConfig | None = None,
    parse_cache: ParseCache | None = None,
) -> AnalyzedDocument:
    """
    Parse ``text`` as the document at ``path``, following ``call`` includes.

    When ``config`` declares ``top-level`` entries and ``path`` is reachable
    from them, the document is analyzed within the whole project tree so that
    cross-file references resolve.  Otherwise the document is analyzed
    standalone, as its own top-level entry point.

    Parameters
    ----------
    path : pathlib.Path or str
        Path of the document being analyzed; used to resolve relative includes.
    text : str
        The current (possibly unsaved) contents of the document.
    overlay : dict of pathlib.Path to str, optional
        Contents of other open buffers, so cross-file resolution prefers live
        editor state over what is on disk.
    config : LatformProjectConfig, optional
        Project config; enables project-tree resolution and lint settings.

    Returns
    -------
    AnalyzedDocument
    """
    resolved = pathlib.Path(path).resolve()
    contents: dict[pathlib.Path, str] = {
        pathlib.Path(p).resolve(): t for p, t in (overlay or {}).items()
    }
    contents[resolved] = text

    if config is not None and config.top_level:
        files, error = _build_project(config, contents, parse_cache)
        if files is not None:
            key = _document_key(files, resolved)
            if key is not None:
                return AnalyzedDocument(
                    path=key, files=files, config=config, project_root=config.root
                )
            # Not part of the project tree; fall through to standalone.
        else:
            logger.debug(
                "Project parse failed (%s); using standalone for %s", config.source, resolved
            )

    files, error = _parse_files([resolved], contents, parse_cache)
    if files is None:
        logger.debug("Parse failed for %s: %s", resolved, error)
        return AnalyzedDocument(path=resolved, files=None, error=error, config=config)
    key = _document_key(files, resolved) or resolved
    return AnalyzedDocument(path=key, files=files, config=config)
