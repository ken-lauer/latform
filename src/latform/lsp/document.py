"""
Analysis core: parse a document (with its recursive ``call`` includes) into an
`AnalyzedDocument`, with an incremental parse/annotate cache.

Pure latform — importable without ``pygls``.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass

from ..config import LatformProjectConfig
from ..parser import MemoryFiles, _resolve_lattice_paths
from ..statements import Constant, Element, ElementList, Line, Statement
from ..tao import TaoInit, is_init_file, looks_like_namelist
from .positions import definition_name_token

logger = logging.getLogger(__name__)

ParseCache = dict[pathlib.Path, "tuple[str, list[Statement]]"]


def _definition_signature(by_filename: dict) -> tuple:
    """
    A signature of everything that affects cross-file annotation.

    Two builds with the same signature annotate identically: the set of defined
    names, their kinds, element inheritance keywords, and file order.  When it is
    unchanged, files whose contents did not change keep their prior annotation.
    """
    sig: list[tuple] = []
    for filename, statements in by_filename.items():
        for st in statements:
            if isinstance(st, Element):
                sig.append((filename, "E", str(st.name).upper(), str(st.keyword).upper()))
            elif isinstance(st, Constant):
                sig.append((filename, "C", str(st.name).upper()))
            elif isinstance(st, (Line, ElementList)):
                tok = definition_name_token(st)
                sig.append((filename, "L", str(tok).upper() if tok is not None else ""))
    return tuple(sig)


class _OverlayFiles(MemoryFiles):
    """
    `MemoryFiles` with overlay-tolerant reads and incremental parse/annotate.

    Overlay lookup falls back to a resolved path so ``call`` targets containing
    ``..`` or symlinks still match open editor buffers.

    With ``_parse_cache`` set, per-file parsing reuses cached statements for
    files whose contents are unchanged, so an edit only re-parses the changed
    file.  With ``_annotate_state`` also set, the cross-file annotation pass
    re-annotates only the re-parsed files when the definition signature is
    unchanged (an edit that touched no definition), reusing the prior annotation
    of every other file.
    """

    _parse_cache: ParseCache | None = None
    _annotate_state: dict | None = None
    _reparsed: set | None = None
    _named_cache: dict | None = None

    def _get_file_contents(self, filepath: pathlib.Path) -> str:
        for candidate in (filepath, filepath.resolve()):
            if candidate in self.initial_contents:
                return self.initial_contents[candidate]
        return filepath.read_text()

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
        signature = _definition_signature(self.by_filename)
        # Only reuse prior annotation when no definition changed anywhere.
        incremental = state.get("signature") == signature
        state["signature"] = signature

        defined: dict[str, Element] = {}
        for filename, statements in self.by_filename.items():
            if incremental and filename not in self._reparsed:
                # Prior annotation is still valid; just feed the type accumulator
                # so re-parsed files can resolve inheritance from this one.
                for st in statements:
                    if isinstance(st, Element):
                        defined[str(st.name).upper()] = st
                continue
            self._annotate_file(filename, named, defined)


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
    comparison is needed to match a document against the parsed tree.
    """
    for key in files.by_filename:
        if key == resolved or key.resolve() == resolved:
            return key
    return None


def _parse_files(
    top_files: list[pathlib.Path],
    contents: dict[pathlib.Path, str],
    parse_cache: ParseCache | None = None,
    annotate_state: dict | None = None,
) -> tuple[MemoryFiles | None, Exception | None]:
    """Parse and annotate a file set, returning ``(files, error)``."""
    files = _OverlayFiles(top_files=top_files, initial_contents=dict(contents))
    files._parse_cache = parse_cache
    files._annotate_state = annotate_state
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
) -> tuple[MemoryFiles | None, Exception | None]:
    """Parse a project's lattice tree, expanding any ``tao.init`` entries."""
    top_files, tao_inits = _expand_top_files(config, contents)
    files, error = _parse_files(top_files, contents, parse_cache, annotate_state)
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
