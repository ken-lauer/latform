"""Open-buffer state + project discovery (pure latform)."""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field

from ..config import LatformProjectConfig, discover_config
from ..parser import MemoryFiles
from .document import (
    AnalyzedDocument,
    ParseCache,
    _build_project,
    _document_key,
    analyze,
)

logger = logging.getLogger(__name__)


@dataclass
class Workspace:
    """
    Tracks open editor buffers and resolves each document against its project.

    Configuration is discovered by walking up from each document's directory
    (see `discover_config`); a project's parsed file set is cached and shared
    across all its open documents, so analyzing several files of one project
    costs a single parse per edit.
    """

    config_enabled: bool = True
    open_texts: dict[pathlib.Path, str] = field(default_factory=dict)
    _config_by_dir: dict[pathlib.Path, LatformProjectConfig] = field(default_factory=dict)
    # Per-project cache: config source -> (open-buffer signature, parsed files).
    _project_cache: dict[pathlib.Path | None, tuple[tuple, MemoryFiles]] = field(
        default_factory=dict
    )
    # Per-file parse cache: path -> (contents, statements). Survives edits so an
    # edit only re-parses the changed file; unchanged files reuse their
    # statements.
    _parse_cache: ParseCache = field(default_factory=dict)
    # Incremental-annotation state (last definition signature). Lets a rebuild
    # re-annotate only the re-parsed files when no definition changed.
    _annotate_state: dict = field(default_factory=dict)

    def set_text(self, path: pathlib.Path | str, text: str) -> pathlib.Path:
        """Record the current text of an open document; returns its resolved path."""
        resolved = pathlib.Path(path).resolve()
        self.open_texts[resolved] = text
        self._project_cache.clear()
        return resolved

    def close(self, path: pathlib.Path | str) -> None:
        """Forget an open document."""
        self.open_texts.pop(pathlib.Path(path).resolve(), None)
        self._project_cache.clear()

    def invalidate(self) -> None:
        """
        Drop cached configs and parsed projects.

        Called when files change on disk (or a config file is edited) so the
        next analysis re-discovers config and re-reads unopened files.
        """
        self._config_by_dir.clear()
        self._project_cache.clear()
        self._parse_cache.clear()
        self._annotate_state.clear()

    def config_for(self, path: pathlib.Path | str) -> LatformProjectConfig:
        """Discover (and cache) the project config applicable to ``path``."""
        directory = pathlib.Path(path).resolve().parent
        if directory not in self._config_by_dir:
            self._config_by_dir[directory] = discover_config(
                start=directory, enabled=self.config_enabled
            )
        return self._config_by_dir[directory]

    def text_of(self, path: pathlib.Path | str) -> str:
        """Current text of an open document, or its on-disk contents."""
        return self._text_of(pathlib.Path(path).resolve())

    def _text_of(self, resolved: pathlib.Path) -> str:
        text = self.open_texts.get(resolved)
        if text is not None:
            return text
        try:
            return resolved.read_text()
        except OSError:
            return ""

    def _signature(self) -> tuple:
        return tuple(sorted(self.open_texts.items()))

    def _project_files(self, config: LatformProjectConfig) -> MemoryFiles | None:
        """Parse (and cache) the project tree with all open buffers overlaid."""
        signature = self._signature()
        cached = self._project_cache.get(config.source)
        if cached is not None and cached[0] == signature:
            return cached[1]
        files, error = _build_project(
            config, self.open_texts, self._parse_cache, self._annotate_state
        )
        if files is None:
            logger.debug("Project parse failed (%s): %s", config.source, error)
            return None
        self._project_cache[config.source] = (signature, files)
        return files

    def analyze(self, path: pathlib.Path | str) -> AnalyzedDocument:
        """
        Analyze an open (or on-disk) document, within its project when possible.
        """
        resolved = pathlib.Path(path).resolve()
        text = self._text_of(resolved)
        config = self.config_for(resolved)

        if config.top_level:
            files = self._project_files(config)
            if files is not None:
                key = _document_key(files, resolved)
                if key is not None:
                    logger.debug(
                        "Analyze %s: project mode (root=%s, %d files in tree)",
                        resolved,
                        config.root,
                        len(files.by_filename),
                    )
                    return AnalyzedDocument(
                        path=key, files=files, config=config, project_root=config.root
                    )
                logger.debug(
                    "Analyze %s: not in project tree (%s); standalone", resolved, config.source
                )
        else:
            logger.debug("Analyze %s: no project config; standalone", resolved)

        overlay = {p: t for p, t in self.open_texts.items() if p != resolved}
        return analyze(resolved, text, overlay, config=config, parse_cache=self._parse_cache)
