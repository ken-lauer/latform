"""Logging configuration for the language server."""

from __future__ import annotations

import logging
import pathlib
import sys

LOG_LEVELS = ("debug", "info", "warning", "error")
_ENV_LOG_LEVEL = "LATFORM_LSP_LOG_LEVEL"
_ENV_LOG_FILE = "LATFORM_LSP_LOG_FILE"


def configure_logging(level: str = "warning", log_file: str | pathlib.Path | None = None) -> int:
    """
    Route latform/pygls logging to stderr (or a file) at ``level``.

    Logging must never go to stdout: that is the JSON-RPC channel for the
    stdio transport.  Both stderr and a log file are surfaced by LSP clients
    (Neovim's ``:LspLog``, VS Code's output channel).

    Parameters
    ----------
    level : str
        One of ``LOG_LEVELS`` (case-insensitive).
    log_file : path-like, optional
        Write logs here instead of stderr.

    Returns
    -------
    int
        The resolved numeric logging level.
    """
    numeric = getattr(logging, str(level).upper(), logging.WARNING)
    if log_file:
        handler: logging.Handler = logging.FileHandler(pathlib.Path(log_file), encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    for name in ("latform", "pygls"):
        package_logger = logging.getLogger(name)
        package_logger.handlers[:] = [handler]  # replace so restarts don't stack handlers
        package_logger.setLevel(numeric)
        package_logger.propagate = False
    return numeric
