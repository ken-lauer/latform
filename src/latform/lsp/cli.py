"""
Command-line entry point for the latform language server.

Language Server Protocol implementation for Bmad lattice files, exposing
go-to-definition, references, hover, completion, rename, formatting, symbols,
semantic tokens, diagnostics, and code actions over stdio.  ``pygls`` is an
optional dependency; install it with ``pip install latform[lsp]``.
"""

from __future__ import annotations

import argparse
import logging
import os

from .logconfig import _ENV_LOG_FILE, _ENV_LOG_LEVEL, LOG_LEVELS, configure_logging
from .server import create_server

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the ``latform-lsp`` command-line parser."""
    parser = argparse.ArgumentParser(prog="latform-lsp", description=__doc__)
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=os.environ.get(_ENV_LOG_LEVEL, "warning").lower(),
        help="Logging verbosity (default: warning, or $%s)." % _ENV_LOG_LEVEL,
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get(_ENV_LOG_FILE),
        help="Write logs to this file instead of stderr (or $%s)." % _ENV_LOG_FILE,
    )
    parser.add_argument(
        "--no-client-log",
        action="store_true",
        help="Do not mirror log messages to the client via window/logMessage.",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Communicate over stdio (the default and only transport; accepted "
        "for clients that pass it explicitly).",
    )
    parser.add_argument(
        "--publish-delay",
        type=float,
        default=0.15,
        help="Debounce interval in seconds for republishing diagnostics after "
        "document changes; 0 publishes on every change (default: 0.15).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Console-script entry point: start the language server on stdio.

    The log level and log file default to the ``LATFORM_LSP_LOG_LEVEL`` and
    ``LATFORM_LSP_LOG_FILE`` environment variables, and can be overridden with
    ``--log-level`` / ``--log-file``.  Both mechanisms work from any LSP client
    that can pass process arguments or environment (e.g. Neovim's ``cmd`` or
    VS Code's ``latform.server.args``).
    """
    # Ignore any other flags a client injects (e.g. --clientProcessId=...).
    args, unknown = build_arg_parser().parse_known_args(argv)

    level = configure_logging(args.log_level, args.log_file)
    if unknown:
        logger.debug("Ignoring unrecognized arguments: %s", " ".join(unknown))
    logger.info(
        "Starting latform-lsp (log-level=%s%s)",
        args.log_level,
        f", log-file={args.log_file}" if args.log_file else "",
    )
    server = create_server(
        client_log_level=None if args.no_client_log else level,
        publish_delay=args.publish_delay,
    )
    server.start_io()


if __name__ == "__main__":
    main()
