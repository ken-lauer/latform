"""
Tests for the language-server analysis layer and pygls glue.

The analysis layer is pure latform and always runs; tests that require
``pygls`` are skipped when it is not installed.
"""

from __future__ import annotations

import pathlib

import pytest

from latform import lsp

FILES = pathlib.Path(__file__).parent / "files"

FODO = """\
L_tot = 2
Lq = 0.1

q0: quad, L = Lq
q1: q0
q2: q0

fodo1: line = (q1, q2)

use, fodo1
"""


@pytest.fixture
def fodo(tmp_path: pathlib.Path) -> lsp.AnalyzedDocument:
    """
    An analyzed in-memory ``fodo`` lattice document.
    """
    path = tmp_path / "fodo.bmad"
    return lsp.analyze(path, FODO)


def _pos(needle: str, occurrence: int = 0) -> tuple[int, int]:
    """
    (line, column) of the ``occurrence``-th ``needle`` in the FODO source.
    """
    lines = FODO.splitlines()
    seen = -1
    for i, line in enumerate(lines):
        col = line.find(needle)
        while col != -1:
            seen += 1
            if seen == occurrence:
                return i, col
            col = line.find(needle, col + 1)
    raise AssertionError(f"{needle!r} occurrence {occurrence} not found")


def test_analyze_success(fodo: lsp.AnalyzedDocument) -> None:
    assert fodo.error is None
    assert fodo.files is not None
    assert fodo.statements


def test_analyze_reports_parse_error(tmp_path: pathlib.Path) -> None:
    analyzed = lsp.analyze(tmp_path / "bad.bmad", "q0: quad, l = (((")
    diags = list(lsp.iter_diagnostics(analyzed))
    assert analyzed.error is not None
    assert any(d.code == "parse-error" for d in diags)


def test_token_at_position_finds_reference(fodo: lsp.AnalyzedDocument) -> None:
    line, col = _pos("q0", occurrence=1)  # the reference in "q1: q0"
    tok = lsp.token_at_position(fodo.statements, line, col)
    assert tok is not None
    assert str(tok).upper() == "Q0"


def test_token_at_position_off_token_is_none(fodo: lsp.AnalyzedDocument) -> None:
    assert lsp.token_at_position(fodo.statements, 1, 200) is None


@pytest.mark.parametrize(
    ("needle", "occurrence", "expected_def_needle"),
    [
        ("q0", 1, "q0: quad"),  # element reference -> element definition
        ("Lq", 1, "Lq = 0.1"),  # constant reference -> constant definition
        ("q1", 1, "q1: q0"),  # reference in the line -> element definition
    ],
)
def test_resolve_definition(
    fodo: lsp.AnalyzedDocument, needle: str, occurrence: int, expected_def_needle: str
) -> None:
    line, col = _pos(needle, occurrence)
    loc = lsp.resolve_definition(fodo, line, col)
    assert loc is not None
    def_line = next(
        i for i, ln in enumerate(FODO.splitlines()) if ln.startswith(expected_def_needle)
    )
    assert loc.line == def_line


def test_resolve_definition_none_off_symbol(fodo: lsp.AnalyzedDocument) -> None:
    assert lsp.resolve_definition(fodo, 200, 0) is None


def test_hover_element_and_constant(fodo: lsp.AnalyzedDocument) -> None:
    line, col = _pos("q0", 0)  # element definition
    assert "QUAD" in (lsp.hover_text(fodo, line, col) or "").upper()

    line, col = _pos("Lq", 0)  # constant definition
    hover = lsp.hover_text(fodo, line, col) or ""
    assert "constant" in hover


def test_document_symbols(fodo: lsp.AnalyzedDocument) -> None:
    symbols = {name: kind for name, kind, _ in lsp.document_symbols(fodo)}
    assert symbols["q0"] == "element"
    assert symbols["fodo1"] == "line"
    assert symbols["Lq"] == "constant"


def test_loc_contains_inclusive_end() -> None:
    from latform.location import Location

    loc = Location(line=0, column=2, end_line=0, end_column=4)  # inclusive end
    assert lsp.loc_contains(loc, 0, 2)
    assert lsp.loc_contains(loc, 0, 4)
    assert not lsp.loc_contains(loc, 0, 5)
    assert not lsp.loc_contains(loc, 0, 1)


def test_diagnostics_use_real_lattice() -> None:
    """
    The linter runs against a real fixture and yields located diagnostics.
    """
    path = FILES / "fodo.bmad"
    analyzed = lsp.analyze(path, path.read_text())
    for diag in lsp.iter_diagnostics(analyzed):
        assert diag.location is not None
        assert diag.message


def _diags_by_code(analyzed: lsp.AnalyzedDocument) -> dict[str, lsp.Diagnostic]:
    return {diag.code: diag for diag in lsp.iter_diagnostics(analyzed)}


@pytest.mark.parametrize(
    ("text", "override_line"),
    [
        ("q1: quad, k1 = 2\nq1[k1] = 3\n", 1),  # overrides the value set in the definition
        ("q1: quad\nq1[k1] = 2\nq1[k1] = 3\n", 2),  # set twice by parameter statements
    ],
)
def test_lf008_override_anchors_on_offending_line(
    tmp_path: pathlib.Path, text: str, override_line: int
) -> None:
    """
    An attribute-override lint anchors on the offending line (not a span from
    the original definition), with the earlier occurrence as related info.
    """
    path = tmp_path / "a.bmad"
    path.write_text(text)
    diag = _diags_by_code(lsp.analyze(path, text)).get("LF008")
    assert diag is not None
    assert diag.location.line == override_line
    assert diag.location.end_line == override_line  # a single-line range, not a merged span
    assert diag.related  # points back to where the value was first set
    assert all(loc.line < override_line for loc, _ in diag.related)


def test_lf008_updates_in_workspace_state(tmp_path: pathlib.Path) -> None:
    """
    Editing away the override clears the diagnostic from workspace analysis.
    """
    path = tmp_path / "a.bmad"
    workspace = lsp.Workspace()

    workspace.set_text(path, "q1: quad, k1 = 2\nq1[k1] = 3\n")
    assert "LF008" in _diags_by_code(workspace.analyze(path))

    workspace.set_text(path, "q1: quad, k1 = 2\n")
    assert "LF008" not in _diags_by_code(workspace.analyze(path))


def test_lint_locations_tolerate_empty_and_nonstatement_context() -> None:
    """
    A lint with no relevant tokens and a non-statement ``context`` must not
    crash (regression: the fallback referenced the removed ``statement`` field).
    """
    from latform.lint import Lint, LintCode

    lint = Lint(
        code=LintCode.undefined_reference,
        context=object(),
        message="x",
        relevant_tokens=[],
    )
    assert lsp._lint_locations(lint) == (None, [])


# --------------------------------------------------------------------------- #
# tao.init (namelist) projects
# --------------------------------------------------------------------------- #

TAO_INIT = '&tao_start\n/\n&tao_design_lattice\n  design_lattice(1)%file = "lat.bmad"\n/\n'
LATTICE_WITH_OVERRIDE = "q1: quad, k1 = 2\nq1[k1] = 3\n"


@pytest.fixture
def tao_project(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    A project whose ``top-level`` is a ``tao.init`` referencing ``lat.bmad``
    (which has an LF008 attribute override).  Returns the project root.
    """
    (tmp_path / "latform.toml").write_text('top-level = ["tao.init"]\n')
    (tmp_path / "tao.init").write_text(TAO_INIT)
    (tmp_path / "lat.bmad").write_text(LATTICE_WITH_OVERRIDE)
    return tmp_path


def test_tao_init_project_lints_referenced_lattice(tao_project: pathlib.Path) -> None:
    """
    A lattice referenced by a ``tao.init`` top-level resolves in project mode
    (the tao.init is expanded to its lattices, not parsed as Bmad) and lints.
    """
    lat = tao_project / "lat.bmad"
    workspace = lsp.Workspace()
    workspace.set_text(lat, LATTICE_WITH_OVERRIDE)

    analyzed = workspace.analyze(lat)
    assert analyzed.project_root == tao_project.resolve()
    assert analyzed.files is not None and analyzed.files.tao_init is not None

    diag = _diags_by_code(analyzed).get("LF008")
    assert diag is not None
    assert diag.location.line == 1


def test_tao_init_opened_directly_does_not_crash(tao_project: pathlib.Path) -> None:
    """
    Opening the ``tao.init`` itself must not parse it as a lattice (regression:
    the recursive parse raised ``Unhandled - unknown: & tao_start``).
    """
    tao = tao_project / "tao.init"
    workspace = lsp.Workspace()
    workspace.set_text(tao, TAO_INIT)

    analyzed = workspace.analyze(tao)
    assert analyzed.error is None
    assert list(lsp.iter_diagnostics(analyzed)) == []


def test_files_parse_skips_namelist_top_level(tmp_path: pathlib.Path) -> None:
    """
    The lattice parser skips a namelist file rather than raising on it.
    """
    from latform.parser import MemoryFiles

    tao = (tmp_path / "tao.init").resolve()
    tao.write_text("&tao_start\n  foo = 1\n/\n")
    files = MemoryFiles(top_files=[tao], initial_contents={tao: tao.read_text()})
    files.parse(raise_if_missing=False)  # must not raise
    files.annotate()
    assert files.by_filename[tao] == []


def test_create_server() -> None:
    pytest.importorskip("pygls")
    server = lsp.create_server()
    assert server is not None


def test_server_advertises_full_sync() -> None:
    """
    Full-document sync must be advertised.  Under pygls's default (incremental)
    the change handler receives per-edit deltas it does not apply, so live edits
    would silently fail to update diagnostics.
    """
    lsp_types = pytest.importorskip("lsprotocol.types")
    server = lsp.create_server()
    # pygls reads this (private) field when building server capabilities.
    assert server._text_document_sync_kind == lsp_types.TextDocumentSyncKind.Full


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def test_configure_logging_sets_level_and_avoids_stdout() -> None:
    import logging
    import sys

    level = lsp.configure_logging("debug")
    assert level == logging.DEBUG
    assert logging.getLogger("latform.lsp").getEffectiveLevel() == logging.DEBUG
    for handler in logging.getLogger("latform").handlers:
        assert getattr(handler, "stream", None) is not sys.stdout
    lsp.configure_logging("warning")  # restore


def test_configure_logging_to_file(tmp_path: pathlib.Path) -> None:
    import logging

    log_file = tmp_path / "lsp.log"
    lsp.configure_logging("debug", log_file)
    logging.getLogger("latform.lsp").debug("marker-line-xyz")
    for handler in logging.getLogger("latform").handlers:
        handler.flush()
    assert "marker-line-xyz" in log_file.read_text()
    lsp.configure_logging("warning")  # restore (closes file handler)


@pytest.mark.parametrize(
    ("argv", "expected_level", "expect_unknown"),
    [
        (["--stdio"], "warning", []),
        (["--log-level", "debug", "--stdio"], "debug", []),
        (["--clientProcessId=123", "--stdio"], "warning", ["--clientProcessId=123"]),
        ([], "warning", []),
    ],
)
def test_arg_parser_tolerates_client_flags(
    argv: list[str],
    expected_level: str,
    expect_unknown: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Client-injected flags such as ``--stdio`` must not crash the server.
    """
    monkeypatch.delenv("LATFORM_LSP_LOG_LEVEL", raising=False)
    monkeypatch.delenv("LATFORM_LSP_LOG_FILE", raising=False)
    args, unknown = lsp.build_arg_parser().parse_known_args(argv)
    assert args.log_level == expected_level
    assert unknown == expect_unknown


def test_client_log_handler_forwards_to_client() -> None:
    import logging

    lsp_types = pytest.importorskip("lsprotocol.types")

    class FakeServer:
        def __init__(self) -> None:
            self.messages: list[tuple] = []

        def show_message_log(self, message: str, msg_type) -> None:
            self.messages.append((msg_type, message))

    server = FakeServer()
    lsp._attach_client_log_handler(server, lsp_types, logging.DEBUG)
    try:
        logging.getLogger("latform.lsp").warning("hello-client")
        assert server.messages
        assert any("hello-client" in msg for _, msg in server.messages)
    finally:
        logging.getLogger("latform").handlers.clear()
        lsp.configure_logging("warning")


# --------------------------------------------------------------------------- #
# Workspace / project discovery
# --------------------------------------------------------------------------- #

MAIN_BMAD = "L_tot = 2\nq0: quad, L = 0.1\ncall, file = sub.bmad\n"
SUB_BMAD = "q1: q0\nfodo: line = (q1)\nuse, fodo\n"


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    A minimal two-file project: ``main.bmad`` defines ``q0`` and calls
    ``sub.bmad``, which references it.  Returns the project root.
    """
    (tmp_path / "latform.toml").write_text('top-level = ["main.bmad"]\n')
    (tmp_path / "main.bmad").write_text(MAIN_BMAD)
    (tmp_path / "sub.bmad").write_text(SUB_BMAD)
    return tmp_path


def test_standalone_reference_does_not_resolve(project: pathlib.Path) -> None:
    sub = project / "sub.bmad"
    analyzed = lsp.analyze(sub, SUB_BMAD)  # no config -> standalone
    assert analyzed.project_root is None
    assert lsp.resolve_definition(analyzed, 0, SUB_BMAD.index("q0")) is None


def test_project_mode_resolves_cross_file(project: pathlib.Path) -> None:
    sub = project / "sub.bmad"
    workspace = lsp.Workspace()
    workspace.set_text(sub, SUB_BMAD)

    analyzed = workspace.analyze(sub)
    assert analyzed.project_root == project.resolve()

    loc = lsp.resolve_definition(analyzed, 0, SUB_BMAD.index("q0"))
    assert loc is not None
    assert loc.filename.name == "main.bmad"
    assert loc.line == 1  # 'q0: quad' is the second line of main.bmad


def test_project_mode_clears_false_undefined(project: pathlib.Path) -> None:
    sub = project / "sub.bmad"
    workspace = lsp.Workspace()
    workspace.set_text(sub, SUB_BMAD)
    analyzed = workspace.analyze(sub)
    codes = {d.code for d in lsp.iter_diagnostics(analyzed)}
    assert "LF003" not in codes  # q0 is defined in main.bmad


def test_file_outside_project_tree_is_standalone(project: pathlib.Path) -> None:
    other = project / "other.bmad"
    other.write_text("z0: quad, L = 1\n")  # not reached from main.bmad
    workspace = lsp.Workspace()
    workspace.set_text(other, other.read_text())
    analyzed = workspace.analyze(other)
    assert analyzed.project_root is None


def test_workspace_overlay_uses_unsaved_buffer(project: pathlib.Path) -> None:
    """
    An unsaved edit to ``main.bmad`` is visible when analyzing ``sub.bmad``.
    """
    sub = project / "sub.bmad"
    main = project / "main.bmad"
    # sub only *references* s9 (in its line); s9 is defined solely in main's buffer.
    sub_text = "q1: q0\nfodo: line = (q1, s9)\nuse, fodo\n"

    workspace = lsp.Workspace()
    workspace.set_text(sub, sub_text)
    # Add a new element to main only in the buffer (not on disk).
    workspace.set_text(main, MAIN_BMAD + "s9: marker\n")

    analyzed = workspace.analyze(sub)
    loc = lsp.resolve_definition(analyzed, 1, sub_text.splitlines()[1].index("s9"))
    assert loc is not None
    assert loc.filename.name == "main.bmad"


def test_find_references_single_file(fodo: lsp.AnalyzedDocument) -> None:
    # q0 is defined on line 3 ("q0: quad") and referenced on lines 4 and 5.
    line, col = _pos("q0", 0)
    refs = lsp.find_references(fodo, line, col)
    ref_lines = sorted(loc.line for loc in refs)
    assert ref_lines == [3, 4, 5]

    without_decl = lsp.find_references(fodo, line, col, include_declaration=False)
    assert sorted(loc.line for loc in without_decl) == [4, 5]


def test_find_references_ignores_non_names(fodo: lsp.AnalyzedDocument) -> None:
    line, col = _pos("quad", 0)  # an element-type keyword, not a name
    assert lsp.find_references(fodo, line, col) == []


def test_find_references_cross_file(project: pathlib.Path) -> None:
    sub = project / "sub.bmad"
    workspace = lsp.Workspace()
    workspace.set_text(sub, SUB_BMAD)
    analyzed = workspace.analyze(sub)

    # Cursor on the q0 reference in sub.bmad; q0 is defined in main.bmad.
    refs = lsp.find_references(analyzed, 0, SUB_BMAD.index("q0"))
    by_file = sorted((loc.filename.name, loc.line) for loc in refs)
    assert ("main.bmad", 1) in by_file  # the definition
    assert ("sub.bmad", 0) in by_file  # the reference


def test_workspace_invalidate_picks_up_disk_change(project: pathlib.Path) -> None:
    sub = project / "sub.bmad"
    main = project / "main.bmad"
    workspace = lsp.Workspace()
    workspace.set_text(sub, SUB_BMAD)

    before = workspace.analyze(sub).files
    main.write_text(MAIN_BMAD + "q9: marker\n")  # change an unopened file on disk
    assert workspace.analyze(sub).files is before  # cached until invalidated

    workspace.invalidate()
    after = workspace.analyze(sub)
    assert after.files is not before
    assert "Q9" in after.files.get_named_items()


def test_workspace_caches_project_parse(project: pathlib.Path) -> None:
    sub = project / "sub.bmad"
    workspace = lsp.Workspace()
    workspace.set_text(sub, SUB_BMAD)
    first = workspace.analyze(sub).files
    second = workspace.analyze(sub).files
    assert first is second  # same parse reused until a buffer changes
    workspace.set_text(sub, SUB_BMAD + "\n")
    assert workspace.analyze(sub).files is not first
