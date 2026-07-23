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


HOVER_DOC = (
    "q0: quad, k1 = 0.5, L = 1\nq0[tilt] = 0.1\nc = pi * sqrt(2)\nparameter[geometry] = closed\n"
)


@pytest.fixture
def hover_doc(tmp_path: pathlib.Path) -> lsp.AnalyzedDocument:
    return lsp.analyze(tmp_path / "h.bmad", HOVER_DOC)


def _hover(analyzed: lsp.AnalyzedDocument, needle: str, occurrence: int = 0) -> str:
    lines = HOVER_DOC.splitlines()
    seen = -1
    for line_idx, line in enumerate(lines):
        col = line.find(needle)
        while col != -1:
            seen += 1
            if seen == occurrence:
                return lsp.hover_text(analyzed, line_idx, col) or ""
            col = line.find(needle, col + 1)
    raise AssertionError(f"{needle!r} occurrence {occurrence} not found")


def test_hover_attribute_in_definition(hover_doc: lsp.AnalyzedDocument) -> None:
    hover = _hover(hover_doc, "k1")
    assert "K1" in hover
    assert "QUADRUPOLE" in hover
    assert "1/m^2" in hover  # units


def test_hover_attribute_in_brackets(hover_doc: lsp.AnalyzedDocument) -> None:
    hover = _hover(hover_doc, "tilt")
    assert "TILT" in hover and "attribute" in hover


def test_hover_length_attribute(hover_doc: lsp.AnalyzedDocument) -> None:
    hover = _hover(hover_doc, "L = 1")  # the standalone length attribute
    assert "**L**" in hover
    assert "QUADRUPOLE" in hover


def test_hover_builtin_function(hover_doc: lsp.AnalyzedDocument) -> None:
    hover = _hover(hover_doc, "sqrt")
    assert "sqrt(x)" in hover
    assert "function" in hover
    assert "Square Root" in hover


def test_hover_builtin_constant(hover_doc: lsp.AnalyzedDocument) -> None:
    hover = _hover(hover_doc, "pi")
    assert "builtin constant" in hover
    assert "3.14159" in hover


def test_hover_parameter_attribute(hover_doc: lsp.AnalyzedDocument) -> None:
    hover = _hover(hover_doc, "geometry")
    assert "geometry" in hover
    assert "Open or closed" in hover  # from Parameter.known metadata


def test_hover_element_type_keyword(hover_doc: lsp.AnalyzedDocument) -> None:
    assert "element type" in _hover(hover_doc, "quad")


def test_hover_attribute_in_simple_statement(tmp_path: pathlib.Path) -> None:
    # ``ele[attr]`` with no value parses as a Simple statement, not a Parameter;
    # hover still resolves the attribute from the bracket context.
    doc = "q0: quad\nq0[k1]\n"
    analyzed = lsp.analyze(tmp_path / "s.bmad", doc)
    hover = lsp.hover_text(analyzed, 1, doc.splitlines()[1].index("k1"), doc) or ""
    assert "K1" in hover and "QUADRUPOLE" in hover


def test_hover_attribute_when_document_unparsed(tmp_path: pathlib.Path) -> None:
    # An incomplete ``ele[attr] = `` fails to parse; the element type is still
    # recovered by scanning the buffer text.
    doc = "q0: quad\nq0[tilt] = \n"
    analyzed = lsp.analyze(tmp_path / "e.bmad", doc)
    assert analyzed.files is None  # the document did not parse
    hover = lsp.hover_text(analyzed, 1, doc.splitlines()[1].index("tilt"), doc) or ""
    assert "TILT" in hover


def test_hover_owner_still_resolves_in_simple_statement(tmp_path: pathlib.Path) -> None:
    doc = "q0: quad\nq0[k1]\n"
    analyzed = lsp.analyze(tmp_path / "s.bmad", doc)
    hover = lsp.hover_text(analyzed, 1, 0, doc) or ""  # cursor on the owner 'q0'
    assert "element" in hover


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


# --------------------------------------------------------------------------- #
# Completion
# --------------------------------------------------------------------------- #

COMPLETE_DOC = "L_tot = 2\nq0: quad, k1 = 0.5\nq1: q0\nfodo: line = (q0, q1)\n"


@pytest.fixture
def complete_doc(tmp_path: pathlib.Path) -> lsp.AnalyzedDocument:
    return lsp.analyze(tmp_path / "c.bmad", COMPLETE_DOC)


def _labels(
    analyzed: lsp.AnalyzedDocument, prefix: str, kind: str | None = None, doc: str = COMPLETE_DOC
) -> set[str]:
    return {c.label for c in lsp.complete(analyzed, prefix, doc) if kind is None or c.kind == kind}


# Labels are cased per the default FormatOptions (names upper; types, attributes
# and builtins lower; length ``l`` always ``L``).


def test_complete_element_types(complete_doc: lsp.AnalyzedDocument) -> None:
    labels = _labels(complete_doc, "qz: ")
    assert {"quadrupole", "sbend", "marker"} <= labels  # element types (kind-case: lower)
    assert "Q0" in labels  # existing element (name-case: upper), for inheritance


def test_complete_attributes_in_body(complete_doc: lsp.AnalyzedDocument) -> None:
    labels = _labels(complete_doc, "qz: quad, ", kind="attribute")
    assert {"k1", "tilt"} <= labels  # attribute-case: lower


def test_complete_special_cases_length_l(complete_doc: lsp.AnalyzedDocument) -> None:
    # ``l`` (length) is always rendered ``L``, like the formatter, even though
    # other attributes are lowercased.
    labels = _labels(complete_doc, "qz: quad, ", kind="attribute")
    assert "L" in labels
    assert "l" not in labels


def test_complete_attributes_in_brackets(complete_doc: lsp.AnalyzedDocument) -> None:
    assert "k1" in _labels(complete_doc, "q0[", kind="attribute")


def test_complete_attributes_follow_inheritance(complete_doc: lsp.AnalyzedDocument) -> None:
    # q1 inherits q0 (a quadrupole), so its attribute set is the quadrupole's.
    assert "k1" in _labels(complete_doc, "q1[k", kind="attribute")


def test_complete_line_contents(complete_doc: lsp.AnalyzedDocument) -> None:
    labels = _labels(complete_doc, "beam: line = (")
    assert {"Q0", "Q1", "FODO"} <= labels  # name-case: upper


def test_complete_value_context(complete_doc: lsp.AnalyzedDocument) -> None:
    labels = _labels(complete_doc, "qz: quad, k1 = ")
    assert "sqrt" in labels  # intrinsic function (builtin-case: lower)
    assert "L_TOT" in labels  # defined constant (name-case: upper)


def test_complete_respects_project_case(tmp_path: pathlib.Path) -> None:
    """
    Completion labels follow the project's ``[format]`` case settings.
    """
    (tmp_path / "latform.toml").write_text('[format]\nname-case = "lower"\nkind-case = "upper"\n')
    doc = "q0: quad, k1 = 0.5\n"
    path = tmp_path / "a.bmad"
    path.write_text(doc)
    workspace = lsp.Workspace()
    workspace.set_text(path, doc)
    analyzed = workspace.analyze(path)

    types = {c.label for c in lsp.complete(analyzed, "zz: ", doc) if c.kind == "type"}
    names = {c.label for c in lsp.complete(analyzed, "zz: ", doc) if c.kind == "element"}
    assert "QUADRUPOLE" in types  # kind-case: upper
    assert "q0" in names  # name-case: lower


def test_complete_after_comma_space(complete_doc: lsp.AnalyzedDocument) -> None:
    # The canonical spacing is ``, `` — attribute completion must be available
    # once the space is typed, not only immediately after the comma.
    assert _labels(complete_doc, "qz: quad,", kind="attribute")
    assert _labels(complete_doc, "qz: quad, ", kind="attribute")


def test_complete_suppressed_in_comment(complete_doc: lsp.AnalyzedDocument) -> None:
    assert lsp.complete(complete_doc, "qz: quad ! a comment ", COMPLETE_DOC) == []


def test_complete_attribute_in_bracket_midedit(tmp_path: pathlib.Path) -> None:
    # ``ele[attr`` mid-edit completes attributes even though the line has not
    # parsed as a Parameter (the element type is resolved from the buffer text).
    doc = "q0: quad, k1=1\nq0[k\n"
    analyzed = lsp.analyze(tmp_path / "m.bmad", doc)
    labels = {c.label for c in lsp.complete(analyzed, "q0[k", doc) if c.kind == "attribute"}
    assert "k1" in labels


def test_complete_robust_without_parse(tmp_path: pathlib.Path) -> None:
    """
    Completions are available even when the document does not parse (mid-edit):
    element types and attributes come from the buffer text, not the AST.
    """
    doc = "q0: quad, k1 = 0.5\n"
    analyzed = lsp.AnalyzedDocument(path=tmp_path / "b.bmad", files=None)  # parse failed
    assert "quadrupole" in _labels(analyzed, "zz: ", doc=doc)
    assert "k1" in _labels(analyzed, "zz: quad, ", kind="attribute", doc=doc)
    assert "k1" in _labels(analyzed, "q0[", kind="attribute", doc=doc)  # type scanned from text


# --------------------------------------------------------------------------- #
# Rename & formatting
# --------------------------------------------------------------------------- #

RENAME_DOC = "q0: quad, k1 = 1\nll: line = (q0, q0)\nuse, ll\n"


@pytest.fixture
def rename_doc(tmp_path: pathlib.Path) -> lsp.AnalyzedDocument:
    return lsp.analyze(tmp_path / "r.bmad", RENAME_DOC)


@pytest.mark.parametrize(
    ("needle", "line", "renameable"),
    [
        ("q0", 0, True),  # element definition name
        ("ll", 1, True),  # line name
        ("q0", 1, True),  # a reference
        ("quad", 0, False),  # element-type keyword
        ("k1", 0, False),  # attribute name
        ("line", 1, False),  # the 'line' keyword
    ],
)
def test_prepare_rename(
    rename_doc: lsp.AnalyzedDocument, needle: str, line: int, renameable: bool
) -> None:
    col = RENAME_DOC.splitlines()[line].index(needle)
    assert (lsp.prepare_rename(rename_doc, line, col) is not None) == renameable


def test_rename_finds_all_occurrences(rename_doc: lsp.AnalyzedDocument) -> None:
    col = RENAME_DOC.splitlines()[0].index("q0")
    locs = lsp.find_references(rename_doc, 0, col, include_declaration=True)
    assert len(locs) == 3  # the definition plus two references in the line
    assert (0, 0) in [(loc.line, loc.column) for loc in locs]


def test_format_document_applies_case(tmp_path: pathlib.Path) -> None:
    (tmp_path / "latform.toml").write_text('[format]\nname-case = "upper"\nkind-case = "lower"\n')
    doc = "q0:quad,k1=1,l=2\n"
    path = tmp_path / "x.bmad"
    path.write_text(doc)
    workspace = lsp.Workspace()
    workspace.set_text(path, doc)
    assert lsp.format_document(workspace.analyze(path)) == "Q0: quad, k1=1, L=2\n"


def test_format_document_none_on_parse_error(tmp_path: pathlib.Path) -> None:
    analyzed = lsp.analyze(tmp_path / "b.bmad", "q0: quad, k1=(((")
    assert lsp.format_document(analyzed) is None


def test_format_range_subset(tmp_path: pathlib.Path) -> None:
    doc = "q0: quad\nq1: quad\nq2: quad\n"
    analyzed = lsp.analyze(tmp_path / "x.bmad", doc)
    result = lsp.format_range(analyzed, 1, 1)  # only the middle statement
    assert result is not None
    first, last, text = result
    assert (first, last) == (1, 1)
    assert "Q1" in text and "Q0" not in text


# --------------------------------------------------------------------------- #
# Semantic tokens
# --------------------------------------------------------------------------- #

SEMANTIC_DOC = "L_tot = 2\nq0: quad, k1 = sqrt(2)\nq1: q0\nll: line = (q0, undef)\nuse, ll\n"


def _semantic(
    tmp_path: pathlib.Path, doc: str = SEMANTIC_DOC
) -> dict[tuple[int, int], tuple[str, bool]]:
    """Map (line, col) -> (token-type-name, is_definition) for a document."""
    analyzed = lsp.analyze(tmp_path / "s.bmad", doc)
    return {
        (ln, col): (lsp.SEMANTIC_TOKEN_TYPES[ti], bool(mods))
        for ln, col, _length, ti, mods in lsp.semantic_tokens(analyzed)
    }


def test_semantic_tokens_classify_roles(tmp_path: pathlib.Path) -> None:
    tokens = _semantic(tmp_path)
    assert tokens[(0, 0)] == ("variable", True)  # L_tot constant definition
    assert tokens[(1, 0)] == ("class", True)  # q0 element definition
    assert tokens[(1, 4)] == ("type", False)  # quad (element type)
    assert tokens[(1, 10)] == ("property", False)  # k1 attribute
    assert tokens[(1, 15)] == ("function", False)  # sqrt builtin
    assert tokens[(2, 4)] == ("class", False)  # q0 reference -> a defined element
    assert tokens[(3, 0)] == ("namespace", True)  # ll line definition


def test_semantic_tokens_valid_names_stand_out(tmp_path: pathlib.Path) -> None:
    # A reference that resolves to an element is `class`; an unresolved name
    # stays a plain `variable`, so valid element names are visibly distinct.
    tokens = _semantic(tmp_path)
    assert tokens[(3, 12)] == ("class", False)  # q0 (defined) inside the line
    assert tokens[(3, 16)] == ("variable", False)  # undef (not defined)


def test_semantic_tokens_lengths_and_order(tmp_path: pathlib.Path) -> None:
    analyzed = lsp.analyze(tmp_path / "s.bmad", SEMANTIC_DOC)
    lines = SEMANTIC_DOC.splitlines()
    toks = lsp.semantic_tokens(analyzed)
    # Lengths cover exactly the token text.
    assert lines[0][0:6] == "L_tot "  # sanity on the doc
    for ln, col, length, _ti, _mods in toks:
        assert lines[ln][col : col + length].strip() == lines[ln][col : col + length]
    assert lines[1][15 : 15 + 4] == "sqrt"
    # Sorted by (line, col) so delta encoding is monotonic.
    assert toks == sorted(toks, key=lambda t: (t[0], t[1]))


def test_semantic_tokens_parameter_attr_not_definition(tmp_path: pathlib.Path) -> None:
    # k1 in `q1[k1] = 3` is an attribute (property), not a definition; q1 is the
    # element (class).
    tokens = _semantic(tmp_path, "q1: quad\nq1[k1] = 3\n")
    assert tokens[(1, 0)] == ("class", False)  # q1 (element reference)
    assert tokens[(1, 3)] == ("property", False)  # k1 attribute, not a definition


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

        def window_log_message(self, params) -> None:
            self.messages.append((params.type, params.message))

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


def test_workspace_reparses_only_changed_file(project: pathlib.Path) -> None:
    """
    Editing one file reuses cached statements for unchanged files; only the
    edited file is re-parsed (the cross-file annotation pass still re-runs).
    """
    main = project / "main.bmad"
    sub = project / "sub.bmad"
    workspace = lsp.Workspace()
    workspace.set_text(main, MAIN_BMAD)
    workspace.set_text(sub, SUB_BMAD)

    files = workspace.analyze(main).files
    sub_ids = [id(st) for st in files.by_filename[sub.resolve()]]
    main_ids = [id(st) for st in files.by_filename[main.resolve()]]

    workspace.set_text(main, MAIN_BMAD + "extra_const = 5\n")  # edit main only
    files2 = workspace.analyze(main).files
    assert [id(st) for st in files2.by_filename[sub.resolve()]] == sub_ids  # reused
    assert [id(st) for st in files2.by_filename[main.resolve()]] != main_ids  # re-parsed

    # Cross-file resolution and diagnostics remain correct after the edit.
    loc = lsp.resolve_definition(workspace.analyze(sub), 0, SUB_BMAD.index("q0"))
    assert loc is not None and loc.filename.name == "main.bmad"


def test_incremental_annotation_correctness(tmp_path: pathlib.Path) -> None:
    """
    Incremental annotation keeps cross-file resolution correct: a non-definition
    edit reuses unchanged files' annotation, while a definition change forces a
    full re-annotation so references update.
    """
    (tmp_path / "latform.toml").write_text('top-level = ["main.bmad"]\n')
    main = tmp_path / "main.bmad"
    sub = tmp_path / "sub.bmad"
    main.write_text("q0: quad, k1 = 1\ncall, file = sub.bmad\n")
    sub.write_text("q1: q0\nll: line = (q0, q1)\nuse, ll\n")

    workspace = lsp.Workspace()
    workspace.set_text(main, main.read_text())
    workspace.set_text(sub, sub.read_text())

    def q0_def_file() -> str | None:
        loc = lsp.resolve_definition(workspace.analyze(sub), 0, workspace.text_of(sub).index("q0"))
        return loc.filename.name if loc is not None else None

    assert q0_def_file() == "main.bmad"

    # Non-definition edit to main (sub is not re-parsed → its annotation is
    # reused); sub's reference to q0 must still resolve into main.
    workspace.set_text(main, "q0: quad, k1 = 99\ncall, file = sub.bmad\n")
    assert q0_def_file() == "main.bmad"

    # Renaming the definition changes the signature → full re-annotation.
    workspace.set_text(main, "q9: quad, k1 = 1\ncall, file = sub.bmad\n")
    named = workspace.analyze(sub).files.get_named_items()
    assert "Q9" in named and "Q0" not in named


def test_workspace_invalidate_clears_parse_cache(project: pathlib.Path) -> None:
    sub = project / "sub.bmad"
    workspace = lsp.Workspace()
    workspace.set_text(sub, SUB_BMAD)
    workspace.analyze(sub)
    assert workspace._parse_cache  # populated
    workspace.invalidate()
    assert not workspace._parse_cache  # cleared


def test_workspace_caches_project_parse(project: pathlib.Path) -> None:
    sub = project / "sub.bmad"
    workspace = lsp.Workspace()
    workspace.set_text(sub, SUB_BMAD)
    first = workspace.analyze(sub).files
    second = workspace.analyze(sub).files
    assert first is second  # same parse reused until a buffer changes
    workspace.set_text(sub, SUB_BMAD + "\n")
    assert workspace.analyze(sub).files is not first
