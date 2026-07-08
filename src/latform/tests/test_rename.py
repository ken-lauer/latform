from __future__ import annotations

import pytest

from ..output import default_options, format_statements
from ..parser import MemoryFiles


def rename(src: str, renames: dict[str, str], **kwargs) -> str:
    """Parse ``src``, apply ``renames``, and return the reformatted text."""
    files = MemoryFiles.from_contents(src, "test.bmad")
    files.parse()
    files.annotate()
    files.rename(renames, **kwargs)
    (statements,) = files.by_filename.values()
    return format_statements(statements, default_options)


def test_rename_updates_definition_and_references():
    text = rename(
        "\n".join(
            [
                "q1: quad, l = 0.5",
                "ln: line = (q1, q1)",
            ]
        ),
        {"q1": "zz"},
    )
    assert text.lower().splitlines() == [
        "zz: quad, l=0.5",
        "ln: line = (zz, zz)",
    ]


def test_rename_leaves_other_names_untouched():
    text = rename("q1: quad\nq2: quad", {"q1": "zz"})
    assert "q2" in text.lower()
    assert "q1" not in text.lower()


def test_rename_is_case_insensitive_by_default():
    text = rename("q1: quad", {"Q1": "zz"})
    assert text.strip() == "ZZ: quad"


def test_rename_only_touches_name_role_by_default():
    text = rename("q1: quad, l = 0.5", {"l": "length"})
    assert "length" not in text.lower()
    assert "l=0.5" in text.lower()


def test_rename_all_roles_touches_keyword():
    text = rename("q1: quad", {"quad": "sextupole"}, only_name_role=False)
    assert text.strip() == "Q1: sextupole"


def test_rename_all_roles_touches_attribute_name():
    text = rename("q1: quad, l = 0.5", {"l": "length"}, only_name_role=False)
    assert "length=0.5" in text.lower()


def test_rename_regex_with_capture_group():
    text = rename(
        "\n".join(
            [
                "b1: sbend",
                "b2: sbend",
            ]
        ),
        {
            r"b(.*)": r"bend\1",
        },
    )
    assert text.splitlines() == [
        "BEND1: sbend",
        "BEND2: sbend",
    ]


def test_rename_no_match_is_noop():
    src = "q1: quad, l = 0.5"
    assert rename(src, {"does_not_exist": "zz"}) == rename(src, {})


@pytest.mark.parametrize("only_name_role", [True, False])
def test_rename_empty_mapping_is_noop(only_name_role: bool):
    src = "q1: quad\nln: line = (q1)"
    assert rename(src, {}, only_name_role=only_name_role) == rename(src, {})


def test_rename_parameter_target_and_expression_reference():
    # The element name appears as the parameter target and as a reference on the RHS.
    text = rename("K1: quad\nK2: quad\nK1[k2] = k1[k2] + 2", {"k1": "knew"})
    assert text.lower().splitlines() == [
        "knew: quad",
        "k2: quad",
        "knew[k2] = knew[k2] + 2",
    ]


def test_rename_leaves_bracketed_attribute_name_untouched():
    # `k2` is an attribute name -> no replace
    text = rename("K1: quad\nK1[k2] = k1[k2] + 2", {"k2": "znew"})
    assert "znew" not in text.lower()
    assert text.lower().count("k2") == 2


def test_rename_overlay_nested_targets():
    code = "\n".join(
        [
            "QUA2: quad",
            "ov: overlay = {QUA2[hkick]:kick, QUA2[b1]:x*kick}, var={kick}, kick=0",
        ]
    )
    text = rename(code, {"qua2": "magnet"})

    expected = [
        "MAGNET: quad",
        "OV: overlay = {MAGNET[hkick]:kick, MAGNET[b1]:x*kick}, var={kick}, kick=0",
    ]
    assert text.splitlines() == expected


def test_rename_line_with_repetition_and_reversal():
    code = "\n".join(
        [
            "a: quad",
            "b: quad",
            "c: quad",
            "l1: line = (a, 2*b, --c)",
        ]
    )
    text = rename(code, {"b": "bb", "c": "cc"})
    assert text.splitlines() == [
        "A: quad",
        "BB: quad",
        "CC: quad",
        "L1: line = (A, 2*BB, --CC)",
    ]


def test_rename_bracketed_reference_always_applies():
    # `BX_QUA1[k1]` is structurally a name (the bracket proves it), so it is
    # renamed even in strict mode and even though BX_QUA1 is not defined here.
    src = "B0_QUA1[k1] = BX_QUA1[k1]*3"
    for assume_defined in (True, False):
        text = rename(src, {"BX_QUA1": "B0_QUA1"}, assume_defined=assume_defined)
        assert text.strip() == "B0_QUA1[k1] = B0_QUA1[k1]*3"


def test_rename_bare_token_applies_when_assuming_defined():
    # `bar` has no role and no `[attr]`; under assume_defined it is treated as a
    # name and renamed on an exact match.
    text = rename("foo = bar + 3", {"bar": "bbb"})
    assert text.strip() == "FOO = bbb + 3"


def test_rename_bare_token_untouched_when_strict():
    text = rename("foo = bar + 3", {"bar": "bbb"}, assume_defined=False)
    assert text.strip() == "FOO = bar + 3"


def test_rename_regex_does_not_touch_bare_tokens():
    # Regex renames only apply to name-role tokens, so a broad pattern cannot
    # rewrite an unannotated token even when assuming references are defined.
    text = rename("foo = bar + 3", {r"ba.*": "X"})
    assert text.strip() == "FOO = bar + 3"
