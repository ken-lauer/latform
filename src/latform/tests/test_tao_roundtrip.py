from __future__ import annotations

import pathlib
import tempfile

import pytest

from ..output import format_file
from ..types import FormatOptions
from .conftest import LATTICE_FILES

try:
    from pytao import SubprocessTao as Tao
except ImportError:
    pytest.skip("pytao is unavailable", allow_module_level=True)

lattice_param = pytest.mark.parametrize(
    ("lattice_fn",), [pytest.param(fn, id=fn.name) for fn in LATTICE_FILES]
)


def tao_save_lattice(tao: Tao) -> str:
    with tempfile.NamedTemporaryFile() as fp:
        tao.cmd(f'write bmad "{fp.name}"')
        fp.seek(0)
        return fp.read().decode()


def roundtrip_lattice(lattice_fn: pathlib.Path, options: FormatOptions):
    with Tao(lattice_file=lattice_fn, noplot=True) as tao:
        orig_lat = tao_save_lattice(tao)

    formatted = format_file(lattice_fn, options)
    with Tao.from_lattice_contents(formatted, noplot=True) as tao:
        formatted_lat = tao_save_lattice(tao)

    print("Original lattice, as written by Tao:")
    print(orig_lat)
    print("Formatted lattice, as written by Tao:")
    print(formatted_lat)

    assert len(orig_lat)
    assert orig_lat == formatted_lat


@lattice_param
def test_roundtrip_tao(lattice_fn: pathlib.Path):
    options = FormatOptions()
    roundtrip_lattice(lattice_fn, options)
