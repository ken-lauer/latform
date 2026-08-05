"""
Tao ``*.init`` namelist support.

`file` models a ``tao.init`` (and its auxiliary namelist files); `schema`
provides type validation of namelist assignments against the generated
`latform.tao._schema`. The public names of both are re-exported here, so
``from latform.tao import TaoInit, resolve_path`` works.
"""

from __future__ import annotations

from .file import *  # noqa: F401,F403
from .file import __all__ as _file_all
from .rename import *  # noqa: F401,F403
from .rename import __all__ as _rename_all
from .schema import *  # noqa: F401,F403
from .schema import __all__ as _schema_all

__all__ = [*_file_all, *_rename_all, *_schema_all]
