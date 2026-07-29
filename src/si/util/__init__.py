# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Utility package: general helpers, metrics, and model-selection tools.

The helpers live in `helpers.py`, not in a `util.py` inside this `util`
package. A module that shares its package's name is a trap: `from .util import
*` binds the name `util` on this package to the inner module, and any
star-import of this package one level up would then rebind `si.util` to that
inner module -- leaving `si.util.train_test_split` an AttributeError while
`from si.util import train_test_split` still worked. That is exactly what had
happened to `si.supervised.nn`; see the note in `si/supervised/nn/__init__.py`.

`__all__` below is the second half of the guard: it exports the functions and
classes gathered here without the module objects (`helpers`, `metrics`, `np`,
`pd`) that importing them leaves behind.
"""
# ---------------------------------------------------------------------------

import types as _types

from .helpers import *
from .metrics import *
from .cv import CrossValidationScore, GridSearchCV

__all__ = sorted(
    name for name, value in list(globals().items())
    if not name.startswith('_') and not isinstance(value, _types.ModuleType)
)
