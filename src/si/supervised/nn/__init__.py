# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Vítor Pereira
# Created Date: 01-09-2021
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Neural network package: the NN container, layers, activations and optimizers.

Note on the module names: the NN container lives in `network.py`, NOT in a file
called `nn.py`. A module may not share its name with the package that holds it.
When it did, `from .nn import NN` here bound the name `nn` on this package to the
inner module, and the `from .nn import *` in `si/supervised/__init__.py` then
copied that binding up one level -- so `si.supervised.nn` resolved to the small
inner module instead of this package. The failure was silent and confusing:

    from si.supervised.nn import Dense    # worked
    from si.supervised import nn
    nn.Dense                             # AttributeError

Both spellings now reach this package. `__all__` below is what keeps the fix
from quietly coming back.
"""
# ---------------------------------------------------------------------------

import types as _types

from .network import NN
from .layers import *
from .activation import *
from .cnn import *
from .optimizers import *
from .rnn import *
from .attention import *
from .transformer import *
from .language_model import *

# Export the classes and functions gathered above, but never the module objects
# that importing them leaves behind (`network`, `layers`, `numpy`, ...). Without
# this, `from .nn import *` one level up re-exports those module names into
# `si.supervised` and can shadow a package -- which is the bug described above.
#
# Computed rather than hand-listed on purpose: a newly added layer is exported
# automatically, so nobody has to remember to update a list here.
__all__ = sorted(
    name for name, value in list(globals().items())
    if not name.startswith('_') and not isinstance(value, _types.ModuleType)
)
