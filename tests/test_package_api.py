# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Tests for the shape of the import surface itself, rather than for any
# algorithm. They exist because a broken import surface fails SILENTLY: the
# library imports fine, some spellings work, and only an unlucky one raises
# AttributeError deep in a student's notebook.
#
# The specific bug guarded here: `si/supervised/nn/` (a package) once contained
# a module also called `nn.py`. `from .nn import NN` inside the package bound
# the name `nn` on the package to that inner module, and `from .nn import *` in
# `si/supervised/__init__.py` copied the binding up a level, leaving
# `si.supervised.nn` pointing at the inner module. Result:
#
#     from si.supervised.nn import Dense   # fine
#     from si.supervised import nn
#     nn.Dense                            # AttributeError
#
# The module is now `network.py`, and `nn/__init__.py` sets an `__all__` that
# excludes module objects so the re-export cannot reintroduce the shadowing.
# ---------------------------------------------------------------------------

import importlib
import os
import pkgutil
import sys
import types
import unittest


class TestNNPackageIsNotShadowed(unittest.TestCase):
    """`si.supervised.nn` must resolve to the package, by every spelling."""

    def test_attribute_access_reaches_the_package(self):
        # The spelling that used to break. `from X import Y` falls back to an
        # attribute lookup on X, which is exactly what the shadowing corrupted.
        from si.supervised import nn
        self.assertEqual(nn.__name__, 'si.supervised.nn')
        self.assertIs(nn, sys.modules['si.supervised.nn'])

    def test_import_as_reaches_the_package(self):
        # `import a.b as x` is also an attribute lookup on `a`, so it broke too.
        import si.supervised.nn as nn
        self.assertIs(nn, sys.modules['si.supervised.nn'])

    def test_layers_are_reachable_through_the_package_attribute(self):
        # The actual symptom: these raised AttributeError while the equivalent
        # `from si.supervised.nn import Dense` succeeded.
        from si.supervised import nn
        for name in ('NN', 'Dense', 'Conv2D', 'LayerNorm', 'Adam',
                     'SelfAttention', 'TransformerBlock'):
            with self.subTest(name=name):
                self.assertTrue(hasattr(nn, name),
                                f'si.supervised.nn.{name} is not reachable')

    def test_both_import_spellings_yield_the_same_class(self):
        from si.supervised import nn
        from si.supervised.nn import Dense
        self.assertIs(nn.Dense, Dense)

    def test_container_module_still_reachable_by_its_own_name(self):
        # Renaming nn.py -> network.py must not hide the module itself.
        network = importlib.import_module('si.supervised.nn.network')
        from si.supervised.nn import NN
        self.assertIs(network.NN, NN)


class TestExportHygiene(unittest.TestCase):
    """`__all__` is what stops the shadowing from silently returning."""

    def test_all_contains_no_module_objects(self):
        # A module object in __all__ is the mechanism of the original bug: it
        # lets a `import *` one level up rebind a package name.
        import si.supervised.nn as nn
        offenders = [name for name in nn.__all__
                     if isinstance(getattr(nn, name, None), types.ModuleType)]
        self.assertEqual(offenders, [],
                         f'__all__ re-exports module objects: {offenders}')

    def test_star_import_exports_classes_not_plumbing(self):
        namespace = {}
        exec('from si.supervised.nn import *', namespace)
        # `np`/`warnings` are implementation imports, not part of the API.
        for leaked in ('np', 'numpy', 'warnings', 'types', 'math'):
            with self.subTest(name=leaked):
                self.assertNotIn(leaked, namespace)
        # ...and the things students actually import are present.
        for wanted in ('NN', 'Dense', 'Adam', 'SelfAttention'):
            with self.subTest(name=wanted):
                self.assertIn(wanted, namespace)

    def test_si_supervised_does_not_export_nn_internals(self):
        # Before the fix, `from .nn import *` dragged `layers`, `activation`,
        # `cnn`, `optimizers`, `np` and `math` into si.supervised as well.
        import si.supervised as supervised
        for internal in ('layers', 'activation', 'cnn', 'optimizers',
                         'attention', 'transformer', 'np', 'math'):
            with self.subTest(name=internal):
                self.assertFalse(hasattr(supervised, internal),
                                 f'si.supervised leaks {internal}')


class TestUtilPackageIsNotShadowed(unittest.TestCase):
    """`si/util/util.py` was the same collision, one step from breaking.

    It never actually broke, purely because `si/__init__.py` happens not to
    star-import anything. Adding one `from .util import *` there would have
    rebound `si.util` to the inner module. The helpers now live in
    `helpers.py`, so the trap is gone rather than merely unsprung.
    """

    def test_attribute_access_reaches_the_package(self):
        from si import util
        self.assertEqual(util.__name__, 'si.util')
        self.assertIs(util, sys.modules['si.util'])

    def test_helpers_and_metrics_are_reachable(self):
        from si import util
        for name in ('train_test_split', 'minibatch', 'add_intercept',
                     'accuracy_score', 'mse', 'METRICS',
                     'CrossValidationScore', 'GridSearchCV'):
            with self.subTest(name=name):
                self.assertTrue(hasattr(util, name),
                                f'si.util.{name} is not reachable')

    def test_star_import_exports_functions_not_plumbing(self):
        namespace = {}
        exec('from si.util import *', namespace)
        for leaked in ('np', 'pd', 'numpy', 'pandas', 'types', 'util'):
            with self.subTest(name=leaked):
                self.assertNotIn(leaked, namespace)
        self.assertIn('train_test_split', namespace)

    def test_helpers_module_reachable_by_its_own_name(self):
        helpers = importlib.import_module('si.util.helpers')
        from si.util import train_test_split
        self.assertIs(helpers.train_test_split, train_test_split)


class TestNoModuleShadowsItsPackage(unittest.TestCase):
    """A general guard, so the next `foo/foo.py` is caught on arrival."""

    def test_no_package_contains_a_module_of_the_same_name(self):
        import si
        root = os.path.dirname(si.__file__)
        collisions = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != '__pycache__']
            if '__init__.py' not in filenames:
                continue
            package_name = os.path.basename(dirpath)
            if f'{package_name}.py' in filenames:
                collisions.append(os.path.join(dirpath, f'{package_name}.py'))
        self.assertEqual(collisions, [],
                         'a module may not share its package name: '
                         f'{collisions}')

    def test_every_submodule_imports(self):
        # Cheap insurance that the rename left no dangling `from .nn import`.
        # cvxopt-dependent svm is skipped rather than failing the suite.
        import si
        failures = []
        for info in pkgutil.walk_packages(si.__path__, prefix='si.'):
            if info.name.endswith('.svm'):
                continue
            try:
                importlib.import_module(info.name)
            except ImportError as error:      # pragma: no cover - diagnostic
                failures.append(f'{info.name}: {error}')
        self.assertEqual(failures, [])


if __name__ == '__main__':
    unittest.main()
