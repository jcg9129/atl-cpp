#!/usr/bin/env python3
"""Fixtures for the include/name guard; also run with python -O."""
import importlib.util
import unittest
import sys

sys.dont_write_bytecode = True
from pathlib import Path

spec = importlib.util.spec_from_file_location('splits', Path(__file__).with_name('check-header-splits.py'))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.ROOT = Path(__file__).resolve().parent / 'fixtures/header-splits'


class SplitCheckerTests(unittest.TestCase):
    def test_include_styles(self):
        for name in ('angle.cpp', 'quoted.cpp', 'relative.cpp'):
            with self.subTest(name=name), self.assertRaisesRegex(checker.SplitError, 'includes atl/forbidden.hpp'):
                checker.check(name, forbidden_headers=('atl/forbidden.hpp',))

    def test_names_in_included_files(self):
        with self.assertRaisesRegex(checker.SplitError, 'spells excluded_implementation'):
            checker.check('relative.cpp', forbidden_names=('excluded_implementation',))

    def test_declarations_are_insufficient(self):
        with self.assertRaisesRegex(checker.SplitError, 'lost use of required_route'):
            checker.check('declaration.cpp', required_names=('required_route',))

    def test_expression_use(self):
        checker.check('use.cpp', required_names=('required_route',))

    def test_missing_required_name(self):
        with self.assertRaisesRegex(checker.SplitError, 'lost use of missing'):
            checker.check('use.cpp', required_names=('missing',))

    def test_compatibility_comparison(self):
        spec = importlib.util.spec_from_file_location('compat', Path(__file__).with_name('check-header-compat.py'))
        compat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compat)
        before = {'atl/a.hpp': '#include "atl/b.hpp"', 'atl/b.hpp': ''}
        self.assertEqual(compat.compare(before, dict(before)), {})
        quoted = {'atl/a.hpp': '#include \"b.hpp\"', 'atl/b.hpp': ''}
        self.assertEqual(compat.compare(quoted, {'atl/a.hpp': '', 'atl/b.hpp': ''}), {'atl/a.hpp': ['atl/b.hpp']})
        self.assertEqual(compat.compare(before, {'atl/a.hpp': '', 'atl/b.hpp': ''}), {'atl/a.hpp': ['atl/b.hpp']})


if __name__ == '__main__':
    unittest.main()
