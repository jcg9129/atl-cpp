#!/usr/bin/env python3
"""Fixtures for the include/name guard; also run with python -O."""
import importlib.util
import unittest
from unittest.mock import patch
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

    def test_forbidden_declarations_are_checked(self):
        # Definitions can survive distillation without a call (constructors and standard names).
        with self.assertRaisesRegex(checker.SplitError, 'spells required_route'):
            checker.check('declaration.cpp', forbidden_names=('required_route',))

    def test_expression_use(self):
        checker.check('use.cpp', required_names=('required_route',))

    def test_missing_required_name(self):
        with self.assertRaisesRegex(checker.SplitError, 'lost use of missing'):
            checker.check('use.cpp', required_names=('missing',))

    def test_discarded_arms_are_checked(self):
        with self.assertRaisesRegex(checker.SplitError, 'spells unused_path'):
            checker.check('discarded.cpp', forbidden_names=('unused_path',))

    def test_qualified_cluster_name(self):
        with self.assertRaisesRegex(checker.SplitError, 'spells top_clusters::sum'):
            checker.check('discarded.cpp', forbidden_names=('top_clusters::sum',))

    def test_comments_are_ignored(self):
        checker.check('comments.cpp', forbidden_headers=('atl/forbidden.hpp',),
                      forbidden_names=('unused_path',))

    def test_example_boundaries(self):
        with patch.object(checker, 'check') as check, patch('builtins.print'):
            checker.main()
        calls = check.call_args_list
        for name in ('quick', 'partitions'):
            root = f'docs/examples/online-convolution/{name}.cpp'
            self.assertTrue(any(call.args[0] == root and 'montgomery64' in call.args[2]
                                and 'convolution' in call.args[2] for call in calls))
            self.assertTrue(any(call.args[0] == root and 'convolution_ntt' in call.args[3]
                                for call in calls))
        self.assertTrue(any(call.args[0] == 'docs/examples/linear-recurrence/quick.cpp'
                            and 'atl/nt/convolution_ntt.hpp' in call.args[1]
                            and 'convolution' in call.args[3] for call in calls))
        self.assertTrue(any(call.args[0] == 'docs/examples/power-series/prime-field-helpers.cpp'
                            and 'fft_split_convolution' in call.args[2] and 'atl/nt/convolution_ntt.hpp' in call.args[1]
                            for call in calls))

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
