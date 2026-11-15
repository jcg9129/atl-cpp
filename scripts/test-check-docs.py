#!/usr/bin/env python3
"""Regression checks for discovery and diagnostics without a header allowlist."""
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location("check_docs", Path(__file__).with_name("check-docs.py"))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class CoverageTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="atl-docs-check-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.write("README.md", "# Main\n[guide](docs/guide.md)\n")
        self.write("include/atl/future/widget.hpp", "// Public header\n")
        self.write("include/atl/future/detail/helper.hpp", "// Internal header\n")
        self.write("experimental/atl/experimental/choice.hpp", "// Prototype\n")
        self.write("docs/guide.md", "# Guide\n#include <atl/future/widget.hpp>\n"
                   "`atl/experimental/choice.hpp`\n[main](../README.md)\n")

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def check(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            status = checker.check(self.root, self.root / "generated/coverage.md")
        return status, output.getvalue()

    def test_new_folder_and_detail_exclusion(self):
        self.assertEqual(self.check()[0], 0)
        table = (self.root / "generated/coverage.md").read_text()
        self.assertIn("atl/future/widget.hpp", table)
        self.assertIn("docs/guide.md", table)
        self.assertNotIn("helper.hpp", table)

    def test_relative_header_link_supplies_coverage(self):
        self.write("docs/guide.md", "# Guide\n[header](../include/atl/future/widget.hpp)\n"
                   "`experimental/atl/experimental/choice.hpp`\n")
        self.assertEqual(self.check()[0], 0)

    def test_new_header_gap_is_listed(self):
        self.write("include/atl/another/component.hpp", "")
        status, log = self.check()
        self.assertEqual(status, 1)
        self.assertIn("Uncovered public header: atl/another/component.hpp", log)

    def test_all_missing_targets_are_reported(self):
        with (self.root / "docs/guide.md").open("a") as guide:
            guide.write('[page](absent.md)\n[header](../include/atl/absent.hpp)\n'
                        '[ref]: another.md\n<a href="third.md">third</a>\n'
                        '#include <atl/future/absent.hpp>\n')
        status, log = self.check()
        self.assertEqual(status, 1)
        for target in ["absent.md", "absent.hpp", "another.md", "third.md", "atl/future/absent.hpp"]:
            self.assertIn(target, log)

    def test_readme_does_not_supply_guide_coverage(self):
        self.write("docs/guide.md", "# Guide\n")
        self.write("README.md", "# Main\n#include <atl/future/widget.hpp>\n")
        self.assertIn("Uncovered public header: atl/future/widget.hpp", self.check()[1])


if __name__ == "__main__":
    unittest.main()
