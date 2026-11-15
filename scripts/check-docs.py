#!/usr/bin/env python3
"""Check guide/header coverage and local Markdown links; emit a Doxygen table.

A guide references a header by spelling atl/path.hpp (typically in an #include)
or by linking to include/atl/path.hpp. Only docs/*.md supplies coverage; README
is checked for broken links as the site's main page. No folder/page allowlists.
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

HEADER = re.compile(r"(?<!\w)(?:include/)?(atl/[\w./-]+\.(?:hpp|h))\b")
# Inline Markdown links/images, reference definitions, and HTML links.
LINK = re.compile(
    r"\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)"
    r"|^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)"
    r"|\b(?:href|src)=[\"']([^\"']+)[\"']", re.MULTILINE
)


def check(root, output):
    guides = sorted((root / "docs").glob("*.md"))
    headers = {
        p.relative_to(root / "include").as_posix(): []
        for p in sorted((root / "include/atl").rglob("*"))
        if p.suffix in {".hpp", ".h"} and "detail" not in p.relative_to(root / "include").parts
    }
    errors = []
    for page in [root / "README.md", *guides]:
        content = page.read_text(encoding="utf-8")
        label = page.relative_to(root).as_posix()
        for header in sorted(set(HEADER.findall(content))):
            base = root / ("experimental" if header.startswith("atl/experimental/") else "include")
            if not (base / header).is_file():
                errors.append(f"{label}: missing header {header}")
            elif page in guides and header in headers:
                headers[header].append(label)
        # Code fences contain sample syntax and shell commands, not site links.
        prose = re.sub(r"^(`{3,}|~{3,})[^\n]*\n.*?^\1\s*$", "", content,
                       flags=re.MULTILINE | re.DOTALL)
        for match in LINK.finditer(prose):
            target = next(group for group in match.groups() if group).strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            path = (page.parent / unquote(url.path)).resolve()
            if path.suffix in {".md", ".hpp", ".h"} and not path.is_file():
                errors.append(f"{label}: broken link {target}")
    for header, pages in headers.items():
        if not pages:
            errors.append(f"Uncovered public header: {header}")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Guide coverage {#atl_guide_coverage}", "",
                 "Generated from the public headers and guide pages present in this checkout.",
                 "Headers in `detail/` directories are implementation details and are excluded.",
                 "", "| Public header | Guide pages |", "|---|---|"]
        # \ref uses the source path: identical README basenames remain unambiguous.
        for header, pages in headers.items():
            links = ", ".join(f'\\ref {page} "{Path(page).stem}"' for page in pages)
            lines.append(f"| `{header}` | {links or '**Uncovered**'} |")
        lines.extend(["", "## Guide pages", ""])
        for page in guides:
            title = re.search(r"^#\s+(.+)", page.read_text(encoding="utf-8"), re.MULTILINE)
            title = title[1].replace('"', '') if title else page.stem
            explicit_id = re.search(r"\{#([\w-]+)\}", title)
            target = explicit_id[1] if explicit_id else page.relative_to(root).as_posix()
            title = re.sub(r"\s*\{#[\w-]+\}", "", title)
            lines.append(f'- \\ref {target} "{title}"')
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if errors:
        print("Documentation coverage/link check failed:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    print(f"Documentation coverage: {len(headers)} public headers, {len(guides)} guides; all links valid.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    return check(args.root.resolve(), args.output)


if __name__ == "__main__":
    sys.exit(main())
