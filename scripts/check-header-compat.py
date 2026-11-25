#!/usr/bin/env python3
"""Preserve the transitive atl include surface of every header in a git baseline."""
import argparse
import re
import posixpath
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDES = re.compile(r'^\s*#\s*include\s*([<"])([^>"\n]+)[>"]', re.MULTILINE)


def closure(name, contents, seen=None):
    if seen is None:
        seen = set()
    if name not in seen:
        seen.add(name)
        for delimiter, header in INCLUDES.findall(contents[name]):
            if delimiter == '"':
                relative = posixpath.normpath(posixpath.join(posixpath.dirname(name), header))
                if relative in contents:
                    header = relative
            if header in contents:
                closure(header, contents, seen)
            elif header.startswith('atl/'):
                raise ValueError(f'{name}: missing library include {header}')
    return seen


def compare(before, after):
    return {name: sorted(closure(name, before) - closure(name, after))
            for name in before if closure(name, before) - closure(name, after)}


def baseline(ref):
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', ref, 'include/atl'], cwd=ROOT, text=True, encoding="utf-8").splitlines()
    return {name.removeprefix('include/'): subprocess.check_output(
        ['git', 'show', f'{ref}:{name}'], cwd=ROOT, text=True, encoding="utf-8")
        for name in names if name.endswith('.hpp')}


def current():
    return {p.relative_to(ROOT / 'include').as_posix(): p.read_text(encoding="utf-8") for p in (ROOT / 'include/atl').rglob('*.hpp')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', help='git baseline ref (default: main or origin/main)')
    args = parser.parse_args()
    ref = args.baseline
    if ref is None:
        for candidate in ('main', 'refs/remotes/origin/main'):
            if subprocess.run(['git', 'rev-parse', '--verify', '--quiet', candidate],
                              cwd=ROOT, stdout=subprocess.DEVNULL).returncode == 0:
                ref = candidate
                break
        if ref is None:
            parser.exit(1, 'Header compatibility requires main or origin/main; fetch the baseline or pass --baseline.\n')
    before, after = baseline(ref), current()
    missing = sorted(set(before) - set(after))
    if missing:
        parser.exit(1, 'Missing existing headers: ' + ', '.join(missing) + '\n')
    failures = compare(before, after)
    for header, lost in failures.items():
        print(f'{header} lost: {", ".join(lost)}')
    if failures:
        parser.exit(1, f'{len(failures)} header include surfaces regressed\n')
    print(f'Header compatibility: all {len(before)} baseline closures preserved.')


if __name__ == '__main__':
    main()
