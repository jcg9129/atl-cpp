#!/usr/bin/env python3
"""Verify optional implementations fail during compilation, and work when included."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--compiler-id', required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    args = parser.parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for key in ('double', 'std::string', 'std::string_view', 'int*',
                'std::pair<int,int>', 'std::tuple<int,double>',
                'std::vector<int>', 'std::array<int,2>'):
        cases.append(('atl/utils/hash_core.hpp', 'atl/utils/hash_extensions.hpp',
                      f'using key = {key}; int main() {{ (void)atl::hash<key>{{}}(key{{}}); }}'))
    for operation in ('os << m', 'is >> m'):
        cases.append(('atl/nt/mod_int_core.hpp', 'atl/nt/mod_int_io.hpp',
                      'int main() { atl::mod_int32<17> m; std::ostream os(nullptr); '
                      f'std::istream is(nullptr); {operation}; }}'))
    cases.append(('atl/nt/power_series_ntt.hpp', 'atl/nt/power_series_native.hpp',
                  '#include <atl/nt/mod_int_static.hpp>\n'
                  'int main() { using C = atl::mod_int32<1000000007>; '
                  '(void)atl::series_inverse(std::vector<C>{1, -1}, 129); }'))
    cases.append(('atl/nt/online_convolution_ntt.hpp', 'atl/nt/online_convolution_native.hpp',
                  '#include <atl/nt/mod_int_static.hpp>\n'
                  'int main() { using C = atl::mod_int32<1000000007>; '
                  'atl::online_convolution<C> product; (void)product.push(C(1), C(1)); }'))
    with tempfile.TemporaryDirectory(dir=args.build_dir) as directory:
        scratch = Path(directory)
        env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
        for index, (core, extension, body) in enumerate(cases):
            for full in (False, True):
                source = scratch / f'case-{index}.cpp'
                source.write_text(f'#include <{core}>\n' +
                                  (f'#include <{extension}>\n' if full else '') + body)
                if args.compiler_id == 'MSVC':
                    options = ['/nologo', '/std:c++latest', '/Zs', '/I' + str(ROOT / 'include')]
                else:
                    options = ['-std=c++23', '-fsyntax-only', '-I' + str(ROOT / 'include')]
                result = subprocess.run([args.compiler, *options, str(source)], env=env,
                                        capture_output=True, text=True, timeout=60)
                output = result.stdout + result.stderr
                if full and result.returncode != 0:
                    parser.exit(1, f'Extension case {index} failed:\n{output}')
                diagnostic = 'atl/nt/power_series_native.hpp' if core == 'atl/nt/online_convolution_ntt.hpp' else extension
                if not full and (result.returncode == 0 or f'Include <{diagnostic}>' not in output):
                    parser.exit(1, f'Core case {index} did not give its compile-time include diagnostic:\n{output}')
    print(f'Header hooks: {len(cases)} core failures and {len(cases)} extension successes verified.')


if __name__ == '__main__':
    main()
