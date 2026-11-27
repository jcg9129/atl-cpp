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
                  'atl::online_convolution<C> product; (void)product.push(C(1), C(1)); }',
                  'atl/nt/power_series_native.hpp'))
    geometry_cases = (
        ('predicates_core', 'predicates', 'return atl::geometry::orientation(p,p,p);'),
        ('segment_core', 'segment', 'return atl::geometry::contains(atl::geometry::segment<double>{p,p},p);'),
        ('line_core', 'line', 'return int(atl::geometry::side_of(atl::geometry::line<double>{p,{1.,0.}},p));'),
        ('polygon_core', 'polygon', 'return int(atl::geometry::contains(points,p));'),
        ('convex_hull_core', 'convex_hull', '(void)atl::geometry::convex_hull(points);'),
        ('convex_layers_core', 'convex_layers', '(void)atl::geometry::convex_layers(points);'),
        ('calipers_core', 'calipers', '(void)atl::geometry::farthest_pair(points);'),
        ('circle_core', 'circle', 'return atl::geometry::exact{}.orientation(points[0],points[1],points[2]);'),
        ('ray_core', 'ray', '(void)atl::geometry::contains(atl::geometry::ray<double>{p,{1.,0.}},p);'),
        ('triangulation_core', 'triangulation', '(void)atl::geometry::triangulation_of(points);'),
        ('triangulation_integral', 'triangulation', '(void)atl::geometry::triangulation_of(points);'),
        ('convex_polygon_operations_core', 'convex_polygon_operations', '(void)atl::geometry::intersection_of_convex_polygons(points,points);'),
        ('convex_polygon_contact_core', 'convex_polygon_contact',
         'auto d=atl::geometry::detail::difference_of(points,points,atl::geometry::exact{}); (void)atl::geometry::detail::overlap_of(d,atl::geometry::exact{});'),
        ('halfplane_core', 'halfplane', '(void)atl::geometry::halfplane<double>::left_of(p,points[1]);'),
        ('halfplane_intersection_core', 'halfplane_intersection',
         'std::array h{atl::geometry::halfplane<double>::at_most(1,0,1)}; (void)atl::geometry::halfplane_intersection(h);'),
    )
    for core, full, operation in geometry_cases:
        # Most policy layers fail in the shared exact-predicate hook, whose
        # diagnostic names predicates.hpp. The specialized kernels name their
        # owning umbrella. Accept only that documented diagnostic.
        diagnostic = ('halfplane_intersection' if core == 'halfplane_intersection_core' else
                      'halfplane' if core == 'halfplane_core' else 'predicates')
        cases.append((f'atl/geometry/{core}.hpp', f'atl/geometry/{full}.hpp',
                      '#include <array>\nint main() { '
                      'atl::geometry::point<double> p{0.,0.}; '
                      'std::array points{p,atl::geometry::point{1.,0.},atl::geometry::point{0.,1.}}; '
                      + operation + ' }', f'atl/geometry/{diagnostic}.hpp'))
    with tempfile.TemporaryDirectory(dir=args.build_dir) as directory:
        scratch = Path(directory)
        env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
        for index, case in enumerate(cases):
            core, extension, body = case[:3]
            diagnostic = case[3] if len(case) == 4 else extension
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
                if not full and (result.returncode == 0 or f'Include <{diagnostic}>' not in output):
                    parser.exit(1, f'Core case {index} did not give its compile-time include diagnostic:\n{output}')
    print(f'Header hooks: {len(cases)} core failures and {len(cases)} extension successes verified.')


if __name__ == '__main__':
    main()
