#!/usr/bin/env python3
"""Check that narrow example surfaces do not pull optional implementations back in.

Distillation follows names even in discarded template arms, so checking only
whether a particular instantiation compiles cannot enforce these boundaries.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDES = re.compile(r"^\s*#\s*include\s*<(atl/[^>]+)>", re.MULTILINE)
COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def sources(path, seen):
    if path in seen:
        return
    content = path.read_text(encoding="utf-8")
    seen[path] = COMMENTS.sub("", content)
    for header in INCLUDES.findall(content):
        sources(ROOT / "include" / header, seen)


def check(root, forbidden_headers=(), forbidden_names=(), required_names=()):
    seen = {}
    sources(ROOT / root, seen)
    included = {str(path.relative_to(ROOT / "include")) for path in seen
                if path.is_relative_to(ROOT / "include")}
    for header in forbidden_headers:
        assert header not in included, f"{root} includes {header}"
    for name in required_names:
        assert any(re.search(r"\b" + re.escape(name) + r"\b", content)
                   for content in seen.values()), f"{root} lost {name}"
    for path, content in seen.items():
        for name in forbidden_names:
            assert not re.search(r"\b" + re.escape(name) + r"\b", content), (
                f"{root}: {path.relative_to(ROOT)} spells {name}")


def main():
    arithmetic_headers = (
        "atl/nt/mod_int_io.hpp", "atl/nt/detail/uint128_stream.hpp",
        "atl/nt/sieve.hpp", "atl/ds/static_vector.hpp",
        "atl/nt/power_series_wide.hpp", "atl/nt/convolution_uint64.hpp",
        "atl/nt/series_composition_brent_kung.hpp",
    )
    arithmetic_names = (
        "write_uint128", "read_stream_magnitude", "convolution_uint64",
        "composition_brent_kung", "composition_block_size",
    )
    for folder, files in {
        "linear-recurrence": ("quick",),
        "polynomial-points": ("quick", "sampling"),
        "polynomial-gcd": ("quick", "pade"),
        "power-series": ("quick", "powers-and-roots"),
        "online-convolution": ("quick", "partitions"),
        "series-composition": ("quick",),
    }.items():
        for file in files:
            check(f"docs/examples/{folder}/{file}.cpp", arithmetic_headers, arithmetic_names,
                  ("fft_split_convolution",))
    check("docs/examples/series-composition/quick.cpp",
          required_names=("composition_kinoshita_li",))
    check("docs/examples/linear-recurrence/quick.cpp",
          ("atl/nt/matrix.hpp", "atl/nt/gaussian_elimination.hpp"))
    for folder in ("point-location", "polygon-overlay"):
        check(f"docs/examples/{folder}/quick.cpp",
              ("atl/utils/hash_extensions.hpp", "atl/geometry/predicates_floating.hpp"),
              ("hash_float_value", "hash_bytes", "adaptive_orientation", "adaptive_cross_sign"))
    check("docs/examples/link-cut-tree/basic.cpp",
          ("atl/ds/detail/forest_edge_registry.hpp", "atl/ds/detail/hash_table.hpp",
           "atl/utils/hash_extensions.hpp"),
          ("forest_edge_hash", "hash_float_value", "hash_bytes"),
          ("initial_hash_seed", "hash_seed_value"))
    print("Narrow header include and name boundaries verified.")


if __name__ == "__main__":
    main()
