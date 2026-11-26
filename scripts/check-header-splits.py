#!/usr/bin/env python3
"""Check that narrow example surfaces do not pull optional implementations back in.

Distillation follows names even in discarded template arms, so checking only
whether a particular instantiation compiles cannot enforce these boundaries.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDES = re.compile(r'^\s*#\s*include\s*([<"])([^>"\n]+)[>"]', re.MULTILINE)
COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


class SplitError(ValueError):
    """A narrow include or name boundary was violated."""


def sources(path, seen):
    path = path.resolve()
    if path in seen:
        return
    content = COMMENTS.sub("", path.read_text(encoding="utf-8"))
    seen[path] = content
    for delimiter, header in INCLUDES.findall(content):
        candidate = path.parent / header if delimiter == '"' else ROOT / "include" / header
        if not candidate.is_file():
            candidate = ROOT / "include" / header
        if candidate.is_file() and candidate.resolve().is_relative_to(ROOT):
            sources(candidate, seen)
        elif header.startswith("atl/"):
            raise SplitError(f"{path.relative_to(ROOT)}: missing library include {header}")


def uses_name(content, name):
    # This is deliberately a lexical check, not a C++ parser. Require an
    # expression context: a bare forward declaration/definition is insufficient.
    # Qualified calls, return expressions, initializers and argument uses cover
    # the implementation routes guarded below.
    return bool(re.search(r"(?:\breturn\s+|[=({,.;]\s*|::\s*)"
                          + r"\b" + re.escape(name) + r"\b", content))


def check(root, forbidden_headers=(), forbidden_names=(), required_names=()):
    seen = {}
    sources(ROOT / root, seen)
    included = {path.relative_to(ROOT / "include").as_posix() for path in seen
                if path.is_relative_to(ROOT / "include")}
    for header in forbidden_headers:
        if header in included:
            raise SplitError(f"{root} includes {header}")
    for name in required_names:
        if not any(uses_name(content, name) for content in seen.values()):
            raise SplitError(f"{root} lost use of {name}")
    for path, content in seen.items():
        for name in forbidden_names:
            if re.search(r"\b" + re.escape(name) + r"\b", content):
                raise SplitError(f"{root}: {path.relative_to(ROOT)} spells {name}")


def main():
    arithmetic_headers = (
        "atl/nt/mod_int_io.hpp", "atl/nt/detail/uint128_stream.hpp",
        "atl/nt/sieve.hpp", "atl/ds/static_vector.hpp",
        "atl/nt/power_series_wide.hpp", "atl/nt/convolution_uint64.hpp",
        "atl/nt/detail/series_composition_brent_kung.hpp",
        "atl/nt/ntt_engine.hpp", "atl/nt/prime_mod_int.hpp",
    )
    arithmetic_names = (
        "write_uint128", "read_stream_magnitude", "convolution_uint64",
        "composition_brent_kung", "composition_block_size",
    )
    for folder, files in {
        "linear-recurrence": ("quick",),
        "polynomial-points": ("quick", "sampling"),
        "polynomial-gcd": ("quick", "pade"),
        "power-series": ("quick", "powers-and-roots", "prime-field-helpers"),
        "online-convolution": ("quick", "partitions"),
        "series-composition": ("quick",),
    }.items():
        for file in files:
            check(f"docs/examples/{folder}/{file}.cpp", arithmetic_headers, arithmetic_names,
                  ("batch_inverse", "square_root_mod") if file == "prime-field-helpers" else ("fft_split_convolution",))
    check("docs/examples/series-composition/quick.cpp",
          required_names=("composition_kinoshita_li",))
    check("docs/examples/linear-recurrence/quick.cpp",
          ("atl/nt/matrix.hpp", "atl/nt/gaussian_elimination.hpp"))
    hash_headers = (
        "atl/ds/flat_hash_map_core.hpp", "atl/ds/detail/hash_table_core.hpp",
        "atl/utils/hash_core.hpp", "atl/utils/hash_extensions.hpp",
        "atl/ds/detail/heterogeneous_key.hpp",
    )
    hash_names = (
        "flat_hash_map", "hash_table", "identity_hash", "hashable_integer",
        "hash_float_value", "hash_bytes", "hash_combine", "byte_hashable",
        "nothrow_hashable", "packable_hash_pair", "tuple_hash_argument_t", "tuple_hash_key_t",
        "hashed_heterogeneous_key_for", "hash_word", "hash_packed_elements",
        "forest_edge_hash",
    )
    for folder, file in (("top-tree", "basic"), ("subtree-link-cut-tree", "basic"),
                         ("link-cut-tree", "basic"), ("point-location", "quick"),
                         ("polygon-overlay", "quick"), ("euler-tour-tree", "basic")):
        check(f"docs/examples/{folder}/{file}.cpp", hash_headers, hash_names)
    cluster_names = {
        "sum": ("sum", "sum_aggregate", "sum_path", "sum_tag", "sum_top_tree"),
        "extremes": ("extremes", "extremes_aggregate", "extremes_path", "extremes_tag", "extremes_top_tree"),
        "diameter": ("diameter", "diameter_value", "diameter_path", "diameter_point",
                     "diameter_span", "farthest_vertex", "diameter_top_tree"),
        "marked_distance": ("marked_distance", "marked_value", "marked_path", "marked_point",
                            "marked_distance_top_tree"),
    }
    for file, used in {"basic": ("extremes", "diameter"), "path": ("sum",),
                       "subtree": ("sum",), "diameter": ("diameter",)}.items():
        excluded = tuple(cluster for cluster in cluster_names if cluster not in used)
        # `sum` is also a field of the extremes algebra and of unrelated generic
        # aggregates. Qualify the set name and guard concrete types and aliases.
        names = tuple("top_clusters::sum" if name == "sum" else name
                      for cluster in excluded for name in cluster_names[cluster])
        check(f"docs/examples/top-tree/{file}.cpp",
              ("atl/ds/top_clusters.hpp",) + tuple(f"atl/ds/top_clusters/{cluster}.hpp" for cluster in excluded),
              names)
        check(f"docs/examples/top-tree/{file}.cpp", hash_headers, hash_names)
    check("include/atl/ds/top_tree_engine.hpp",
          ("atl/ds/top_clusters.hpp",) + tuple(f"atl/ds/top_clusters/{cluster}.hpp" for cluster in cluster_names),
          tuple("top_clusters::sum" if name == "sum" else name
                for names in cluster_names.values() for name in names))
    for folder in ("point-location", "polygon-overlay"):
        check(f"docs/examples/{folder}/quick.cpp",
              ("atl/utils/hash_extensions.hpp", "atl/geometry/predicates_floating.hpp"),
              ("hash_float_value", "hash_bytes", "hash_combine", "byte_hashable",
               "nothrow_hashable", "packable_hash_pair", "tuple_hash_argument_t", "tuple_hash_key_t",
               "adaptive_orientation", "adaptive_cross_sign"))
    check("docs/examples/link-cut-tree/basic.cpp",
          ("atl/ds/detail/forest_edge_registry.hpp", "atl/ds/detail/hash_table.hpp",
           "atl/ds/detail/hash_table_core.hpp",
           "atl/utils/hash_extensions.hpp"),
          ("forest_edge_hash", "hash_float_value", "hash_bytes"),
          ("initial_hash_seed", "hash_seed_value"))
    print("Narrow header include and name boundaries verified.")


if __name__ == "__main__":
    import sys
    try:
        main()
    except (SplitError, OSError) as error:
        print(f"Header split check failed: {error}", file=sys.stderr)
        sys.exit(1)
