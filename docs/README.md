# Guide to atl {#atl_guides}

`atl` is a header-only C++23 library. Include a public component with
`#include <atl/…>`; the generated API reference describes its contracts and
constraints. The handwritten guides explain semantics, tradeoffs, examples
and measurements.

The site's **Related Pages** tree discovers all guide pages in this checkout.
The generated **Guide coverage** page lists every public header and the guides
that discuss it, with a complete guide index. Both grow automatically when
components and their pages arrive.

See [Library conventions](conventions.md) for contribution rules, including
where to put guide prose, executable examples and API comments.

- [128-bit integers](int128.md): native aliases, portable software integers and formatting.

| Guide | Components |
|---|---|
| [Flat hash containers](flat-hash.md) | `flat_hash_set`, `flat_hash_map` |
| [Static vector](static-vector.md) | `static_vector` |
| [Hierarchical bitset](hierarchical-bitset.md) | `hierarchical_bitset` |
| [Coordinate compression](coordinate-compression.md) | `coordinate_compression`, `compress` |
| [Disjoint sets](dsu.md) | `dsu`, `rollback_dsu`, `persistent_dsu` |
| [Weighted disjoint sets](weighted-dsu.md) | `weighted_dsu` |
| [Persistent arrays](persistent-array.md) | `persistent_array`, `semi_persistent_array` |
| [Time divide and conquer](time-divide-and-conquer.md) | `time_divide_and_conquer` |
