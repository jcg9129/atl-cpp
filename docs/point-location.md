# `atl::geometry::point_location`

Classify queries as arrangement vertices, edge interiors or open faces. Build
persistent slabs for repeated queries, or use `locations_of` for a single
batch with less allocated memory.

```cpp
#include <atl/geometry/point_location.hpp>
```

\include point-location/quick.cpp

Expected stdout:

\verbinclude point-location/quick.out

`<atl/geometry/point_location.hpp>` is the normal entry point for all coordinate
types. The example uses the reduced alternative
`<atl/geometry/point_location_integral.hpp>` without adaptive floating kernels.
To also use floating coordinates in that translation unit, include
`<atl/geometry/predicates_floating.hpp>` before the first instantiation or
constraint query involving floating predicates. Integral instantiations may
precede the extension.

## API

All names are in `atl::geometry`. `location_kind::{vertex, edge, face}` selects
the interpretation of `location{location_kind kind, uint32 index}`. Equality
is memberwise and constexpr. Indices refer to the supplied arrangement.

| Declaration | Meaning |
|---|---|
| `template<coordinate T> class point_location` | Persistent slab index |
| `template<segment_range R> requires segment_range<R const> && std::same_as<segment_coordinate_t<R>, T> point_location(arrangement const &arr, R const &segments)` | Build from arr and its exact input segments; deduction infers T |
| `location location_of(point<T> q) const` | Classify one query, vertices taking precedence over edges |
| `template<segment_range R, vertex_range Q> requires segment_range<R const> && vertex_range<Q const> && std::same_as<segment_coordinate_t<R>, vertex_coordinate_t<Q>> std::vector<location> locations_of(arrangement const &arr, R const &segments, Q const &queries)` | Results in query input order |

The constructor taking `arrangement const &&` is deleted to reject temporary
arrangements. Segment/query ranges are sized random-access ranges of the
[admitted coordinate types](geometry.md), readable through their const form.
Segments need not outlive the constructed index: it stores supporting lines.
The arrangement must outlive it and stay unchanged. An unbounded face result
uses `arrangement::unbounded_face`.

## Complexity

For n original segments, V vertices, E edges and Q queries: slab build takes
expected O(n + V + E log(E + 2)) time and storage. The n term remains when
many duplicate/point segments collapse to a tiny subdivision. One query takes
O(log(V + 1)) plus expected O(log(E + 1)) treap work. These are randomized
height bounds, not deterministic worst-case bounds. Treap updates use allocated
paths, so even an unlucky tree cannot cause recursion per element.
Batch mode takes O(n + (V + E + Q) log(V + E + Q + 2)) time and
O(n + V + E + Q) storage with an iterative status sweep. All query/build APIs are runtime-only.

## Preconditions

Asserted: fewer than `2^32` segments, vertices, queries and arena nodes;
int64 **segments and queries** satisfy `|x|, |y| < 2^32`. **int32 accepts
its full range**. The arrangement must have been built from exactly these
segments in this order, with valid half-edge invariants (documented only).

Integer boundary decisions are exact: vertex rationals are reconstructed
from incident supporting lines, rather than comparing rounded positions.
The degree-three factors and degree-five int256 comparisons obey the
[sweep bounds](segment-intersections.md); a query's line-side expression
has degree two. Float/double use rounded rational comparisons and best-effort
topology, with finite input and finite intermediates required. No tolerance
parameter repairs an inaccurately constructed subdivision. This family has
no exact floating-input tier; use the primitive polygon predicate when that
is the required guarantee.

## Measured performance

`benchmarks/geometry/point_location.cpp` races persistent slabs against
`atl::experimental::point_location_by_trapezoid_map` in
`atl/experimental/geometry/trapezoid_point_location.hpp`. The contender
randomly inserts edges into a trapezoidal decomposition and queries a search
DAG; symbolic shear handles verticals. It accepts integral coordinates only,
with the same bounds, lifetime and classification semantics. Expected build
O(n + V + E log(E + 2)), query O(log(E + 2) + log(V + 2)), storage O(n + V + E).
Its walks are iterative and its APIs runtime-only.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Operation / shape | library -O2 | by trapezoid map -O2 | library native | by trapezoid map native |
|---|---:|---:|---:|---:|
| build sparse, V=10k | 1.5 | 4.7 | 1.4 | 4.5 |
| build dense, V=60k E=118k | 300 | 472 | 312 | 446 |
| 1e5 queries, sparse | 35.1 | 72.2 | 39.3 | 64.9 |
| 1e5 queries, dense | 112 | 187 | 128 | 212 |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Batch mode | -O2 | native |
|---|---:|---:|
| 1e5 queries, sparse | 36.1 | 41.8 |
| 1e5 queries, dense | 136 | 143 |
| no queries, dense sweep | 71.9 | 76.4 |

Slabs beat the trapezoid map on every measured build/query shape. For a
dense single batch, `locations_of` takes 136/143 ms versus
300 + 112 = 412 ms plain and 312 + 128 = 440 ms native for building
and querying slabs separately. Batch timings
include frame construction; the no-query row measures that plus status updates.

The dense build and sparse/dense query comparisons include rows that vary
between runs (spreads up to 1.37x). The quoted minima and their sums do
not establish a stable single-batch speedup by themselves.

## Pitfalls and notes

A rounded crossing returned by an arrangement is not an exact query coordinate.
Queries have the same T as the original segments; integer queries cannot name
nonintegral vertices. Copying the index retains a reference to the same arr.
No ordering or id relationship between different arrangements is guaranteed.

## How it works

Sarnak and Tarjan (1986, Planar point location using persistent search trees)
organize vertical slabs between consecutive vertex x values. This implementation
keeps one root per slab in a path-copying randomized treap. Supporting-line
comparisons reconstruct exact rational boundaries; vertical edges and vertices
have separate y-sorted boundary catalogs. Coexisting arrangement edges cannot
cross, so their order at the later left boundary is a stateless pair order.
A descent finds the topmost edge below a query, whose forward half-edge names
the containing face. Batch mode buckets queries by slab and erases leaving
edges through saved insertion iterators, avoiding a redundant tree search.

## See also

[Arrangements](arrangement.md), [Polygon overlay](polygon-overlay.md),
[Polygons](polygon.md), [Geometry overview](geometry.md).

The integral include path also exposes `<atl/geometry/angular_sort_core.hpp>`,
`<atl/geometry/segment_intersections_core.hpp>` and
`<atl/geometry/arrangement_core.hpp>`. These policy-based layers share the
same definitions as their general umbrellas; include the floating predicate
extension before using exact floating predicates.

Exact floating predicates through a reduced header require
`<atl/geometry/predicates.hpp>` before their first instantiation; missing
floating kernels fail at compile time with an include diagnostic.
