# `atl::geometry` convex hulls and convex layers

Build the boundary of a point set, query a strict convex polygon, merge finished
hulls or peel a cloud into boundary layers. Hull vertices remain input points;
no floating construction is needed. For an arbitrary simple polygon use the
[polygon functions](polygon.md). Coordinate types and precision policies follow
the [geometry overview](geometry.md).

```cpp
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/convex_layers.hpp>
```

\include convex-hull/quick.cpp

Output:

\verbinclude convex-hull/quick.out

## Convex hull API

All functions are `[[nodiscard]] constexpr` in `atl::geometry`. `R`, `RA` and
`RB` model `vertex_range`; `T = vertex_coordinate_t<R>` (or of `RA`). A
`Policy` models `sign_policy<T>` and defaults to `exact`; binary functions
require the same coordinate type on both inputs. Indexed proxies convertible
to `point<T>` may be used. `vertex_index` is `uint32`.

| Signature | Return and conventions |
|---|---|
| `std::vector<vertex_index> convex_hull_indices(R &&points, Policy p = {})` | strict extreme vertices, counterclockwise from the lexicographically smallest point |
| `std::vector<vertex_index> convex_hull_indices(R &&points, keep_collinear_t, Policy p = {})` | every distinct boundary position, same traversal |
| `std::vector<point<T>> convex_hull(R &&points, Policy p = {})` | strict hull materialized as points |
| `std::vector<point<T>> convex_hull(R &&points, keep_collinear_t, Policy p = {})` | boundary hull materialized as points |
| `containment convex_contains(R &&hull, point<vertex_coordinate_t<R>> q, Policy p = {})` | `outside`, `boundary` or `inside` a nonempty strict CCW hull |
| `std::vector<point<T>> merge_convex_hulls(RA &&a, RB &&b, Policy p = {})` | strict hull of two finished hulls, in `convex_hull` form |
| `std::vector<point<T>> minkowski_sum(RA &&a_points, RB &&b_points, Policy p = {})` | strict CCW boundary of all pairwise sums, starting at the (y, x)-minimum |

`keep_collinear_t` is an explicit default-constructible tag; pass its constant
`keep_collinear`. Duplicates use raw `==` under every policy and keep the
smallest original input index. Empty input yields an empty hull; equal points
yield one vertex; collinear input yields two endpoints, or every distinct
point in lexicographic order with `keep_collinear`. The closing vertex is not
repeated. Ties among coordinates use lexicographic (x, y) order.

`merge_convex_hulls` accepts empty hulls, points and segments. `minkowski_sum`
accepts nonempty convex inputs of either winding, including duplicates and
straight vertices, normalized internally; point and segment summands are
legal. `convex_contains` accepts point and segment hulls, returning only
`boundary` or `outside` for those.

### Complexity and preconditions

Hull construction takes O(n log n) time and O(n) extra memory; containment
O(log h) time and O(1) memory. Merge takes O(h_a + h_b) time and output memory.
Separated integral hulls with the exact policy and at least 1024 vertices in
total use O(log h_a log h_b + h_out) time, excluding assertion checks.
Minkowski sum takes O(n + m) time and memory. The library algorithms use
constant call-stack space, apart from the standard sort's O(log n) depth.
Allocation failures propagate standard exceptions.

Asserted under `ATL_ASSERT`: hull inputs have fewer than 2^32 elements;
`convex_contains` has at least one vertex; merge inputs have distinct vertices,
strict CCW turns and increase lexicographically to their maximum then decrease;
Minkowski inputs remain nonempty and have nonzero area when normalization
leaves at least three vertices. The `int64` bound is `|x|, |y| < 2^62`,
tightened to `2^61` for Minkowski input coordinates. Hull construction checks
the bound even on singleton and duplicate-only inputs. Under another policy,
merge omits the turn checks at the two chain joins, matching the hull builder.

Documented only: the queried polygon must be strictly convex and CCW; checking
that would cost O(h). Minkowski inputs must be convex. Minkowski sums, edge
differences and accumulated steps must fit the coordinate type `T`, including
narrow integer types. Floating coordinates must satisfy the finite, normal
product requirements of the geometry overview, and constructions must remain
finite. Under `exact`, the hull and containment predicates are exact for
signed integer types through 64 bits, `float` and `double`; character types,
unsigned integers and `long double` are rejected. Differences use
`difference_product_t<T>`: `int64` for 16-bit coordinates, `int128` for 32/64-bit,
and `double` for floating coordinates. Full-range 32-bit input signs are exact.
`fast` may misclassify floating ties. `approx{eps}` applies determinant dead
zones and is intended for rounded input, without an exact geometric guarantee;
its equality never controls hull deduplication. For floating Minkowski sums,
edge differences, coordinate sums and accumulated vertices are rounded in T;
exact support predicates on those rounded directions cannot certify the true
sum's topology, strictness or absence of duplicate output vertices. Treat the
floating sum as a construction.

### Measured performance

`benchmarks/geometry/convex_hull.cpp` compares adaptive octagon pruning with
an unpruned chain, angular-sort Graham scan, Quickhull splitting, Chan
mini-hulls and Jarvis wrapping. Each call constructs one hull. Jarvis omits
hull-heavy circles because of its O(n h) cost. A final row forms a Minkowski
sum of finished circle hulls. Divide and conquer is not raced: it has the
same bound as the monotone chain with more merge machinery.

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo,
assertions off, 2026-10-15. Flags `-O2 -g -DNDEBUG -std=c++23`, optionally
`-march=native`. Minimum per-run means, three runs of 25 samples; milliseconds.

| Row | -O2 | native |
|---|---:|---:|
| library, uniform 1e6 int64 | 12.1 | 11.3 |
| by unpruned chain, uniform | 136 | 136 |
| library, circle 1e6 | 132 | 132 |
| by unpruned chain, circle | 133 | 132 |
| library, Gaussian cluster 1e6 | 12.5 | 11.5 |
| by unpruned chain, cluster | 142 | 139 |
| library, uniform double 1e6 | 11.3 | 12.5 |
| by Graham scan, uniform | 256 | 287 |
| by Graham scan, circle | 257 | 287 |
| by Quickhull, uniform | 37.5 | 42.2 |
| by Quickhull, cluster | 32.3 | 36.3 |
| by Quickhull, circle | 289 | 342 |
| by Chan, uniform | 278 | 279 |
| by Chan, circle | 462 | 479 |
| by Jarvis march, uniform (h about 30) | 333 | 406 |
| Minkowski sum, two circle hulls | 12.8 | 13.6 |

Pruning is about 11.2 times faster on the plain uniform row (136 / 12.1),
while its sample gate avoids an expensive full scan on hull-heavy inputs.
Quickhull uses explicit heap tasks and Graham scan orders same-ray ties
with raw coordinates;
the supplied measurements establish the algorithm comparison, while their
current implementation costs need a new measurement.

The same source record also measures hull-union paths (now reproduced by
`benchmarks/geometry/convex_hull_merge.cpp`); CPU, compiler and flags as above.

| Union of two circle sets | -O2 | native |
|---|---:|---:|
| by concatenation, raw 1e6 + 1e6 | 320 | 317 |
| by materialized chains with rehull, raw | 303 | 301 |
| by copodal walk with rehull, raw | 334 | 332 |
| by concatenation, finished hulls | 48.2 | 47.8 |
| by materialized chains with rehull, finished hulls | 87.4 | 87.2 |
| by materialized chains, finished hulls | 26.4 | 26.1 |
| by copodal walk with rehull, finished hulls | 117 | 117 |
| by copodal walk, finished hulls | 56.2 | 56.6 |

The merge dispatch-size race compares streamed chains with binary bridges on
separated hulls. Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0,
RelWithDebInfo, assertions off, 2026-11-02. Same flag sets and sampling method;
milliseconds. Source: `benchmarks/geometry/convex_hull_merge.cpp`.

| Type, shape, total vertices | by streamed chains -O2 / native | library -O2 / native |
|---|---:|---:|
| int64, halves, 256 | 0.001 / 0.001 | 0.001 / 0.001 |
| int64, circles, 256 | 0.002 / 0.002 | 0.002 / 0.002 |
| int64, halves, 512 | 0.002 / 0.002 | 0.002 / 0.002 |
| int64, circles, 512 | 0.004 / 0.004 | 0.004 / 0.005 |
| int64, halves, 1024 | 0.004 / 0.004 | 0.001 / 0.002 |
| int64, circles, 1024 | 0.009 / 0.009 | 0.001 / 0.002 |
| int64, halves, 2048 | 0.008 / 0.008 | 0.002 / 0.003 |
| int64, circles, 2048 | 0.018 / 0.018 | 0.002 / 0.002 |
| int64, halves, 8192 | 0.031 / 0.031 | 0.005 / 0.006 |
| int64, circles, 8192 | 0.071 / 0.071 | 0.003 / 0.004 |
| int64, halves, 32768 | 0.121 / 0.123 | 0.016 / 0.017 |
| int64, circles, 32768 | 0.282 / 0.282 | 0.009 / 0.010 |
| int64, halves, 100000 | 0.364 / 0.377 | 0.122 / 0.111 |
| int64, circles, 100000 | 0.860 / 0.862 | 0.052 / 0.049 |
| int64, halves, 300000 | 1.1 / 1.1 | 0.365 / 0.331 |
| int64, circles, 300000 | 2.2 / 2.6 | 0.186 / 0.168 |
| int32, halves, 256 | 0.001 / 0.001 | 0.001 / 0.001 |
| int32, circles, 256 | 0.001 / 0.001 | 0.001 / 0.001 |
| int32, halves, 1024 | 0.003 / 0.003 | 0.001 / 0.001 |
| int32, circles, 1024 | 0.005 / 0.005 | 0.001 / 0.001 |
| int32, halves, 8192 | 0.020 / 0.024 | 0.003 / 0.003 |
| int32, circles, 8192 | 0.039 / 0.042 | 0.002 / 0.002 |

The 300000-vertex disjoint-circle row saves about 91.5% with plain -O2
(1 - 0.186 / 2.2). Overlapping, nested, identical, disk and interleaved rows
remain in the benchmark; their current dispatch comparison is not measured
yet in the supplied record.

### Pitfalls and notes

Keep-collinear output is unsuitable for strict-hull caliper views: use the
default hull or normalize a convex boundary. A merge takes finished hulls;
for raw point sets concatenate and hull, or hull each side then merge.
Minkowski coordinates are sums in `T`, so inputs that fit individually can
still overflow the result. None of the return values borrows input storage.

## Convex layers API

`std::vector<std::vector<vertex_index>> convex_layers(R &&points, Policy p = {})`
is `[[nodiscard]] constexpr`, with the same range and policy constraints as
hull construction. Layer zero contains the entire hull boundary, including
straight boundary positions; each later layer peels the remainder. Every
input index occurs once. Equal positions peel together, consecutive in each
layer with indices ascending. Layers walk CCW from their lexicographic minimum;
a collinear remainder is one final lexicographic layer. Empty input has no
layers; duplicate-only input has one layer.

### Complexity and preconditions

O(n log n + n L) time for L layers, O(n) extra memory excluding the output;
no recursion. The input size and coordinate bounds are asserted as for hulls.
Floating bounds and policy limitations are the same. Normalization is not
needed, because the input is a point cloud.

### Measured performance

`benchmarks/geometry/convex_layers.cpp` measures complete peeling on square
and disk clouds and nested octagons; there is no rival implementation.
Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo,
assertions off, 2026-10-15. Flags `-O2 -g -DNDEBUG -std=c++23`, optionally
`-march=native`. Minimum per-run means, three runs of 25 samples; milliseconds.

| Input | -O2 | native |
|---|---:|---:|
| uniform square, 100k | 1991 | 2009 |
| uniform disk, 100k | 1797 | 1812 |
| 2500 nested octagons, 20k | 641 | 662 |

### Pitfalls and notes

Peeling removes boundary points, including all duplicates at each position;
it does not remove just the strict corner indices. Many layers make repeated
scans expensive. A deletion-only dynamic hull could improve the asymptotic
bound, but requires a separate data structure and has not been raced here.

## How it works

Andrew's monotone chain sorts (point, index) records by raw coordinates and
index, deduplicates equal positions, and builds lower and upper stacks.
Akl–Toussaint pruning uses the eight extremes of x, y, x+y and x-y. A rectangle
inside their polygon accepts interior points with comparisons alone; a stride
sample tests its own octagon and enables the full scan only if it discards
at least half. Boundary points are never deliberately pruned.

Merge splits each input at its lexicographic maximum, merges the two lower
streams and then the upper streams. Dominated points are dropped against the
other chain's current edge; consecutive surviving vertices of one strict
chain can bypass the stack turn test. Separated exact integral hulls use
nested binary tangent searches, then copy four surviving runs. Minkowski sum
normalizes both boundaries, rotates to the (y, x)-minimum and merges edge
directions, combining parallel edges into one step. Convex containment binary
searches a fan wedge from vertex zero and tests its outer edge.

Layers sort and group equal positions once, run the keep-collinear stack on
the sorted survivors, expand each boundary position to its input indices,
and compact the survivors. Each layer costs linear time in its remainder.
The tests use triangle/segment enumeration for extreme vertices and an
independent supporting-line oracle for layer membership.

References: Andrew (1979), *Another Efficient Algorithm for Convex Hulls in
Two Dimensions*; Akl and Toussaint (1978), *A Fast Convex Hull Algorithm*;
Chan (1996), *Optimal Output-Sensitive Convex Hull Algorithms in Two and Three
Dimensions*. Implementations are independent.

The contender implementations are in
`atl/experimental/geometry/convex_hull_variants.hpp`.

## See also

[Geometry overview](geometry.md), [Polygons](polygon.md),
[Rotating calipers](calipers.md).
