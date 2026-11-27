# `atl::geometry` polygons: area, containment, winding, convexity, normalization, centroid and lattice points

Functions on a polygon given as its vertex list: the signed doubled area
(`polygon_area2`, exact for integers), the point-in-polygon test (`contains`,
with an exact boundary class), the winding direction, a convexity test, the
removal of duplicate and straight vertices, the area centroid and Pick's
lattice-point counts. Reach for them whenever a decision about a polygon must
not misread a vertex or an edge: every predicate is exact for integral and
floating coordinates under the default policy. There is no polygon class: a
polygon is any `vertex_range` (a sized random-access range of `point<T>`,
vertices in order, closing edge implicit, either orientation), since a vertex
list carries no invariant to protect. The coordinate types, the sign policies
and the precision tiers are explained in the [geometry overview](geometry.md).

```cpp
#include <atl/geometry/polygon.hpp> // polygon_area2, contains, winding_of, convexity_of, normalized_polygon, ...
```

Use `<atl/geometry/polygon.hpp>` as the normal entry point. For reduced
integer-coordinate solutions, `<atl/geometry/polygon_core.hpp>` shares the
same definitions while omitting adaptive floating predicates. Exact floating
predicates require the full header before their first instantiation; missing
extensions produce a compile-time include diagnostic. The `_core` suffix
matches the existing shared policy layers (`segment_core.hpp` and
`angular_sort_core.hpp`).

\include polygon/quick.cpp

Output:

\verbinclude polygon/quick.out

## API

Everything is in namespace `atl::geometry`, `constexpr` and `[[nodiscard]]`.
`R` is a `vertex_range` with coordinate type `T = vertex_coordinate_t<R>`;
`Policy` is a `sign_policy<T>`, `exact` by default; `W` is
`difference_product_t<T>`.

```cpp
enum class containment : int { outside = -1, boundary = 0, inside = 1 };
enum class winding : int { clockwise = -1, none = 0, counterclockwise = 1 };

struct convexity {
	bool is_convex;    // a convex simple polygon, straight vertices allowed
	bool is_strict;    // and no straight vertex; false when not convex
	winding direction; // winding::none when not convex
};

template<signed_integral Count>
struct lattice_count {
	Count boundary; // lattice points on the boundary
	Count interior; // lattice points strictly inside
};
```

| Signature | Result | Tier |
|---|---|---|
| `W polygon_area2(R &&poly)` | twice the signed area, positive counterclockwise; 0 below 3 vertices | exact for integers |
| `float64 polygon_area(R &&poly)` | the absolute area | construction |
| `containment contains(R &&poly, point<vertex_coordinate_t<R>> q, Policy p = {})` | where `q` lies; vertices and edges are `boundary` | predicate |
| `winding winding_of(R &&poly, Policy p = {})` | the traversal direction, `none` when degenerate | predicate |
| `convexity convexity_of(R &&poly, Policy p = {})` | convex simple polygon or not, strictly or not, which way | predicate |
| `std::vector<point<T>> normalized_polygon(R &&poly, Policy p = {})` | duplicates and straight vertices removed, counterclockwise, first survivor first | predicate-driven, exact output |
| `point<float64> polygon_centroid(R &&poly)` | the area centroid (not the vertex average) | construction |
| `lattice_count<W> lattice_count_of(R &&poly)` | boundary and interior lattice points; requires integral `T` | exact |

`contains` takes its query coordinate type from the polygon, so a braced
query such as `contains(poly, {1, 2})` works. A lossless integer-widening
overload also accepts `contains(poly64, point{1, 2})`; it converts the query
before testing. A wider integer query against narrower vertices does not
compile. Every
function is O(n) time (`lattice_count_of` O(n log C) for coordinate magnitude
C, one gcd per edge); `normalized_polygon` uses O(n) memory, the others O(1).
None recurses.

Details per function:

- `polygon_area2` is defined on ANY closed vertex sequence: by Green's theorem
  it is the winding-number weighted signed area (a symmetric bowtie gives 0,
  a doubly wound triangle twice its area, a hole traversed through a cut is
  subtracted). For integers it is exact whenever the result fits `W`, which
  every simple polygon within the coordinate bounds does.
- `contains` accepts fewer than 3 vertices: an empty range has everything
  outside, one or two vertices report `boundary` on the point or segment they
  span.
- `winding_of` orients the lowest-then-rightmost vertex, skipping copies of
  it; `none` means fewer than 3 vertices or a zero-area corner.
- `convexity_of` rejects (`{false, false, winding::none}`) fewer than 3
  vertices, consecutive equal vertices (`==` whatever the policy), mixed turns,
  collinear reversals (which covers every zero-area sequence) and sequences
  that wind more than once, the gap of a bare same-turn test (a pentagram
  turns left everywhere).
- `normalized_polygon` keeps the boundary as a point set, so the area, and is
  idempotent. Equality of vertices goes through `policy.equal`. With fewer
  than 3 distinct consecutive vertices (an empty input, one point, `{p, q, q,
  p}`) the deduplicated sequence is returned unchanged.

For strict counterclockwise hulls, `convex_contains` in
`atl/geometry/convex_hull.hpp` answers the same containment query in O(log n).
`convex_hull(points)` supplies that form; normalizing its keep-collinear
output yields the strict hull. See [Convex hulls and layers](convex-hull.md).

## Preconditions

Asserted (`atl_assert`, active under `-DATL_ASSERT`): for `int64` coordinates
`|x|, |y| < 2^62` in every function, as for the predicates;
`size(poly) < 2^32` in `convexity_of`; at least 3 vertices and nonzero area in
`lattice_count_of` and `polygon_centroid` (the centroid checks its rounded
doubled area, which the division needs); nonzero area in `normalized_polygon`
once 3 distinct vertices remain.

Documented only: every function except `polygon_area2` and `convexity_of`
assumes a SIMPLE polygon (no self-intersection, no repeated traversal).
`convexity_of` decides simplicity for convex candidates itself; for a general
simplicity test use `simplicity_of` from `atl/geometry/segment_intersections.hpp`
([segment intersections](segment-intersections.md)). Floating coordinates follow
the floating predicate precondition (finite values, no overflow or underflow in
products). The centroid additionally requires its area-weighted moment
products to be finite and normal unless a factor is zero (asserted), and its
accumulated moments and area to remain finite. Large coordinates can satisfy
an area bound while violating the stronger centroid bound; no rescaling is
performed. Indexed proxy references convertible to `point<T>` are accepted.

Exactness per coordinate type under `exact`: every predicate (`contains`,
`winding_of`, `convexity_of`, the decisions of `normalized_polygon`) is exact
for all coordinate types. `polygon_area2` and `lattice_count_of` are exact
integers for integral coordinates (`int16` squares of side 65535 need the `int64`
product type, `int64` boxes up to `2^63` per side give doubled areas just
below `2^127`); for `float` and `double` the area is rounded, and
`polygon_area` and `polygon_centroid` are constructions for every type.

## Measured performance

`benchmarks/geometry/segments_polygons.cpp` times the polygon queries
alongside the segment predicate and the polar-angle sorts of the
[primitives](geometry-primitives.md), whose numbers are kept here with the
benchmark. One race is in it: `sort_by_polar_angle` (an unstable
`std::ranges::sort`) against the same comparator in `std::ranges::stable_sort`
("by stable_sort"). Sort rows sort a copy of 1e6 points uniform in a 2e9
square, or 1e6 multiples of 441 directions for the same-ray ties;
`intersects` rows call the predicate on 1e6 pairs of short segments, densely
crossing in a 1000 square or scattered over the full range (mostly early
exits); the polygon rows run 1000 `contains` queries against a 1e5-vertex star
and one `polygon_area2` over it.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Row | -O2 | native |
|---|---:|---:|
| `contains`, 1e5-vertex star, 1000 queries | 479 | 552 |
| `polygon_area2`, 1e5 vertices | 0.248 | 0.331 |
| `sort_by_polar_angle` library, int64 | 129 | 122 |
| `sort_by_polar_angle` library, int32 | 119 | 108 |
| `sort_by_polar_angle` by stable_sort, int64 | 133 | 123 |
| `sort_by_polar_angle` library, double, exact | 204 | 170 |
| `sort_by_polar_angle` library, double, fast | 151 | 150 |
| `polar_angle_order`, int64 | 130 | 118 |
| `polar_angle_order`, double | 203 | 198 |
| `polar_angle_order`, int64 same-ray ties | 130 | 114 |
| `intersects`, int64, dense hits, exact | 16.6 | 18.1 |
| `intersects`, int64, dense hits, fast | 16.5 | 17.9 |
| `intersects`, double, dense hits, exact | 37.2 | 38.4 |
| `intersects`, double, dense hits, fast | 14 | 13.7 |
| `intersects`, int64, sparse hits, exact | 8 | 9.3 |
| `intersects`, int64, sparse hits, fast | 8 | 9.3 |
| `intersects`, double, sparse hits, exact | 19.1 | 19.9 |
| `intersects`, double, sparse hits, fast | 2.7 | 2.3 |

`contains` costs 4.79/5.52 ns per visited vertex (479/552 ms over
1000 queries of 1e5 vertices); `polygon_area2` costs 2.48/3.31 ns per
vertex. Stable sort takes 3.10%/0.82% longer than the library int64 sort
(133/123 versus 129/122 ms). Int32 sort takes 7.75%/11.48% less time than
int64; exact double sort takes 35.10%/13.33% longer than fast double.
The `intersects` rows are also ns per call: exact double takes
37.2/14 = 2.66 times as long as fast on dense hits plain (38.4/13.7 = 2.80
native), and 19.1/2.7 = 7.07 times on sparse hits (19.9/2.3 = 8.65
native). For int64 both policies use the same arithmetic.


## Pitfalls and notes

- The shoelace sign of `double` vertices is a rounded construction: a
  sliver near `1e16` can have its sign lost. Decide orientation with
  `winding_of`, which is exact, and use `polygon_area2` for the value.
- Feeding a self-intersecting sequence to a simple-polygon function is a
  precondition break, not a defined result: `winding_of` then reports its
  corner's turn, `contains` an even-odd parity.
- `normalized_polygon` is the canonical form for comparing polygons as point
  sets: same start vertex rule, counterclockwise, strict corners only.
- `polygon_centroid` is the lamina's center; for the average of the vertices,
  average them.

## How it works

**Area.** `polygon_area2` sums the triangles of a fan from the first vertex,
each `cross(v0, v_i, v_i+1)` exact on widened differences. Partial sums of a
fan are the doubled areas of the prefix chains closed by a chord, and a spiral
can wind those several times, beyond the final value; the sum is therefore
accumulated in the unsigned type of the same width, where wraparound is
defined, and converted back once: the result is exact whenever it fits.

**Containment** walks the edges once. Each edge is first tested with the
segment predicate (`contains(segment, q)`, exact orientation plus an exact box
test), so a boundary query returns before any parity is counted. Otherwise an
edge straddling the horizontal line through `q` under the half-open rule
(`u.y > q.y` differs from `v.y > q.y`: horizontal edges never count, a shared
vertex counts once) toggles the parity when `q` is on its left for an upward
edge, on its right for a downward one: one orientation, never zero here since a
zero with a straddle is a boundary point.

**Winding** uses one orientation at the lowest-then-rightmost vertex, which is
always a strictly convex corner of a simple polygon; copies of the corner are
skipped in both directions so the nearest distinct neighbors decide.

**Convexity** makes one pass of orientations: all turns must share one sign,
and a collinear vertex is accepted as straight only when its two edges point
the same way, decided by the signs of the coordinate differences, which are
exact even for doubles. That leaves sequences winding k > 1 times, which pass
every local test; a second pass counts the vertices where the edge direction
wraps past angle 0 in the sweep order (an angular class plus one exact cross
sign), exactly once for a convex simple polygon.

**Normalization** removes consecutive duplicates first, the cyclic wrap
included (a duplicated corner would make both copies' neighbor triples
degenerate), then every vertex collinear with its two neighbors, testing each
against its original neighbors: in a simple polygon a straight vertex lies
strictly between them, so no removal straightens a survivor. A clockwise
result is reversed behind its first vertex.

**Centroid and lattice points.** The centroid sums each fan triangle's doubled
area times the sum of its vertex offsets from the anchor (`u + v`, differences
taken exactly before rounding), divided by three times the total doubled area.
`lattice_count_of` adds `gcd(|dx|, |dy|)` per edge (the lattice points on the
edge with one endpoint counted) and solves Pick's theorem `A = I + B/2 - 1`
(Pick, 1899) as `I = (2A - B + 2) / 2` on the exact doubled area.

`triangulation_of` in `atl/geometry/triangulation.hpp` decomposes a simple
polygon into positive CCW triangles; the centroid tests also certify against
area-weighted triangle centroids from that independent decomposition.

## See also

[Triangulation](triangulation.md), [Segment intersections](segment-intersections.md);
[Geometry overview](geometry.md) for the coordinate types, the policies and
the precision tiers; [Geometry primitives](geometry-primitives.md) for the
segment predicate `contains` builds on and the polar-angle sorts timed above.
