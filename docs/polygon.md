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

Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, assertions
off, 2026-10-15. Flags `-O2 -g -DNDEBUG -std=c++23`, optionally
`-march=native`. Minimum per-run mean over three runs of 25 samples;
milliseconds. Spreads 1.00x to 1.06x.

| Row | -O2 | native |
|---|---:|---:|
| `contains`, 1e5-vertex star, 1000 queries | 671 | 794 |
| `polygon_area2`, 1e5 vertices | 0.370 | 0.389 |
| `sort_by_polar_angle` library, int64 | 147 | 148 |
| `sort_by_polar_angle` library, int32 | 129 | 129 |
| `sort_by_polar_angle` by stable_sort, int64 | 150 | 149 |
| `sort_by_polar_angle` library, double, exact | 194 | 179 |
| `sort_by_polar_angle` library, double, fast | 172 | 164 |
| `polar_angle_order`, int64 | 151 | 151 |
| `polar_angle_order`, double | 218 | 210 |
| `polar_angle_order`, int64 same-ray ties | 150 | 151 |
| `intersects`, int64, dense hits, exact | 24.1 | 30.9 |
| `intersects`, int64, dense hits, fast | 24.1 | 30.7 |
| `intersects`, double, dense hits, exact | 39.0 | 38.6 |
| `intersects`, double, dense hits, fast | 14.9 | 15.2 |
| `intersects`, int64, sparse hits, exact | 12.7 | 17.6 |
| `intersects`, int64, sparse hits, fast | 12.7 | 17.6 |
| `intersects`, double, sparse hits, exact | 21.7 | 21.1 |
| `intersects`, double, sparse hits, fast | 6.4 | 6.9 |

`contains` costs 6.7 ns per vertex at `-O2` (7.9 native), `polygon_area2`
3.7 ns (3.9). The stable sort takes 2% longer than the library sort at `-O2`
(150 against 147 ms) and 1% with `-march=native`; `int32` sorts 12% faster
than `int64`, and the exact double sort takes 13% longer than the fast one at
`-O2`, 9% native. The `intersects` rows are ns per call: exactness costs 2.6x
over `fast` on dense double hits and 3.4x on sparse ones at `-O2`; for `int64`
both policies are the same code. A bounding-box-first `contains` lost on the
int64 star (937 against 664 ms, `-O2`, 2026-10-14) and is not kept.

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
