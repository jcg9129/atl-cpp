# `atl::geometry` rotating calipers

Rotate supporting lines around strict convex polygons through lazy event
ranges, or use the supplied diameter, strip, hull-distance and enclosing
rectangle folds. Use these when extrema occur on convex boundaries. Hull
region distance ignores interior input points; the [closest pair](closest-pair.md)
problem has different semantics. Coordinate types and the three precision
tiers follow the [geometry overview](geometry.md).

```cpp
#include <atl/geometry/calipers.hpp>
```

Use `<atl/geometry/calipers.hpp>` as the normal entry point. For reduced
integer-coordinate solutions, `<atl/geometry/calipers_core.hpp>` shares the
same definitions while omitting adaptive floating predicates. Exact floating
predicates require the full header before their first instantiation; missing
extensions produce a compile-time include diagnostic. The `_core` suffix
matches the existing shared policy layers (`segment_core.hpp` and
`angular_sort_core.hpp`).

\include calipers/quick.cpp

Output:

\verbinclude calipers/quick.out

## API

Everything is in `atl::geometry`. `R`, `RP`, `RQ`, `RA`, `RB` model
`vertex_range`; `T` is the first range's `vertex_coordinate_t`. `Policy`
models `sign_policy<T>` and defaults to `exact`. Two-input APIs require the
same coordinate type. Functions returning results are `[[nodiscard]]`.

```cpp
struct antipodal_event {
	vertex_index edge_start, edge_end, opposite;
	constexpr std::array<std::array<vertex_index, 2>, 2> antipodal_pairs() const;
};
struct support_event {
	vertex_index edge_start, edge_end;
	bool is_on_second;
	vertex_index p_support, q_support;
	constexpr std::array<std::array<vertex_index, 2>, 2> support_pairs() const;
};
template<coordinate T>
struct strip {
	point<T> edge_start, edge_end, opposite;
};
```

Events and strips have value equality. Event indices are positions in the
polygons passed to the views. `antipodal_pairs()` returns the edge endpoints
paired with `opposite`. A support event's pairs always use (P index, Q index),
including events whose edge is on Q. P/Q parallel ties emit consecutive events
covering the extremal support combinations; they do not enumerate every
crossed endpoint combination. This suffices for the supplied folds.

| Expression or signature | Result |
|---|---|
| `antipodal_view(R &&poly, Policy policy = {})` | constexpr forward event range for one strict CCW polygon |
| `dual_support_view(RP &&p, RQ &&q, Policy policy = {})` | constexpr forward event range, parallel supports |
| `dual_support_view(RP &&p, RQ &&q, anti_parallel_t, Policy policy = {})` | constexpr forward event range, opposed supports |
| `std::array<vertex_index, 2> farthest_pair_indices(R &&points, Policy p = {})` | constexpr diameter pair in the input |
| `std::array<point<T>, 2> farthest_pair(R &&points, Policy p = {})` | constexpr diameter pair as points |
| `std::array<vertex_index, 2> farthest_pair_between_indices(RA &&a_points, RB &&b_points, Policy p = {})` | constexpr maximum-distance pair, one index per input |
| `std::array<point<T>, 2> farthest_pair_between(RA &&a_points, RB &&b_points, Policy p = {})` | constexpr maximum-distance pair as points |
| `strip<T> narrowest_strip(R &&points, Policy p = {})` | two parallel boundary lines through input points |
| `float64 distance_between_convex_hulls(RA &&a_points, RB &&b_points, Policy p = {})` | rounded distance between filled hull regions; zero on overlap or touch |
| `std::array<point<float64>, 4> min_area_enclosing_rectangle(R &&points, Policy p = {})` | four constructed corners, CCW, minimizing area |
| `std::array<point<float64>, 4> min_perimeter_enclosing_rectangle(R &&points, Policy p = {})` | four constructed corners minimizing perimeter |

The views deduce their range types from the argument value categories; their
constructors constrain storage with `std::constructible_from`. `anti_parallel_t`
is an explicit default-constructible tag with constant `anti_parallel`.
The antipodal constructor is explicit. Both views expose constexpr const
`begin()` and `end()`. Their nested `iterator` types are default constructible
forward iterators, with `value_type` the event, `difference_type = ptrdiff_t`,
`iterator_concept = forward_iterator_tag`; dereference returns an event by
value, prefix/postfix increment advance, and equality compares stream
positions. Only compare nonsingular iterators belonging to the same stream.

Lvalue polygons are referenced and rvalues owned; rvalue initializer lists are
materialized into vectors. Other non-owning ranges, including rvalue spans and
subranges, require their backing storage to remain alive. Keep referenced ranges alive
and unchanged; moving or destroying a view invalidates its
iterators. The views do not derive from `std::ranges::view_interface` and do
not promise `std::ranges::view` modeling. Returned application results own all
their data. Applications hull raw inputs internally.

Antipodal polygons with fewer than three vertices emit no events. Dual views
accept empty polygons (no events), distinct-endpoint segments and points;
point/point emits no events, while one point remains a constant support against
a polygon. A diameter has two distinct input indices whenever at least two
inputs exist, including duplicates; a singleton pairs with itself. Diameter
ties are arbitrary. Strip and rectangle ties are also arbitrary.

A strip has one boundary through `edge_start` and `edge_end`, the other through
`opposite`, parallel. Its width is
`double(cross(s.edge_start, s.edge_end, s.opposite)) / distance(s.edge_start, s.edge_end)`
when the endpoints differ. A singleton gives three equal points and width zero;
a collinear cloud gives a zero-width strip. Rectangle corners start at the
low-projection end of a flush hull edge. A point gives four equal corners and
a collinear cloud a zero-height rectangle. Minimum area and minimum perimeter
can choose different rectangles.

## Complexity

Antipodal iteration emits O(h) events, one per edge plus a parallel-opposite
tie event; advancing is O(1) amortized. Construction computes the initial
support in O(h), so every `begin()` call is O(1). Dual initialization scans both polygons, and iteration emits
hP + hQ events when each has at least two vertices; a point contributes zero
edges. Both streams use O(1) extra storage apart from any owned inputs and
O(hP + hQ) total time. Assertion checks scan convexity.

Applications take O(n log n) time, or O((n + m) log(n + m)) for two sets,
and O(n + m) extra memory including hulls. Sweeps and folds have constant
call-stack depth; hull sorting uses O(log n) depth. Standard allocation
failures propagate. Strip, distance and rectangle functions are runtime-only
because they use square roots unavailable in constant evaluation on some
supported compilers.

## Preconditions and precision

Asserted under `ATL_ASSERT`: view ranges have fewer than 2^32 vertices;
polygons with at least three vertices must be strictly convex CCW; dual
segments must have distinct endpoints. Applications require nonempty point
sets, each smaller than 2^32. Use the default [convex hull](convex-hull.md)
for valid view input. Each input `int64` coordinate must satisfy
`|x|, |y| < 2^62`; hull-region distance tightens this to `2^61`, checked prior
to negating the second set. For other integral types, negation and all
coordinate differences used in the Minkowski difference must fit `T`.

Documented only: floating inputs are finite and all required products of
coordinates and differences are normal and finite unless a factor is zero;
sums must stay finite. Rectangle objective calculations additionally require
finite projection products, objective intermediates and a finite, nonzero squared edge length.
The output corners can lose small offsets when large coordinates are rounded.
There is no uniform relative error bound on nearly degenerate constructions;
use an application-scale absolute tolerance as well as a relative tolerance.

Under `exact`, event order and support predicates are exact for accepted
signed integer types, `float` and `double`, within those bounds. Character
types, unsigned integers and `long double` are rejected. Squared-distance
selection for integral diameters is exact in `difference_product_t<T>`:
`int64` for 16-bit coordinates and `int128` for 32/64-bit coordinates, including
the portable software alias. Floating diameters compare rounded squared
distances and may select a different pair on a rounded tie. Strip selection
compares rounded widths, though its defining points are exact input vertices.
Floating hull-region distance also constructs a rounded Minkowski difference
in T; even its overlap/contact classification can differ at rounding-scale
ties. Integral overlap/contact classification is exact within the bounds.
Rectangle objectives, corners and nonzero hull distances are constructions
for every coordinate type. `fast` and `approx{eps}` change predicate decisions;
neither certifies floating geometry. Approximate equality is not a sort key.

## Measured performance

`benchmarks/geometry/calipers.cpp` measures whole event streams on large
strict hulls formed by angle-sorted paired edge vectors and end-to-end
applications on scattered clouds. Parallel and opposed dual streams differ
in their second support direction; applications fold those events for their
own objectives. There is no alternative implementation in this benchmark.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Row | -O2 | native |
|---|---:|---:|
| antipodal sweep, 1e5-vertex hull | 1.2 | 1.2 |
| dual sweep parallel, 1e5 + 1e5 | 3.3 | 3.4 |
| dual sweep opposed, 1e5 + 1e5 | 3.3 | 3.4 |
| farthest pair, 1e6 cloud | 4.4 | 5 |
| narrowest strip, 1e6 cloud | 4.5 | 5.3 |
| farthest pair between two 5e5 clouds | 4.1 | 5.1 |
| hull-region distance, two 5e5 clouds | 5 | 5.7 |
| minimum-area rectangle, 1e6 cloud | 4.2 | 5.1 |

The antipodal row is 12 ns per input hull vertex in both modes
(1.2 ms / 1e5). The cloud rows include hull construction. The perimeter
rectangle is not measured yet.

## Pitfalls and notes

Keep-collinear hull output is not strict view input. Events can repeat pairs;
collect into a set if uniqueness is needed. A dual support stream is not a
list of all vertex pairs. Hull distance measures regions, so overlapping
sets of hulls have distance zero even if their input point sets are disjoint.
For rounded corners and distances compare within a scale-aware tolerance;
exact predicates cannot recover detail lost in a construction.

## How it works

Toussaint's rotating-support-line mechanism replaces angles with exact edge
cross signs. In one polygon the opposite support advances monotonically as
each edge aligns; exactly parallel opposite edges add a tie event. The dual
stream merges edge directions using half-plane classes and cross signs;
opposed rotation reverses Q's candidate direction. Ties consume P first.

Diameter folds maximize squared distances over certified pairs, mapping hull
indices back to input indices. The strip fold compares edge-to-support
heights divided by edge lengths. Hull-region distance builds A + (-B) by
the parallel stream, tests the origin for containment, then minimizes its
distance to the boundary segments. The rectangle sweep maintains three
monotone supports per flush edge: minimum/maximum projection along the edge
and maximum perpendicular height. Dot comparisons reduce to a cross sign
against rotated endpoints; offsets are differenced in widened arithmetic
prior to conversion, preserving short edges at large translations.

Tests enumerate normal-cone intersections for antipodal pairs, all pairs for
diameters, and all flush directions/features for strip, rectangle and distance
checks. Degenerate inputs and coordinate-bound configurations are included.
Reference: Toussaint (1983), *Solving Geometric Problems with the Rotating
Calipers*. Implementations are independent.

## See also

[Geometry overview](geometry.md), [Convex hulls and layers](convex-hull.md),
[Polygons](polygon.md), [Closest pair](closest-pair.md).
