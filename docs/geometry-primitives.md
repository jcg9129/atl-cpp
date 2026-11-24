# `atl::geometry` primitives: points, predicates, segments, lines, boxes and polar angles

The building blocks of the geometry folder: `point<T>` with its widened cross
and dot products, the exact `orientation` predicate and the sign policies
(`exact`, `fast`, `approx`), closed segments and directed lines with exact
predicates and rounded constructions, the area of a union of axis-aligned
boxes (`union_area`), and the polar-angle comparator and sorts. Reach for them
whenever a decision depends on the sign of a determinant: the predicates never
misclassify a degenerate input under the default policy, integer or `double`.
The coordinate types, the widened product types, the policies and the
precision tiers are explained once in the [geometry overview](geometry.md).

```cpp
#include <atl/geometry/point.hpp>        // point, cross, dot, squared_distance, distance
#include <atl/geometry/predicates.hpp>   // orientation, exact, fast, approx, sign_policy
#include <atl/geometry/segment.hpp>      // segment, contains, intersects, distance
#include <atl/geometry/line.hpp>         // line, side_of, intersection, line_coefficients
#include <atl/geometry/box.hpp>          // box, union_area
#include <atl/geometry/angular_sort.hpp> // polar_angle_less, sort_by_polar_angle, polar_angle_order
```

\include geometry-primitives/quick.cpp

## Shared interface

Everything is in namespace `atl::geometry`; every function except those
returning a `double` length or distance (which call `std::sqrt`) is
`constexpr`, and every function whose only effect is its result is
`[[nodiscard]]`.

```cpp
template<typename T>
concept coordinate = /* non-character signed integers up to 64 bits, float or double */;

template<coordinate T> using product_t = /* int32, int64, int128 or double */;
template<coordinate T> using difference_product_t = /* int64, int128 or double */;

template<typename Policy, typename T>
concept sign_policy = coordinate<T> && std::copyable<Policy>
	&& requires(Policy const p, point<T> const q, product_t<T> const w, difference_product_t<T> const v) {
		{ p.orientation(q, q, q) } -> std::same_as<int>;
		{ p.cross_sign(q, q, q, q) } -> std::same_as<int>;
		{ p.sign(w) } -> std::same_as<int>;
		{ p.sign(v) } -> std::same_as<int>;
		{ p.equal(q, q) } -> std::same_as<bool>;
	};

template<typename R>
concept vertex_range = std::ranges::random_access_range<R> && std::ranges::sized_range<R>
	&& /* range_value_t<R> is some point<T> */;
template<vertex_range R> using vertex_coordinate_t = /* T of the points */;
using vertex_index = uint32;   // index into a vertex_range
using triangle_index = uint32; // index of a triangle in a triangulation result
```

Predicates take the policy as their last parameter, `Policy const p = {}`,
constrained as `sign_policy<T> Policy = exact`. Every `int64` operation on
coordinate differences asserts `|x|, |y| < 2^62`; floating predicates require
finite values whose products neither overflow nor underflow (the
[precision tiers](geometry.md)).

## Points and vectors

`point<T>` is an aggregate `{x, y}` with CTAD (`point{3, 4}`), used both as a
point and as a vector; `<=>` is lexicographic (`x`, then `y`).

| Signature | Result |
|---|---|
| `point &operator+=(point)`, `-=`, `operator*=(T)`; `+`, binary and unary `-`, `point * T`, `T * point` | arithmetic in `T` |
| `point<T> perp(point<T> v)` | `(-y, x)`, `v` turned a quarter counterclockwise |
| `product_t<T> cross(point<T> a, point<T> b)` | `a.x b.y - a.y b.x`; exact for integers and total (extreme `2^(2w-1) - 2^(w-1)` for `w`-bit coordinates); for `float` one rounding, so its sign is exact; for `double` rounded |
| `product_t<T> dot(point<T> a, point<T> b)` | `a.x b.x + a.y b.y`, exactness as `cross` |
| `difference_product_t<T> cross(point<T> o, point<T> a, point<T> b)` | twice the signed area of `o, a, b`, on widened differences; exact for integers, a rounded construction for floats |
| `product_t<T> squared_length(point<T> v)` | `dot(v, v)` |
| `difference_product_t<T> squared_distance(point<T> a, point<T> b)` | exact for integers: compare these, not `distance` |
| `float64 length(point<T> v)`, `float64 distance(point<T> a, point<T> b)` | rounded constructions; not constexpr |
| `point<float64> to_float64(point<T> p)` | coordinatewise; exact for floats and for integers up to `2^53` |

All O(1). Asserted: for integer coordinates, `dot` and `squared_length` reject
the one input whose value overflows `product_t` (all four coordinates at the
type's minimum, value `2^(2w-1)`); the three-point `cross` and
`squared_distance` assert the `int64` bound. To convert a coordinate
difference to `double`, subtract in the widened type first: converting each
coordinate rounds above `2^53` and can cancel a short offset to zero.

## Orientation and sign policies

| Signature | Result |
|---|---|
| `int orientation(point<T> o, point<T> a, point<T> b, Policy p = {})` | +1 if `b` is counterclockwise of `a` seen from `o` (left turn), -1 if clockwise, 0 if collinear or any two coincide |
| `int Policy::orientation(o, a, b) const` | the same, as a member |
| `int Policy::cross_sign(a, b, c, d) const` | sign of `(b - a) x (d - c)`, four independent points |
| `int Policy::sign(V v) const` | sign of a scalar (dead zone for `approx`) |
| `bool Policy::equal(point<T> a, point<T> b) const` | `==` for `exact` and `fast`; distance at most `epsilon` for `approx` |
| `bool approx::equal(float64 a, float64 b) const` | `a` and `b` differ by at most `epsilon` |
| `float64 approx::epsilon` | the dead-zone half-width, in the units of the compared value |

`exact::sign` and `fast::sign` accept any value constructible from `int` and
totally ordered; `approx::sign` any value convertible to `double`. Every
predicate is O(1). Exactness per coordinate type under `exact`:

| coordinates | orientation and `cross_sign` | bound |
|---|---|---|
| `int16`, `int32` | exact: differences in `product_t`, products in `difference_product_t` | none |
| `int64` | exact: differences and products in `int128` | coordinates below `2^62` in magnitude, asserted |
| `float` | exact: converted to `double` without rounding, then as `double` | as `double` |
| `double` | exact: the static filter certifies the common case, adaptive stages the rest | finite, no overflow or underflow in products |

Under `fast`, integer coordinates are exact as above and floating ones are
the rounded determinant. Under `approx{eps}`, integer coordinates compute the
exact determinant and apply the dead zone to its `double` value, floating ones
apply it to the rounded determinant. Asserted for `double` under `exact`: a
product that overflows, a NaN, and a zero product of nonzero factors (an
underflow that would read as collinear).

### Measured performance

`benchmarks/geometry/predicates.cpp` times 1e6 orientations per row under each
policy on independent triples (throughput; the loops can vectorize) and, in
the serialized rows, on a chain where each result picks the next triple
(latency, closer to a hull loop). The contender "by branchless filter" bounds
the rounding error by `|left| + |right|` whatever the signs of the two products,
instead of first returning on opposite signs; it shares the adaptive stages.

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, assertions
off, 2026-10-15. Flags `-O2 -g -DNDEBUG -std=c++23`, optionally
`-march=native`. Minimum per-run mean over three runs of 25 samples; ms per
1e6 calls, which is ns per call. Spreads 1.01x to 1.15x.

| Shape | Policy or contender | -O2 | native |
|---|---|---:|---:|
| int32 uniform within 1e9 | exact | 2.7 | 2.6 |
| int64 uniform within 2^61 | exact (int128 products) | 7.7 | 7.7 |
| double generic position | exact | 16.6 | 16.5 |
| double generic position | fast | 5.8 | 5.7 |
| double generic position | approx{1e-9} | 5.8 | 5.6 |
| double generic position | by branchless filter | 6.6 | 6.1 |
| double collinear grid | exact | 24.1 | 18.1 |
| double collinear grid | fast | 5.5 | 5.4 |
| double collinear grid | by branchless filter | 24.0 | 18.1 |
| double grid nudged one ulp | exact | 24.2 | 18.0 |
| double grid nudged one ulp | fast | 5.6 | 5.6 |
| double grid nudged one ulp | by branchless filter | 28.1 | 22.8 |
| double mixed-exponent collinear | exact | 85.7 | 68.6 |
| double mixed-exponent collinear | fast | 5.6 | 5.4 |
| double generic, serialized | exact | 25.9 | 24.8 |
| double generic, serialized | fast | 16.3 | 15.7 |
| double generic, serialized | by branchless filter | 17.4 | 16.4 |
| int32, serialized | exact | 11.7 | 11.6 |

Exactness costs 2.9x over `fast` in throughput on generic doubles but 1.6x in
the serialized loop, where latency dominates. The collinear grid exits after
the exact product of the rounded differences; about half of the
mixed-exponent triples reach the full expansion, the worst case at about
86 ns. The branchless filter runs generic position 2.5x faster (6.6 against
16.6 ns) and the serialized loop 33% faster, but it is 16% slower one ulp from
degeneracy at `-O2` (28.1 against 24.2 ns) and 27% slower with
`-march=native`; whether to adopt it is open.

### Pitfalls

The dead zone of `approx` is in the units of the value it signs: an
orientation compares a product of two coordinate differences, so
`approx{1e-9}` on coordinates near 1e6 is far tighter than the rounding error
of the determinant. Use `approx` only for constructed values; for input data,
`exact` is both correct and, in generic position, within 3x of `fast`.

## Segments

`segment<T>` is an aggregate `{a, b}` with CTAD and memberwise `==`: the
closed segment between two points, the degenerate `a == b` (a point) legal
everywhere.

| Signature | Result |
|---|---|
| `point<T> direction(segment<T> s)` | `b - a`, in `T` |
| `difference_product_t<T> squared_length(segment<T> s)` | exact for integers |
| `float64 length(segment<T> s)` | a construction; not constexpr |
| `bool contains(segment<T> s, point<T> q, Policy p = {})` | `q` on the closed segment: the policy's orientation plus an exact bounding-box test |
| `bool intersects(segment<T> s, segment<T> t, Policy p = {})` | the closed segments share a point: touching endpoints and collinear overlap count |
| `float64 distance(segment<T> s, point<T> q)` | nearest point of `s`, chosen exactly; exactly 0 when `q` is on `s`; not constexpr |
| `float64 distance(segment<T> s, segment<T> t)` | exactly 0 when they intersect, else the least of the four endpoint-to-segment distances; not constexpr |

All O(1). With `approx{eps}` both tests in `contains` and `intersects` are
tolerant: collinearity within the dead zone (cross-product units) and the box
inflated by eps (coordinate units), so a crossing within the dead zone reads
as touching.

## Lines

`line<T>` is an aggregate `{a, b}`: the infinite DIRECTED line from `a`
through `b`, `a != b` asserted by every operation. `side_of` is signed relative
to the direction, so a line also names the closed half-plane on its left.
`==` is representational (the same two points in the same order); geometric
coincidence is `same_line`.

| Signature | Kind | Result |
|---|---|---|
| `point<T> direction(line<T> l)` | data | `b - a`, in `T` |
| `side side_of(line<T> l, point<T> q, Policy p = {})` | predicate | `side::left` (+1), `side::on` (0) or `side::right` (-1); `int(side)` is the orientation sign |
| `side side_of(segment<T> s, point<T> q, Policy p = {})` | predicate | relative to the line spanned by `s` (`s.a != s.b` asserted) |
| `bool contains(line<T> l, point<T> q, Policy p = {})` | predicate | `q` on the line |
| `bool parallel(line<T> l, line<T> m, Policy p = {})` | predicate | same or opposite direction, coinciding included |
| `bool same_line(line<T> l, line<T> m, Policy p = {})` | predicate | the lines coincide as point sets |
| `bool intersects(line<T> l, line<T> m, Policy p = {})` | predicate | not parallel, or coinciding |
| `bool intersects(line<T> l, segment<T> s, Policy p = {})` and `(s, l, p)` | predicate | the line meets the closed segment |
| `point<float64> intersection(line<T> l, line<T> m)` | construction | the crossing point; `!parallel(l, m)` asserted |
| `segment_intersection<T> intersection(segment<T> s, segment<T> t, Policy p = {})` | mixed | empty, one point, or the exact collinear overlap |
| `segment_intersection<T> intersection(line<T> l, segment<T> s, Policy p = {})` and `(s, l, p)` | mixed | empty, one point, or the whole segment when it lies on the line |
| `float64 distance(line<T> l, point<T> q)` | construction | exactly 0 on the line; not constexpr |
| `float64 distance(line<T> l, line<T> m)` | construction | 0 unless parallel and distinct; not constexpr |
| `float64 distance(line<T> l, segment<T> s)` and `(s, l)` | construction | exactly 0 on contact, else the nearer endpoint; not constexpr |
| `point<float64> projection(line<T> l, point<T> q)` | construction | foot of the perpendicular |
| `std::array<difference_product_t<T>, 3> line_coefficients(line<T> l)` | exact for integers | `{A, B, C}` of `A x + B y + C = 0`, oriented so `side_of(l, q) == sign(A q.x + B q.y + C)` |
| `line<float64> line_from_coefficients(std::array<Coefficient, 3> const &c)` | construction | the directed line of the equation, left side positive; `A`, `B` not both 0 (asserted) |
| `line<float64> line_from_coefficients(T a, T b, T c)` | construction | the same from three coordinate values |

```cpp
enum class side : int { right = -1, on = 0, left = 1 };
enum class intersection_kind { empty, at_point, along_segment, along_ray };

template<coordinate T>
struct segment_intersection {
	intersection_kind kind{intersection_kind::empty};
	point<float64> at{};  // kind == at_point: rounded, exact when it is an input point
	segment<T> along{};   // kind == along_segment: exact input points, along.a < along.b
};
```

`Coefficient` is any signed integer (`atl::int128` included) or floating
type, so `line_from_coefficients(line_coefficients(l))` round-trips within the floating magnitude preconditions;
`line_from_coefficients` anchors the line at the foot of the perpendicular
from the origin, a construction, since an integer-coefficient line need not
pass through any lattice point. `along_ray` is never produced by these
operations. All O(1).

\include geometry-primitives/segments-and-lines.cpp

Preconditions: `a != b` for every line, asserted; for `int64` the
`|x|, |y| < 2^62` bound, asserted (it also keeps `|C| < 2^126` in
`line_coefficients`); for floating coordinates the floating predicate
precondition.

Integer intersection constructions compute numerator and denominator
exactly in the widened integer type before converting and dividing once.
Floating determinants use double: with `u = 2^-53`, absolute error is at
most `4u / (1 - 4u)` times the sum of exact product magnitudes, including
errors from differencing. Division amplifies it near parallel lines, and
final coordinate cancellation can increase relative error. Their rounded denominator must be nonzero (asserted).
Coefficient conversion requires finite, normal norm and anchor products, except
zero-factor products; the norm and endpoints must be finite. The tangent step
scales with the anchor magnitude so that the endpoints remain distinct.

## Boxes: the area of a union

`box<T>` is an aggregate `{lo, hi}` with CTAD and memberwise `==`: the closed
axis-aligned box with min corner `lo` and max corner `hi`. A `box_range` is a
sized random-access range whose value type is some `box<T>`.

```cpp
template<box_range R>
constexpr difference_product_t<T> union_area(R &&boxes); // T: the boxes' coordinate type
```

The area covered by at least one box (Klee's measure in the plane), in
O(n log n) time and O(n) memory: exact for integer coordinates, a rounded
double construction for floating ones: both lengths and area accumulate in
`double` for `float` and `double` boxes, and the result type is `double`. Boxes of zero area contribute nothing; an
empty range gives 0.

Asserted: each box ordered (`lo.x <= hi.x`, `lo.y <= hi.y`), fewer than 2^29
boxes (the tree indexes at most 8n nodes with `uint32`), and for `int64`
coordinates `|x|, |y| < 2^62`, so every covered length times an x-gap fits
`int128`.

### Measured performance

`benchmarks/geometry/box.cpp` races `union_area` against
`atl::experimental::union_area_by_top_down_tree` (in
`experimental/atl/experimental/geometry/box_variants.hpp`), which differs only
in the tree: a recursive top-down update on one interleaved node array that
reads node lengths from the y values, instead of the iterative bottom-up
update with two leaf-to-root re-pulls over a precomputed length array. Each
row is one full call on random boxes in a 1e9 square: small boxes (sides below
1e7) touch few nodes per update, nested boxes span random intervals of the
whole square; the int32 rows narrow the same boxes. Tests certify both against
a per-cell brute force.

Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, assertions
off, 2026-10-15. Flags `-O2 -g -DNDEBUG -std=c++23`, optionally
`-march=native`. Minimum per-run mean over three runs of 25 samples;
milliseconds. Spreads 1.00x to 1.03x.

| Shape | Contender | -O2 | native |
|---|---|---:|---:|
| int64 small boxes 2e5 | library | 259 | 261 |
| int64 small boxes 2e5 | by top-down tree | 346 | 346 |
| int64 small boxes 1e6 | library | 2001 | 1995 |
| int64 small boxes 1e6 | by top-down tree | 2587 | 2561 |
| int64 nested boxes 2e5 | library | 276 | 283 |
| int64 nested boxes 2e5 | by top-down tree | 368 | 369 |
| int64 nested boxes 1e6 | library | 2145 | 2143 |
| int64 nested boxes 1e6 | by top-down tree | 2743 | 2716 |
| int32 small boxes 1e6 | library | 1603 | 1577 |
| int32 small boxes 1e6 | by top-down tree | 2206 | 2205 |
| int32 nested boxes 1e6 | library | 1727 | 1723 |
| int32 nested boxes 1e6 | by top-down tree | 2328 | 2370 |

The top-down tree takes 28-38% longer at `-O2` and 27-40% longer with
`-march=native`, on every shape.

## Polar-angle sort

| Signature | Result |
|---|---|
| `template<typename Policy = exact> struct polar_angle_less { Policy policy{}; bool operator()(point<T> a, point<T> b) const; }` | `a` has a strictly smaller angle than `b` in `atan2` order; `operator()` requires `sign_policy<Policy, T>` |
| `polar_angle_less(Policy) -> polar_angle_less<Policy>` | deduction guide |
| `void sort_by_polar_angle(R &&points, Policy p = {})` | sorts in place with `std::ranges::sort`, no allocation; requires `std::sortable<iterator_t<R>, polar_angle_less<Policy>>` |
| `std::vector<vertex_index> polar_angle_order(R &&points, Policy p = {})` | the STABLE argsort: same-direction points keep their input order |

`R` is a `vertex_range`, `Policy` a `sign_policy<vertex_coordinate_t<R>>`.
The order is the angle in `(-pi, pi]`, from signs only: no `atan2`, no
epsilon. Points on one ray from the origin compare equivalent (magnitude is
no tiebreak), duplicates included; the zero vector is equivalent to the
positive x axis, angle 0; the negative x axis is last. Integer coordinates use
the two-point `cross`, exact over the whole range of the type (`int64`
included, no `2^62` bound); floating coordinates use the policy's orientation
about the origin. A comparison is O(1); both sorts are O(n log n),
`polar_angle_order` with O(n) memory and `size(points) < 2^32` asserted.

\include geometry-primitives/boxes-and-angles.cpp

The sort needs a strict weak ordering: `exact` always gives one, `fast` on
floating coordinates only when rounding misclassifies no pair of directions,
and `approx` in general does not, since its dead zone is not transitive. For
a stable in-place sort, pass the comparator to `std::ranges::stable_sort`
yourself. The polar-angle sorts and `intersects` are timed in
`benchmarks/geometry/segments_polygons.cpp`; its table is on the
[polygons page](polygon.md#measured-performance).

## How it works

**Widened products.** For integer coordinates every quantity a predicate
signs is computed in a type wide enough to hold it: differences of `w`-bit
coordinates need `w + 1` bits and live in `product_t`, their products need
`2w + 2` bits and live in `difference_product_t`. Only `int64` runs out of
width (`int128` is the widest type), which is the source of the `2^62` bound:
differences then stay below `2^63`, each product below `2^126` and the
determinant below `2^127`. The two-point `cross` and `dot` multiply raw
coordinates, which need only `2w` bits, so they have no bound; the polar sort
uses that `cross`, which is why it accepts every `int64`. Narrow types
promote to `int` in C++ arithmetic, so the code converts every intermediate
back to its widened type explicitly.

**The exact floating orientation** follows Shewchuk's adaptive predicates
(Shewchuk, 1997, "Adaptive Precision Floating-Point Arithmetic and Fast
Robust Geometric Predicates"). The two products `left = (b.x - a.x)(d.y - c.y)`
and `right = (b.y - a.y)(d.x - c.x)` are evaluated in plain `double`. If their
signs differ, the difference has the true sign (a rounded nonzero product
keeps its sign). Otherwise the static filter compares `|left - right|` with
`(3 + 16 eps) eps (|left| + |right|)`, eps = 2^-53, which certifies almost
every generic input with one comparison. A failing input goes to the tail:
the exact determinant of the ROUNDED differences, built with error-free
products (one `fma` each at run time, the Veltkamp-Dekker split in constant
evaluation, where `fma` is unavailable) and checked against a smaller bound;
if the four differences were exact (their rounding errors, recovered by
two-difference, are 0, as for any integer-valued doubles) that determinant is
already the answer. Otherwise a second-order correction with its own bound
follows, and the last stage sums the remaining cross terms as expansions of
nonoverlapping components, whose leading component carries the exact sign.
Four independent points cost the same as an orientation, which is the
shared-origin special case.

**Segments and lines** reduce to orientations. `intersects` returns false as
soon as both ends of one segment lie strictly on one side of the other's line
(for `exact` and `fast`), declares a proper crossing on two strict straddles,
and otherwise tests whether a collinear endpoint lies in the other segment's
bounding box, an exact comparison of widened differences. `intersection`
returns input points exactly whenever the common point is one (touches and
collinear overlaps, whose lexicographic order is the order along the common
line), and constructs only proper crossings. The floating constructions start
from differences taken in the widened type and rounded once, so a short
segment far from the origin keeps its direction. The point distances check
the exact orientation first, so a point on the segment or line gives exactly
0 even when the rounded cross of the triple is not 0.

**`union_area`** sweeps the box edges by x and keeps, over the compressed y
values, a segment tree whose nodes store a cover count and the covered
length (the full node length when the count is positive, else the sum of the
children). The tree is a power-of-two array updated bottom-up: an edge adds
its delta to the O(log n) canonical nodes of its y range and then recomputes
the two leaf-to-root paths at the range boundaries. Lengths are held in the
unsigned type of the coordinate width, where the difference of two sorted
coordinates is exact by wraparound, and widened only for the area product.

**The polar-angle comparator** classifies each direction into the lower half
plane, the positive x axis (with the zero vector), the upper half plane or
the negative x axis; directions in different classes compare by class, and
within an open half plane one cross-product sign orders them, since every
pair there spans less than pi. `polar_angle_order` sorts `{point, index}`
records with the index as the tiebreak, which makes it stable while keeping
`std::ranges::sort`; sorting bare indices would gather both points from the
input on every comparison.

## See also

[Geometry overview](geometry.md) for the coordinate types, the policies and
the precision tiers; [Lexicographic order](lexicographic-order.md) for the
stable argsort convention that `polar_angle_order` follows.

The `intersection_kind::along_ray` result is used by `ray_intersection<T>`
in `atl/geometry/ray.hpp`; see [Circles and rays](circle.md) for exact ray
predicates, intersection shapes and distances.

The policy interface is available in `<atl/geometry/predicates_core.hpp>`;
`<atl/geometry/segment_core.hpp>` builds on it without the adaptive floating
specialization. The corresponding general headers install that specialization
and keep support for all coordinate types.
