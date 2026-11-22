# `atl::geometry::circle` and `atl::geometry::ray`

Closed disks, circle boundaries and directed half-infinite rays. Use integral
coordinates for exact circle classifications; use a tolerance for constructed
centers, tangent lines and crossings. Rays are documented on this page.

```cpp
#include <atl/geometry/circle.hpp>
#include <atl/geometry/ray.hpp>
```

\include circle/quick.cpp

Expected stdout:

\verbinclude circle/quick.out

## API

All entities are in `atl::geometry`. `T` models `coordinate` (non-character
signed integers up to 64 bits, `float`, `double`); `Policy` models
`sign_policy<Policy, T>` and defaults to `exact`. `R` models `vertex_range`.
Functions whose only effect is their result are `[[nodiscard]]`.
The following signatures use `T`, `Policy`, `R` for those constrained template
parameters; `intersection_area` additionally requires
`std::same_as<vertex_coordinate_t<R>, T>`.

```cpp
template<coordinate T> struct circle {
    using coordinate_type = T;
    point<T> center{};
    T radius{};
    friend constexpr bool operator==(circle const &, circle const &) = default;
};
template<coordinate T> struct ray {
    using coordinate_type = T;
    point<T> origin{}, through{};
    friend constexpr bool operator==(ray const &, ray const &) = default;
};
template<coordinate T> struct ray_intersection {
    intersection_kind kind{intersection_kind::empty};
    point<float64> at{};
    segment<T> along{};
    ray<T> beam{};
    friend constexpr bool operator==(ray_intersection const &, ray_intersection const &) = default;
};
```

Equality compares representation. In `ray_intersection`, only `kind` and its
selected field are meaningful: `at_point` selects rounded `at`,
`along_segment` selects exact lexicographically ordered endpoints in `along`,
`along_ray` selects the exact overlap ray `beam`. `empty` selects no field.
Same-direction collinear rays return the later origin's ray; facing rays
return the segment between their origins, a point, or nothing.

| Circle signature | Result |
|---|---|
| `constexpr containment contains(circle<T> const c, point<T> const q, Policy const p = {})` | `outside`, `boundary`, `inside` for the closed disk |
| `constexpr circle_relation relation(circle<T> const a, circle<T> const b, Policy const p = {})` | `disjoint`, `externally_tangent`, `overlapping`, `internally_tangent`, `nested`, `coincident` |
| `constexpr tangency relation(circle<T> const c, line<T> const l, Policy const p = {})` | `disjoint`, `tangent`, `secant`; reverse argument order also provided |
| `circle<float64> diametral_circle(point<T> const a, point<T> const b)` | Circle with diameter `a,b` |
| `circle<float64> circumcircle(point<T> const a, point<T> const b, point<T> const c)` | Circle through a noncollinear triple |
| `static_vector<point<float64>, 2> intersection(circle<T> const c, line<T> const l)` | Boundary crossings, increasing parameter along the directed line; reverse order also provided |
| `static_vector<point<float64>, 2> intersection(circle<T> const c, segment<T> const s)` | Boundary crossings in segment order; reverse order also provided |
| `static_vector<point<float64>, 2> intersection(circle<T> const a, circle<T> const b)` | Boundary crossings, left of `a.center -> b.center` first; coincident circles give empty |
| `static_vector<point<float64>, 2> tangent_points(circle<T> const c, point<T> const q, Policy const p = {})` | None inside, `q` on the boundary, two outside, left first; a point circle can give repeated touch points |
| `static_vector<line<float64>, 4> common_tangents(circle<T> const a, circle<T> const b)` | Outer tangents then inner: 4/3/2/1/0 for disjoint/external touch/overlap/internal touch/nested; coincident gives empty |
| `float64 intersection_area(circle<T> const c, R &&poly)` | Disk intersected with a simple polygon, either winding; reverse order also provided |
| `circle<float64> minimum_enclosing_circle(R &&points)` | Best-effort enclosing disk of a nonempty set |

Point-circle common tangents suppress duplicate sign choices: two distinct
point circles give their single connecting line. An infinite tangent family
is represented by an empty result, so classify coincident circles first.

| Ray signature | Result |
|---|---|
| `constexpr point<T> direction(ray<T> const r)` | `through - origin` in `T` |
| `constexpr side side_of(ray<T> const r, point<T> const q, Policy const p = {})` | Side of the spanning directed line |
| `constexpr bool contains(ray<T> const r, point<T> const q, Policy const p = {})` | On the ray, including the origin |
| `constexpr bool intersects(ray<T> const r, line<T> const l, Policy const p = {})` | Nonempty intersection; reverse order also provided |
| `constexpr bool intersects(ray<T> const r, segment<T> const s, Policy const p = {})` | Closed segment contact; reverse order also provided |
| `constexpr bool intersects(ray<T> const a, ray<T> const b, Policy const p = {})` | Ray contact |
| `constexpr ray_intersection<T> intersection(ray<T> const r, line<T> const l, Policy const p = {})` | Empty / point / ray; reverse order also provided |
| `constexpr ray_intersection<T> intersection(ray<T> const r, segment<T> const s, Policy const p = {})` | Empty / point / exact clipped segment; reverse order also provided |
| `constexpr ray_intersection<T> intersection(ray<T> const a, ray<T> const b, Policy const p = {})` | Empty / point / exact segment / exact ray |
| `static_vector<point<float64>, 2> intersection(circle<T> const c, ray<T> const r)` | Boundary crossings in ray order; reverse order also provided |
| `float64 distance(ray<T> const r, point<T> const q)` | Clamped projection distance |
| `float64 distance(ray<T> const r, line<T> const l)` | Minimum distance; reverse order also provided |
| `float64 distance(ray<T> const r, segment<T> const s)` | Minimum distance; reverse order also provided |
| `float64 distance(ray<T> const a, ray<T> const b)` | Minimum distance |

## Complexity

Constant time and space except disk-polygon area, O(n) time/O(1) space, and
minimum enclosing circle, expected O(n) with a random permutation, O(n^3)
worst-case time and O(n) heap storage. Its three nested loops never recurse.
Ray queries use O(1) stack. Predicate and ray classification functions are
`constexpr`; runtime constructions use square roots, trigonometry or seeded
random state and are not `constexpr`.

## Preconditions

Asserted: radii nonnegative, lines and rays nondegenerate for geometric
queries, circumcircle triples noncollinear, enclosing-circle input nonempty.
Integral circle containment, circle-circle relations and common tangents
accept the full `int16`/`int32` range; `int64` coordinates and radii must have
magnitude below `2^62`. Quadratic differences are below `2^127` in int128.
Circle-line relations and circle-line/segment/ray crossings tighten to
`|x|, |y|, radius < 2^30` for **both int32 and int64**, and `< 2^14` for int16.
The degree-four discriminant subtracts two nonnegative terms: with bound B,
`cross^2 < 16 B^4`, `r^2 |direction|^2 < 8 B^4`; these fit int128 or int64
respectively. A degenerate segment uses only containment's quadratic bound.
Ray predicates without circles accept full int16/int32 and int64 `< 2^62`.
`direction` is point arithmetic: its difference must fit `T` (documented only).

Documented only: polygons simple; all floating intermediates finite, nonzero
products normal, round-to-nearest and no unsafe reassociation. Cheap radius,
coordinate and orientation magnitude checks are asserted. Constructions do
not rescale and require representable divisions and nonzero rounded
denominators. There is no universal accuracy bound near collinearity,
tangency, or coordinates whose conversion loses a small displacement.

Precision tiers: integral classifications and crossing **counts** are exact
within these bounds. Floating circle expressions are rounded even with
`exact` (unlike adaptive-exact ray orientations); `fast` also rounds, and
`approx{eps}` applies its dead zone in squared-distance or discriminant units.
Constructed outputs require an application tolerance. The enclosing-circle
coverage check allows `1e-12 * radius^2 + 1e-30` squared slack: very tiny
clouds can return radius zero. `set_hash_seed` pins the shuffle; a seed does
not promise identical permutations across standard libraries.

## Measured performance

`benchmarks/geometry/circle.cpp` measures complete-call throughput, with no
contender: enclosing a uniform int64 cloud, classifications on circle pairs
and circle-point pairs. Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0,
RelWithDebInfo, assertions off, 2026-10-15. Flags `-O2 -g -DNDEBUG -std=c++23`,
optionally `-march=native`; minimum per-run means over three runs, 25 samples.

| Operation | -O2 ms | native ms | Largest spread |
|---|---:|---:|---:|
| minimum enclosing circle, 1e6 | 12.5 | 11.7 | 2.59x |
| relation, 1e6 pairs | 16.0 | 18.7 | 1.01x |
| contains, 1e6 pairs | 4.3 | 4.8 | 1.04x |

Ray operations are not measured yet. Enclosing-circle timing is unstable.

## Pitfalls and notes

Circle intersections concern the boundary, not the disk. `contains` is the
disk query. Endpoints returned in `double` are exact only when their input
coordinates are exactly representable (integers through `2^53`). Never use
approximate equality as a sorting relation.

## How it works

Radii are stored unsquared so the integral circle relation can compare
squared center distance to `(r1 + r2)^2` and `(r1 - r2)^2` exactly. Circle-line
classification uses a degree-four discriminant. Segment and ray crossings
count quadratic roots with exact signs, then select rounded line crossings.
Common tangents enumerate center-side sign pairs and suppress the vacuous
sign of a point circle. Disk-polygon area splits each chord into straight
triangle pieces and circular sectors. The enclosing-circle loops rebuild
with one, two or three support points in randomized order (Welzl, 1991,
Smallest enclosing disks).

Ray intersection parameters need only determinant signs; dot signs reduce
to cross signs by a perpendicular. Collinear cases select input endpoints
and origins. Distances test exact contact first, then minimize over endpoint
or origin projections: an interior minimum between nonparallel supports
would be a crossing.

## See also

[Primitives](geometry-primitives.md), [Delaunay and Voronoi](delaunay.md),
[Radial sweep](radial-sweep.md), [Geometry overview](geometry.md).
