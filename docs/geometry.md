# `atl::geometry`

Planar geometry with exact predicates for integer and floating coordinates.
This page holds what the whole folder shares: the coordinate types and the
widened products, the sign policies every predicate takes, and the three
precision tiers that decide which policy to use. Each family of components has
its own page:

- [Geometry primitives](geometry-primitives.md): points and vectors, the
  orientation predicate and sign policies, closed segments, directed lines,
  the area of a union of boxes and the polar-angle sort.
- [Circles and rays](circle.md): disk relations, crossings, tangents and enclosing circles; exact ray predicates and overlap shapes.
- [Delaunay and Voronoi](delaunay.md): triangulations, dual edges, Euclidean MSTs, bichromatic closest pairs and empty circles.
- [Polygons](polygon.md): signed area, exact point-in-polygon, winding,
  convexity, normalization, the area centroid and lattice-point counts.
- [Radial sweep](radial-sweep.md): the most overlapping arcs of directions and
  the most points one circle of a fixed radius covers.
- [Closest pair](closest-pair.md): the closest pair of a point set, exact for
  integer coordinates.
- [Manhattan MST](manhattan-mst.md): the minimum spanning tree under the L1
  distance, exact over the full integer range.

- [Convex hulls and layers](convex-hull.md): hulls, merging, Minkowski sums and onion peeling.
- [Rotating calipers](calipers.md): support events, diameters, strips and rectangles.
- [Convex polygon operations](convex-polygon-operations.md): intersection, union and closest hull-region points.
- [Half-plane intersection](halfplane-intersection.md): coefficient constraints, exact region kinds and boundary indices.
- [Segment intersections](segment-intersections.md): lazy exact integral events and polygon simplicity.
- [Arrangements](arrangement.md): half-edge subdivisions, holes, overlaps and isolated vertices.
- [Point location](point-location.md): persistent slabs and batch classification.
- [Polygon overlay](polygon-overlay.md): Boolean membership and areas of simple polygons.
- [Triangulation](triangulation.md): simple-polygon triangles and their dual tree.

The library separates two kinds of computation:

- **Predicates** (orientations, comparisons, equality) drive control flow: one
  wrong sign can corrupt a hull or a sweep for good. The orientation-based predicates are exact by
  default; higher-degree predicates and circle expressions have their own
  bounds and floating guarantees on the family pages.
- **Constructions** (lengths, distances, intersection points) are output
  values: a rounded `double` is the honest answer, and no exactness is
  claimed. Each function documents which kind it is.

## Coordinates and widened products

`atl::geometry::coordinate` admits non-character signed integers of at most
64 bits, `float` and `double`. All character types are excluded (`char`,
`signed char`, `unsigned char`, `wchar_t`, `char8_t`, `char16_t`, `char32_t`),
including `int8_t` when it aliases `signed char`. Unsigned types are excluded
because cross products and differences are signed; `long double` because its format differs by target
(80-bit x87, binary128, or plain `double` on MSVC and Apple), so no exactness
argument about it is portable; 128-bit integers because their products have no
wider type. Two alias templates name the exact result types:

| alias | meaning | `int16` | `int32` | `int64` | `float` | `double` |
|---|---|---|---|---|---|---|
| `product_t<T>` | one product of raw coordinates | `int32` | `int64` | `int128` | `double`, each product exact | `double`, rounded |
| `difference_product_t<T>` | a product of coordinate differences | `int64` | `int128` | `int128` (bounded) | `double` | `double` |

`atl::int128` is the native 128-bit integer where the compiler has one and a
portable software class elsewhere (MSVC, 32-bit targets); the geometry code
is the same on both. `int64` has no wider type to grow its differences into,
so every operation on `int64` differences carries the precondition
`|x|, |y| < 2^62`, checked with `atl_assert`. Narrower integers have no
magnitude restriction at all: any `int32` triple has an exact orientation.

`point<T>` arithmetic (`+`, `-`, `*`) works in `T` and can overflow near the
type's limits like any `T` arithmetic. The predicates and the widened value
functions take the original points and widen internally, so write
`orientation(o, a, b)`, not `cross(a - o, b - o)`.

## Sign policies

Every predicate-driven decision goes through a sign policy, a small value
passed last where a comparator would be, `exact` unless the call names another:

- **`exact`** (the default): integral coordinates use the widened products;
  floating coordinates use filtered adaptive evaluation (Shewchuk's method).
  A fast path evaluates the determinant normally and certifies its sign
  against a proven error bound; near-degenerate inputs fall through to
  error-free transformations whose last stage is the exact sign of the true
  determinant. Every `double` is an exact rational, so "collinear" and "same
  point" have exact answers for floating input data, and this policy gives
  them. Equality is plain `==`.
- **`fast`**: floating orientation is the raw IEEE evaluation, with no filter
  and no fallback: near-degenerate triples may be misclassified, and the
  result on exact ties can depend on floating contraction (`-ffp-contract`).
  Integral coordinates behave exactly as under `exact`. Name it only where
  the input is known non-degenerate or a wrong tie sign is harmless.
- **`approx{eps}`**: dead-zone comparisons for constructed values, quantities
  already rounded (intersection points, rotated coordinates) where no
  predicate can recover the lost information. The epsilon is a member,
  visible at the call site; there is no default and no global. Mind the
  units: orientations compare products of two coordinate differences,
  `equal` compares a distance. `approx::equal` is not transitive, so use it
  for boundary and output decisions, never as a sort predicate, map key or
  deduplication relation; to order doubles, use their exact `<`. For integral
  coordinates the determinant is computed exactly and only its conversion to
  `double` meets the dead zone, so `orientation` and `cross_sign` agree.

`sign_policy<Policy, T>` is the concept the predicates constrain on. A policy
is a copyable type with const members `orientation(o, a, b) -> int`,
`cross_sign(a, b, c, d) -> int` (the sign of `(b - a) x (d - c)` for four
independent points), `sign(v) -> int` for both `product_t<T>` and
`difference_product_t<T>` values, and `equal(p, q) -> bool`.

## Precision guidance: the three tiers

1. **Integer coordinates.** When the input permits (integers, or decimals with
   a fixed number of digits scaled by a power of ten), use an integral
   coordinate type: every predicate is exact on the widened products and
   floating point never enters. The general bound is `|x|, |y| < 2^62` for
   `int64`; higher-degree circle and Delaunay predicates have tighter bounds
   on their family pages.
2. **Floating input data.** Orientation-based predicates stay exact under the default policy, so
   orientation-based sorts on `double` input are safe, exact ties included.
   Segment sweeps, arrangements, point location and overlay also compare
   constructed rational values: their floating topology is best effort, as
   their family pages document. `float`
   coordinates convert to `double` exactly and get the same guarantee. The
   precondition is finite coordinates and finite, normal squares and pairwise
   products of the coordinates and differences formed by the function, with
   zero-factor products allowed. Sums must remain finite too; centroid moments
   and coefficient anchors have additional product bounds documented at their
   declarations. There is no rescaling. Cheap magnitude checks are asserted.
   The exact policy asserts overflow, NaN and a zero product of nonzero factors.
3. **Constructed values.** Rounding has already happened: compare them through
   one `approx{eps}` chosen from the application's tolerance, and shape the
   algorithm so that a misclassified tie stays local instead of corrupting a
   structure.

On targets with `FLT_EVAL_METHOD != 0`, runtime error-free transforms force
binary64 rounding through a store barrier. Constant evaluation keeps the plain
arithmetic path; other targets need no barrier. Round-to-nearest and no unsafe
floating reassociation remain required.

## Reduced includes

The family headers above are the normal entry points for every coordinate
type. Integer-coordinate programs that need smaller sources can use the documented
`*_core.hpp` layers (shared policy implementations) or `*_integral.hpp`
headers (integer kernels). These remove adaptive floating implementations by
include structure, including uninstantiated templates and discarded arms.
Constructed results may still be `double`; their coordinate input remains
integral. Include the full header before instantiating an exact floating
predicate; an unavailable extension fails at compile time and names the
required header. The mixed-coordinate primitives example keeps its full
predicate umbrella deliberately.
