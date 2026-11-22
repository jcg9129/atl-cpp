# `atl::geometry` convex polygon operations

Intersect or unite the convex hulls of two point sets, or recover a pair of
closest points on those hull regions. Inputs are point clouds, hulled internally;
use these functions when interior points should be ignored. Under `exact` or `fast`, predicates on
integral inputs are exact within the bounds below; output vertices are rounded
`float64` constructions.

```cpp
#include <atl/geometry/convex_polygon_operations.hpp>
```

\include convex-polygon-operations/quick.cpp

Output:

\verbinclude convex-polygon-operations/quick.out

## API

All functions are `[[nodiscard]]` in `atl::geometry`. `RA` and `RB` model
`vertex_range`, with the same `vertex_coordinate_t`; `Policy` models
`sign_policy<vertex_coordinate_t<RA>>` and defaults to `exact`. Both inputs
are passed as forwarding references and are not modified.

| Signature | Return |
|---|---|
| `std::vector<point<float64>> intersection_of_convex_polygons(RA &&a_points, RB &&b_points, Policy p = {})` | Empty, one contact point, two segment endpoints, or a CCW polygon; closing edge implicit |
| `std::vector<std::vector<point<float64>>> union_of_convex_polygons(RA &&a_points, RB &&b_points, Policy p = {})` | One CCW component for proper overlap, containment or segment contact; two for disjoint or point-touching polygons |
| `std::array<point<float64>, 2> closest_points_between_convex_hulls(RA &&a_points, RB &&b_points, Policy p = {})` | One point in each hull, attaining their region distance up to construction error; overlapping hulls return a common point twice |

Duplicates and collinear clouds are legal. Equal points form a point hull;
a collinear cloud forms a segment. A degenerate hull wholly contained in a
polygon contributes no extra union component; a protruding segment is returned
as its own component. Two touching collinear degenerate hulls merge into one
point or segment. Crossing segments remain two union components. Point-touching
polygons share their contact point in both components, since a pinched boundary
is not a simple polygon. Intersection segments have lexicographic endpoint order.
Polygon starting vertices, union component order and closest-point ties are
unspecified.

## Complexity

Each operation takes O((n + m) log(n + m)) time and O(n + m) extra memory,
including hull construction. The edge merge and deque passes are linear;
union containment shortcuts can take O(h log k), within the total bound.
Standard sorting uses O(log(n + m)) stack depth; the remaining passes are
iterative. Allocation failures propagate. These functions are not constexpr:
they use runtime deque storage, square roots or construction cleanup.

## Preconditions and precision

Asserted under `ATL_ASSERT`: both clouds are nonempty, their sizes fit the
hull's `uint32` index bound, and integral coordinates satisfy
`|x|, |y| < 2^61` for int64, `< 2^30` for int32 and `< 2^14` for int16.
The storage envelope for an 8-bit integer would be `< 2^6`; `int8` is a
character alias and is rejected by `coordinate`, as are other character types,
unsigned integers, `long double` and 128-bit coordinate types. These limits keep
Minkowski vertices in T. Int64 difference vertices stay below `2^62`, their
edge differences below `2^63`, within the widened quadratic predicates' bounds.
Contact directions always use supporting input edges, whose coordinates stay
below `2^62`, even when a Minkowski edge exceeds that bound.

The entry bounds scan automatically selects the integral arithmetic path.
When every raw coordinate is below `2^30`, crossing-side and rational parameter
comparisons use int128 arithmetic. Otherwise quadratic factors still fit int128,
while every quartic comparison uses exact signed 128x128 products compared as
256-bit unsigned magnitudes through one portable helper. This also works with
software `atl::int128` and 32-bit targets. Crossing-side signs use the factored
expression `(E D + N F) / D`, avoiding cubic intermediates. Clipping and union
parameters compare products directly without subtraction in int128.
Input-grid contact positions are selected exactly and then rounded to binary64;
large integral coordinates need not be exactly representable.

The three precision tiers need different expectations:

1. Integral input within the bound has exact overlap classification and
   stitching, crossing-side and clipping-parameter decisions under exact/fast.
   Intersection coordinates are rounded. Closest-point projection, clamping and
   minimum-distance selection are floating constructions even for integral inputs.
2. Floating input uses exact policy predicates on input points, but the
   Minkowski differences and crossing-side expressions are rounded. Even
   `exact` cannot certify the resulting topology. Require finite coordinates,
   finite normal products (zero factors allowed) and finite intermediate sums
   and constructions, under round-to-nearest without unsafe reassociation.
3. For constructed inputs, choose `approx{eps}` from application tolerances;
   it provides no exact topology guarantee. Output cleanup uses `2^-44` times
   the largest coordinate magnitude. It may merge nearby corners, or retain
   duplicates and straight vertices when fewer than three would survive.
   Highly translated, thin polygons therefore need particular care.

## Measured performance

Not measured yet with a complete publication record for this family. The
half-plane benchmark includes a comparison on two finished circle hulls, but
the supplied dispatch-size record does not measure that row. The wide path is
not measured yet; the benchmark includes a coordinates-near-`2^60` shape.

## Pitfalls and notes

A hull-region closest pair can lie inside input edges or inside overlapping
hulls; it is not a closest pair of original sites. Proper overlap need not
produce a point on both boundaries. Strictness of a rounded result is not
guaranteed. Compare constructions with a tolerance appropriate to their scale,
including under FMA contraction and excess evaluation precision.

## How it works

Hull construction removes duplicate and interior input sites. A parallel
support sweep forms the Minkowski difference, whose origin containment
classifies disjointness, contact and proper overlap. Contact is recovered from
supporting input faces. For proper overlap, angularly merged hull edges feed
a half-plane deque; integral pops compare rational crossings without division.
Intersection corners are constructed only once the active constraints are known.
Union walks outside arcs between alternating polygon transitions. Degenerate
segments use rational parameter clipping. Closest points split a minimum-distance
Minkowski edge back into its two supporting hull edges. The implementation uses
heap buffers and iterative walks, avoiding recursion proportional to cloud size.

## See also

[Geometry overview](geometry.md), [Convex hulls and layers](convex-hull.md),
[Rotating calipers](calipers.md), [Half-plane intersection](halfplane-intersection.md).
