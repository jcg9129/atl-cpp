# `atl::geometry::triangulation_of`

Triangulate a simple polygon into positive CCW index triples and a dual tree.
Use the triples for area decomposition or sampling and the tree for adjacency
walks; use Delaunay for an unconstrained point cloud.

```cpp
#include <atl/geometry/triangulation.hpp>
```

\include triangulation/quick.cpp

Expected stdout:

\verbinclude triangulation/quick.out

## API

```cpp
struct triangulation {
    std::vector<std::array<vertex_index, 3>> triangles;
    std::vector<std::array<triangle_index, 2>> dual_edges;
};
template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
triangulation triangulation_of(R &&poly, Policy const p = {});
template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
constexpr triangulation triangulation_of_by_ear_clipping(R &&poly, Policy const p = {});
```

All names are in `atl::geometry`. Input is a sized random-access point range;
T models [coordinate](geometry.md), Policy models `sign_policy<Policy, T>`.
The closing edge is implicit, either winding is accepted, and collinear straight
vertices are retained. A positive-area n-vertex polygon gives n - 2 strictly
positive CCW triangles and n - 3 dual edges, each connecting triangles sharing
a diagonal. Indices refer to the original range. Triangle order and choice of
diagonals are unspecified. Fewer than three vertices and all-collinear inputs
give empty vectors. The result owns its data and retains no input reference.

## Complexity

One pass counts reflex and straight vertices r. When `r <= 8 bit_width(n)`,
ear clipping costs O(n(r + 1)); otherwise monotone decomposition costs
O(n log(n + 1)). Thus the public dispatch takes O(n log(n + 1)) geometric work, expected
time with constant-time hash maps, and O(n) allocated memory. All polygon walks and work stacks are iterative, so
long thin polygons and collinear runs do not recurse per vertex. The public
entry is runtime-only because the sweep uses ordered/hash containers. The
`triangulation_of_by_ear_clipping` entry is constexpr, takes O(n(r + 1)) time
(r counts reflex vertices; quadratic worst case) and O(n) storage,
with the same bounds, orientation and result contract. Prefer `triangulation_of`
for large reflex-heavy inputs to retain the O(n log n) bound.
For example, `auto ears = atl::geometry::triangulation_of_by_ear_clipping(polygon);`.

## Preconditions

Asserted: fewer than `2^32` vertices, int64 `|x|, |y| < 2^62` when used by
predicates. **int32 accepts its full range** and narrow signed coordinates
are supported. Simplicity is documented only: no repeated vertices, edge
crossings or retraversals; use `simplicity_of` for validation. Failure to find
an ear asserts; no valid result is promised for a violated precondition.

Orientation has degree two; differences below `2^63` give determinants below
`2^127` under the int64 bound, within int128. Angular stitching compares
original points rather than subtracting directions in T. No higher-degree
predicate is needed. Winding and piece orientation use policy signs, avoiding
a rounded shoelace sign for floating input.

Precision tiers: integral signs exact within bounds; float/double signs exact
under `exact` within the [orientation expansion requirements](geometry.md);
no new coordinates are constructed, only indices. For `fast` or `approx`,
signs must remain consistent with the actual simple-polygon topology; these
policies cannot guarantee a triangulation on ambiguous ties. Validate a polygon
of constructed vertices using a tolerance appropriate to its source first.

## Measured performance

`benchmarks/geometry/triangulation.cpp` measures full-call throughput on convex,
reflex-heavy and collinear-repair inputs. It measures the adaptive library
entry, except the last row directly exercises the monotone sweep. The convex
construction merges parallel edge directions, so its actual vertex count can
be below the nominal 1e5 generated edges. No separate contender is timed.

Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, assertions
off, 2026-10-15; `-O2 -g -DNDEBUG -std=c++23`, optionally `-march=native`.
Minimum per-run means, three runs of 25 samples; milliseconds.

| Shape | -O2 | native |
|---|---:|---:|
| exactly convex, nominal 1e5 | 4.4 | 4.9 |
| random star, 1e5 | 184 | 187 |
| random star, 1e5, double | 192 | 194 |
| adversarial comb, 2e4 | 22.7 | 23.7 |
| collinear-top comb, 1e5, sweep/repair | 131 | 134 |

These records establish shape costs, not an independently reproducible numeric
optimum for the dispatch threshold. The benchmark retains every shape.

## Pitfalls and notes

Do not repeat the initial vertex at the end. Collinear boundary vertices must
participate in nonzero triangles: deleting them changes index coverage. Tests
compare certificates rather than requiring one triangulation on ties. Rounded
triangle moments, such as a centroid accumulated from these triples, still
need an application tolerance despite exact index selection.

## How it works

Ear clipping maintains an intrusive list of reflex blockers; closed-triangle
containment blocks diagonals through their vertices. A straight chain on a
prospective diagonal has a reflex endpoint there unless the whole remainder
would be collinear. A count of non-straight corners guards that exception in
constant time, retaining every straight vertex without scanning straight chains
for each ear.
The sweep follows de Berg et al., Computational Geometry: Algorithms and
Applications (2008), classifying vertices into five types and inserting helper
diagonals. Exact angular adjacency extracts y-monotone faces, each triangulated
by a two-chain stack. Straight vertices can produce zero-area stack triangles;
a local flip across the interior long edge repairs them using an edge-owner
map. A final shared-edge map yields the dual tree. Independent exact
certificates check area partition, manifold edges, no T-vertices and coverage.

## See also

[Polygons](polygon.md), [Segment intersections](segment-intersections.md),
[Delaunay](delaunay.md), [Geometry overview](geometry.md).
