# `atl::geometry::arrangement_of`

Build the planar subdivision induced by closed segments, including shared
pieces, holes, slits and isolated points. Use the half-edge result for boundary
walks, polygon overlay or repeated point location.

```cpp
#include <atl/geometry/arrangement.hpp>
```

\include arrangement/quick.cpp

Expected stdout:

\verbinclude arrangement/quick.out

## API

`template<segment_range R> arrangement arrangement_of(R &&segments)` accepts
sized random-access ranges of segments and owns its result. Coordinate types
are those of [the overview](geometry.md). No input lifetime is retained.

`arrangement` contains these public members:

| Member / signature | Meaning |
|---|---|
| `static constexpr uint32 none` | Absent-index sentinel |
| `static constexpr uint32 unbounded_face` | Zero, the unbounded face |
| `std::vector<point<float64>> vertices` | Rounded positions, ids in exact sweep order for integer inputs |
| `std::vector<uint32> origin, next, face` | Per half-edge source vertex, cycle successor, left face |
| `std::vector<std::vector<uint32>> supporting` | Per undirected edge, every input segment containing that piece |
| `std::vector<face_data> faces` | Regions, unbounded first |
| `static constexpr uint32 twin(uint32 h)` | `h ^ 1` |
| `static constexpr uint32 edge_of(uint32 h)` | `h >> 1` |
| `constexpr uint32 to(uint32 h) const` | Target vertex; h must be a valid half-edge |
| `constexpr uint32 edge_count() const` | Number of undirected edges |

`face_data` has `uint32 outer` (a half-edge, none for the unbounded face),
`std::vector<uint32> holes` (one representative per inner cycle, including
zero-area slits), `std::vector<uint32> isolated` (isolated vertex ids), and
`float64 area` (signed cycle sum, bounded area net of holes). `next` walks
with face interior on the left: bounded outer cycles CCW, holes CW. Half-edge
`2e` points lexicographically forward, with its left side above the edge.
Empty input still has one unbounded face. Non-vertex output ordering is not
an API guarantee; compare structural certificates when changing builders.

## Complexity

Expected O((n + I) log(n + I + 1)) time with constant-time hashing, O(n + I) allocated storage, including angular
stitching and support lists; I counts segment/event incidences, which can be
quadratic for nested overlaps. All subdivision walks are iterative. Building
is runtime-only; the four scalar accessors are constexpr.

## Preconditions

Asserted: int64 endpoints `|x|, |y| < 2^32`; **full-range int32** and narrower
admitted integers; fewer than `2^32` input segments and vertices and `2^31`
undirected edges. All narrowing bounds are checked at their assembly points.
Allocation failure propagates. The [sweep's degree-five int256 argument](segment-intersections.md)
applies; angular directions widen to int64, avoiding subtraction in int32.

Topology is exact for integral input within bounds; positions and areas are
rounded constructions without a universal relative error bound for tiny faces.
Floating topology is best effort and inherits the sweep's finite-intermediate
requirements. Use tolerances for area certificates, including translated inputs.

## Measured performance

`benchmarks/geometry/arrangement.cpp` races the library sweep against
`atl::experimental::arrangement_of_by_all_pairs` from
`atl/experimental/geometry/arrangement_all_pairs.hpp`: direct pair intersections,
one vertex sort, and the same half-edge assembly. It avoids a sweep status
but pays for every pair and scans below-anchors on demand.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape / contender | -O2 | native |
|---|---:|---:|
| sparse 5000, library | 7.9 | 7.8 |
| sparse 5000, by all pairs | 1300 | 1130 |
| dense 700 chords, library | 175 | 182 |
| dense 700 chords, by all pairs | 151 | 160 |

The library wins the sparse shape (7.9/7.8 versus 1300/1130 ms), but loses
the dense shape to all pairs in both modes (175/182 versus 151/160 ms). The
contender accepts integral coordinates only, with **both int32 and int64
magnitudes below `2^26`**, asserted. Its anchor numerator has degree four
and magnitude below `64 B^4`, within int128; subsequent products use int256.
Its expected cost is O(n^2 + (n + k) log(n + k + 1) + n H) for k pair intersections
and H anchors, using O(n + k) transient storage; it owns the same result type
and is runtime-only.

## Pitfalls and notes

Do not infer topology from rounded vertex equality or area signs. Overlapping
segments share one piece with multiple supports; point segments can attach
to an edge or form isolated vertices. For geometric queries retain the exact
input segments beside the arrangement. Mutating arrays can invalidate its
invariants and any point-location structure that references it.

## How it works

A singleton-reporting segment sweep emits every subdivision vertex and its
below-neighbor anchor. Consecutive cuts on each segment form pieces; identical
vertex pairs merge. Exact supporting directions sort outgoing half-edges,
then clockwise rotation at the target defines each left-face walk. At a cycle's
leftmost vertex, cancelling twin pairs removes slit contributions and the
remaining angular extreme determines its orientation. Inner cycles and
isolated vertices attach through below-anchors using iterative component
compression. Each cycle's area uses a local anchor to reduce cancellation.
Tests check rational brute-force V/E/components, Euler's formula, support
coverage, cycle invariants and zero signed total area.

## See also

[Segment intersections](segment-intersections.md), [Point location](point-location.md),
[Polygon overlay](polygon-overlay.md), [Geometry overview](geometry.md).
