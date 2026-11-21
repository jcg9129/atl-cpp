# `atl::geometry::max_arc_overlap` and `max_points_in_circle`: the radial sweep

`max_arc_overlap` finds the direction covered by the most closed arcs of
directions, and `max_points_in_circle` the most points one disk of a fixed
radius can cover, built on it. Reach for them for angular coverage questions
(visibility cones, beams, "how many points fit in a circle of radius r");
both work on rounded angles, so they are construction-grade: there is no exact
tier for irrational distances and angles, and counts are best-effort only for
points within rounding of a disk's boundary. The coordinate types are explained
in the [geometry overview](geometry.md).

```cpp
#include <atl/geometry/radial_sweep.hpp> // arc, max_arc_overlap, max_points_in_circle
```

\include radial-sweep/quick.cpp

Expected stdout:

\verbinclude radial-sweep/quick.out

## API

Everything is in namespace `atl::geometry`, `[[nodiscard]]`, and not
`constexpr` (`std::fmod`, `std::atan2`, `std::acos`, `std::sqrt`).

```cpp
struct arc { float64 from{}, to{}; };          // closed, counterclockwise from `from` to `to`, radians
template<typename R> concept arc_range = /* sized random-access range whose value type is arc */;
struct arc_overlap { uint32 count{}; float64 angle{}; };
struct circle_cover { uint32 count{}; point<float64> center{}; };
```

| Signature | Result | Complexity |
|---|---|---|
| `template<arc_range R> arc_overlap max_arc_overlap(R &&arcs)` | the most arcs containing one direction, and such a direction in `[0, 2 pi)` (the `from` of some arc); `{0, 0.0}` for no arcs | O(n log n) time, O(n) memory |
| `template<vertex_range R> circle_cover max_points_in_circle(R &&points, float64 radius)` | the most points one closed disk of the radius covers, and a center achieving it (rounded) | O(n log n + n k log(k + 1)) for at most k points within `2 radius` of each point; O(n^2 + n k log(k + 1)) when the grid is skipped; radius zero O(n log n); O(n) memory |

An arc's span is the counterclockwise distance from `from` to `to` reduced
into `[0, 2 pi)`: any finite radians are accepted, `from == to` is a single
direction, and a full circle is not representable (count elements that cover
every direction outside the sweep). Arcs touching at an endpoint overlap.

`max_points_in_circle` counts points at equal positions separately (all of
them), and `radius == 0` gives the size of the largest group of equal points.
Coordinate differences are taken exactly in the widened type before they are
rounded, so there is no coordinate bound: a cloud near the `int64` limits gets
the same arcs as near the origin. The grid of cell side a little over
`2 radius` is skipped, and every pair scanned, when the bounding box spans
`2^30` cells or more on a side. Radius zero sorts the original points and
counts equal runs using exact point equality in O(n log n), including tiny
floating coordinates whose squares would underflow.

## Preconditions

Asserted: `size(arcs) < 2^32`; `1 <= size(points) < 2^32` and
`radius >= 0`; finite angles, radius and coordinates. For a positive radius,
all squares and pairwise products of coordinates, differences and radius
formed by the function must be finite and normal unless a factor is zero;
their sums must remain finite. Radius and coordinate squares and visited
difference squares are asserted. There is no rescaling.

## Measured performance

`benchmarks/geometry/radial_sweep.cpp` times one full call per row; nothing is
raced. Shapes: uniform clouds in a 10k box at radius 300; a dense 2000-point
cloud where every pair is within `2 radius` (the worst case: each anchor sorts
n - 1 arcs and the grid cannot help); a 500-point cluster in a uniform
background, where most anchors cannot beat the cluster's count and skip their
sweep; 1e5 points in a 1e6 box; and `max_arc_overlap` on 1e6 random arcs.

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, assertions
off, 2026-10-15. Flags `-O2 -g -DNDEBUG -std=c++23`, optionally
`-march=native`. Minimum per-run mean over three runs of 25 samples;
milliseconds. Spreads 1.00x to 1.02x.

| Shape | -O2 | native |
|---|---:|---:|
| `max_points_in_circle`, uniform 2000, r = 300 | 6.5 | 6.4 |
| `max_points_in_circle`, uniform 5000, r = 300 | 42.3 | 41.5 |
| `max_points_in_circle`, dense 2000, r = 400 (all arcs) | 283 | 278 |
| `max_points_in_circle`, cluster 500 + background 4500, r = 300 | 60.1 | 59.3 |
| `max_points_in_circle`, uniform 1e5 in a 1e6 box, r = 300 | 56.0 | 52.8 |
| `max_arc_overlap`, 1e6 random arcs | 187 | 190 |

The dense shape costs 44x the uniform 2000 shape of the same size at `-O2`
(283 against 6.5 ms): all n - 1 arcs per anchor against a handful.

## Pitfalls and notes

- Angles compare as rounded `float64` values: an arc ending within rounding
  of another's start may or may not meet it. Where an exact angular order is
  needed, use the polar-angle comparator of the
  [primitives](geometry-primitives.md) on direction vectors instead.
- Many coincident points make every anchor scan its whole group: an input of
  n equal points with a positive radius costs O(n^2).

## How it works

**The sweep.** Each arc becomes an entry and exit at its separately reduced
endpoints. Signed remainders of both original angles are taken before
subtracting to find the span, so extreme finite angles neither lose the other
endpoint nor overflow. A wrapping arc is counted open at angle 0. Rounding can carry such a wrapped exit to
the arc's own entry, which would count it twice there, so it is clamped just
below the entry. Entries and exits are sorted as two plain arrays and merged:
before each entry, every exit strictly smaller closes (ties keep the arc open,
since arcs are closed), then the entry opens; the best count is recorded at an
entry, which is where a maximum is always reached.

**The circle cover.** Some optimal disk has an input point on its boundary
(move the disk until one touches). For each anchor point the possible centers
form the circle of radius r around it, and a neighbor at distance d up to 2r is
covered by the centers on an arc of half-width `acos(d / 2r)` around the
direction to it; the best count is the anchor and its copies plus the best arc
overlap. Neighbors come from a grid of cell side `2r (1 + 2^-16)` over exact
coordinate differences to the bounding box's lower left corner: each anchor
scans its 3 by 3 block of cells, found by binary search in the cells sorted by
(column, row). With cell quotients below `2^30` the rounding of a quotient is
far below the side's slack, so a true neighbor is never more than one cell
away. An anchor whose arc count cannot beat the best so far skips its sort and
sweep, and the entry and exit arrays are reused across anchors.

## See also

[Geometry overview](geometry.md) for the coordinate types and the precision
tiers; [Closest pair](closest-pair.md) for the other point-set search over
widened differences.
