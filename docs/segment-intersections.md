# `atl::geometry::segment_intersections`

Enumerate common points of closed segments, or stop a lazy sweep at the first
useful event. `simplicity_of` checks the simple-polygon precondition used by
polygon algorithms. Use pairwise `intersects` for just two segments.

```cpp
#include <atl/geometry/segment_intersections.hpp>
```

\include segment-intersections/quick.cpp

Expected stdout:

\verbinclude segment-intersections/quick.out

## API

All names are in `atl::geometry`. `segment_range<R>` is a sized random-access
range whose value type is `segment<T>`; `segment_coordinate_t<R>` names T.
T models the [coordinate concept](geometry.md). `segment_index` is uint32.

| Declaration | Meaning |
|---|---|
| `template<segment_range R> class segment_intersections_view` | Construct with `R &&segments`; input is copied, or moved from an rvalue segment vector |
| `template<segment_range R> std::vector<materialized_intersection> segment_intersections(R &&segments)` | Own every event and its segment lists, in sweep order |
| `template<vertex_range R> simplicity simplicity_of(R &&poly)` | Verify a cyclic polygon of either winding; straight vertices allowed |

An `intersection_event` has `point<float64> at`, spans of `segment_index const`
`starting`, `ending`, `passing`, and `segment_index below`. The partition is
relative to lexicographic endpoint order: starts, ends, and segment interiors.
Point segments appear in both starting and ending. List order is unspecified.
`below` is the highest active segment strictly below the event, with slope
then input index breaking height ties; `intersection_event::no_below` means none.
Spans expire on iterator increment. `materialized_intersection` has the same
fields with vectors replacing spans. `simplicity` contains `bool is_simple`
and `point<float64> conflict` (meaningful only for a reported conflict; fewer
than three vertices give false with a zero-initialized witness).

`begin()` consumes the first reportable event; call it once. The nested input
iterator dereferences to an event, supports pre/post increment and comparison
to `std::default_sentinel_t`. Copies consume the same engine. The range owns
its input but is neither copyable nor movable, because its comparator points
at its engine; it does not model `std::ranges::view`.

## Complexity

With I total segment/event incidences, the sweep takes O((n + I) log(n + 1))
time and O(n + I) memory; eager storage also includes all copied incidences.
Counting distinct events alone understates the cost of overlapping segments:
I can be quadratic with only linearly many event positions. Simplicity takes
O(n log(n + 1)) on simple inputs and stops at the first invalid event.
Ordered-container operations have logarithmic call depth; no recursion per
segment is used. These APIs are runtime-only because they use ordered containers.

## Preconditions

Asserted: fewer than `2^32` segments/vertices; int64 endpoints satisfy
`|x|, |y| < 2^32`. **int32 accepts its full range**, as do narrower admitted
signed integers. Point, vertical, duplicate and overlapping segments are legal.
For simplicity, a repeated closing vertex is a repeated vertex and fails.

Integral decisions are exact. If B bounds coordinate magnitudes, line
coefficients have degree one/two, crossing numerators degree three, and
denominators degree two: loose bounds are `|a|,|b| < 2B`, `|c| < 4B^2`,
`d < 8B^2`, `|nx|,|ny| < 16B^3`. These factors fit int128 at B = `2^32`;
status and event comparisons have degree at most five and fit int256
(magnitude below `256 B^5`). Software int128 uses the same code.

Floating sweeps use rounded rational arithmetic and **best-effort topology**,
even though their initial pairwise predicates use exact orientation. Finite
input and finite intermediate products/divisors are documented requirements;
no universal floating coordinate envelope protects arbitrarily close crossings.
The exact floating-input tier of the primitives therefore does not extend to
this sweep. Constructions require an application tolerance; integral endpoints
convert exactly, other events are rounded divisions with no uniform relative
error bound near zero.

## Measured performance

Not measured yet with a complete benchmark record for this family. The
[arrangement race](arrangement.md) exercises the same sweep with singleton
endpoints enabled; those whole-call timings are not intersection-only timings.

## Pitfalls and notes

Overlap continua are represented by set-change endpoints and crossings inside
the overlap; their interior points are not enumerated. Events are ordered by
exact rational positions for integers, not by their rounded output coordinates.
Distinct events may round to the same point. `simplicity_of` accepts only edges
meeting at their common cyclically adjacent endpoints and rejects backtracking,
pinches, zero-length edges and crossings.

## How it works

The Bentley–Ottmann event-point formulation in de Berg et al., Computational
Geometry: Algorithms and Applications (2008), maintains a rational event map
and a transparent status set. A point probe finds the entire active run
through an event. The run is removed, endings discarded, and passing/starting
segments reinserted with the slope tie order reversed. Only flanking neighbors
need new crossing events. Collinear overlaps need no continuum queue entries:
their endpoints already mark every set change. Simplicity accepts an event
only when exactly two cyclically adjacent edges meet at their shared vertex.

## See also

[Polygons](polygon.md), [Arrangements](arrangement.md), [Geometry overview](geometry.md).
