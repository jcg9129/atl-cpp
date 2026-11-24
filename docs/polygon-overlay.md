# `atl::geometry::overlay_of`

Combine two simple polygon boundaries into one subdivision, with exact integral
face-membership flags. Read several Boolean areas from one overlay, or inspect
the half-edge graph for components and holes. Use convex polygon operations
when both inputs are convex and that simpler representation suffices.

```cpp
#include <atl/geometry/polygon_overlay.hpp>
```

\include polygon-overlay/quick.cpp

Expected stdout:

\verbinclude polygon-overlay/quick.out

For integral coordinates, `<atl/geometry/polygon_overlay_integral.hpp>` provides
the same overlay operations without adaptive floating-point kernels.
`<atl/geometry/polygon_overlay.hpp>` retains support for every coordinate type.

## API

All names are in `atl::geometry`. `R1`, `R2` model `vertex_range` and require
`std::same_as<vertex_coordinate_t<R1>, vertex_coordinate_t<R2>>`.

```cpp
template<coordinate T> struct polygon_overlay {
    arrangement arr;
    std::vector<segment<T>> segments;
    uint32 first_edge_count;
    std::vector<char> in_first;
    std::vector<char> in_second;
};
template<vertex_range R1, vertex_range R2>
    requires std::same_as<vertex_coordinate_t<R1>, vertex_coordinate_t<R2>>
polygon_overlay<vertex_coordinate_t<R1>> overlay_of(R1 &&first_polygon, R2 &&second_polygon);
```

`segments` owns cyclic edges of the first polygon followed by the second's,
with `first_edge_count` marking the split. `arr` is built from that list;
`in_first[f]` and `in_second[f]` are 0/1 membership of face f's open interior.
They are both zero for the unbounded face. Either winding is accepted;
collinear straight vertices and shared input boundary pieces are legal.

Each of `intersection_area`, `union_area`, `difference_area` and
`symmetric_difference_area` has these forms (substitute its name for area):

```cpp
template<coordinate T> constexpr float64 area(polygon_overlay<T> const &overlay);
template<vertex_range R1, vertex_range R2>
    requires std::same_as<vertex_coordinate_t<R1>, vertex_coordinate_t<R2>>
float64 area(R1 &&a, R2 &&b);
```

Difference means first minus second. Results are rounded float64 constructions;
the membership flags are exact for integral inputs within bounds. No boundary
polygon extraction function is exposed: the arrangement retains its full graph.

The four area functions taking a `polygon_overlay<T> const &` are constexpr
face sums. `overlay_of` itself requires a runtime arrangement sweep; a manually
assembled constexpr face/membership fixture can exercise the sums at compile time.
Runtime-built overlays use the same functions and retain rounded face areas.

## Complexity

For n combined edges and k boundary intersections, build takes expected
O((n + k) log(n + k + 1)) time and O(n + k) memory. Each area read takes
O(faces) time and O(1) extra space; a two-polygon call builds an overlay first.
Parity propagation is iterative. APIs are runtime-only, using arrangement
containers even when an area read only loops over faces.

## Preconditions

Asserted: both inputs are simple (`simplicity_of` under ATL_ASSERT), combined
edge count below `2^32`, and the arrangement's index bounds. int64 needs
`|x|, |y| < 2^32`; **int32 accepts its full range**. The
[sweep's degree-five bound](segment-intersections.md) supplies exact integral
topology. Empty/degenerate polygons are outside the simple-polygon contract.

Float/double topology is best effort, with finite input and finite intermediate
products required. Area uses rounded vertices even for exact integer topology;
there is no uniform relative error bound for arbitrarily thin intersections.
Choose application tolerances for all area comparisons. There is no exact
floating-input tier for the overlay sweep.

## Measured performance

Not measured yet. The [arrangement benchmark](arrangement.md) measures the
underlying builder; it does not measure polygon validation or membership BFS.

## Pitfalls and notes

Retain the overlay while a `point_location(o.arr, o.segments)` references it.
Face flags describe open interiors: boundary membership must use the edge's
supports or a polygon predicate. Reading multiple combinations from one overlay
avoids repeated sweeps. A difference can have holes and disconnected pieces.

## How it works

The arrangement merges overlapping boundary pieces. Starting at the unbounded
face, a breadth-first walk crosses each edge and toggles membership in each
input whose boundary supports the edge. Jordan parity makes every route agree;
shared pieces toggle both flags. Selected face areas are summed for each Boolean
operation, so no rounded representative point is needed to classify a face.
Independent exact cell/triangle oracles and area identities validate the flags
and rounded constructions on random small inputs, including shared boundaries.

## See also

[Arrangements](arrangement.md), [Point location](point-location.md),
[Convex polygon operations](convex-polygon-operations.md), [Polygons](polygon.md).
