# `atl::geometry::delaunay`

Delaunay triangulations and their Voronoi dual, with Euclidean minimum
spanning trees, bichromatic closest pairs and largest empty circles.
Use these for neighborhood graphs and empty-disk queries; use the simpler
closest-pair sweep for one nearest pair in one set.

```cpp
#include <atl/geometry/delaunay.hpp>
```

\include delaunay/quick.cpp

Expected stdout:

\verbinclude delaunay/quick.out

## API

All names are in `atl::geometry`. `R`, `RA`, `RB` model `vertex_range`;
bichromatic calls require `std::same_as<vertex_coordinate_t<RA>,
vertex_coordinate_t<RB>>`. Coordinates model `coordinate`, including narrow
signed integers but excluding character types, unsigned and long double.
All result-only functions are `[[nodiscard]]`. None of these algorithms is
`constexpr`: they build runtime quad-edge state or use runtime math.

```cpp
struct delaunay_result {
    static constexpr triangle_index no_neighbor = ~triangle_index(0);
    std::vector<std::array<vertex_index, 3>> triangles;
    std::vector<std::array<triangle_index, 3>> neighbors;
};
struct voronoi_edge {
    vertex_index site_a, site_b;
    point<float64> from, to, direction;
    bool is_unbounded;
};
struct voronoi_diagram {
    std::vector<point<float64>> vertices;
    std::vector<voronoi_edge> edges;
};
```

`triangles` are CCW triples into the original input. `neighbors[t][e]`
is the triangle across the directed edge from `triangles[t][e]` to
`triangles[t][(e + 1) % 3]`, or `no_neighbor` on the hull. Duplicate positions
use their smallest input index. Triangle order and cocircular diagonals are
unspecified; the empty-circumcircle certificate is non-strict. Collinear
sets, and sets with fewer than three distinct sites, have no triangles.

`vertices[t]` is triangle t's rounded circumcenter. A dual edge names its two
input sites; bounded edges use `from,to`, hull rays use `from,direction`
(outward, unnormalized). Only the fields selected by `is_unbounded` matter.
Group edges by either site to obtain cell boundaries; no cell order is
promised. Cocircular triangles can give zero-length dual segments. With no
triangles the diagram is empty: bisector lines are outside this representation.

| Signature | Result |
|---|---|
| `template<vertex_range R> delaunay_result delaunay(R &&points)` | Triangles and across-edge adjacency |
| `template<vertex_range R> std::vector<std::array<vertex_index, 2>> delaunay_edges(R &&points)` | Undirected edges; collinear sets give the sorted path, duplicates collapse |
| `template<vertex_range R> voronoi_diagram voronoi_from_delaunay(R &&points, delaunay_result const &d)` | Rounded dual of d, which must be produced from these same points |
| `template<vertex_range R> std::vector<std::array<vertex_index, 2>> euclidean_mst(R &&points)` | n - 1 edges spanning every input index, duplicates linked by zero-length edges |
| `template<vertex_range RA, vertex_range RB> std::array<vertex_index, 2> closest_pair_between_indices(RA &&a_points, RB &&b_points)` | One index per input minimizing distance; shared positions give zero |
| `template<vertex_range RA, vertex_range RB> std::array<point<vertex_coordinate_t<RA>>, 2> closest_pair_between(RA &&a_points, RB &&b_points)` | The same pair as points |
| `template<vertex_range R> circle<float64> largest_empty_circle(R &&points)` | Largest empty disk with center in the filled hull, a rounded construction |

MST and closest-pair choices on ties, endpoint order and edge order are
unspecified. Integral squared-distance selection is exact; floating distance
keys are rounded and near-ties can select either pair/tree. MST duplicate
edges are appended, so the returned tree is not globally sorted by length.
The empty-circle construction has no uniform accuracy bound for thin hulls.
Collinear inputs optimize the largest consecutive gap; a single site gives
radius zero.

## Complexity

Delaunay, edges, MST and bichromatic queries take O(n log n) time (n is the
combined size for bichromatic calls). Allocated memory is O(n log n) worst
case: deleted edge ids are retained through all merge levels. Recursion is
balanced, O(log n) deep, with no recursion per point; large expansion arrays
are thread-local or bounded stack buffers. Voronoi extraction takes O(t)
time and memory for t triangles. Largest empty circle takes O(n log n + n h)
worst-case time and O(n log n) storage for h hull vertices: dual edges crossing
the hull are clipped against its boundary.

## Preconditions

Asserted: fewer than `2^32` input points (combined for bichromatic calls),
nonempty MST and empty-circle inputs, both bichromatic inputs nonempty.
**int32 accepts its full range**; **int64 requires `|x|, |y| < 2^61`** at
each entry, including the Voronoi conversion and shared-position shortcuts.
Storage limits: fewer than `2^32` directed edges, including retained deleted
ones, and at most `2^32 - 2` triangles. These limits are asserted before
narrowing to uint32 ids. Allocation failure propagates normally.

Incircle has degree four. The input bounds scan selects one of two paths:
when every coordinate has magnitude below `2^32`, the adaptive binary64
filter handles exactly converted integers; otherwise three exact signed
128-by-128 products are summed in int256. The wide path is not measured yet.
With coordinates below `2^61`, differences are below `2^62`, quadratic
factors below `2^125`, and the determinant below `2^252`. Orientation and
squared distances widen differences before multiplication and fit int128.
The standalone incircle sign uses the same bounds and paths, with int256
also used during constant evaluation. Integer-valued doubles of magnitude
below `2^52` have exact differences; larger ones can need subtraction tails.

Documented only: the diagram argument must be the result for the same
unchanged input. Floating expansion intermediates must stay finite and
nonzero products normal, including tails and degree-four products; the
orientation-only magnitude guidance is insufficient for incircle.
A sufficient scale envelope is nonzero coordinate magnitudes from `2^-200`
to `2^200` (their binary64 product quanta stay above subnormal range).
Cheap coordinate-square and main fourth-degree product checks are asserted;
there is no rescaling. Round-to-nearest, no unsafe reassociation, and the
[overview's](geometry.md) x87 rounding barriers apply.

Precision tiers: exact integral signs within bounds; exact adaptive signs
for float/double input within the expansion preconditions; rounded Voronoi
and empty-circle outputs require application tolerances. `largest_empty_circle`
clips already constructed values, so its numerical decisions are best effort.

## Measured performance

`benchmarks/geometry/delaunay.cpp` measures full-call throughput and races
the library incircle filter against `incircle_sign_by_int256`, which always
uses integer products. Exactly representable integer inputs use the adaptive
double filter, with int256 for larger integers and constant evaluation.
Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, assertions
off, 2026-10-15; `-O2 -g -DNDEBUG -std=c++23`, optionally `-march=native`.
Minimum per-run means over three runs of 25 samples; ms, spreads 1.00–1.06x.

| Operation / shape | -O2 | native |
|---|---:|---:|
| incircle int64 library, 1e6 | 26.2 | 23.2 |
| incircle int64 by int256, 1e6 | 64.6 | 61.5 |
| incircle int32 library, 1e6 | 24.1 | 20.9 |
| incircle int32 by int256, 1e6 | 59.8 | 59.3 |
| Delaunay int64 uniform 2e5, 2e9 span | 346 | 360 |
| Delaunay double, same cloud | 331 | 322 |
| Delaunay int64 grid 450 by 450 | 160 | 159 |
| Voronoi extraction, uniform 2e5 | 50.8 | 49.5 |
| Largest empty circle, uniform 2e4 | 32.4 | 33.3 |
| Largest empty circle, uniform 2e5 | 432 | 445 |

`benchmarks/geometry/manhattan_mst.cpp`, same CPU, compiler, build and date,
runs `euclidean_mst` beside the L1 race on identical clouds. These compute
different trees; the Euclidean agreement check is only n - 1 edges.

| Euclidean MST shape | -O2 ms | native ms |
|---|---:|---:|
| uniform 100k, 1e6 span | 172 | 172 |
| gridded 100k, 300 by 300 | 67.2 | 67.3 |

The edge-sort comparison (2026-10-14, same CPU, GCC 15.2, RelWithDebInfo,
assertions off, `-O2`) measured 25.9 ms for precomputed squared-length keys
versus 73.4 ms for a comparator recomputing both lengths on uniform 1e5.
Precomputation also gives stable comparisons on x87. Bichromatic pair
selection is not measured yet. An all-sites nearest-site scan per empty-circle
candidate measured 3751 ms at 2e4 points (same CPU/compiler/build, assertions
off, `-O2`, 2026-10-14); using the candidate's known nearest site avoids that
quadratic scan.

## Pitfalls and notes

Voronoi coordinates are constructions, and exact topology does not make
rounded circumcenters exact. Never sort using an epsilon relation or
recompute floating construction keys inside a comparator. A collinear
Delaunay edge path is useful even though its triangle/dual results are empty.

## How it works

A lexicographic sort deduplicates positions and preserves the smallest input
index. Guibas and Stolfi (1985, Primitives for the manipulation of general
subdivisions and the computation of Voronoi diagrams) divide the sorted sites
in halves. A primal quad-edge store maintains origin rings and their inverses;
the merge rises from the lower common tangent, removing candidates whose
circumcircles contain the opposing vertex. Triangular CCW face walks recover
indices and adjacency. Balanced splits bound stack depth even on collinear
or duplicate-heavy million-point inputs.

Incircle uses a static error filter, exact expansions of rounded differences,
then expansions of two-component differences when subtraction tails matter.
Integral coordinates exactly convertible to binary64 share that filter;
the fallback accumulates widened products in int256. Voronoi vertices are
circumcenters and hull edges dualize to outward normals. Kruskal uses
precomputed squared-length keys and then attaches duplicates. A closest
bichromatic pair lies on a Delaunay edge of the colored union; shared
positions are detected separately. Empty-circle candidates are Voronoi
vertices in the hull and dual/hull crossings; their nearest site is already
known, avoiding an all-sites scan at every candidate.

## See also

[Circles and rays](circle.md), [Closest pair](closest-pair.md),
[Manhattan MST](manhattan-mst.md), [Geometry overview](geometry.md).
