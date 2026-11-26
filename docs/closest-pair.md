# `atl::geometry::closest_pair`: the closest pair of a point set

`closest_pair_indices` returns the indices of two points at the smallest
distance, `closest_pair` the points themselves, in O(n log n) by an x-sweep
over a window ordered by y. Reach for it for nearest-pair questions on one set
(collision checks, minimum spacing); the selection is EXACT for integral
coordinates, since every decision compares exact squared distances in the
widened types. For two sets, use `closest_pair_between_indices(a, b)` or
`closest_pair_between(a, b)` from [Delaunay](delaunay.md): the bichromatic
closest pair is an edge of the colored union's Delaunay triangulation.
Both inputs must be nonempty; integral selection is exact within that
family's bounds, shared positions give zero and ties are unspecified.
The coordinate types are explained in the [geometry overview](geometry.md).

```cpp
#include <atl/geometry/closest_pair.hpp> // closest_pair_indices, closest_pair
```

\include closest-pair/quick.cpp

Expected stdout:

\verbinclude closest-pair/quick.out

## API

| Signature | Result | Complexity |
|---|---|---|
| `template<vertex_range R> std::array<vertex_index, 2> closest_pair_indices(R &&points)` | indices of a closest pair, distinct | O(n log n) time, O(n) memory |
| `template<vertex_range R> std::array<point<T>, 2> closest_pair(R &&points)` | the two points, in the same order | as above |

Both are in namespace `atl::geometry`, `[[nodiscard]]`, and not `constexpr`
(`std::set`, `std::sqrt`). Equal points are at distance 0 and end the search at
once. Which of several equally close pairs is returned, and the order of its two
indices, is unspecified: it follows the order `std::ranges::sort` leaves equal
points in, which differs between standard libraries. Neither recurses.

Exactness per coordinate type: integral coordinates (`int16` to `int64`) give
an exactly closest pair; both `float` and `double` coordinates use double arithmetic,
where differences, squares and their sum can each round, so among pairs within rounding of the minimum any may be returned.

## Preconditions

Asserted: `2 <= size(points) < 2^32`; for `int64` coordinates
`|x|, |y| < 2^62` on EVERY point (a far point is never a candidate, but its x
gap is still squared). Floating coordinates and differences require finite,
normal squares and pairwise products unless a factor is zero; sums must stay
finite. Coordinate squares are checked before sorting.

## Measured performance

`benchmarks/geometry/closest_pair.cpp` races the library against two
contenders in `experimental/atl/experimental/geometry/closest_pair_variants.hpp`:
`closest_pair_indices_by_divide_and_conquer` ("by divide and conquer": a
recursion on the x-sorted points with the mergesort by y riding along,
16-point brute-force leaves and an integer strip radius; constexpr-capable)
and `closest_pair_indices_by_randomized_grid` ("by randomized grid": a hash
grid with a power-of-two side covering the running best distance in a seeded random order,
rebuilt on every improvement). Power-of-two division avoids cell-index
rounding gaps, and integer neighbor offsets remain distinct above `2^53`.
The grid has O(n^2) worst-case time; expected O(n) requires that its minimum
side and saturated indices do not merge unboundedly many nearby points.
Each row is one call on 1e6 int64 points:
uniform in a 2e9 square; a Gaussian cluster with duplicates, where the
distance-0 exit decides; the cluster without duplicates ("dedup"); a strip
x < 1000 and one vertical line, where the window holds almost every point.
Every contender must find the library's distance.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | Contender | -O2 | native |
|---|---|---:|---:|
| uniform 1e6 | library | 117 | 115 |
| uniform 1e6 | by divide and conquer | 221 | 218 |
| uniform 1e6 | by randomized grid | 804 | 724 |
| cluster 1e6 | library | 83.7 | 90.2 |
| cluster 1e6 | by divide and conquer | 116 | 119 |
| cluster 1e6 | by randomized grid | 36.7 (spread 2.47x) | 44 (spread 2.01x) |
| dedup cluster | library | 117 | 114 |
| dedup cluster | by divide and conquer | 236 | 234 |
| strip x < 1000, 1e6 | library | 216 | 219 |
| strip x < 1000, 1e6 | by divide and conquer | 233 | 228 |
| vertical line, 1e6 | library | 336 | 270 |
| vertical line, 1e6 | by divide and conquer | 167 | 159 |

Divide and conquer takes 88.89%/89.57% longer than the library on uniform
points, 38.59%/31.93% on the cluster, 101.71%/105.26% on the dedup cluster
and 7.87%/4.11% on the strip (plain/native). The library loses the vertical
line: 336/270 versus 167/159 ms, or 101.20%/69.81% longer. The grid takes
804/117 = 6.87 times the library's time plain on uniform points
(724/115 = 6.30 native). On the duplicate cluster the library loses to the
grid (83.7/90.2 versus 36.7/44 ms), but the grid row varies between runs,
with 2.47x/2.01x spread. The sweep wins the uniform,
deduplicated-cluster and strip comparisons with divide and conquer; there
is no winner across all shapes.


## Pitfalls and notes

- For many queries of "the nearest point to q", build a spatial structure
  instead; this is a one-shot answer for the whole set.
- Compare `squared_distance` values, not `distance`: the squared distance is
  exact for integers, the distance a rounded construction.

## How it works

The points are sorted by (x, y) with their indices. The sweep keeps the best
squared distance `best`, a window radius `w` with `w^2 >= best`, and a
`std::set` of active points keyed by (y, x, index). Each new point first
evicts from the left every point whose x gap squared is at least `best` (a
two-pointer walk, each point evicted once), then visits the active points
whose y lies within `w` of its own, from a `lower_bound` up to the first y
beyond, and finally joins the set. Active points within `w` in x and y are pairwise at least `sqrt(best)`
apart, so O(1) of them are visited (a packing argument) and each step is
O(log n); on one vertical line the set holds every point and only its size
costs.

For integral coordinates `w` is an INTEGER, the rounded-up square root of
`best` with a relative slack of `2^-50` plus 1, compared against exact widened
y differences: a `double` window would round by up to 512 at `|y|` near
`2^61` and could skip the true partner. For floating coordinates `w` is the
square root rounded up by one ulp, which by monotone rounding never drops a
candidate (a fixed slack of 1 would admit every point at small scales). Equal
points give `best == 0` and stop the sweep. The decisive comparisons are
`squared_distance` values in `difference_product_t`, exact for integers.

`distance_between_convex_hulls(a, b)` in `atl/geometry/calipers.hpp` measures
the distance between the filled hull regions, returning zero on overlap or
touch. See [Rotating calipers](calipers.md). Interior input points do not
affect that distance; it solves a different problem from a closest pair
with one point from each set.

## See also

[Geometry overview](geometry.md) for the coordinate types and the widened
products; [Manhattan MST](manhattan-mst.md) for the minimum spanning tree under
the L1 distance, another sweep over the same widened differences.
