# `atl::geometry::closest_pair`: the closest pair of a point set

`closest_pair_indices` returns the indices of two points at the smallest
distance, `closest_pair` the points themselves, in O(n log n) by an x-sweep
over a window ordered by y. Reach for it for nearest-pair questions on one set
(collision checks, minimum spacing); the selection is EXACT for integral
coordinates, since every decision compares exact squared distances in the
widened types. The closest pair between two different sets is not provided.
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
Its cell-scale correction needs fresh performance measurements. Each row is one call on 1e6 int64 points:
uniform in a 2e9 square; a Gaussian cluster with duplicates, where the
distance-0 exit decides; the cluster without duplicates ("dedup"); a strip
x < 1000 and one vertical line, where the window holds almost every point.
Every contender must find the library's distance.

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, assertions
off, 2026-10-15. Flags `-O2 -g -DNDEBUG -std=c++23`, optionally
`-march=native`. Minimum per-run mean over three runs of 25 samples;
milliseconds. Spreads 1.00x to 1.07x except where marked.

| Shape | Contender | -O2 | native |
|---|---|---:|---:|
| uniform 1e6 | library | 162 | 163 |
| uniform 1e6 | by divide and conquer | 290 | 275 |
| uniform 1e6 | by randomized grid | 1201 | 1126 |
| cluster 1e6 | library | 110 | 111 |
| cluster 1e6 | by divide and conquer | 146 | 142 |
| cluster 1e6 | by randomized grid | 144 (spread 1.74x) | 52.8 (spread 4.20x) |
| dedup cluster | library | 161 | 160 |
| dedup cluster | by divide and conquer | 314 | 301 |
| strip x < 1000, 1e6 | library | 283 | 280 |
| strip x < 1000, 1e6 | by divide and conquer | 317 | 311 |
| vertical line, 1e6 | library | 519 | 516 |
| vertical line, 1e6 | by divide and conquer | 261 | 264 |

Divide and conquer takes 79% longer than the library on uniform points at
`-O2` (69% native), 33% (28%) on the cluster, 95% (88%) on the dedup cluster
and 12% (11%) on the strip; it wins only the vertical line, where the library
takes 99% longer (95% native) because its window holds every point. The grid
takes 7.4x the library's time on uniform points (6.9x native); on the cluster
it can stop at the first distance 0, but how soon depends on the shuffle, and
its runs spread up to 4.2x. The library keeps the sweep: the vertical line is
not worth a slower common case, and the constexpr capability of divide and
conquer does not pay for 79% at run time.

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
