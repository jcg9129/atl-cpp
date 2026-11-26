# `atl::geometry::manhattan_mst`: the minimum spanning tree under the L1 distance

`manhattan_mst` returns a minimum spanning tree of a point set under the
Manhattan distance `|dx| + |dy|` in O(n log n), without building the complete
graph: four octant sweeps give every point a nearest neighbor per octant, and
Kruskal's algorithm runs over those at most 4n candidate edges. Reach for it
for rectilinear wiring, grid routing or any L1 clustering; for integral
coordinates the tree is EXACTLY minimal over the full range of the type, with
no coordinate bound. The coordinate types are explained in the
[geometry overview](geometry.md).

```cpp
#include <atl/geometry/manhattan_mst.hpp> // manhattan_mst
```

\include manhattan-mst/quick.cpp

Expected stdout:

\verbinclude manhattan-mst/quick.out

## API

| Signature | Result | Complexity |
|---|---|---|
| `template<vertex_range R> std::vector<std::array<vertex_index, 2>> manhattan_mst(R &&points)` | `n - 1` edges as index pairs, spanning every index, in nondecreasing length order | O(n log n) time, O(n) memory |

In namespace `atl::geometry`, `[[nodiscard]]`, not `constexpr` (`std::map`).
Equal points join through edges of length 0. Which of several minimum
spanning trees is returned, and the order of the two indices of an edge, is
unspecified. No recursion.

Exactness per coordinate type: the sweep keys and lengths are sums and
differences of coordinates in `product_t<T>` (`int32` for `int16`, up to
`int128` for `int64`), exact for every integral input, the full `int64` range
included. Floating coordinates use rounded `double` keys (exact for `float`
coordinates whose sums fit 53 bits), so near-ties may be broken either way and
the tree is minimal up to that rounding.

## Preconditions

Asserted: `1 <= size(points) < 2^32`. Floating coordinate squares
and pairwise products must be finite and normal unless a factor is zero.
Squares are checked before sorting in both the library and the Fenwick
contender, rejecting NaNs and overflow before they can invalidate ordering.
There is no rescaling.

## Measured performance

`benchmarks/geometry/manhattan_mst.cpp` races the library against
`atl::experimental::manhattan_mst_by_fenwick_prefix_minimum` (in
`experimental/atl/experimental/geometry/manhattan_mst_variants.hpp`), which
collects the same candidate edges through a Fenwick prefix minimum over ranked
x instead of an ordered map ("by Fenwick prefix minimum"). Each row is one call
on int64 points: 1e5 uniform in a 1e6 square; 4e5 uniform over a 1e9 span; 1e5
on a 300 by 300 grid, heavy with duplicates and ties; a noisy descending
staircase of 1e5 points with every x distinct. Both must give trees of equal
weight. The [Euclidean minimum spanning tree](delaunay.md) runs beside these on the
same uniform and gridded clouds. It builds a different tree; its agreement
check is only `size() == n - 1`.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | Contender | -O2 | native |
|---|---|---:|---:|
| uniform 1e5 | library | 127 | 127 |
| uniform 1e5 | by Fenwick prefix minimum | 136 | 141 |
| uniform 1e5 | euclidean_mst | 108 | 121 |
| uniform 4e5, 1e9 span | library | 627 | 669 |
| uniform 4e5, 1e9 span | by Fenwick prefix minimum | 726 | 803 |
| gridded 1e5 (many ties) | library | 85.3 | 90.5 |
| gridded 1e5 (many ties) | by Fenwick prefix minimum | 92.2 | 97.6 |
| gridded 1e5 (many ties) | euclidean_mst | 48.3 | 47.5 |
| staircase 1e5 (distinct x) | library | 83.2 | 89.8 |
| staircase 1e5 (distinct x) | by Fenwick prefix minimum | 83.9 | 87.9 |

Fenwick takes 7.09%/11.02% longer on 1e5 uniform points and
15.79%/20.03% longer at 4e5 (plain/native). It also loses the grid
(92.2/97.6 versus 85.3/90.5 ms). The staircase favors the library plain
(83.2 versus 83.9 ms), but the library loses native (89.8 versus 87.9 ms).
No speed comparison is established for the Euclidean edge-sort experiment.


## Pitfalls and notes

- The tree is over the L1 distance; for the Euclidean tree the candidate set
  is different (the Delaunay edges).
- Edges come sorted by length, so a single-linkage clustering into k groups
  is the tree minus its last k - 1 edges.

## How it works

By the octant lemma, for each point it suffices to connect it to a nearest
point in each of the eight 45-degree octants around it: some minimum spanning
tree uses only such edges. Edges are undirected, so the four octants of the
upper half plane cover all eight. Each of four sweeps maps one octant onto the
canonical one (up and to the right, at least 45 degrees steep) by swapping x
and y and negating x, then visits the points by increasing `x + y`: inside the
octant the L1 distance from a point to a later one is exactly the difference of
their sums. Active points wait in a `std::map` keyed by x in decreasing order.
A new point looks at the active keys at most its own x, from the greatest
down; along them `x - y` strictly decreases, so it matches each active point
that has it in its octant (`x_j - y_j >= x_i - y_i`) until the first that does
not. Every match records a candidate edge and erases the matched point: the
new point is at least as close to every later arrival, so each point enters
and leaves the map once, and a sweep is O(n log n). The candidates (at most
4n) are sorted by length and Kruskal's algorithm over the library's
[disjoint-set union](dsu.md) keeps `n - 1` of them.

## See also

[Geometry overview](geometry.md) for the coordinate types and the widened
products; [Closest pair](closest-pair.md) for the nearest pair alone; [Disjoint-set
union](dsu.md) for the Kruskal step.
