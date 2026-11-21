# `atl::convex_piecewise_linear_forest` (slope trick on trees)

**Many convex piecewise-linear functions over one shared breakpoint pool**,
for tree DPs whose state `f_v(x)` is convex in `x`: children combine by a
SUM of functions, and passing an edge up is an infimal convolution with a
convex kernel such as `D|t - C|`. Breakpoints are weighted (a slope change
of 1e6 is one breakpoint, not 1e6 copies), a sum is O(1), and every clip
reports where it cut, which is what a top-down reconstruction of the
optimal choices needs. Function values are not tracked, only the shape. For
one function of a sequence DP, with the minimum value and O(1) minimizers,
use [`convex_piecewise_linear`](convex-piecewise-linear.md), which shares
this vocabulary.

Two worked examples (experimental, not part of the library) apply it:
`atl::experimental::balance_leaf_paths` makes every root-to-leaf path of a
weighted tree equally long at the least weighted change of the edge costs,
and its dual `atl::experimental::tree_transport_cost` moves units along a
tree at the least cost so that every vertex meets its demand.

```cpp
#include <atl/dp/convex_piecewise_linear_forest.hpp>
// worked examples, experimental (needs -I experimental):
#include <atl/experimental/dp/leaf_path_balance.hpp> // balance_leaf_paths
#include <atl/experimental/dp/tree_transport.hpp>    // tree_transport_cost
```

Leaf-path balancing with free new costs, written out on the forest: the
bottom-up pass clips each child's slopes to the edge weight and records the
clip range, the top-down pass clamps each child's total into it.

\include convex-piecewise-linear-forest/quick.cpp

Output:

\include convex-piecewise-linear-forest/quick.out

## API

All entities are in namespace `atl` and `constexpr`.

`template<piecewise_linear_value Position = int64, signed_integral Slope = int64>
class convex_piecewise_linear_forest`. `Position` is the coordinate type
(`int64`, `int128`, `double`, a narrow `int16`; not unsigned), `Slope` the
slope and weight type, a signed integer (`int128` included). Aliases:
`position_type`, `slope_type`.

| member | effect | time |
|---|---|---|
| `explicit convex_piecewise_linear_forest(std::size_t function_count = 0)` | `function_count` zero functions, `id(0) .. id(function_count - 1)` | O(function_count) |
| `void reserve(std::size_t elements)` | room for `elements` breakpoint records in the shared pool | O(elements) |
| `std::size_t function_count() const noexcept` | the number of functions | O(1) |
| `function_id id(std::size_t index) const` | the id of function `index` | O(1) |
| `function_id add_function()` | a new zero function, `id(function_count() - 1)` | O(1) amortized |
| `function_reference operator[](function_id f)` | the function `f`, by reference | O(1) |
| `const_function_reference operator[](function_id f) const` | the function `f` of a const forest, read-only | O(1) |
| `void merge(function_id into, function_id from)` | `into += from`, `from := 0`; a no-op when equal | O(1) |

`function_id` is a strong id nested in the forest: `std::size_t index() const
noexcept` gives the number back, ids compare and order as their numbers, and
a default-constructed id names function 0.

`function_reference` (`forest[f]`) is a forest pointer and an index: cheap to
copy, valid across `add_function` while the forest lives and does not move.
Its members are the vocabulary of
[`convex_piecewise_linear`](convex-piecewise-linear.md) with `Position` and
`Slope` in place of `Value` and `slope_type`, all `const`; `k` is the number
of breakpoints removed or moved, `n` the pool's size, all bounds amortized:

| member | effect | time |
|---|---|---|
| `function_id id() const noexcept` | the function's id | O(1) |
| `void add_rising_hinge(Position const &a, Slope weight = 1)` | `f += weight max(0, x - a)` | O(1) |
| `void add_falling_hinge(Position const &a, Slope weight = 1)` | `f += weight max(0, a - x)` | O(1) |
| `void add_absolute(Position const &a, Slope weight = 1)` | `f += weight \|x - a\|`, one breakpoint of weight `2 weight` | O(1) |
| `void translate(Position const &d)` | `f(x) := f(x - d)` | O(1) |
| `slope_clip_interval<Position> clip_slopes(Slope min_slope, Slope max_slope)` | `f := min_y f(y) + k(x - y)`, `k(t) = max_slope t` for `t >= 0` and `min_slope t` for `t <= 0`; returns where the slopes were unchanged, whose `clamp(x)` is an optimal `y` | O((k + 1) log n) |
| `slope_clip_interval<Position> take_prefix_minimum()` | `clip_slopes(leftmost_slope(), 0)`: `{none, upper}`, `upper` the largest minimizer | O((k + 1) log n) |
| `slope_clip_interval<Position> take_suffix_minimum()` | `clip_slopes(0, rightmost_slope())`: `{lower, none}`, `lower` the smallest minimizer | O((k + 1) log n) |
| `std::optional<Position> insert_segment(Slope slope, Position const &length)` | a segment of slope `slope` and length `length` inserted where `f`'s slope rises through `slope`; returns that point `p` (none when `slope <= leftmost_slope()`); an optimal `t` for `x` is `x - clamp(p, x - length, x)` | O((k + 1) log n), `k` on the lighter side |
| `std::optional<Position> take_window_minimum(Position const &lo, Position const &hi)` | `f(x) := min over lo <= d <= hi of f(x - d)`: `insert_segment(0, hi - lo)`, then `translate(lo)` | as `insert_segment` |
| `Slope leftmost_slope() const`, `Slope rightmost_slope() const` | the slopes at minus and plus infinity | O(1) |
| `minimizer_interval<Position> minimizers() const` | the argmin interval (an absent end: unbounded); walks to the minimum from the lighter end and puts the walked breakpoints back | O((k + 1) log n), `k <= min(-leftmost, rightmost)` |
| `std::vector<weighted_breakpoint<Position, Slope>> take_breakpoints() const` | the breakpoints by increasing position, equal positions combined; `f` reset to 0 | O((k + 1) log n) |

`const_function_reference` (`forest[f]` on a const forest) has the O(1)
queries, with the same results: `id()`, `leftmost_slope()` and
`rightmost_slope()`. `minimizers()` needs a non-const forest, because its
walk pops breakpoints of the shared pool and pushes them back. Concurrent
const queries are safe.

`slope_clip_interval`, `minimizer_interval` and `weighted_breakpoint` are
the vocabulary types of `convex_piecewise_linear`. The concept
`breakpoint_sink<Sink, Position, Slope>` (`std::invocable<Sink &, Position
const &, Slope const &>`) constrains the callbacks that receive eaten or
visited breakpoints inside the pool; the public vocabulary takes none.

Ties. Where `f` has a piece of slope exactly `slope`, `insert_segment` (and
so `take_window_minimum`) returns its right end when it moves the right side
(`rightmost_slope() - slope <= slope - leftmost_slope()`) and its left end
otherwise; both are optimal. A clip's range ends are the breakpoints where
the slope first reaches `min_slope` from the left and `max_slope` from the
right.

## Complexity

A breakpoint is pushed once per heap when added and again each time
`insert_segment` moves it; clips pop it at most once per heap, and
`minimizers` and `take_breakpoints` walk by popping and pushing back. Pops
are O(log n) amortized (pairing heaps), everything else is O(1). Memory is one
weight record and two heap nodes per push, so a run that moves breakpoints
grows by the moves. Nothing recurses.

**Values are not tracked, only the shape.** The natural value to carry (the
intercept of the left asymptote) is a slope of up to 2e11 times a position
of up to 2e11 on trees of 2e5 vertices with weights and costs up to 1e6,
past int64 although every answer fits. The optimum's cost comes from its
reconstruction instead: the per-edge clip ranges give the optimal choices,
and summing the choices' costs is exact.

## Preconditions

Asserted (under `ATL_ASSERT`):

- `clip_slopes(min_slope, max_slope)`: `min_slope <= max_slope`,
  `leftmost_slope() <= max_slope` and `min_slope <= rightmost_slope()`
  (otherwise the infimum is minus infinity). So `take_prefix_minimum` needs
  `leftmost_slope() <= 0` and `take_suffix_minimum` `rightmost_slope() >= 0`.
- `insert_segment(slope, length)`: `length >= 0`, `slope <=
  rightmost_slope()`; `slope <= leftmost_slope()` is allowed (the whole graph
  moves; returns none). `take_window_minimum(lo, hi)`: `lo <= hi` and
  `rightmost_slope() >= 0`.
- `minimizers()`: `leftmost_slope() <= 0 <= rightmost_slope()`.
- Hinge weights `>= 0` (0 is a no-op), `add_absolute`'s at most
  `max(Slope) / 2`.
- `forest.id(i)`, `forest[f]` and `merge` check that the index is in range;
  fewer than `2^32 - 1` functions and breakpoint records over the forest's
  life: every hinge is a record, and so is every piece `insert_segment`
  moves.

Documented only: every position and translation fits `Position`, the sum of
the absolute slopes of all functions fits `Slope`, and an id passed to a
forest names one of its functions (an id of another forest passes the check
when its index is in range).

## Worked examples (experimental)

The two headers live in `experimental/` (namespace `atl::experimental`):
they show the forest and its dual at work, with their contenders and tests,
and are not part of the library.

### Weighted leaf-path balancing

A rooted tree; edge `v` (from `parent[v]` to `v`) has cost `C_v` and weight
`D_v`. Change the costs to any integers `X_v`, paying `D_v |X_v - C_v|`, so
that every root-to-leaf path has the same total; minimize, and report the
`X_v`.

Let `f_v(x)` be the least cost of making every path from `v` down to a leaf
total `x`. A leaf is the indicator of `x = 0`. An inner vertex is the sum
of its children's functions after their edges, and the edge above a child
`v` turns `f_v` into `h_v(x) = min_y f_v(y) + D_v |x - y - C_v|`, the
infimal convolution with `D_v |t - C_v|`: the slopes of `f_v` are clipped
to `[-D_v, D_v]` and the graph moves right by `C_v`. The clip range
`[lo_v, hi_v]` is where `f_v`'s slope lies within `D_v` of 0, and the best
`y` for a given `x` is `clamp(x - C_v, lo_v, hi_v)`: inside, `f_v` is too
flat to be worth moving off `x - C_v`; outside, paying `D_v` per unit on the
edge is cheaper. That is the loop of the quick example;
`atl::experimental::balance_leaf_paths(parent, cost, weight)` adds the
breadth-first order, the checks and the edge cases, and returns
`{total_cost, path_sum, new_cost}`.

**A lower bound on the new costs** (`balance_leaf_paths(parent, cost,
weight, min_cost)`, for example nonnegative lengths with unit weights)
changes answers. A root with one edge of length 1 to a vertex with ten
leaves of length 5, and twenty leaves of length 1 on the root: shortening
the first edge to -4 costs 5; with lengths `>= 0` the best is 41. Under
`X_v >= m` the edge kernel is `D|t - C|` over `t >= m` only (`C >= m`):
clip to slopes `<= D` (the left side stays), insert a segment of slope `-D`
and length `C - m`, translate by `m`; the best predecessor gains a cap,
`y = min(x - m, clamp(x - C, lo, hi))` with `lo` the point
`insert_segment` returned. A leaf's domain wall (its own edge `>= m`) is a
left slope no clip may cross: `2 (sum of weights) + 1` in general, since
walls at different depths sit at different positions and sibling slopes add
on top. At `m = 0` every wall sits at 0 with no breakpoint left of it, so
`max weight - D_leaf` suffices (nothing at all for unit weights), and
`x_v >= 0` settles the one possible tie.

`insert_segment` moves the lighter side by weight, at most `2 D_v`, and every
moved breakpoint becomes a new record: O(log n) per edge with unit weights,
but with weights up to 1e6 a path can move of order `n^2` breakpoints in
total, in time and in memory. The shape: a path with unit costs, `m = 0`,
`D_1 = 4n` and `D_v = n - v` below. The light side (by weight) splinters into
pieces, one more per edge, while the heavy side is the leaf's wall in a
record or two, so the weight rule moves 2, 3, ... pieces per edge, about
`5e9` at `n = 1e5`, past the pool's `2^32 - 1` records (asserted).

The experimental `balance_leaf_paths_by_fewer_entries` (in
`leaf_path_balance_variants.hpp`) inserts each segment by
`insert_segment_by_fewer_entries` (in
`convex_piecewise_linear_forest_variants.hpp`): it walks both ends of the
function in turn, one live entry each, and moves the side that reaches the
crossing first. Bound: O((min(k_left, k_right) + 1) log n) amortized per
segment, `k` the live entries between each end and the crossing, and as many
new records, so the run is O((n + M') log n) for `M'` the sum of those
minima: O(n log n) on the path above, never more than the weight rule's `M`
plus O(n). It returns the same total cost and path total (the new costs can
differ where a piece of slope exactly `-D_v` makes two reconstructions
optimal). Not measured yet; the library keeps the weight rule until it is.

Ties: the root takes the smallest minimizer of its function (unbounded
below only with zero weights; then its leftmost breakpoint, or 0), and each
edge keeps its old cost whenever that is optimal given its parent's total.

### The dual: minimum-cost tree transport

Moving one unit across edge `v` (between `parent[v]` and `v`) costs `c_v`
in either direction. Vertex `v` starts with `supply[v]` units and must end
with at least `demand[v]` (`sum supply >= sum demand`); minimize the total
moving cost. `atl::experimental::tree_transport_cost(parent, cost, supply,
demand)` returns it.

Let `f_v(k)` be the least cost inside `v`'s subtree when `k` units leave it
upward (`k < 0`: `-k` enter). It is convex. The DP is the mirror image of
leaf-path balancing with free costs:

| DP operation | leaf-path balancing (`f_v(x)`, x = path total) | tree transport (`f_v(k)`, k = flow out) |
|---|---|---|
| children | **sum**: every path has the same total `x` | **infimal convolution**: the flows add |
| the edge above | **infimal convolution** with `D \|t - C\|` | **sum** with `c \|k\|` |
| breakpoints | positions `x` (a few, weighted) | slopes (marginal costs), with lengths |

An infimal convolution of convex functions merges their pieces sorted by
slope (the epigraphs add as sets), so the representation that makes it O(1)
is the dual one: `f_v` as a multiset of (slope, length) SEGMENTS. Kept split
at `k = 0`, in a min-heap of the right side's segments keyed by slope and a
min-heap of the left side's keyed by minus the slope, an edge adds `c` to
every key on both sides: one lazy `add_to_all` per heap, on
[`meldable_heap_forest`](meldable-heap-forest.md). A child's two heaps are
melded into its parent's, O(1) each.

What the union does not give for free is the split at the parent's `k = 0`:
the merged left side may hold a slope above some right-side slope (the
children were each split at their own 0). Re-centering swaps them: while the
least right key plus the least left key is negative (a left slope `a` beats
a right slope `b`), move `m = min` of their lengths across, a segment
`(a, m)` into the right heap and `(b, m)` into the left one. Each swap lowers
`f_v(0)` by `m (a - b)`, which is how the VALUE is tracked: `f_v(0)` is the
sum of the children's values at 0 (`c |0| = 0` on every edge) minus the swap
gains. The swapped segments never swap again in the same re-centering (each
key is at least the opposite side's current top).

A vertex itself contributes `0 for k <= supply - demand`: with a surplus, a
right segment of slope 0 and length `supply - demand`; with a deficit, the
domain ends left of 0, so the hard wall is priced instead: a left segment
of slope `P = sum of costs + 1` and length `demand - supply`, with `P
(demand - supply)` added to the value. `P` beats every path, so the optimum
meets every demand, the domain stays the whole line, and no segment ever
needs dropping.

The example holds the segment being eaten on each side out of its heap
instead of popping it and pushing the rest back after every partial swap,
and pushes the reverse segments after the loop (so the loop's pops do not
pair them), not at all at the root. O(n + s log n) for `s` swaps; no bound
on `s` better than the number of augmentations of the matching greedy is
claimed.

**Why it is not a member of the forest.** The forest stores breakpoints by
position and adds functions by melding; the dual stores segments by slope
and needs the re-centering after every meld, plus value tracking at 0,
which the forest deliberately does not do. A dual class (segments, an
infimal convolution `merge`, an `add_absolute_at_zero(c)`, a
`value_at_zero()`) would be the 60 lines of the example's loop around
`meldable_heap_forest`; with one application using it, it stays an example.

### Preconditions of the examples

All asserted:

- `balance_leaf_paths`: spans of one size `n >= 1` forming a tree rooted at
  0 (a cycle or an unreachable vertex is caught), weights `>= 0`, the sum of
  weights at most `max(int64) / 2`; with `min_cost`, costs `>= min_cost`
  and `4 n (sum of weights)` within int64 (the domain walls); fewer than
  `2^32 - 1` breakpoint records over the run, `2n + M` for `M` the moved
  pieces (checked by the pool; the bounded path above exceeds it at `n =
  1e5`). Path totals, translated breakpoints and the total cost must fit
  int64 (documented; the free-cost worst case at n = 2e5 with weights and
  costs up to 1e6 is about 2e17); the reconstruction forms `x_parent -
  cost[v]`, the clamp and `|new_cost - cost|` in `int128` (an edge of cost
  `INT64_MIN` and weight 0 is exact) and asserts that the three results fit.
  The ordered-map contender stores positions and its offset in `int128`, as
  a meld can store a breakpoint two `int64` ranges away from its offset.
- `tree_transport_cost`: spans of one size `n >= 1` forming a tree rooted
  at 0, costs, supplies and demands `>= 0` with sums within int64, `sum
  supply >= sum demand`, and the answer within int64 (checked on the
  result). Keys, reverse records, gains and the running total are `int128`
  (a reverse key is minus an opposite key and gains a cost per edge above,
  so a sum of two keys can pass int64 when the answer is 0); the price of
  unmet demand, `sum cost + 1`, fits int64. A heap forest whose `top`
  returns a proxy is read through the proxy's conversion to
  `transport_segment const &`.

Neither recurses: both order the vertices by a breadth-first or Kahn scan,
so a path of 10^6 vertices or a star of 10^6 leaves runs in one stack frame
(the tests run both).

## Measured performance

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2, RelWithDebInfo, assertions
off. Milliseconds, minimum of three run means, plain `-O2` /
`-O2 -march=native`; both also use `-g -DNDEBUG -std=c++23`.

**The breakpoint pool** ([benchmark](../benchmarks/dp/convex_piecewise_linear_forest.cpp),
2026-11-01). The library pool empties a discarded heap (`clear`, and the
right heap of `take_all`, whose left heap pops to emit the sorted
breakpoints) with `meldable_heap_forest::release`, one walk over the links;
the contenders empty it by repeated pops, each pairing the root's children,
on the library's heaps (by repeated pops) and on the experimental
`meldable_heap_forest_by_pairing` at its defaults (by tunable pairing pops).
Unit breakpoints at positions uniform in [0, 1e9); the tree shape melds 2e5
sets of 3 bottom-up along random parents and discards the root.

| shape | library (release) | by repeated pops | by tunable pairing pops |
|---|---|---|---|
| clear: 1 set x 1e6 | 74.1 / 75.3 | 1357 / 1348 | 1332 / 1340 |
| clear: 1000 sets x 1000 | 222 / 222 | 393 / 382 | 364 / 370 |
| clear: 1e5 sets x 10 | 88.8 / 91.9 | 132 / 127 | 122 / 124 |
| clear + refill: 1000 sets x 1000, 3 rounds | 632 / 631 | 1724 / 1693 | 1666 / 1683 |
| clear: melded random tree n=2e5 (root) | 104 / 79.0 | 706 / 706 | 698 / 690 |
| take_all: 1 set x 1e6 | 812 / 807 | 1463 / 1449 | 1447 / 1435 |
| take_all: 1000 sets x 1000 | 315 / 315 | 409 / 406 | 392 / 394 |
| take_all: 1e5 sets x 10 | 143 / 142 | 162 / 162 | 152 / 150 |
| take_all: melded random tree n=2e5 (root) | 493 / 469 | 803 / 791 | 797 / 790 |

Release wins every shape: 17.8x to 18.3x on clearing one large set, 6.7x to
8.9x on the melded tree, 1.06x to 1.80x on `take_all`, where the sorted pops
of the left heap remain.

**Leaf-path balancing** ([benchmark](../benchmarks/dp/leaf_path_balance.cpp),
2026-10-24). The example against by ordered map (each function a `std::map`
from position to weight with a lazy offset, children merged
smaller-into-larger, O(n log^2 n) without a bound; with one, O((n + M)
log^2 (n + M)) for the `M` moved pieces). The benchmark also races by fewer
entries and a bounded rising-weight path of `n = 2e3`: not measured yet. n = 2e5, costs uniform in [1, 1e6];
free-cost weighted: weights uniform in [1, 1e6], any new costs; nonnegative
unit-weight: unit weights, new costs `>= 0`. Caterpillar = a spine of n/2
with one leaf per spine vertex; random = parent uniform among the earlier
vertices.

| shape (n = 2e5) | example (heap forest) | by ordered map |
|---|---|---|
| free-cost weighted path | 5.9 / 5.9 | 7.4 / 7.9 |
| free-cost weighted star | 44.4 / 43.7 | 109 / 106 |
| free-cost weighted caterpillar | 11.5 / 12.1 | 16.0 / 16.4 |
| free-cost weighted random | 53.6 / 54.6 | 62.3 / 63.0 |
| nonnegative unit-weight path | 7.1 / 6.9 | 7.3 / 7.6 |
| nonnegative unit-weight star | 44.8 / 44.9 | 105 / 106 |
| nonnegative unit-weight caterpillar | 19.9 / 21.1 | 48.3 / 50.6 |
| nonnegative unit-weight random | 58.9 / 57.9 | 127 / 132 |

The heap forest wins every row: 1.15x to 2.46x free-cost weighted, 1.03x to
2.43x nonnegative unit-weight (the path is the near tie).

**Tree transport** ([benchmark](../benchmarks/dp/tree_transport.cpp),
2026-10-24). The example on the library's pairing heaps against the same
loop on the other engines of the meldable heap race (by skew, by rank
leftist, by binary: `std::vector` heaps melded smaller-into-larger), and
against the loop without the example's two savings (by pop and push) and
that loop with the lengths in a side array (by length records). n = 2.5e5,
costs uniform in [1, 1e6]; mixed: every vertex has a supply and a demand
uniform in [0, 7]; split: supplies on the first half of the vertices,
demands on the second.

| instance | example (pairing) | by pop and push | by length records | by skew | by rank leftist | by binary |
|---|---|---|---|---|---|---|
| mixed path | 41.6 / 44.7 | 47.2 / 43.6 | 72.3 / 72.7 | 37.1 / 39.7 | 59.4 / 59.5 | 108 / 108 |
| mixed star | 74.3 / 77.6 | 107 / 103 | 129 / 130 | 77.5 / 80.0 | 99.8 / 135 | 51.6 / 52.0 |
| mixed caterpillar | 37.7 / 39.5 | 44.2 / 41.7 | 73.1 / 73.3 | 37.9 / 40.1 | 58.7 / 59.1 | 97.0 / 96.9 |
| mixed random | 69.9 / 71.6 | 71.6 / 69.8 | 85.9 / 90.0 | 62.0 / 66.3 | 77.6 / 88.1 | 153 / 153 |
| split path | 16.9 / 18.2 | 18.7 / 17.5 | 17.6 / 17.8 | 12.7 / 14.0 | 17.8 / 17.8 | 48.5 / 48.9 |
| split star | 71.6 / 75.2 | 104 / 102 | 134 / 130 | 75.2 / 75.4 | 99.2 / 131 | 50.3 / 49.6 |
| split caterpillar | 18.0 / 18.9 | 20.1 / 18.9 | 19.5 / 20.0 | 17.9 / 19.0 | 22.2 / 22.6 | 52.2 / 52.1 |
| split random | 70.4 / 69.9 | 71.4 / 71.7 | 87.2 / 89.6 | 62.9 / 67.5 | 76.2 / 84.0 | 153 / 153 |

- **The savings** buy 1.44x / 1.33x on the mixed star and 1.45x / 1.36x on
  the split star (all the work is the root's re-centering), 1.01x to 1.17x
  under plain `-O2` on the other shapes and 0.96x to 1.06x with
  `-march=native`. The side array of lengths is slower except on the native
  split path (1.04x to 1.94x): the indirection costs more than the pops it
  saves.
- **Engines.** Skew heaps beat pairing on the paths (1.12x to 1.33x) and the
  random trees (1.04x to 1.13x), stay within 2% on the caterpillars and lose
  the stars by up to 1.05x. Binary heaps win the stars 1.42x to 1.52x (every
  child brings one segment, and the root's swaps pop a contiguous array) and
  lose every other shape by 2.14x to 2.90x. Rank-leftist heaps lose
  everywhere except the native split path (17.8 against 18.2). The example
  stays on the library's pairing heaps, which won the meldable heap race;
  switching it to skew heaps remains an open choice.

## Pitfalls and notes

- **What needs the minimum is not O(1) here.** The forest keeps no split at
  the minimum, so `minimizers()` walks from the end carrying less weight to
  where the slope crosses 0, popping and pushing the walked entries back. It
  is `const` on the reference, and on a const forest's read-only view, but
  not on the pool (it drops stale entries; the function is unchanged). `take_window_minimum` moves the lighter side of
  slope 0 and pushes those breakpoints again, NOT the O(1) of the single
  class: a window DP belongs on `convex_piecewise_linear`.
- **Eat the final minimizer from the lighter end.** `clip_slopes(0, 0)`
  returns the minimizers and empties both ends of the function. When the
  function is dropped afterwards (a tree DP's root), eat only the end with
  less weight: `take_suffix_minimum()` if `-leftmost_slope() <=
  rightmost_slope()`, else `take_prefix_minimum()` and read `minimizers()`
  (then O(1)). A bounded leaf-path root is heavy on the left, where its
  min-heap holds every stale entry of the run; `balance_leaf_paths` does
  this, and its ordered-map contender, whose ends are cheap, does not need to.
- **Reserve what the run pushes, not more.** `balance_leaf_paths` reserves
  `n` records: one per leaf, enough for the free-cost run (a bounded run
  grows past it on some shapes, about 1.5n on the unit-weight caterpillar).
  Larger hints measured slower on the benchmark's repeated calls, which
  release and fault in the larger blocks again.
- **Bounded weighted runs can be quadratic.** With a lower bound on the new
  costs and large weights, every edge may move many breakpoints (see the
  worked example); with unit weights it moves at most two units of weight.
  The experimental `balance_leaf_paths_by_fewer_entries` avoids the known
  quadratic path (not measured yet).
- **`take_all` combines equal positions only**, also with an output tail it
  did not write: a tail at another position is left alone, whatever its
  order relative to the set.
- **Choosing.** One function of a sequence DP, with its minimum value:
  `convex_piecewise_linear` (O(1) minimizers and window minimum). Many
  functions that are summed, or clips of the far ends: this forest. A tree
  DP whose children combine by infimal convolution: the dual segment form of
  `tree_transport_cost` on [`meldable_heap_forest`](meldable-heap-forest.md).

## How it works

`f` is stored as `(leftmost slope, rightmost slope, breakpoint multiset)`,
the slope rising by `weight` at `position` (so the weights sum to
`rightmost - leftmost`). Each breakpoint is one weight record entered in a min-heap and
a max-heap of its function by position (two `meldable_heap_forest` pools with
a lazy translation, `detail::breakpoint_pool`). Clipping removes the
FARTHEST breakpoints (the leftmost raise the slope at minus infinity), which
a max-heap of the left half around the minimum cannot reach; with both heaps
over the whole multiset, eating from one end decrements the shared record,
and an entry whose record the other end already emptied is dropped when it
surfaces. No split at the minimum is kept, so a sum is two O(1) melds with no
rebalancing.

Sum: the slopes add and the multisets unite. Clip to `[lo, hi]`: if the
slope at minus infinity is below `lo`, eat `lo - leftmost` of weight from the
left end (the leftmost breakpoint first; the last one touched may be split);
symmetrically on the right. The last breakpoint touched on each side is the
reported range end: `f`'s slope is below `lo` left of it and at least `lo`
right of it, which is exactly the optimality condition of the infimal
convolution's inner minimum. Insert a segment at slope `s`: the breakpoints
carrying the slope from `s` up to the rightmost slope move right by `len`
(eaten from the right and pushed again, shifted), or, if that side is
heavier, the ones below `s` are taken out, everything is translated, and
they are put back where they were.

`clear` and the discarded heap of `take_all` use `meldable_heap_forest`'s
`release`, which returns the nodes in one walk instead of pairing them pop
by pop (Measured). The experimental
`convex_piecewise_linear_by_breakpoint_pool` (a contender of
`convex_piecewise_linear`) keeps two sets of the same pool, split at the
minimum; the pool's own contenders are in
`atl/experimental/dp/convex_piecewise_linear_forest_variants.hpp`, those of
the worked examples in `atl/experimental/dp/leaf_path_balance_variants.hpp`
and `atl/experimental/dp/tree_transport_variants.hpp`.

In a free-cost leaf-path run each leaf creates one breakpoint (its
`D|x - C|`) and nothing else does, so at most `n` records are pushed and each
entry is popped at most once: O(n log n) amortized over the pairing heaps.

References: the slope trick for convex piecewise-linear DP states is folklore
of competitive programming; pairing heaps are from M. L. Fredman, R.
Sedgewick, D. D. Sleator and R. E. Tarjan, "The pairing heap: a new form of
self-adjusting heap" (1986).

## See also

- [`convex_piecewise_linear`](convex-piecewise-linear.md): the single-function slope trick with the same vocabulary, O(1) minimizers and window minimum, and the minimum value tracked.
- [`meldable_heap_forest`](meldable-heap-forest.md): the pairing-heap pools underneath, and the engines of the transport example.
- [DP optimization](dp-optimization.md): the other DP-optimization components.
