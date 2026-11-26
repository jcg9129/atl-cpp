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
optimal). The library uses the weight rule.

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

**The breakpoint pool** ([benchmark](../benchmarks/dp/convex_piecewise_linear_forest.cpp),
2026-11-22). The library pool empties a discarded heap (`clear`, and the
right heap of `take_all`, whose left heap pops to emit the sorted
breakpoints) with `meldable_heap_forest::release`, one walk over the links;
the contenders empty it by repeated pops, each pairing the root's children,
on the library's heaps (by repeated pops) and on the experimental
`meldable_heap_forest_by_pairing` at its defaults (by tunable pairing pops).
Unit breakpoints at positions uniform in [0, 1e9); the tree shape melds 2e5
sets of 3 bottom-up along random parents and discards the root.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | library (release) | by repeated pops | by tunable pairing pops |
|---|---|---|---|
| clear: 1 set x 1e6 | 83.3 / 71.2 | 1820 / 1830 | 1760 / 1840 |
| clear: 1000 sets x 1000 | 97.3 / 89.2 | 245 / 232 | 219 / 214 |
| clear: 1e5 sets x 10 | 103 / 95.5 | 132 / 124 | 108 / 105 |
| clear + refill: 1000 sets x 1000, 3 rounds | 170 / 157 | 1330 / 1400 | 1280 / 1390 |
| clear: melded random tree n=2e5 (root) | 79.4 / 94.4 | 854 / 861 | 817 / 854 |
| take_all: 1 set x 1e6 | 956 / 1030 | 1790 / 1960 | 1750 / 1980 |
| take_all: 1000 sets x 1000 | 168 / 161 | 241 / 234 | 219 / 216 |
| take_all: 1e5 sets x 10 | 123 / 116 | 138 / 128 | 117 / 115 |
| take_all: melded random tree n=2e5 (root) | 545 / 571 | 918 / 1010 | 921 / 1000 |

Release wins all five clearing shapes. Against repeated pops, clearing one
large set is 1820/83.3 = 21.85 times faster plain and 1830/71.2 = 25.70
native; clearing the melded tree is 10.76/9.12 times faster. The library
loses `take_all` on 1e5 sets of 10 to tunable pairing pops (123/116 versus
117/115 ms); these rows vary between runs, so the minima do not establish a
stable ranking. The other `take_all` shapes favor release in both modes.

**Leaf-path balancing** ([benchmark](../benchmarks/dp/leaf_path_balance.cpp),
2026-11-25). The example against by ordered map (each function a `std::map`
from position to weight with a lazy offset, children merged
smaller-into-larger, O(n log^2 n) without a bound; with one, O((n + M)
log^2 (n + M)) for the `M` moved pieces). The benchmark also measures by
fewer entries and a bounded rising-weight
path of `n = 2e3`. n = 2e5, costs uniform in [1, 1e6];
free-cost weighted: weights uniform in [1, 1e6], any new costs; nonnegative
unit-weight: unit weights, new costs `>= 0`. Caterpillar = a spine of n/2
with one leaf per spine vertex; random = parent uniform among the earlier
vertices.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape (n = 2e5) | example (heap forest) | by ordered map |
|---|---|---|
| free-cost weighted path | 4.6 / 3.1 | 4.4 / 3.8 |
| free-cost weighted star | 32.4 / 28.2 | 69.8 / 66.2 |
| free-cost weighted caterpillar | 8.6 / 8.2 | 10.3 / 8.4 |
| free-cost weighted random | 28.3 / 24.1 | 44 / 33.8 |
| nonnegative unit-weight path | 5.3 / 4.5 | 4.6 / 4.1 |
| nonnegative unit-weight star | 32.2 / 28.3 | 71 / 67 |
| nonnegative unit-weight caterpillar | 17.8 / 15.6 | 30.7 / 27.8 |
| nonnegative unit-weight random | 34.8 / 27.9 | 96.1 / 83.1 |

The heap forest loses the nonnegative unit-weight path to ordered maps
(5.3/4.5 versus 4.6/4.1 ms). The weighted path favors maps plain (4.4 versus
4.6 ms) and the forest native (3.1 versus 3.8 ms). The forest wins the other
six shapes in both modes; no engine wins every row.

**Tree transport** ([benchmark](../benchmarks/dp/tree_transport.cpp),
2026-11-25). The example on the library's pairing heaps against the same
loop on the other engines of the meldable heap race (by skew, by rank
leftist, by binary: `std::vector` heaps melded smaller-into-larger), and
against the loop without the example's two savings (by pop and push) and
that loop with the lengths in a side array (by length records). n = 2.5e5,
costs uniform in [1, 1e6]; mixed: every vertex has a supply and a demand
uniform in [0, 7]; split: supplies on the first half of the vertices,
demands on the second.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| instance | example (pairing) | by pop and push | by length records | by skew | by rank leftist | by binary |
|---|---|---|---|---|---|---|
| mixed path | 60.3 / 52.6 | 64 / 54.1 | 53.2 / 52.2 | 59.3 / 45.7 | 65.5 / 61 | 101 / 97.8 |
| mixed star | 59.3 / 55.8 | 82.3 / 76.2 | 90.2 / 92.8 | 67.8 / 69.8 | 107 / 119 | 50.2 / 44.6 |
| mixed caterpillar | 51.8 / 45.7 | 54.9 / 48.6 | 48.1 / 47.3 | 53.6 / 45.4 | 59.4 / 56.9 | 88.4 / 83.8 |
| mixed random | 46.4 / 55.7 | 45.9 / 52.8 | 48.8 / 61 | 47.7 / 63.8 | 57.4 / 88.4 | 153 / 186 |
| split path | 19.7 / 15.6 | 19.6 / 15.8 | 14 / 13.5 | 18.1 / 11.2 | 19 / 16.8 | 42.6 / 44 |
| split star | 55.8 / 76.4 | 82.7 / 78.5 | 87.6 / 83.4 | 64 / 59.2 | 104 / 98 | 48.4 / 42 |
| split caterpillar | 18.7 / 15.6 | 19.5 / 16.3 | 14.4 / 14.2 | 19.8 / 15.4 | 20.6 / 18.9 | 45.6 / 41.5 |
| split random | 45.2 / 43.2 | 45.2 / 52.8 | 48.7 / 61.5 | 47.9 / 48.5 | 58.2 / 89.3 | 152 / 161 |

The direct example beats pop/push on the mixed star (59.3/55.8 versus
82.3/76.2 ms), but loses on mixed random trees (46.4/55.7 versus
45.9/52.8 ms). Length records beat it on both paths and the split
caterpillar; the mixed caterpillar favors length records plain
(48.1 versus 51.8 ms) and the direct example native (45.7 versus 47.3 ms).

Skew beats the library's pairing heaps on both paths, but loses both random
trees. Caterpillar and split-star rankings depend on flags. Binary heaps
beat pairing on both stars (50.2/44.6 versus 59.3/55.8 ms mixed;
48.4/42 versus 55.8/76.4 ms split), and lose every other shape. Rank-leftist
wins the split path plain (19 versus 19.7 ms) but loses it native
(16.8 versus 15.6 ms), and loses every other shape. Some transport rows vary
between runs, with spreads reaching 1.46x plain and 2.55x native; these
minimum-time comparisons do not establish stable rankings on those rows.

The weighted random leaf-path comparison varies between runs (spread up
to 1.27x); its minima alone do not establish a stable engine ranking.

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
  quadratic path.
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
