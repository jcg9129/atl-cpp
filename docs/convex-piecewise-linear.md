# `atl::convex_piecewise_linear` (slope trick)

**A convex piecewise-linear function of one variable** with integer slopes,
kept as its breakpoints in two heaps around its minimum, with the minimum
value tracked. This is the "slope trick" representation of a DP row
`dp_i(x)` that stays convex in `x`: it replaces an O(range) array with
O(log n) work per step, and the domain may be all of `int64`, `int128` or the
reals. Reach for it for sequence DPs (fit a nondecreasing sequence, a window
of predecessors, a cost per unit moved). For many functions that are summed
(tree DPs), or for a DP that clips the far ends of large sides on every step,
use [`convex_piecewise_linear_forest`](convex-piecewise-linear-forest.md),
which shares this vocabulary.

```cpp
#include <atl/dp/convex_piecewise_linear.hpp>
```

\include convex-piecewise-linear/quick.cpp

Output:

\include convex-piecewise-linear/quick.out

A cost per unit moved (`clip_slopes`) and a bounded step (`take_window_minimum`):

\include convex-piecewise-linear/moving.cpp

## API

All entities are in namespace `atl` and `constexpr`.

```cpp
template<typename Value>
concept piecewise_linear_value = std::semiregular<Value> && std::totally_ordered<Value> && !unsigned_integral<Value>
	&& requires(Value const a, Value &b) {
		{ a + a } -> std::convertible_to<Value>;
		{ a - a } -> std::convertible_to<Value>;
		b += a;
		b -= a;
	};

template<typename Value, typename Slope>
concept scalable_by = piecewise_linear_value<Value> && signed_integral<Slope>
	&& std::constructible_from<Value, Slope const &> && requires(Value const a) { { a * a } -> std::convertible_to<Value>; };

struct unit_weight {};
template<typename Slope>
concept piecewise_linear_slope = std::same_as<Slope, unit_weight> || signed_integral<Slope>;

template<piecewise_linear_value Value, piecewise_linear_slope Slope = unit_weight>
	requires scalable_by<Value, detail::slope_count_t<Slope>>
class convex_piecewise_linear;
```

`Value` is the coordinate and value type: any signed integer (`int128`
included) or floating-point type; unsigned integers are rejected, since
offsets and values go negative. `detail::slope_count_t<Slope>` is `Slope`, or
`int64` for `unit_weight`: the public `slope_type`. `Slope` picks the storage:

- `atl::unit_weight` (the default): unit breakpoints stored as bare
  positions (a breakpoint of multiplicity `m` stored `m` times); the hinges
  take no weight argument, and a weighted call does not compile.
  `slope_type` is `int64`. The fastest storage for unit hinges.
- a signed integer (`int64`, `int16`, ...): `(position, weight)` entries; the
  hinges take a weight, default 1, and `slope_type` is `Slope`.

Unit mode drops the weight argument instead of accepting only `w == 1`: a
weight that would be checked at run time is a weight the caller should not be
writing, and the compile error points to `Slope`. `is_unit` tells the two
modes apart. `value_type` (and its alias `position_type`, the name the forest
uses) is `Value`. The function starts as `f = 0`; `n` is the number of stored
breakpoints, `m` the number on one side of the minimum.

The shared vocabulary (identical names, arguments and return values on a
forest function, `forest[f]`):

| member | effect | time |
|---|---|---|
| `add_rising_hinge(a)` (unit), `add_rising_hinge(a, w = 1)` (weighted) | `f += w max(0, x - a)` | unit: O(log n); weighted: O((k + 1) log n), `k` breakpoints carried across the minimum |
| `add_falling_hinge(a)`, `add_falling_hinge(a, w = 1)` | `f += w max(0, a - x)` | as above |
| `add_absolute(a)`, `add_absolute(a, w = 1)` | `f += w \|x - a\|` | as above |
| `translate(d)` | `f(x) := f(x - d)` | O(1) |
| `clip_slopes(lo, hi) -> slope_clip_interval<Value>` | slopes clipped into `[lo, hi]`: `f := min_y f(y) + k(x - y)`, `k(t) = hi t` (`t >= 0`), `lo t` (`t <= 0`); returns where nothing changed, an end present exactly when that side was clipped | O(1) per side clipped to 0 or not at all; else O(m) per entry eaten for the first four (a leaf scan), then one O(m log m) sort |
| `take_prefix_minimum() -> slope_clip_interval<Value>` | `f(x) := min_{y <= x} f(y)`; returns `{none, largest minimizer before}` | O(1) |
| `take_suffix_minimum() -> slope_clip_interval<Value>` | `f(x) := min_{y >= x} f(y)`; returns `{smallest minimizer before, none}` | O(1) |
| `take_window_minimum(lo, hi) -> std::optional<Value>` | `f(x) := min_{lo <= d <= hi} f(x - d)`; returns the smallest minimizer before, `p` (none: minus infinity) | O(1) |
| `insert_segment(s, len) -> std::optional<Value>` | a segment of slope `s`, length `len` inserted where the slope rises through `s`: `f := min_{0 <= t <= len} f(x - t) + s t`; returns that point `p` (none when `s <= leftmost_slope()`) | O(1) for `s = 0` or `s <= leftmost_slope()`; O((k + 1) log n) for the `k` breakpoints between the minimum and `p`; O(m) for `s = rightmost_slope()` |
| `leftmost_slope()`, `rightmost_slope()` | slopes at minus / plus infinity (`<= 0`, `>= 0`) | O(1) |
| `take_breakpoints() -> std::vector<weighted_breakpoint<Value, slope_type>>` | the breakpoints by increasing position, equal positions combined (in unit mode too: the weight counts them); `f` reset to 0 | O(n log n) |
| `minimizers() const -> minimizer_interval<Value>` | the closed argmin interval, an absent end meaning unbounded | here O(1); on the forest it walks to the minimum and back |

This class only (value tracking, and storage that differs from the forest):

| member | effect | time |
|---|---|---|
| `reserve(elements)` | room for `elements` stored breakpoints on each side (the forest's `reserve` counts records in its one shared pool) | O(elements) |
| `add_constant(c)` | `f += c` | O(1) |
| `minimum() const -> Value const &` | the minimum value | O(1) |
| `operator()(x) const -> Value` | `f(x)`, by summing every hinge | O(n) |

```cpp
template<piecewise_linear_value Position>
struct slope_clip_interval {
	std::optional<Position> lower;    // none: unbounded below
	std::optional<Position> upper;    // none: unbounded above
	Position clamp(Position x) const; // x moved into [lower, upper]
};
template<piecewise_linear_value Position>
using minimizer_interval = slope_clip_interval<Position>;
template<piecewise_linear_value Position, signed_integral Slope>
struct weighted_breakpoint { Position position; Slope weight; };
```

**Reconstruction.** Every operation that takes a minimum reports where the
minimum is attained, so a top-down pass can rebuild the optimal choices (the
quick example does this):

- `clip_slopes`, `take_prefix_minimum`, `take_suffix_minimum`: an optimal
  `y` for `x` is `range.clamp(x)`.
- `take_window_minimum(lo, hi)` returning `p`: an optimal `y` is
  `clamp(p, x - hi, x - lo)` (`x - hi` when `p` is none).
- `insert_segment(s, len)` returning `p`: an optimal `t` is `x - clamp(p, x
  - len, x)` (`len` when `p` is none).

**Ties.** Where several arguments are optimal the choice is fixed and
tested: `take_window_minimum` returns the smallest minimizer, the prefix
minimum the largest, the suffix minimum the smallest. When a stretch of
slope exactly `s` lets several points qualify for `insert_segment(s, len)`,
`p` is the one nearest the minimum: the smallest minimizer for `s == 0`,
else the end of that stretch on the minimum's side.

## Preconditions

Asserted (`atl_assert`, active under `-DATL_ASSERT`): hinge weights `>= 0`
(weighted mode; 0 changes nothing); `take_window_minimum`: `lo <= hi`;
`clip_slopes`: `lo <= 0 <= hi` (the class tracks the minimum, so it keeps
`leftmost_slope() <= 0 <= rightmost_slope()`, where a forest function may
clip to `lo > 0`); `insert_segment`: `len >= 0`, `s <= rightmost_slope()`;
in unit mode, a side's breakpoint count fits `int64`; `take_breakpoints`:
every combined weight fits `slope_type` (a position at the minimum can carry
both sides' weight: `add_absolute(0, 127)` on `Slope = int8` makes 254).

Documented only: every breakpoint, offset, stored breakpoint (its position
minus its side's offset, i.e. the translations since the side was last
emptied) and the minimum fit `Value`, and so does `f(x)` for every evaluated
`x` (positions up to the ends of `int64` work, as the tests check at
`INT64_MAX - 10`); the sum of the weights on each side fits `slope_type`
(tested at 127 for `int8`). Narrow types (`int16`, `int8`) are accepted.

The intermediates need not fit: for an integral `Value` up to 64 bits the
distance `p - a` and the product `weight * |p - a|` of a hinge crossing the
minimum, the `slope * length` of a segment and the hinge terms `f(x)` sums
are formed in a wider integer (`int128` for `int64`, `int32` below 32 bits)
and only the minimum or `f(x)` is narrowed back. So `add_constant(INT64_MIN)`,
`add_rising_hinge(INT64_MIN)` and then `f(0) == 0` is exact, as is
`insert_segment(-INT64_MAX, 2)` on the constant `INT64_MAX`. An `int128` or
floating `Value` has no wider type: there they must fit too. With a floating
`Value`, offsets accumulate rounding like any running sum; an emptied side
starts again from offset 0.

Every coordinate argument is read before the function changes, so
`f.add_rising_hinge(f.minimum())` and `f.add_absolute(f.minimum())` act on
the minimum at the call, not on the value the first hinge leaves.

## Measured performance

[`benchmarks/dp/convex_piecewise_linear.cpp`](../benchmarks/dp/convex_piecewise_linear.cpp),
tuning race: the library against Floyd's bottom-up `replace_top` (to a leaf
along the larger children without comparing the new top, then back up)
on the hinge shapes, and against the whole-heap scan for the farthest right
breakpoint (`std::ranges::max` over the heap, where the library scans only
its leaves) on segments at the rightmost slope. n = 1e6 operations with a_i
uniform in `[0, 1e9)`, weights in `[1, 1000]`; the segment shape builds each
function from 20,000 random hinges (about 1e4 right entries) and then
inserts 2000 segments.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-26.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | library | by Floyd replace_top |
|---|---|---|
| prefix minimum + \|x - a\|, unit | 53.7 / 53.6 | 54.1 / 56 |
| window [-1000, 1000] + \|x - a\|, unit | 83 / 87 | 85.1 / 86.8 |
| random hinges, unit | 46.7 / 46.4 | 56.4 / 58.8 |
| prefix minimum + w\|x - a\|, weighted | 91.6 / 88.3 | 91.5 / 87.8 |
| random weighted hinges | 70.4 / 67.1 | 70.8 / 67.4 |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-26.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | library (leaf scan) | by whole-heap scan |
|---|---|---|
| segments at the rightmost slope, unit | 6.3 / 6.3 | 11.9 / 11.9 |
| segments at the rightmost slope, weighted | 6.7 / 6.6 | 12.1 / 12.1 |

The library's leaf scan beats the whole-heap scan: 6.3/6.3 versus
11.9/11.9 ms on unit entries, 1.89 times faster with both flag sets, and
6.7/6.6 versus 12.1/12.1 ms on weighted entries, 1.81/1.83 times faster.
Both are O(m); the leaf scan reads half the entries.
Floyd replacement loses the unit prefix-minimum, unit random-hinge and
weighted random-hinge shapes with both flag sets, and the plain unit window
(85.1 versus 83 ms). It narrowly beats the library on the native unit window
(86.8 versus 87 ms) and on weighted prefix minima (91.5/87.8 versus
91.6/88.3 ms).

`benchmarks/dp/convex_piecewise_linear.cpp` also measures the library's two
modes against the textbook `std::priority_queue` form, unit-only heaps and
the forest's breakpoint pool. The contenders are defined in
`atl/experimental/dp/convex_piecewise_linear_variants.hpp`.

The native unit random-hinge library row varies between runs (1.27x
spread); its lead over Floyd (46.4 versus 58.8 ms) is larger than that.

## Pitfalls and notes

- **Concave functions need no separate class.** Maximizing a concave
  piecewise-linear `g` is minimizing `f = -g`, so keep `f`:
  `g += w min(0, x - a)` (rising up to `a`, then flat) is
  `f.add_falling_hinge(a, w)`; `g += w min(0, a - x)` (flat, then falling)
  is `f.add_rising_hinge(a, w)`; `g -= w |x - a|` is `f.add_absolute(a, w)`;
  `g += c` is `f.add_constant(-c)`. The maximum of `g` is `-f.minimum()`,
  its maximizers are `f.minimizers()`, `g(x)` is `-f(x)`, and a prefix /
  suffix / window maximum of `g` is the prefix / suffix / window minimum of
  `f` (the same calls, the same reconstruction ranges).
- `f(x)` sums every hinge, so it is O(n): use it for a final answer at a
  fixed point, not inside a loop.
- A `clip_slopes` that is neither a prefix / suffix minimum nor a no-op eats
  the far ends of a heap, which a binary heap does not reach: up to four
  entries are found by scanning the heap's leaves, past that the side is
  sorted. Cheap when the sides stay small (a clip after every hinge with a
  small speed); a DP whose sides grow to thousands of breakpoints and clips
  every step belongs on a one-function
  [`convex_piecewise_linear_forest`](convex-piecewise-linear-forest.md)
  (amortized O(log n) per breakpoint removed, but no tracked value).
- Merging two functions is left out: a sum of two split-at-the-minimum
  heaps must move every misplaced breakpoint across the new minimum. Sum
  functions on the forest.
- `take_breakpoints` empties the function; read `minimum()` and the slopes
  first.

## How it works

`f` is stored as its minimum value and its breakpoints `(a, w)`, the slope
rising by `w` at `a`. The breakpoints left of the minimum live in a max-heap
and those right of it in a min-heap, each heap with a lazy offset and its
total weight (minus the slope at minus infinity, the slope at plus
infinity). The slope is 0 between the two tops, so `minimizers()` reads the
tops and the minimum value is a number kept on the side.

- **A hinge** `w max(0, x - a)` whose `a` lies left of the left top carries
  weight across the minimum: the left top `(p, t)` gives `min(t, w)` units
  to the right heap, each unit raising the minimum by `p - a`, until the
  hinge's weight is spent or the left top is no longer right of `a`; the
  carried weight lands at `a` on the left, the rest of the hinge at `a` on
  the right. Every carried piece becomes the top of its new side, so a piece
  at the same position as that top is merged into it: without that, a
  position crossing back and forth splits into many small entries that every
  later crossing walks (weighted random hinges ran superlinearly).
- **Unit mode** stores bare positions (a weight-`m` breakpoint is `m`
  copies, so the side's weight is its heap's size): a hinge carries at most
  one copy, the top replaced in place by one sift-down (half a pop and a
  push) and the old top pushed on the other side, with no weight
  bookkeeping. `insert_segment` and `clip_slopes` eat copies one by one.
- **`take_window_minimum(lo, hi)`** moves the decreasing part right by `lo`
  and the increasing part by `hi` (two offsets), which widens the flat
  bottom by `hi - lo`; `translate(d)` is `lo = hi = d`. **Prefix and suffix
  minimum** drop one heap.
- **`clip_slopes(lo, hi)`** eats `left weight + lo` from the far end of the
  left heap (the smallest positions) and `right weight - hi` from the far
  end of the right heap, splitting the last breakpoint touched, whose
  position it reports; the minimum is unchanged. A heap's farthest entry is
  one of its leaves (indices from `size / 2` on), so the first four are found
  by a leaf scan and removed by moving the last entry into the hole (still a
  leaf) and sifting it up; past four the side is sorted nearest first (a
  sorted array is a valid heap) and eaten from the back.
- **`insert_segment(s, len)`** with `s > 0` takes the `s` units of the right
  heap nearest the minimum out, moves the right heap by `len`, and puts them
  back; with `s < 0` it takes the `-s` nearest units of the left heap out and
  puts them back `len` further right, moves the right heap by `len`, and
  lowers the minimum by `-s len`. At `s = rightmost_slope()` nothing is
  steeper, so the graph is unchanged and `p` is the largest right
  breakpoint, found by the same leaf scan.

The vocabulary and the reconstruction rules follow the slope-trick
literature; the code here is independent.

## See also

- [`convex_piecewise_linear_forest`](convex-piecewise-linear-forest.md): the
  same vocabulary for many functions; use it for sums and repeated clipping
  of large sides.
- [DP optimization](dp-optimization.md): DPs whose cost is Monge rather than
  convex in the state.
