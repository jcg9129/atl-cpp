# `atl` DP optimization: row minima, 1D/1D, layered and interval DPs

Four speedups for dynamic programs whose transition cost obeys the quadrangle
(Monge) inequality: row minima of implicit monotone matrices
(`atl::monotone_row_minima`, `atl::totally_monotone_row_minima`), the 1D/1D DP
`dp[i] = min_j dp[j] + cost(j, i)` (`atl::optimal_segmentation`), cuts into
exactly `k` segments (`atl::optimal_partition`, `atl::optimal_partition_cuts`)
and interval merges (`atl::optimal_interval_merge`, `atl::optimal_merge_order`).
Reach for them when the plain DP is quadratic or cubic in the cost
evaluations and the cost has the structure; without the quadrangle inequality
they return wrong answers, not errors, so prove it first. For linear
transitions `a_j * x_i + b_j` use [Li Chao trees](li-chao-tree.md); for a DP
row that is a convex function of its index, the
[slope trick](convex-piecewise-linear.md).

```cpp
#include <atl/dp/concepts.hpp>          // matrix_cost, additive_cost, cost_value_t
#include <atl/dp/row_minima.hpp>        // monotone_row_minima, totally_monotone_row_minima
#include <atl/dp/segmentation.hpp>      // optimal_segmentation
#include <atl/dp/optimal_partition.hpp> // optimal_partition, optimal_partition_cuts
#include <atl/dp/interval_merge.hpp>    // optimal_interval_merge, optimal_merge_order
```

\include dp-optimization/quick.cpp

## Shared interface

All entities are in namespace `atl` and every function is `constexpr`.

```cpp
template<typename Cost>
using cost_value_t = std::remove_cvref_t<std::invoke_result_t<Cost &, uint32, uint32>>;

template<typename Cost, typename Compare>
concept matrix_cost =
	std::invocable<Cost &, uint32, uint32> && std::semiregular<cost_value_t<Cost>>
	&& std::strict_weak_order<Compare const &, cost_value_t<Cost> const &, cost_value_t<Cost> const &>;

template<typename Cost, typename Compare>
concept additive_cost = matrix_cost<Cost, Compare> && requires(cost_value_t<Cost> const a) {
	{ a + a } -> std::convertible_to<cost_value_t<Cost>>;
};
```

- **The cost is a callable** `cost(i, j)` on two `uint32` indices, called as
  an lvalue: a stateful cost (a two-pointer window, a memo), a mutable lambda,
  a function or a `std::reference_wrapper` is accepted. Each function states
  the order it queries entries in.
- **`Value` is whatever the cost returns** (`cost_value_t<Cost>`). The DP
  solvers add values (`dp[j] + cost(j, i)`) and start from `Value{}`; sums
  are converted back to `Value`, so narrow types (`int16`, `uint8`) work, as
  do `int64`, `int128` and `double`. No infinity is ever formed:
  unreachable states are left out of the scan, not padded, so `Value` only
  has to hold real totals, and costs may be negative or zero. `bool` is
  rejected by the solvers that index raw rows (`totally_monotone_row_minima`,
  `optimal_partition`, `optimal_partition_cuts`, `optimal_interval_merge`):
  `std::vector<bool>` has no `data()`.
- **Candidate totals are formed wide.** For an integral `Value` ordered by
  `std::less` or `std::greater` (the `std::ranges` forms included), every
  candidate `dp[j] + cost(j, i)` (and the interval merge's child sums) is
  formed and compared in a wider integer: `int32` for `int8` and `int16`,
  `int64` for `int32`, `int128` (software on MSVC and 32-bit targets) for
  `int64`. Only the `dp` values the result holds must fit `Value`; a losing
  candidate may overflow it. A 128-bit `Value`, a floating `Value` or any
  other comparator has no wider type, and there every candidate must fit.
- **`Compare` picks min or max**, default `std::less<>`; `comp(a, b)` reads
  "a is better than b", called through `std::invoke` (a member-function
  pointer `&V::less` works). `std::greater<>` maximizes, and every
  precondition below flips with it (Monge becomes inverse Monge).
- **Ties go to the smallest index** (strict comparisons throughout), so
  argmins, decisions and cuts are deterministic and identical across the
  algorithms and the experimental contenders.
- **Segments are half-open over boundaries `0 .. n`**: `cost(j, i)` prices
  the segment `[j, i)` of items `j .. i-1`.
- **The structural law is a documented precondition.** No concept can check
  monotone minima or the quadrangle inequality, and a sampled debug check
  would add cost evaluations that break the query order a stateful cost
  relies on, while passing most broken costs anyway. A cost that breaks the
  law gives a wrong answer, not an abort.

The quadrangle inequality, for `a <= b < c <= d`:

```
cost(a, c) + cost(b, d)  <=  cost(a, d) + cost(b, c)
```

Typical sources: `f(S[i] - S[j])` with `f` convex and `S` nondecreasing
(prefix sums); sums of a non-negative density over sub-segments; "distinct
values in `[j, i)`" satisfies the reverse (maximize it).

| function | recurrence | precondition | evaluations |
|---|---|---|---|
| `monotone_row_minima(rows, cols, cost)` | each row's best column | monotone leftmost row argmins | O((rows + cols) log rows) |
| `totally_monotone_row_minima(rows, cols, cost)` | each row's best column | totally monotone (Monge suffices) | O(rows + cols) |
| `optimal_segmentation(n, cost)` | `dp[i] = min_{j<i} dp[j] + cost(j, i)` | quadrangle inequality | O(n log n) |
| `optimal_partition(n, k, cost)` | exactly `1 .. k` segments | quadrangle inequality | O(min(k n log n, n² + k n)) |
| `optimal_interval_merge(n, cost)` | `dp[l][r] = min_{l<m<r} dp[l][m] + dp[m][r] + cost(l, r)` | QI + monotone on inclusion | O(n²) |

Every index count is asserted below `2^32 - 1` (the family's `uint32`
indices keep one value free); the per-function bounds follow.

Measurements on this page: Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2.0,
RelWithDebInfo (`-O2 -g -DNDEBUG -std=c++23`, optionally `-march=native`),
assertions off; each cell is the minimum per-run mean over 3 runs of 25
samples, in ms, as "plain `-O2` / `-march=native`". Records of 2026-10-15,
except the partition engines (2026-11-01). Those records formed candidate totals in
the cost's own type; the wide totals of the shared interface are not
measured yet (the `int64` rows now add and compare in `int128`).

## Row minima

For every row of the implicit `rows x cols` matrix `cost(i, j)`, its leftmost
best column and value. The engine of the divide-and-conquer optimization
(one call per layer of `dp_k[i] = min_j dp_{k-1}[j] + w(j, i)`) and of SMAWK
applications such as nearest neighbours between sorted sets.

\include dp-optimization/row-minima.cpp

### API

```cpp
template<std::semiregular Value>
struct row_minima_result {
	std::vector<uint32> column; // column[i]: leftmost best column of row i
	std::vector<Value> value;   // value[i]: the entry at column[i]
};

template<typename Cost, typename Compare = std::less<>>
	requires matrix_cost<Cost, Compare>
[[nodiscard]] constexpr row_minima_result<cost_value_t<Cost>> monotone_row_minima(std::size_t rows, std::size_t cols,
                                                                                  Cost &&cost, Compare comp = {});

template<typename Cost, typename Compare = std::less<>>
	requires matrix_cost<Cost, Compare> && detail::contiguously_storable<cost_value_t<Cost>>
[[nodiscard]] constexpr row_minima_result<cost_value_t<Cost>> totally_monotone_row_minima(std::size_t rows, std::size_t cols,
                                                                                          Cost &&cost, Compare comp = {});
```

`detail::contiguously_storable<Value>` is every type but `bool`.

- `monotone_row_minima` (divide and conquer) needs only that each row's
  leftmost best column is at least the previous row's. It scans the middle
  row left to right and splits the column window at its winner for the two
  halves, so a two-pointer cost moves O((rows + cols) log rows) steps.
- `totally_monotone_row_minima` (SMAWK) needs total monotonicity: in every
  2x2 submatrix (rows `i < i'`, columns `j < j'`), if `j'` is strictly better
  in row `i` it is strictly better in row `i'`, and if it is at least as good
  in row `i` it is at least as good in row `i'`. A Monge matrix satisfies it.
  Monotone argmins are not enough (the tests show SMAWK missing rows of such
  matrices). Entries are queried in no locality-friendly order.

Both return the leftmost best column of every row, so they agree on every
totally monotone matrix.

### Complexity and preconditions

`monotone_row_minima`: O((rows + cols) log rows) evaluations;
`totally_monotone_row_minima`: O(rows + cols) evaluations and O(rows + cols)
memory. Both recurse at most log2(rows) + 1 levels deep (divide and conquer
recurses into the smaller half and loops on the larger; SMAWK halves the
rows), so a million rows need about twenty frames.

Asserted: `rows, cols < 2^32 - 1`; `cols >= 1` whenever `rows >= 1`.
Documented only: monotone leftmost argmins, or total monotonicity.

### Measured performance

[`benchmarks/dp/row_minima.cpp`](../benchmarks/dp/row_minima.cpp) times one
whole call of each library function. `A[i][j] = f(x_i - y_j)` over sorted
random points: "cheap" `f = d²`, "pow1.5" `f = |d|^1.5` (a sqrt).

| rows x cols | cost | `totally_monotone_row_minima` | `monotone_row_minima` |
|---|---|---|---|
| 1e5 x 1e5 | cheap | 2.9 / 2.9 | 3.3 / 2.9 |
| 1e5 x 1e5 | pow1.5 | 3.8 / 3.8 | 4.4 / 4.4 |
| 1e6 x 1e6 | cheap | 37.5 / 37.3 | 41.0 / 36.2 |
| 1e6 x 1e6 | pow1.5 | 44.3 / 43.9 | 54.1 / 54.7 |
| 1e5 x 1e6 | cheap | 10.6 / 10.4 | 21.5 / 17.6 |
| 1e5 x 1e6 | pow1.5 | 13.9 / 13.5 | 32.0 / 31.4 |
| 1e6 x 1e5 | cheap | 11.4 / 12.0 | 12.5 / 11.8 |
| 1e6 x 1e5 | pow1.5 | 12.6 / 12.5 | 15.7 / 15.8 |

SMAWK wins or ties every shape: 1.7-2.3x when columns outnumber rows, and
1.16-1.26x with the pow1.5 cost on the other shapes. On square and tall
shapes with the cheap cost it leads by at most 1.14x at `-O2`, and the
divide and conquer is up to 3% faster with `-march=native`.

### Pitfalls and notes

Use SMAWK whenever the matrix is totally monotone. Keep the divide and
conquer for matrices that are only monotone, and for two-pointer costs,
which need its scan order.

## `optimal_segmentation`: the 1D/1D DP

The best way to cut the boundaries `0 .. n` into segments when `cost(j, i)`
prices the segment `[j, i)`: `dp[0] = 0`, `dp[i] = min over j < i of dp[j] + cost(j, i)`. Line breaking, partitioning with a convex penalty, any
`f(S[i] - S[j])` transition with `f` convex: O(n log n) evaluations where
the plain DP needs O(n²). For *exactly* `k` segments use `optimal_partition`
below or [`exact_count_optimum`](exact-count-optimum.md) over this solver.

### API

```cpp
template<std::semiregular Value>
struct segmentation_result {
	std::vector<Value> value;     // value[i] = dp[i], value[0] = Value{}
	std::vector<uint32> decision; // smallest best j for each i, decision[0] = 0
	[[nodiscard]] constexpr std::vector<uint32> boundaries() const; // 0 = b_0 < ... < b_k = n
};

template<typename Cost, typename Compare = std::less<>>
	requires additive_cost<Cost, Compare>
[[nodiscard]] constexpr segmentation_result<cost_value_t<Cost>> optimal_segmentation(std::size_t n, Cost &&cost,
                                                                                     Compare comp = {});
```

- `n`: the number of items; boundaries are `0 .. n`.
- `cost(j, i)`: the price of the segment `[j, i)`, called for `0 <= j < i <= n`.
- Returns `value[i]`, the best cost of segmenting the prefix `[0, i)`, and
  `decision[i]`, the start of the last segment of that optimum: the smallest
  best `j`. `boundaries()` walks `i = decision[i]` from `n` and returns the
  cuts in increasing order, segment `t` being `[b_t, b_{t+1})`; it is `{0}`
  for `n = 0` and empty for a default-constructed result.

### Complexity and preconditions

O(n log n) evaluations of `cost` (each comparison evaluates two), in no
locality-friendly order; O(n) memory (`value`, `decision` and the two queue
arrays hold `n + 1` entries each); no recursion. `boundaries()` is O(k) for
`k` segments.

Asserted: `n < 2^32 - 2`. Documented only: the quadrangle inequality (the
reverse one with `std::greater<>`); `Value` holds every `dp` value (and,
without a wider type, every `dp` value plus one cost).

Floating-point costs: the takeover search relies on the comparison of two
candidates crossing once along the boundaries, as the exact comparisons do
under the quadrangle inequality. Rounding can break that near ties: with
`n = 4`, `B = {0, 3, 2^54, 0, 2^54}` and `cost(j, i) = double(B[i] - (j ==
1 ? 4 : 0))` (additively separable, so the inequality holds with equality),
candidate 1 ties candidate 0 at boundaries 2 and 4 after rounding but wins
at 3, the search discards it, and `value[3]` comes out 0 instead of -1. A
`double` cost must keep the comparisons exact (every `dp` value plus cost
representable, e.g. integers below `2^53`) or single-crossing after
rounding.

### Measured performance

[`benchmarks/dp/segmentation.cpp`](../benchmarks/dp/segmentation.cpp) races
the library's decision queue against three contenders in
`experimental/atl/experimental/dp/segmentation_variants.hpp` that differ in
how a boundary's best decision is found: halving divide and conquer with the
monotone divide and conquer or with SMAWK across the halves, and SMAWK on
doubling blocks (the linear-time method). Segment cost `f(S_i - S_j - L)`
over prefix sums: "cheap" `f = d²`, "pow1.5" `f = |d|^1.5`.

| n | cost | library | by halving divide and conquer | by halving SMAWK | by SMAWK blocks |
|---|---|---|---|---|---|
| 1e5 | cheap | 4.3 / 4.4 | 20.9 / 21.6 | 25.6 / 26.2 | 7.7 / 7.1 |
| 1e5 | pow1.5 | 7.9 / 7.7 | 29.4 / 28.4 | 30.0 / 30.0 | 8.3 / 8.2 |
| 1e6 | cheap | 50.2 / 52.4 | 276 / 282 | 295 / 302 | 81.9 / 74.7 |
| 1e6 | pow1.5 | 93.7 / 91.0 | 390 / 379 | 349 / 344 | 87.7 / 86.4 |

The decision queue is fastest on three shapes of four, by 1.05-1.8x over the
SMAWK blocks and 3.7-6.0x over the halving contenders. The SMAWK blocks pull
ahead only at n = 1e6 with the pow1.5 cost, by 5-6%: their fewer
evaluations pay off once an evaluation is dear.

### Pitfalls and notes

- It is **online**: `dp[i]` is final before any boundary past `i` is asked
  about, which the recurrence needs (row `j` of the matrix exists only once
  `dp[j]` is known). That is why the 1D/1D DP cannot simply call
  `monotone_row_minima`.
- Each comparison evaluates the cost twice, so make it cheap (prefix sums,
  a precomputed table).

## `optimal_partition`: exactly `k` segments

Element `k - 1` of the result is the best cost of cutting `[0, n)` into
exactly `k` non-empty segments, for every `k = 1 .. segments`; the whole
curve comes out of one run. `optimal_partition_cuts` returns one optimum's
boundaries for one `k`.

\include dp-optimization/partition.cpp

### API

```cpp
template<typename Cost, typename Compare = std::less<>>
	requires additive_cost<Cost, Compare> && detail::contiguously_storable<cost_value_t<Cost>>
[[nodiscard]] constexpr std::vector<cost_value_t<Cost>> optimal_partition(std::size_t n, std::size_t segments, Cost &&cost,
                                                                          Compare comp = {});

template<std::semiregular Value>
struct partition_cuts_result {
	Value value{};                  // best cost with exactly `segments` segments
	std::vector<uint32> boundaries; // 0 = b_0 < ... < b_k = n
};

template<typename Cost, typename Compare = std::less<>>
	requires additive_cost<Cost, Compare> && detail::contiguously_storable<cost_value_t<Cost>>
[[nodiscard]] constexpr partition_cuts_result<cost_value_t<Cost>> optimal_partition_cuts(std::size_t n, std::size_t segments,
                                                                                         Cost &&cost, Compare comp = {});
```

The layered DP `dp_k[i] = min over k-1 <= j < i of dp_{k-1}[j] + cost(j, i)`, with `opt_k[i]` its leftmost best `j`, one layer at a time, each on one
of two engines: the row minima of the staircase matrix by divide and conquer
(O(n log n) evaluations per layer whatever `k` is), or Knuth's layered bound
`opt_{k-1}[i] <= opt_k[i] <= opt_k[i+1]` (each row scans only that window;
the windows shrink as `k` grows). Both find leftmost argmins, so the values
and cuts do not depend on which engine ran.

The cuts are read back from the leftmost argmins: `b_{k-1} = opt_k[n]`, then
`b_{t-1} = opt_t[b_t]`. The last cut is the smallest one of any optimum, and
each earlier cut is the smallest best one for the prefix it ends.

Within a layer, rows are settled one at a time, each scanning its candidate
`j` upward over a window, so a two-pointer cost (a sliding window that moves
its ends to the asked segment, as for "distinct values in a segment") moves
O(evaluations) steps.

### Complexity and preconditions

O(min(segments n log n, n² + segments n)) evaluations of `cost`.
`optimal_partition` keeps O(n) memory; `optimal_partition_cuts` keeps every
layer's argmins over the rows it fills, `(k - 1) (n - k + 1) + k` `uint32`s
(about 40 MB at `k = 100`, `n = 1e5`). Recursion depth O(log n).

Asserted: `n < 2^32 - 2`; `segments <= n` (`1 <= segments` for the cuts;
`segments = 0` gives an empty curve); the argmin table's size fits
`std::size_t` (binds on 32-bit targets). Documented only: the quadrangle
inequality, which gives both monotonicities the engines rest on; a cost that
only has monotone layer argmins (enough for divide and conquer alone) is not
enough. `Value` holds every `dp_k[i]` (and, without a wider type, every
candidate `dp_{k-1}[j] + cost(j, i)`); not `bool` (rejected).

### Measured performance

[`benchmarks/dp/optimal_partition.cpp`](../benchmarks/dp/optimal_partition.cpp)
races the library against contenders in
`experimental/atl/experimental/dp/optimal_partition_variants.hpp` that change
only how a layer is solved. The 2026-11-01 record ("layer engines") covers
two of them: the divide-and-conquer rows also floored at the previous
layer's argmin ("by floored divide and conquer"), and SMAWK on every layer
but the last ("by SMAWK", opt-in only: it evaluates out of order). Costs over
random prefix sums `S` (steps below 1000): "cheap" `(S_i - S_j)²`, "pow1.5"
`|S_i - S_j - L|^1.5`.

| n | K | cost | library | by floored divide and conquer | by SMAWK |
|---|---|---|---|---|---|
| 1e3 | 2 | cheap | 0.002 / 0.002 | 0.002 / 0.002 | 0.002 / 0.002 |
| 1e3 | 2 | pow1.5 | 0.004 / 0.004 | 0.004 / 0.004 | 0.004 / 0.004 |
| 1e3 | 20 | cheap | 0.289 / 0.268 | 0.308 / 0.294 | 0.596 / 0.644 |
| 1e3 | 20 | pow1.5 | 0.419 / 0.420 | 0.424 / 0.440 | 0.977 / 0.905 |
| 1e3 | 100 | cheap | 1.0 / 1.0 | 1.0 / 0.992 | 3.2 / 3.4 |
| 1e3 | 100 | pow1.5 | 1.5 / 1.5 | 1.5 / 1.5 | 5.2 / 4.8 |
| 1e3 | 1e3 | cheap | 3.3 / 3.3 | 3.2 / 3.1 | 15.0 / 16.6 |
| 1e3 | 1e3 | pow1.5 | 4.8 / 5.1 | 4.7 / 5.0 | 25.1 / 21.8 |
| 5e3 | 2 | cheap | 0.012 / 0.011 | 0.012 / 0.010 | 0.012 / 0.010 |
| 5e3 | 2 | pow1.5 | 0.023 / 0.022 | 0.023 / 0.023 | 0.022 / 0.022 |
| 5e3 | 20 | cheap | 2.0 / 1.8 | 2.0 / 2.0 | 3.0 / 3.3 |
| 5e3 | 20 | pow1.5 | 3.1 / 3.2 | 3.0 / 3.0 | 5.0 / 4.6 |
| 5e3 | 100 | cheap | 6.6 / 6.3 | 6.6 / 6.2 | 16.7 / 18.3 |
| 5e3 | 100 | pow1.5 | 9.7 / 9.8 | 9.7 / 9.7 | 28.2 / 25.9 |
| 5e3 | 500 | cheap | 24.1 / 23.8 | 23.8 / 23.2 | 81.5 / 89.2 |
| 5e3 | 500 | pow1.5 | 34.7 / 36.3 | 34.6 / 36.5 | 135 / 124 |
| 5e3 | 5e3 | cheap | 84.8 / 80.8 | 77.8 / 80.6 | 386 / 427 |
| 5e3 | 5e3 | pow1.5 | 116 / 128 | 116 / 128 | 647 / 566 |
| 2e4 | 2 | cheap | 0.050 / 0.044 | 0.048 / 0.044 | 0.049 / 0.044 |
| 2e4 | 2 | pow1.5 | 0.094 / 0.093 | 0.092 / 0.092 | 0.092 / 0.093 |
| 2e4 | 20 | cheap | 8.7 / 7.7 | 8.8 / 8.9 | 12.4 / 13.4 |
| 2e4 | 20 | pow1.5 | 14.1 / 14.1 | 13.3 / 13.4 | 20.4 / 19.1 |
| 2e4 | 100 | cheap | 33.7 / 31.0 | 33.3 / 31.4 | 69.1 / 76.8 |
| 2e4 | 100 | pow1.5 | 50.5 / 51.3 | 49.0 / 49.4 | 115 / 108 |
| 2e4 | 2000 | cheap | 385 / 384 | 382 / 364 | 1344 / 1474 |
| 2e4 | 2000 | pow1.5 | 558 / 569 | 559 / 572 | 2217 / 2038 |
| 2e4 | 2e4 | cheap | 1538 / 1521 | 1473 / 1460 | 6298 / 7004 |
| 2e4 | 2e4 | pow1.5 | 2147 / 2263 | 2142 / 2268 | 10509 / 9184 |

The floor ties the library: between 0.92x and 1.16x of its time on every
shape, with no consistent direction (the largest gap, n = 2e4, K = 20, cheap,
`-march=native`: 8.9 against 7.7 ms). SMAWK ties at K = 2, where every
contender runs the last layer's single row, and is 1.35-5.6x slower from
K = 20 up: its linear evaluation count does not make up for its constant
factor and out-of-order access.

The race against Knuth's bound on every layer and divide and conquer on
every layer, the switch-budget calibration, the Knuth engine's tuning steps
and the two large workloads in the same benchmark are not measured yet in
the current records; the experimental header's docstrings keep the race's
verdict.

### Pitfalls and notes

Without the cut positions `optimal_partition` keeps O(n) memory. When the
`(k - 1) (n - k + 1)` argmins of `optimal_partition_cuts` are too much, the
Lagrangian route ([`exact_count_optimum`](exact-count-optimum.md) over
`optimal_segmentation`) keeps O(n), with the reconstruction caveats on its
page. The whole curve `k -> best(k)` is what such a Lagrangian solver needs
to be convex; it is tested against this one.

## `optimal_interval_merge`: Knuth's optimization

`dp[l][l+1] = 0`, `dp[l][r] = min over l < m < r of dp[l][m] + dp[m][r] + cost(l, r)`: merging `[l, r)` costs `cost(l, r)` on top of its two halves
(stone merging, optimal binary search trees, optimal alphabetic codes), in
O(n²) instead of O(n³).

\include dp-optimization/interval-merge.cpp

### API

```cpp
template<std::semiregular Value, typename Compare = std::less<>>
	requires std::copy_constructible<Compare> && std::strict_weak_order<Compare const &, Value const &, Value const &>
struct interval_merge_result {
	uint32 items = 0;
	std::vector<Value> table; // row l holds r = l+1 .. items
	Compare comp{};           // the order the table was built with
	[[nodiscard]] constexpr std::size_t index(std::size_t l, std::size_t r) const;
	[[nodiscard]] constexpr Value const &value(std::size_t l, std::size_t r) const; // best cost of [l, r)
};

template<typename Cost, typename Compare = std::less<>>
	requires additive_cost<Cost, Compare> && detail::contiguously_storable<cost_value_t<Cost>>
			 && std::copy_constructible<Compare>
[[nodiscard]] constexpr interval_merge_result<cost_value_t<Cost>, Compare>
optimal_interval_merge(std::size_t items, Cost &&cost, Compare comp = {});

struct merge_step {
	uint32 left = 0, split = 0, right = 0; // [left, split) + [split, right) -> [left, right)
};

template<std::semiregular Value, typename Compare>
	requires std::strict_weak_order<Compare const &, Value const &, Value const &>
		  && requires(Value const a) {
				 { a + a } -> std::convertible_to<Value>;
			 }
[[nodiscard]] constexpr std::vector<merge_step> optimal_merge_order(interval_merge_result<Value, Compare> const &result);
```

- `cost(l, r)` is queried once for every `0 <= l`, `l + 2 <= r <= n`, `l`
  descending and `r` ascending within an `l`. Single items cost `Value{}`.
- The closed form `f[i][j] = min_{i <= k < j} f[i][k] + f[k+1][j] + w(i, j)`
  is `dp[i][j+1]` with `cost(l, r) = w(l, r - 1)`.
- `optimal_merge_order` returns the `items - 1` merges of one optimal tree in
  an executable order (each step's halves are single items or earlier
  steps' results; the last makes `[0, items)`), from the table alone: the
  merge cost does not depend on the split, so a best split of `[l, r)` is any
  `m` minimizing `value(l, m) + value(m, r)`. The leftmost one is taken,
  under the comparator the table stores (`result.comp`, the one the solver
  was called with), the sums formed in the same wide type as the solver's.

### Complexity and preconditions

O(n²) time and evaluations, O(n²) memory: `items (items + 1) / 2` values,
about 100 MB of `int64` at `n = 5000`; only two rows of split points are
kept. `optimal_merge_order` costs O(sum of the tree's segment lengths):
O(n log n) for a balanced tree, O(n²) at worst, with an explicit stack.

Asserted: `items < 2^32 - 2`; the table's size fits `std::size_t` (binds on
32-bit targets, from about `n = 65000`); `value(l, r)` needs `l < r <= items`.
Documented only: for `a <= b < c <= d`, the quadrangle inequality and
monotonicity on inclusion, `cost(b, c)` no worse than `cost(a, d)` (with the
unqueried single-item costs chosen to fit). A cost that breaks them is not
detected: the windows are never empty, so the scan silently returns a
non-optimal table. `Value` holds every segment's best cost; the child sums
`dp[l][m] + dp[m][r]` are formed wide (see the shared interface), so with
merge costs `-30000`, `0`, `30000` for two, three and four `int16` items the
balanced tree's `-60000` is seen and the answer is `-30000`. Without a wider
type every candidate total must fit `Value`.

### Measured performance

[`benchmarks/dp/interval_merge.cpp`](../benchmarks/dp/interval_merge.cpp)
times one whole table on stone merging with random weights; there is no
race.

| workload | ms |
|---|---|
| stone merging, n = 2000 | 18.1 / 19.5 |
| stone merging, n = 5000 | 214 / 222 |

## How it works

### Row minima

**Divide and conquer.** The middle row of a row range is scanned over its
column window left to right, keeping the leftmost best; since leftmost
argmins are nondecreasing, the rows above take columns up to the winner and
the rows below from it. The column windows of one recursion level overlap
only at their ends, so each level costs O(rows + cols) evaluations, over
log rows levels. The code recurses into the smaller half and loops on the
larger, which bounds the depth. A per-row column cap lets
`optimal_partition` run its staircase matrix (row `i` defined on columns
`< i`) through the same loop without padding the undefined entries.

**SMAWK** (Aggarwal, Klawe, Moran, Shor and Wilber, 1987, "Geometric
applications of a matrix-searching algorithm"). One level on a set of rows:
REDUCE, when there are more columns than rows, makes a left-to-right stack
pass that keeps at most one column per row. A column strictly better than
the stack top in the top's row kills the top (by total monotonicity the top
can then be best in no row at or below), otherwise it is pushed if the stack
has room. Ties keep the left column, so the survivors include every row's
leftmost best. The odd rows recurse on the survivors, and each even row scans
the survivors between its neighbours' answers. The stack caches each entry's
value in its own row, so a comparison evaluates one new entry, and the work
per level is linear in the rows and survivors, which halve with the rows.

### The decision queue

Candidate `j` offers `dp[j] + cost(j, x)` at every later boundary `x`. The
quadrangle inequality makes the candidates decision-monotone: for `j < i`,
if `i` is strictly better than `j` at some boundary `x`, it is strictly
better at every `x' > x` (add `cost(j, x) + cost(i, x') <= cost(j, x') + cost(i, x)` to the strict inequality at `x`). So against any older candidate
a newer one wins on a suffix of the boundaries.

The code keeps a queue of candidates in two flat `uint32` arrays,
`candidate[k]` owning the run of boundaries `[start[k], start[k+1])`, the
last run extending to `n`. Processing boundary `i`:

1. Drop candidates from the front while the next run starts at or before `i`.
2. The front candidate is the best for `i`: record `decision[i]` and
   `value[i]`.
3. Offer `i` for the boundaries `i + 1 .. n`: from the back, while `i` is
   strictly better than the back candidate at that run's first boundary, it
   wins the whole run (the suffix property), and the run is popped.
4. In the surviving back run, binary-search the first boundary where `i` is
   strictly better and push `i` from there; if there is none, `i` is not
   pushed. The last boundary `n` skips steps 3 and 4.

Every candidate is pushed and popped at most once (O(n) evaluations), and
the one binary search per boundary costs O(log n) comparisons: O(n log n)
in all. Strict comparisons everywhere give the smallest best `j`. Values are
recomputed through `value[j] + cost(j, i)`, so nothing but `value` is stored
per candidate and no infinity exists.

References: Hirschberg and Larmore, 1987, "The least weight subsequence
problem" (the quadrangle inequality and the queue); Galil and Park, 1990, "A
linear-time algorithm for concave one-dimensional dynamic programming" (the
linear-time method the SMAWK-blocks contender follows); Knuth and Plass,
1981, "Breaking paragraphs into lines".

### Exactly `k` segments

**Row monotonicity, `opt_k[i] <= opt_k[i+1]`.** If `c = opt_k[i+1] < d = opt_k[i]`, then `c` is no worse than `d` for row `i + 1` and `d` strictly
better than `c` for row `i` (leftmost ties). Adding the two contradicts the
quadrangle inequality for `c < d < i < i + 1`.

**Layer monotonicity, `opt_{k-1}[i] <= opt_k[i]`.** For `b < a`, `dp_k[a] + dp_{k-1}[b] <= dp_{k-1}[a] + dp_k[b]`: take an optimal `(k-1)`-segment path
`p` to `a` and an optimal `k`-segment path `q` to `b`; some segment `[q_t, q_{t+1})` nests in `[p_{t-1}, p_t)`, and swapping the tails there gives a
`k`-segment path to `a` and a `(k-1)`-segment path to `b` costing no more
together (the quadrangle inequality on `p_{t-1} < q_t < q_{t+1} <= p_t`). If
`b = opt_k[i] < a = opt_{k-1}[i]`, optimality and leftmost ties sum to the
opposite strict inequality.

**The engine switch.** Row `i` of layer `k` scans `opt_k[i+1] - opt_{k-1}[i] + 1` candidates. Along a diagonal `i - k` these telescope to at
most `n`, so Knuth's bound costs O(n² + k n) over all layers, but on balanced
costs layer `k` alone takes about `n² / (2 k²)`: Θ(n²) for any `k >= 3`.
After each layer the solver adds up its Knuth windows (stopping the count
once past the threshold, so early layers pay only a few rows for it) and
switches to Knuth's bound once that total is at most `rows * bit_width(rows)`, an estimate of a divide-and-conquer layer. A Knuth layer
gets that much as a budget; past it, its remaining lower rows go to divide
and conquer, capped by the argmin of the last row it finished. So no layer
costs more than about two divide-and-conquer layers, and the switch comes
near `k = sqrt(n / (2 log n))`. The last layer needs only `dp_k[n]`: one
window.

**What this code does.** Two rolling rows of values and of `uint32` argmins
(`optimal_partition_cuts` keeps one argmin row per layer, only over the rows
`k .. n - (segments - k)` each layer fills, as every later layer needs an
item). In the Knuth scan the previous row's argmin stays in a register
rather than being reloaded from the row just written. A window's upper end
is raised to its lower end, so even a cost that breaks the precondition
gets a real partition's cost, never an empty scan.

References: Knuth, 1971, "Optimum binary search trees" (monotone splits);
Yao, 1980, "Efficient dynamic programming using quadrangle inequalities".

### Interval merges

Under both conditions the leftmost best split is monotone, `opt[l][r-1] <= opt[l][r] <= opt[l+1][r]` (Knuth, 1971; Yao, 1980), and the windows
telescope to O(n) per left end. Rows are filled from `l = items - 1` down
with `r` increasing, so only the split rows of `l` and `l + 1` are kept
(`opt[l][l+1] = l` makes the `r = l + 2` window the single split `l + 1`),
and the window is never empty by construction (`opt[l][r-1] <= opt[l+1][r-1] <= opt[l+1][r]`). The table is stored row by row,
`[l, r)` at `l (2 items - l + 1) / 2 + (r - l - 1)`.
`optimal_merge_order` walks the tree in preorder from `[0, items)` with an
explicit stack, picking each split from the table, and reverses the steps
so that every step follows its halves.

## See also

[Exact-count optimum](exact-count-optimum.md) (exactly `k` items by
Lagrangian relaxation), [Li Chao trees](li-chao-tree.md) (the convex hull
trick), [convex piecewise-linear functions](convex-piecewise-linear.md) and
[their forest](convex-piecewise-linear-forest.md) (the slope trick, also on
trees).
