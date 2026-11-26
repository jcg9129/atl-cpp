# `atl::exact_count_optimum`

The exact optimum of a problem that must use exactly `k` items (segments,
chosen elements, groups), given a solver for the same problem with the count
left free and every item priced instead: Lagrangian relaxation, also called
the WQS binary search. It drops the count from the DP state, so an O(k n)
state space becomes O(n), solved O(log V) times for a price range of size V.
Reach for it when the best value is convex (minimizing) or concave
(maximizing) in the count and `k` is large; for small `k` with a
quadrangle-inequality cost, the layered
[`optimal_partition`](dp-optimization.md) is faster and returns the cuts.

```cpp
#include <atl/dp/exact_count_optimum.hpp>
```

\include exact-count-optimum/quick.cpp

## API

All entities are in namespace `atl`; the function is `constexpr`.

```cpp
template<typename Solve, typename Price>
concept penalized_solver =
	integral<Price> && std::invocable<Solve &, Price>
	&& detail::pair_like<std::remove_cvref_t<std::invoke_result_t<Solve &, Price>>>
	&& integral<std::remove_cvref_t<std::tuple_element_t<1, std::remove_cvref_t<std::invoke_result_t<Solve &, Price>>>>>;

template<typename Solve, typename Price>
using penalized_value_t = /* the first element of solve(price)'s result, cv-ref dropped */;
template<typename Solve, typename Price>
using penalized_count_t = /* the second element, an integer */;

template<std::movable Value, integral Price, integral Count>
struct exact_count_result {
	Value value; // the best value with exactly `count` items
	Price price; // the smallest price in [lo, hi] at which `count` is optimal
	Count count; // the count the solver reported at `price`, at most the asked one
};

template<typename Price, typename Solve, typename Compare = std::less<>>
	requires penalized_solver<Solve, Price>
		  && detail::penalty_arithmetic<penalized_value_t<Solve, Price>, Price, penalized_count_t<Solve, Price>>
		  && std::strict_weak_order<Compare const &, penalized_value_t<Solve, Price> const &,
									penalized_value_t<Solve, Price> const &>
[[nodiscard]] constexpr exact_count_result<penalized_value_t<Solve, Price>, Price, penalized_count_t<Solve, Price>>
exact_count_optimum(std::type_identity_t<penalized_count_t<Solve, Price>> count, Price lo, Price hi, Solve &&solve,
					Compare comp = {});
```

Let `f(c)` be the best value with exactly `c` items.

- **Minimizing** (`std::less<>`, the default): `solve(price)` returns `{min over solutions of value + price * count, the count of a minimizer}`.
- **Maximizing** (`std::greater<>`): `solve(price)` returns `{max of value - price * count, the count of a maximizer}`. The direction is read off
  `comp(Value(1), Value(0))` (through `std::invoke`), so any comparator
  ordering the values numerically works.
- `solve` returns any pair-like value (`std::pair`, a two-element
  `std::tuple`) whose second element is an integer; it is called as an
  lvalue, so it may reuse buffers. `Price` is any integer type (`int128`
  included); `Value` must be constructible from `Price`, from the count type
  and from `int`, and support `+`, `-` and `*` (the `penalty_arithmetic`
  detail concept).
- The search finds the smallest integer price in `[lo, hi]` whose reported
  count is at most `count`, and returns `penalized - price * count`
  (`penalized + price * count` when maximizing), where `penalized` is the
  solver's value at that price. `price` is that price: `f`'s slope just
  right of `count` (`f(count + 1) - f(count)`) is `-price` when minimizing
  and `+price` when maximizing (maximizing `f(c) = 3c` over `0 .. 2`, count
  1 in `[0, 4]`, returns price 3 and the slope is 3), unless `count` is the
  largest count the range reaches, where the search stops at `lo`.
- `count` in the result is what the solver reported at `price`: `count`
  itself when the asked count is a vertex of `f`, and smaller -- the
  fewest-items end of the tie -- when it lies inside a collinear stretch.
  A caller that must build an exactly-`count` solution needs it to tell the
  two cases apart without another solver call (see Pitfalls).
- The final `price * count` and its sum with the solver's value are formed
  in a wider integer for an integral `Value` up to 64 bits (`int128` for
  `int64`), so only the answer has to fit: `Value = int64`, `Count =
  uint64`, count `2^63`, price `-1` gives the exact `0`.

## Complexity

At most `ceil(log2(hi - lo + 1)) + 1` calls of `solve`, nothing else of note.
The midpoint is formed without `lo + hi`, so `[INT64_MIN, INT64_MAX]` is a
valid range.

## Preconditions

Asserted: `lo <= hi`; the solver's count at `hi` is at most `count`; when
the search ends at `lo`, the count reported there is `count` exactly. Both
checks use answers the search already has. An infeasible `count`, below or
above every count the solver can reach in the range, fails one of them.
With every slope `f(c + 1) - f(c)` in `[-M, M]`, `[-M - 1, M + 1]` always
works when `count` is feasible.

Documented only:

- `f` is convex when minimizing (concave when maximizing) over the feasible
  counts, which form one interval.
- `f` is integer-valued, so its slopes are integers and integer prices reach
  every one.
- Among the solutions tied for the penalized optimum, `solve` reports the
  one with the **fewest** items, e.g. by comparing `(penalized value, count)` lexicographically (as the example's `counted` type does).
- `Value` holds the solver's penalized value at every probed price (the
  solver forms those itself) and the answer `f(count)`. A 128-bit or
  non-integral `Value` has no wider type, so there it also holds
  `price * count`, `Value(price)` and `Value(count)` at the final price.

## Measured performance

[`benchmarks/dp/exact_count_optimum.cpp`](../benchmarks/dp/exact_count_optimum.cpp)
cuts `n = 1e5` prefix-summed items (steps below 1000) into exactly `k`
segments of squared sum: the relaxation searches the price range `[-1, S² + 1]` (about 52 probes) over `optimal_segmentation`, against the layered
`optimal_partition` at one `k`.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| workload | ms |
|---|---|
| `exact_count_optimum` over `optimal_segmentation`, k = 20 | 161 / 165 |
| `exact_count_optimum` over `optimal_segmentation`, k = 1000 | 158 / 163 |
| `exact_count_optimum` over `optimal_segmentation`, k = 50000 | 143 / 147 |
| `optimal_partition`, k = 20 | 46.3 / 50.4 |

The relaxation varies less with `k` than the layered algorithm's layer
count would suggest. At k=20, relaxation loses to the layered solver
(161/165 versus 46.3/50.4 ms), taking 3.48/3.27 times as long. The table
does not locate a crossover at larger `k`. A tighter valid price range
reduces the binary search's probe bound.


## Pitfalls and notes

**The tie-break rule is the part that is easy to get wrong.** With the
most-items rule instead, the search stops one price past the breakpoint, and
on a collinear stretch of `f` the value is wrong (see How it works).

**Rebuilding an exactly-`count` solution.** `price` is returned for this.

1. Re-run your solver once at `price`, keeping its own reconstruction data
   (decisions, argmins). With the fewest-items tie-break it yields a
   solution optimal at `price` with `c_lo <= count` items, `c_lo` being the
   result's `count`; if `c_lo == count` (the usual case: `count` is a vertex
   of the curve), that solution is an exact optimum, and its value is
   `value`. The result's `count` says which case holds before the re-run.
2. Otherwise `count` sits inside a collinear stretch of `f`: at `price`
   every count in `[c_lo, c_hi]` ties, where `c_hi` is what a most-items
   tie-break reports (a second run with the tie-break flipped). No price
   makes the solver produce `count` items itself, so the solution has to be
   assembled from the two tied ones, and that step is problem-specific. For
   partitions under a quadrangle-inequality cost the two boundary lists can
   be spliced (a prefix of one, a suffix of the other, joined where they
   interleave) into one with exactly `count` segments and the same penalized
   value, the exchange argument behind the curve's convexity; atl does not
   provide the splice. Other relaxations (matchings, independent sets,
   flows) need a different exchange step, or none is known.
3. When O(count n) memory is affordable, skip all of this:
   `optimal_partition_cuts(n, count, cost)` in
   [DP optimization](dp-optimization.md) returns the boundaries of an
   exact-`count` optimum directly.

**Real-valued `f`.** Integer prices assume integer slopes. For a real-valued
`f`, run the same search on a fixed-point scaled value; it is not built in,
because exactness then needs care the caller has to supply.

## How it works

Minimizing (maximizing is the mirror image). Penalizing every item by `p`
turns the curve into `g_p(c) = f(c) + p c`. With integer slopes `s_c = f(c + 1) - f(c)`, nondecreasing by convexity, `c` minimizes `g_p` exactly when
`s_{c-1} <= -p <= s_c`, so a higher price can only lower the minimizers,
and the fewest-items rule reports the smallest of them. That is at most
`count` exactly when `p >= -s_count`: the reported count is monotone in the
price, the binary search finds `p = -s_count` (clamped to the range), and
`count` is a minimizer there because `s_{count-1} <= s_count`. So `g_p(count) = penalized` and `f(count) = penalized - p count`.

The value is computed from the penalized optimum, not from the solution the
solver found, and that is what makes the answer exact on a collinear stretch
of `f`: there several counts tie at every price, no price makes the solver
report `count` itself, yet at the found price `count` ties for the penalized
optimum.

The search keeps the smallest price that reported at most `count` and its
value, so the final answer needs no extra call unless no probe ever
succeeded (then `hi` is asked once, which is also what the assertion
checks). A search that ends at the range's first price had no probe below
it reporting more than `count`, so the count reported there is the only
evidence that `count` is optimal, and it must be `count` itself.

The underlying result is Lagrangian relaxation of a cardinality
constraint, as for the k-link shortest path in Aggarwal, Schieber and
Tokuyama, 1994, "Finding a minimum-weight k-link path in graphs with the
concave Monge property and applications".

## See also

[DP optimization](dp-optimization.md) (`optimal_segmentation` as the
penalized solver, `optimal_partition` as the exact-`k` alternative and test
oracle).
