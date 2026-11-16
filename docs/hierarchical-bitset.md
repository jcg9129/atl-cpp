# `atl::hierarchical_bitset`

An ordered set of positions in a bounded universe, supporting membership and inclusive
predecessor and successor queries. Use it when the universe fits in a bitset; use a
sparse ordered container when allocating in proportion to the universe is too expensive.

```cpp
#include <atl/ds/hierarchical_bitset.hpp>
```

\include hierarchical-bitset/basic.cpp

Expected stdout:

\verbinclude hierarchical-bitset/basic.out

## API

Every operation is `constexpr`.

| signature | behavior |
|---|---|
| `explicit hierarchical_bitset(std::size_t universe)` | Empty set on `[0, universe)`; zero is allowed |
| `template<std::ranges::input_range Range> hierarchical_bitset(std::size_t universe, Range&& elements)` | Build bottom-up; duplicates are harmless; range values must convert to `size_t`; `std::size_t(e)` must also work on the named element lvalue |
| `bool contains(std::size_t x) const` | Membership |
| `void insert(std::size_t x)`, `void erase(std::size_t x)` | Add or remove; duplicates and missing keys are harmless |
| `std::size_t next(std::size_t x) const` | Smallest present position at least `x` |
| `std::size_t prev(std::size_t x) const` | Largest present position at most `x` |
| `std::size_t min() const`, `std::size_t max() const` | Smallest or largest present position |
| `void clear()` | Empty the set, retaining storage |
| `std::size_t size() const noexcept`, `bool empty() const noexcept` | Present count and emptiness |
| `std::size_t universe() const noexcept` | Exclusive upper bound |
| `static constexpr std::size_t npos = std::size_t(-1)` | No matching position |

Neighbor and extremum queries return `npos` when none exists. For a strict successor
use `next(x + 1)`; for a strict predecessor use `prev(x - 1)` after checking `x > 0`.
`next(universe())` returns `npos`, including for a zero universe.

The bulk constructor retains
`std::convertible_to<std::ranges::range_value_t<Range>, std::size_t>` and also checks
`std::size_t(e)` for `e` of type
`std::remove_reference_t<std::ranges::range_reference_t<Range>>&`, matching its loop.
Explicit lvalue conversions work; rvalue-only conversions do not.

## Complexity

For universe size N, storage and construction cost O(1 + N/64) words and word
operations; bulk construction also costs O(number of inputs). Membership, `size`,
`empty` and `universe` cost O(1). Insert, erase, next, prev, min and max cost
O(1 + log_64 N) word operations. `clear` costs O(1 + N/64), or O(1) if already empty.
A full 32-bit universe needs six levels, rather than four.

## Preconditions

Asserted under `ATL_ASSERT`: positions are below `universe()`, except `next` accepts
`universe()` as well. Each converted bulk position must be in range.

Documented only: input conversions must preserve the intended position; allocations
must fit the backing vectors. Allocation failures propagate. No rank or select
operations are provided.

## Measured performance

The library uses heap-stored offsets and stops summary propagation when the
word is already nonzero. By full propagation writes every summary level;
by inline offsets keeps offsets in an array. `std::set` uses an ordered tree.
Loops measure throughput of random inserts, insert/erase churn, and next/prev
queries in dense and sparse universes.

Source: [`benchmarks/ds/hierarchical_bitset.cpp`](../benchmarks/ds/hierarchical_bitset.cpp).
CPU: Intel Xeon Platinum 8280; GCC 15.2; RelWithDebInfo, asserts off; 2026-10-15.
Flags: `-O2 -g -DNDEBUG -std=c++23` and
`-O2 -march=native -g -DNDEBUG -std=c++23`. Cells are minimum milliseconds
over three runs of 25 samples, plain / native.

| Shape (1e6 operations) | library | by full propagation | by inline offsets |
|---|---|---|---|
| insert, universe 1e6 | 5.2 / 5.2 | 7.9 / 6.0 | 4.9 / 4.7 |
| insert, universe 2^26 | 11.3 / 10.7 | 14.8 / 13.3 | not measured yet |
| insert/erase churn, universe 1e6 | 5.7 / 5.3 | 8.5 / 7.0 | 5.3 / 5.7 |
| next, universe 1e6 | 2.6 / 2.9 | 3.3 / 2.8 | 3.3 / 2.8 |
| next, universe 2^26 | 16.8 / 14.8 | 16.5 / 14.4 | 16.8 / 14.6 |
| prev, universe 1e6 | 2.0 / 1.7 | 2.1 / 2.5 | 2.4 / 1.7 |
| prev, universe 2^26 | 14.3 / 13.5 | 14.2 / 14.1 | 14.9 / 13.3 |

| Shape (1e6 operations, universe 1e6) | std::set |
|---|---|
| insert | 537 / 557 |
| lower_bound | 482 / 484 |

## Pitfalls and notes

Both neighbor operations are inclusive. `npos` is a sentinel, never an input position.
Memory depends on the universe bound, not the number of elements. Compress sparse
coordinates first if all possible coordinates are known. XOR queries and order
statistics require different structures.

## How it works

Level zero packs positions into 64-bit words. Each higher level has one bit per
nonzero word below it; levels are consecutive slices of one vector, with offsets in
a second vector. Rounding uses division and remainder to avoid overflowing `size_t`.
Updates stop once a summary cannot change. Neighbor queries climb until a nonzero
word is found, then descend with count-leading/trailing-zero operations. At most
one ascent and descent through the hierarchy explains the logarithmic bound.

## See also

- [Coordinate compression](coordinate-compression.md)
- [Types and utilities](types-and-utilities.md)
