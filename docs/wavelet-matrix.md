# `atl::wavelet_matrix`

Static range order statistics over unsigned integers: k-th smallest and
counts below a value or within a value interval. Each query walks the bit
levels without pointers or allocation. Use another structure for updates.

```cpp
#include <atl/ds/wavelet_matrix.hpp>
```

\include wavelet-matrix/basic.cpp

## API

`template<unsigned_integral T> class wavelet_matrix` accepts the library's
unsigned integers, including native and software atl::uint128. The aliases
are `value_type = T` and `size_type = std::size_t`; all operations are constexpr.

| signature | meaning |
|---|---|
| `constexpr wavelet_matrix()` | empty sequence |
| `constexpr explicit wavelet_matrix(R &&values)` | sized range with forwarded references convertible to T; CTAD uses range_value_t |
| `constexpr std::size_t size() const noexcept` | number of elements |
| `constexpr bool empty() const noexcept` | whether size is zero |
| `constexpr T at(std::size_t i) const` | reconstruct the value at i |
| `constexpr T kth(std::size_t l, std::size_t r, std::size_t k) const` | zero-based k-th smallest in [l,r) |
| `constexpr std::size_t count_less(std::size_t l, std::size_t r, T upper) const` | count values < upper in [l,r) |
| `constexpr std::size_t count(std::size_t l, std::size_t r, T lo, T hi) const` | count values in [lo,hi) among positions [l,r) |

## Complexity

Let B = max(1, bit_width(maximum)). Build takes O(n B) time and
O(n B / 64 + B) machine words, plus O(n) temporary values. Accessors take
O(1); all other queries take O(B). Each level uses roughly 1.5 stored bits
per sequence bit: a 64-bit word plus a 32-bit cumulative rank per 64 bits,
with padding. The representation remains fully 64-bit on 32-bit targets.

## Preconditions

Asserted under ATL_ASSERT: size <= 2^32-1 before narrowing; integral source
values fit T; i < size(); l <= r <= size() for counts; l < r <= size() and
k < r-l for kth; lo <= hi. Non-integral conversions must be valid and
representable (documented only). Range constructors copy source views,
even as rvalues; owning ranges and yielded prvalues may move.

Bounds are half-open: count_less and count cannot include a value equal to
numeric_limits<T>::max(). To count the closed interval [lo,max], use
`(r-l) - matrix.count_less(l, r, lo)`.

## Measured performance

`benchmarks/ds/wavelet_matrix.cpp` races raw values against compression to
rank space and races two-pass versus one-pass stable partition, separate
versus interleaved rank cells, and branching versus branchless queries.
The loops measure construction separately from independent query throughput.
The persistent-segment-tree comparison returns with that structure's page.

Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-15.
Minimum of three runs, milliseconds, plain -O2 / -O2 -march=native;
both also use -g -DNDEBUG -std=c++23. Compression work is outside the query timing.

| shape | phase | library | prefix versions | by rank compression |
|---|---|---:|---:|---:|
| balanced n=q=2e5 | build | 10.0 / 8.4 | 148 / 150 | — |
| balanced n=q=2e5 | kth | 117 / 99.9 | 216 / 236 | 66.1 / 54.8 |
| balanced n=q=2e5 | count_less | 80.4 / 57.8 | 427 / 440 | 71.9 / 53.8 |
| query-heavy n=5e4 q=1e6 | build | 2.4 / 2.0 | 18.0 / 17.8 | — |
| query-heavy n=5e4 q=1e6 | kth | 515 / 419 | 443 / 471 | 260 / 209 |
| query-heavy n=5e4 q=1e6 | count_less | 367 / 213 | 796 / 817 | 288 / 192 |
| build-heavy n=1e6 q=1e4 | build | 52.7 / 44.4 | 1245 / 1247 | — |
| build-heavy n=1e6 q=1e4 | kth | 10.3 / 9.3 | 17.8 / 19.0 | 6.7 / 5.7 |
| build-heavy n=1e6 q=1e4 | count_less | 8.3 / 6.8 | 32.5 / 32.1 | 7.1 / 5.6 |
| alphabet<256 n=q=2e5 | kth | 27.5 / 22.3 | 61.6 / 62.6 | — |
| narrow r-l<=64 n=q=2e5 | kth | 118 / 100 | 199 / 202 | 65.5 / 54.2 |

The following race table isolates construction and query methods on the
balanced 30-bit shape. A build with branchless queries uses the same one-pass
partition as the library; only its query code differs. Interleaved cells
place a word and its cumulative rank together in a 16-byte cell.

| phase, n=q=2e5 | library | by two pass partition | by one pass partition / branchless queries | by interleaved cells | by branchless queries and cells |
|---|---:|---:|---:|---:|---:|
| build | 10.0 / 8.4 | 52.3 / 51.5 | 9.8 / 8.7 | — | 9.6 / 8.2 |
| kth | 118 / 99.7 | 113 / 94.6 | 113 / 93.0 | 132 / 117 | 134 / 117 |

One-pass construction wins on this shape; interleaving increases cell size
and loses on kth. The benchmark retains the larger, rank-space and k=0
shapes to test whether that conclusion changes with the workload.

## Pitfalls and notes

Compress first when the alphabet is sparse: levels scale with maximum
value, not distinct count. Translate kth ranks back through the sorted
universe and translate count bounds with lower_bound. No mutation or
version interface is provided. Empty count ranges are allowed.

## How it works

Each bit level stably partitions the current sequence into zero and one
values. A two-cursor pass writes both destinations and advances only the
matching cursor, assembling each bit word in a register. The ones buffer
is then copied after the zeros. A rank directory records ones strictly
before each word; rank combines that prefix with popcount of a masked word.
Unused tail bits and a padding word are zero. Queries map [l,r) into the
next level using zero ranks or the level's zero count plus one ranks.
Choosing a zero or one branch either locates kth or accumulates count_less.

## See also

- [Coordinate compression](coordinate-compression.md)
