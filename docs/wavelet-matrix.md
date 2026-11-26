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
Prefix count versions of [persistent_segment_tree](persistent-segment-tree.md)
provide the static order-statistic comparison: compressed values, a lockstep
kth descent and two prefix-fold differences for count_less.

The benchmark also prints an estimate of the payload storage in decimal MB: for
the wavelet matrix, the packed bit words and rank prefixes of every level; for
the persistent segment tree, its arena nodes plus the version handles and the
sorted value universe the comparison keeps. Allocator overhead is not counted.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | phase | library | by persistent segment tree | by rank compression |
|---|---|---:|---:|---:|
| balanced n=q=2e5 | build | 7.5 / 6.5 | 114 / 114 | — |
| balanced n=q=2e5 | kth | 122 / 92.2 | 200 / 181 | 63.8 / 43.6 |
| balanced n=q=2e5 | count_less | 88 / 58.6 | 287 / 280 | 70.8 / 48.6 |
| query-heavy n=5e4 q=1e6 | build | 1.9 / 1.6 | 13.5 / 13.4 | — |
| query-heavy n=5e4 q=1e6 | kth | 406 / 300 | 280 / 291 | 218 / 157 |
| query-heavy n=5e4 q=1e6 | count_less | 301 / 172 | 413 / 417 | 235 / 158 |
| build-heavy n=1e6 q=1e4 | build | 37.7 / 32.3 | 960 / 975 | — |
| build-heavy n=1e6 q=1e4 | kth | 8.2 / 6.7 | 14.5 / 14.2 | 5.2 / 4.2 |
| build-heavy n=1e6 q=1e4 | count_less | 6.7 / 4.9 | 15.4 / 16.4 | 5.6 / 4.3 |
| alphabet<256 n=q=2e5 | build | 2.2 / 1.9 | 20.7 / 20 | — |
| alphabet<256 n=q=2e5 | kth | 23 / 16.4 | 37.7 / 36.6 | 23.8 / 16.1 |
| alphabet<256 n=q=2e5 | count_less | 16.1 / 7.7 | 47.6 / 42.9 | 21 / 12.7 |
| narrow r-l<=64 n=q=2e5 | build | 7.5 / 6.5 | 113 / 116 | — |
| narrow r-l<=64 n=q=2e5 | kth | 122 / 87.6 | 175 / 170 | 62 / 42.6 |
| narrow r-l<=64 n=q=2e5 | count_less | 87.8 / 56.4 | 235 / 222 | 71.3 / 50.8 |

The following race table isolates construction and query methods on the
balanced 30-bit shape. A build with branchless queries uses the same one-pass
partition as the library; only its query code differs. Interleaved cells
place a word and its cumulative rank together in a 16-byte cell.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| phase, n=q=2e5 | library | by two pass partition | by one pass partition / branchless queries | by interleaved cells | by branchless queries and cells |
|---|---:|---:|---:|---:|---:|
| build | 7.5 / 6.6 | 41.9 / 42.4 | 7.6 / 6.5 | — | 7.6 / 6.5 |
| kth | 120 / 88.8 | 122 / 90.5 | 117 / 84.3 | 128 / 93.3 | 127 / 92.7 |

One-pass construction wins on this shape; interleaving increases cell size
and loses on kth. The benchmark also measures larger, rank-space and k=0
shapes to test whether that conclusion changes with the workload.

The library loses query-heavy kth to the persistent segment tree
(406/300 versus 280/291 ms), and loses balanced kth to rank compression
(122/92.2 versus 63.8/43.6 ms). These query rows vary between runs
(spreads up to 2.90x), so their minima alone do not establish stable
rankings or an automatic compression threshold.

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
