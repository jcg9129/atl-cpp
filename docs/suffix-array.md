# `atl::suffix_array`

Build the sorted suffix positions, their inverse ranks and adjacent LCP lengths.
Use this flat static index for lexicographic queries and RMQ geometry.

```cpp
#include <atl/strings/suffix_array.hpp>
```

\include suffix-array/quick.cpp

## API

`suffix_array` owns positions, inverse ranks and adjacent LCP lengths; it retains no text.
`R` satisfies `sortable_text<R, Proj>` (sized random-access range, invocable
projection, totally ordered projected symbols). Construction accepts non-contiguous
ranges, all character types, integer alphabets and ordered tokens.

```cpp
constexpr suffix_array() = default;
template<typename R, typename Proj = std::identity>
requires sortable_text<R, Proj>
constexpr explicit suffix_array(R &&text, Proj proj = {});
[[nodiscard]] constexpr std::span<uint32 const> positions() const;
[[nodiscard]] constexpr std::span<uint32 const> ranks() const;
[[nodiscard]] constexpr std::span<uint32 const> lcp_lengths() const;
[[nodiscard]] constexpr uint32 position(uint32 r) const;
[[nodiscard]] constexpr uint32 rank(uint32 p) const;
constexpr std::size_t size() const;
constexpr bool empty() const;
```

`positions()[r]` and `position(r)` give the rank-r suffix start; `ranks()[p]`
and `rank(p)` give the inverse. `lcp_lengths()[r]` compares ranks r and r+1.
The LCP array has max(n-1, 0) entries; the empty suffix is excluded.

## Complexity

Construction takes O(n) time for compact integral alphabets and O(n log n)
otherwise, including compression; auxiliary storage is O(n). Stored arrays use
about 12n bytes. Accessors are O(1). SA-IS recurses only on reduced problems
at most half as large, so stack depth is O(log n); Kasai is iterative and linear,
including a unary million-symbol input.

## Preconditions

Asserted under `ATL_ASSERT`: text length n < 2^32 - 1 and accessor indices
below n. Allocations must fit `size_t` and the standard containers' limits.
Documented only: projection results remain stable and their order is consistent
with equality. Character order follows the projected type's signedness; cast to
`unsigned char` for byte order. Returned spans are invalidated by assignment or
destruction of their owner. Allocation failures propagate as standard exceptions.

## Measured performance

[The benchmark](../benchmarks/strings/suffix_array.cpp) measures full construction,
prefix-doubling sort alone, and isolated Kasai/phi LCP passes. By index compression
sorts indices even for sparse integral symbols and copies the reduced sequence and
final suffix array. By prefix doubling performs stable counting-sort rounds;
by phi builds permuted LCPs in text order. The experimental certifying oracle is
`suffix_sort_by_prefix_doubling` in
`experimental/atl/experimental/strings/suffix_array_variants.hpp`.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape / operation | Contender | -O2 ms | native ms |
|---|---|---|---|
| 1e6 random2 | library build | 63.8 | 63.1 |
| 1e6 random2 | by index compression, build | 63.4 | 63.2 |
| 1e6 random2 | by prefix doubling, sort only | 36.4 | 40.6 |
| 1e6 random2 | by Kasai, LCP only | 11.2 | 12.7 |
| 1e6 random2 | by phi, LCP only | 10.7 | 11.6 |
| 1e6 random26 | library build | 58.7 | 61.1 |
| 1e6 random26 | by index compression, build | 58.8 | 61.6 |
| 1e6 random26 | by prefix doubling, sort only | 36.2 | 39.7 |
| 1e6 random26 | by Kasai, LCP only | 10.1 | 11.9 |
| 1e6 random26 | by phi, LCP only | 9.4 | 10.3 |
| 1e6 unary | library build | 9.6 | 11.7 |
| 1e6 unary | by index compression, build | 9.5 | 11.9 |
| 1e6 unary | by prefix doubling, sort only | 53.6 | 59.5 |
| 1e6 unary | by Kasai, LCP only | 0.943 | 1.4 |
| 1e6 unary | by phi, LCP only | 1.5 | 2.1 |
| 1e6 alternating | library build | 15.5 | 19.2 |
| 1e6 alternating | by index compression, build | 16 | 19.3 |
| 1e6 alternating | by prefix doubling, sort only | 55.7 | 62.1 |
| 1e6 alternating | by Kasai, LCP only | 0.94 | 1.6 |
| 1e6 alternating | by phi, LCP only | 1.6 | 2.1 |
| 1e6 Fibonacci | library build | 23.9 | 28.9 |
| 1e6 Fibonacci | by index compression, build | 24.8 | 28.9 |
| 1e6 Fibonacci | by prefix doubling, sort only | 117 | 130 |
| 1e6 Fibonacci | by Kasai, LCP only | 2.9 | 2.7 |
| 1e6 Fibonacci | by phi, LCP only | 3 | 3.8 |
| 5e6 random26 | library build | 604 | 606 |
| 5e6 random26 | by index compression, build | 589 | 607 |
| 5e6 random26 | by Kasai, LCP only | 137 | 144 |
| 5e6 random26 | by phi, LCP only | 55.6 | 73.1 |
| 5e6 Fibonacci | library build | 257 | 294 |
| 5e6 Fibonacci | by index compression, build | 259 | 287 |
| 5e6 Fibonacci | by Kasai, LCP only | 34.4 | 41.9 |
| 5e6 Fibonacci | by phi, LCP only | 24.8 | 37.8 |
| 1e5 int64 range 32n | library build | 10.3 | 10.2 |
| 1e5 int64 range 32n | by index compression, build | 10.9 | 10.9 |
| 1e5 int64 range 2^40 | library build | 10.3 | 10 |
| 1e5 int64 range 2^40 | by index compression, build | 11.1 | 11 |
| 1e6 int64 range 32n | library build | 124 | 125 |
| 1e6 int64 range 32n | by index compression, build | 137 | 143 |
| 1e6 int64 range 2^40 | library build | 120 | 121 |
| 1e6 int64 range 2^40 | by index compression, build | 134 | 141 |
| 1e5 string tokens | library build | 18.3 | 19.5 |
| 1e5 string tokens | by index compression, build | 18.4 | 19.4 |


Phi beats Kasai on random texts but loses unary and alternating texts at
1e6. For random26 at 1e6, the library Kasai pass loses (10.1/11.9 versus
9.4/10.3 ms); unary favors Kasai (0.943/1.4 versus 1.5/2.1 ms).
Fibonacci favors Kasai at 1e6 (2.9/2.7 versus 3/3.8 ms), but Phi at 5e6
(24.8/37.8 versus 34.4/41.9 ms). Kasai remains the library LCP pass,
with no uniform performance win. The native alternating Phi row has a
1.01x spread.

The plain 5e6 Fibonacci Kasai row varies between runs (1.29x spread);
its comparison with Phi alone does not establish a stable ranking.

## Pitfalls and notes

For suffixes at distinct positions p and q, their LCP is the minimum adjacent
LCP between their ranks. For p == q it is n-p. The number of distinct nonempty
substrings is n(n+1)/2 minus the LCP sum; widen to `uint64` prior to multiplying.
Pattern searches need the caller's original text.

## How it works

Symbols are mapped to positive integers with zero reserved for the unique
sentinel. Integral ranges below 8n use offset mapping when the alphabet bound
also fits `uint32`; other integral symbols sort (symbol, index) pairs and other
types sort an index permutation. SA-IS classifies suffixes, sorts LMS substrings
by induced bucket passes, names them and solves the smaller sequence if names
repeat. A second induced pass sorts all suffixes. The sentinel entry is erased
in place. Inverse ranks then drive Kasai's scan: advancing a text position loses
at most one known prefix symbol. Reference: Nong, Zhang and Chan (2009),
“Linear Suffix Array Construction by Almost Pure Induced-Sorting”.

## See also

[Suffix automaton](suffix-automaton.md), [Suffix tree](suffix-tree.md), [RMQ](rmq.md).
