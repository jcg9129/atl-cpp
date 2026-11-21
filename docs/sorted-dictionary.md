# `atl::sorted_dictionary`

Find the interval of sorted words beginning with a query in O(m + log n)
symbol work, including an insertion point on a miss. Use it when static prefix
search and predictable comparison bounds matter.

```cpp
#include <atl/strings/sorted_dictionary.hpp>
```

\include sorted-dictionary/quick.cpp

## API

```cpp
template<std::totally_ordered Symbol> class sorted_dictionary;
```

```cpp
template<typename R, typename Proj = std::identity>
requires sortable_word_range<R, Proj> && std::same_as<word_symbol_t<R, Proj>, Symbol>
constexpr explicit sorted_dictionary(R &&words, Proj proj = {});
sorted_dictionary(sorted_dictionary const &) = delete;
sorted_dictionary &operator=(sorted_dictionary const &) = delete;
constexpr sorted_dictionary(sorted_dictionary &&) = default;
constexpr sorted_dictionary &operator=(sorted_dictionary &&) = default;
template<typename Q, typename QProj = std::identity>
requires sortable_text<Q, QProj> && std::totally_ordered_with<text_symbol_t<Q, QProj>, Symbol>
[[nodiscard]] constexpr index_interval prefix_interval(Q &&x, QProj proj = {}) const;
constexpr std::size_t size() const;
constexpr bool empty() const;
```

The deduction guide is:

```cpp
template<typename R, typename Proj = std::identity>
    requires sortable_word_range<R, Proj>
sorted_dictionary(R &&, Proj = {}) -> sorted_dictionary<word_symbol_t<R, Proj>>;
```

Construction copies projected symbols into an owned buffer (symbols that are not
default constructible or not move assignable are emplaced one by one); identity projection
moves nontrivial symbols from an rvalue owning word collection. Each word is a
sized random-access range; the outer collection may be input-only. Query projection
results must be totally ordered with `Symbol`; when both are integral types of
different signedness they are compared by mathematical value (a negative query
symbol sorts below every unsigned stored symbol), not through the built-in
conversion. `prefix_interval` returns word
positions [begin, end); an empty query matches every word. A miss returns its
lexicographic insertion point twice. `size` counts words and `empty` tests zero.
Copy operations are deleted; moving transfers buffers and their RMQ view. A moved-from
object may only be assigned or destroyed.

## Complexity

Construction takes O(L+n) symbol work and storage for L total symbols and
n words. Querying m symbols takes O(m + log n) symbol work and O(1) scratch;
projection and symbol comparison costs multiply that work. Size and empty are O(1).
Build and search use loops or heap stacks. No recursion grows with word length.

## Preconditions

Asserted under `ATL_ASSERT`: words are sorted by projected-symbol order;
word count, total symbols and query length are below 2^32. Checked sums/reserve
sizes must fit `size_t` and vector limits. Documented only: projections remain
stable, and symbol equality agrees with the total order. Signed byte order follows
the projected type; sort and query with the same projection. Materialized symbols that themselves borrow storage (such as string-view tokens
or proxy symbols) require that storage to stay valid and unchanged for the
dictionary's lifetime; ordinary character values are owned. Internal RMQ views
move with their buffers. Allocation
failure propagates as a standard exception.

## Measured performance

[The benchmark](../benchmarks/strings/sorted_dictionary.cpp) measures full
build throughput and query-stream throughput. By parts build returns materialized
arrays, optionally changing vectorization, reserve, descent-LCP arrays or linear
upper scans. By inplace build isolates the upper scan with member-built arrays;
by query options switches individual search shortcuts. These contenders live in
`experimental/atl/experimental/strings/sorted_dictionary_variants.hpp` and the
benchmark; the parts-build RMQ variant is the certifying experimental oracle.

Intel Xeon Platinum 8168 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
2026-11-03. Flags: `-O2 -g -DNDEBUG -std=c++23`, optionally `-march=native`.
Minimum per-run mean milliseconds over three runs of 25 samples.

| Shape / operation | Contender | -O2 ms | native ms |
|---|---|---|---|
| char 1e6 build | library | 97.7 | 97.1 |
| char 1e6 build | library, queried once | 98.6 | 98.4 |
| char 1e6 build | by parts build, RMQ descent | 82.2 | 90.6 |
| char 1e6 build | by parts build, novector | 82.2 | 90.9 |
| char 1e6 build | by query options | 77.2 | 80.4 |
| char 1e6 build | by parts build, symbol reserve | 79.0 | 79.5 |
| char 1e6 build | by parts build, descent LCP | 201 | 204 |
| char 1e6 build | library, repeat | 99.2 | 96.7 |
| char 1e6 build | by inplace build, trivial symbol reserve | 99.8 | 95.8 |
| char 1e6 build | by inplace build, all symbol reserve | 98.9 | 95.9 |
| uint64 prefix500 2e4 build | library | 74.2 | 74.8 |
| uint64 prefix500 2e4 build | library, queried once | 74.5 | 75.6 |
| uint64 prefix500 2e4 build | by parts build, RMQ descent | 134 | 136 |
| uint64 prefix500 2e4 build | by parts build, novector | 136 | 137 |
| uint64 prefix500 2e4 build | by query options | 136 | 138 |
| uint64 prefix500 2e4 build | by parts build, symbol reserve | 76.3 | 77.0 |
| uint64 prefix500 2e4 build | by parts build, descent LCP | 139 | 140 |
| uint64 prefix500 2e4 build | library, repeat | 76.6 | 77.3 |
| uint64 prefix500 2e4 build | by inplace build, trivial symbol reserve | 77.1 | 78.0 |
| uint64 prefix500 2e4 build | by inplace build, all symbol reserve | 77.3 | 78.6 |
| tokens 2e4x50 build | library | 97.4 | 99.3 |
| tokens 2e4x50 build | library, queried once | 98.1 | 99.2 |
| tokens 2e4x50 build | by parts build, RMQ descent | 97.3 | 97.5 |
| tokens 2e4x50 build | by parts build, novector | 97.3 | 97.4 |
| tokens 2e4x50 build | by query options | 97.3 | 97.4 |
| tokens 2e4x50 build | by parts build, symbol reserve | 61.8 | 62.6 |
| tokens 2e4x50 build | by parts build, descent LCP | 97.9 | 96.4 |
| tokens 2e4x50 build | library, repeat | 97.5 | 97.6 |
| tokens 2e4x50 build | by inplace build, trivial symbol reserve | 97.4 | 97.2 |
| tokens 2e4x50 build | by inplace build, all symbol reserve | 61.7 | 60.9 |
| char 5e5 hits | library | 394 | 411 |
| char 5e5 hits | by parts build, linear upper scan | 513 | 532 |
| char 5e5 hits | by parts build, descent LCP | 395 | 415 |
| char 5e5 hits | by parts build, linear upper scan and descent LCP | 357 | 356 |
| char 5e5 hits | by inplace build | 384 | 409 |
| char 5e5 hits | by inplace build, linear upper scan | 347 | 367 |
| char 5e5 misses | library | 227 | 247 |
| char 5e5 misses | by parts build, linear upper scan | 409 | 425 |
| char 5e5 misses | by parts build, descent LCP | 230 | 233 |
| char 5e5 misses | by parts build, linear upper scan and descent LCP | 235 | 225 |
| char 5e5 misses | by inplace build | 219 | 232 |
| char 5e5 misses | by inplace build, linear upper scan | 226 | 240 |
| char 5e5 whole words | library | 348 | 357 |
| char 5e5 whole words | by parts build, linear upper scan | 495 | 527 |
| char 5e5 whole words | by parts build, descent LCP | 346 | 379 |
| char 5e5 whole words | by parts build, linear upper scan and descent LCP | 316 | 337 |
| char 5e5 whole words | by inplace build | 340 | 371 |
| char 5e5 whole words | by inplace build, linear upper scan | 307 | 303 |
| uint64 2e4 misses | library | 10.3 | 10.3 |
| uint64 2e4 misses | by parts build, linear upper scan | 10.3 | 10.4 |
| uint64 2e4 misses | by parts build, descent LCP | 10.3 | 10.3 |
| uint64 2e4 misses | by parts build, linear upper scan and descent LCP | 10.3 | 10.4 |
| uint64 2e4 misses | by inplace build | 10.1 | 10.5 |
| uint64 2e4 misses | by inplace build, linear upper scan | 10.1 | 10.3 |
| uint64 2e4 unique hits | library | 26.2 | 26.7 |
| uint64 2e4 unique hits | by parts build, linear upper scan | 26.1 | 27.3 |
| uint64 2e4 unique hits | by parts build, descent LCP | 26.1 | 26.2 |
| uint64 2e4 unique hits | by parts build, linear upper scan and descent LCP | 26.0 | 26.1 |
| uint64 2e4 unique hits | by inplace build | 26.4 | 26.6 |
| uint64 2e4 unique hits | by inplace build, linear upper scan | 26.2 | 26.3 |
| uint64 2e4 wide | library | 15.0 | 15.5 |
| uint64 2e4 wide | by parts build, linear upper scan | 15.8 | 16.6 |
| uint64 2e4 wide | by parts build, descent LCP | 15.0 | 15.8 |
| uint64 2e4 wide | by parts build, linear upper scan and descent LCP | 15.0 | 15.4 |
| uint64 2e4 wide | by inplace build | 14.6 | 15.6 |
| uint64 2e4 wide | by inplace build, linear upper scan | 14.8 | 15.5 |
| tokens 2e4 queries | library | 22.3 | 25.1 |
| tokens 2e4 queries | by parts build, linear upper scan | 22.2 | 24.3 |
| tokens 2e4 queries | by parts build, descent LCP | 21.9 | 22.1 |
| tokens 2e4 queries | by parts build, linear upper scan and descent LCP | 22.1 | 22.3 |
| tokens 2e4 queries | by inplace build | 21.4 | 22.4 |
| tokens 2e4 queries | by inplace build, linear upper scan | 22.1 | 23.4 |


Other query-option micro-variant rows are not measured yet. The supplied records
include both trivial-symbol and all-symbol reserve rows; the library reserves
all symbols on forward ranges. Token build times in that comparison are
97.4 versus 61.7 ms at -O2 and 97.2 versus 60.9 ms native. Linear upper scans
have no consistent win: the inplace comparison's hit and whole-word rows improve,
but miss times rise from 219 to 226 ms and 232 to 240 ms, within substantial noise.
The char query spreads reach 1.22x; these figures do not establish small speedups.

## Pitfalls and notes

Sorted input is essential; use `lexicographic_order` to reorder unsorted
words. A trie supplies word/prefix counts incrementally; this structure gives
static intervals and miss insertion points. Nontrivial token comparisons can
cost more than one elementary symbol operation.

## How it works

The constructor materializes directly into members and computes adjacent
LCPs. Their total length is O(L), since every adjacent LCP is bounded by its
shorter word. A `direct_rmq` answers pairwise word LCPs. Two arrays store each
fixed binary descent midpoint's LCP with its boundaries, costing 8n bytes for
n >= 3. Manber–Myers boundary search resumes symbol scans at a known matched
length, so the query is not rescanned per probe. The upper boundary uses only
LCP minima: a gallop capped at 16 positions, then bisection, bounded by the first
word known to exceed the prefix interval. Misses skip that search; a prefix of
the last word ends at n. Reference: Manber and Myers (1993), “Suffix arrays:
A new method for on-line string searches”.

## See also

[Lexicographic order](lexicographic-order.md), [Static trie](static-trie.md), [RMQ](rmq.md).
