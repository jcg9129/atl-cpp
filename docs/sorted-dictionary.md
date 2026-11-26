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

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape / operation | Contender | -O2 ms | native ms |
|---|---|---|---|
| char 1e6 build | library | 65.4 | 61.8 |
| char 1e6 build | library, queried once | 65.6 | 62.6 |
| char 1e6 build | by parts build, RMQ descent | 62.8 | 56.6 |
| char 1e6 build | by parts build, novector | 62.9 | 56.2 |
| char 1e6 build | by query options | 51.9 | 48.6 |
| char 1e6 build | by parts build, symbol reserve | 58.1 | 51.1 |
| char 1e6 build | by parts build, descent LCP | 193 | 138 |
| char 1e6 build | library, repeat | 65.4 | 62.6 |
| char 1e6 build | by inplace build | 65.4 | 62 |
| uint64 prefix500 2e4 build | library | 57.1 | 59.7 |
| uint64 prefix500 2e4 build | library, queried once | 57.5 | 59.7 |
| uint64 prefix500 2e4 build | by parts build, RMQ descent | 99.5 | 100 |
| uint64 prefix500 2e4 build | by parts build, novector | 99.9 | 101 |
| uint64 prefix500 2e4 build | by query options | 99.7 | 101 |
| uint64 prefix500 2e4 build | by parts build, symbol reserve | 59 | 60.1 |
| uint64 prefix500 2e4 build | by parts build, descent LCP | 102 | 103 |
| uint64 prefix500 2e4 build | library, repeat | 58.5 | 60.6 |
| uint64 prefix500 2e4 build | by inplace build | 58.1 | 60.8 |
| tokens 2e4x50 build | library | 40.9 | 42.3 |
| tokens 2e4x50 build | library, queried once | 40.8 | 41.7 |
| tokens 2e4x50 build | by parts build, RMQ descent | 64.4 | 66.4 |
| tokens 2e4x50 build | by parts build, novector | 64.7 | 66.4 |
| tokens 2e4x50 build | by query options | 65 | 66.2 |
| tokens 2e4x50 build | by parts build, symbol reserve | 41.3 | 41.3 |
| tokens 2e4x50 build | by parts build, descent LCP | 65.1 | 66.3 |
| tokens 2e4x50 build | library, repeat | 40.8 | 42.6 |
| tokens 2e4x50 build | by inplace build | 40.5 | 42.4 |
| char 5e5 hits | library | 353 | 234 |
| char 5e5 hits | by parts build, linear upper scan | 379 | 264 |
| char 5e5 hits | by parts build, descent LCP | 349 | 227 |
| char 5e5 hits | by parts build, linear upper scan and descent LCP | 301 | 207 |
| char 5e5 hits | by inplace build | 337 | 221 |
| char 5e5 hits | by inplace build, linear upper scan | 296 | 202 |
| char 5e5 misses | library | 197 | 135 |
| char 5e5 misses | by parts build, linear upper scan | 305 | 207 |
| char 5e5 misses | by parts build, descent LCP | 201 | 136 |
| char 5e5 misses | by parts build, linear upper scan and descent LCP | 197 | 136 |
| char 5e5 misses | by inplace build | 198 | 135 |
| char 5e5 misses | by inplace build, linear upper scan | 193 | 136 |
| char 5e5 whole words | library | 351 | 216 |
| char 5e5 whole words | by parts build, linear upper scan | 396 | 253 |
| char 5e5 whole words | by parts build, descent LCP | 349 | 214 |
| char 5e5 whole words | by parts build, linear upper scan and descent LCP | 292 | 184 |
| char 5e5 whole words | by inplace build | 336 | 214 |
| char 5e5 whole words | by inplace build, linear upper scan | 282 | 180 |
| uint64 2e4 misses | library | 9.4 | 8.4 |
| uint64 2e4 misses | by parts build, linear upper scan | 9.5 | 8.5 |
| uint64 2e4 misses | by parts build, descent LCP | 9.2 | 8.1 |
| uint64 2e4 misses | by parts build, linear upper scan and descent LCP | 9.4 | 8.5 |
| uint64 2e4 misses | by inplace build | 9.5 | 8.3 |
| uint64 2e4 misses | by inplace build, linear upper scan | 9.5 | 8.5 |
| uint64 2e4 unique hits | library | 14.3 | 14 |
| uint64 2e4 unique hits | by parts build, linear upper scan | 17 | 17.2 |
| uint64 2e4 unique hits | by parts build, descent LCP | 14.4 | 14 |
| uint64 2e4 unique hits | by parts build, linear upper scan and descent LCP | 14 | 14.1 |
| uint64 2e4 unique hits | by inplace build | 14.4 | 13.9 |
| uint64 2e4 unique hits | by inplace build, linear upper scan | 13.9 | 13.6 |
| uint64 2e4 wide | library | 11.9 | 11.5 |
| uint64 2e4 wide | by parts build, linear upper scan | 13.1 | 12.6 |
| uint64 2e4 wide | by parts build, descent LCP | 11.9 | 11.2 |
| uint64 2e4 wide | by parts build, linear upper scan and descent LCP | 12 | 11.3 |
| uint64 2e4 wide | by inplace build | 11.8 | 11.3 |
| uint64 2e4 wide | by inplace build, linear upper scan | 11.8 | 11.3 |
| tokens 2e4 queries | library | 20.1 | 15.8 |
| tokens 2e4 queries | by parts build, linear upper scan | 21 | 16.8 |
| tokens 2e4 queries | by parts build, descent LCP | 19.8 | 15.7 |
| tokens 2e4 queries | by parts build, linear upper scan and descent LCP | 18.9 | 15.5 |
| tokens 2e4 queries | by inplace build | 18.9 | 15.8 |
| tokens 2e4 queries | by inplace build, linear upper scan | 17.9 | 15.5 |


The library loses char construction to by query options (65.4/61.8 versus
51.9/48.6 ms). On queries, inplace linear upper scanning improves char hits
relative to inplace without scanning (296/202 versus 337/221 ms) and whole
words (282/180 versus 336/214 ms); the library loses both comparisons
(353/234 ms hits, 351/216 ms whole words). Misses favor linear scanning
plain (193 versus 198 ms inplace, library 197 ms) but lose native
(136 versus 135 ms for both). These minimum-time comparisons do not
establish small stable speedups; some char-query rows vary between runs.


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
