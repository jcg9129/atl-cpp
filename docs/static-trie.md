# `atl::static_trie`

Build a fixed multiset of integral-symbol words with sorted child runs,
lexicographic word intervals and preorder node IDs. Use it for ordered prefix
queries over a known dictionary; interleaved updates require `trie`.

```cpp
#include <atl/strings/static_trie.hpp>
```

\include static-trie/quick.cpp

## API

```cpp
template<std::size_t Alphabet>
requires(Alphabet <= (uint64(1) << 32))
class static_trie;
```

`node_id` is `uint32`. Child spans contain:

```cpp
struct edge {
    uint32 symbol;
    uint32 target;
};
```

The constructor accepts `trie_word_range<R, Proj>`: an input range of input
ranges whose projected symbols are integral. Query words require
`trie_word<R, Proj>`. Projections are invoked as lvalues; captured lambdas and
`std::reference_wrapper` work. Every operation is constexpr.

```cpp
constexpr static_trie();
template<typename R, typename Proj = std::identity>
requires trie_word_range<R, Proj>
constexpr explicit static_trie(R &&words, Proj proj = {});
```

Each whole-word query has this template header:

```cpp
template<typename R, typename Proj = std::identity>
requires trie_word<R, Proj>
```

| Signature | Result |
|---|---|
| `constexpr std::size_t count(R &&word, Proj proj = {}) const` | multiplicity |
| `constexpr bool contains(R &&word, Proj proj = {}) const` | positive multiplicity |
| `constexpr std::optional<node_id> find(R &&word, Proj proj = {}) const` | prefix node, or nullopt for absent path |
| `constexpr std::size_t count_with_prefix(R &&prefix, Proj proj = {}) const` | occurrences carrying prefix |
| `constexpr node_id root() const` | empty prefix, id zero |
| `template<std::integral Sym> constexpr std::optional<node_id> walk(node_id n, Sym sym) const` | child or nullopt |
| `constexpr std::span<edge const> children(node_id n) const` | ascending-symbol edges |
| `constexpr std::size_t word_count(node_id n) const` | occurrences ending at node |
| `constexpr std::size_t prefix_count(node_id n) const` | occurrences carrying node prefix |
| `constexpr index_interval word_interval(node_id n) const` | half-open positions in lexicographic multiset order |
| `constexpr std::span<uint32 const> word_order() const` | input index at each lexicographic position |
| `constexpr node_id subtree_end(node_id n) const` | one past last node in preorder subtree |
| `constexpr std::size_t size() const` / `constexpr bool empty() const` | input word count / zero words |
| `constexpr std::size_t node_count() const` / `constexpr std::size_t edge_count() const` | nodes including root / edges |

Pure queries have `[[nodiscard]]`, except trivial `size` and `empty`. Equal words
have unspecified relative order. The root is always present, so `find` of the
empty word succeeds even in an empty dictionary. Every subtree is exactly
[n, subtree_end(n)). Copying owns independent arrays; moving transfers them.
A moved-from object may only be assigned or destroyed.

## Complexity

For n words and L symbols, build costs O(n + L + sum of slice log slice over
branching nodes), bounded by O(n + L log(1 + n)). Word lookup takes
O(m log(1 + d)) for length m and maximum encountered degree d; child runs of at
most eight edges use a linear scan. Other queries are O(1). Build uses O(n + L)
heap scratch; persistent storage is five uint32 arrays per node (about 20 bytes),
eight bytes per edge and four per word, plus vector capacity. Build and lookup
use loops and heap stacks, with no recursion growing with depth.

## Preconditions

Asserted under `ATL_ASSERT`: projected symbols are in [0, Alphabet), total
symbols are at most 2^32 - 2, word count fits uint32, and node handles are valid.
Materialization checks each bound ahead of narrowing. Reservations and sentinel
arrays must fit `size_t` and vector limits. Queries outside the alphabet miss.

Documented only: use a consistent projection for build and lookup. Character
values are numeric symbols; cast signed `char` to unsigned char when indexing
arbitrary bytes. A negative or wider-than-64-bit symbol is compared with the
alphabet at its own width and is out of alphabet (it never aliases a smaller
symbol). All standard character widths and integral alphabets work,
including non-contiguous and single-pass word ranges. Proxies require an
integral-valued projection. The structure owns its materialized symbols.
Returned spans survive const queries; assignment/destruction invalidate them,
and moving transfers their storage. Allocation/projection exceptions propagate.

## Measured performance

[The benchmark](../benchmarks/strings/static_trie.cpp) measures full build
throughput and counting the input words. Library construction gathers integer
keys, skips sorting single-symbol slices and emits chains directly. The
benchmark-local `static_trie_by_slice_comparison` in
`benchmarks/strings/baseline/static_trie.hpp` sorts word IDs by indirect symbol
comparisons at each level. Dynamic hashed insertion supplies a third build.

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
2026-10-15. Flags: `-O2 -g -DNDEBUG -std=c++23`, optionally `-march=native`.
Minimum per-run mean over three runs of 25 samples; milliseconds.

| Shape | Operation / contender | -O2 | native |
|---|---|---:|---:|
| U=1e6 1e5x10 | build library | 28.5 | 28.9 |
| U=1e6 1e5x10 | build by slice comparison | 53.5 | 54.2 |
| U=1e6 1e5x10 | build dynamic trie | 41.0 | 40.8 |
| U=1e6 1e5x10 | count all, library | 24.4 | 23.9 |
| sigma=26 1e5x(5-15) | build library | 34.0 | 34.3 |
| sigma=26 1e5x(5-15) | build by slice comparison | 44.7 | 45.4 |
| sigma=26 1e5x(5-15) | build dynamic trie | 33.0 | 33.2 |
| sigma=26 1e5x(5-15) | count all, library | 24.7 | 24.7 |
| prefix500 2e4 | build library | 82.5 | 84.3 |
| prefix500 2e4 | build by slice comparison | 598 | 602 |
| prefix500 2e4 | build dynamic trie | 40.8 | 44.6 |
| prefix500 2e4 | count all, library | 44.9 | 44.8 |
| identical 1e5x10 | build library | 3.2 | 3.1 |
| identical 1e5x10 | build by slice comparison | 26.3 | 26.4 |
| identical 1e5x10 | build dynamic trie | 2.5 | 2.9 |
| identical 1e5x10 | count all, library | 2.9 | 2.9 |

Spreads range from 1.00x to 1.17x. Integer-key partitioning improves these builds
relative to slice comparison. Dynamic insertion is faster on the shared-prefix
and identical shapes; these records do not compare dynamic and static queries.

## Pitfalls and notes

Lexicographic order is numeric projected-symbol order. Duplicate words count
with multiplicity, but their input-index order is unspecified. `word_interval`
indexes `word_order`, while `subtree_end` indexes nodes; these are different
coordinate systems. A fixed dictionary does not imply a smaller node payload
than the dynamic trie: select by traversal needs and measured workload.

## How it works

Materialization makes a single pass over projected symbols. An explicit DFS
stack holds slices of a word-ID permutation. Branching slices sort packed
64-bit `(symbol, word ID)` keys; equal-symbol runs become CSR child spans.
A lone word emits its suffix as a chain. Single-symbol slices measure their
shared continuation word by word and emit its chain without sorting. Reverse
preorder propagates each last child's subtree end. Node slices directly give
word intervals and occurrence counts. Queries scan short child spans and
binary-search longer ones.

## See also

[Trie](trie.md), [Sorted dictionary](sorted-dictionary.md),
[Lexicographic order](lexicographic-order.md), [Aho–Corasick](aho-corasick.md).
