# `atl::trie`

Maintain a multiset of words over a sparse integral alphabet, with online
insertion, erasure and prefix counts. Use a static trie when ordered traversal
of a fixed dictionary matters; this trie's child order is unspecified.

```cpp
#include <atl/strings/trie.hpp>
```

\include trie/quick.cpp

## API

```cpp
template<std::size_t Alphabet>
requires(Alphabet <= (uint64(1) << 32))
class trie;
```

`node_id` is `uint32`. The concepts `trie_word<R, Proj = std::identity>` and
`trie_word_range<R, Proj = std::identity>` require an input range of integral
projected symbols and an input range of such words respectively. The projection
is invoked as an lvalue on the range reference. `child_visitor<Visit>` requires
`std::invocable<Visit &, uint32, uint32>`.

Each word operation has this template header:

```cpp
template<typename R, typename Proj = std::identity>
requires trie_word<R, Proj>
```

`erase` reads the word once, so it also accepts an input-only range; the
visited path is kept in O(word length) scratch. The two-descent baseline in
`benchmarks/strings/baseline/trie.hpp` still requires a forward range.

| Signature | Result / effect |
|---|---|
| `trie()` | empty dictionary, one root |
| `void reserve(std::size_t total_symbols)` | reserve edges plus root |
| `void clear()` | discard nodes and words, keep capacity and hash seed |
| `node_id insert(R &&word, Proj proj = {})` | add an occurrence, return terminal handle |
| `bool erase(R &&word, Proj proj = {})` | remove one occurrence, return whether present |
| `std::size_t count(R &&word, Proj proj = {}) const` | word multiplicity |
| `bool contains(R &&word, Proj proj = {}) const` | positive multiplicity |
| `std::optional<node_id> find(R &&word, Proj proj = {}) const` | allocated path, including paths whose counts are zero |
| `std::size_t count_with_prefix(R &&prefix, Proj proj = {}) const` | occurrences carrying prefix |
| `node_id root() const` | empty prefix, id zero |
| `template<std::integral Sym> std::optional<node_id> walk(node_id n, Sym sym) const` | child or nullopt |
| `template<child_visitor Visit> void for_each_child(node_id n, Visit visit) const` | invoke visitor on each `(symbol, child)` in unspecified order |
| `std::size_t word_count(node_id n) const` | occurrences ending at node |
| `std::size_t prefix_count(node_id n) const` | occurrences passing through node |
| `std::size_t size() const` / `bool empty() const` | total occurrences / zero occurrences |
| `std::size_t node_count() const` | allocated nodes, including root |

Pure queries carry `[[nodiscard]]`; `size` and `empty` are trivial-accessor
exceptions. All operations are runtime-only because construction captures the
process hash seed. Copying owns an independent pool; moving transfers buffers.
A moved-from trie may only be assigned or destroyed.

## Complexity

A word operation of length m takes expected O(m log(1 + d)) time for maximum
encountered degree d. Worst-case sibling height is d, so hashing is no
adversarial worst-case guarantee. Erase uses O(m) reusable heap scratch and one
descent. Child visitation takes O(d) time and heap scratch. Count/handle accessors
are O(1); clear destroys O(nodes) storage. Reserve may move O(nodes) entries.
Storage is O(nodes), with six uint32 fields (24 bytes) per node plus spare capacity
and erase scratch. Nodes remain allocated when counts reach zero.

## Preconditions

Asserted with `ATL_ASSERT`: inserted projected symbols lie in [0, Alphabet),
stored occurrence count fits uint32, node creation has fewer than 2^32 - 1
existing nodes, and node handles are valid. Reservation must fit the stored node
bound and `size_t`. Query and erase symbols outside the alphabet miss.

Documented only: projections must produce the intended stable symbol mapping;
callbacks must not clear or mutate the trie during traversal. Construct from
`main` or later, when process seed initialization is complete. Allocation and
projection exceptions propagate; mutation has no transactional recovery guarantee.
Handles survive insert, erase and reserve; clear and assignment invalidate them.
`find` returns root for the empty word even when the dictionary is empty.

## Measured performance

[The benchmark](../benchmarks/strings/trie.cpp) measures build/query streams for
hashed sibling trees, a global `(parent,symbol)` edge hash map and a raw-symbol
sibling BST. Erase compares path scratch with two descents. Experimental storage
contenders are in `experimental/atl/experimental/strings/trie_variants.hpp`;
tests certify both against a map-backed multiset oracle. The raw-symbol BST is
excluded from sorted timing because sorted symbols form a linear sibling chain.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape / operation | Contender | -O2 | native |
|---|---|---:|---:|
| random build | library | 26.2 | 27.4 |
| random 1e5 queries | library | 28.7 | 25.6 |
| random build | by global hash | 52.6 | 55.7 |
| random 1e5 queries | by global hash | 55.9 | 61.2 |
| random build | by symbol BST | 20.4 | 22 |
| random 1e5 queries | by symbol BST | 16.3 | 16.2 |
| sorted build | library | 28.8 | 30 |
| sorted 1e5 queries | library | 34.4 | 37.3 |
| sorted build | by global hash | 58.2 | 59.5 |
| sorted 1e5 queries | by global hash | 61.2 | 61.8 |
| binary 2e5x30 erase | library | 200 | 192 |
| binary 2e5x30 erase | by two descents | 217 | 200 |
| U=1e6 1e5x10 insert+erase | library | 56.2 | 52 |
| U=1e6 1e5x10 insert+erase | by two descents | 68 | 62.9 |

Random and sorted storage races use U=1e6, 1e5 words of length 10. Spreads range from 1.04x to 1.81x. The library loses random queries to
the raw-symbol BST (28.7/25.6 versus 16.3/16.2 ms); those library rows
vary between runs (1.25x spread in both modes), so minima alone do not
establish a stable ratio. Hashing avoids the BST's sorted-input chain.
The global hash map has larger query minima on both random and sorted
inputs; those rows also vary between runs.
The benchmark-local `trie_by_two_descents` lives in
`benchmarks/strings/baseline/trie.hpp`; its projection must remain stable across
both descents. It retains nodes just as the library does.

## Pitfalls and notes

`std::string_view` works directly for nonnegative character values. Plain `char`
may be signed: use an unsigned-char projection for arbitrary bytes. Signed char,
unsigned char, char8_t, wchar_t, char16_t, char32_t and integer alphabets retain
their numeric values; negative values are invalid for insert. Word ranges may be
non-contiguous and single-pass, erase included. A symbol wider than 64 bits
(a 128-bit integer) or negative is compared with the alphabet at its own width
and is out of alphabet: it never aliases a smaller symbol. Words are iterated
through `std::ranges::begin` and `std::ranges::end`, so a range whose `end` is
found only by argument-dependent lookup is accepted.
Proxy ranges work when their projection returns an integral value. Empty words
are stored at the root. No operation recurses per character or node.

## How it works

A shared `include/atl/ds/detail/hashed_child_pool.hpp` arena stores sibling BST
links ordered by a seeded, bijective 64-bit symbol hash. Distinct symbols have
distinct order keys, and storage scales with actual edges. Each trie node fuses
its incoming edge, child-tree root and two counts. Insert updates counts along
one path. Erase records that path on the heap, checks the terminal count and
then decrements it. Children use an explicit traversal stack. The hash seed is
captured per pool; changing the process seed does not reorder an existing trie.

## See also

[Static trie](static-trie.md), [Aho–Corasick](aho-corasick.md),
[Sorted dictionary](sorted-dictionary.md), [Hash utilities](types-and-utilities.md).
