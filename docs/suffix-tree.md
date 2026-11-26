# `atl::suffix_tree`

Build the compressed trie of every suffix of a word collection online. Use it
when explicit substring hierarchy, edge labels and suffix positions matter.

```cpp
#include <atl/strings/suffix_tree.hpp>
```

\include suffix-tree/quick.cpp

## API

`template<std::size_t Alphabet> requires(Alphabet <= (uint64(1) << 32))`
selects the exclusive integral alphabet bound. `node_id` is `uint32`.
`insert` requires `trie_word<R, Proj>`; the child callback requires
`std::invocable<Visit &, node_id>`.

```cpp
struct suffix_ref { uint32 word, offset; };
struct edge_ref { uint32 word, offset, length; };
```

```cpp
using node_id = uint32;
suffix_tree();
template<std::integral Sym>
uint32 extend(Sym sym);
uint32 finish_word();
template<typename R, typename Proj = std::identity>
requires trie_word<R, Proj>
uint32 insert(R &&word, Proj proj = {});
[[nodiscard]] node_id root() const;
[[nodiscard]] node_id parent(node_id v) const;
[[nodiscard]] std::optional<node_id> suffix_link(node_id v) const;
[[nodiscard]] uint32 length(node_id v) const;
template<std::integral Sym>
[[nodiscard]] std::optional<node_id> walk(node_id v, Sym sym) const;
template<typename Visit>
requires std::invocable<Visit &, node_id>
void for_each_child(node_id v, Visit visit) const;
[[nodiscard]] edge_ref edge(node_id v) const;
[[nodiscard]] std::span<uint32 const> word(uint32 w) const;
[[nodiscard]] uint64 suffix_count(node_id v) const;
[[nodiscard]] suffix_ref suffix(node_id v) const;
[[nodiscard]] std::vector<uint64> occurrence_counts() const;
[[nodiscard]] std::vector<node_id> nodes_by_length() const;
std::size_t node_count() const;
std::size_t word_count() const;
std::size_t size() const;
void reserve(std::size_t symbols);
void clear();
```

`extend` returns the longest suffix repeated at an earlier position, possibly
in another word. `finish_word` seals every nonempty suffix and returns its word
index, including for an empty word; `insert` extends and seals.
`walk` finds a child by the first edge symbol, rather than consuming its whole
label. `edge` addresses a retained word slice; `word` views a finished word.
`parent`, `suffix_link` and `length` navigate the structure. Suffix links may be
absent for open leaves; non-root sealed suffix ends have links.
`suffix_count` counts sealed suffixes ending exactly at a node; `suffix` gives
one representative when this count is positive. `occurrence_counts` sums sealed
suffixes below each node; the root counts nonempty sealed suffixes.
`nodes_by_length` supplies a parent-first order. Children have unspecified order.

## Complexity

For L fed symbols, storage is O(L + words + 1); at most 2L+1 nodes and
2L transitions. Expected amortized extension is O(log(degree+1)); insertion of
m symbols and sealing together take expected O(m log(degree+1)). A standalone
seal may perform O(m) splits and expected O(m log(degree+1)) pool work.
`walk` is expected O(log(degree+1)); child visitation is O(degree).
`edge` and `suffix` take O(log(words+1)); scalar queries and `word` are O(1).
Length ordering and occurrence propagation take O(nodes + maximum word length)
time and scratch, including an implicit unary tree with few nodes.
Clear is O(nodes + transitions + L + words), retaining capacity.
All walks and sealing passes are iterative; no call stack grows per suffix or node.

## Preconditions

Asserted under `ATL_ASSERT`: symbols inserted are in [0, Alphabet),
handles are valid, edge/parent/link queries exclude the root, and representative
suffix queries have positive suffix multiplicity. Positions, word counts and node
ids fit `uint32`, with its largest value reserved for sentinels; suffix multiplicities
fit `uint32`. Reserve arithmetic and counting-sort allocations must fit `size_t`.
`word(w)` requires w < word_count(). Queries treat out-of-alphabet symbols as absent; negative and wider-than-64-bit symbols are out of alphabet (compared at their own width).
Documented only: projections are stable; character code units require the same
alphabet mapping as integers. Negative signed-char values need a projection for
byte semantics. Node handles survive growth and are invalidated by clear,
assignment or destruction. Word spans are invalidated by extend, insert, clear,
assignment or destruction. The hash seed makes all operations runtime-only;
allocation failures propagate as standard exceptions.

## Measured performance

[The benchmark](../benchmarks/strings/suffix_tree.cpp) measures construction
throughput on about 1e6 symbols. By sibling list scans first-child/next-sibling
links; by separate lookups repeats pool descents and allocates sealing scratch
locally. The library retargets found pool entries and reuses scratch. The
certifying list contender is `suffix_tree_by_sibling_list` in
`experimental/atl/experimental/strings/suffix_tree_variants.hpp`.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape / operation | Contender | -O2 ms | native ms |
|---|---|---|---|
| sigma=2 child storage | library | 483 | 403 |
| sigma=2 child storage | by sibling list | 326 | 259 |
| sigma=26 child storage | library | 638 | 427 |
| sigma=26 child storage | by sibling list | 1250 | 1030 |
| sigma=1000000 child storage | library | 795 | 511 |
| one word sigma=2 | library | 487 | 419 |
| one word sigma=2 | by separate lookups | 504 | 433 |
| one word sigma=26 | library | 643 | 423 |
| one word sigma=26 | by separate lookups | 680 | 464 |
| one word sigma=1000000 | library | 799 | 519 |
| one word sigma=1000000 | by separate lookups | 1130 | 706 |
| one Fibonacci word | library | 105 | 56.1 |
| one Fibonacci word | by separate lookups | 117 | 60.9 |
| 1e6 symbols, words length 1-5 sigma=26 | library | 195 | 110 |
| 1e6 symbols, words length 1-5 sigma=26 | by separate lookups | 202 | 113 |
| 1e5 words x10 sigma=26 | library | 433 | 266 |
| 1e5 words x10 sigma=26 | by separate lookups | 461 | 289 |


The sibling list wins the two-symbol storage race and loses the 26-symbol race.
It is excluded at sigma=1e6 because root scans take linear degree; the pool is
measured there. The separate-lookup contender has no consistent advantage.

The library loses two-symbol child storage to the sibling list
(483/403 versus 326/259 ms), but wins 26-symbol storage
(638/427 versus 1250/1030 ms). The plain 26-symbol sibling-list row varies
between runs (1.37x spread). Separate lookups lose every shown minimum;
the plain sigma=1e6 separate-lookup row also varies (1.25x spread).

## Pitfalls and notes

Mid-word queries see an implicit tree: substring structure is valid but
pending suffixes are not sealed or counted. Edge labels can refer to the current
word, whereas `word` exposes only finished words; seal prior to reconstructing
labels. Empty words contribute no nonempty suffixes. Splits can create parents
with larger ids than their children, so numeric id order is unsuitable for DPs.

## How it works

Ukkonen's active point tracks a node, edge and offset. Each new symbol
creates leaves for pending suffixes until an existing extension is found; suffix
links and skip/count edge walks amortize the work. Sealing simulates a terminator
without storing it: split each pending suffix endpoint, mark multiplicities, close
open leaf edges and complete suffix links, including leaf links needed when later
words enter a closed leaf. All pending work lives in vectors. Reference: Ukkonen
(1995), “On-line construction of suffix trees”.

## See also

[Suffix automaton](suffix-automaton.md), [Suffix array](suffix-array.md), [Lexicographic order](lexicographic-order.md).
