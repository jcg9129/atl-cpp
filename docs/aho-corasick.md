# `atl::aho_corasick`

Recognize a fixed multiset of patterns while scanning integral-symbol text.
Suffix links handle sparse alphabets; output links enumerate matches without
walking nonmatching suffixes. Build a dense transition snapshot for repeated
scans or automaton dynamic programming over a modest alphabet.

```cpp
#include <atl/strings/aho_corasick.hpp>
#include <atl/strings/dense_transitions.hpp>
```

\include aho-corasick/quick.cpp

## API

```cpp
template<std::size_t Alphabet>
requires(Alphabet <= (uint64(1) << 32))
class aho_corasick;
```

`node_id` is `uint32`; nodes are distinct pattern prefixes. Root is zero;
other IDs follow first-touch insertion order. Pattern IDs are input indices.
`trie_word_range<R, Proj>` requires an input range of input ranges whose
projected symbols are integral. Projections are invoked as lvalues; captured
lambdas and `std::reference_wrapper` work.

```cpp
aho_corasick();
template<typename R, typename Proj = std::identity>
requires trie_word_range<R, Proj>
explicit aho_corasick(R &&words, Proj proj = {});
```

| Signature | Result |
|---|---|
| `node_id root() const` | scan start, empty prefix |
| `template<std::integral Sym> node_id step(node_id v, Sym sym) const` | longest suffix of the extended text that is a pattern prefix |
| `template<std::integral Sym> std::optional<node_id> walk(node_id v, Sym sym) const` | pure trie edge, no suffix fallback |
| `uint32 length(node_id v) const` | prefix depth |
| `std::optional<node_id> output_link(node_id v) const` | nearest proper suffix ending a pattern, nullopt at root |
| `std::span<uint32 const> words_at(node_id v) const` | pattern IDs ending here, ascending |
| `std::size_t word_count(node_id v) const` | patterns ending here, including duplicates |
| `std::size_t match_count(node_id v) const` | patterns ending here or at suffixes |
| `node_id suffix_link(node_id v) const` | longest proper suffix that is a pattern prefix |
| `node_id parent(node_id v) const` | prefix with final symbol removed |
| `uint32 symbol(node_id v) const` | last symbol of prefix |
| `node_id word_node(std::size_t i) const` | terminal node for pattern i |
| `std::span<node_id const> nodes_by_depth() const` | BFS order; trie and suffix parents precede children |
| `dense_transitions transition_table(std::size_t sigma) const` | snapshot over [0, sigma) |
| `dense_transitions transition_table() const requires(Alphabet > 0)` | snapshot over the entire alphabet |
| `std::size_t size() const` / `bool empty() const` | pattern count / zero patterns |
| `std::size_t node_count() const` | distinct prefixes including root |

All pure queries have `[[nodiscard]]`, except trivial `size` and `empty`.
These APIs are runtime-only because the pool captures a runtime hash seed.
Copying owns independent arrays and nodes; moving transfers their storage.
A moved-from automaton may only be assigned or destroyed.

`dense_transitions` owns a row-major table. Its members are constexpr:

```cpp
class dense_transitions {
public:
    static constexpr uint32 none = uint32(-1);
    constexpr dense_transitions();
    constexpr dense_transitions(std::vector<uint32> next, std::size_t const alphabet_size);
    constexpr dense_transitions(dense_transitions const &other);
    constexpr dense_transitions &operator=(dense_transitions const &other);
    constexpr dense_transitions(dense_transitions &&other) noexcept;
    constexpr dense_transitions &operator=(dense_transitions &&other) noexcept;
    [[nodiscard]] constexpr uint32 operator()(uint32 const v, uint32 const c) const;
    [[nodiscard]] constexpr std::span<uint32 const> row(uint32 const v) const;
    [[nodiscard]] constexpr std::size_t state_count() const;
    [[nodiscard]] constexpr std::size_t alphabet_size() const;
};
```

The default table has zero states and alphabet size one. Cells may be valid
state IDs or `none`; automaton snapshots are complete. Copies are independent;
moves leave zero states. Row spans follow transferred storage through a move
and expire when their storage is replaced or destroyed.

## Complexity

For n patterns, L total symbols and maximum sibling degree d, construction is
expected O(n + L log(1 + d)); storage is O(nodes + n), independent of Alphabet.
A forward scan of m symbols takes expected O(m log(1 + d)). One `step` can walk
O(length(v)) suffix links. Hash trees can have linear worst-case sibling height;
the expected bounds assume well-distributed hashed order. `walk` takes expected
O(log(1 + d)). Remaining node/count queries are O(1).

Output enumeration takes O(1 + matches) by checking the scan state and following
output links. Repeatedly stepping every state/symbol pair can take quadratic
work in pattern depth. A snapshot instead takes O(nodes * sigma) time and four
bytes per cell, independent of depth. Dense cell/row queries are O(1), copies
O(cells), and moves O(1) apart from releasing destination storage. Table
assertion scans are O(cells). All depth-dependent traversal is iterative with
heap buffers; no recursion grows with pattern length or node count.

## Preconditions

Asserted with `ATL_ASSERT`: projected pattern symbols lie in [0, Alphabet),
pattern count fits uint32, node creation has fewer than 2^32 - 1 existing
nodes, handles are valid, and pattern indices are in range. `suffix_link`,
`parent` and `symbol` require a nonroot node. Snapshot sigma must be positive
and at most Alphabet; nodes * sigma must fit `size_t` in bytes. A zero alphabet
can store only empty patterns and cannot produce a dense snapshot: the
no-argument `transition_table()` is constrained to `Alphabet > 0`. Pattern and
query symbols are compared with the alphabet at their own width, so a negative
or 128-bit value is out of alphabet and never aliases a smaller symbol.

Dense table construction asserts a positive alphabet size, a cell count
divisible by it, at most 2^32 - 1 states, and cells that are valid state IDs or
`none`. Cell queries assert valid state and symbol indices. Vector and sentinel
allocations must fit their implementation limits.

Documented only: projections give the intended consistent mapping. All standard
character types and integral alphabets retain numeric values, including signed
char; project plain char to unsigned char for arbitrary bytes. Non-contiguous
and single-pass pattern ranges work; proxy ranges need an integral-valued
projection. The automaton owns projected symbols. Text symbols outside the
alphabet reset to root. Construct from `main` or later, once process hash seed
initialization is complete. Allocation/projection exceptions propagate.
Automaton spans survive const queries, follow storage through moves, and expire
on assignment/destruction. Snapshots own storage independently of the automaton.

## Measured performance

[The benchmark](../benchmarks/strings/aho_corasick.cpp) measures whole-table
allocation/fill throughput, comparing library payload lookahead with public-API
row fills. `transition_table_by_row_options` in
`experimental/atl/experimental/strings/aho_corasick_variants.hpp` varies
zeroing, prefetch threshold and cache-line read/write hints. Its zeroed
first-line instantiation is the certifying experimental oracle. The same header
contains `aho_corasick_by_dense_table`, which stores all transitions permanently;
its build/scan race is not measured yet in the supplied records. Sparse
build/scan and step/walk table-fill comparisons are also not measured yet.

Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
2026-11-02. Flags: `-O2 -g -DNDEBUG -std=c++23`, optionally `-march=native`.
Minimum per-run mean milliseconds over three runs of 25 samples. The record
filter covers only the table-build variants. Rows below retain all measured
shapes and contenders. Z = by row options, zeroed first line; O = by row
options, overwrite first line; N = by row options, never prefetch;
A = by row options, always first line; L = by row options, all lines;
W = by row options, all lines write; OW = by row options, overwrite all lines
write. Z/O use a 2^18-cell prefetch threshold; A/L/W/OW use zero.

### Plain -O2

| sigma, patterns | Nodes / cells | library | Z | O | N | A | L | W | OW |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2, 1000x10 | 1582 / 3164 | 0.004 | 0.010 | 0.010 | 0.005 | 0.009 | 0.006 | 0.011 | 0.004 |
| 2, 10000x10 | 2047 / 4094 | 0.005 | 0.011 | 0.011 | 0.006 | 0.011 | 0.007 | 0.013 | 0.006 |
| 2, 100000x10 | 2047 / 4094 | 0.005 | 0.011 | 0.011 | 0.006 | 0.011 | 0.007 | 0.013 | 0.006 |
| 2, 10000x20 | 78124 / 156248 | 0.582 | 0.808 | 0.788 | 0.610 | 0.824 | 0.667 | 0.922 | 0.583 |
| 2, 100000x20 | 455054 / 910108 | 8.2 | 9.6 | 9.3 | 8.0 | 9.4 | 8.0 | 9.5 | 7.4 |
| 2, 100000x30 | 1450237 / 2900474 | 45.8 | 54.1 | 52.0 | 48.2 | 52.9 | 46.6 | 55.2 | 46.3 |
| 8, 1000x10 | 7388 / 59104 | 0.043 | 0.053 | 0.052 | 0.046 | 0.052 | 0.049 | 0.057 | 0.045 |
| 8, 10000x10 | 62684 / 501472 | 0.750 | 0.838 | 0.778 | 0.803 | 0.839 | 0.784 | 0.861 | 0.703 |
| 26, 1000x10 | 8502 / 221052 | 0.078 | 0.094 | 0.091 | 0.094 | 0.094 | 0.091 | 0.103 | 0.078 |
| 26, 10000x10 | 78250 / 2034500 | 1.5 | 1.8 | 1.6 | 1.8 | 1.8 | 1.7 | 1.8 | 1.3 |
| 26, 100000x10 | 707617 / 18398042 | 72.3 | 76.4 | 73.9 | 75.8 | 76.0 | 77.3 | 79.3 | 75.6 |
| 26, a^2000 | 2001 / 52026 | 0.018 | 0.021 | 0.018 | 0.021 | 0.021 | 0.022 | 0.021 | 0.018 |
| 26, a^20000 | 20001 / 520026 | 0.175 | 0.245 | 0.177 | 0.242 | 0.255 | 0.251 | 0.254 | 0.175 |

### -O2 -march=native

| sigma, patterns | Nodes / cells | library | Z | O | N | A | L | W | OW |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2, 1000x10 | 1582 / 3164 | 0.004 | 0.010 | 0.010 | 0.004 | 0.010 | 0.012 | 0.012 | 0.004 |
| 2, 10000x10 | 2047 / 4094 | 0.004 | 0.011 | 0.011 | 0.005 | 0.011 | 0.014 | 0.013 | 0.006 |
| 2, 100000x10 | 2047 / 4094 | 0.005 | 0.011 | 0.011 | 0.005 | 0.011 | 0.014 | 0.013 | 0.006 |
| 2, 10000x20 | 78124 / 156248 | 0.583 | 0.809 | 0.813 | 0.605 | 0.831 | 0.916 | 0.910 | 0.585 |
| 2, 100000x20 | 455054 / 910108 | 7.8 | 10.2 | 10.1 | 8.0 | 9.6 | 10.6 | 10.0 | 8.1 |
| 2, 100000x30 | 1450237 / 2900474 | 46.3 | 52.2 | 51.5 | 46.9 | 51.3 | 55.5 | 54.0 | 45.2 |
| 8, 1000x10 | 7388 / 59104 | 0.033 | 0.052 | 0.053 | 0.041 | 0.053 | 0.058 | 0.057 | 0.033 |
| 8, 10000x10 | 62684 / 501472 | 0.750 | 0.839 | 0.790 | 0.814 | 0.839 | 0.875 | 0.877 | 0.647 |
| 26, 1000x10 | 8502 / 221052 | 0.078 | 0.090 | 0.087 | 0.085 | 0.091 | 0.105 | 0.102 | 0.070 |
| 26, 10000x10 | 78250 / 2034500 | 1.5 | 1.8 | 1.6 | 1.9 | 1.8 | 1.9 | 1.9 | 1.4 |
| 26, 100000x10 | 707617 / 18398042 | 71.9 | 75.2 | 74.5 | 75.8 | 75.6 | 79.5 | 79.9 | 75.9 |
| 26, a^2000 | 2001 / 52026 | 0.018 | 0.021 | 0.018 | 0.021 | 0.021 | 0.022 | 0.021 | 0.018 |
| 26, a^20000 | 20001 / 520026 | 0.177 | 0.246 | 0.182 | 0.245 | 0.255 | 0.248 | 0.242 | 0.174 |

The plain -O2 overwrite-first-line row at sigma=2 100000x10 has an unstable
3.42x spread. Other spreads reach 1.33x at -O2 and 1.26x native; tiny differences
are not reliable speedups. No prefetch policy wins uniformly. Library row and
payload lookahead keeps the large sigma=26 table at 72.3/71.9 ms compared with
76.4/75.2 for zeroed first-line public-API filling.

## Pitfalls and notes

Duplicates count separately. Empty patterns end at root and match at every
text boundary: include `match_count(root())` for the initial boundary, then add
`match_count(state)` per symbol. To enumerate, visit `words_at(state)` and each
output-linked node. Root has no output link. To count each pattern over an
entire text, accumulate visits at scan states and traverse `nodes_by_depth()`
in reverse, adding each nonroot total into its suffix-link target. Creation
order alone is not topological for suffix links.

A restricted table still falls back through pattern prefixes containing larger
symbols; it agrees with `step` on every retained symbol at every node.
Permanent dense storage costs O(nodes * Alphabet); use the sparse engine for
large alphabets. Tables are independent snapshots, so their memory adds to the
source automaton's memory.

## How it works

The shared `include/atl/ds/detail/hashed_child_pool.hpp` stores hashed sibling
BSTs in a flat arena. Each node fuses incoming symbol, parent, suffix/output
links, length and match counts. BFS resolves suffix links using completed
shallower nodes, propagates match counts and records an order valid for both
trees. A stable counting-sort bucket maps pattern IDs to terminal-node spans.
Output links skip suffixes with no terminating pattern.

Dense construction initializes only the root row. Every other row is copied
from its suffix row and its own child columns are overwritten. BFS groups each
parent's children in one contiguous run, letting a single cursor traverse them
without sibling-tree probes. Large tables use row and node lookahead prefetch.
Completed allocations are adopted by `dense_transitions` without a second fill.
Reference: Aho and Corasick (1975), “Efficient string matching: An aid to
bibliographic search”.

## See also

[Trie](trie.md), [Static trie](static-trie.md), [String matching](matching.md),
[Suffix automaton](suffix-automaton.md).
