# `atl::palindromic_tree`

An online generalized palindromic tree, with one node per distinct nonempty
palindrome across all inserted words. Use it for occurrence counts, suffix
links and per-prefix answers over sparse alphabets; static substring
predicates are simpler with palindrome_lengths.

```cpp
#include <atl/strings/palindromic_tree.hpp>
```

\include palindromic-tree/quick.cpp

## API

```cpp
template<std::size_t Alphabet>
    requires(Alphabet <= (uint64(1) << 32))
class palindromic_tree;
```

Symbols are standard integral values in `[0, Alphabet)`; Alphabet itself must
fit size_t. Character signedness matters: negative char values must be
projected to nonnegative integers. char8_t, wide characters, signed and
unsigned integer alphabets work when their values fit. A `trie_word<R, Proj>`
is an input range with integral projected symbols, including single-pass
ranges. Software int128 is not a standard integral type for these APIs.
All operations are runtime-only because the child pool captures the process
hash seed; the seed remains fixed for the pool's lifetime.

Member signatures (inside `palindromic_tree<Alphabet>`):

```cpp
using node_id = uint32;
palindromic_tree();
[[nodiscard]] constexpr node_id odd_root() const;
[[nodiscard]] constexpr node_id even_root() const;
template<std::integral Sym> node_id extend(Sym sym);
node_id finish_word();
[[nodiscard]] node_id last() const;
template<typename R, typename Proj = std::identity>
    requires trie_word<R, Proj>
node_id insert(R &&word, Proj proj = {});
[[nodiscard]] int32 length(node_id v) const;
[[nodiscard]] node_id suffix_link(node_id v) const;
[[nodiscard]] node_id parent(node_id v) const;
[[nodiscard]] uint32 symbol(node_id v) const;
[[nodiscard]] uint32 palindromic_suffix_count(node_id v) const;
template<std::integral Sym>
[[nodiscard]] std::optional<node_id> walk(node_id v, Sym sym) const;
template<child_visitor Visit> void for_each_child(node_id v, Visit visit) const;
[[nodiscard]] std::vector<node_id> series_links() const;
[[nodiscard]] std::vector<uint64> occurrence_counts() const;
std::size_t palindrome_count() const;
std::size_t node_count() const;
void reserve(std::size_t symbols);
void clear();
```

| Operation | Meaning |
|---|---|
| `odd_root()` / `even_root()` | Ids 0 and 1, lengths -1 and 0; constexpr, they read no tree state. |
| `extend(sym)` | Append a symbol and return the longest palindromic suffix node. |
| `finish_word()` | Return the current suffix node, clear the current word and reset last() to even_root(). |
| `insert(word, proj)` | Extend each projected symbol, then finish_word(). |
| `last()` | Current longest-palindromic-suffix node. |
| `length(v)` / `symbol(v)` | Palindrome length / wrapping symbol. |
| `parent(v)` | Remove the two outer symbols; singletons have odd_root() as parent. |
| `suffix_link(v)` | Longest proper palindromic suffix; even_root() links to odd_root(). |
| `palindromic_suffix_count(v)` | Number of nonempty palindromic suffixes; zero for roots. |
| `walk(v, sym)` | Node for sym·v·sym, or nullopt, including out-of-alphabet symbols. |
| `for_each_child(v, visit)` | Invoke `visit(uint32 symbol, node_id child)` in unspecified hashed order. `child_visitor` checks invocability. |
| `series_links()` | For each node, first suffix whose length difference changes; roots map to themselves. |
| `occurrence_counts()` | Per-node counts across words with multiplicity; ignore root entries. |
| `palindrome_count()` / `node_count()` | Distinct nonempty palindrome count / count including roots. |
| `reserve(symbols)` / `clear()` | Reserve for at most symbols + 2 nodes / reset while keeping capacity. |

## Complexity

With D the maximum sibling degree, child probes take expected O(log(D + 1))
time; worst-case hashed BST height can be D. Extend is expected amortized
O(log(D + 1)); one call can walk O(word length) suffix links. Insert is
expected amortized O(word length · log(D + 1)). Child enumeration takes
O(actual children) time and an explicit O(actual children) traversal stack.
Other accessors and finish_word take O(1); series_links and occurrence_counts
take O(nodes) time and result storage. Clear takes O(nodes + current word
length); reserve has linear worst-case relocation cost.

Memory is O(nodes + longest word), at most L + 2 nodes for L input symbols.
The node payload is 40 bytes on the measured 64-bit ABI with widened counters;
allocation capacity adds overhead. All traversals are iterative, with no
recursion per symbol, sibling or node. Node ids remain valid across growth
and finish_word, and are invalidated by clear or assignment. Returned vectors
own their results and remain independent of later updates.

## Preconditions

Asserted: append symbols lie in `[0, Alphabet)`; each word has at most
INT32_MAX symbols; the pool contains at most 2^32 - 1 nodes including roots;
node handles are valid; suffix_link excludes odd_root; parent and symbol
exclude both roots; reserve addition fits size_t. Seed increments and count
propagation are checked for uint64 overflow.

Documented only: occurrence counts fit uint64, projection preserves the
intended alphabet, and a child visitor does not mutate the tree while
traversing it. Count queries include the current unfinished word. Starting
insert with an unfinished word continues that word and then finishes it.
Allocation failures propagate standard container exceptions; clear or replace
the tree if a mutating allocation fails. Moved-from instances may only be
destroyed or assigned to.

## Measured performance

`benchmarks/strings/palindromic_tree.cpp` races reserve plus whole-word build.
The library files a node in the same descent that probes for it; by find
then emplace probes again for missing nodes.
The native sigma=26 10^6 find-then-emplace row varies between runs
(1.29x spread); its minimum alone does not establish a stable ranking.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | library -O2 | by find then emplace -O2 | library native | by find then emplace native |
|---|---:|---:|---:|---:|
| 10^6 sigma=2 | 12.1 (1.03x) | 13.4 (1.01x) | 12.7 (1.11x) | 13.7 (1.11x) |
| 10^6 sigma=26 | 30.4 (1.04x) | 34.5 (1.23x) | 30.4 (1.15x) | 36.5 (1.29x) |
| 10^6 all-a | 26.3 (1.00x) | 27.2 (1.02x) | 26.8 (1.05x) | 27.4 (1.06x) |
| 10^6 sigma=10^6 | 341 (1.04x) | 446 (1.04x) | 404 (1.10x) | 526 (1.15x) |
| 5*10^6 sigma=26 | 164 (1.05x) | 197 (1.18x) | 172 (1.07x) | 193 (1.22x) |

## Pitfalls and notes

Words share palindrome nodes, and duplicates contribute occurrences again.
finish_word prevents palindromes from crossing word boundaries. Child order
is unspecified; compare child sets rather than relying on enumeration order.
For palindromic factorization, series links group suffixes with equal
`length(v) - length(suffix_link(v))` into arithmetic progressions; compute
the returned vector again when new nodes have been added.

## How it works

Each node fuses a hashed sibling-BST entry with its palindrome payload.
A suffix-link walk tests the symbol preceding a candidate suffix in the
current word; the length -1 root always extends. Each appended symbol
creates at most one new palindrome. Its longest proper suffix already has
a node, so parent and suffix links point to smaller ids for nonroot nodes.
Reverse id order propagates occurrence seeds without sorting or recursion;
forward order computes series links. Only the current word's symbols need
to be retained for mirror checks. Child enumeration uses a heap stack.
Reference: Rubinchik and Shur (2018), “EERTREE: An efficient data structure
for processing palindromes in strings”.

## See also

[Palindrome radii](palindrome-lengths.md), [String matching](matching.md),
[Tries](trie.md), [Static tries](static-trie.md), [Suffix automaton](suffix-automaton.md).
